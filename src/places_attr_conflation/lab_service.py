"""Loopback-only HTTP service for the local-first hybrid lab."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
from typing import Any

from .lab_config import LabConfig
from .lab_policy import PolicyError, audit_untrusted_text, validate_embed_request, validate_generate_request, validate_rerank_request
from .lab_protocol import EmbedRequest, GenerateRequest, LabPolicyInput, RerankRequest
from .lab_providers import BaseProvider, build_provider
from .lab_router import route_generate_request


class LabRuntime:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.providers: dict[str, BaseProvider] = {
            provider.name: build_provider(provider) for provider in config.providers if provider.enabled
        }
        self.cache_path = Path(config.cache_path)
        self.audit_log_path = Path(config.audit_log_path)
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        return json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _write_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True), encoding="utf-8")

    def _append_audit(self, payload: dict[str, Any]) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _cache_key(self, prefix: str, payload: dict[str, Any]) -> str:
        return f"{prefix}:{json.dumps(payload, sort_keys=True)}"

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        validate_generate_request(self.config, request)
        audit = audit_untrusted_text(request.context)
        route = route_generate_request(self.config, request)
        cache_key = self._cache_key("generate", asdict(request))
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            return cached
        provider = self.providers[route.provider_name]
        response = asdict(provider.generate(request, route_reason=route.reason))
        response["prompt_audit"] = asdict(audit)
        self._cache[cache_key] = response
        self._write_cache()
        self._append_audit(
            {
                "event": "generate",
                "provider": route.provider_name,
                "route_reason": route.reason,
                "used_hosted_fallback": route.used_hosted_fallback,
                "prompt_chars": len(request.prompt),
                "context_chars": len(request.context),
                "prompt_injection_flagged": audit.flagged,
            }
        )
        return response

    def embed(self, request: EmbedRequest) -> dict[str, Any]:
        validate_embed_request(self.config, request)
        provider = next(provider for provider in self.providers.values() if provider.kind == "local")
        response = asdict(provider.embed(request))
        self._append_audit({"event": "embed", "provider": provider.config.name, "texts": len(request.texts)})
        return response

    def rerank(self, request: RerankRequest) -> dict[str, Any]:
        validate_rerank_request(self.config, request)
        provider = next(provider for provider in self.providers.values() if provider.kind == "local")
        response = asdict(provider.rerank(request))
        self._append_audit({"event": "rerank", "provider": provider.config.name, "candidates": len(request.candidates)})
        return response

    def route(self, request: GenerateRequest) -> dict[str, Any]:
        return asdict(route_generate_request(self.config, request))

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "providers": sorted(self.providers)}


def _policy_from_payload(payload: dict[str, Any]) -> LabPolicyInput:
    return LabPolicyInput(**payload) if payload else LabPolicyInput()


def _generate_request_from_payload(payload: dict[str, Any]) -> GenerateRequest:
    return GenerateRequest(
        task=payload["task"],
        prompt=payload["prompt"],
        context=payload.get("context", ""),
        policy=_policy_from_payload(payload.get("policy", {})),
        metadata=payload.get("metadata", {}),
    )


def _embed_request_from_payload(payload: dict[str, Any]) -> EmbedRequest:
    return EmbedRequest(texts=tuple(payload.get("texts", ())), policy=_policy_from_payload(payload.get("policy", {})))


def _rerank_request_from_payload(payload: dict[str, Any]) -> RerankRequest:
    return RerankRequest(
        query=payload["query"],
        candidates=tuple(payload.get("candidates", ())),
        policy=_policy_from_payload(payload.get("policy", {})),
    )


def build_handler(runtime: LabRuntime) -> type[BaseHTTPRequestHandler]:
    class LabHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject_remote_client(self) -> bool:
            if not runtime.config.security_policy.block_remote_clients:
                return False
            client_host = self.client_address[0]
            if client_host not in {"127.0.0.1", "::1"}:
                self._send_json(403, {"error": "remote clients are blocked"})
                return True
            return False

        def do_GET(self) -> None:  # noqa: N802
            if self._reject_remote_client():
                return
            if self.path == "/v1/health":
                self._send_json(200, runtime.health())
                return
            if self.path == "/v1/models":
                self._send_json(200, {"providers": sorted(runtime.providers)})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self._reject_remote_client():
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/v1/generate":
                    self._send_json(200, runtime.generate(_generate_request_from_payload(payload)))
                    return
                if self.path == "/v1/embed":
                    self._send_json(200, runtime.embed(_embed_request_from_payload(payload)))
                    return
                if self.path == "/v1/rerank":
                    self._send_json(200, runtime.rerank(_rerank_request_from_payload(payload)))
                    return
                if self.path == "/v1/route":
                    self._send_json(200, runtime.route(_generate_request_from_payload(payload)))
                    return
                self._send_json(404, {"error": "not found"})
            except PolicyError as exc:
                self._send_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return LabHandler


def create_server(config: LabConfig) -> ThreadingHTTPServer:
    runtime = LabRuntime(config)
    handler = build_handler(runtime)
    return ThreadingHTTPServer((config.security_policy.bind_host, config.security_policy.bind_port), handler)
