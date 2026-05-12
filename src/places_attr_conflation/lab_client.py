"""Thin client for the local hybrid lab API."""

from __future__ import annotations

from dataclasses import asdict
import json
import urllib.request

from .lab_protocol import EmbedRequest, GenerateRequest, RerankRequest


class LabClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, request: GenerateRequest) -> dict[str, object]:
        return self._post("/v1/generate", asdict(request))

    def embed(self, request: EmbedRequest) -> dict[str, object]:
        return self._post("/v1/embed", asdict(request))

    def rerank(self, request: RerankRequest) -> dict[str, object]:
        return self._post("/v1/rerank", asdict(request))
