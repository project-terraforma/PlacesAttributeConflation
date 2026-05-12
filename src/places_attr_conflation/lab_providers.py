"""Provider abstractions for local and hosted model lanes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os

from .lab_config import ProviderConfig
from .lab_protocol import EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, RerankRequest, RerankResponse


class ProviderError(RuntimeError):
    pass


@dataclass
class BaseProvider:
    config: ProviderConfig

    @property
    def kind(self) -> str:
        return self.config.kind

    def generate(self, request: GenerateRequest, *, route_reason: str) -> GenerateResponse:
        raise NotImplementedError

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise NotImplementedError

    def rerank(self, request: RerankRequest) -> RerankResponse:
        raise NotImplementedError


def _hash_score(text: str) -> float:
    digest = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _vectorize(text: str, dimensions: int = 8) -> tuple[float, ...]:
    digest = sha256(text.encode("utf-8")).digest()
    values = []
    for idx in range(dimensions):
        start = idx * 2
        chunk = digest[start : start + 2]
        values.append(int.from_bytes(chunk, "big") / 65535.0)
    return tuple(values)


class LocalEchoProvider(BaseProvider):
    """Deterministic local provider scaffold used until a real local LLM is wired."""

    def generate(self, request: GenerateRequest, *, route_reason: str) -> GenerateResponse:
        context_note = f" context_chars={len(request.context)}" if request.context else ""
        return GenerateResponse(
            provider_name=self.config.name,
            provider_kind=self.kind,
            output_text=f"[local:{self.config.model}] task={request.task}{context_note}\n{request.prompt[: self.config.max_output_chars]}",
            route_reason=route_reason,
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(provider_name=self.config.name, vectors=tuple(_vectorize(text) for text in request.texts))

    def rerank(self, request: RerankRequest) -> RerankResponse:
        ranked = sorted(request.candidates, key=lambda candidate: _hash_score(f"{request.query}\n{candidate}"), reverse=True)
        scores = tuple(_hash_score(f"{request.query}\n{candidate}") for candidate in ranked)
        return RerankResponse(provider_name=self.config.name, ranked_candidates=tuple(ranked), scores=scores)


class HostedStubProvider(BaseProvider):
    """Hosted provider scaffold that stays disabled until explicit configuration exists."""

    def _ensure_enabled(self) -> None:
        if self.config.api_key_env and not os.getenv(self.config.api_key_env):
            raise ProviderError(f"Hosted provider '{self.config.name}' is missing env var {self.config.api_key_env}.")

    def generate(self, request: GenerateRequest, *, route_reason: str) -> GenerateResponse:
        self._ensure_enabled()
        return GenerateResponse(
            provider_name=self.config.name,
            provider_kind=self.kind,
            output_text=f"[hosted:{self.config.model}] task={request.task}\n{request.prompt[: self.config.max_output_chars]}",
            route_reason=route_reason,
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        self._ensure_enabled()
        return EmbedResponse(provider_name=self.config.name, vectors=tuple(_vectorize(f"hosted:{text}") for text in request.texts))

    def rerank(self, request: RerankRequest) -> RerankResponse:
        self._ensure_enabled()
        ranked = sorted(request.candidates, key=lambda candidate: len(candidate), reverse=True)
        scores = tuple(float(len(candidate)) for candidate in ranked)
        return RerankResponse(provider_name=self.config.name, ranked_candidates=tuple(ranked), scores=scores)


def build_provider(config: ProviderConfig) -> BaseProvider:
    if config.kind == "local":
        return LocalEchoProvider(config)
    if config.kind == "hosted":
        return HostedStubProvider(config)
    raise ProviderError(f"Unknown provider kind: {config.kind}")
