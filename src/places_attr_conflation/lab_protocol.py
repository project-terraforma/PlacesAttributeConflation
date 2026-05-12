"""Typed request and response schemas for the hybrid lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabPolicyInput:
    project: str = "default"
    allow_hosted_fallback: bool = False
    capture_debug_prompt: bool = False


@dataclass(frozen=True)
class GenerateRequest:
    task: str
    prompt: str
    context: str = ""
    policy: LabPolicyInput = LabPolicyInput()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbedRequest:
    texts: tuple[str, ...]
    policy: LabPolicyInput = LabPolicyInput()


@dataclass(frozen=True)
class RerankRequest:
    query: str
    candidates: tuple[str, ...]
    policy: LabPolicyInput = LabPolicyInput()


@dataclass(frozen=True)
class RouteRequest:
    task: str
    prompt_chars: int
    context_chars: int
    policy: LabPolicyInput = LabPolicyInput()


@dataclass(frozen=True)
class RouteDecision:
    provider_name: str
    provider_kind: str
    reason: str
    used_hosted_fallback: bool = False


@dataclass(frozen=True)
class GenerateResponse:
    provider_name: str
    provider_kind: str
    output_text: str
    route_reason: str
    cached: bool = False


@dataclass(frozen=True)
class EmbedResponse:
    provider_name: str
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RerankResponse:
    provider_name: str
    ranked_candidates: tuple[str, ...]
    scores: tuple[float, ...]


def to_payload(value: Any) -> dict[str, Any]:
    return asdict(value)
