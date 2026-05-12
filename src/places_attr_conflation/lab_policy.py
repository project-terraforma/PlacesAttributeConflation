"""Security and policy enforcement for the hybrid lab."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .lab_config import LabConfig
from .lab_protocol import EmbedRequest, GenerateRequest, RerankRequest


PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"exfiltrate",
    r"reveal\s+secrets",
)


class PolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptAudit:
    flagged: bool
    matches: tuple[str, ...]


def audit_untrusted_text(text: str) -> PromptAudit:
    lowered = (text or "").lower()
    matches = tuple(pattern for pattern in PROMPT_INJECTION_PATTERNS if re.search(pattern, lowered))
    return PromptAudit(flagged=bool(matches), matches=matches)


def validate_generate_request(config: LabConfig, request: GenerateRequest) -> None:
    if len(request.prompt) > config.security_policy.max_context_chars:
        raise PolicyError("Prompt exceeds max_context_chars.")
    if len(request.context) > config.security_policy.max_context_chars:
        raise PolicyError("Context exceeds max_context_chars.")
    if request.policy.capture_debug_prompt and not config.project_policy.allow_prompt_debug_capture:
        raise PolicyError("Prompt debug capture is disabled by policy.")


def validate_embed_request(config: LabConfig, request: EmbedRequest) -> None:
    if any(len(text) > config.security_policy.max_context_chars for text in request.texts):
        raise PolicyError("Embedding text exceeds max_context_chars.")


def validate_rerank_request(config: LabConfig, request: RerankRequest) -> None:
    if len(request.candidates) > config.security_policy.max_candidate_count:
        raise PolicyError("Too many rerank candidates.")
    if any(len(candidate) > config.security_policy.max_context_chars for candidate in request.candidates):
        raise PolicyError("Rerank candidate exceeds max_context_chars.")
