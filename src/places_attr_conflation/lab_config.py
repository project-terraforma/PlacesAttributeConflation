"""Configuration contracts for the local-first hybrid lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str
    model: str
    enabled: bool = True
    endpoint: str | None = None
    api_key_env: str | None = None
    max_input_chars: int = 20000
    max_output_chars: int = 8000
    timeout_seconds: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectPolicy:
    mode: str = "local_only"
    allow_hosted_fallback: bool = False
    allow_prompt_debug_capture: bool = False


@dataclass(frozen=True)
class RoutingPolicy:
    default_task: str = "generate"
    long_context_chars: int = 12000
    high_reasoning_tasks: tuple[str, ...] = ("research_synthesis", "paper_to_code", "long_context_generate")
    fallback_task_prefixes: tuple[str, ...] = ("high_reasoning", "hosted_")


@dataclass(frozen=True)
class SecurityPolicy:
    bind_host: str = "127.0.0.1"
    bind_port: int = 8765
    max_context_chars: int = 30000
    max_candidate_count: int = 100
    block_remote_clients: bool = True
    capture_full_prompts: bool = False


@dataclass(frozen=True)
class LabConfig:
    providers: tuple[ProviderConfig, ...]
    project_policy: ProjectPolicy = ProjectPolicy()
    routing_policy: RoutingPolicy = RoutingPolicy()
    security_policy: SecurityPolicy = SecurityPolicy()
    cache_path: str = ".lab_cache.json"
    audit_log_path: str = ".lab_audit.jsonl"

    def provider_by_name(self, name: str) -> ProviderConfig | None:
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None


def load_lab_config(path: str | Path) -> LabConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    providers = tuple(ProviderConfig(**item) for item in payload.get("providers", []))
    return LabConfig(
        providers=providers,
        project_policy=ProjectPolicy(**payload.get("project_policy", {})),
        routing_policy=RoutingPolicy(**payload.get("routing_policy", {})),
        security_policy=SecurityPolicy(**payload.get("security_policy", {})),
        cache_path=payload.get("cache_path", ".lab_cache.json"),
        audit_log_path=payload.get("audit_log_path", ".lab_audit.jsonl"),
    )
