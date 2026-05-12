"""Deterministic routing logic for local-first model selection."""

from __future__ import annotations

from .lab_config import LabConfig
from .lab_policy import PolicyError
from .lab_protocol import GenerateRequest, RouteDecision, RouteRequest


def route_request(config: LabConfig, request: RouteRequest) -> RouteDecision:
    local_provider = next((provider for provider in config.providers if provider.enabled and provider.kind == "local"), None)
    hosted_provider = next((provider for provider in config.providers if provider.enabled and provider.kind == "hosted"), None)
    if local_provider is None:
        raise PolicyError("No enabled local provider configured.")

    total_chars = request.prompt_chars + request.context_chars
    high_reasoning = request.task in config.routing_policy.high_reasoning_tasks or any(
        request.task.startswith(prefix) for prefix in config.routing_policy.fallback_task_prefixes
    )

    if total_chars > config.routing_policy.long_context_chars or high_reasoning:
        if (
            hosted_provider is not None
            and config.project_policy.allow_hosted_fallback
            and request.policy.allow_hosted_fallback
        ):
            reason = "hosted fallback allowed for long-context or high-reasoning task"
            return RouteDecision(hosted_provider.name, hosted_provider.kind, reason, used_hosted_fallback=True)
    return RouteDecision(local_provider.name, local_provider.kind, "default local-first route")


def route_generate_request(config: LabConfig, request: GenerateRequest) -> RouteDecision:
    return route_request(
        config,
        RouteRequest(
            task=request.task,
            prompt_chars=len(request.prompt),
            context_chars=len(request.context),
            policy=request.policy,
        ),
    )
