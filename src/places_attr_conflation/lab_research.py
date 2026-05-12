"""Structured research and experiment records for paper-to-code workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchSourceRecord:
    source_id: str
    source_type: str
    title: str
    reference: str
    claims: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    source_refs: tuple[str, ...]
    claim_under_test: str
    implementation_scope: str
    target_modules: tuple[str, ...]
    expected_metric_change: str
    datasets_or_fixtures: tuple[str, ...]
    baseline_name: str
    eval_command: str
    success_criteria: tuple[str, ...]
    failure_criteria: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    baseline_metrics: dict[str, float]
    experiment_metrics: dict[str, float]
    promotion_decision: str
    decision_reason: str
    artifacts: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_record(record: ResearchSourceRecord | ExperimentSpec | ExperimentResult) -> dict[str, Any]:
    return asdict(record)
