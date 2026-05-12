"""Synthetic authoritative evidence benchmarks for resolver validation.

Synthetic evidence is a systems test: it verifies resolver behavior, replay
shape, abstention, and edge-case handling. It is not a live evidence claim.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .golden import _as_float, _normalize
from .manifest import EvidenceItem
from .resolver import resolve_attribute


EDGE_SCENARIOS = (
    "authoritative_truth",
    "truth_with_decoy",
    "tied_authority",
    "decoy_only",
    "no_matching_evidence",
    "truth_not_candidate",
)


@dataclass(frozen=True)
class SyntheticCaseMetrics:
    total: int
    covered: int
    correct: int
    abstained: int
    accuracy: float
    coverage: float
    abstention_rate: float
    high_confidence_wrong: int
    high_confidence_wrong_rate: float


def load_conflict_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _other_candidate(row: dict[str, str]) -> str:
    truth = row.get("truth", "")
    current = row.get("current_value", "")
    base = row.get("base_value", "")
    attribute = row.get("attribute", "")
    if _normalize(attribute, current) != _normalize(attribute, truth):
        return current
    if _normalize(attribute, base) != _normalize(attribute, truth):
        return base
    return current or base


def _candidate_values(row: dict[str, str], *, include_truth: bool = True) -> list[str]:
    attribute = row.get("attribute", "")
    candidates: list[str] = []
    for value in (row.get("current_value", ""), row.get("base_value", "")):
        if value and _normalize(attribute, value) not in {_normalize(attribute, existing) for existing in candidates}:
            candidates.append(value)
    truth = row.get("truth", "")
    if include_truth and truth and _normalize(attribute, truth) not in {_normalize(attribute, existing) for existing in candidates}:
        candidates.append(truth)
    return candidates


def _evidence_item(row: dict[str, str], source_type: str, value: str, suffix: str, source_rank: float | None = None) -> dict[str, object]:
    case_id = f"{row.get('id', '')}:{row.get('attribute', '')}"
    url = f"synthetic://{source_type}/{case_id}/{suffix}"
    item = EvidenceItem(
        source_type=source_type,
        url=url,
        attribute=row.get("attribute", ""),
        extracted_value=value,
        query=f"{row.get('id', '')} {row.get('attribute', '')}",
        source_rank=source_rank,
        notes="Synthetic evidence generated from project_a conflict truth.",
    )
    return item.to_dict()


def _scenario_for_index(index: int, include_edges: bool) -> str:
    if not include_edges:
        return "truth_with_decoy"
    return EDGE_SCENARIOS[index % len(EDGE_SCENARIOS)]


def _case_evidence(row: dict[str, str], scenario: str) -> list[dict[str, object]]:
    truth = row.get("truth", "")
    decoy = _other_candidate(row)
    if scenario == "authoritative_truth":
        return [_evidence_item(row, "official_site", truth, "truth")]
    if scenario == "truth_with_decoy":
        return [
            _evidence_item(row, "official_site", truth, "truth"),
            _evidence_item(row, "aggregator", decoy, "decoy"),
        ]
    if scenario == "tied_authority":
        return [
            _evidence_item(row, "official_site", truth, "truth"),
            _evidence_item(row, "official_site", decoy, "tie-decoy"),
        ]
    if scenario == "decoy_only":
        return [_evidence_item(row, "aggregator", decoy, "decoy")]
    if scenario == "no_matching_evidence":
        return [_evidence_item(row, "official_site", "__unmatched_synthetic_value__", "unmatched")]
    if scenario == "truth_not_candidate":
        return [_evidence_item(row, "official_site", truth, "new-truth", source_rank=1.0)]
    raise ValueError(f"Unknown synthetic scenario: {scenario}")


def generate_synthetic_evidence_cases(
    conflict_rows: Iterable[dict[str, str]],
    *,
    limit: int | None = None,
    include_edges: bool = True,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for index, row in enumerate(conflict_rows):
        if limit is not None and len(cases) >= limit:
            break
        scenario = _scenario_for_index(index, include_edges)
        cases.append(
            {
                "case_id": f"{row.get('id', '')}:{row.get('attribute', '')}:{index}",
                "id": row.get("id", ""),
                "base_id": row.get("base_id", ""),
                "attribute": row.get("attribute", ""),
                "scenario": scenario,
                "truth": row.get("truth", ""),
                "baseline": row.get("baseline", ""),
                "baseline_prediction": row.get("prediction", ""),
                "baseline_confidence": _as_float(row.get("confidence"), 0.0),
                "current_value": row.get("current_value", ""),
                "base_value": row.get("base_value", ""),
                "candidates": _candidate_values(row, include_truth=True),
                "evidence": _case_evidence(row, scenario),
            }
        )
    return {
        "mode": "synthetic_authoritative_evidence",
        "warning": "Synthetic evidence validates system behavior only; it is not live evidence.",
        "case_count": len(cases),
        "cases": cases,
    }


def write_synthetic_evidence(payload: dict[str, object], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def load_synthetic_evidence(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Synthetic evidence file must contain a top-level cases list.")
    return payload


def _baseline_correct(case: dict[str, object]) -> bool:
    attribute = str(case.get("attribute", ""))
    return _normalize(attribute, str(case.get("baseline_prediction", ""))) == _normalize(attribute, str(case.get("truth", "")))


def _case_correct(case: dict[str, object], decision: Any) -> bool:
    if decision.abstained:
        return False
    attribute = str(case.get("attribute", ""))
    return _normalize(attribute, decision.decision) == _normalize(attribute, str(case.get("truth", "")))


def _score_cases(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    covered = sum(1 for row in rows if not row["abstained"])
    correct = sum(1 for row in rows if row["correct"])
    abstained = sum(1 for row in rows if row["abstained"])
    high_confidence_wrong = sum(1 for row in rows if (not row["correct"] and not row["abstained"] and row["confidence"] >= 0.75))
    return asdict(
        SyntheticCaseMetrics(
            total=total,
            covered=covered,
            correct=correct,
            abstained=abstained,
            accuracy=correct / covered if covered else 0.0,
            coverage=covered / total if total else 0.0,
            abstention_rate=abstained / total if total else 0.0,
            high_confidence_wrong=high_confidence_wrong,
            high_confidence_wrong_rate=high_confidence_wrong / covered if covered else 0.0,
        )
    )


def evaluate_synthetic_evidence(
    payload: dict[str, object],
    *,
    min_confidence: float = 0.55,
    min_support_score: float = 0.55,
) -> dict[str, object]:
    decisions: list[dict[str, object]] = []
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        evidence = [
            EvidenceItem(
                source_type=str(item.get("source_type", "unknown")),
                url=str(item.get("url", "")),
                attribute=str(item.get("attribute", "")),
                extracted_value=str(item.get("extracted_value", "")),
                query=str(item.get("query", "")),
                observed_at=str(item.get("observed_at", "")),
                source_rank=_as_float(item.get("source_rank"), None) if item.get("source_rank") is not None else None,
                recency_days=_as_float(item.get("recency_days"), None) if item.get("recency_days") is not None else None,
                zombie_score=_as_float(item.get("zombie_score"), 0.0),
                identity_change_score=_as_float(item.get("identity_change_score"), 0.0),
                notes=str(item.get("notes", "")),
            )
            for item in case.get("evidence", [])
            if isinstance(item, dict)
        ]
        candidates = [str(value) for value in case.get("candidates", []) if value]
        decision = resolve_attribute(
            str(case.get("attribute", "")),
            candidates,
            evidence,
            min_confidence=min_confidence,
            min_support_score=min_support_score,
        )
        decisions.append(
            {
                "case_id": case.get("case_id", ""),
                "id": case.get("id", ""),
                "attribute": case.get("attribute", ""),
                "scenario": case.get("scenario", ""),
                "truth": case.get("truth", ""),
                "baseline_prediction": case.get("baseline_prediction", ""),
                "baseline_correct": _baseline_correct(case),
                "decision": decision.decision,
                "confidence": decision.confidence,
                "abstained": decision.abstained,
                "correct": _case_correct(case, decision),
                "reason": decision.reason,
                "evidence_count": len(evidence),
            }
        )

    by_scenario = {
        scenario: _score_cases([row for row in decisions if row["scenario"] == scenario])
        for scenario in sorted({str(row["scenario"]) for row in decisions})
    }
    by_attribute = {
        attribute: _score_cases([row for row in decisions if row["attribute"] == attribute])
        for attribute in sorted({str(row["attribute"]) for row in decisions})
    }
    baseline_rows = [
        {
            "correct": row["baseline_correct"],
            "abstained": not row["baseline_prediction"],
            "confidence": 1.0 if row["baseline_prediction"] else 0.0,
        }
        for row in decisions
    ]
    return {
        "mode": payload.get("mode", "synthetic_authoritative_evidence"),
        "warning": payload.get("warning", "Synthetic evidence validates system behavior only; it is not live evidence."),
        "total": len(decisions),
        "resolver": _score_cases(decisions),
        "baseline": _score_cases(baseline_rows),
        "by_scenario": by_scenario,
        "by_attribute": by_attribute,
        "decisions": decisions,
    }
