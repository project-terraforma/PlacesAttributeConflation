"""Audit prioritized PAC workplan batches before evidence collection."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


FIRST50_GATE_THRESHOLDS = {
    "min_website_case_share": 0.70,
    "min_p0_case_share": 0.60,
    "max_empty_query_count": 0,
    "max_missing_priority_bucket_count": 0,
    "max_duplicate_query_rate": 0.20,
    "max_template_query_duplicates": 3,
}


def _clean(value: object) -> str:
    return str(value or "").strip()


def _case_id(row: dict[str, object]) -> str:
    return _clean(row.get("case_id")) or _clean(row.get("id"))


def _pair(row: dict[str, object]) -> tuple[str, str]:
    return _case_id(row), _clean(row.get("attribute"))


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((key, value) for key, value in counter.items() if key))


def _query_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    query_counts = Counter(_clean(row.get("query")) for row in rows if _clean(row.get("query")))
    duplicate_query_count = sum(count - 1 for count in query_counts.values() if count > 1)
    return {
        "duplicate_query_count": duplicate_query_count,
        "duplicate_query_rate": _rate(duplicate_query_count, len(rows)),
        "max_query_occurrences": max(query_counts.values(), default=0),
        "most_common_queries": [
            {"query": query, "count": count}
            for query, count in sorted(query_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def load_workplan_csv(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_workplan_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    rows_total = len(rows)
    query_stats = _query_stats(rows)

    pairs: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        case_id, attribute = _pair(row)
        if not case_id or not attribute:
            continue
        pair = pairs.setdefault((case_id, attribute), {"case_id": case_id, "attribute": attribute, "p0": False})
        if _clean(row.get("priority_bucket")).upper().startswith("P0"):
            pair["p0"] = True

    unique_pairs = len(pairs)
    website_pairs = sum(1 for _, attribute in pairs if attribute.lower() == "website")
    p0_pairs = sum(1 for pair in pairs.values() if pair["p0"])

    return {
        "rows_total": rows_total,
        "unique_case_attribute_pairs": unique_pairs,
        "duplicate_query_count": query_stats["duplicate_query_count"],
        "duplicate_query_rate": query_stats["duplicate_query_rate"],
        "max_query_occurrences": query_stats["max_query_occurrences"],
        "most_common_queries": query_stats["most_common_queries"],
        "empty_query_count": sum(1 for row in rows if not _clean(row.get("query"))),
        "missing_priority_bucket_count": sum(1 for row in rows if not _clean(row.get("priority_bucket"))),
        "missing_case_type_guess_count": sum(1 for row in rows if not _clean(row.get("case_type_guess"))),
        "missing_website_label_guess_for_website_count": sum(
            1
            for row in rows
            if _clean(row.get("attribute")).lower() == "website" and not _clean(row.get("website_label_guess"))
        ),
        "rows_by_priority_bucket": _sorted_counts(Counter(_clean(row.get("priority_bucket")) for row in rows)),
        "rows_by_attribute": _sorted_counts(Counter(_clean(row.get("attribute")) for row in rows)),
        "rows_by_layer": _sorted_counts(Counter(_clean(row.get("layer")) for row in rows)),
        "website_case_share": _rate(website_pairs, unique_pairs),
        "p0_case_share": _rate(p0_pairs, unique_pairs),
        "selected_case_ids": sorted({_case_id(row) for row in rows if _case_id(row)}),
        "selected_attributes": sorted({_clean(row.get("attribute")) for row in rows if _clean(row.get("attribute"))}),
    }


def evaluate_first50_gate(report: dict[str, object]) -> dict[str, object]:
    template_rows = int(report.get("template_rows_total", 0) or 0)
    if template_rows:
        query_rate = float(report.get("template_duplicate_query_rate", 0.0))
        cap_respected = bool(report.get("template_duplicate_query_cap_respected", False))
        query_check = query_rate <= FIRST50_GATE_THRESHOLDS["max_duplicate_query_rate"] or cap_respected
        query_scope = "evidence_template"
    else:
        query_rate = float(report["duplicate_query_rate"])
        cap_respected = False
        query_check = query_rate <= FIRST50_GATE_THRESHOLDS["max_duplicate_query_rate"]
        query_scope = "input_rows"
    checks = {
        "website_case_share": float(report["website_case_share"]) >= FIRST50_GATE_THRESHOLDS["min_website_case_share"],
        "p0_case_share": float(report["p0_case_share"]) >= FIRST50_GATE_THRESHOLDS["min_p0_case_share"],
        "empty_query_count": int(report["empty_query_count"]) <= FIRST50_GATE_THRESHOLDS["max_empty_query_count"],
        "missing_priority_bucket_count": int(report["missing_priority_bucket_count"])
        <= FIRST50_GATE_THRESHOLDS["max_missing_priority_bucket_count"],
        "duplicate_query_rate": query_check,
    }
    return {
        "name": "first50",
        "passed": all(checks.values()),
        "checks": checks,
        "query_evaluation_scope": query_scope,
        "query_duplicate_rate_evaluated": query_rate,
        "template_duplicate_query_cap_respected": cap_respected,
        "thresholds": FIRST50_GATE_THRESHOLDS,
    }


def _template_path_for_input(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.name.startswith("evidence_template_"):
        return candidate
    if candidate.name.startswith("batch_"):
        template = candidate.with_name(candidate.name.replace("batch_", "evidence_template_", 1))
        return template if template.exists() else None
    return None


def audit_workplan_files(paths: Sequence[str | Path]) -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    inputs = [str(Path(path)) for path in paths]
    for path in paths:
        all_rows.extend(load_workplan_csv(path))
    report = audit_workplan_rows(all_rows)
    report["batch_duplicate_query_count"] = report["duplicate_query_count"]
    report["batch_duplicate_query_rate"] = report["duplicate_query_rate"]

    template_paths: list[Path] = []
    seen_templates: set[Path] = set()
    for path in paths:
        template_path = _template_path_for_input(path)
        if template_path is not None and template_path not in seen_templates:
            template_paths.append(template_path)
            seen_templates.add(template_path)
    template_rows: list[dict[str, object]] = []
    for path in template_paths:
        template_rows.extend(load_workplan_csv(path))
    template_stats = _query_stats(template_rows)
    report.update(
        {
            "template_inputs": [str(path) for path in template_paths],
            "template_input_count": len(template_paths),
            "template_rows_total": len(template_rows),
            "template_duplicate_query_count": template_stats["duplicate_query_count"],
            "template_duplicate_query_rate": template_stats["duplicate_query_rate"],
            "template_max_query_occurrences": template_stats["max_query_occurrences"],
            "template_duplicate_query_cap_respected": template_stats["max_query_occurrences"]
            <= FIRST50_GATE_THRESHOLDS["max_template_query_duplicates"],
            "most_common_template_queries": template_stats["most_common_queries"],
        }
    )
    report["inputs"] = inputs
    report["input_count"] = len(inputs)
    report["gate"] = evaluate_first50_gate(report)
    return report
