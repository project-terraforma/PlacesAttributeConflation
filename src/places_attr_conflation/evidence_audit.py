"""Audit enriched evidence CSVs without fetching URLs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


EVIDENCE_GATE_THRESHOLDS = {
    "first50": {
        "min_rows_with_url": 25,
        "min_fetch_ok_rate": 0.70,
        "min_content_hash_rate": 0.70,
        "min_page_text_rate": 0.70,
    },
    "v1": {
        "min_fetch_ok_rate": 0.75,
        "min_content_hash_rate": 0.75,
        "min_page_text_rate": 0.75,
    },
}

IDENTITY_STATUSES = {"moved", "permanently_closed", "temporarily_closed", "closed", "stale", "inactive"}
STALE_WEBSITE_LABELS = {"OFFICIAL_STALE", "OFFICIAL_DEAD", "OFFICIAL_WRONG_ENTITY", "PARKED_DOMAIN"}
IDENTITY_LABEL_SIGNALS = {
    "MOVED_ENTITY",
    "RENAMED_ENTITY",
    "OWNERSHIP_CHANGE",
    "NEW_ENTITY_SAME_ADDRESS",
    "STALE_OFFICIAL_SITE",
    "BRANCH_AMBIGUITY",
    "TEMPORARY_CLOSURE",
    "PERMANENT_CLOSURE",
}


def _clean(value: object) -> str:
    return str(value or "").strip()


def _truthy(value: object) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "y"}


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((key, value) for key, value in counter.items() if key))


def _field_present(rows: list[dict[str, object]], field: str) -> bool:
    return any(field in row for row in rows)


def _count_present(rows: list[dict[str, object]], field: str) -> int:
    return sum(1 for row in rows if _clean(row.get(field)))


def _has_real_page_text(row: dict[str, object]) -> bool:
    text = _clean(row.get("page_text"))
    return bool(text and not text.startswith("Public website URL evidence:"))


def _has_identity_signal(row: dict[str, object]) -> bool:
    status = _clean(row.get("detected_status")).lower()
    website_label = _clean(row.get("website_label_guess")).upper()
    identity_label = _clean(row.get("identity_label_guess")).upper()
    return bool(
        _clean(row.get("identity_claims"))
        or status in IDENTITY_STATUSES
        or website_label in STALE_WEBSITE_LABELS
        or identity_label in IDENTITY_LABEL_SIGNALS
    )


def load_evidence_csv(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_evidence_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    rows_total = len(rows)
    rows_with_url = _count_present(rows, "url")
    denominator = rows_with_url
    fetch_status_counts = Counter(_clean(row.get("fetch_status")).lower() for row in rows)
    fetch_ok_count = fetch_status_counts.get("ok", 0)
    fetch_error_count = sum(
        1 for row in rows if _clean(row.get("fetch_status")).lower() == "error" or bool(_clean(row.get("fetch_error")))
    )
    rows_with_page_text = sum(1 for row in rows if _has_real_page_text(row))

    report: dict[str, object] = {
        "rows_total": rows_total,
        "rows_with_url": rows_with_url,
        "rows_missing_url": rows_total - rows_with_url,
        "fetch_ok_count": fetch_ok_count,
        "fetch_error_count": fetch_error_count,
        "fetch_ok_rate": _rate(fetch_ok_count, denominator),
        "rows_with_final_url": _count_present(rows, "final_url"),
        "rows_with_content_hash": _count_present(rows, "content_hash"),
        "rows_with_page_text": rows_with_page_text,
        "rows_with_full_text_path": _count_present(rows, "page_text_full_path"),
        "rows_with_page_text_excerpt": _count_present(rows, "page_text_excerpt"),
        "rows_with_schema_org": sum(1 for row in rows if _truthy(row.get("schema_org_detected"))),
        "rows_with_localbusiness_schema": sum(1 for row in rows if _truthy(row.get("localbusiness_schema_detected"))),
        "rows_with_detected_phone": _count_present(rows, "detected_phone"),
        "rows_with_detected_address": _count_present(rows, "detected_address"),
        "rows_with_detected_name": _count_present(rows, "detected_name"),
        "rows_with_detected_status": sum(
            1 for row in rows if _clean(row.get("detected_status")) and _clean(row.get("detected_status")).lower() != "unknown"
        ),
        "rows_with_identity_claims": _count_present(rows, "identity_claims"),
        "stale_or_identity_signal_count": sum(1 for row in rows if _has_identity_signal(row)),
        "rows_by_detected_status": _sorted_counts(Counter(_clean(row.get("detected_status")) for row in rows)),
        "rows_by_fetch_status": _sorted_counts(Counter(_clean(row.get("fetch_status")).lower() for row in rows)),
        "url_rate": _rate(rows_with_url, rows_total),
        "content_hash_rate": _rate(_count_present(rows, "content_hash"), denominator),
        "page_text_rate": _rate(rows_with_page_text, denominator),
        "full_text_path_rate": _rate(_count_present(rows, "page_text_full_path"), denominator),
        "schema_org_rate": _rate(sum(1 for row in rows if _truthy(row.get("schema_org_detected"))), denominator),
        "identity_claim_rate": _rate(_count_present(rows, "identity_claims"), denominator),
    }

    for field in ("source_type", "priority_bucket", "website_label_guess", "identity_label_guess"):
        if _field_present(rows, field):
            report[f"rows_by_{field}"] = _sorted_counts(Counter(_clean(row.get(field)) for row in rows))
    return report


def evaluate_evidence_gate(report: dict[str, object], gate_name: str = "first50") -> dict[str, object]:
    thresholds = EVIDENCE_GATE_THRESHOLDS[gate_name]
    checks: dict[str, bool] = {}
    if "min_rows_with_url" in thresholds:
        checks["rows_with_url"] = int(report["rows_with_url"]) >= thresholds["min_rows_with_url"]
    checks["fetch_ok_rate"] = float(report["fetch_ok_rate"]) >= thresholds["min_fetch_ok_rate"]
    checks["content_hash_rate"] = float(report["content_hash_rate"]) >= thresholds["min_content_hash_rate"]
    checks["page_text_rate"] = float(report["page_text_rate"]) >= thresholds["min_page_text_rate"]
    return {
        "name": gate_name,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
    }


def audit_evidence_files(paths: Sequence[str | Path], *, gate_name: str = "first50") -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    inputs = [str(Path(path)) for path in paths]
    for path in paths:
        all_rows.extend(load_evidence_csv(path))
    report = audit_evidence_rows(all_rows)
    report["inputs"] = inputs
    report["input_count"] = len(inputs)
    report["gate"] = evaluate_evidence_gate(report, gate_name)
    return report
