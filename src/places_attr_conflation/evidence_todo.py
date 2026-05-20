"""Build human-friendly evidence URL todo sheets from PAC workplans."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


DEFAULT_TARGETS = [
    "official source for the candidate value",
    "corroborating directory or registry page",
]

TARGETS_BY_BUCKET = {
    "P0_WEBSITE_MISSING": [
        "official website",
        "contact/about page",
        "social page only if official unavailable",
    ],
    "P0_WEBSITE_AGGREGATOR_OR_SOCIAL": [
        "official website",
        "aggregator/social page that caused conflict",
        "contact/location page",
    ],
    "P0_WEBSITE_CHAIN_OR_BRANCH": [
        "brand location page",
        "branch-specific page",
        "chain homepage if it is the wrong candidate",
    ],
    "P0_IDENTITY_DRIFT_WEBSITE": [
        "moved/formerly/new ownership page",
        "old official domain",
        "current official page",
        "directory/social corroboration",
    ],
    "P1_CATEGORY_TAXONOMY": [
        "official services/about/menu page",
        "Overture/category context if available",
    ],
}

DISPLAY_FIELDS = [
    "case_id",
    "attribute",
    "priority_bucket",
    "case_type_guess",
    "identity_label_guess",
    "website_label_guess",
    "difficulty",
    "query",
    "current_value",
    "base_value",
    "prediction",
]


def _clean(value: object) -> str:
    return str(value or "").strip()


def _case_id(row: dict[str, object]) -> str:
    return _clean(row.get("case_id")) or _clean(row.get("id"))


def recommended_targets(priority_bucket: str) -> list[str]:
    return list(TARGETS_BY_BUCKET.get(_clean(priority_bucket), DEFAULT_TARGETS))


def load_todo_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_key(row: dict[str, object]) -> tuple[str, str]:
    return _case_id(row), _clean(row.get("attribute"))


def _merge_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    by_key: OrderedDict[tuple[str, str], dict[str, object]] = OrderedDict()
    for row in rows:
        key = _row_key(row)
        if not key[0] or not key[1]:
            continue
        if key not in by_key:
            merged = dict(row)
            merged["case_id"] = key[0]
            by_key[key] = merged
            continue
        merged = by_key[key]
        for field in DISPLAY_FIELDS:
            if not _clean(merged.get(field)) and _clean(row.get(field)):
                merged[field] = row.get(field)
    return list(by_key.values())


def render_evidence_todo(rows: Iterable[dict[str, object]], *, title: str = "Evidence URL Todo") -> str:
    merged_rows = _merge_rows(rows)
    lines = [f"# {title}", "", "Fill only explicit evidence URLs found by human review. Do not decide truth labels here."]
    for index, row in enumerate(merged_rows, start=1):
        case_id = _case_id(row)
        attribute = _clean(row.get("attribute"))
        bucket = _clean(row.get("priority_bucket"))
        lines.extend(["", f"## {index}. {case_id} / {attribute}", ""])
        for field in DISPLAY_FIELDS:
            value = case_id if field == "case_id" else _clean(row.get(field))
            if value:
                lines.append(f"- {field}: {value}")
        lines.extend(["- recommended evidence targets:"])
        for target in recommended_targets(bucket):
            lines.append(f"  - {target}")
        lines.extend(["- evidence URL(s):", "  - "])
    return "\n".join(lines).rstrip() + "\n"


def write_evidence_todo(input_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_evidence_todo(load_todo_rows(input_path)), encoding="utf-8")
    return output
