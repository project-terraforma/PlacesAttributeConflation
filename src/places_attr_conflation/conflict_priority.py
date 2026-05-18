"""PAC-specific conflict prioritization for replay-corpus construction.

This module is intentionally side-effect free. It can be used as a post-process
on conflict-dork CSV rows before changing the existing dork-generation path.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PAC_PRIORITY_BUCKETS = (
    "P0_WEBSITE_BASELINE_WRONG",
    "P0_WEBSITE_MISSING",
    "P0_WEBSITE_AGGREGATOR_OR_SOCIAL",
    "P0_WEBSITE_CHAIN_OR_BRANCH",
    "P0_OVERTURE_WEBSITE_GAP",
    "P0_IDENTITY_DRIFT_WEBSITE",
    "P1_CATEGORY_TAXONOMY",
    "P1_NAME_IDENTITY",
    "P1_WEBSITE_CONFLICT",
    "P2_OTHER",
)

PRIORITY_ORDER = {bucket: idx for idx, bucket in enumerate(PAC_PRIORITY_BUCKETS)}

AGGREGATOR_OR_SOCIAL_DOMAINS = (
    "yelp.com",
    "tripadvisor.com",
    "facebook.com",
    "instagram.com",
    "doordash.com",
    "ubereats.com",
    "grubhub.com",
    "foursquare.com",
    "opentable.com",
    "linktr.ee",
)

IDENTITY_DRIFT_TERMS = (
    "moved",
    "formerly",
    "former",
    "closed",
    "new ownership",
    "grand opening",
    "renamed",
)

CHAIN_OR_BRANCH_TERMS = (
    "/locations",
    "/location",
    "store-locator",
    "branches",
    "branch",
    "near-me",
)


@dataclass(frozen=True)
class ConflictPriority:
    priority_bucket: str
    case_type_guess: str
    identity_label_guess: str
    website_label_guess: str
    difficulty: str


def _text(*values: object) -> str:
    return " ".join(str(value or "").lower() for value in values)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _missing(value: object) -> bool:
    return not str(value or "").strip()


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def classify_conflict_priority(row: dict[str, object]) -> ConflictPriority:
    """Assign a PAC priority bucket and weak label guesses to one conflict row."""

    attribute = str(row.get("attribute") or "").strip()
    current_value = str(row.get("current_value") or "")
    base_value = str(row.get("base_value") or "")
    prediction = str(row.get("prediction") or "")
    correct = _truthy(row.get("correct"))
    truth_source = str(row.get("truth_source") or "")
    row_text = _text(*row.values())
    values_text = _text(current_value, base_value, prediction)

    if attribute == "website":
        if _missing(current_value) or _missing(base_value) or _missing(prediction):
            bucket = "P0_WEBSITE_MISSING"
        elif _has_any(values_text, AGGREGATOR_OR_SOCIAL_DOMAINS):
            bucket = "P0_WEBSITE_AGGREGATOR_OR_SOCIAL"
        elif _has_any(values_text, CHAIN_OR_BRANCH_TERMS) or "branch" in row_text:
            bucket = "P0_WEBSITE_CHAIN_OR_BRANCH"
        elif "overture" in truth_source.lower() or "gers" in row_text:
            bucket = "P0_OVERTURE_WEBSITE_GAP"
        elif not correct:
            bucket = "P0_WEBSITE_BASELINE_WRONG"
        else:
            bucket = "P1_WEBSITE_CONFLICT"
    elif _has_any(row_text, IDENTITY_DRIFT_TERMS):
        bucket = "P0_IDENTITY_DRIFT_WEBSITE"
    elif attribute == "category":
        bucket = "P1_CATEGORY_TAXONOMY"
    elif attribute == "name":
        bucket = "P1_NAME_IDENTITY"
    else:
        bucket = "P2_OTHER"

    return ConflictPriority(
        priority_bucket=bucket,
        case_type_guess=_case_type_guess(bucket),
        identity_label_guess=_identity_label_guess(bucket, values_text),
        website_label_guess=_website_label_guess(bucket, values_text),
        difficulty=_difficulty(bucket),
    )


def _case_type_guess(bucket: str) -> str:
    return {
        "P0_WEBSITE_BASELINE_WRONG": "WEBSITE_BASELINE_WRONG",
        "P0_WEBSITE_MISSING": "MISSING_BUT_DISCOVERABLE_WEBSITE",
        "P0_WEBSITE_AGGREGATOR_OR_SOCIAL": "AGGREGATOR_OR_SOCIAL_AS_WEBSITE",
        "P0_WEBSITE_CHAIN_OR_BRANCH": "CHAIN_OR_BRANCH_WEBSITE_AMBIGUITY",
        "P0_OVERTURE_WEBSITE_GAP": "OVERTURE_WEBSITE_GAP",
        "P0_IDENTITY_DRIFT_WEBSITE": "IDENTITY_DRIFT_WEBSITE",
        "P1_CATEGORY_TAXONOMY": "CATEGORY_TAXONOMY_CONFLICT",
        "P1_NAME_IDENTITY": "NAME_IDENTITY_CONFLICT",
        "P1_WEBSITE_CONFLICT": "WEBSITE_CONFLICT",
    }.get(bucket, "OTHER_CONFLICT")


def _identity_label_guess(bucket: str, values_text: str) -> str:
    if bucket == "P0_WEBSITE_CHAIN_OR_BRANCH":
        return "BRANCH_AMBIGUITY"
    if "formerly" in values_text or "former" in values_text or "renamed" in values_text:
        return "RENAMED_ENTITY"
    if "moved" in values_text:
        return "MOVED_ENTITY"
    if "closed" in values_text:
        return "PERMANENT_CLOSURE"
    if bucket == "P0_IDENTITY_DRIFT_WEBSITE":
        return "UNKNOWN_IDENTITY"
    return ""


def _website_label_guess(bucket: str, values_text: str) -> str:
    if bucket == "P0_WEBSITE_MISSING":
        return "NO_WEBSITE_FOUND"
    if "facebook.com" in values_text or "instagram.com" in values_text:
        return "SOCIAL_ONLY_CURRENT"
    if bucket == "P0_WEBSITE_AGGREGATOR_OR_SOCIAL":
        return "AGGREGATOR_ONLY"
    if bucket == "P0_WEBSITE_CHAIN_OR_BRANCH":
        return "OFFICIAL_CHAIN_ONLY"
    if bucket == "P0_WEBSITE_BASELINE_WRONG":
        return "AMBIGUOUS_WEBSITE"
    return ""


def _difficulty(bucket: str) -> str:
    if bucket.startswith("P0_"):
        return "HARD"
    if bucket.startswith("P1_"):
        return "MEDIUM"
    return "EASY"


def enrich_conflict_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        priority = classify_conflict_priority(row)
        enriched.append({**row, **asdict(priority)})
    return sorted(enriched, key=_sort_key)


def _sort_key(row: dict[str, object]) -> tuple[int, str, str, str]:
    return (
        PRIORITY_ORDER.get(str(row.get("priority_bucket") or ""), 99),
        str(row.get("attribute") or ""),
        str(row.get("id") or ""),
        str(row.get("query") or ""),
    )


def summarize_priorities(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    by_bucket = Counter(str(row.get("priority_bucket") or "") for row in rows if row.get("priority_bucket"))
    by_attribute = Counter(str(row.get("attribute") or "") for row in rows if row.get("attribute"))
    return {
        "rows_total": len(rows),
        "rows_by_priority_bucket": dict(sorted(by_bucket.items())),
        "rows_by_attribute": dict(sorted(by_attribute.items())),
        "p0_rows": sum(count for bucket, count in by_bucket.items() if bucket.startswith("P0_")),
    }


def load_conflict_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_conflict_rows(rows: list[dict[str, object]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out
