"""Build PAC-prioritized evidence workplans from conflict dork rows."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .conflict_priority import PRIORITY_ORDER, enrich_conflict_rows, summarize_priorities


EVIDENCE_TEMPLATE_FIELDS = [
    "case_id",
    "attribute",
    "priority_bucket",
    "case_type_guess",
    "identity_label_guess",
    "website_label_guess",
    "difficulty",
    "layer",
    "query",
    "url",
    "title",
    "page_text",
    "source_type",
    "extracted_value",
    "notes",
]

DEFAULT_MAX_TEMPLATE_QUERY_DUPLICATES = 3


def load_workplan_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row.get("id") or ""), str(row.get("attribute") or "")


def _query_sort_key(row: dict[str, object]) -> tuple[int, str]:
    layer_order = {
        "official": 0,
        "website_validation": 1,
        "identity_drift": 2,
        "corroboration": 3,
        "freshness": 4,
        "fallback": 5,
    }
    return layer_order.get(str(row.get("layer") or ""), 99), str(row.get("query") or "")


def _quote(value: str) -> str:
    return f'"{value}"' if value and " " in value else value


def _query_value(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if parsed.netloc:
        domain = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.strip("/")
        return f"{domain}/{path}" if path else domain
    return raw


def _case_specific_template_query(row: dict[str, object]) -> str:
    values: list[str] = []
    for field in ("current_value", "base_value", "prediction"):
        value = _query_value(row.get(field))
        if value and value not in values:
            values.append(value)
    if not values:
        return str(row.get("query") or "")
    attribute = str(row.get("attribute") or "")
    if attribute == "website":
        return f"{' OR '.join(_quote(value) for value in values[:3])} official website"
    return " ".join(_quote(value) for value in values[:2])


def _select_template_row(
    group: list[dict[str, object]],
    query_counts: Counter[str],
    max_query_duplicates: int,
) -> dict[str, object]:
    ordered = sorted(group, key=_query_sort_key)
    if not ordered:
        return {}
    if max_query_duplicates <= 0:
        return dict(ordered[0])

    for row in ordered:
        query = str(row.get("query") or "").strip()
        if query and query_counts[query] < max_query_duplicates:
            return dict(row)

    best = dict(ordered[0])
    generated_query = _case_specific_template_query(best)
    if generated_query and query_counts[generated_query] < max_query_duplicates:
        best["query"] = generated_query
        return best

    fallback = min(
        ordered,
        key=lambda row: (query_counts[str(row.get("query") or "").strip()],) + _query_sort_key(row),
    )
    return dict(fallback)


def _template_query_stats(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    query_counts = Counter(str(row.get("query") or "").strip() for row in rows if str(row.get("query") or "").strip())
    duplicate_query_count = sum(count - 1 for count in query_counts.values() if count > 1)
    return {
        "template_duplicate_query_count": duplicate_query_count,
        "template_duplicate_query_rate": duplicate_query_count / len(rows) if rows else 0.0,
        "template_max_query_occurrences": max(query_counts.values(), default=0),
        "most_common_template_queries": [
            {"query": query, "count": count}
            for query, count in sorted(query_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def _case_sort_key(rows: list[dict[str, object]]) -> tuple[int, str, str, str]:
    head = rows[0]
    return (
        PRIORITY_ORDER.get(str(head.get("priority_bucket") or ""), 99),
        str(head.get("difficulty") or ""),
        str(head.get("attribute") or ""),
        str(head.get("id") or ""),
    )


def _write_csv(rows: list[dict[str, object]], path: str | Path) -> Path:
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


def _write_evidence_template(
    groups: list[list[dict[str, object]]],
    path: str | Path,
    *,
    query_counts: Counter[str],
    max_query_duplicates: int,
) -> tuple[Path, list[dict[str, object]]]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    template_rows: list[dict[str, object]] = []
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_TEMPLATE_FIELDS)
        writer.writeheader()
        for group in groups:
            best = _select_template_row(group, query_counts, max_query_duplicates)
            template_row = {
                "case_id": best.get("id", ""),
                "attribute": best.get("attribute", ""),
                "priority_bucket": best.get("priority_bucket", ""),
                "case_type_guess": best.get("case_type_guess", ""),
                "identity_label_guess": best.get("identity_label_guess", ""),
                "website_label_guess": best.get("website_label_guess", ""),
                "difficulty": best.get("difficulty", ""),
                "layer": best.get("layer", ""),
                "query": best.get("query", ""),
                "url": "",
                "title": "",
                "page_text": "",
                "source_type": "",
                "extracted_value": "",
                "notes": "",
            }
            writer.writerow(template_row)
            template_rows.append(template_row)
            query = str(template_row.get("query") or "").strip()
            if query:
                query_counts[query] += 1
    return out, template_rows


def build_prioritized_evidence_workplan(
    rows: Iterable[dict[str, object]],
    output_dir: str | Path,
    *,
    cases_per_batch: int = 25,
    batch_count: int = 2,
    max_template_query_duplicates: int = DEFAULT_MAX_TEMPLATE_QUERY_DUPLICATES,
) -> dict[str, object]:
    """Write prioritized conflict/evidence batches for first-pass PAC review.

    The unit of selection is a case/attribute pair. All queries for a selected
    pair are kept together, then sorted so authoritative layers appear first.
    """

    enriched = enrich_conflict_rows(rows)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in enriched:
        key = _row_key(row)
        if not key[0] or not key[1]:
            continue
        grouped.setdefault(key, []).append(row)

    sorted_groups = sorted(grouped.values(), key=_case_sort_key)
    selected_groups = sorted_groups[: max(0, batch_count) * max(1, cases_per_batch)]
    batches = [
        selected_groups[index : index + max(1, cases_per_batch)]
        for index in range(0, len(selected_groups), max(1, cases_per_batch))
    ][: max(0, batch_count)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    selected_rows = 0
    selected_case_attributes = 0
    selected_bucket_counts: Counter[str] = Counter()
    template_query_counts: Counter[str] = Counter()
    all_template_rows: list[dict[str, object]] = []

    for idx, groups in enumerate(batches, start=1):
        batch_rows: list[dict[str, object]] = []
        for group in groups:
            ordered_group = sorted(group, key=_query_sort_key)
            batch_rows.extend(ordered_group)
            selected_bucket_counts[str(ordered_group[0].get("priority_bucket") or "")] += 1
        selected_rows += len(batch_rows)
        selected_case_attributes += len(groups)
        batch_path = out_dir / f"batch_{idx:03d}.csv"
        evidence_template = out_dir / f"evidence_template_{idx:03d}.csv"
        _write_csv(batch_rows, batch_path)
        evidence_template, template_rows = _write_evidence_template(
            groups,
            evidence_template,
            query_counts=template_query_counts,
            max_query_duplicates=max_template_query_duplicates,
        )
        all_template_rows.extend(template_rows)
        template_stats = _template_query_stats(template_rows)
        files.append(
            {
                "batch": idx,
                "case_attributes": len(groups),
                "rows": len(batch_rows),
                "template_rows": len(template_rows),
                "template_duplicate_query_count": template_stats["template_duplicate_query_count"],
                "template_duplicate_query_rate": template_stats["template_duplicate_query_rate"],
                "priority_buckets": dict(sorted(Counter(str(group[0].get("priority_bucket") or "") for group in groups).items())),
                "path": str(batch_path),
                "evidence_template": str(evidence_template),
            }
        )

    template_stats = _template_query_stats(all_template_rows)
    manifest = {
        "output_dir": str(out_dir),
        "ranking_strategy": "pac_priority_bucket",
        "cases_per_batch": cases_per_batch,
        "batch_count_requested": batch_count,
        "max_template_query_duplicates": max_template_query_duplicates,
        "input_rows": len(enriched),
        "candidate_case_attributes": len(sorted_groups),
        "selected_case_attributes": selected_case_attributes,
        "selected_rows": selected_rows,
        "selected_priority_buckets": dict(sorted(selected_bucket_counts.items())),
        **template_stats,
        "priority_summary": summarize_priorities(enriched),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_prioritized_evidence_workplan_from_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    cases_per_batch: int = 25,
    batch_count: int = 2,
    max_template_query_duplicates: int = DEFAULT_MAX_TEMPLATE_QUERY_DUPLICATES,
) -> dict[str, object]:
    manifest = build_prioritized_evidence_workplan(
        load_workplan_rows(input_csv),
        output_dir,
        cases_per_batch=cases_per_batch,
        batch_count=batch_count,
        max_template_query_duplicates=max_template_query_duplicates,
    )
    manifest["input_csv"] = str(Path(input_csv))
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
