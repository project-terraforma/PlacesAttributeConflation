"""Export targeted dork queries for labeled attribute conflicts.

This is the bridge between "we have lots of labeled conflicts" and
"we have replayable evidence to improve retrieval/resolution".

It intentionally does not fetch the web. It generates:
- which cases need evidence,
- which attribute is disputed,
- candidates (current/base),
- layered authoritative queries (official/corroboration/freshness/fallback).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .dorking import build_multi_layer_plan
from .golden import _normalize


EVIDENCE_ATTRIBUTES = ("website", "name", "category")
EVIDENCE_TEMPLATE_FIELDS = ["case_id", "attribute", "layer", "query", "url", "title", "page_text", "source_type", "extracted_value", "notes"]
PRIORITY_ORDER = {"needs_evidence": 0, "baseline_wrong": 1, "baseline_missing": 2, "low": 3}
ATTRIBUTE_ORDER = {attribute: idx for idx, attribute in enumerate(EVIDENCE_ATTRIBUTES)}
LAYER_ORDER = {"official": 0, "government": 1, "business_registry": 2, "registry": 3, "corroboration": 4, "freshness": 5, "fallback": 6}


@dataclass(frozen=True)
class ConflictDorkRow:
    id: str
    base_id: str
    attribute: str
    truth: str
    truth_source: str
    prediction: str
    baseline: str
    correct: bool
    needs_evidence: bool
    current_value: str
    base_value: str
    preferred_sources: str
    layer: str
    query: str
    priority: str


def load_conflict_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _priority(row: dict[str, str]) -> str:
    # The queue should focus on what will move metrics:
    # 1) baseline wrong
    # 2) baseline abstained / missing
    # 3) labeled conflict needing evidence
    correct = str(row.get("correct", "")).lower() in {"true", "1", "yes"}
    prediction = str(row.get("prediction", "")).strip()
    if not correct:
        return "baseline_wrong"
    if not prediction:
        return "baseline_missing"
    if str(row.get("needs_evidence", "")).lower() in {"true", "1", "yes"}:
        return "needs_evidence"
    return "low"


def build_conflict_dork_rows(
    conflict_rows: Iterable[dict[str, str]],
    *,
    max_queries_per_case: int = 8,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in conflict_rows:
        attribute = str(row.get("attribute", "")).strip()
        place = {
            "name": str(row.get("name") or ""),
            "city": str(row.get("city") or ""),
            "region": str(row.get("region") or ""),
            "address": str(row.get("current_value") if attribute == "address" else ""),
            "phone": str(row.get("current_value") if attribute == "phone" else ""),
            "website": str(row.get("current_value") if attribute == "website" else ""),
        }

        # When conflictset rows come from project_a, name/city/region aren't present.
        # Anchor with what we reliably have: current/base candidates and attribute.
        # The dorking plan will still include authority operators and site-restricted
        # queries when a domain can be derived from a candidate website.
        if attribute in {"name", "category"}:
            place["name"] = str(row.get("current_value") or row.get("base_value") or "")
        if attribute == "address":
            place["address"] = str(row.get("current_value") or row.get("base_value") or "")
        if attribute == "phone":
            place["phone"] = str(row.get("current_value") or row.get("base_value") or "")
        if attribute == "website":
            place["website"] = str(row.get("current_value") or row.get("base_value") or "")

        plan = build_multi_layer_plan(place, attribute if attribute else "website")
        preferred_sources = ",".join(plan.layers[0].preferred_sources) if plan.layers else "official_site,government,business_registry"
        priority = _priority(row)

        case_id = str(row.get("id") or "")
        base_id = str(row.get("base_id") or "")
        truth = str(row.get("truth") or "")
        truth_source = str(row.get("truth_source") or "")
        prediction = str(row.get("prediction") or "")
        baseline = str(row.get("baseline") or "")
        correct = str(row.get("correct", "")).lower() in {"true", "1", "yes"}
        needs_evidence = str(row.get("needs_evidence", "")).lower() in {"true", "1", "yes"}
        current_value = str(row.get("current_value") or "")
        base_value = str(row.get("base_value") or "")

        emitted = 0
        for layer in plan.layers:
            for query in layer.queries:
                if not query.strip():
                    continue
                output.append(
                    asdict(
                        ConflictDorkRow(
                            id=case_id,
                            base_id=base_id,
                            attribute=attribute,
                            truth=truth,
                            truth_source=truth_source,
                            prediction=prediction,
                            baseline=baseline,
                            correct=correct,
                            needs_evidence=needs_evidence,
                            current_value=current_value,
                            base_value=base_value,
                            preferred_sources=preferred_sources,
                            layer=layer.name,
                            query=query,
                            priority=priority,
                        )
                    )
                )
                emitted += 1
                if emitted >= max_queries_per_case:
                    break
            if emitted >= max_queries_per_case:
                break

        # When attribute field is missing or unparsable, skip quietly.
        # Also skip degenerate rows where candidates normalize equal (not a true conflict).
        if attribute and _normalize(attribute, current_value) == _normalize(attribute, base_value):
            # Keep only if baseline was wrong; otherwise it's not a useful evidence target.
            if correct:
                output = output[:-emitted]

    return output


def write_conflict_dork_csv(rows: list[dict[str, str]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    return str(row.get("id", "")), str(row.get("attribute", ""))


def _batch_sort_key(row: dict[str, str]) -> tuple[int, str]:
    return LAYER_ORDER.get(str(row.get("layer", "")), 99), str(row.get("query", ""))


def _episode_sort_key(rows: list[dict[str, str]]) -> tuple[int, int, int, str, str, str]:
    head = rows[0]
    best_layer = min((LAYER_ORDER.get(str(row.get("layer", "")), 99) for row in rows), default=99)
    return (
        PRIORITY_ORDER.get(str(head.get("priority", "")), 99),
        ATTRIBUTE_ORDER.get(str(head.get("attribute", "")), 99),
        best_layer,
        str(head.get("base_id", "")),
        str(head.get("id", "")),
        min((str(row.get("query", "")) for row in rows), default=""),
    )


def _load_batch_dir_rows(batch_dir: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(Path(batch_dir).glob("batch_*.csv")):
        for row in load_conflict_csv(path):
            rows.append({key: ("" if value is None else str(value)) for key, value in row.items()})
    return rows


def _evidenced_episode_keys(replay_dir: str | Path | None) -> set[tuple[str, str]]:
    if replay_dir is None:
        return set()
    root = Path(replay_dir)
    if not root.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            pages = sum(
                len(attempt.get("fetched_pages", []))
                for attempt in episode.get("search_attempts", [])
                if isinstance(attempt, dict)
            )
            if pages:
                keys.add((str(episode.get("case_id", "")), str(episode.get("attribute", ""))))
    return keys


def _write_evidence_template(rows: list[dict[str, str]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_key.setdefault(_row_key(row), []).append(row)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_TEMPLATE_FIELDS)
        writer.writeheader()
        for (case_id, attribute), grouped_rows in sorted(by_key.items(), key=lambda item: _episode_sort_key(item[1])):
            best = sorted(grouped_rows, key=_batch_sort_key)[0]
            writer.writerow(
                {
                    "case_id": case_id,
                    "attribute": attribute,
                    "layer": best.get("layer", ""),
                    "query": best.get("query", ""),
                    "url": "",
                    "title": "",
                    "page_text": "",
                    "source_type": "",
                    "extracted_value": "",
                    "notes": "",
                }
            )
    return out


def build_evidence_workplan_batches(
    batch_dir: str | Path,
    output_dir: str | Path,
    *,
    replay_dir: str | Path | None = None,
    attributes: Iterable[str] = EVIDENCE_ATTRIBUTES,
    batch_count: int = 25,
    cases_per_batch: int = 25,
) -> dict[str, object]:
    """Create small deterministic evidence work queues from conflict dork batches."""
    allowed_attributes = {str(attribute) for attribute in attributes}
    evidenced = _evidenced_episode_keys(replay_dir)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in _load_batch_dir_rows(batch_dir):
        key = _row_key(row)
        if not key[0] or not key[1] or key[1] not in allowed_attributes or key in evidenced:
            continue
        grouped.setdefault(key, []).append(row)

    sorted_groups = sorted(grouped.values(), key=_episode_sort_key)
    selected_groups = sorted_groups[: max(0, batch_count) * max(1, cases_per_batch)]
    batches = [selected_groups[index : index + max(1, cases_per_batch)] for index in range(0, len(selected_groups), max(1, cases_per_batch))]
    batches = batches[: max(0, batch_count)]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    total_rows = 0
    for idx, groups in enumerate(batches, start=1):
        rows: list[dict[str, str]] = []
        for group in groups:
            rows.extend(sorted(group, key=_batch_sort_key))
        total_rows += len(rows)
        batch_path = out_dir / f"batch_{idx:03d}.csv"
        evidence_template = out_dir / f"evidence_template_{idx:03d}.csv"
        write_conflict_dork_csv(rows, batch_path)
        _write_evidence_template(rows, evidence_template)
        files.append(
            {
                "batch": idx,
                "case_attributes": len(groups),
                "rows": len(rows),
                "path": str(batch_path),
                "evidence_template": str(evidence_template),
            }
        )

    manifest = {
        "input_batch_dir": str(Path(batch_dir)),
        "output_dir": str(out_dir),
        "replay_dir": "" if replay_dir is None else str(Path(replay_dir)),
        "attributes": sorted(allowed_attributes),
        "excluded_evidenced_case_attributes": len(evidenced),
        "remaining_case_attributes": len(sorted_groups),
        "selected_case_attributes": sum(int(item["case_attributes"]) for item in files),
        "selected_rows": total_rows,
        "batch_count_requested": batch_count,
        "cases_per_batch": cases_per_batch,
        "batches": len(files),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def split_conflict_dork_csv_by_case(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    cases_per_batch: int = 250,
) -> dict[str, object]:
    """Split a conflict dork CSV into batches, grouping by id."""
    rows = load_conflict_csv(input_csv)
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        case_id = str(row.get("id") or "")
        by_id.setdefault(case_id, []).append(row)

    case_ids = [case_id for case_id in by_id.keys() if case_id]
    batches: list[list[str]] = []
    for i in range(0, len(case_ids), max(1, cases_per_batch)):
        batches.append(case_ids[i : i + cases_per_batch])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[dict[str, object]] = []
    total_rows = 0
    for idx, batch_case_ids in enumerate(batches, start=1):
        batch_rows: list[dict[str, str]] = []
        for case_id in batch_case_ids:
            batch_rows.extend(by_id.get(case_id, []))
        total_rows += len(batch_rows)
        batch_path = out_dir / f"batch_{idx:02d}.csv"
        write_conflict_dork_csv(batch_rows, batch_path)
        batch_files.append(
            {
                "batch": idx,
                "cases": len(batch_case_ids),
                "rows": len(batch_rows),
                "path": str(batch_path),
            }
        )

    manifest = {
        "input_csv": str(Path(input_csv)),
        "output_dir": str(out_dir),
        "cases_per_batch": cases_per_batch,
        "total_cases": len(case_ids),
        "total_rows": total_rows,
        "batches": len(batches),
        "files": batch_files,
    }
    (out_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
