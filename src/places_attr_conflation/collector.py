"""Human-in-the-loop replay evidence collector.

This module bridges the current project blocker:
we can generate *dork query queues* deterministically, but we still need
replayable evidence at scale to prove targeted retrieval beats loose search.

Because wide crawling is not always available (or reproducible), we provide a
local collector workflow:

1) Load a batch CSV of queued queries (grouped by conflict case id).
2) Present an extremely simple local web UI where a human can:
   - click a search link for a query (Google/Bing/etc.)
   - paste authoritative URLs and minimal extracted values/snippets
3) Save a replay corpus JSON in the repo's stable schema (replay.py).

The saved replay corpus can be benchmarked by the existing harness
(`compare`, `rerank`, and resolver evaluation).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus, load_replay_corpus


@dataclass(frozen=True)
class CollectorCase:
    case_id: str
    base_id: str
    attribute: str
    truth: str
    prediction: str
    baseline: str
    current_value: str
    base_value: str
    preferred_sources: str
    query_rows: list[dict[str, str]]


def load_collector_cases(batch_csv: str | Path) -> list[CollectorCase]:
    path = Path(batch_csv)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("id", ""), []).append(row)

    cases: list[CollectorCase] = []
    for case_id, items in grouped.items():
        if not case_id:
            continue
        head = items[0]
        cases.append(
            CollectorCase(
                case_id=case_id,
                base_id=head.get("base_id", ""),
                attribute=head.get("attribute", ""),
                truth=head.get("truth", ""),
                prediction=head.get("prediction", ""),
                baseline=head.get("baseline", ""),
                current_value=head.get("current_value", ""),
                base_value=head.get("base_value", ""),
                preferred_sources=head.get("preferred_sources", ""),
                query_rows=items,
            )
        )
    return sorted(cases, key=lambda c: (c.attribute, c.case_id))


def _load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(key): ("" if value is None else str(value)) for key, value in row.items()})
    return rows


def _load_evidence_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    evidence_path = Path(path)
    if evidence_path.suffix.lower() == ".json":
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("evidence") or payload.get("pages") or payload.get("rows") or []
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError("Evidence JSON must be a list or an object with evidence/pages/rows")
        return [{str(key): str(value) for key, value in row.items()} for row in rows if isinstance(row, dict)]
    return _load_csv_rows(evidence_path)


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _float_or_default(value: str, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _evidence_page(row: dict[str, str], attribute: str) -> FetchedPage | None:
    url = row.get("url") or row.get("source_url") or row.get("evidence_url") or ""
    if not url:
        return None
    extracted_value = (
        row.get("extracted_value")
        or row.get(f"extracted_{attribute}")
        or row.get(attribute)
        or row.get("candidate_value")
        or ""
    )
    return FetchedPage(
        url=url,
        title=row.get("title", ""),
        page_text=row.get("page_text") or row.get("snippet") or row.get("text") or "",
        source_type=row.get("source_type") or "unknown",
        extracted_values={attribute: extracted_value} if extracted_value else {},
        recency_days=_float_or_none(row.get("recency_days", "")),
        zombie_score=_float_or_default(row.get("zombie_score", "")),
        identity_change_score=_float_or_default(row.get("identity_change_score", "")),
        notes=row.get("notes", ""),
    )


def _group_evidence_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        case_id = row.get("case_id") or row.get("id") or ""
        attribute = row.get("attribute") or ""
        if case_id and attribute:
            grouped.setdefault((case_id, attribute), []).append(row)
    return grouped


def build_seed_replay_episodes(
    batch_csv: str | Path,
    evidence_path: str | Path | None = None,
) -> list[ReplayEpisode]:
    """Build schema-valid replay episodes directly from a conflict dork batch."""
    batch_rows = _load_csv_rows(batch_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in batch_rows:
        case_id = row.get("id", "")
        attribute = row.get("attribute", "")
        if case_id and attribute:
            grouped.setdefault((case_id, attribute), []).append(row)

    evidence_by_case = _group_evidence_rows(_load_evidence_rows(evidence_path))
    episodes: list[ReplayEpisode] = []
    for (case_id, attribute), rows in sorted(grouped.items()):
        head = rows[0]
        pages_by_attempt: dict[tuple[str, str], list[FetchedPage]] = {}
        for evidence_row in evidence_by_case.get((case_id, attribute), []):
            page = _evidence_page(evidence_row, attribute)
            if page is None:
                continue
            layer = evidence_row.get("layer") or "imported"
            query = evidence_row.get("query") or ""
            pages_by_attempt.setdefault((layer, query), []).append(page)

        attempt_keys = {(row.get("layer", ""), row.get("query", "")) for row in rows if row.get("query", "")}
        attempt_keys.update(pages_by_attempt)
        attempts = [
            SearchAttempt(layer=layer or "fallback", query=query, fetched_pages=pages_by_attempt.get((layer, query), []))
            for layer, query in sorted(attempt_keys)
        ]
        episodes.append(
            ReplayEpisode(
                case_id=case_id,
                attribute=attribute,
                place={
                    "base_id": head.get("base_id", ""),
                    "truth_source": head.get("truth_source", ""),
                    "prediction": head.get("prediction", ""),
                    "baseline": head.get("baseline", ""),
                    "correct": head.get("correct", ""),
                    "needs_evidence": head.get("needs_evidence", ""),
                    "current_value": head.get("current_value", ""),
                    "base_value": head.get("base_value", ""),
                    "preferred_sources": head.get("preferred_sources", ""),
                    "priority": head.get("priority", ""),
                },
                gold_value=head.get("truth", ""),
                search_attempts=attempts,
            )
        )
    return episodes


def write_seed_replay_from_batch(
    batch_csv: str | Path,
    out_replay_json: str | Path,
    *,
    evidence_path: str | Path | None = None,
) -> dict[str, object]:
    episodes = build_seed_replay_episodes(batch_csv, evidence_path=evidence_path)
    out = Path(out_replay_json)
    dump_replay_corpus(episodes, out)
    validated = load_replay_corpus(out)
    return {
        "batch": str(batch_csv),
        "evidence_input": "" if evidence_path is None else str(evidence_path),
        "output_replay": str(out),
        "episodes": len(validated),
        "attempts": sum(len(episode.search_attempts) for episode in validated),
        "pages": sum(len(attempt.fetched_pages) for episode in validated for attempt in episode.search_attempts),
        "attributes": sorted({episode.attribute for episode in validated}),
    }


def collector_cases_payload(cases: list[CollectorCase]) -> dict[str, Any]:
    return {
        "cases": [
            {
                "case_id": c.case_id,
                "base_id": c.base_id,
                "attribute": c.attribute,
                "truth": c.truth,
                "prediction": c.prediction,
                "baseline": c.baseline,
                "current_value": c.current_value,
                "base_value": c.base_value,
                "preferred_sources": c.preferred_sources,
                "queries": [
                    {
                        "layer": row.get("layer", ""),
                        "query": row.get("query", ""),
                        "priority": row.get("priority", ""),
                    }
                    for row in c.query_rows
                ],
            }
            for c in cases
        ]
    }


def write_collector_payload(cases: list[CollectorCase], out_json: str | Path) -> Path:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collector_cases_payload(cases), indent=2, sort_keys=True), encoding="utf-8")
    return out


def save_replay_from_collector_payload(payload: dict[str, Any], out_replay_json: str | Path) -> Path:
    episodes: list[ReplayEpisode] = []
    for row in payload.get("episodes", []):
        place = row.get("place", {})
        attempts_payload = row.get("search_attempts", [])
        attempts: list[SearchAttempt] = []
        for attempt in attempts_payload:
            pages: list[FetchedPage] = []
            for page in attempt.get("fetched_pages", []):
                pages.append(FetchedPage.from_dict(page))
            attempts.append(SearchAttempt(layer=str(attempt.get("layer", "")), query=str(attempt.get("query", "")), fetched_pages=pages))
        episodes.append(
            ReplayEpisode(
                case_id=str(row.get("case_id", "")),
                attribute=str(row.get("attribute", "")),
                place={str(k): str(v) for k, v in (place.items() if isinstance(place, dict) else [])},
                gold_value=str(row.get("gold_value", "")),
                search_attempts=attempts,
                final_decision=None,
            )
        )

    out = Path(out_replay_json)
    dump_replay_corpus(episodes, out)
    return out
