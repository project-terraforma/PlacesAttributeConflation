#!/usr/bin/env python3
"""Build replay episodes from PAC workplan batches and explicit evidence rows."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus, load_replay_corpus


def _clean(value: object) -> str:
    return str(value or "").strip()


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{str(key): _clean(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _case_id(row: dict[str, str]) -> str:
    return row.get("case_id") or row.get("id") or ""


def _key(row: dict[str, str]) -> tuple[str, str]:
    return _case_id(row), row.get("attribute", "")


def _notes(row: dict[str, str]) -> str:
    fields = [
        ("notes", row.get("notes", "")),
        ("final_url", row.get("final_url", "")),
        ("http_status", row.get("http_status", "")),
        ("content_hash", row.get("content_hash", "")),
        ("detected_status", row.get("detected_status", "")),
        ("identity_claims", row.get("identity_claims", "")),
        ("fetch_status", row.get("fetch_status", "")),
        ("fetch_error", row.get("fetch_error", "")),
    ]
    return "; ".join(f"{name}={value}" if name != "notes" else value for name, value in fields if value)


def _page(row: dict[str, str], attribute: str) -> FetchedPage | None:
    url = row.get("url", "")
    if not url:
        return None
    extracted_values = {}
    if row.get("extracted_value", ""):
        extracted_values[attribute] = row["extracted_value"]
    if row.get("content_hash", ""):
        extracted_values["content_hash"] = row["content_hash"]
    if row.get("identity_claims", ""):
        extracted_values["identity_claims"] = row["identity_claims"]
    return FetchedPage(
        url=url,
        title=row.get("title", ""),
        page_text=row.get("page_text_excerpt") or row.get("page_text", ""),
        source_type=row.get("source_type") or "unknown",
        extracted_values=extracted_values,
        notes=_notes(row),
    )


def build_replay(batch_paths: list[str], evidence_paths: list[str]) -> list[ReplayEpisode]:
    if len(batch_paths) != len(evidence_paths):
        raise ValueError("--batch and --evidence must be supplied the same number of times")

    grouped: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for batch_path in batch_paths:
        for row in _load_csv(batch_path):
            key = _key(row)
            if key[0] and key[1]:
                grouped.setdefault(key, []).append(row)

    evidence_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for evidence_path in evidence_paths:
        for row in _load_csv(evidence_path):
            key = _key(row)
            if key[0] and key[1]:
                evidence_by_key.setdefault(key, []).append(row)

    episodes: list[ReplayEpisode] = []
    for key, rows in grouped.items():
        case_id, attribute = key
        head = rows[0]
        pages_by_attempt: dict[tuple[str, str], list[FetchedPage]] = {}
        for evidence_row in evidence_by_key.get(key, []):
            page = _page(evidence_row, attribute)
            if page is None:
                continue
            attempt_key = (evidence_row.get("layer") or "imported", evidence_row.get("query", ""))
            pages_by_attempt.setdefault(attempt_key, []).append(page)

        attempt_keys = {(row.get("layer") or "fallback", row.get("query", "")) for row in rows if row.get("query", "")}
        attempt_keys.update(pages_by_attempt)
        attempts = [
            SearchAttempt(layer=layer, query=query, fetched_pages=pages_by_attempt.get((layer, query), []))
            for layer, query in sorted(attempt_keys)
        ]
        episodes.append(
            ReplayEpisode(
                case_id=case_id,
                attribute=attribute,
                place={
                    "base_id": head.get("base_id", ""),
                    "prediction": head.get("prediction", ""),
                    "baseline": head.get("baseline", ""),
                    "current_value": head.get("current_value", ""),
                    "base_value": head.get("base_value", ""),
                    "priority_bucket": head.get("priority_bucket", ""),
                },
                gold_value=head.get("truth", ""),
                search_attempts=attempts,
                identity_label=head.get("identity_label_guess", ""),
                case_type=head.get("case_type_guess", ""),
                website_label=head.get("website_label_guess", ""),
                difficulty=head.get("difficulty", ""),
                label_origin="workplan_guess_unreviewed",
                review_status="unreviewed",
                reviewer_notes="Workplan guess metadata only; human truth review not completed.",
            )
        )
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="append", required=True, help="Prioritized batch CSV; repeatable")
    parser.add_argument("--evidence", action="append", required=True, help="Matching enriched evidence CSV; repeatable")
    parser.add_argument("--output", required=True, help="Replay corpus JSON output")
    args = parser.parse_args()

    episodes = build_replay(args.batch, args.evidence)
    dump_replay_corpus(episodes, args.output)
    loaded = load_replay_corpus(args.output)
    pages = sum(len(attempt.fetched_pages) for episode in loaded for attempt in episode.search_attempts)
    print({"output": args.output, "episodes": len(loaded), "pages": pages})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
