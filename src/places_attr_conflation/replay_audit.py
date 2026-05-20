"""Audit replay corpora before human truth review."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .replay import ReplayEpisode, load_replay_corpus


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((key, value) for key, value in counter.items() if key))


def _page_count(episode: ReplayEpisode) -> int:
    return sum(len(attempt.fetched_pages) for attempt in episode.search_attempts)


def _has_note_token(notes: str, token: str) -> bool:
    return f"{token}=" in (notes or "")


def audit_replay_episodes(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    episodes = list(episodes)
    pages = [page for episode in episodes for attempt in episode.search_attempts for page in attempt.fetched_pages]
    return {
        "episodes_total": len(episodes),
        "episodes_by_attribute": _sorted_counts(Counter(episode.attribute for episode in episodes)),
        "episodes_with_fetched_pages": sum(1 for episode in episodes if _page_count(episode) > 0),
        "episodes_with_case_type": sum(1 for episode in episodes if episode.case_type),
        "episodes_with_website_label": sum(1 for episode in episodes if episode.website_label),
        "episodes_with_identity_label": sum(1 for episode in episodes if episode.identity_label),
        "episodes_with_expected_abstain": sum(1 for episode in episodes if episode.expected_abstain is not None),
        "pages_total": len(pages),
        "pages_with_source_type": sum(1 for page in pages if page.source_type and page.source_type != "unknown"),
        "pages_with_content_hash": sum(
            1 for page in pages if "content_hash" in page.extracted_values or _has_note_token(page.notes, "content_hash")
        ),
        "pages_with_identity_claims": sum(
            1 for page in pages if "identity_claims" in page.extracted_values or _has_note_token(page.notes, "identity_claims")
        ),
    }


def audit_replay_file(path: str | Path) -> dict[str, object]:
    report = audit_replay_episodes(load_replay_corpus(path))
    report["input"] = str(Path(path))
    return report
