"""Corpus-quality statistics for replay corpora.

These stats answer a different question than retrieval metrics: whether the
replay corpus has enough hard, labeled, website-heavy evidence to be a useful
PAC benchmark.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .corpus_labels import IDENTITY_DRIFT_LABELS, is_hard_case, is_reviewed_status
from .replay import ReplayEpisode


def replay_corpus_label_stats(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    """Return label-coverage and hard-case coverage for replay episodes."""

    episodes = list(episodes)
    case_type_counts = Counter(episode.case_type for episode in episodes if episode.case_type)
    identity_label_counts = Counter(episode.identity_label for episode in episodes if episode.identity_label)
    website_label_counts = Counter(episode.website_label for episode in episodes if episode.website_label)
    review_status_counts = Counter(episode.review_status for episode in episodes if episode.review_status)
    identity_drift_count = sum(count for label, count in identity_label_counts.items() if label in IDENTITY_DRIFT_LABELS)
    reviewed_count = sum(count for status, count in review_status_counts.items() if is_reviewed_status(status))

    return {
        "episodes_total": len(episodes),
        "episodes_with_case_type": sum(1 for episode in episodes if episode.case_type),
        "episodes_with_identity_label": sum(1 for episode in episodes if episode.identity_label),
        "episodes_with_website_label": sum(1 for episode in episodes if episode.website_label),
        "episodes_with_expected_abstain": sum(1 for episode in episodes if episode.expected_abstain is not None),
        "episodes_with_truth_source_type": sum(1 for episode in episodes if episode.truth_source_type),
        "episodes_by_case_type": dict(sorted(case_type_counts.items())),
        "episodes_by_identity_label": dict(sorted(identity_label_counts.items())),
        "episodes_by_website_label": dict(sorted(website_label_counts.items())),
        "episodes_by_review_status": dict(sorted(review_status_counts.items())),
        "website_heavy_count": sum(1 for episode in episodes if episode.attribute == "website" or episode.website_label),
        "identity_drift_count": identity_drift_count,
        "abstention_expected_count": sum(1 for episode in episodes if episode.expected_abstain is True),
        "reviewed_count": reviewed_count,
        "hard_case_count": sum(1 for episode in episodes if is_hard_case(episode)),
    }
