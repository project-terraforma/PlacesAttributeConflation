"""Corpus-shape gates for the replay-corpus v1 benchmark.

These gates are deliberately separate from model accuracy gates. They answer:
"is the replay corpus large, labeled, reviewed, and hard enough to support a
credible PAC benchmark?"
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from .corpus_stats import replay_corpus_label_stats
from .replay import ReplayEpisode


@dataclass(frozen=True)
class ReplayCorpusV1Thresholds:
    min_total_replay_cases: int = 650
    min_website_cases: int = 425
    min_identity_labeled_cases: int = 100
    min_reviewed_cases: int = 100
    min_expected_abstain_cases: int = 50
    min_stale_website_cases: int = 40
    min_wrong_branch_cases: int = 25
    min_aggregator_echo_cases: int = 25
    min_new_entity_same_address_cases: int = 25
    min_website_label_coverage_rate: float = 0.80
    min_case_type_coverage_rate: float = 0.90


def evaluate_website_label_coverage(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    """Measure website-specific label coverage and hard website case counts."""

    episodes = list(episodes)
    website_episodes = [episode for episode in episodes if episode.attribute == "website" or episode.website_label]
    by_label = Counter(episode.website_label for episode in website_episodes if episode.website_label)
    labeled_total = sum(by_label.values())
    return {
        "website_episodes_total": len(website_episodes),
        "website_labeled_total": labeled_total,
        "website_label_coverage_rate": labeled_total / len(website_episodes) if website_episodes else 0.0,
        "episodes_by_website_label": dict(sorted(by_label.items())),
        "official_current_cases": by_label.get("OFFICIAL_CURRENT", 0),
        "stale_website_cases": by_label.get("OFFICIAL_STALE", 0),
        "dead_domain_cases": by_label.get("OFFICIAL_DEAD", 0),
        "wrong_entity_cases": by_label.get("OFFICIAL_WRONG_ENTITY", 0),
        "chain_homepage_cases": by_label.get("OFFICIAL_CHAIN_ONLY", 0),
        "location_page_cases": by_label.get("OFFICIAL_LOCATION_PAGE", 0),
        "social_only_cases": by_label.get("SOCIAL_ONLY_CURRENT", 0),
        "aggregator_only_cases": by_label.get("AGGREGATOR_ONLY", 0),
        "parked_domain_cases": by_label.get("PARKED_DOMAIN", 0),
        "no_website_found_cases": by_label.get("NO_WEBSITE_FOUND", 0),
        "ambiguous_website_cases": by_label.get("AMBIGUOUS_WEBSITE", 0),
    }


def evaluate_replay_corpus_v1_gate(
    episodes: Iterable[ReplayEpisode],
    thresholds: ReplayCorpusV1Thresholds | None = None,
) -> dict[str, object]:
    """Evaluate whether a replay corpus meets v1 operational shape targets."""

    thresholds = thresholds or ReplayCorpusV1Thresholds()
    episodes = list(episodes)
    stats = replay_corpus_label_stats(episodes)
    website = evaluate_website_label_coverage(episodes)
    identity_counts = stats.get("episodes_by_identity_label", {}) or {}
    case_type_counts = stats.get("episodes_by_case_type", {}) or {}
    total = int(stats.get("episodes_total", 0))
    website_total = int(website.get("website_episodes_total", 0))
    checks = {
        "total_replay_cases": total >= thresholds.min_total_replay_cases,
        "website_cases": int(stats.get("website_heavy_count", 0)) >= thresholds.min_website_cases,
        "identity_labeled_cases": int(stats.get("episodes_with_identity_label", 0)) >= thresholds.min_identity_labeled_cases,
        "reviewed_cases": int(stats.get("reviewed_count", 0)) >= thresholds.min_reviewed_cases,
        "expected_abstain_cases": int(stats.get("abstention_expected_count", 0)) >= thresholds.min_expected_abstain_cases,
        "stale_website_cases": int(website.get("stale_website_cases", 0)) >= thresholds.min_stale_website_cases,
        "wrong_branch_cases": (int(identity_counts.get("BRANCH_AMBIGUITY", 0)) + int(case_type_counts.get("WRONG_BRANCH", 0))) >= thresholds.min_wrong_branch_cases,
        "aggregator_echo_cases": int(case_type_counts.get("AGGREGATOR_ECHO", 0)) >= thresholds.min_aggregator_echo_cases,
        "new_entity_same_address_cases": int(identity_counts.get("NEW_ENTITY_SAME_ADDRESS", 0)) >= thresholds.min_new_entity_same_address_cases,
        "website_label_coverage_rate": float(website.get("website_label_coverage_rate", 0.0)) >= thresholds.min_website_label_coverage_rate,
        "case_type_coverage_rate": (int(stats.get("episodes_with_case_type", 0)) / total if total else 0.0) >= thresholds.min_case_type_coverage_rate,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": asdict(thresholds),
        "stats": stats,
        "website_labels": website,
        "coverage": {
            "case_type_coverage_rate": int(stats.get("episodes_with_case_type", 0)) / total if total else 0.0,
            "identity_label_coverage_rate": int(stats.get("episodes_with_identity_label", 0)) / total if total else 0.0,
            "website_label_coverage_rate": float(website.get("website_label_coverage_rate", 0.0)),
            "reviewed_rate": int(stats.get("reviewed_count", 0)) / total if total else 0.0,
            "website_case_share": website_total / total if total else 0.0,
        },
    }


def build_replay_corpus_v1_report(episodes: Iterable[ReplayEpisode]) -> dict[str, object]:
    episodes = list(episodes)
    return {
        "stats": replay_corpus_label_stats(episodes),
        "website_labels": evaluate_website_label_coverage(episodes),
        "gate": evaluate_replay_corpus_v1_gate(episodes),
    }
