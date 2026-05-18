"""Shared replay-corpus label constants and helpers.

These labels define the first operational website-heavy PAC corpus. They are
kept in one module so replay loading, review dashboards, corpus statistics,
and release gates can share the same vocabulary without duplicating strings.
"""

from __future__ import annotations

from typing import Protocol


class LabeledEpisode(Protocol):
    attribute: str
    case_type: str
    identity_label: str
    expected_abstain: bool | None
    website_label: str
    review_status: str


WEBSITE_LABELS = {
    "OFFICIAL_CURRENT",
    "OFFICIAL_STALE",
    "OFFICIAL_DEAD",
    "OFFICIAL_WRONG_ENTITY",
    "OFFICIAL_CHAIN_ONLY",
    "OFFICIAL_LOCATION_PAGE",
    "SOCIAL_ONLY_CURRENT",
    "AGGREGATOR_ONLY",
    "PARKED_DOMAIN",
    "NO_WEBSITE_FOUND",
    "AMBIGUOUS_WEBSITE",
}

WEBSITE_HARD_LABELS = {
    "OFFICIAL_STALE",
    "OFFICIAL_DEAD",
    "OFFICIAL_WRONG_ENTITY",
    "OFFICIAL_CHAIN_ONLY",
    "SOCIAL_ONLY_CURRENT",
    "AGGREGATOR_ONLY",
    "PARKED_DOMAIN",
    "AMBIGUOUS_WEBSITE",
}

IDENTITY_LABELS = {
    "SAME_ENTITY",
    "MOVED_ENTITY",
    "RENAMED_ENTITY",
    "OWNERSHIP_CHANGE",
    "NEW_ENTITY_SAME_ADDRESS",
    "STALE_OFFICIAL_SITE",
    "BRANCH_AMBIGUITY",
    "TEMPORARY_CLOSURE",
    "PERMANENT_CLOSURE",
    "UNKNOWN_IDENTITY",
}

IDENTITY_DRIFT_LABELS = {
    "MOVED_ENTITY",
    "RENAMED_ENTITY",
    "OWNERSHIP_CHANGE",
    "NEW_ENTITY_SAME_ADDRESS",
    "STALE_OFFICIAL_SITE",
    "BRANCH_AMBIGUITY",
    "TEMPORARY_CLOSURE",
    "PERMANENT_CLOSURE",
}

REQUIRED_PAC_CASE_TYPES = {
    "OFFICIAL_CORRECT_DIRECTORY_STALE",
    "OFFICIAL_STALE_DIRECTORY_CURRENT",
    "MOVED_ENTITY",
    "RENAMED_ENTITY",
    "NEW_ENTITY_SAME_ADDRESS",
    "WRONG_BRANCH",
    "AGGREGATOR_ECHO",
    "WEAK_EVIDENCE_ABSTAIN",
    "TIED_AUTHORITY_ABSTAIN",
    "CLOSURE_AMBIGUITY",
}

REVIEWED_STATUSES = {
    "accepted",
    "rejected",
    "needs_more_evidence",
}


def is_identity_drift_label(label: str) -> bool:
    return label in IDENTITY_DRIFT_LABELS


def is_reviewed_status(status: str) -> bool:
    return bool(status) and status in REVIEWED_STATUSES


def is_hard_case(episode: LabeledEpisode) -> bool:
    """Return True when an episode should count as a PAC hard case."""

    if episode.expected_abstain is True:
        return True
    if episode.case_type and episode.case_type != "SIMPLE_AGREEMENT":
        return True
    if episode.identity_label and episode.identity_label != "SAME_ENTITY":
        return True
    return episode.website_label in WEBSITE_HARD_LABELS
