"""Utilities for turning fetched pages into manifest evidence items."""

from __future__ import annotations

from .dorking import classify_source, rank_source
from .manifest import EvidenceItem


def evidence_from_page(
    url: str,
    attribute: str,
    extracted_value: str,
    query: str = "",
    page_text: str = "",
    recency_days: float | None = None,
    zombie_score: float = 0.0,
    identity_change_score: float = 0.0,
    notes: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        source_type=classify_source(url),
        url=url,
        attribute=attribute,
        extracted_value=extracted_value,
        query=query,
        source_rank=rank_source(url, page_text=page_text, query=query),
        recency_days=recency_days,
        zombie_score=zombie_score,
        identity_change_score=identity_change_score,
        notes=notes,
    )


def evidence_from_source_type(
    source_type: str,
    url: str,
    attribute: str,
    extracted_value: str,
    query: str = "",
    recency_days: float | None = None,
    zombie_score: float = 0.0,
    identity_change_score: float = 0.0,
    notes: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        source_type=source_type,
        url=url,
        attribute=attribute,
        extracted_value=extracted_value,
        query=query,
        recency_days=recency_days,
        zombie_score=zombie_score,
        identity_change_score=identity_change_score,
        notes=notes,
    )
