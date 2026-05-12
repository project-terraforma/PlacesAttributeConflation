"""Evidence manifest schema for attribute-level resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .freshness import adjusted_evidence_score


SOURCE_RANK = {
    "official_site": 1.0,
    "government": 0.95,
    "business_registry": 0.9,
    "google_places": 0.8,
    "osm": 0.65,
    "social": 0.45,
    "aggregator": 0.35,
    "unknown": 0.2,
}


@dataclass(frozen=True)
class EvidenceItem:
    source_type: str
    url: str
    attribute: str
    extracted_value: str
    query: str = ""
    observed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_rank: float | None = None
    recency_days: float | None = None
    zombie_score: float = 0.0
    identity_change_score: float = 0.0
    notes: str = ""

    def score(self) -> float:
        base = self.source_rank if self.source_rank is not None else SOURCE_RANK.get(self.source_type, SOURCE_RANK["unknown"])
        return adjusted_evidence_score(
            base,
            recency_days=self.recency_days,
            zombie_score=self.zombie_score,
            identity_change_score=self.identity_change_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "url": self.url,
            "attribute": self.attribute,
            "extracted_value": self.extracted_value,
            "query": self.query,
            "observed_at": self.observed_at,
            "source_rank": self.source_rank if self.source_rank is not None else SOURCE_RANK.get(self.source_type, SOURCE_RANK["unknown"]),
            "recency_days": self.recency_days,
            "zombie_score": self.zombie_score,
            "identity_change_score": self.identity_change_score,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceItem":
        return cls(
            source_type=str(payload.get("source_type", "unknown")),
            url=str(payload.get("url", "")),
            attribute=str(payload.get("attribute", "")),
            extracted_value=str(payload.get("extracted_value", "")),
            query=str(payload.get("query", "")),
            observed_at=str(payload.get("observed_at", datetime.now(UTC).isoformat())),
            source_rank=payload.get("source_rank"),
            recency_days=payload.get("recency_days"),
            zombie_score=float(payload.get("zombie_score", 0.0) or 0.0),
            identity_change_score=float(payload.get("identity_change_score", 0.0) or 0.0),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class AttributeDecision:
    attribute: str
    decision: str
    confidence: float
    reason: str
    evidence: list[EvidenceItem]
    abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "abstained": self.abstained,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class EvidenceManifest:
    poi_id: str
    candidate: dict[str, str]
    decisions: list[AttributeDecision]

    def to_dict(self) -> dict[str, Any]:
        return {
            "poi_id": self.poi_id,
            "candidate": self.candidate,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
