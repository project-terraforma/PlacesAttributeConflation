"""Stable replay corpus schema for retrieval benchmarking.

The replay corpus is intentionally simple:

* top-level metadata plus a list of episodes
* each episode stores the place, attempts, fetched pages, and final decision
* each fetched page stores the evidence fields needed to re-run ranking offline

The loader accepts both the new corpus object and the older list-of-episodes
fixture format so we can migrate the sample data without breaking callers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .retrieval import SearchResult


SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str = ""
    page_text: str = ""
    source_type: str = "unknown"
    extracted_values: dict[str, str] = field(default_factory=dict)
    evidence_role: str = ""
    source_family_id: str = ""
    recency_days: float | None = None
    zombie_score: float = 0.0
    identity_change_score: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "page_text": self.page_text,
            "source_type": self.source_type,
            "extracted_values": dict(self.extracted_values),
            "evidence_role": self.evidence_role,
            "source_family_id": self.source_family_id,
            "recency_days": self.recency_days,
            "zombie_score": self.zombie_score,
            "identity_change_score": self.identity_change_score,
            "notes": self.notes,
        }

    def to_search_result(self, layer: str) -> SearchResult:
        return SearchResult(
            url=self.url,
            title=self.title,
            snippet=self.page_text,
            layer=layer,
            recency_days=self.recency_days,
            zombie_score=self.zombie_score,
            identity_change_score=self.identity_change_score,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FetchedPage":
        return cls(
            url=str(payload.get("url", "")),
            title=str(payload.get("title", "")),
            page_text=str(payload.get("page_text", payload.get("snippet", ""))),
            source_type=str(payload.get("source_type", "unknown")),
            extracted_values=_coerce_mapping(payload.get("extracted_values", {})),
            evidence_role=str(payload.get("evidence_role", "")),
            source_family_id=str(payload.get("source_family_id", "")),
            recency_days=payload.get("recency_days"),
            zombie_score=_coerce_float(payload.get("zombie_score", 0.0)),
            identity_change_score=_coerce_float(payload.get("identity_change_score", 0.0)),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class SearchAttempt:
    layer: str
    query: str
    fetched_pages: list[FetchedPage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "query": self.query,
            "fetched_pages": [page.to_dict() for page in self.fetched_pages],
        }

    @property
    def results(self) -> list[FetchedPage]:
        return self.fetched_pages

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchAttempt":
        pages = payload.get("fetched_pages", payload.get("results", []))
        return cls(
            layer=str(payload.get("layer", "fallback")),
            query=str(payload.get("query", "")),
            fetched_pages=[FetchedPage.from_dict(page) for page in pages or []],
        )


@dataclass(frozen=True)
class FinalDecision:
    attribute: str
    decision: str
    confidence: float
    reason: str
    abstained: bool = False
    selected_url: str = ""
    selected_source_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "abstained": self.abstained,
            "selected_url": self.selected_url,
            "selected_source_type": self.selected_source_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FinalDecision":
        return cls(
            attribute=str(payload.get("attribute", "")),
            decision=str(payload.get("decision", "")),
            confidence=_coerce_float(payload.get("confidence", 0.0)),
            reason=str(payload.get("reason", "")),
            abstained=bool(payload.get("abstained", False)),
            selected_url=str(payload.get("selected_url", "")),
            selected_source_type=str(payload.get("selected_source_type", "")),
        )


@dataclass(frozen=True)
class ReplayEpisode:
    case_id: str
    attribute: str
    place: dict[str, str]
    gold_value: str
    search_attempts: list[SearchAttempt]
    final_decision: FinalDecision | None = None
    identity_label: str = ""
    case_type: str = ""
    expected_decision: str = ""
    expected_abstain: bool | None = None
    truth_source_type: str = ""
    label_origin: str = ""
    website_label: str = ""
    difficulty: str = ""
    review_status: str = ""
    reviewer_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "case_id": self.case_id,
            "attribute": self.attribute,
            "place": dict(self.place),
            "gold_value": self.gold_value,
            "search_attempts": [attempt.to_dict() for attempt in self.search_attempts],
        }
        if self.identity_label:
            payload["identity_label"] = self.identity_label
        if self.case_type:
            payload["case_type"] = self.case_type
        if self.expected_decision:
            payload["expected_decision"] = self.expected_decision
        if self.expected_abstain is not None:
            payload["expected_abstain"] = self.expected_abstain
        if self.truth_source_type:
            payload["truth_source_type"] = self.truth_source_type
        if self.label_origin:
            payload["label_origin"] = self.label_origin
        if self.website_label:
            payload["website_label"] = self.website_label
        if self.difficulty:
            payload["difficulty"] = self.difficulty
        if self.review_status:
            payload["review_status"] = self.review_status
        if self.reviewer_notes:
            payload["reviewer_notes"] = self.reviewer_notes
        if self.final_decision is not None:
            payload["final_decision"] = self.final_decision.to_dict()
        return payload

    @property
    def attempts(self) -> list[SearchAttempt]:
        return self.search_attempts

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayEpisode":
        attempts = payload.get("search_attempts", payload.get("attempts", []))
        final_decision = payload.get("final_decision")
        return cls(
            case_id=str(payload.get("case_id", "")),
            attribute=str(payload.get("attribute", "")),
            place=_coerce_mapping(payload.get("place", {})),
            gold_value=str(payload.get("gold_value", "")),
            search_attempts=[SearchAttempt.from_dict(attempt) for attempt in attempts or []],
            final_decision=FinalDecision.from_dict(final_decision) if isinstance(final_decision, dict) else None,
            identity_label=str(payload.get("identity_label", "")),
            case_type=str(payload.get("case_type", "")),
            expected_decision=str(payload.get("expected_decision", "")),
            expected_abstain=payload.get("expected_abstain") if isinstance(payload.get("expected_abstain"), bool) else None,
            truth_source_type=str(payload.get("truth_source_type", "")),
            label_origin=str(payload.get("label_origin", "")),
            website_label=str(payload.get("website_label", "")),
            difficulty=str(payload.get("difficulty", "")),
            review_status=str(payload.get("review_status", "")),
            reviewer_notes=str(payload.get("reviewer_notes", "")),
        )


def _load_payload(path: str | Path) -> dict[str, Any] | list[Any]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return json.loads(text)


def load_replay_corpus(path: str | Path) -> list[ReplayEpisode]:
    payload = _load_payload(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        rows = payload["episodes"]
    else:
        raise ValueError("Replay corpus must be a list of episodes or an object with an 'episodes' list")
    return [ReplayEpisode.from_dict(row) for row in rows]


def dump_replay_corpus(
    episodes: Iterable[ReplayEpisode],
    path: str | Path,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> None:
    payload = {
        "schema_version": schema_version,
        "generated_at": _now_iso(),
        "episodes": [episode.to_dict() for episode in episodes],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def replay_episode_from_legacy_dict(payload: dict[str, Any]) -> ReplayEpisode:
    return ReplayEpisode.from_dict(payload)


def replay_episode_to_legacy_dict(episode: ReplayEpisode) -> dict[str, Any]:
    return episode.to_dict()


def replay_page_payload(page: FetchedPage) -> dict[str, Any]:
    return page.to_dict()


def replay_episode_summary(episode: ReplayEpisode) -> dict[str, Any]:
    return {
        "case_id": episode.case_id,
        "attribute": episode.attribute,
        "attempt_count": len(episode.search_attempts),
        "page_count": sum(len(attempt.fetched_pages) for attempt in episode.search_attempts),
        "has_final_decision": episode.final_decision is not None,
    }
