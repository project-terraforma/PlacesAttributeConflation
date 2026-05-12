"""Search-result ranking for authoritative evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from .dorking import rank_source
from .freshness import adjusted_evidence_score
from .small_model import TinyLinearModel, build_feature_vector


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    layer: str = "fallback"
    recency_days: float | None = None
    zombie_score: float = 0.0
    identity_change_score: float = 0.0


def score_search_result(
    result: SearchResult,
    query: str = "",
    model: TinyLinearModel | None = None,
) -> float:
    text = " ".join(part for part in [result.title, result.snippet] if part).strip()
    score = rank_source(result.url, page_text=text, query=query)
    score = adjusted_evidence_score(
        score,
        recency_days=result.recency_days,
        zombie_score=result.zombie_score,
        identity_change_score=result.identity_change_score,
    )
    if result.layer == "official":
        score += 0.05
    elif result.layer == "corroboration":
        score += 0.02
    elif result.layer == "freshness":
        score += 0.03
    if model is not None:
        model_score = model.score(build_feature_vector(result, query=query, page_text=text))
        score = (score * 0.4) + (model_score * 0.6)
    return min(1.0, score)


def rank_search_results(
    results: list[SearchResult],
    query: str = "",
    model: TinyLinearModel | None = None,
) -> list[SearchResult]:
    return sorted(results, key=lambda result: score_search_result(result, query=query, model=model), reverse=True)


def select_authoritative_result(
    results: list[SearchResult],
    query: str = "",
    threshold: float = 0.75,
    model: TinyLinearModel | None = None,
) -> SearchResult | None:
    ranked = rank_search_results(results, query=query, model=model)
    if not ranked:
        return None
    best = ranked[0]
    if score_search_result(best, query=query, model=model) < threshold:
        return None
    return best
