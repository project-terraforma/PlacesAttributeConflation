"""Small evidence-backed resolver prototype."""

from __future__ import annotations

from collections import defaultdict

from .manifest import AttributeDecision, EvidenceItem
from .normalization import (
    normalize_address,
    normalize_category,
    normalize_name,
    normalize_phone,
    normalize_website,
)


NORMALIZERS = {
    "phone": normalize_phone,
    "website": normalize_website,
    "address": normalize_address,
    "name": normalize_name,
    "category": normalize_category,
}

ATTRIBUTE_MIN_SUPPORT = {
    "website": 0.65,
    "category": 0.65,
    "name": 0.60,
    "phone": 0.55,
    "address": 0.55,
}

ATTRIBUTE_MIN_CONFIDENCE = {
    "website": 0.58,
    "category": 0.60,
    "name": 0.58,
    "phone": 0.55,
    "address": 0.55,
}

ATTRIBUTE_MIN_MARGIN = {
    "website": 0.08,
    "category": 0.08,
    "name": 0.06,
    "phone": 0.05,
    "address": 0.05,
}


def resolve_attribute(
    attribute: str,
    candidates: list[str],
    evidence: list[EvidenceItem],
    min_confidence: float = 0.55,
    min_support_score: float = 0.55,
) -> AttributeDecision:
    normalizer = NORMALIZERS.get(attribute, lambda value: (value or "").strip().lower())
    candidate_by_norm = {normalizer(candidate): candidate for candidate in candidates if candidate}
    scores: defaultdict[str, float] = defaultdict(float)
    supporting: defaultdict[str, list[EvidenceItem]] = defaultdict(list)

    for item in evidence:
        if item.attribute != attribute:
            continue
        normalized_value = normalizer(item.extracted_value)
        if not normalized_value:
            continue
        if normalized_value in candidate_by_norm:
            scores[normalized_value] += item.score()
            supporting[normalized_value].append(item)

    if not scores:
        return AttributeDecision(attribute, "", 0.0, "No evidence matched candidate values.", [], abstained=True)

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    best_value, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score else 0.0
    margin = best_score - second_score
    support_floor = max(min_support_score, ATTRIBUTE_MIN_SUPPORT.get(attribute, min_support_score))
    confidence_floor = max(min_confidence, ATTRIBUTE_MIN_CONFIDENCE.get(attribute, min_confidence))
    margin_floor = ATTRIBUTE_MIN_MARGIN.get(attribute, 0.0)
    best_item_score = max((item.score() for item in supporting[best_value]), default=0.0)

    if best_item_score < support_floor:
        return AttributeDecision(
            attribute,
            "",
            confidence,
            "Best evidence support is below the minimum authority threshold; abstaining.",
            supporting[best_value],
            abstained=True,
        )

    if confidence < confidence_floor or margin <= margin_floor:
        return AttributeDecision(
            attribute,
            "",
            confidence,
            "Evidence is too weak or tied; abstaining.",
            supporting[best_value],
            abstained=True,
        )

    return AttributeDecision(
        attribute,
        candidate_by_norm[best_value],
        confidence,
        f"Selected value supported by {len(supporting[best_value])} evidence item(s).",
        supporting[best_value],
    )
