"""Replayable live-search URL evidence finder for PAC corpus construction."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .dorking import classify_source
from .search_provider import SearchProvider, SearchResult, utc_now
from .website_evidence import detect_identity_claims, detect_status, domain_from_url, registered_domain


CANDIDATE_FIELDS = [
    "case_id",
    "attribute",
    "priority_bucket",
    "candidate_url",
    "candidate_title",
    "candidate_snippet",
    "query_used",
    "search_layer",
    "search_rank",
    "search_provider",
    "source_type",
    "evidence_role",
    "domain",
    "registered_domain",
    "match_name",
    "match_address",
    "match_phone",
    "match_city",
    "match_state",
    "ca_local_signal",
    "santa_cruz_signal",
    "identity_claims",
    "detected_status",
    "url_score",
    "confidence_bucket",
    "rejection_reason",
    "retrieved_at",
]


@dataclass(frozen=True)
class QuerySpec:
    case_id: str
    attribute: str
    priority_bucket: str
    search_layer: str
    query: str

    def to_dict(self, *, provider: str, retrieved_at: str) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "attribute": self.attribute,
            "priority_bucket": self.priority_bucket,
            "search_layer": self.search_layer,
            "query": self.query,
            "provider": provider,
            "retrieved_at": retrieved_at,
        }


def _clean(value: object) -> str:
    return str(value or "").strip()


def _quote(value: str) -> str:
    value = _clean(value)
    return f'"{value}"' if value and not (value.startswith('"') and value.endswith('"')) else value


def _row_case_id(row: dict[str, str]) -> str:
    return _clean(row.get("case_id")) or _clean(row.get("id"))


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return bool(parsed.netloc and "." in parsed.netloc)


def _row_context(row: dict[str, str]) -> dict[str, str]:
    attribute = _clean(row.get("attribute"))
    current = _clean(row.get("current_value"))
    base = _clean(row.get("base_value"))
    prediction = _clean(row.get("prediction"))
    website = _clean(row.get("website")) or next((value for value in [current, base, prediction] if _looks_like_url(value)), "")
    return {
        "case_id": _row_case_id(row),
        "attribute": attribute,
        "priority_bucket": _clean(row.get("priority_bucket")),
        "name": _clean(row.get("name") or row.get("business_name") or row.get("current_name") or row.get("base_name")),
        "city": _clean(row.get("city") or row.get("locality")),
        "state": _clean(row.get("state") or row.get("region") or "CA"),
        "address": _clean(row.get("address") or row.get("current_address") or row.get("base_address") or (current if attribute == "address" else "")),
        "phone": _clean(row.get("phone") or row.get("current_phone") or row.get("base_phone") or (current if attribute == "phone" else "")),
        "website": website,
        "query": _clean(row.get("query")),
    }


def _domain(value: str) -> str:
    return domain_from_url(value) if value else ""


def _add_query(queries: list[QuerySpec], seen: set[tuple[str, str]], context: dict[str, str], layer: str, query: str) -> None:
    query = re.sub(r"\s+", " ", query).strip()
    query = query.replace('""', "").strip()
    if not query or query in {"CA", '"CA"', "official website"}:
        return
    key = (layer, query)
    if key in seen:
        return
    seen.add(key)
    queries.append(
        QuerySpec(
            case_id=context["case_id"],
            attribute=context["attribute"],
            priority_bucket=context["priority_bucket"],
            search_layer=layer,
            query=query,
        )
    )


def ca_santa_cruz_query_specs(row: dict[str, str]) -> list[QuerySpec]:
    context = _row_context(row)
    name = _quote(context["name"])
    city = _quote(context["city"] or "Santa Cruz")
    state = _clean(context["state"] or "CA")
    address = _quote(context["address"])
    phone = _quote(context["phone"])
    domain = _domain(context["website"])
    queries: list[QuerySpec] = []
    seen: set[tuple[str, str]] = set()

    for query in [
        f"{name} {city} {state} official website -site:yelp.com -site:facebook.com -site:instagram.com",
        f"{name} {address} {city} {state}",
        f"{name} {phone}",
        f"{name} {city} contact",
        f"{name} {city} locations",
    ]:
        _add_query(queries, seen, context, "official", query)

    if domain:
        for query in [
            f"site:{domain} {name}",
            f"site:{domain} {address}",
            f"site:{domain} {phone}",
            f"site:{domain} contact OR about OR locations",
            f"site:{domain} schema.org OR ld+json",
        ]:
            _add_query(queries, seen, context, "website_validation", query)

    for query in [
        f"{name} {city} moved to",
        f"{name} {city} formerly",
        f"{name} {city} under new ownership",
        f"{address} {city} formerly",
        f"{address} {city} grand opening",
        f"{name} {city} permanently closed",
    ]:
        _add_query(queries, seen, context, "identity_drift", query)

    for query in [
        f"{name} {city} \"business license\"",
        f"{name} {city} \"Santa Cruz\" \"business license\"",
        f"{address} \"Santa Cruz\" business",
        f"site:cityofsantacruz.com {name}",
        f"site:santacruzcountyca.gov {name}",
        f"site:ca.gov {name} {city}",
    ]:
        _add_query(queries, seen, context, "ca_public_registry", query)

    for query in [
        f"site:yelp.com {name} {city}",
        f"site:facebook.com {name} {city}",
        f"site:instagram.com {name} {city}",
        f"site:tripadvisor.com {name} {city}",
    ]:
        _add_query(queries, seen, context, "aggregator_conflict", query)

    if not queries and context["query"]:
        _add_query(queries, seen, context, "workplan", context["query"])
    return queries


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle and needle.lower() in haystack.lower())


def _phone_digits(value: str) -> str:
    return re.sub(r"\D+", "", value or "")


def _path_has_local_hint(url: str) -> bool:
    path = urlparse(url if "://" in url else f"https://{url}").path.lower()
    return any(token in path for token in ("contact", "about", "location", "locations", "store", "hours"))


def _wrong_city_signal(text: str, city: str) -> bool:
    city = city.lower()
    known = {"san jose", "los angeles", "san francisco", "oakland", "watsonville", "monterey", "scotts valley"}
    return bool(city and city not in text.lower() and any(other in text.lower() for other in known if other != city))


def _evidence_role(source_type: str, search_layer: str, detected_status: str, identity_claims: str, url: str) -> str:
    if source_type == "government" or source_type == "business_registry" or search_layer == "ca_public_registry":
        return "public_registry_candidate"
    if source_type == "social":
        return "social_conflict_candidate"
    if source_type == "aggregator":
        return "aggregator_conflict_candidate"
    if search_layer == "identity_drift" or identity_claims:
        return "identity_drift_candidate"
    if detected_status in {"moved", "permanently_closed", "temporarily_closed"}:
        return "stale_candidate"
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if source_type == "official_site" and parsed.path.strip("/") in {"", "home", "index.html"}:
        return "official_chain_homepage_candidate"
    if source_type == "official_site" and _path_has_local_hint(url):
        return "official_location_candidate"
    if source_type == "official_site":
        return "official_candidate"
    return "rejected_wrong_entity"


def score_candidate(row: dict[str, str], result: SearchResult, *, search_layer: str) -> dict[str, str]:
    context = _row_context(row)
    text = " ".join([result.title, result.snippet, result.url])
    source_type = classify_source(result.url)
    domain = domain_from_url(result.url)
    reg_domain = registered_domain(domain)
    identity_claims = ",".join(detect_identity_claims(text))
    detected_status = detect_status(text)

    match_name = _contains(text, context["name"])
    match_address = _contains(text, context["address"])
    phone_digits = _phone_digits(context["phone"])
    match_phone = bool(phone_digits and phone_digits in _phone_digits(text))
    match_city = _contains(text, context["city"])
    match_state = _contains(text, context["state"])
    ca_local_signal = "ca" in text.lower() or ".ca.gov" in result.url.lower() or "california" in text.lower()
    santa_cruz_signal = "santa cruz" in text.lower()

    score = 0.0
    rejection_reasons: list[str] = []
    if source_type in {"official_site", "government", "business_registry"}:
        score += 0.35
    if match_name:
        score += 0.25
    if match_address:
        score += 0.20
    if match_phone:
        score += 0.20
    if match_city:
        score += 0.12
    if match_state or ca_local_signal:
        score += 0.05
    if _path_has_local_hint(result.url):
        score += 0.08

    wants_identity = search_layer == "identity_drift" or context["priority_bucket"] == "P0_IDENTITY_DRIFT_WEBSITE"
    wants_aggregator = search_layer == "aggregator_conflict" or context["priority_bucket"] == "P0_WEBSITE_AGGREGATOR_OR_SOCIAL"
    if source_type in {"aggregator", "social"} and not wants_aggregator:
        score -= 0.35
        rejection_reasons.append("aggregator_as_official")
    if _wrong_city_signal(text, context["city"]):
        score -= 0.35
        rejection_reasons.append("wrong_city_state")
    if detected_status in {"moved", "permanently_closed", "temporarily_closed"} and not wants_identity:
        score -= 0.25
        rejection_reasons.append("stale_or_identity_signal")
    if re.search(r"\b(domain for sale|parked domain|buy this domain)\b", text, re.IGNORECASE):
        score -= 0.50
        rejection_reasons.append("parked_domain")

    role = _evidence_role(source_type, search_layer, detected_status, identity_claims, result.url)
    if rejection_reasons:
        confidence = "rejected"
        if role == "official_candidate":
            role = "rejected_wrong_entity"
    elif score >= 0.70:
        confidence = "high"
    elif score >= 0.45:
        confidence = "medium"
    elif score >= 0.20:
        confidence = "low"
    else:
        confidence = "rejected"
        role = "rejected_wrong_entity"

    return {
        "case_id": context["case_id"],
        "attribute": context["attribute"],
        "priority_bucket": context["priority_bucket"],
        "candidate_url": result.url,
        "candidate_title": result.title,
        "candidate_snippet": result.snippet,
        "query_used": result.query,
        "search_layer": search_layer,
        "search_rank": str(result.rank),
        "search_provider": result.provider,
        "source_type": source_type,
        "evidence_role": role,
        "domain": domain,
        "registered_domain": reg_domain,
        "match_name": str(match_name).lower(),
        "match_address": str(match_address).lower(),
        "match_phone": str(match_phone).lower(),
        "match_city": str(match_city).lower(),
        "match_state": str(match_state).lower(),
        "ca_local_signal": str(ca_local_signal).lower(),
        "santa_cruz_signal": str(santa_cruz_signal).lower(),
        "identity_claims": identity_claims,
        "detected_status": detected_status,
        "url_score": f"{max(0.0, min(score, 1.0)):.3f}",
        "confidence_bucket": confidence,
        "rejection_reason": ",".join(dict.fromkeys(rejection_reasons)),
        "retrieved_at": result.retrieved_at,
    }


def _load_rows(paths: list[str | Path]) -> list[tuple[str, int, dict[str, str]]]:
    rows: list[tuple[str, int, dict[str, str]]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle), start=2):
                rows.append((str(Path(path)), idx, {str(key): _clean(value) for key, value in row.items()}))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_candidates(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _eligible_for_autofill(candidate: dict[str, str], row: dict[str, str]) -> bool:
    if candidate["confidence_bucket"] != "high":
        return False
    if candidate["source_type"] not in {"official_site", "government", "business_registry"}:
        return False
    if candidate["rejection_reason"]:
        return False
    is_chain_homepage = candidate["evidence_role"] == "official_chain_homepage_candidate"
    return not (is_chain_homepage and row.get("priority_bucket") != "P0_WEBSITE_CHAIN_OR_BRANCH")


def _write_autofill(
    input_paths: list[str | Path],
    output_dir: Path,
    rows_by_input: dict[str, list[dict[str, str]]],
    candidates_by_key: dict[tuple[str, str], list[dict[str, str]]],
) -> list[str]:
    outputs: list[str] = []
    for input_path in input_paths:
        path = Path(input_path)
        output = output_dir / f"{path.stem}.autofilled.csv"
        rows = [dict(row) for row in rows_by_input[str(path)]]
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        for field in ("url", "notes", "source_type"):
            if field not in fields:
                fields.append(field)
        for row in rows:
            key = (_row_case_id(row), row.get("attribute", ""))
            eligible = [candidate for candidate in candidates_by_key.get(key, []) if _eligible_for_autofill(candidate, row)]
            by_url: dict[str, dict[str, str]] = {}
            for candidate in sorted(eligible, key=lambda item: (-float(item["url_score"]), int(item["search_rank"]), item["candidate_url"])):
                by_url.setdefault(candidate["candidate_url"], candidate)
            if len(by_url) != 1:
                continue
            candidate = next(iter(by_url.values()))
            row["url"] = candidate["candidate_url"]
            row["source_type"] = candidate["source_type"]
            note = (
                f"query_used={candidate['query_used']}; search_provider={candidate['search_provider']}; "
                f"search_rank={candidate['search_rank']}; evidence_role={candidate['evidence_role']}"
            )
            row["notes"] = "; ".join(part for part in [row.get("notes", ""), note] if part)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        outputs.append(str(output))
    return outputs


def run_url_evidence_finder(
    *,
    input_paths: list[str | Path],
    output_dir: str | Path,
    provider: SearchProvider,
    limit: int = 5,
    region: str = "ca-santa-cruz",
    write_autofill: bool = False,
    provider_info: dict[str, object] | None = None,
    retrieved_at: str | None = None,
) -> dict[str, object]:
    if region != "ca-santa-cruz":
        raise ValueError(f"Unsupported region: {region}")
    output_root = Path(output_dir)
    timestamp = retrieved_at or utc_now()
    loaded_rows = _load_rows(input_paths)

    query_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    candidates: list[dict[str, str]] = []
    rows_by_input: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidates_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for source_path, _line_no, row in loaded_rows:
        rows_by_input[source_path].append(row)
        specs = ca_santa_cruz_query_specs(row)
        layer_by_query = {spec.query: spec.search_layer for spec in specs}
        for spec in specs:
            query_rows.append(spec.to_dict(provider=provider.name, retrieved_at=timestamp))
            results = provider.search(spec.query, case_id=spec.case_id, limit=limit)
            for result in results:
                result_rows.append(result.to_dict())
                candidate = score_candidate(row, result, search_layer=layer_by_query.get(result.query, spec.search_layer))
                candidates.append(candidate)
                candidates_by_key[(candidate["case_id"], candidate["attribute"])].append(candidate)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "search_queries.jsonl", query_rows)
    _write_jsonl(output_root / "search_results_snapshot.jsonl", result_rows)
    _write_candidates(output_root / "url_candidates.csv", candidates)

    autofill_outputs = _write_autofill(input_paths, output_root, rows_by_input, candidates_by_key) if write_autofill else []
    candidate_counts = Counter(candidate["confidence_bucket"] for candidate in candidates)
    report = {
        "inputs": [str(Path(path)) for path in input_paths],
        "output_dir": str(output_root),
        "provider": provider.name,
        "provider_info": provider_info or {},
        "region": region,
        "limit": limit,
        "rows_total": len(loaded_rows),
        "queries_total": len(query_rows),
        "results_total": len(result_rows),
        "candidates_total": len(candidates),
        "candidates_by_confidence": dict(sorted(candidate_counts.items())),
        "autofill_written": bool(write_autofill),
        "autofill_outputs": autofill_outputs,
        "notes": "Live search discovers candidate URLs only. It does not create truth labels or review decisions.",
    }
    (output_root / "url_finder_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "url_finder_notes.md").write_text(_render_notes(report), encoding="utf-8")
    return report


def _render_notes(report: dict[str, object]) -> str:
    return "\n".join(
        [
            "# URL Evidence Finder Notes",
            "",
            "Live search is used only to discover candidate evidence URLs.",
            "",
            f"- Provider: `{report['provider']}`",
            f"- Region: `{report['region']}`",
            f"- Input rows: {report['rows_total']}",
            f"- Queries written: {report['queries_total']}",
            f"- Results snapshotted: {report['results_total']}",
            f"- Candidate URLs: {report['candidates_total']}",
            f"- Autofill written: {str(report['autofill_written']).lower()}",
            "",
            "No `gold_value`, `expected_decision`, `expected_abstain`, or accepted `review_status` is written by this tool.",
            "",
        ]
    )
