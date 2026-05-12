"""Targeted query generation and source classification for evidence retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from .normalization import website_domain


AGGREGATOR_DOMAINS = {
    "yelp.com",
    "tripadvisor.com",
    "foursquare.com",
    "facebook.com",
    "instagram.com",
    "doordash.com",
    "ubereats.com",
    "grubhub.com",
}

GOVERNMENT_SUFFIXES = (".gov", ".ca.gov", ".nyc.gov")
GOOGLE_PLACES_DOMAINS = {"google.com", "maps.google.com"}
OSM_DOMAINS = {"openstreetmap.org", "osm.org"}
BUSINESS_REGISTRY_DOMAINS = {
    "bbb.org",
    "opencorporates.com",
    "bizapedia.com",
    "dnb.com",
    "corporationwiki.com",
}
EXCLUDED_AGGREGATOR_SITES = (
    "yelp.com",
    "tripadvisor.com",
    "facebook.com",
    "instagram.com",
    "doordash.com",
    "ubereats.com",
    "grubhub.com",
)
SEARCH_OPERATORS = ("site:", "-site:", "intitle:", "inurl:", "OR", '"')
CONTACT_HINTS = ("contact", "about", "locations", "location", "store locator", "store-locator", "directions")
CATEGORY_HINTS = ("services", "menu", "about", "schema.org", "ld+json")
FRESH_HINTS = ("current", "updated", "open now", "now open", "hours", "verified", "today", "latest")
STALE_HINTS = (
    "permanently closed",
    "temporarily closed",
    "moved",
    "former",
    "formerly",
    "old location",
    "under new ownership",
    "duplicate listing",
    "claimed listing",
    "directory",
    "listing",
    "reviews",
    "review",
    "aggregate",
    "outdated",
    "stale",
)


@dataclass(frozen=True)
class DorkQueryPlan:
    attribute: str
    loose: str
    targeted: list[str]
    preferred_sources: list[str]


@dataclass(frozen=True)
class DorkLayer:
    name: str
    queries: list[str]
    preferred_sources: list[str]


@dataclass(frozen=True)
class MultiLayerDorkPlan:
    attribute: str
    layers: list[DorkLayer]


@dataclass(frozen=True)
class DorkPlanAudit:
    attribute: str
    total_queries: int
    operator_queries: int
    quoted_anchor_queries: int
    site_restricted_queries: int
    exclusion_queries: int
    fallback_queries: int
    authority_queries: int
    operator_coverage: float
    quoted_anchor_coverage: float
    site_restricted_coverage: float
    exclusion_coverage: float
    authority_coverage: float
    fallback_share: float


def quoted(value: str | None) -> str:
    value = (value or "").strip()
    return f'"{value}"' if value else ""


def _known_domain(website: str | None) -> str:
    return website_domain(website)


def _site_query(domain: str, *parts: str) -> str:
    terms = " ".join(part for part in parts if part).strip()
    return f"site:{domain} {terms}".strip()


def loose_query(place: dict[str, str]) -> str:
    return " ".join(filter(None, [place.get("name", ""), place.get("city", ""), place.get("region", "")])).strip()


def targeted_queries(place: dict[str, str], attribute: str) -> list[str]:
    name = quoted(place.get("name"))
    city = quoted(place.get("city"))
    region = quoted(place.get("region"))
    address = quoted(place.get("address"))
    phone = quoted(place.get("phone"))
    website = place.get("website", "")
    domain = _known_domain(website)
    exclusions = " ".join(f"-site:{domain}" for domain in EXCLUDED_AGGREGATOR_SITES)

    if attribute == "website":
        queries = [
            f"{name} {city} official website {exclusions}",
            f"{name} {address} {city} {exclusions}",
            f"intitle:{name} {city} contact OR locations {exclusions}",
            f"inurl:locations {name} {city} contact",
            f"site:.gov {name} {city} business license OR registry",
            f"site:bbb.org {name} {city}",
        ]
        if domain:
            queries.extend(
                [
                    _site_query(domain, "contact OR about OR locations OR store-locator"),
                    _site_query(domain, "schema.org OR ld+json"),
                    _site_query(domain, phone) if phone else "",
                ]
            )
    elif attribute == "phone":
        queries = [
            f"{phone} {name} official contact",
            f"{name} {city} phone {exclusions}",
            f"{name} {city} contact OR hours {exclusions}",
            f"site:.gov {name} {city} license OR registry",
            f"site:google.com/maps {name} {city} phone",
            f"site:openstreetmap.org {name} {city} phone",
        ]
        if domain:
            queries.extend(
                [
                    _site_query(domain, phone),
                    _site_query(domain, "contact OR locations OR hours OR tel"),
                    _site_query(domain, "schema.org OR ld+json"),
                ]
            )
    elif attribute == "address":
        queries = [
            f"{name} {address}",
            f"{name} {city} address {exclusions}",
            f"{address} {city} {region} {exclusions}",
            f"site:.gov {name} {city} address OR permit OR license",
        ]
        if domain:
            queries.extend(
                [
                    _site_query(domain, address),
                    _site_query(domain, "directions OR locations OR contact"),
                    _site_query(domain, "schema.org OR ld+json"),
                ]
            )
    elif attribute == "category":
        queries = [
            f"{name} {city} services menu about {exclusions}",
            f"{name} {city} {region} category",
            f"{name} {city} schema.org LocalBusiness OR Organization {exclusions}",
            f"site:openstreetmap.org {name} {city}",
        ]
        if domain:
            queries.extend(
                [
                    _site_query(domain, "about OR services OR menu"),
                    _site_query(domain, "schema.org LocalBusiness OR Organization"),
                    _site_query(domain, "contact OR locations"),
                ]
            )
    elif attribute == "name":
        queries = [
            f"{address} {city} business name",
            f"{phone} {address}",
            f"{name} {address} {city} {exclusions}",
            f"{name} {city} official OR contact {exclusions}",
            f"site:.gov {address} {city} business OR license",
            f"site:opencorporates.com {name} {region}",
            f"site:bbb.org {name} {city}",
            f"site:google.com/maps {name} {address}",
            f"site:openstreetmap.org {address} {city}",
        ]
        if domain:
            queries.extend(
                [
                    _site_query(domain, "about OR contact OR locations"),
                    _site_query(domain, "schema.org LocalBusiness OR Organization"),
                    _site_query(domain, name),
                ]
            )
    else:
        queries = [loose_query(place)]
    return [query.strip() for query in queries if query.strip()]


def build_query_plan(place: dict[str, str], attribute: str) -> DorkQueryPlan:
    loose = loose_query(place)
    targeted = targeted_queries(place, attribute)
    preferred_sources = ["official_site", "government", "business_registry", "google_places", "osm", "social", "aggregator"]
    return DorkQueryPlan(attribute=attribute, loose=loose, targeted=targeted, preferred_sources=preferred_sources)


def build_multi_layer_plan(place: dict[str, str], attribute: str) -> MultiLayerDorkPlan:
    name = quoted(place.get("name"))
    city = quoted(place.get("city"))
    region = quoted(place.get("region"))
    address = quoted(place.get("address"))
    phone = quoted(place.get("phone"))
    website = place.get("website", "")
    domain = _known_domain(website)

    official_layers = DorkLayer(
        name="official",
        queries=targeted_queries(place, attribute),
        preferred_sources=["official_site", "government", "business_registry"],
    )

    corroboration_queries: list[str] = []
    if attribute in {"website", "phone", "address"}:
        corroboration_queries.extend(
            [
                f"{name} {city} {region} official OR contact OR about",
                f"{name} {address} {city}",
                f"{phone} {name}" if phone else "",
                f"site:google.com/maps {name} {city}",
                f"site:openstreetmap.org {name} {city}",
                _site_query(domain, "contact OR about OR locations") if domain else "",
            ]
        )
    elif attribute == "category":
        corroboration_queries.extend(
            [
                f"{name} {city} services OR menu OR about",
                f"site:{domain} schema.org LocalBusiness" if domain else "",
                f"site:openstreetmap.org {name} {city}",
                f"site:google.com/maps {name} {city} category",
                _site_query(domain, "services OR menu OR about") if domain else "",
            ]
        )
    else:
        corroboration_queries.extend(
            [
                f"{name} {city}",
                f"{address} {city}",
                f"site:google.com/maps {name}",
                _site_query(domain, "about OR contact OR locations") if domain else "",
            ]
        )

    corroboration_layer = DorkLayer(
        name="corroboration",
        queries=[q.strip() for q in corroboration_queries if q.strip()],
        preferred_sources=["official_site", "google_places", "osm", "social"],
    )

    freshness_queries: list[str] = []
    if attribute in {"website", "phone", "address"}:
        freshness_queries.extend(
            [
                f"{name} {city} open now hours contact",
                f"{name} {city} updated contact hours",
                f"{name} {city} current address phone",
                f"{name} {city} moved OR permanently closed OR formerly",
            ]
        )
    elif attribute == "category":
        freshness_queries.extend(
            [
                f"{name} {city} current menu services hours",
                f"{name} {city} updated about services",
                f"site:{domain} hours menu updated" if domain else "",
                f"{name} {city} new menu OR current services OR latest",
            ]
        )
    else:
        freshness_queries.extend([f"{name} {city} updated", f"{name} {city} current", f"{name} {city} former OR formerly OR moved"])

    freshness_layer = DorkLayer(
        name="freshness",
        queries=[q.strip() for q in freshness_queries if q.strip()],
        preferred_sources=["official_site", "google_places", "osm"],
    )

    fallback_layer = DorkLayer(
        name="fallback",
        queries=[loose_query(place)],
        preferred_sources=["google_places", "osm", "social", "aggregator"],
    )

    return MultiLayerDorkPlan(
        attribute=attribute,
        layers=[official_layers, corroboration_layer, freshness_layer, fallback_layer],
    )


def classify_source(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc.lower().removeprefix("www.")
    if any(domain.endswith(suffix) for suffix in GOVERNMENT_SUFFIXES):
        return "government"
    if any(domain == registry or domain.endswith(f".{registry}") for registry in BUSINESS_REGISTRY_DOMAINS):
        return "business_registry"
    if domain in GOOGLE_PLACES_DOMAINS or domain.endswith(".google.com"):
        path = parsed.path.lower()
        if "/maps" in path or "/place" in path or "/search" in path:
            return "google_places"
    if domain in OSM_DOMAINS or domain.endswith(".openstreetmap.org"):
        return "osm"
    if any(domain == agg or domain.endswith(f".{agg}") for agg in AGGREGATOR_DOMAINS):
        return "aggregator" if domain not in {"facebook.com", "instagram.com"} else "social"
    if domain:
        return "official_site"
    return "unknown"


def _has_operator(query: str) -> bool:
    return any(operator in query for operator in SEARCH_OPERATORS)


def _has_quoted_anchor(query: str) -> bool:
    return query.count('"') >= 2


def _has_authority_surface(query: str) -> bool:
    lowered = query.lower()
    return any(
        token in lowered
        for token in (
            "official",
            "site:",
            "contact",
            "locations",
            "directions",
            "schema.org",
            "license",
            "registry",
            "permit",
        )
    )


def audit_multi_layer_plan(plan: MultiLayerDorkPlan) -> DorkPlanAudit:
    layer_queries = [(layer.name, query) for layer in plan.layers for query in layer.queries]
    total = len(layer_queries)
    if total == 0:
        return DorkPlanAudit(
            attribute=plan.attribute,
            total_queries=0,
            operator_queries=0,
            quoted_anchor_queries=0,
            site_restricted_queries=0,
            exclusion_queries=0,
            fallback_queries=0,
            authority_queries=0,
            operator_coverage=0.0,
            quoted_anchor_coverage=0.0,
            site_restricted_coverage=0.0,
            exclusion_coverage=0.0,
            authority_coverage=0.0,
            fallback_share=0.0,
        )

    operator_queries = sum(1 for _, query in layer_queries if _has_operator(query))
    quoted_anchor_queries = sum(1 for _, query in layer_queries if _has_quoted_anchor(query))
    site_restricted_queries = sum(1 for _, query in layer_queries if "site:" in query)
    exclusion_queries = sum(1 for _, query in layer_queries if "-site:" in query)
    fallback_queries = sum(1 for layer, _ in layer_queries if layer == "fallback")
    authority_queries = sum(1 for _, query in layer_queries if _has_authority_surface(query))
    return DorkPlanAudit(
        attribute=plan.attribute,
        total_queries=total,
        operator_queries=operator_queries,
        quoted_anchor_queries=quoted_anchor_queries,
        site_restricted_queries=site_restricted_queries,
        exclusion_queries=exclusion_queries,
        fallback_queries=fallback_queries,
        authority_queries=authority_queries,
        operator_coverage=operator_queries / total,
        quoted_anchor_coverage=quoted_anchor_queries / total,
        site_restricted_coverage=site_restricted_queries / total,
        exclusion_coverage=exclusion_queries / total,
        authority_coverage=authority_queries / total,
        fallback_share=fallback_queries / total,
    )


def audit_dorking_plans(places: list[dict[str, str]], attributes: list[str]) -> dict[str, object]:
    audits: list[DorkPlanAudit] = []
    for place in places:
        for attribute in attributes:
            audits.append(audit_multi_layer_plan(build_multi_layer_plan(place, attribute)))
    totals = {
        "plans": len(audits),
        "queries": sum(audit.total_queries for audit in audits),
        "operator_queries": sum(audit.operator_queries for audit in audits),
        "quoted_anchor_queries": sum(audit.quoted_anchor_queries for audit in audits),
        "site_restricted_queries": sum(audit.site_restricted_queries for audit in audits),
        "exclusion_queries": sum(audit.exclusion_queries for audit in audits),
        "fallback_queries": sum(audit.fallback_queries for audit in audits),
        "authority_queries": sum(audit.authority_queries for audit in audits),
    }
    query_count = max(1, int(totals["queries"]))
    summary = {
        "operator_coverage": totals["operator_queries"] / query_count,
        "quoted_anchor_coverage": totals["quoted_anchor_queries"] / query_count,
        "site_restricted_coverage": totals["site_restricted_queries"] / query_count,
        "exclusion_coverage": totals["exclusion_queries"] / query_count,
        "authority_coverage": totals["authority_queries"] / query_count,
        "fallback_share": totals["fallback_queries"] / query_count,
    }
    return {
        "summary": summary,
        "totals": totals,
        "plans": [asdict(audit) for audit in audits],
    }


def rank_source(url: str, page_text: str = "", query: str = "") -> float:
    """Return a coarse source authority score for a fetched page.

    The score is intentionally simple: it lets the resolver prefer official or
    government evidence, then use page-level attribute evidence to break ties.
    """
    source_type = classify_source(url)
    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    base = {
        "official_site": 0.78,
        "government": 0.96,
        "business_registry": 0.9,
        "google_places": 0.82,
        "osm": 0.7,
        "social": 0.45,
        "aggregator": 0.35,
        "unknown": 0.2,
    }.get(source_type, 0.2)

    text = (page_text or "").lower()
    query_text = (query or "").lower()
    bonus = 0.0
    if any(token in text for token in CONTACT_HINTS) or any(token in path for token in CONTACT_HINTS):
        bonus += 0.03
    if any(token in text for token in ("phone", "tel", "address", "hours", "menu", "services")):
        bonus += 0.03
    if "schema.org" in text or "ld+json" in text:
        bonus += 0.04
    if any(token in text for token in FRESH_HINTS) or any(token in path for token in ("current", "updated", "hours")):
        bonus += 0.02
    if "site:" in query_text and domain and domain in query_text:
        bonus += 0.03
    if any(token in text for token in ("reviews", "review", "directory", "listing", "aggregate")):
        bonus -= 0.04
    if any(token in text for token in STALE_HINTS):
        bonus -= 0.08
    if "official" in query_text and source_type == "official_site":
        bonus += 0.02
    if query and query.lower() in text:
        bonus += 0.02

    return min(1.0, base + bonus)
