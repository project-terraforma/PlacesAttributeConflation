from places_attr_conflation.dorking import build_multi_layer_plan, classify_source


PLACE = {
    "name": "Moose Cafe",
    "city": "Santa Cruz",
    "region": "CA",
    "address": "123 Ocean St",
    "phone": "831-555-0199",
    "website": "https://moosecafe.example/locations/santa-cruz",
}


def _layer_names(attribute: str) -> list[str]:
    return [layer.name for layer in build_multi_layer_plan(PLACE, attribute).layers]


def _layer_queries(attribute: str, layer_name: str) -> list[str]:
    plan = build_multi_layer_plan(PLACE, attribute)
    return next(layer.queries for layer in plan.layers if layer.name == layer_name)


def test_website_plan_adds_validation_and_identity_layers_before_fallback() -> None:
    names = _layer_names("website")

    assert names == ["official", "website_validation", "identity_drift", "corroboration", "freshness", "fallback"]


def test_website_validation_queries_check_known_domain_evidence() -> None:
    queries = _layer_queries("website", "website_validation")
    joined = "\n".join(queries)

    assert "site:moosecafe.example" in joined
    assert "contact OR about OR locations" in joined
    assert "schema.org OR ld+json" in joined
    assert "official website" in joined
    assert "-site:yelp.com" in joined


def test_identity_drift_queries_cover_moved_formerly_and_closure_signals() -> None:
    queries = _layer_queries("website", "identity_drift")
    joined = "\n".join(queries)

    assert "moved to" in joined
    assert "formerly" in joined
    assert "under new ownership" in joined
    assert "permanently closed" in joined
    assert "site:moosecafe.example we moved" in joined


def test_phone_plan_keeps_existing_layers_without_website_identity_expansion() -> None:
    names = _layer_names("phone")

    assert names == ["official", "corroboration", "freshness", "fallback"]


def test_source_classification_still_distinguishes_core_source_types() -> None:
    assert classify_source("https://www.cityofsantacruz.com/business-license") == "official_site"
    assert classify_source("https://www.ca.gov/business") == "government"
    assert classify_source("https://www.yelp.com/biz/moose-cafe") == "aggregator"
    assert classify_source("https://www.facebook.com/moosecafe") == "social"
    assert classify_source("https://www.openstreetmap.org/node/123") == "osm"
