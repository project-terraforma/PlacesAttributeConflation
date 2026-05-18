from places_attr_conflation.conflict_priority import classify_conflict_priority, enrich_conflict_rows, summarize_priorities


def test_prioritizes_aggregator_or_social_website_conflict() -> None:
    row = {
        "id": "case-1",
        "attribute": "website",
        "current_value": "https://www.yelp.com/biz/demo-cafe",
        "base_value": "https://democafe.example",
        "prediction": "https://www.yelp.com/biz/demo-cafe",
        "correct": "false",
    }

    priority = classify_conflict_priority(row)

    assert priority.priority_bucket == "P0_WEBSITE_AGGREGATOR_OR_SOCIAL"
    assert priority.website_label_guess == "AGGREGATOR_ONLY"
    assert priority.difficulty == "HARD"


def test_prioritizes_missing_website_before_generic_wrong_prediction() -> None:
    row = {
        "id": "case-2",
        "attribute": "website",
        "current_value": "",
        "base_value": "https://democafe.example",
        "prediction": "",
        "correct": "false",
    }

    priority = classify_conflict_priority(row)

    assert priority.priority_bucket == "P0_WEBSITE_MISSING"
    assert priority.website_label_guess == "NO_WEBSITE_FOUND"


def test_prioritizes_branch_ambiguity_for_location_urls() -> None:
    row = {
        "id": "case-3",
        "attribute": "website",
        "current_value": "https://chain.example/locations/santa-cruz",
        "base_value": "https://chain.example",
        "prediction": "https://chain.example",
        "correct": "false",
    }

    priority = classify_conflict_priority(row)

    assert priority.priority_bucket == "P0_WEBSITE_CHAIN_OR_BRANCH"
    assert priority.identity_label_guess == "BRANCH_AMBIGUITY"
    assert priority.website_label_guess == "OFFICIAL_CHAIN_ONLY"


def test_enriched_rows_sort_p0_before_p1_and_summary_counts() -> None:
    rows = [
        {"id": "category", "attribute": "category", "current_value": "restaurant", "base_value": "cafe", "prediction": "restaurant", "correct": "true"},
        {"id": "website", "attribute": "website", "current_value": "https://facebook.com/demo", "base_value": "https://demo.example", "prediction": "https://facebook.com/demo", "correct": "false"},
    ]

    enriched = enrich_conflict_rows(rows)
    summary = summarize_priorities(enriched)

    assert enriched[0]["id"] == "website"
    assert enriched[0]["priority_bucket"] == "P0_WEBSITE_AGGREGATOR_OR_SOCIAL"
    assert enriched[0]["website_label_guess"] == "SOCIAL_ONLY_CURRENT"
    assert summary["p0_rows"] == 1
    assert summary["rows_by_attribute"]["website"] == 1
