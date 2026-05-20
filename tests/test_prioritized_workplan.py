import csv
import json

from places_attr_conflation.prioritized_workplan import build_prioritized_evidence_workplan


def test_build_prioritized_workplan_keeps_queries_grouped_and_selects_p0_first(tmp_path):
    rows = [
        {
            "id": "cat-1",
            "attribute": "category",
            "current_value": "restaurant",
            "base_value": "cafe",
            "prediction": "restaurant",
            "correct": "true",
            "layer": "official",
            "query": "category query",
        },
        {
            "id": "web-1",
            "attribute": "website",
            "current_value": "https://www.yelp.com/biz/demo",
            "base_value": "https://demo.example",
            "prediction": "https://www.yelp.com/biz/demo",
            "correct": "false",
            "layer": "fallback",
            "query": "fallback query",
        },
        {
            "id": "web-1",
            "attribute": "website",
            "current_value": "https://www.yelp.com/biz/demo",
            "base_value": "https://demo.example",
            "prediction": "https://www.yelp.com/biz/demo",
            "correct": "false",
            "layer": "official",
            "query": "official query",
        },
    ]

    manifest = build_prioritized_evidence_workplan(rows, tmp_path, cases_per_batch=1, batch_count=1)

    assert manifest["selected_case_attributes"] == 1
    assert manifest["selected_priority_buckets"] == {"P0_WEBSITE_AGGREGATOR_OR_SOCIAL": 1}

    batch_path = tmp_path / "batch_001.csv"
    with batch_path.open(newline="", encoding="utf-8") as handle:
        batch_rows = list(csv.DictReader(handle))

    assert [row["query"] for row in batch_rows] == ["official query", "fallback query"]
    assert all(row["priority_bucket"] == "P0_WEBSITE_AGGREGATOR_OR_SOCIAL" for row in batch_rows)

    template_path = tmp_path / "evidence_template_001.csv"
    with template_path.open(newline="", encoding="utf-8") as handle:
        template_rows = list(csv.DictReader(handle))

    assert len(template_rows) == 1
    assert template_rows[0]["case_id"] == "web-1"
    assert template_rows[0]["query"] == "official query"
    assert template_rows[0]["website_label_guess"] == "AGGREGATOR_ONLY"

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["ranking_strategy"] == "pac_priority_bucket"


def test_evidence_template_query_selection_caps_duplicates_with_case_specific_fallback(tmp_path):
    rows = []
    for idx in range(5):
        rows.extend(
            [
                {
                    "id": f"web-{idx}",
                    "attribute": "website",
                    "current_value": f"https://current-{idx}.example",
                    "base_value": f"https://base-{idx}.example",
                    "prediction": f"https://base-{idx}.example",
                    "correct": "false",
                    "layer": "official",
                    "query": "generic official query",
                },
                {
                    "id": f"web-{idx}",
                    "attribute": "website",
                    "current_value": f"https://current-{idx}.example",
                    "base_value": f"https://base-{idx}.example",
                    "prediction": f"https://base-{idx}.example",
                    "correct": "false",
                    "layer": "fallback",
                    "query": "generic fallback query",
                },
            ]
        )

    manifest = build_prioritized_evidence_workplan(
        rows,
        tmp_path,
        cases_per_batch=5,
        batch_count=1,
        max_template_query_duplicates=2,
    )

    with (tmp_path / "evidence_template_001.csv").open(newline="", encoding="utf-8") as handle:
        template_rows = list(csv.DictReader(handle))

    queries = [row["query"] for row in template_rows]
    assert queries.count("generic official query") == 2
    assert queries.count("generic fallback query") == 2
    assert any("current-4.example" in query for query in queries)
    assert manifest["max_template_query_duplicates"] == 2
    assert manifest["template_max_query_occurrences"] == 2
    assert manifest["template_duplicate_query_count"] == 2
    assert manifest["most_common_template_queries"][0] == {"query": "generic fallback query", "count": 2}
