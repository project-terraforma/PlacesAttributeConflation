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
