import csv
import json
from pathlib import Path

from places_attr_conflation.search_provider import QueryOnlyProvider, SearchResult
from places_attr_conflation.url_evidence_finder import (
    ca_santa_cruz_query_specs,
    run_url_evidence_finder,
    score_candidate,
)


class StaticProvider:
    name = "static"

    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, query: str, *, case_id: str, limit: int) -> list[SearchResult]:
        return [
            SearchResult(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                rank=result.rank,
                provider=self.name,
                query=query,
                case_id=case_id,
                retrieved_at=result.retrieved_at,
            )
            for result in self.results[:limit]
        ]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "case_id": "case-1",
        "attribute": "website",
        "priority_bucket": "P0_WEBSITE_BASELINE_WRONG",
        "name": "Demo Cafe",
        "address": "123 Ocean St",
        "city": "Santa Cruz",
        "state": "CA",
        "phone": "831-555-0100",
        "current_value": "https://demo.example",
        "base_value": "https://old.example",
        "query": "Demo Cafe official",
        "url": "",
        "notes": "",
    }
    row.update(overrides)
    return row


def _write_template(path: Path, rows: list[dict[str, str]]) -> Path:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_query_only_finder_writes_deterministic_query_snapshots(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "evidence_template_001.csv", [_row()])
    report = run_url_evidence_finder(
        input_paths=[template],
        output_dir=tmp_path / "finder",
        provider=QueryOnlyProvider(),
        retrieved_at="2026-01-01T00:00:00Z",
    )

    query_lines = (tmp_path / "finder" / "search_queries.jsonl").read_text(encoding="utf-8").splitlines()
    result_lines = (tmp_path / "finder" / "search_results_snapshot.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(query_lines[0])

    assert report["queries_total"] > 0
    assert result_lines == []
    assert first["case_id"] == "case-1"
    assert first["provider"] == "query-only"
    assert first["retrieved_at"] == "2026-01-01T00:00:00Z"


def test_ca_santa_cruz_and_aggregator_dorks_are_generated() -> None:
    specs = ca_santa_cruz_query_specs(_row())
    by_layer = {spec.search_layer: [] for spec in specs}
    for spec in specs:
        by_layer[spec.search_layer].append(spec.query)

    assert any('"Demo Cafe" "Santa Cruz" CA official website' in query for query in by_layer["official"])
    assert any("site:demo.example" in query for query in by_layer["website_validation"])
    assert any("business license" in query for query in by_layer["ca_public_registry"])
    assert any("site:yelp.com" in query for query in by_layer["aggregator_conflict"])


def test_candidate_scoring_rewards_name_address_phone_city_matches() -> None:
    result = SearchResult(
        url="https://demo.example/contact",
        title="Demo Cafe Contact",
        snippet="Demo Cafe at 123 Ocean St, Santa Cruz, CA. Call 831-555-0100.",
        rank=1,
        provider="static",
        query="Demo Cafe Santa Cruz",
        case_id="case-1",
        retrieved_at="2026-01-01T00:00:00Z",
    )

    candidate = score_candidate(_row(), result, search_layer="official")

    assert candidate["confidence_bucket"] == "high"
    assert candidate["source_type"] == "official_site"
    assert candidate["match_name"] == "true"
    assert candidate["match_address"] == "true"
    assert candidate["match_phone"] == "true"
    assert candidate["match_city"] == "true"


def test_candidate_scoring_penalizes_wrong_city_and_aggregator_as_official() -> None:
    result = SearchResult(
        url="https://www.yelp.com/biz/demo-cafe",
        title="Demo Cafe San Jose",
        snippet="Demo Cafe in San Jose, CA reviews.",
        rank=1,
        provider="static",
        query="Demo Cafe Santa Cruz official",
        case_id="case-1",
        retrieved_at="2026-01-01T00:00:00Z",
    )

    candidate = score_candidate(_row(), result, search_layer="official")

    assert candidate["confidence_bucket"] == "rejected"
    assert "wrong_city_state" in candidate["rejection_reason"]
    assert "aggregator_as_official" in candidate["rejection_reason"]


def test_autofill_refuses_low_confidence_candidates(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "evidence_template_001.csv", [_row()])
    provider = StaticProvider(
        [
            SearchResult(
                url="https://irrelevant.example",
                title="Other Business",
                snippet="No matching local details.",
                rank=1,
                provider="static",
                query="",
                case_id="case-1",
                retrieved_at="2026-01-01T00:00:00Z",
            )
        ]
    )

    run_url_evidence_finder(input_paths=[template], output_dir=tmp_path / "finder", provider=provider, write_autofill=True)

    autofill = tmp_path / "finder" / "evidence_template_001.autofilled.csv"
    rows = list(csv.DictReader(autofill.open(newline="", encoding="utf-8")))
    original_rows = list(csv.DictReader(template.open(newline="", encoding="utf-8")))
    assert rows[0]["url"] == ""
    assert original_rows[0]["url"] == ""


def test_autofill_writes_only_explicit_high_confidence_candidates_when_opted_in(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "evidence_template_001.csv", [_row()])
    provider = StaticProvider(
        [
            SearchResult(
                url="https://demo.example/contact",
                title="Demo Cafe Contact",
                snippet="Demo Cafe at 123 Ocean St, Santa Cruz, CA. Call 831-555-0100.",
                rank=1,
                provider="static",
                query="",
                case_id="case-1",
                retrieved_at="2026-01-01T00:00:00Z",
            )
        ]
    )

    run_url_evidence_finder(input_paths=[template], output_dir=tmp_path / "finder-default", provider=provider)
    assert not (tmp_path / "finder-default" / "evidence_template_001.autofilled.csv").exists()

    run_url_evidence_finder(input_paths=[template], output_dir=tmp_path / "finder", provider=provider, write_autofill=True)
    autofill = tmp_path / "finder" / "evidence_template_001.autofilled.csv"
    rows = list(csv.DictReader(autofill.open(newline="", encoding="utf-8")))
    original_rows = list(csv.DictReader(template.open(newline="", encoding="utf-8")))

    assert rows[0]["url"] == "https://demo.example/contact"
    assert "query_used=" in rows[0]["notes"]
    assert "search_provider=static" in rows[0]["notes"]
    assert "search_rank=1" in rows[0]["notes"]
    assert "evidence_role=" in rows[0]["notes"]
    assert original_rows[0]["url"] == ""
