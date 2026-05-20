import json
import sys
from pathlib import Path

import pytest

from places_attr_conflation.search_provider import (
    CommandSearchProvider,
    QueryOnlyProvider,
    build_search_provider,
)


def test_query_only_provider_returns_no_live_results() -> None:
    provider = QueryOnlyProvider()

    assert provider.name == "query-only"
    assert provider.search("demo query", case_id="case-1", limit=5) == []


def test_command_search_provider_reads_jsonl_and_returns_results(tmp_path: Path) -> None:
    script = tmp_path / "fake_search.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "request = json.loads(sys.stdin.readline())",
                "print(json.dumps({'url': 'https://demo.example/contact', 'title': 'Demo Cafe', 'snippet': request['query'], 'rank': 1, 'retrieved_at': '2026-01-01T00:00:00Z'}))",
            ]
        ),
        encoding="utf-8",
    )
    provider = CommandSearchProvider(f"{sys.executable} {script}")

    results = provider.search("Demo Cafe Santa Cruz", case_id="case-1", limit=3)

    assert len(results) == 1
    assert results[0].url == "https://demo.example/contact"
    assert results[0].provider == "command"
    assert results[0].query == "Demo Cafe Santa Cruz"
    assert results[0].case_id == "case-1"


def test_missing_api_keys_fall_back_to_query_only() -> None:
    provider, info = build_search_provider("brave", env={}, require_live_search=False)

    assert provider.name == "query-only"
    assert info["fallback"] is True
    assert info["reason"] == "missing_brave_api_key"


def test_missing_api_keys_fail_when_live_search_required() -> None:
    with pytest.raises(ValueError):
        build_search_provider("google-cse", env={}, require_live_search=True)


def test_google_provider_is_not_live_without_credentials() -> None:
    provider, info = build_search_provider("google-cse", env={}, require_live_search=False)

    assert provider.name == "query-only"
    assert info["reason"] == "missing_google_cse_credentials"
