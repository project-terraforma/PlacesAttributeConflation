"""Pluggable search providers for replayable evidence URL discovery."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Mapping, Protocol


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    rank: int
    provider: str
    query: str
    case_id: str
    retrieved_at: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        provider: str,
        query: str,
        case_id: str,
        rank: int,
        retrieved_at: str | None = None,
    ) -> "SearchResult":
        return cls(
            url=str(payload.get("url") or payload.get("link") or ""),
            title=str(payload.get("title") or payload.get("name") or ""),
            snippet=str(payload.get("snippet") or payload.get("description") or ""),
            rank=int(payload.get("rank") or rank),
            provider=str(payload.get("provider") or provider),
            query=str(payload.get("query") or query),
            case_id=str(payload.get("case_id") or case_id),
            retrieved_at=str(payload.get("retrieved_at") or retrieved_at or utc_now()),
        )


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, case_id: str, limit: int) -> list[SearchResult]:
        ...


class QueryOnlyProvider:
    """Provider that records queries but performs no live search."""

    name = "query-only"

    def search(self, query: str, *, case_id: str, limit: int) -> list[SearchResult]:
        return []


class CommandSearchProvider:
    """Run a local JSONL command adapter for search.

    The command receives one JSON line on stdin with `query`, `case_id`, and
    `limit`. It should return JSONL rows containing URL result fields.
    """

    name = "command"

    def __init__(self, command: str):
        if not command.strip():
            raise ValueError("CommandSearchProvider requires a command")
        self.command = command

    def search(self, query: str, *, case_id: str, limit: int) -> list[SearchResult]:
        request = {"query": query, "case_id": case_id, "limit": limit}
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(request, sort_keys=True) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        results: list[SearchResult] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rows = payload.get("results") if isinstance(payload, dict) else None
            if isinstance(rows, list):
                for idx, row in enumerate(rows[:limit], start=1):
                    if isinstance(row, dict):
                        results.append(SearchResult.from_dict(row, provider=self.name, query=query, case_id=case_id, rank=idx))
            elif isinstance(payload, dict):
                results.append(
                    SearchResult.from_dict(payload, provider=self.name, query=query, case_id=case_id, rank=len(results) + 1)
                )
        return [result for result in results if result.url][:limit]


class BraveSearchProvider:
    name = "brave"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, *, case_id: str, limit: int) -> list[SearchResult]:
        if not self.available:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is not set")
        params = urllib.parse.urlencode({"q": query, "count": max(1, min(limit, 20))})
        request = urllib.request.Request(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
        )
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = ((payload.get("web") or {}).get("results") or []) if isinstance(payload, dict) else []
        return [
            SearchResult.from_dict(
                {"url": row.get("url", ""), "title": row.get("title", ""), "snippet": row.get("description", "")},
                provider=self.name,
                query=query,
                case_id=case_id,
                rank=idx,
            )
            for idx, row in enumerate(rows[:limit], start=1)
            if isinstance(row, dict) and row.get("url")
        ]


class GoogleProgrammableSearchProvider:
    name = "google-cse"

    def __init__(self, api_key: str | None = None, cse_id: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_CSE_API_KEY", "")
        self.cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID", "")

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.cse_id)

    def search(self, query: str, *, case_id: str, limit: int) -> list[SearchResult]:
        if not self.available:
            raise RuntimeError("GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID are required")
        params = urllib.parse.urlencode(
            {"key": self.api_key, "cx": self.cse_id, "q": query, "num": max(1, min(limit, 10))}
        )
        request = urllib.request.Request(f"https://www.googleapis.com/customsearch/v1?{params}")
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("items") or [] if isinstance(payload, dict) else []
        return [
            SearchResult.from_dict(row, provider=self.name, query=query, case_id=case_id, rank=idx)
            for idx, row in enumerate(rows[:limit], start=1)
            if isinstance(row, dict) and (row.get("link") or row.get("url"))
        ]


def build_search_provider(
    provider_name: str,
    *,
    search_command: str = "",
    require_live_search: bool = False,
    env: Mapping[str, str] | None = None,
) -> tuple[SearchProvider, dict[str, object]]:
    env = env or os.environ
    requested = provider_name
    if provider_name == "query-only":
        return QueryOnlyProvider(), {"requested_provider": requested, "provider": "query-only", "fallback": False, "reason": ""}
    if provider_name == "command":
        if not search_command:
            if require_live_search:
                raise ValueError("--search-command is required for command provider")
            return QueryOnlyProvider(), {
                "requested_provider": requested,
                "provider": "query-only",
                "fallback": True,
                "reason": "missing_search_command",
            }
        return CommandSearchProvider(search_command), {"requested_provider": requested, "provider": "command", "fallback": False, "reason": ""}
    if provider_name == "brave":
        api_key = env.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            if require_live_search:
                raise ValueError("BRAVE_SEARCH_API_KEY is required for --provider brave")
            return QueryOnlyProvider(), {
                "requested_provider": requested,
                "provider": "query-only",
                "fallback": True,
                "reason": "missing_brave_api_key",
            }
        return BraveSearchProvider(api_key), {"requested_provider": requested, "provider": "brave", "fallback": False, "reason": ""}
    if provider_name == "google-cse":
        api_key = env.get("GOOGLE_CSE_API_KEY", "")
        cse_id = env.get("GOOGLE_CSE_ID", "")
        if not api_key or not cse_id:
            if require_live_search:
                raise ValueError("GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID are required for --provider google-cse")
            return QueryOnlyProvider(), {
                "requested_provider": requested,
                "provider": "query-only",
                "fallback": True,
                "reason": "missing_google_cse_credentials",
            }
        return GoogleProgrammableSearchProvider(api_key, cse_id), {
            "requested_provider": requested,
            "provider": "google-cse",
            "fallback": False,
            "reason": "",
        }
    raise ValueError(f"Unsupported provider: {provider_name}")
