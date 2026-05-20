#!/usr/bin/env python3
"""Find replayable candidate evidence URLs from explicit search providers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.search_provider import build_search_provider
from places_attr_conflation.url_evidence_finder import run_url_evidence_finder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Evidence template or prioritized batch CSV. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Directory for query snapshots, candidates, and report.")
    parser.add_argument("--region", default="ca-santa-cruz", choices=["ca-santa-cruz"])
    parser.add_argument("--provider", default="query-only", choices=["query-only", "command", "brave", "google-cse"])
    parser.add_argument("--search-command", default="", help="Command provider executable; accepts JSONL stdin and emits JSONL stdout.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--require-live-search", action="store_true", help="Fail instead of falling back to query-only.")
    parser.add_argument("--write-autofill", action="store_true", help="Write opt-in *.autofilled.csv copies next to finder outputs.")
    args = parser.parse_args()

    provider, provider_info = build_search_provider(
        args.provider,
        search_command=args.search_command,
        require_live_search=args.require_live_search,
    )
    report = run_url_evidence_finder(
        input_paths=args.input,
        output_dir=args.output_dir,
        provider=provider,
        provider_info=provider_info,
        limit=args.limit,
        region=args.region,
        write_autofill=args.write_autofill,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
