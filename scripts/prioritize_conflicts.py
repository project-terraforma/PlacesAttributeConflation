#!/usr/bin/env python3
"""Post-process conflict dork rows with PAC priority buckets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.conflict_priority import enrich_conflict_rows, load_conflict_rows, summarize_priorities, write_conflict_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prioritize conflict dork rows for PAC replay-corpus collection.")
    parser.add_argument("--input", required=True, help="Conflict dork CSV or conflict CSV")
    parser.add_argument("--output", required=True, help="Prioritized CSV output path")
    parser.add_argument("--report", help="Optional JSON summary report path")
    args = parser.parse_args()

    rows = load_conflict_rows(args.input)
    enriched = enrich_conflict_rows(rows)
    out = write_conflict_rows(enriched, args.output)
    report = {"input": args.input, "output": str(out), **summarize_priorities(enriched)}
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
