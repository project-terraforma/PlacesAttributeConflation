#!/usr/bin/env python3
"""Build PAC-prioritized evidence workplan batches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.prioritized_workplan import build_prioritized_evidence_workplan_from_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PAC-prioritized evidence workplan batches.")
    parser.add_argument("--input", required=True, help="Conflict dork CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cases-per-batch", type=int, default=25)
    parser.add_argument("--batch-count", type=int, default=2)
    parser.add_argument("--max-template-query-duplicates", type=int, default=3)
    args = parser.parse_args()

    manifest = build_prioritized_evidence_workplan_from_csv(
        args.input,
        args.output_dir,
        cases_per_batch=args.cases_per_batch,
        batch_count=args.batch_count,
        max_template_query_duplicates=args.max_template_query_duplicates,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
