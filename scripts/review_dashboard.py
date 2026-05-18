#!/usr/bin/env python3
"""Generate the static replay review dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.review_dashboard import write_review_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a keyboard-driven replay review dashboard.")
    parser.add_argument("--input", required=True, help="Replay corpus JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory for index.html")
    parser.add_argument("--title", default="Replay Review Dashboard")
    args = parser.parse_args()

    report = write_review_dashboard(args.input, args.output_dir, title=args.title)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
