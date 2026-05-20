#!/usr/bin/env python3
"""Audit prioritized PAC workplan batches before evidence collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.workplan_audit import audit_workplan_files


def _inputs(values: list[list[str]]) -> list[str]:
    return [item for group in values for item in group]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", nargs="+", help="Prioritized batch CSV path")
    parser.add_argument("--output", help="Optional JSON report output path")
    parser.add_argument("--fail-on-gate", action="store_true", help="Return nonzero if the first50 gate fails")
    args = parser.parse_args()

    report = audit_workplan_files(_inputs(args.input))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.fail_on_gate and not report["gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
