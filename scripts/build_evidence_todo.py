#!/usr/bin/env python3
"""Build a human-friendly evidence URL todo sheet from a PAC workplan CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.evidence_todo import write_evidence_todo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Evidence template or prioritized batch CSV")
    parser.add_argument("--output", required=True, help="Markdown todo output path")
    args = parser.parse_args()

    out = write_evidence_todo(args.input, args.output)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
