#!/usr/bin/env python3
"""Report replay corpus v1 shape, label coverage, and gate status."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.corpus_gates import ReplayCorpusV1Thresholds, build_replay_corpus_v1_report, evaluate_replay_corpus_v1_gate
from places_attr_conflation.replay import load_replay_corpus


def _thresholds_from_args(args: argparse.Namespace) -> ReplayCorpusV1Thresholds:
    thresholds = ReplayCorpusV1Thresholds()
    updates = {
        key: value
        for key, value in {
            "min_total_replay_cases": args.min_total_replay_cases,
            "min_website_cases": args.min_website_cases,
            "min_identity_labeled_cases": args.min_identity_labeled_cases,
            "min_reviewed_cases": args.min_reviewed_cases,
            "min_expected_abstain_cases": args.min_expected_abstain_cases,
        }.items()
        if value is not None
    }
    return replace(thresholds, **updates) if updates else thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate replay corpus v1 operational shape.")
    parser.add_argument("--input", required=True, help="Replay corpus JSON file")
    parser.add_argument("--output", help="Optional JSON report output path")
    parser.add_argument("--fail-on-gate", action="store_true", help="Return nonzero if corpus gate fails")
    parser.add_argument("--min-total-replay-cases", type=int)
    parser.add_argument("--min-website-cases", type=int)
    parser.add_argument("--min-identity-labeled-cases", type=int)
    parser.add_argument("--min-reviewed-cases", type=int)
    parser.add_argument("--min-expected-abstain-cases", type=int)
    args = parser.parse_args()

    episodes = load_replay_corpus(args.input)
    report = build_replay_corpus_v1_report(episodes)
    report["gate"] = evaluate_replay_corpus_v1_gate(episodes, _thresholds_from_args(args))
    report["input"] = args.input
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
