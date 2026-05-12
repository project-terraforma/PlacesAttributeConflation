#!/usr/bin/env python3
"""Unified benchmark harness for baseline, replay, reranking, and smoke checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.harness import (
    DorkAuditThresholds,
    build_ranker_dataset_rows,
    compare_arms,
    compare_reranker_on_replay,
    dump_retrieval_episodes,
    evaluate_dork_audit_gate,
    evaluate_final_decisions,
    evaluate_harness_report,
    evaluate_resolver_on_replay,
    evaluate_retrieval_proof,
    evaluate_retrieval_quality_gate,
    load_retrieval_episodes,
    merge_replay_corpora,
    merge_replay_files,
    replay_stats,
    write_ranker_dataset_csv,
)
from places_attr_conflation.dashboard import write_dashboard
from places_attr_conflation.dataset import (
    export_project_a_review_rows,
    find_project_a_parquet,
    summarize_project_a,
    write_dataset_summary,
    write_review_csv,
)
from places_attr_conflation.dorking import audit_dorking_plans
from places_attr_conflation.conflict_dorks import (
    build_evidence_workplan_batches,
    build_conflict_dork_rows,
    load_conflict_csv,
    split_conflict_dork_csv_by_case,
    write_conflict_dork_csv,
)
from places_attr_conflation.collector import write_seed_replay_from_batch
from places_attr_conflation.collector_static import write_static_collector_html
from places_attr_conflation.golden import (
    PROJECT_A_BASELINES,
    build_project_a_agreement_labels,
    build_project_a_conflict_review_rows,
    build_project_a_evaluation_rows,
    build_project_a_labels_from_david_finalized,
    build_project_a_labels_from_james_golden,
    evaluate_project_a_golden,
    write_conflict_csv,
    write_label_csv,
)
from places_attr_conflation.overture_context import (
    build_overture_context_replay,
    build_overture_gap_dork_rows,
    connect_overture_duckdb,
    dump_overture_context_replay,
    evaluate_overture_gap_dorks,
    evaluate_overture_context,
    fetch_overture_context,
    load_overture_context_replay,
    write_overture_gap_dork_csv,
    write_overture_context_decisions,
)
from places_attr_conflation.synthetic_evidence import (
    evaluate_synthetic_evidence,
    generate_synthetic_evidence_cases,
    load_conflict_rows,
    load_synthetic_evidence,
    write_synthetic_evidence,
)


DEFAULT_SMOKE_URLS = [
    "https://example.com/",
    "https://www.usa.gov/",
]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")


def _default_output_path(command: str) -> Path:
    if command == "golden":
        return ROOT / "reports" / "golden" / f"project_a_golden_{_timestamp()}.json"
    if command in {"synth-evidence", "evidence-eval"}:
        return ROOT / "reports" / "evidence" / f"{command}_{_timestamp()}.json"
    if command == "replay-merge-report":
        return ROOT / "reports" / "replay" / f"merge_report_{_timestamp()}.json"
    if command == "replay-stats":
        return ROOT / "reports" / "replay_stats" / f"replay_stats_{_timestamp()}.json"
    if command == "compare":
        return ROOT / "reports" / "retrieval_compare" / f"compare_{_timestamp()}.json"
    if command == "resolver-on-replay":
        return ROOT / "reports" / "resolver_replay" / f"resolver_on_replay_{_timestamp()}.json"
    if command == "replay-seed":
        return ROOT / "reports" / "harness" / f"replay-seed_{_timestamp()}.json"
    if command == "replay-batch":
        return ROOT / "reports" / "harness" / f"replay-batch_{_timestamp()}.json"
    return ROOT / "reports" / "harness" / f"{command}_{_timestamp()}.json"


def _write_report(report: dict[str, object], output: str | None, command: str) -> Path:
    out = Path(output) if output else _default_output_path(command)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _project_a_places(dataset_path: Path, limit: int) -> list[dict[str, str]]:
    rows = export_project_a_review_rows(dataset_path, limit=limit)
    return [
        {
            "name": str(row.get("name") or row.get("base_name") or ""),
            "city": "",
            "region": "",
            "address": str(row.get("address") or row.get("base_address") or ""),
            "phone": str(row.get("phone") or row.get("base_phone") or ""),
            "website": str(row.get("website") or row.get("base_website") or ""),
        }
        for row in rows
    ]


def _replay_pages_count(path: Path) -> int:
    try:
        episodes = load_retrieval_episodes(path)
    except Exception:
        return 0
    return sum(len(attempt.fetched_pages) for episode in episodes for attempt in episode.search_attempts)


def _discover_replay_inputs(root: Path, *, include_empty: bool) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        name = path.name
        # Prefer canonical batch directories; avoid merging copied combined artifacts or reruns.
        if "evidence_batch_" in str(path.parent) and "evidence_combined_" not in str(path.parent) and name.startswith("replay_seed_evidence_batch_") and name.endswith(".json"):
            candidates.append(path)
    selected: list[Path] = []
    for path in candidates:
        pages = _replay_pages_count(path)
        if pages or include_empty:
            selected.append(path)
    return selected


def _audit_thresholds(args: argparse.Namespace) -> DorkAuditThresholds:
    return DorkAuditThresholds(
        min_operator_coverage=args.min_operator_coverage,
        min_quoted_anchor_coverage=args.min_quoted_anchor_coverage,
        min_site_restricted_coverage=args.min_site_restricted_coverage,
        min_authority_coverage=args.min_authority_coverage,
        max_fallback_share=args.max_fallback_share,
    )


def _add_audit_threshold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-operator-coverage", type=float, default=0.75)
    parser.add_argument("--min-quoted-anchor-coverage", type=float, default=0.70)
    parser.add_argument("--min-site-restricted-coverage", type=float, default=0.35)
    parser.add_argument("--min-authority-coverage", type=float, default=0.60)
    parser.add_argument("--max-fallback-share", type=float, default=0.12)


def _fetch_smoke_url(url: str, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "MLAttributes/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(4096)
            text = body.decode("utf-8", errors="replace")
            title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
            return {
                "url": url,
                "status": "ok",
                "http_status": getattr(response, "status", 200),
                "bytes": len(body),
                "title": title_match.group(1).strip() if title_match else "",
            }
    except urllib.error.URLError as exc:
        return {
            "url": url,
            "status": "error",
            "error": str(exc),
        }


def _run_smoke(urls: list[str], timeout: float, replay_input: str | None) -> dict[str, object]:
    live_results = [_fetch_smoke_url(url, timeout) for url in urls]
    live_ok = any(result.get("status") == "ok" for result in live_results)
    if live_ok:
        return {
            "mode": "live",
            "urls": urls,
            "results": live_results,
        }
    if replay_input:
        replay_report = evaluate_harness_report(retrieval_path=replay_input, retrieval_arm="targeted")
        return {
            "mode": "replay",
            "urls": urls,
            "results": live_results,
            "replay": replay_report,
        }
    return {
        "mode": "offline",
        "urls": urls,
        "results": live_results,
        "message": "Live retrieval was unavailable and no replay fixture was provided.",
    }


def _overture_context_rows(args: argparse.Namespace, dataset_path: Path, attributes: list[str]) -> list[dict[str, object]]:
    rows = build_project_a_evaluation_rows(dataset_path, args.labels, args.baseline)
    if not getattr(args, "all_labeled", False):
        rows = [
            row
            for row in rows
            if any(row.get(f"{attribute}_truth") and row.get(f"{attribute}_pair_differs") for attribute in attributes)
        ]
    return rows[: max(0, args.limit)]


def _fetch_context_for_rows(
    rows: list[dict[str, object]],
    *,
    bbox_margin: float,
) -> tuple[dict[str, dict[str, list[dict[str, object]]]], list[dict[str, str]]]:
    con = connect_overture_duckdb()
    context_by_id: dict[str, dict[str, list[dict[str, object]]]] = {}
    fetch_errors: list[dict[str, str]] = []
    cached_by_source_id: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        case_id = str(row.get("id") or "")
        merged_context = {"places": [], "addresses": []}
        for source_id in [case_id, str(row.get("base_id") or "")]:
            if not source_id:
                continue
            if source_id in cached_by_source_id:
                context = cached_by_source_id[source_id]
            else:
                try:
                    context = fetch_overture_context(con, source_id, bbox_margin=bbox_margin)
                except Exception as exc:  # pragma: no cover - live network path
                    fetch_errors.append({"id": source_id, "case_id": case_id, "error": str(exc)})
                    continue
                cached_by_source_id[source_id] = context
            merged_context["places"].extend(context["places"])
            merged_context["addresses"].extend(context["addresses"])
        context_by_id[case_id] = merged_context
    return context_by_id, fetch_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible baseline, replay, reranker, and smoke benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Reproduce a prior ResolvePOI baseline.")
    baseline.add_argument("--truth", required=True)
    baseline.add_argument("--results-dir", required=True)
    baseline.add_argument("--baseline", required=True, choices=["most_recent", "completeness", "confidence", "hybrid"])
    baseline.add_argument("--limit", type=int, default=200)

    record = subparsers.add_parser("record", help="Normalize a replay corpus to the stable schema.")
    record.add_argument("--input", required=True, help="Replay JSON file to normalize.")

    replay = subparsers.add_parser("replay", help="Evaluate a retrieval replay file.")
    replay.add_argument("--input", required=True, help="Retrieval replay JSON file.")
    replay.add_argument("--arm", default="targeted", choices=["targeted", "fallback", "all"])

    compare = subparsers.add_parser("compare", help="Compare retrieval arms from one replay file.")
    compare.add_argument("--input", required=True, help="Retrieval replay JSON file.")

    replay_merge = subparsers.add_parser("replay-merge", help="Merge downloaded collector replay JSON files.")
    replay_merge.add_argument("--input-dir", required=True, help="Directory containing downloaded replay JSON files.")
    replay_merge.add_argument("--output", help="Merged replay JSON path. Defaults under reports/replay/.")

    replay_stats_cmd = subparsers.add_parser("replay-stats", help="Summarize replay coverage and source distribution.")
    replay_stats_cmd.add_argument("--input", required=True, help="Merged replay JSON file.")

    resolver_replay = subparsers.add_parser("resolver-on-replay", help="Evaluate evidence-backed resolver over replay pages.")
    resolver_replay.add_argument("--input", required=True, help="Merged replay JSON file.")

    dork_audit = subparsers.add_parser("dork-audit", help="Audit search-operator quality for generated dorking plans.")
    dork_audit.add_argument("--input", help="Optional project_a parquet path. Defaults to data/project_a_samples.parquet when present.")
    dork_audit.add_argument("--limit", type=int, default=25)
    dork_audit.add_argument("--attribute", action="append", choices=["website", "phone", "address", "category", "name"])
    _add_audit_threshold_args(dork_audit)

    gated = subparsers.add_parser("gated-retrieval", help="Run dork-audit first, then replay retrieval only if the audit passes.")
    gated.add_argument("--audit-input", help="Optional project_a parquet path. Defaults to data/project_a_samples.parquet when present.")
    gated.add_argument("--audit-limit", type=int, default=25)
    gated.add_argument("--attribute", action="append", choices=["website", "phone", "address", "category", "name"])
    gated.add_argument("--replay-input", required=True, help="Retrieval replay JSON file to evaluate after audit passes.")
    gated.add_argument("--ranker-output", help="Optional CSV output for ranker candidate rows.")
    gated.add_argument("--threshold", type=float, default=0.75)
    gated.add_argument("--max-high-confidence-wrong-rate", type=float, default=0.25)
    _add_audit_threshold_args(gated)

    ranker_dataset = subparsers.add_parser("ranker-dataset", help="Export candidate evidence rows from replay for precision/recall ranker training.")
    ranker_dataset.add_argument("--input", required=True, help="Retrieval replay JSON file.")
    ranker_dataset.add_argument("--arm", default="targeted", choices=["targeted", "fallback", "all"])
    ranker_dataset.add_argument("--threshold", type=float, default=0.75)
    ranker_dataset.add_argument("--csv-output", help="Optional CSV output path.")

    rerank = subparsers.add_parser("rerank", help="Train the optional tiny reranker from replay labels.")
    rerank.add_argument("--input", required=True, help="Retrieval replay JSON file.")

    smoke = subparsers.add_parser("smoke", help="Run a small live retrieval smoke check with replay fallback.")
    smoke.add_argument("--url", action="append", dest="urls", help="Allowlisted URL to fetch. May be repeated.")
    smoke.add_argument("--timeout", type=float, default=5.0)
    smoke.add_argument("--replay-input", help="Replay JSON file to use when live fetches fail.")

    dataset = subparsers.add_parser("dataset", help="Summarize the raw project_a matched-pair parquet with DuckDB.")
    dataset.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")

    review = subparsers.add_parser("reviewset", help="Export a user-friendly CSV review set from project_a matched pairs.")
    review.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    review.add_argument("--limit", type=int, default=200)
    review.add_argument("--offset", type=int, default=0)

    golden = subparsers.add_parser("golden", help="Evaluate project_a pair baselines against a labeled review CSV.")
    golden.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    golden.add_argument("--labels", required=True, help="CSV with <attribute>_truth_choice or <attribute>_truth_value columns.")
    golden.add_argument("--baseline", action="append", choices=PROJECT_A_BASELINES, help="Baseline to evaluate. May be repeated. Defaults to all.")
    golden.add_argument("--limit", type=int, help="Optional max project_a rows to scan before joining labels.")

    conflictset = subparsers.add_parser("conflictset", help="Export labeled base/current conflicts for evidence review.")
    conflictset.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    conflictset.add_argument("--labels", required=True, help="CSV with project_a golden label columns.")
    conflictset.add_argument("--baseline", default="hybrid", choices=PROJECT_A_BASELINES)
    conflictset.add_argument("--limit", type=int, help="Optional max project_a rows to scan before joining labels.")

    conflict_dorks = subparsers.add_parser("conflict-dorks", help="Export layered dork queries for conflict rows (no fetching).")
    conflict_dorks.add_argument("--conflicts", required=True, help="CSV produced by the conflictset command.")
    conflict_dorks.add_argument("--max-queries", type=int, default=8, help="Max queries per conflict row.")
    conflict_dorks.add_argument("--csv-output", help="Optional CSV output path.")

    conflict_batches = subparsers.add_parser("conflict-dorks-batch", help="Split a conflict dork CSV into case-grouped batches.")
    conflict_batches.add_argument("--input", required=True, help="CSV produced by conflict-dorks.")
    conflict_batches.add_argument("--output-dir", required=True, help="Output directory for batch CSVs + manifest.json.")
    conflict_batches.add_argument("--cases-per-batch", type=int, default=250)

    collect = subparsers.add_parser("collect", help="Write a self-contained HTML collector for one batch CSV (no server).")
    collect.add_argument("--batch", required=True, help="One batch CSV under reports/ranker/..._batches/")
    collect.add_argument("--html-output", help="Optional output path for the collector HTML.")

    workplan = subparsers.add_parser("evidence-workplan", help="Create small evidence-collection queues from conflict dork batches.")
    workplan.add_argument("--batch-dir", required=True, help="Directory containing conflict dork batch_*.csv files.")
    workplan.add_argument("--output-dir", help="Output directory for workplan CSVs + evidence templates + manifest.json.")
    workplan.add_argument("--replay-dir", help="Replay-collected directory used to exclude already-evidenced episodes.")
    workplan.add_argument("--batches", type=int, default=25, help="Number of workplan batches to create.")
    workplan.add_argument("--cases-per-batch", type=int, default=25, help="Max case-attributes per workplan batch.")
    workplan.add_argument("--attribute", action="append", choices=["website", "name", "category"], help="Attribute to include. May be repeated.")

    replay_seed = subparsers.add_parser("replay-seed", help="Seed schema-valid replay JSON directly from one conflict dork batch.")
    replay_seed.add_argument("--batch", required=True, help="One batch CSV under reports/ranker/..._batches/")
    replay_seed.add_argument("--evidence", help="Optional CSV/JSON with URL, source_type, snippet/page_text, and extracted_value columns.")
    replay_seed.add_argument("--replay-output", help="Optional replay corpus JSON output path.")

    replay_batch = subparsers.add_parser("replay-batch", help="Seed, merge, and evaluate one evidence batch in one command.")
    replay_batch.add_argument("--batch", required=True, help="Batch CSV containing the dork queries for this evidence batch.")
    replay_batch.add_argument("--evidence", help="Evidence CSV/JSON with public page URLs/snippets/extracted_value.")
    replay_batch.add_argument("--seed-output", help="Optional output path for the seeded replay JSON.")
    replay_batch.add_argument("--merged-output", help="Optional output path for the merged replay JSON.")
    replay_batch.add_argument("--merge-replay-dir", default=str(ROOT / "reports" / "replay_collected"), help="Directory to discover existing replay JSON inputs.")
    replay_batch.add_argument("--include-empty", action="store_true", help="Also merge replay JSON inputs with 0 pages.")

    synth = subparsers.add_parser("synth-evidence", help="Generate synthetic authoritative evidence from conflict rows.")
    synth.add_argument("--conflicts", required=True, help="CSV produced by the conflictset command.")
    synth.add_argument("--limit", type=int, default=200)
    synth.add_argument("--no-edges", action="store_true", help="Use only truth-with-decoy cases instead of rotating edge cases.")

    evidence_eval = subparsers.add_parser("evidence-eval", help="Evaluate resolver behavior on synthetic evidence JSON.")
    evidence_eval.add_argument("--input", required=True, help="Synthetic evidence JSON from synth-evidence.")
    evidence_eval.add_argument("--min-confidence", type=float, default=0.55)
    evidence_eval.add_argument("--min-support-score", type=float, default=0.55)

    overture_context = subparsers.add_parser("overture-context", help="Evaluate candidate choices against official nearby Overture Places/Addresses context.")
    overture_context.add_argument("--input", help="Optional project_a parquet path. Defaults to data/project_a_samples.parquet when present.")
    overture_context.add_argument("--labels", required=True, help="CSV with project_a golden label columns.")
    overture_context.add_argument("--baseline", default="hybrid", choices=PROJECT_A_BASELINES)
    overture_context.add_argument("--limit", type=int, default=10, help="Max labeled rows to evaluate.")
    overture_context.add_argument("--attribute", action="append", choices=["website", "phone", "address", "category", "name"])
    overture_context.add_argument("--bbox-margin", type=float, default=0.01)
    overture_context.add_argument("--live", action="store_true", help="Fetch official Overture context from cloud GeoParquet.")
    overture_context.add_argument("--all-labeled", action="store_true", help="Evaluate all labeled rows, not just conflict rows.")
    overture_context.add_argument("--csv-output", help="Optional CSV output for per-attribute decisions.")

    overture_record = subparsers.add_parser("overture-context-record", help="Fetch and cache official Overture context for labeled conflict rows.")
    overture_record.add_argument("--input", help="Optional project_a parquet path. Defaults to data/project_a_samples.parquet when present.")
    overture_record.add_argument("--labels", required=True, help="CSV with project_a golden label columns.")
    overture_record.add_argument("--baseline", default="hybrid", choices=PROJECT_A_BASELINES)
    overture_record.add_argument("--limit", type=int, default=25)
    overture_record.add_argument("--attribute", action="append", choices=["website", "phone", "address", "category", "name"])
    overture_record.add_argument("--bbox-margin", type=float, default=0.01)
    overture_record.add_argument("--all-labeled", action="store_true")
    overture_record.add_argument("--replay-output", help="Optional JSON output path for cached Overture context.")

    overture_replay = subparsers.add_parser("overture-context-replay", help="Evaluate cached Overture context offline.")
    overture_replay.add_argument("--input", required=True, help="JSON produced by overture-context-record.")
    overture_replay.add_argument("--csv-output", help="Optional CSV output for per-attribute decisions.")

    overture_gap_dorks = subparsers.add_parser("overture-gap-dorks", help="Export targeted dork queue for Overture abstentions and high-risk baseline decisions.")
    overture_gap_dorks.add_argument("--input", required=True, help="JSON produced by overture-context-record.")
    overture_gap_dorks.add_argument("--csv-output", help="Optional CSV output for targeted dork queries.")

    agreement = subparsers.add_parser("agreement-labels", help="Generate silver labels where project_a base/current values normalize to agreement.")
    agreement.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    agreement.add_argument("--limit", type=int, default=200)
    agreement.add_argument("--min-attributes", type=int, default=1)

    james = subparsers.add_parser("import-james-golden", help="Convert James ProjectTerra golden CSV into project_a label schema.")
    james.add_argument("--input", help="Optional parquet path. Defaults to data/project_a_samples.parquet when present.")
    james.add_argument("--james-csv", default="/home/anthony/projectterra_repos/James-Places-Attribute-Conflation/output_data/golden_dataset.csv")
    james.add_argument("--limit", type=int)

    david = subparsers.add_parser("import-david-labels", help="Convert David attribute-level finalized labels into project_a label schema.")
    david.add_argument("--david-csv", default="/home/anthony/projectterra_repos/david-places-attributes-conflation-v2/data/labeling/finalized/final_labels.csv")
    david.add_argument("--split-name", default="finalized")

    dashboard = subparsers.add_parser("dashboard", help="Render a compact benchmark dashboard from saved reports.")
    dashboard.add_argument("--reports-root", default=str(ROOT / "reports"))
    dashboard.add_argument("--output-dir", default=str(ROOT / "reports" / "dashboard"))

    gui = subparsers.add_parser("gui", help="Build the interactive local benchmark viewer.")
    gui.add_argument("--reports-root", default=str(ROOT / "reports"))
    gui.add_argument("--output-dir", default=str(ROOT / "reports" / "dashboard"))

    both = subparsers.add_parser("all", help="Run baseline reproduction and replay evaluation together.")
    both.add_argument("--truth", required=True)
    both.add_argument("--results-dir", required=True)
    both.add_argument("--baseline", required=True, choices=["most_recent", "completeness", "confidence", "hybrid"])
    both.add_argument("--limit", type=int, default=200)
    both.add_argument("--input", required=True, help="Retrieval replay JSON file.")
    both.add_argument("--arm", default="targeted", choices=["targeted", "fallback", "all"])

    parser.add_argument("--output", help="Optional JSON report output path.")
    args = parser.parse_args()

    if args.command == "baseline":
        report = evaluate_harness_report(
            truth_path=args.truth,
            results_dir=args.results_dir,
            baseline_name=args.baseline,
            limit=args.limit,
        )
    elif args.command == "record":
        episodes = load_retrieval_episodes(args.input)
        report = {
            "recorded": len(episodes),
            "corpus": [episode.to_dict() for episode in episodes],
        }
    elif args.command == "replay":
        report = evaluate_harness_report(
            retrieval_path=args.input,
            retrieval_arm=args.arm,
        )
    elif args.command == "compare":
        episodes = load_retrieval_episodes(args.input)
        report = evaluate_retrieval_proof(episodes)
        report["input"] = str(args.input)
    elif args.command == "replay-merge":
        merged_output = Path(args.output) if args.output else ROOT / "reports" / "replay" / f"merged_{_timestamp()}.json"
        report = merge_replay_corpora(args.input_dir, merged_output)
        report_path = _write_report(report, None, "replay-merge-report")
        print(f"saved report to {report_path}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    elif args.command == "replay-stats":
        episodes = load_retrieval_episodes(args.input)
        report = replay_stats(episodes)
        report["input"] = str(args.input)
        lines = [
            f"Replay coverage for {args.input}",
            f"Episodes: {report['episodes_total']}",
            f"Attempts: {report['attempts_total']}",
            f"Pages: {report['pages_total']}",
            f"Authoritative pages: {report['authoritative_pages']} ({float(report['authoritative_pages_rate']) * 100:.1f}%)",
        ]
        report["summary"] = "\n".join(lines)
    elif args.command == "resolver-on-replay":
        episodes = load_retrieval_episodes(args.input)
        report = evaluate_resolver_on_replay(episodes)
        report["input"] = str(args.input)
    elif args.command == "dork-audit":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        places = _project_a_places(dataset_path, args.limit)
        report = audit_dorking_plans(places, args.attribute or ["website", "phone", "address", "category", "name"])
        report["path"] = str(dataset_path)
        report["rows"] = len(places)
        report["gate"] = evaluate_dork_audit_gate(report, _audit_thresholds(args))
    elif args.command == "gated-retrieval":
        dataset_path = Path(args.audit_input) if args.audit_input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --audit-input.")
        places = _project_a_places(dataset_path, args.audit_limit)
        audit = audit_dorking_plans(
            places,
            args.attribute or ["website", "phone", "address", "category", "name"],
        )
        audit["path"] = str(dataset_path)
        audit["rows"] = len(places)
        audit_gate = evaluate_dork_audit_gate(audit, _audit_thresholds(args))
        report = {"audit": audit, "audit_gate": audit_gate, "replay_input": str(args.replay_input)}
        if audit_gate["passed"]:
            episodes = load_retrieval_episodes(args.replay_input)
            retrieval = compare_arms(episodes)
            decisions = evaluate_final_decisions(episodes)
            retrieval_gate = evaluate_retrieval_quality_gate(
                retrieval,
                decisions if decisions["total"] else None,
                max_high_confidence_wrong_rate=args.max_high_confidence_wrong_rate,
            )
            ranker_rows = build_ranker_dataset_rows(episodes, arm="targeted", threshold=args.threshold)
            csv_path = Path(args.ranker_output) if args.ranker_output else ROOT / "reports" / "ranker" / f"ranker_candidates_{_timestamp()}.csv"
            write_ranker_dataset_csv(ranker_rows, csv_path)
            report.update(
                {
                    "retrieval": retrieval,
                    "decisions": decisions,
                    "retrieval_gate": retrieval_gate,
                    "ranker_dataset": {
                        "rows": len(ranker_rows),
                        "positive_rows": sum(int(row["is_supporting_gold"]) for row in ranker_rows),
                        "output_csv": str(csv_path),
                    },
                }
            )
        else:
            report["skipped"] = "Replay retrieval skipped because dork-audit gate did not pass."
    elif args.command == "ranker-dataset":
        episodes = load_retrieval_episodes(args.input)
        rows = build_ranker_dataset_rows(episodes, arm=args.arm, threshold=args.threshold)
        csv_path = Path(args.csv_output) if args.csv_output else ROOT / "reports" / "ranker" / f"ranker_candidates_{_timestamp()}.csv"
        write_ranker_dataset_csv(rows, csv_path)
        report = {
            "input": str(args.input),
            "arm": args.arm,
            "rows": len(rows),
            "positive_rows": sum(int(row["is_supporting_gold"]) for row in rows),
            "selected_correct_rows": sum(int(row["selected_correct"]) for row in rows),
            "output_csv": str(csv_path),
        }
    elif args.command == "rerank":
        episodes = load_retrieval_episodes(args.input)
        report = compare_reranker_on_replay(episodes)
    elif args.command == "smoke":
        urls = args.urls or DEFAULT_SMOKE_URLS
        report = _run_smoke(urls, args.timeout, args.replay_input)
    elif args.command == "dataset":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        report = summarize_project_a(dataset_path)
        write_dataset_summary(report, ROOT / "reports" / "data" / f"project_a_summary_{_timestamp()}.json")
    elif args.command == "reviewset":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = export_project_a_review_rows(dataset_path, limit=args.limit, offset=args.offset)
        csv_path = write_review_csv(rows, ROOT / "reports" / "data" / f"project_a_reviewset_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "preview": rows[:3],
        }
    elif args.command == "golden":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        report = evaluate_project_a_golden(
            dataset_path,
            args.labels,
            baselines=args.baseline or PROJECT_A_BASELINES,
            limit=args.limit,
        )
    elif args.command == "conflictset":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = build_project_a_conflict_review_rows(dataset_path, args.labels, baseline=args.baseline, limit=args.limit)
        csv_path = write_conflict_csv(rows, ROOT / "reports" / "golden" / f"project_a_conflictset_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "labels": str(args.labels),
            "baseline": args.baseline,
            "rows": len(rows),
            "output_csv": str(csv_path),
            "preview": rows[:3],
        }
    elif args.command == "conflict-dorks":
        conflict_rows = load_conflict_csv(args.conflicts)
        rows = build_conflict_dork_rows(conflict_rows, max_queries_per_case=args.max_queries)
        csv_path = Path(args.csv_output) if args.csv_output else ROOT / "reports" / "ranker" / f"conflict_dorks_{_timestamp()}.csv"
        write_conflict_dork_csv(rows, csv_path)
        report = {
            "conflicts": str(args.conflicts),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "preview": rows[:5],
        }
    elif args.command == "conflict-dorks-batch":
        report = split_conflict_dork_csv_by_case(
            args.input,
            args.output_dir,
            cases_per_batch=args.cases_per_batch,
        )
    elif args.command == "collect":
        html_path = (
            Path(args.html_output)
            if args.html_output
            else ROOT / "reports" / "replay_collected" / f"collector_{_timestamp()}.html"
        )
        out = write_static_collector_html(args.batch, html_path)
        report = {
            "batch": str(args.batch),
            "collector_html": str(out),
            "note": "Open the HTML in your browser, paste evidence, then download replay JSON from the page.",
        }
    elif args.command == "replay-seed":
        replay_path = (
            Path(args.replay_output)
            if args.replay_output
            else ROOT / "reports" / "replay_collected" / f"replay_seed_{_timestamp()}.json"
        )
        report = write_seed_replay_from_batch(args.batch, replay_path, evidence_path=args.evidence)
        report["note"] = (
            "Seed replay preserves cases and query attempts. If pages is 0, this unblocks schema/merge/coverage tracking "
            "but does not prove authoritative retrieval."
        )
    elif args.command == "evidence-workplan":
        out_dir = (
            Path(args.output_dir)
            if args.output_dir
            else ROOT / "reports" / "replay_collected" / f"evidence_workplan_{_timestamp()}"
        )
        report = build_evidence_workplan_batches(
            args.batch_dir,
            out_dir,
            replay_dir=args.replay_dir or str(ROOT / "reports" / "replay_collected"),
            attributes=args.attribute or ["website", "name", "category"],
            batch_count=args.batches,
            cases_per_batch=args.cases_per_batch,
        )
    elif args.command == "replay-batch":
        seed_path = (
            Path(args.seed_output)
            if args.seed_output
            else ROOT / "reports" / "replay_collected" / f"replay_seed_evidence_batch_{_timestamp()}.json"
        )
        seed_report = write_seed_replay_from_batch(args.batch, seed_path, evidence_path=args.evidence)

        merge_root = Path(args.merge_replay_dir)
        merge_inputs = _discover_replay_inputs(merge_root, include_empty=bool(args.include_empty))
        if seed_path not in merge_inputs:
            merge_inputs.append(seed_path)

        merged_path = (
            Path(args.merged_output)
            if args.merged_output
            else ROOT / "reports" / "replay" / f"merged_{_timestamp()}.json"
        )
        merge_report = merge_replay_files(merge_inputs, merged_path)
        episodes = load_retrieval_episodes(merged_path)
        stats_report = replay_stats(episodes)
        stats_report["input"] = str(merged_path)
        compare_report = evaluate_retrieval_proof(episodes)
        compare_report["input"] = str(merged_path)
        resolver_report = evaluate_resolver_on_replay(episodes)
        resolver_report["input"] = str(merged_path)

        stats_path = _write_report(stats_report, None, "replay-stats")
        compare_path = _write_report(compare_report, None, "compare")
        resolver_path = _write_report(resolver_report, None, "resolver-on-replay")
        dashboard_outputs = write_dashboard(ROOT / "reports", ROOT / "reports" / "dashboard")

        report = {
            "batch": str(args.batch),
            "evidence": "" if not args.evidence else str(args.evidence),
            "seed": seed_report,
            "merge": merge_report,
            "merged_replay": str(merged_path),
            "replay_stats_report": str(stats_path),
            "compare_report": str(compare_path),
            "resolver_report": str(resolver_path),
            "dashboard": dashboard_outputs,
        }
    elif args.command == "synth-evidence":
        payload = generate_synthetic_evidence_cases(
            load_conflict_rows(args.conflicts),
            limit=args.limit,
            include_edges=not args.no_edges,
        )
        evidence_path = write_synthetic_evidence(payload, ROOT / "reports" / "evidence" / f"synthetic_evidence_{_timestamp()}.json")
        report = {
            "mode": payload["mode"],
            "warning": payload["warning"],
            "case_count": payload["case_count"],
            "output_json": str(evidence_path),
            "preview": payload["cases"][:3],
        }
    elif args.command == "evidence-eval":
        report = evaluate_synthetic_evidence(
            load_synthetic_evidence(args.input),
            min_confidence=args.min_confidence,
            min_support_score=args.min_support_score,
        )
    elif args.command == "overture-context":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        attributes = args.attribute or ["website", "phone", "address", "category", "name"]
        rows = _overture_context_rows(args, dataset_path, attributes)
        if not args.live:
            raise SystemExit("Pass --live to fetch official Overture context. Unit tests cover deterministic scoring without network.")
        context_by_id, fetch_errors = _fetch_context_for_rows(rows, bbox_margin=args.bbox_margin)
        report = evaluate_overture_context(
            rows,
            context_by_id,
            attributes=attributes,
            conflicts_only=not args.all_labeled,
        )
        csv_path = Path(args.csv_output) if args.csv_output else ROOT / "reports" / "overture_context" / f"overture_context_decisions_{_timestamp()}.csv"
        write_overture_context_decisions(report["decisions"], csv_path)
        report.update(
            {
                "path": str(dataset_path),
                "labels": str(args.labels),
                "rows": len(rows),
                "context_cases": len(context_by_id),
                "fetch_errors": fetch_errors,
                "output_csv": str(csv_path),
            }
        )
    elif args.command == "overture-context-record":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        attributes = args.attribute or ["website", "phone", "address", "category", "name"]
        rows = _overture_context_rows(args, dataset_path, attributes)
        context_by_id, fetch_errors = _fetch_context_for_rows(rows, bbox_margin=args.bbox_margin)
        replay_payload = build_overture_context_replay(
            rows,
            context_by_id,
            dataset_path=dataset_path,
            labels_path=args.labels,
            baseline=args.baseline,
            attributes=attributes,
            fetch_errors=fetch_errors,
        )
        replay_path = Path(args.replay_output) if args.replay_output else ROOT / "reports" / "overture_context" / f"overture_context_replay_{_timestamp()}.json"
        dump_overture_context_replay(replay_payload, replay_path)
        eval_report = evaluate_overture_context(rows, context_by_id, attributes=attributes, conflicts_only=not args.all_labeled)
        report = {
            **eval_report,
            "path": str(dataset_path),
            "labels": str(args.labels),
            "rows": len(rows),
            "context_cases": len(context_by_id),
            "fetch_errors": fetch_errors,
            "replay_output": str(replay_path),
        }
    elif args.command == "overture-context-replay":
        replay_payload = load_overture_context_replay(args.input)
        attributes = replay_payload.get("attributes") or ["website", "phone", "address", "category", "name"]
        rows = replay_payload["rows"]
        context_by_id = replay_payload["context_by_id"]
        report = evaluate_overture_context(
            rows,
            context_by_id,
            attributes=attributes,
            conflicts_only=True,
        )
        csv_path = Path(args.csv_output) if args.csv_output else ROOT / "reports" / "overture_context" / f"overture_context_decisions_{_timestamp()}.csv"
        write_overture_context_decisions(report["decisions"], csv_path)
        report.update(
            {
                "input": str(args.input),
                "rows": len(rows),
                "context_cases": len(context_by_id),
                "fetch_errors": replay_payload.get("fetch_errors", []),
                "output_csv": str(csv_path),
            }
        )
    elif args.command == "overture-gap-dorks":
        replay_payload = load_overture_context_replay(args.input)
        attributes = replay_payload.get("attributes") or ["website", "phone", "address", "category", "name"]
        eval_report = evaluate_overture_context(
            replay_payload["rows"],
            replay_payload["context_by_id"],
            attributes=attributes,
            conflicts_only=True,
        )
        rows = build_overture_gap_dork_rows(eval_report)
        csv_path = Path(args.csv_output) if args.csv_output else ROOT / "reports" / "overture_context" / f"overture_gap_dorks_{_timestamp()}.csv"
        write_overture_gap_dork_csv(rows, csv_path)
        report = {
            "input": str(args.input),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "gap_dork_audit": evaluate_overture_gap_dorks(eval_report),
            "gated_metrics": eval_report["gated_metrics"],
            "baseline_metrics": eval_report["baseline_metrics"],
        }
    elif args.command == "agreement-labels":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = build_project_a_agreement_labels(dataset_path, limit=args.limit, min_attributes=args.min_attributes)
        csv_path = write_label_csv(rows, ROOT / "reports" / "golden" / f"project_a_agreement_labels_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "label_type": "silver_agreement",
            "preview": rows[:3],
        }
    elif args.command == "import-james-golden":
        dataset_path = Path(args.input) if args.input else find_project_a_parquet(ROOT)
        if dataset_path is None:
            raise SystemExit("No project_a parquet found. Put it under data/project_a_samples.parquet or pass --input.")
        rows = build_project_a_labels_from_james_golden(dataset_path, args.james_csv, limit=args.limit)
        csv_path = write_label_csv(rows, ROOT / "reports" / "golden" / f"project_a_james_golden_labels_{_timestamp()}.csv")
        report = {
            "path": str(dataset_path),
            "source_csv": str(args.james_csv),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "label_type": "prior_projectterra_golden",
            "preview": rows[:3],
        }
    elif args.command == "import-david-labels":
        rows = build_project_a_labels_from_david_finalized(args.david_csv, split_name=args.split_name)
        csv_path = write_label_csv(rows, ROOT / "reports" / "golden" / f"project_a_david_{args.split_name}_labels_{_timestamp()}.csv")
        attribute_counts = {
            attribute: sum(1 for row in rows if row.get(f"{attribute}_truth_choice") or row.get(f"{attribute}_truth_value"))
            for attribute in ["website", "phone", "address", "category", "name"]
        }
        report = {
            "source_csv": str(args.david_csv),
            "rows": len(rows),
            "output_csv": str(csv_path),
            "label_type": "david_attribute_level_labels",
            "split_name": args.split_name,
            "attribute_counts": attribute_counts,
            "preview": rows[:3],
        }
    elif args.command == "dashboard":
        report = write_dashboard(args.reports_root, args.output_dir)
    elif args.command == "gui":
        report = write_dashboard(args.reports_root, args.output_dir)
    else:
        report = evaluate_harness_report(
            truth_path=args.truth,
            results_dir=args.results_dir,
            baseline_name=args.baseline,
            limit=args.limit,
            retrieval_path=args.input,
            retrieval_arm=args.arm,
        )

    output_path = _write_report(report, args.output, args.command)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.output:
        print(f"saved report to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
