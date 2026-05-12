"""Render benchmark reports into a compact review dashboard."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardData:
    dataset: dict[str, object] | None
    baseline: dict[str, object] | None
    compare: dict[str, object] | None
    rerank: dict[str, object] | None
    combined: dict[str, object] | None
    smoke: dict[str, object] | None
    golden: dict[str, object] | None
    evidence: dict[str, object] | None
    paths: dict[str, str]
    batch_progress_rows: list[list[str]]


def _table_rows_without_separator(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def _safe_table_rows(lines: list[str], headers: list[str]) -> list[list[str]]:
    rows = _table_rows_without_separator(lines)
    if rows:
        return rows
    return [headers, ["missing"] + ["-" for _ in headers[1:]]]


def _load_json(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_json(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def latest_report_paths(reports_root: str | Path) -> dict[str, str]:
    root = Path(reports_root)
    harness = root / "harness"
    baseline = root / "baseline_metrics"
    ranker = root / "ranker"
    selected = {
        "dataset": _latest_json(root / "data", "project_a_summary*.json"),
        "baseline": _latest_json(baseline, "resolvepoi_*.json"),
        "compare": _latest_json(root / "retrieval_compare", "compare_*.json") or _latest_json(harness, "compare_*.json"),
        "replay_stats": _latest_json(root / "replay_stats", "replay_stats_*.json"),
        "merged_replay": _latest_json(root / "replay", "merged_*.json"),
        "resolver_replay": _latest_json(root / "resolver_replay", "resolver_on_replay_*.json"),
        "rerank": _latest_json(harness, "rerank_*.json"),
        "combined": _latest_json(harness, "all_*.json"),
        "smoke": _latest_json(harness, "smoke_*.json"),
        "golden": _latest_json(root / "golden", "project_a_golden_*.json") or _latest_json(harness, "golden_*.json"),
        "evidence": _latest_json(root / "evidence", "evidence-eval_*.json"),
        "conflict_dorks": _latest_json(ranker, "conflict_dorks_*.csv"),
    }
    return {name: str(path) for name, path in selected.items() if path is not None}


def build_dashboard_data(reports_root: str | Path) -> DashboardData:
    paths = latest_report_paths(reports_root)
    return DashboardData(
        dataset=_load_json(Path(paths["dataset"])) if "dataset" in paths else None,
        baseline=_load_json(Path(paths["baseline"])) if "baseline" in paths else None,
        compare=_load_json(Path(paths["compare"])) if "compare" in paths else None,
        rerank=_load_json(Path(paths["rerank"])) if "rerank" in paths else None,
        combined=_load_json(Path(paths["combined"])) if "combined" in paths else None,
        smoke=_load_json(Path(paths["smoke"])) if "smoke" in paths else None,
        golden=_load_json(Path(paths["golden"])) if "golden" in paths else None,
        evidence=_load_json(Path(paths["evidence"])) if "evidence" in paths else None,
        paths=paths,
        batch_progress_rows=_batch_progress_rows(reports_root),
    )


def _pct(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "-"


def _num(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "-"


def _pct_points(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f} pts"
    return "-"


def _compare_verdict(compare: dict[str, object] | None) -> str:
    if not compare:
        return "Retrieval verdict unavailable."
    targeted = compare.get("targeted", {})
    fallback = compare.get("fallback", {})
    if not isinstance(targeted, dict) or not isinstance(fallback, dict):
        return "Retrieval report is incomplete."
    if (
        float(targeted.get("authoritative_found_rate", 0.0)) > float(fallback.get("authoritative_found_rate", 0.0))
        and float(targeted.get("citation_precision", 0.0)) >= float(fallback.get("citation_precision", 0.0))
    ):
        return "Targeted search is ahead of loose search on replay."
    if float(targeted.get("authoritative_found_rate", 0.0)) < float(fallback.get("authoritative_found_rate", 0.0)):
        return "Targeted search is behind loose search on replay."
    return "Retrieval is mixed; keep collecting replay labels."


def _compare_highlights(compare: dict[str, object] | None) -> list[str]:
    if not compare:
        return ["No retrieval comparison report found."]
    targeted = compare.get("targeted", {})
    fallback = compare.get("fallback", {})
    if not isinstance(targeted, dict) or not isinstance(fallback, dict):
        return ["Retrieval comparison report is incomplete."]
    auth_delta = float(targeted.get("authoritative_found_rate", 0.0)) - float(fallback.get("authoritative_found_rate", 0.0))
    return [
        f"Authoritative found: {_pct(targeted.get('authoritative_found_rate'))} vs {_pct(fallback.get('authoritative_found_rate'))} ({_pct_points(auth_delta)})",
        f"Citation precision: {_pct(targeted.get('citation_precision'))} vs {_pct(fallback.get('citation_precision'))}",
        f"Top-1 authoritative: {_pct(targeted.get('top1_authoritative_rate'))} vs {_pct(fallback.get('top1_authoritative_rate'))}",
        f"Average attempts: {_num(targeted.get('average_search_attempts'))}",
    ]


def _baseline_table(baseline: dict[str, object] | None) -> list[str]:
    if not baseline:
        return ["No baseline report found."]
    metrics = baseline.get("metrics", {})
    if not isinstance(metrics, dict):
        return ["No baseline metrics found."]
    lines = [
        "| Attribute | Accuracy | Macro F1 | HC Wrong | Abstention |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for attribute in ("website", "phone", "address", "category", "name"):
        row = metrics.get(attribute, {})
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    attribute,
                    _pct(row.get("accuracy")),
                    _num(row.get("macro_f1")),
                    _pct(row.get("high_confidence_wrong_rate")),
                    _pct(row.get("abstention_rate")),
                ]
            )
            + " |"
        )
    return lines


def _dataset_lines(dataset: dict[str, object] | None) -> list[str]:
    if not dataset:
        return ["No project_a dataset summary found."]
    summary = dataset.get("summary", {})
    schema = dataset.get("schema", {})
    if not isinstance(summary, dict):
        return ["No project_a dataset summary found."]
    return [
        f"Path: {dataset.get('path', '-')}",
        f"Rows: {_num(summary.get('row_count'))}",
        f"Distinct id: {_num(summary.get('distinct_id_count'))}",
        f"Distinct base_id: {_num(summary.get('distinct_base_id_count'))}",
        f"Column count: {_num((schema.get('column_count') if isinstance(schema, dict) else None))}",
        f"Websites present: {_pct(summary.get('websites_present_rate'))}",
        f"Base websites present: {_pct(summary.get('base_websites_present_rate'))}",
        f"Phones present: {_pct(summary.get('phones_present_rate'))}",
        f"Base phones present: {_pct(summary.get('base_phones_present_rate'))}",
    ]


def _compare_table(compare: dict[str, object] | None) -> list[str]:
    if not compare:
        return ["No retrieval comparison report found."]
    lines = [
        "| Arm | Auth Found | Useful Found | Citation Precision | Top-1 Authoritative | Avg Attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("targeted", "fallback", "all"):
        row = compare.get(arm, {})
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _pct(row.get("authoritative_found_rate")),
                    _pct(row.get("useful_found_rate")),
                    _pct(row.get("citation_precision")),
                    _pct(row.get("top1_authoritative_rate")),
                    _num(row.get("average_search_attempts")),
                ]
            )
            + " |"
        )
    return lines


def _rerank_lines(rerank: dict[str, object] | None) -> list[str]:
    if not rerank:
        return ["No reranker report found."]
    if not rerank.get("available"):
        return [f"Reranker unavailable: {rerank.get('reason', 'unknown reason')}"]
    return [
        f"Training examples: {_num(rerank.get('training_examples'))}",
        f"Positive labels: {_num(rerank.get('positive_examples'))}",
        f"Negative labels: {_num(rerank.get('negative_examples'))}",
        f"Heuristic top-1 authoritative: {_pct(((rerank.get('heuristic') or {}).get('top1_authoritative_rate')) if isinstance(rerank.get('heuristic'), dict) else None)}",
        f"Reranker top-1 authoritative: {_pct(((rerank.get('reranker') or {}).get('top1_authoritative_rate')) if isinstance(rerank.get('reranker'), dict) else None)}",
        f"Improved top-1 authoritative: {'yes' if rerank.get('improved_top1_authoritative_rate') else 'no'}",
    ]


def _decision_lines(combined: dict[str, object] | None) -> list[str]:
    if not combined:
        return ["No combined report found."]
    decisions = combined.get("decisions", {})
    if not isinstance(decisions, dict) or not decisions:
        return ["No resolver decision summary found."]
    return [
        f"Accuracy: {_pct(decisions.get('accuracy'))}",
        f"Abstention rate: {_pct(decisions.get('abstention_rate'))}",
        f"High-confidence wrong rate: {_pct(decisions.get('high_confidence_wrong_rate'))}",
        f"Cases: {_num(decisions.get('total'))}",
    ]


def _replay_stats_lines(paths: dict[str, str]) -> list[str]:
    stats = _load_json(Path(paths["replay_stats"])) if "replay_stats" in paths else None
    if not stats:
        return ["No replay stats report found."]
    return [
        f"Episodes: {_num(stats.get('episodes_total'))}",
        f"Attempts: {_num(stats.get('attempts_total'))}",
        f"Pages: {_num(stats.get('pages_total'))}",
        f"Authoritative pages rate: {_pct(stats.get('authoritative_pages_rate'))}",
        f"Last merged replay: {paths.get('merged_replay', '-')}",
    ]


def _batch_progress_rows(reports_root: str | Path) -> list[list[str]]:
    root = Path(reports_root)
    manifests = sorted((root / "ranker").glob("conflict_dorks_*_batches/manifest.json"))
    if not manifests:
        return [["Batch", "Cases", "Cases With Pages", "Pages"], ["missing", "-", "-", "-"]]
    manifest_path = max(manifests, key=lambda path: path.stat().st_mtime)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_cases: dict[str, set[tuple[str, str]]] = {}
    for replay_path in sorted((root / "replay_collected").rglob("*.json")):
        try:
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for episode in payload.get("episodes", []) if isinstance(payload, dict) else []:
            if not isinstance(episode, dict):
                continue
            pages = sum(len(attempt.get("fetched_pages", [])) for attempt in episode.get("search_attempts", []) if isinstance(attempt, dict))
            if pages:
                case_id = str(episode.get("case_id", ""))
                attribute = str(episode.get("attribute", ""))
                case_pages = evidence_cases.setdefault(case_id, set())
                for attempt in episode.get("search_attempts", []):
                    if not isinstance(attempt, dict):
                        continue
                    for page in attempt.get("fetched_pages", []):
                        if isinstance(page, dict) and page.get("url"):
                            case_pages.add((attribute, str(page["url"])))
    rows = [["Batch", "Cases", "Cases With Pages", "Pages"]]
    batch_entries = manifest.get("batches", [])
    if isinstance(batch_entries, int):
        batch_entries = manifest.get("files", [])
    for batch in batch_entries if isinstance(batch_entries, list) else []:
        if not isinstance(batch, dict):
            continue
        batch_path = Path(str(batch.get("path", "")))
        if not batch_path.is_absolute():
            batch_path = root.parent / batch_path
        case_ids: set[str] = set()
        if batch_path.exists():
            import csv

            with batch_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("id"):
                        case_ids.add(str(row["id"]))
        cases_with_pages = sum(1 for case_id in case_ids if evidence_cases.get(case_id))
        pages = sum(len(evidence_cases.get(case_id, set())) for case_id in case_ids)
        rows.append([str(batch.get("batch", "")), str(batch.get("cases", len(case_ids))), str(cases_with_pages), str(pages)])
    return rows


def _smoke_lines(smoke: dict[str, object] | None) -> list[str]:
    if not smoke:
        return ["No smoke report found."]
    mode = smoke.get("mode", "unknown")
    lines = [f"Mode: {mode}"]
    results = smoke.get("results", [])
    if isinstance(results, list):
        ok_count = sum(1 for row in results if isinstance(row, dict) and row.get("status") == "ok")
        lines.append(f"Successful live checks: {ok_count}/{len(results)}")
    return lines


def _golden_table(golden: dict[str, object] | None) -> list[str]:
    if not golden:
        return ["No project_a golden report found."]
    baselines = golden.get("baselines", {})
    if not isinstance(baselines, dict):
        return ["No project_a golden baseline metrics found."]
    lines = [
        "| Baseline | Attribute | Accuracy | Conflict Accuracy | Conflict Coverage | Conflict Abstention | HC Wrong | Conflict Labels | Labels |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline_name in sorted(baselines):
        baseline = baselines.get(baseline_name, {})
        if not isinstance(baseline, dict):
            continue
        metrics = baseline.get("metrics", {})
        conflict_metrics = baseline.get("conflict_metrics", {})
        if not isinstance(metrics, dict):
            continue
        for attribute in ("website", "phone", "address", "category", "name"):
            row = metrics.get(attribute, {})
            conflict_row = conflict_metrics.get(attribute, {}) if isinstance(conflict_metrics, dict) else {}
            if not isinstance(row, dict) or not row.get("total"):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        baseline_name,
                        attribute,
                        _pct(row.get("accuracy")),
                        _pct(conflict_row.get("accuracy") if isinstance(conflict_row, dict) else None),
                        _pct(conflict_row.get("coverage") if isinstance(conflict_row, dict) else None),
                        _pct(conflict_row.get("abstention_rate") if isinstance(conflict_row, dict) else None),
                        _pct(row.get("high_confidence_wrong_rate")),
                        _num(conflict_row.get("total") if isinstance(conflict_row, dict) else None),
                        _num(row.get("total")),
                    ]
                )
                + " |"
            )
    return lines if len(lines) > 2 else ["No labeled project_a attributes found."]


def _evidence_lines(evidence: dict[str, object] | None) -> list[str]:
    if not evidence:
        return ["No synthetic evidence evaluation report found."]
    resolver = evidence.get("resolver", {})
    baseline = evidence.get("baseline", {})
    if not isinstance(resolver, dict) or not isinstance(baseline, dict):
        return ["Synthetic evidence report is missing resolver or baseline metrics."]
    return [
        f"Mode: {evidence.get('mode', '-')}",
        f"Cases: {_num(evidence.get('total'))}",
        f"Resolver accuracy: {_pct(resolver.get('accuracy'))}",
        f"Resolver coverage: {_pct(resolver.get('coverage'))}",
        f"Resolver abstention: {_pct(resolver.get('abstention_rate'))}",
        f"Resolver high-confidence wrong: {_pct(resolver.get('high_confidence_wrong_rate'))}",
        f"Baseline accuracy: {_pct(baseline.get('accuracy'))}",
        f"Warning: {evidence.get('warning', 'Synthetic evidence validates system behavior only.')}",
    ]


def render_markdown(data: DashboardData) -> str:
    verdict = _compare_verdict(data.compare)
    lines = [
        "# Benchmark Dashboard",
        "",
        "## What Is Stopping Us",
        "",
    ]
    if data.compare and isinstance(data.compare.get("targeted"), dict) and isinstance(data.compare.get("fallback"), dict):
        targeted = data.compare["targeted"]
        fallback = data.compare["fallback"]
        lines.extend(
            [
                f"- Retrieval proof is still small-sample: targeted authoritative found is {_pct(targeted.get('authoritative_found_rate'))} versus fallback {_pct(fallback.get('authoritative_found_rate'))} on {_num(targeted.get('total'))} replay cases.",
                "- The reranker is still optional because it has not beaten the heuristic on replay.",
                "- Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.",
            ]
        )
    else:
        lines.append("- The next blocker is missing or incomplete benchmark reports.")

    lines.extend(
        [
            "",
            "## At a Glance",
            "",
            f"- Verdict: {verdict}",
            *[f"- {line}" for line in _compare_highlights(data.compare)],
            f"- Conflict dork queue: `{data.paths.get('conflict_dorks', 'missing')}`",
            "",
            "## Current Benchmarks",
            "",
            "### Raw Matched-Pair Dataset",
            "",
            * [f"- {line}" for line in _dataset_lines(data.dataset)],
            "",
            "### ResolvePOI Baseline",
            "",
            * _baseline_table(data.baseline),
            "",
            "### Retrieval Arms",
            "",
            * _compare_table(data.compare),
            "",
            "### Replay Coverage",
            "",
            * [f"- {line}" for line in _replay_stats_lines(data.paths)],
            "",
            "### Batch Progress",
            "",
            *[
                "| " + " | ".join(row) + " |" if idx == 0 else "| " + " | ".join(row) + " |"
                for idx, row in enumerate(data.batch_progress_rows)
            ],
            "",
            "### Reranker",
            "",
            * [f"- {line}" for line in _rerank_lines(data.rerank)],
            "",
            "### Resolver Decisions",
            "",
            * [f"- {line}" for line in _decision_lines(data.combined)],
            "",
            "### Project A Golden Labels",
            "",
            * _golden_table(data.golden),
            "",
            "### Synthetic Evidence Validation",
            "",
            * [f"- {line}" for line in _evidence_lines(data.evidence)],
            "",
            "### Live Smoke",
            "",
            * [f"- {line}" for line in _smoke_lines(data.smoke)],
            "",
            "## Report Files",
            "",
        ]
    )
    for name, path in sorted(data.paths.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def render_html(data: DashboardData) -> str:
    verdict = _compare_verdict(data.compare)
    baseline_rows = _safe_table_rows(
        _baseline_table(data.baseline),
        ["Attribute", "Accuracy", "Macro F1", "HC Wrong", "Abstention"],
    )
    compare_rows = _safe_table_rows(
        _compare_table(data.compare),
        ["Arm", "Auth Found", "Useful Found", "Citation Precision", "Top-1 Authoritative", "Avg Attempts"],
    )
    golden_rows = _safe_table_rows(
        _golden_table(data.golden),
        ["Baseline", "Attribute", "Accuracy", "Coverage", "HC Wrong", "Labels"],
    )
    bundle = {
        "verdict": verdict,
        "compare_highlights": _compare_highlights(data.compare),
        "dataset": _dataset_lines(data.dataset),
        "stoppers": [
            "Retrieval proof is still small-sample." if data.compare else "Retrieval comparison report missing.",
            "The reranker is still optional because it has not beaten the heuristic on replay." if data.rerank else "Reranker report missing.",
            "Resolver improvement over the 200-row ResolvePOI baseline is not yet proven on a larger labeled evidence corpus.",
        ],
        "baseline_rows": baseline_rows,
        "compare_rows": compare_rows,
        "replay_stats": _replay_stats_lines(data.paths),
        "batch_progress_rows": data.batch_progress_rows,
        "rerank": _rerank_lines(data.rerank),
        "decisions": _decision_lines(data.combined),
        "golden_rows": golden_rows,
        "evidence": _evidence_lines(data.evidence),
        "smoke": _smoke_lines(data.smoke),
        "paths": data.paths,
    }
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>MLAttributes Benchmark Dashboard</title>",
            "<style>",
            ":root { --paper:#f4f1ea; --ink:#17201d; --muted:#5f685f; --accent:#1f5c4d; --accent-soft:#d8e8de; --panel:#fffdf8; --line:#d7cfbf; --warn:#8a3b1e; }",
            "body { font-family: Georgia, 'Times New Roman', serif; margin: 0; background: linear-gradient(180deg, #ede7da 0%, var(--paper) 220px); color: var(--ink); }",
            "main { max-width: 1180px; margin: 0 auto; padding: 36px 20px 72px; }",
            "h1, h2, h3, button { font-family: 'Trebuchet MS', 'Gill Sans', sans-serif; letter-spacing: 0; }",
            "h1 { font-size: 2.4rem; margin: 0 0 0.5rem; }",
            "p.lead { color: var(--muted); max-width: 70ch; }",
            ".hero { display:grid; grid-template-columns: 1.7fr 1fr; gap: 16px; align-items:start; margin-bottom: 22px; }",
            ".panel { background: var(--panel); border: 1px solid var(--line); padding: 16px; }",
            ".cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 14px 0 20px; }",
            ".card { background: var(--panel); border: 1px solid var(--line); padding: 14px; min-height: 92px; }",
            ".card .label { color: var(--muted); font-size: 0.88rem; text-transform: uppercase; }",
            ".card .value { font-size: 1.8rem; margin-top: 8px; color: var(--accent); }",
            ".tabs { display:flex; flex-wrap:wrap; gap: 8px; margin: 8px 0 16px; }",
            ".tab { border: 1px solid var(--line); background: #efe8d9; color: var(--ink); padding: 10px 14px; cursor: pointer; }",
            ".tab.active { background: var(--accent); color: white; border-color: var(--accent); }",
            ".view { display:none; }",
            ".view.active { display:block; }",
            "table { width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.2rem; background: var(--panel); }",
            "th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }",
            "th { background: var(--accent-soft); }",
            "ul { padding-left: 1.2rem; }",
            ".stopper { color: var(--warn); }",
            ".path-list li { margin-bottom: 6px; word-break: break-all; }",
            ".split { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }",
            "code { background: #ece6d8; padding: 2px 5px; border-radius: 4px; }",
            "@media (max-width: 860px) { .hero, .split { grid-template-columns: 1fr; } h1 { font-size: 2rem; } }",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<section class='hero'>",
            "<div>",
            "<h1>Benchmark Viewer</h1>",
            "<p class='lead'>Repeatable, reproducible benchmark surface for Overture Places attribute conflation. This GUI reads the latest saved reports and keeps the workflow visible without opening raw JSON.</p>",
            "</div>",
            "<div class='panel'>",
            "<strong>Workflow Cut</strong>",
            "<ul>",
            "<li>Baseline reproduction</li>",
            "<li>Retrieval replay compare</li>",
            "<li>Raw pair review export</li>",
            "<li>Resolver and abstention evaluation</li>",
            "</ul>",
            "</div>",
            "</section>",
            "<section class='cards'>",
            f"<div class='card'><div class='label'>Raw Pair Rows</div><div class='value'>{html.escape(next((line.split(': ',1)[1] for line in bundle['dataset'] if line.startswith('Rows: ')), '-'))}</div></div>",
            f"<div class='card'><div class='label'>Website Baseline</div><div class='value'>{html.escape(baseline_rows[1][1] if len(baseline_rows) > 1 else '-')}</div></div>",
            f"<div class='card'><div class='label'>Targeted Auth Found</div><div class='value'>{html.escape(compare_rows[1][1] if len(compare_rows) > 1 else '-')}</div></div>",
            f"<div class='card'><div class='label'>Replay Episodes</div><div class='value'>{html.escape(next((line.split(': ',1)[1] for line in bundle['replay_stats'] if line.startswith('Episodes: ')), '-'))}</div></div>",
            f"<div class='card'><div class='label'>Resolver Abstention</div><div class='value'>{html.escape(next((line.split(': ',1)[1] for line in bundle['decisions'] if line.startswith('Abstention rate: ')), '-'))}</div></div>",
            "</section>",
            "<section class='panel'>",
            "<h2>At a Glance</h2>",
            f"<p><strong>{html.escape(bundle['verdict'])}</strong></p>",
            "<ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["compare_highlights"]],
            "</ul>",
            "</section>",
            "<section class='panel'>",
            "<h2>What Is Stopping Us</h2>",
            "<ul>",
            *[f"<li class='stopper'>{html.escape(line)}</li>" for line in bundle["stoppers"]],
            "</ul>",
            "</section>",
            "<section>",
            "<div class='tabs'>",
            "<button class='tab active' data-view='overview'>Overview</button>",
            "<button class='tab' data-view='baseline'>Baseline</button>",
            "<button class='tab' data-view='retrieval'>Retrieval</button>",
            "<button class='tab' data-view='progress'>Batch Progress</button>",
            "<button class='tab' data-view='golden'>Golden</button>",
            "<button class='tab' data-view='evidence'>Evidence</button>",
            "<button class='tab' data-view='reports'>Reports</button>",
            "</div>",
            "<div id='overview' class='view active'>",
            "<div class='split'>",
            "<div class='panel'><h3>Raw Dataset</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["dataset"]],
            "</ul></div>",
            "<div class='panel'><h3>Reranker</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["rerank"]],
            "</ul><h3>Live Smoke</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["smoke"]],
            "</ul></div>",
            "</div>",
            "</div>",
            "<div id='baseline' class='view'>",
            "<div class='panel'><h3>ResolvePOI Baseline</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in baseline_rows[0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in baseline_rows[1:]
            ],
            "</tbody></table>",
            "</div></div>",
            "<div id='retrieval' class='view'>",
            "<div class='split'>",
            "<div class='panel'><h3>Retrieval Arms</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in compare_rows[0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in compare_rows[1:]
            ],
            "</tbody></table></div>",
            "<div class='panel'><h3>Resolver Decisions</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["decisions"]],
            "</ul></div>",
            "</div></div>",
            "<div id='progress' class='view'>",
            "<div class='split'>",
            "<div class='panel'><h3>Replay Coverage</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["replay_stats"]],
            "</ul></div>",
            "<div class='panel'><h3>Batch Progress</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in bundle["batch_progress_rows"][0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in bundle["batch_progress_rows"][1:]
            ],
            "</tbody></table></div>",
            "</div></div>",
            "<div id='golden' class='view'>",
            "<div class='panel'><h3>Project A Golden Labels</h3>",
            "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in golden_rows[0]) + "</tr></thead><tbody>",
            *[
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in golden_rows[1:]
            ],
            "</tbody></table></div>",
            "</div>",
            "<div id='evidence' class='view'>",
            "<div class='panel'><h3>Synthetic Evidence Validation</h3><ul>",
            *[f"<li>{html.escape(line)}</li>" for line in bundle["evidence"]],
            "</ul></div>",
            "</div>",
            "<div id='reports' class='view'>",
            "<div class='panel'><h3>Latest Report Files</h3><ul class='path-list'>",
            *[f"<li><strong>{html.escape(name)}</strong>: <code>{html.escape(path)}</code></li>" for name, path in sorted(bundle["paths"].items())],
            "</ul></div></div>",
            "<script>",
            "for (const button of document.querySelectorAll('.tab')) {",
            "  button.addEventListener('click', () => {",
            "    for (const tab of document.querySelectorAll('.tab')) tab.classList.remove('active');",
            "    for (const view of document.querySelectorAll('.view')) view.classList.remove('active');",
            "    button.classList.add('active');",
            "    document.getElementById(button.dataset.view).classList.add('active');",
            "  });",
            "}",
            "</script>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def write_dashboard(reports_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    data = build_dashboard_data(reports_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    markdown_path = output / "index.md"
    html_path = output / "index.html"
    json_path = output / "latest.json"
    markdown_path.write_text(render_markdown(data), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    json_path.write_text(json.dumps(data.paths, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "markdown": str(markdown_path),
        "html": str(html_path),
        "latest": str(json_path),
    }
