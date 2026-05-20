import csv
import json
import subprocess
import sys
from pathlib import Path

from places_attr_conflation.workplan_audit import audit_workplan_files, audit_workplan_rows, evaluate_first50_gate


ROOT = Path(__file__).resolve().parents[1]


def _row(case_id: str, attribute: str = "website", bucket: str = "P0_WEBSITE_BASELINE_WRONG", query: str = "query") -> dict[str, str]:
    return {
        "case_id": case_id,
        "attribute": attribute,
        "priority_bucket": bucket,
        "case_type_guess": "WEBSITE_CONFLICT",
        "website_label_guess": "OFFICIAL_STALE" if attribute == "website" else "",
        "layer": "official",
        "query": query,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_batch_with_mostly_website_p0_passes_first50_gate() -> None:
    rows = [
        _row("web-1", query="q1"),
        _row("web-2", query="q2"),
        _row("web-3", query="q3"),
        _row("web-4", query="q4"),
        _row("cat-1", attribute="category", bucket="P1_CATEGORY_CONFLICT", query="q5"),
    ]

    report = audit_workplan_rows(rows)
    gate = evaluate_first50_gate(report)

    assert report["unique_case_attribute_pairs"] == 5
    assert report["website_case_share"] == 0.8
    assert report["p0_case_share"] == 0.8
    assert gate["passed"] is True


def test_batch_with_empty_queries_fails_gate() -> None:
    rows = [_row(f"web-{idx}", query=f"q{idx}") for idx in range(4)]
    rows.append(_row("web-empty", query=""))

    report = audit_workplan_rows(rows)
    gate = evaluate_first50_gate(report)

    assert report["empty_query_count"] == 1
    assert gate["checks"]["empty_query_count"] is False
    assert gate["passed"] is False


def test_batch_with_missing_priority_bucket_fails_gate() -> None:
    rows = [_row(f"web-{idx}", query=f"q{idx}") for idx in range(4)]
    rows.append(_row("web-missing", bucket="", query="q-missing"))

    report = audit_workplan_rows(rows)
    gate = evaluate_first50_gate(report)

    assert report["missing_priority_bucket_count"] == 1
    assert gate["checks"]["missing_priority_bucket_count"] is False
    assert gate["passed"] is False


def test_duplicate_query_rate_is_computed_against_total_rows() -> None:
    rows = [
        _row("web-1", query="same"),
        _row("web-2", query="same"),
        _row("web-3", query="same"),
        _row("web-4", query="unique-1"),
        _row("web-5", query="unique-2"),
    ]

    report = audit_workplan_rows(rows)

    assert report["duplicate_query_count"] == 2
    assert report["duplicate_query_rate"] == 0.4


def test_missing_website_label_guess_for_website_rows_is_counted() -> None:
    rows = [
        _row("web-1", query="q1"),
        {**_row("web-2", query="q2"), "website_label_guess": ""},
        _row("cat-1", attribute="category", query="q3"),
    ]

    report = audit_workplan_rows(rows)

    assert report["missing_website_label_guess_for_website_count"] == 1


def test_audit_uses_template_query_diversity_when_templates_exist(tmp_path: Path) -> None:
    batch_rows = [
        _row("web-1", query="same"),
        _row("web-2", query="same"),
        _row("web-3", query="same"),
        _row("web-4", query="same"),
        _row("web-5", query="same"),
    ]
    template_rows = [
        _row("web-1", query="q1"),
        _row("web-2", query="q1"),
        _row("web-3", query="q2"),
        _row("web-4", query="q2"),
        _row("web-5", query="q3"),
    ]
    batch = _write_csv(tmp_path / "batch_001.csv", batch_rows)
    _write_csv(tmp_path / "evidence_template_001.csv", template_rows)

    report = audit_workplan_files([batch])

    assert report["batch_duplicate_query_rate"] == 0.8
    assert report["template_duplicate_query_count"] == 2
    assert report["template_duplicate_query_rate"] == 0.4
    assert report["template_max_query_occurrences"] == 2
    assert report["template_duplicate_query_cap_respected"] is True
    assert report["gate"]["query_evaluation_scope"] == "evidence_template"
    assert report["gate"]["checks"]["duplicate_query_rate"] is True
    assert report["gate"]["passed"] is True


def test_workplan_audit_cli_writes_deterministic_report_and_fails_gate(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "batch_001.csv", [_row("web-1", query="")])
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_workplan.py",
            "--input",
            str(csv_path),
            "--output",
            str(output),
            "--fail-on-gate",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == json.loads(result.stdout)
    assert payload["gate"]["passed"] is False
    assert audit_workplan_files([csv_path])["rows_total"] == 1
