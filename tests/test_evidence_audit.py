import csv
import json
import subprocess
import sys
from pathlib import Path

from places_attr_conflation.evidence_audit import audit_evidence_files, audit_evidence_rows, evaluate_evidence_gate


ROOT = Path(__file__).resolve().parents[1]


def _row(case_id: str, *, url: str = "https://example.com", fetch_status: str = "ok") -> dict[str, str]:
    return {
        "case_id": case_id,
        "attribute": "website",
        "priority_bucket": "P0_WEBSITE_BASELINE_WRONG",
        "website_label_guess": "OFFICIAL_CURRENT",
        "identity_label_guess": "",
        "url": url,
        "final_url": url,
        "content_hash": f"hash-{case_id}" if fetch_status == "ok" else "",
        "page_text": "Demo Cafe official page." if fetch_status == "ok" else "",
        "page_text_full_path": f"/tmp/{case_id}.txt" if fetch_status == "ok" else "",
        "page_text_excerpt": "Demo Cafe official page." if fetch_status == "ok" else "",
        "schema_org_detected": "true" if fetch_status == "ok" else "false",
        "localbusiness_schema_detected": "true" if fetch_status == "ok" else "false",
        "detected_phone": "(831) 555-0100" if fetch_status == "ok" else "",
        "detected_address": "123 Ocean St" if fetch_status == "ok" else "",
        "detected_name": "Demo Cafe" if fetch_status == "ok" else "",
        "detected_status": "active" if fetch_status == "ok" else "unknown",
        "identity_claims": "",
        "fetch_status": fetch_status,
        "fetch_error": "" if fetch_status == "ok" else "timeout",
        "source_type": "official_site",
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


def test_enriched_rows_with_good_fetch_pass_pilot_gate() -> None:
    rows = [_row(f"web-{idx}") for idx in range(25)]

    report = audit_evidence_rows(rows)
    gate = evaluate_evidence_gate(report, "first50")

    assert report["rows_total"] == 25
    assert report["rows_with_url"] == 25
    assert report["fetch_ok_rate"] == 1.0
    assert report["content_hash_rate"] == 1.0
    assert report["page_text_rate"] == 1.0
    assert report["rows_by_source_type"] == {"official_site": 25}
    assert gate["passed"] is True


def test_rows_without_urls_are_counted() -> None:
    rows = [_row("web-1"), {**_row("blank"), "url": "", "final_url": ""}]

    report = audit_evidence_rows(rows)

    assert report["rows_total"] == 2
    assert report["rows_with_url"] == 1
    assert report["rows_missing_url"] == 1
    assert report["url_rate"] == 0.5


def test_fetch_errors_lower_fetch_ok_rate() -> None:
    rows = [_row("ok"), {**_row("err", fetch_status="error"), "page_text": "Public website URL evidence: https://example.com"}]

    report = audit_evidence_rows(rows)

    assert report["fetch_ok_count"] == 1
    assert report["fetch_error_count"] == 1
    assert report["fetch_ok_rate"] == 0.5
    assert report["content_hash_rate"] == 0.5
    assert report["page_text_rate"] == 0.5


def test_identity_claims_and_detected_status_are_counted() -> None:
    moved = {
        **_row("moved"),
        "detected_status": "moved",
        "identity_claims": "MOVED",
        "website_label_guess": "OFFICIAL_STALE",
        "identity_label_guess": "MOVED_ENTITY",
    }
    report = audit_evidence_rows([moved, _row("active")])

    assert report["rows_with_identity_claims"] == 1
    assert report["identity_claim_rate"] == 0.5
    assert report["rows_by_detected_status"] == {"active": 1, "moved": 1}
    assert report["stale_or_identity_signal_count"] == 1


def test_evidence_audit_cli_fail_on_gate_exits_nonzero(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "evidence.csv", [_row("err", fetch_status="error")])
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_evidence.py",
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
    assert audit_evidence_files([csv_path])["rows_total"] == 1
