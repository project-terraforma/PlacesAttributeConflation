import csv
import subprocess
import sys
from pathlib import Path

from places_attr_conflation.evidence_todo import recommended_targets, render_evidence_todo


ROOT = Path(__file__).resolve().parents[1]


def test_recommended_evidence_targets_by_priority_bucket() -> None:
    assert recommended_targets("P0_WEBSITE_MISSING") == [
        "official website",
        "contact/about page",
        "social page only if official unavailable",
    ]
    assert recommended_targets("P0_WEBSITE_AGGREGATOR_OR_SOCIAL") == [
        "official website",
        "aggregator/social page that caused conflict",
        "contact/location page",
    ]
    assert recommended_targets("P0_WEBSITE_CHAIN_OR_BRANCH") == [
        "brand location page",
        "branch-specific page",
        "chain homepage if it is the wrong candidate",
    ]
    assert "moved/formerly/new ownership page" in recommended_targets("P0_IDENTITY_DRIFT_WEBSITE")
    assert "Overture/category context if available" in recommended_targets("P1_CATEGORY_TAXONOMY")


def test_output_includes_query_and_label_guesses() -> None:
    markdown = render_evidence_todo(
        [
            {
                "case_id": "case-1",
                "attribute": "website",
                "priority_bucket": "P0_IDENTITY_DRIFT_WEBSITE",
                "case_type_guess": "IDENTITY_DRIFT",
                "identity_label_guess": "MOVED_ENTITY",
                "website_label_guess": "OFFICIAL_STALE",
                "difficulty": "HARD",
                "query": '"Demo Cafe" "moved"',
                "current_value": "https://current.example",
                "base_value": "https://old.example",
                "prediction": "https://old.example",
            }
        ]
    )

    assert "case-1" in markdown
    assert "- query: \"Demo Cafe\" \"moved\"" in markdown
    assert "- identity_label_guess: MOVED_ENTITY" in markdown
    assert "- website_label_guess: OFFICIAL_STALE" in markdown
    assert "old official domain" in markdown


def test_no_fake_urls_inserted() -> None:
    markdown = render_evidence_todo(
        [
            {
                "case_id": "case-1",
                "attribute": "website",
                "priority_bucket": "P0_WEBSITE_MISSING",
                "query": "demo cafe official website",
            }
        ]
    )

    assert "https://" not in markdown
    assert "http://" not in markdown
    assert "- evidence URL(s):\n  -" in markdown


def test_cli_writes_todo_markdown(tmp_path: Path) -> None:
    csv_path = tmp_path / "evidence_template_001.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "attribute", "priority_bucket", "query"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case-1",
                "attribute": "website",
                "priority_bucket": "P0_WEBSITE_CHAIN_OR_BRANCH",
                "query": "demo brand location",
            }
        )
    output = tmp_path / "evidence_url_todo_001.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_evidence_todo.py",
            "--input",
            str(csv_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    markdown = output.read_text(encoding="utf-8")
    assert "branch-specific page" in markdown
    assert "demo brand location" in markdown
