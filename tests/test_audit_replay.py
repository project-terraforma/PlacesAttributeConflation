import csv
import json
import subprocess
import sys
from pathlib import Path

from places_attr_conflation.replay import FetchedPage, ReplayEpisode, SearchAttempt, dump_replay_corpus, load_replay_corpus
from places_attr_conflation.replay_audit import audit_replay_episodes


ROOT = Path(__file__).resolve().parents[1]


def test_replay_audit_counts_episode_labels_and_page_features(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    episode = ReplayEpisode(
        case_id="case-1",
        attribute="website",
        place={},
        gold_value="",
        case_type="WEBSITE_BASELINE_WRONG",
        website_label="AMBIGUOUS_WEBSITE",
        identity_label="MOVED_ENTITY",
        expected_abstain=True,
        search_attempts=[
            SearchAttempt(
                layer="official",
                query="demo official",
                fetched_pages=[
                    FetchedPage(
                        url="https://example.com",
                        source_type="official_site",
                        page_text="Demo moved.",
                        extracted_values={"content_hash": "abc", "identity_claims": "MOVED"},
                    )
                ],
            )
        ],
    )
    dump_replay_corpus([episode], replay)

    report = audit_replay_episodes(load_replay_corpus(replay))

    assert report["episodes_total"] == 1
    assert report["episodes_by_attribute"] == {"website": 1}
    assert report["episodes_with_fetched_pages"] == 1
    assert report["episodes_with_case_type"] == 1
    assert report["episodes_with_website_label"] == 1
    assert report["episodes_with_identity_label"] == 1
    assert report["episodes_with_expected_abstain"] == 1
    assert report["pages_total"] == 1
    assert report["pages_with_source_type"] == 1
    assert report["pages_with_content_hash"] == 1
    assert report["pages_with_identity_claims"] == 1


def test_audit_replay_cli_writes_json(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    output = tmp_path / "audit.json"
    dump_replay_corpus(
        [
            ReplayEpisode(
                case_id="case-1",
                attribute="website",
                place={},
                gold_value="",
                search_attempts=[SearchAttempt(layer="official", query="q", fetched_pages=[])],
            )
        ],
        replay,
    )

    result = subprocess.run(
        [sys.executable, "scripts/audit_replay.py", "--input", str(replay), "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == json.loads(result.stdout)
    assert payload["episodes_total"] == 1
    assert payload["pages_total"] == 0


def test_build_replay_from_workplan_preserves_guesses_and_evidence_pages(tmp_path: Path) -> None:
    batch = tmp_path / "batch_001.csv"
    evidence = tmp_path / "evidence_template_001.csv"
    replay = tmp_path / "merged.json"
    with batch.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "base_id",
                "attribute",
                "truth",
                "prediction",
                "baseline",
                "current_value",
                "base_value",
                "priority_bucket",
                "case_type_guess",
                "identity_label_guess",
                "website_label_guess",
                "difficulty",
                "layer",
                "query",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "case-1",
                "base_id": "base-1",
                "attribute": "website",
                "truth": "",
                "prediction": "https://old.example",
                "baseline": "hybrid",
                "current_value": "https://new.example",
                "base_value": "https://old.example",
                "priority_bucket": "P0_IDENTITY_DRIFT_WEBSITE",
                "case_type_guess": "IDENTITY_DRIFT",
                "identity_label_guess": "MOVED_ENTITY",
                "website_label_guess": "OFFICIAL_STALE",
                "difficulty": "HARD",
                "layer": "official",
                "query": "demo moved official",
            }
        )
    with evidence.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "attribute",
                "layer",
                "query",
                "url",
                "title",
                "page_text_excerpt",
                "source_type",
                "content_hash",
                "identity_claims",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case-1",
                "attribute": "website",
                "layer": "official",
                "query": "demo moved official",
                "url": "https://example.com",
                "title": "Demo",
                "page_text_excerpt": "Demo moved.",
                "source_type": "official_site",
                "content_hash": "abc",
                "identity_claims": "MOVED",
            }
        )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_replay_from_workplan.py",
            "--batch",
            str(batch),
            "--evidence",
            str(evidence),
            "--output",
            str(replay),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    episode = load_replay_corpus(replay)[0]
    assert episode.case_type == "IDENTITY_DRIFT"
    assert episode.identity_label == "MOVED_ENTITY"
    assert episode.website_label == "OFFICIAL_STALE"
    assert episode.review_status == "unreviewed"
    assert episode.search_attempts[0].fetched_pages[0].url == "https://example.com"
