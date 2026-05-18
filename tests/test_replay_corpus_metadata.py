from pathlib import Path

from places_attr_conflation.replay import (
    FetchedPage,
    ReplayEpisode,
    SearchAttempt,
    dump_replay_corpus,
    load_replay_corpus,
)


def test_replay_episode_corpus_metadata_round_trips(tmp_path: Path) -> None:
    episode = ReplayEpisode(
        case_id="case-1",
        attribute="website",
        place={"current_value": "https://old.example", "base_value": "https://new.example"},
        gold_value="https://new.example",
        search_attempts=[
            SearchAttempt(
                layer="official",
                query='"Example" official website',
                fetched_pages=[
                    FetchedPage(
                        url="https://new.example",
                        source_type="official_site",
                        extracted_values={"website": "https://new.example"},
                        evidence_role="supporting_gold",
                        source_family_id="official_site:new.example",
                    )
                ],
            )
        ],
        identity_label="SAME_ENTITY",
        case_type="WEBSITE_OFFICIAL_CURRENT",
        expected_decision="https://new.example",
        expected_abstain=False,
        truth_source_type="official_site",
        label_origin="unit_test",
        website_label="OFFICIAL_CURRENT",
        difficulty="EASY",
        review_status="accepted",
        reviewer_notes="Verified from official contact page.",
    )

    out = tmp_path / "replay.json"
    dump_replay_corpus([episode], out)
    loaded = load_replay_corpus(out)[0]

    assert loaded.website_label == "OFFICIAL_CURRENT"
    assert loaded.difficulty == "EASY"
    assert loaded.review_status == "accepted"
    assert loaded.reviewer_notes == "Verified from official contact page."
    assert loaded.search_attempts[0].fetched_pages[0].evidence_role == "supporting_gold"
    assert loaded.search_attempts[0].fetched_pages[0].source_family_id == "official_site:new.example"


def test_legacy_replay_episode_without_corpus_metadata_still_loads() -> None:
    loaded = ReplayEpisode.from_dict(
        {
            "case_id": "legacy-1",
            "attribute": "website",
            "place": {},
            "gold_value": "https://example.com",
            "search_attempts": [],
        }
    )

    assert loaded.website_label == ""
    assert loaded.difficulty == ""
    assert loaded.review_status == ""
    assert loaded.reviewer_notes == ""
