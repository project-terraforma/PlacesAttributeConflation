from places_attr_conflation.corpus_stats import replay_corpus_label_stats
from places_attr_conflation.replay import ReplayEpisode


def _episode(**overrides: object) -> ReplayEpisode:
    data = {
        "case_id": "case-1",
        "attribute": "website",
        "place": {},
        "gold_value": "https://example.com",
        "search_attempts": [],
    }
    data.update(overrides)
    return ReplayEpisode(**data)  # type: ignore[arg-type]


def test_replay_corpus_label_stats_counts_operational_coverage() -> None:
    episodes = [
        _episode(
            case_id="official-current",
            website_label="OFFICIAL_CURRENT",
            identity_label="SAME_ENTITY",
            case_type="WEBSITE_OFFICIAL_CURRENT",
            expected_abstain=False,
            truth_source_type="official_site",
            review_status="accepted",
        ),
        _episode(
            case_id="stale-official",
            website_label="OFFICIAL_STALE",
            identity_label="STALE_OFFICIAL_SITE",
            case_type="OFFICIAL_STALE_DIRECTORY_CURRENT",
            expected_abstain=True,
            truth_source_type="official_site",
            review_status="needs_more_evidence",
        ),
        _episode(case_id="category", attribute="category", case_type="SIMPLE_AGREEMENT"),
    ]

    stats = replay_corpus_label_stats(episodes)

    assert stats["episodes_total"] == 3
    assert stats["episodes_with_website_label"] == 2
    assert stats["episodes_with_identity_label"] == 2
    assert stats["episodes_with_expected_abstain"] == 2
    assert stats["episodes_with_truth_source_type"] == 2
    assert stats["website_heavy_count"] == 2
    assert stats["identity_drift_count"] == 1
    assert stats["abstention_expected_count"] == 1
    assert stats["reviewed_count"] == 2
    assert stats["hard_case_count"] == 2
    assert stats["episodes_by_website_label"]["OFFICIAL_STALE"] == 1
