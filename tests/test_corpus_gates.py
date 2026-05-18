from places_attr_conflation.corpus_gates import (
    ReplayCorpusV1Thresholds,
    evaluate_replay_corpus_v1_gate,
    evaluate_website_label_coverage,
)
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


def test_website_label_coverage_counts_hard_website_labels() -> None:
    episodes = [
        _episode(case_id="current", website_label="OFFICIAL_CURRENT"),
        _episode(case_id="stale", website_label="OFFICIAL_STALE"),
        _episode(case_id="dead", website_label="OFFICIAL_DEAD"),
        _episode(case_id="agg", website_label="AGGREGATOR_ONLY"),
        _episode(case_id="category", attribute="category"),
    ]

    coverage = evaluate_website_label_coverage(episodes)

    assert coverage["website_episodes_total"] == 4
    assert coverage["website_labeled_total"] == 4
    assert coverage["website_label_coverage_rate"] == 1.0
    assert coverage["stale_website_cases"] == 1
    assert coverage["dead_domain_cases"] == 1
    assert coverage["aggregator_only_cases"] == 1


def test_replay_corpus_v1_gate_fails_thin_corpus() -> None:
    report = evaluate_replay_corpus_v1_gate([_episode(case_id="one", website_label="OFFICIAL_CURRENT")])

    assert report["passed"] is False
    assert report["checks"]["total_replay_cases"] is False
    assert report["checks"]["website_cases"] is False


def test_replay_corpus_v1_gate_can_pass_with_small_custom_thresholds() -> None:
    episodes = [
        _episode(
            case_id="stale",
            website_label="OFFICIAL_STALE",
            identity_label="BRANCH_AMBIGUITY",
            case_type="WRONG_BRANCH",
            expected_abstain=True,
            review_status="accepted",
        ),
        _episode(
            case_id="same-address",
            website_label="AGGREGATOR_ONLY",
            identity_label="NEW_ENTITY_SAME_ADDRESS",
            case_type="AGGREGATOR_ECHO",
            expected_abstain=True,
            review_status="needs_more_evidence",
        ),
    ]
    thresholds = ReplayCorpusV1Thresholds(
        min_total_replay_cases=2,
        min_website_cases=2,
        min_identity_labeled_cases=2,
        min_reviewed_cases=2,
        min_expected_abstain_cases=2,
        min_stale_website_cases=1,
        min_wrong_branch_cases=1,
        min_aggregator_echo_cases=1,
        min_new_entity_same_address_cases=1,
        min_website_label_coverage_rate=1.0,
        min_case_type_coverage_rate=1.0,
    )

    report = evaluate_replay_corpus_v1_gate(episodes, thresholds)

    assert report["passed"] is True
    assert all(report["checks"].values())
