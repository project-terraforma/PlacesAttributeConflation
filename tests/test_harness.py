import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.harness import (
    build_ranker_dataset_rows,
    compare_arms,
    compare_reranker_on_replay,
    dump_retrieval_episodes,
    evaluate_dork_audit_gate,
    evaluate_final_decisions,
    evaluate_harness_report,
    evaluate_resolver_on_replay,
    evaluate_retrieval_episodes,
    evaluate_retrieval_quality_gate,
    load_retrieval_episodes,
)
from places_attr_conflation.replay import FetchedPage, FinalDecision, ReplayEpisode, SearchAttempt


class HarnessTests(unittest.TestCase):
    def _episode(self) -> ReplayEpisode:
        return ReplayEpisode(
            case_id="1",
            attribute="website",
            place={"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA"},
            gold_value="https://example.com",
            search_attempts=[
                SearchAttempt(
                    layer="official",
                    query='"Cafe Rio" official website',
                    fetched_pages=[
                        FetchedPage(
                            url="https://example.com",
                            title="Cafe Rio",
                            page_text="Contact us",
                            source_type="official_site",
                            extracted_values={"website": "https://example.com"},
                            recency_days=10,
                        ),
                        FetchedPage(
                            url="https://yelp.com/biz/example",
                            title="Cafe Rio",
                            page_text="Reviews",
                            source_type="aggregator",
                            extracted_values={"website": "https://yelp.com/biz/example"},
                            recency_days=400,
                            zombie_score=0.9,
                        ),
                    ],
                ),
                SearchAttempt(
                    layer="fallback",
                    query="Cafe Rio Santa Cruz CA",
                    fetched_pages=[
                        FetchedPage(
                            url="https://yelp.com/biz/example",
                            title="Cafe Rio",
                            page_text="Reviews",
                            source_type="aggregator",
                            extracted_values={"website": "https://yelp.com/biz/example"},
                            recency_days=400,
                            zombie_score=0.9,
                        ),
                    ],
                ),
            ],
            final_decision=FinalDecision(
                attribute="website",
                decision="https://example.com",
                confidence=0.9,
                reason="official site",
                selected_url="https://example.com",
                selected_source_type="official_site",
            ),
        )

    def test_retrieval_episode_round_trip(self):
        episode = self._episode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            dump_retrieval_episodes([episode], path)
            loaded = load_retrieval_episodes(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].case_id, "1")
        self.assertEqual(loaded[0].search_attempts[0].fetched_pages[0].url, "https://example.com")

    def test_legacy_list_replay_shape_still_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            path.write_text(
                """
                [
                  {
                    "case_id": "legacy",
                    "attribute": "website",
                    "place": {"name": "Cafe Rio"},
                    "gold_value": "https://example.com",
                    "attempts": [
                      {
                        "layer": "official",
                        "query": "Cafe Rio official",
                        "results": [
                          {"url": "https://example.com", "title": "Cafe Rio", "snippet": "Contact us"}
                        ]
                      }
                    ]
                  }
                ]
                """,
                encoding="utf-8",
            )
            loaded = load_retrieval_episodes(path)
        self.assertEqual(loaded[0].case_id, "legacy")
        self.assertEqual(loaded[0].search_attempts[0].fetched_pages[0].url, "https://example.com")

    def test_targeted_arm_beats_fallback_arm(self):
        episode = self._episode()
        targeted = evaluate_retrieval_episodes([episode], arm="targeted")
        fallback = evaluate_retrieval_episodes([episode], arm="fallback")
        self.assertGreater(targeted["authoritative_found_rate"], fallback["authoritative_found_rate"])
        self.assertGreater(targeted["top1_authoritative_rate"], fallback["top1_authoritative_rate"])

    def test_compare_arms_returns_all_modes(self):
        episode = self._episode()
        report = compare_arms([episode])
        self.assertIn("targeted", report)
        self.assertIn("fallback", report)
        self.assertIn("all", report)

    def test_final_decision_metrics_track_abstention(self):
        episode = self._episode()
        report = evaluate_final_decisions([episode])
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["abstained"], 0)
        self.assertEqual(report["accuracy"], 1.0)

    def test_reranker_can_train_on_replay_labels(self):
        report = compare_reranker_on_replay([self._episode()])
        self.assertIn("available", report)
        self.assertTrue(report["available"])

    def test_harness_report_bundles_baseline_and_replay(self):
        episode = self._episode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "replay.json"
            dump_retrieval_episodes([episode], path)
            report = evaluate_harness_report(retrieval_path=path, retrieval_arm="targeted")
        self.assertIn("replay", report)
        self.assertIn("retrieval", report)
        self.assertGreater(report["replay"]["selected"]["authoritative_found_rate"], 0.0)

    def test_dork_audit_gate_blocks_weak_operator_plans(self):
        weak = {
            "summary": {
                "operator_coverage": 0.1,
                "quoted_anchor_coverage": 0.1,
                "site_restricted_coverage": 0.0,
                "authority_coverage": 0.1,
                "fallback_share": 0.9,
            }
        }
        report = evaluate_dork_audit_gate(weak)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["site_restricted_coverage"])

    def test_retrieval_quality_gate_compares_targeted_to_fallback(self):
        retrieval = compare_arms([self._episode()])
        decisions = evaluate_final_decisions([self._episode()])
        report = evaluate_retrieval_quality_gate(retrieval, decisions)
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["citation_precision_not_worse"])

    def test_ranker_dataset_rows_label_supporting_gold(self):
        rows = build_ranker_dataset_rows([self._episode()], arm="all")
        self.assertGreater(len(rows), 0)
        self.assertGreater(sum(row["is_supporting_gold"] for row in rows), 0)
        self.assertIn("source_url", rows[0])

    def test_resolver_replay_ignores_blank_nonofficial_website_urls(self):
        episode = ReplayEpisode(
            case_id="blank-registry",
            attribute="website",
            place={"current_value": "https://old.example", "base_value": "https://example.com/contact"},
            gold_value="https://example.com/contact",
            search_attempts=[
                SearchAttempt(
                    layer="official",
                    query="example official website",
                    fetched_pages=[
                        FetchedPage(
                            url="https://example.com/contact",
                            title="Example contact",
                            page_text="Official contact page",
                            source_type="official_site",
                            extracted_values={"website": "https://example.com/contact"},
                        )
                    ],
                ),
                SearchAttempt(
                    layer="fallback",
                    query="example business registry",
                    fetched_pages=[
                        FetchedPage(
                            url="https://find-and-update.company-information.service.gov.uk/company/00000000",
                            title="Example Ltd",
                            page_text="Registry page without a website field",
                            source_type="government",
                            extracted_values={},
                        )
                    ],
                ),
            ],
        )

        report = evaluate_resolver_on_replay([episode])

        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["abstention_rate"], 0.0)
        self.assertEqual(report["high_confidence_wrong_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
