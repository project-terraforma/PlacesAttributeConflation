import json
import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.dashboard import build_dashboard_data, render_html, render_markdown, write_dashboard


class DashboardTests(unittest.TestCase):
    def test_dashboard_discovers_latest_reports_and_renders_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "reports"
            harness = root / "harness"
            baseline = root / "baseline_metrics"
            harness.mkdir(parents=True)
            baseline.mkdir(parents=True)

            (baseline / "resolvepoi_hybrid_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "website": {
                                "accuracy": 0.36,
                                "macro_f1": 0.18,
                                "high_confidence_wrong_rate": 0.64,
                                "abstention_rate": 0.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (harness / "compare_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "targeted": {
                            "authoritative_found_rate": 0.75,
                            "useful_found_rate": 1.0,
                            "citation_precision": 0.75,
                            "top1_authoritative_rate": 0.75,
                            "average_search_attempts": 1.0,
                            "total": 4,
                        },
                        "fallback": {
                            "authoritative_found_rate": 0.0,
                            "useful_found_rate": 0.0,
                            "citation_precision": 0.0,
                            "top1_authoritative_rate": 0.0,
                            "average_search_attempts": 1.0,
                            "total": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (harness / "rerank_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "available": True,
                        "training_examples": 9,
                        "positive_examples": 3,
                        "negative_examples": 6,
                        "heuristic": {"top1_authoritative_rate": 0.75},
                        "reranker": {"top1_authoritative_rate": 0.75},
                        "improved_top1_authoritative_rate": False,
                    }
                ),
                encoding="utf-8",
            )
            (harness / "all_20260424_010000.json").write_text(
                json.dumps(
                    {
                        "decisions": {
                            "accuracy": 0.5,
                            "abstention_rate": 0.25,
                            "high_confidence_wrong_rate": 0.25,
                            "total": 4,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (harness / "smoke_20260424_010000.json").write_text(
                json.dumps({"mode": "replay", "results": [{"status": "error"}, {"status": "ok"}]}),
                encoding="utf-8",
            )

            data = build_dashboard_data(root)
            markdown = render_markdown(data)
            outputs = write_dashboard(root, root / "dashboard")

            self.assertIn("What Is Stopping Us", markdown)
            self.assertIn("At a Glance", markdown)
            self.assertIn("Targeted search is ahead", markdown)
            self.assertIn("ResolvePOI Baseline", markdown)
            self.assertIn("Retrieval Arms", markdown)
            self.assertTrue(Path(outputs["markdown"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())
            self.assertTrue(Path(outputs["latest"]).exists())
            html = Path(outputs["html"]).read_text(encoding="utf-8")
            self.assertIn("Benchmark Viewer", html)
            self.assertIn("data-view='baseline'", html)
            self.assertIn("At a Glance", html)

    def test_dashboard_html_renders_when_reports_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = build_dashboard_data(Path(tmpdir) / "reports")

            html = render_html(data)

            self.assertIn("Benchmark Viewer", html)
            self.assertIn("<td>missing</td>", html)

    def test_dashboard_counts_nested_replay_collected_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "reports"
            batch_dir = root / "ranker" / "conflict_dorks_test_batches"
            replay_dir = root / "replay_collected" / "evidence_batch"
            batch_dir.mkdir(parents=True)
            replay_dir.mkdir(parents=True)
            batch_path = batch_dir / "batch_001.csv"
            batch_path.write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value,preferred_sources,layer,query,priority\n"
                "case-1,base-1,website,https://example.com,base,https://old.example,hybrid,False,True,https://old.example,https://example.com,official,official,query,high\n",
                encoding="utf-8",
            )
            (batch_dir / "manifest.json").write_text(
                json.dumps({"batches": [{"batch": 1, "path": str(batch_path.relative_to(root.parent)), "cases": 1}]}),
                encoding="utf-8",
            )
            (replay_dir / "seed.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "episodes": [
                            {
                                "case_id": "case-1",
                                "attribute": "website",
                                "place": {},
                                "gold_value": "https://example.com",
                                "search_attempts": [
                                    {
                                        "layer": "official",
                                        "query": "query",
                                        "fetched_pages": [
                                            {
                                                "url": "https://example.com",
                                                "title": "Example",
                                                "page_text": "Official page",
                                                "source_type": "official_site",
                                                "extracted_values": {"website": "https://example.com"},
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (replay_dir / "copy.json").write_text((replay_dir / "seed.json").read_text(encoding="utf-8"), encoding="utf-8")

            data = build_dashboard_data(root)

            self.assertIn(["1", "1", "1", "1"], data.batch_progress_rows)


if __name__ == "__main__":
    unittest.main()
