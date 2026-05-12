import unittest

from places_attr_conflation.lab_research import ExperimentResult, ExperimentSpec, ResearchSourceRecord, serialize_record


class LabResearchTests(unittest.TestCase):
    def test_research_source_serialization(self):
        record = ResearchSourceRecord(
            source_id="paper-1",
            source_type="paper",
            title="A Better Reranker",
            reference="https://example.com/paper",
            claims=("reranking improves precision",),
            metrics=("precision", "recall"),
        )
        payload = serialize_record(record)
        self.assertEqual(payload["source_id"], "paper-1")
        self.assertEqual(payload["claims"], ("reranking improves precision",))

    def test_experiment_result_keeps_promotion_decision(self):
        spec = ExperimentSpec(
            experiment_id="exp-1",
            title="Add margin abstention",
            source_refs=("paper-1",),
            claim_under_test="margin calibration reduces high-confidence wrong",
            implementation_scope="resolver abstention path",
            target_modules=("resolver.py",),
            expected_metric_change="lower hc wrong",
            datasets_or_fixtures=("tests/fixtures/retrieval_replay_sample.json",),
            baseline_name="heuristic",
            eval_command="python3 scripts/run_harness.py replay --input fixture.json",
            success_criteria=("hc wrong decreases",),
            failure_criteria=("coverage collapses",),
        )
        result = ExperimentResult(
            experiment_id=spec.experiment_id,
            baseline_metrics={"accuracy": 0.5},
            experiment_metrics={"accuracy": 0.6},
            promotion_decision="promote",
            decision_reason="beat baseline without safety regression",
        )
        payload = serialize_record(result)
        self.assertEqual(payload["promotion_decision"], "promote")
        self.assertEqual(payload["baseline_metrics"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
