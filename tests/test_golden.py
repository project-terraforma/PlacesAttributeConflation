import csv
import tempfile
import unittest
from pathlib import Path

import duckdb

from places_attr_conflation.golden import (
    build_project_a_agreement_labels,
    build_project_a_conflict_review_rows,
    build_project_a_evaluation_rows,
    build_project_a_labels_from_david_finalized,
    build_project_a_labels_from_james_golden,
    evaluate_project_a_golden,
    load_label_rows,
    write_conflict_csv,
    write_label_csv,
)


def _write_project_a_parquet(path: Path) -> None:
    duckdb.query(
        f"""
        COPY (
          SELECT
            '1' AS id,
            'b1' AS base_id,
            '[{{}}]' AS sources,
            '{{"primary":"Cafe"}}' AS names,
            '{{"primary":"bakery"}}' AS categories,
            0.9 AS confidence,
            '["https://example.com"]' AS websites,
            NULL AS socials,
            NULL AS emails,
            '["+18315551212"]' AS phones,
            NULL AS brand,
            '[{{"freeform":"1 Main St"}}]' AS addresses,
            '[{{}}]' AS base_sources,
            '{{"primary":"Cafe"}}' AS base_names,
            '{{"primary":"coffee_shop"}}' AS base_categories,
            0.7 AS base_confidence,
            '["https://base.example.com"]' AS base_websites,
            NULL AS base_socials,
            NULL AS base_emails,
            '["8315551212"]' AS base_phones,
            NULL AS base_brand,
            '[{{"freeform":"1 Main Street"}}]' AS base_addresses
          UNION ALL
          SELECT
            '2' AS id,
            'b2' AS base_id,
            '[{{}}]' AS sources,
            '{{"primary":"Shop Current"}}' AS names,
            '{{"primary":"retail"}}' AS categories,
            0.6 AS confidence,
            '["https://current.example.com"]' AS websites,
            NULL AS socials,
            NULL AS emails,
            '["+14085550000"]' AS phones,
            NULL AS brand,
            '[{{"freeform":"2 Main St"}}]' AS addresses,
            '[{{}}]' AS base_sources,
            '{{"primary":"Shop Base"}}' AS base_names,
            '{{"primary":"retail"}}' AS base_categories,
            0.95 AS base_confidence,
            '["https://base.example.com"]' AS base_websites,
            NULL AS base_socials,
            NULL AS base_emails,
            '["4085550000"]' AS base_phones,
            NULL AS base_brand,
            '[{{"freeform":"2 Main St"}}]' AS base_addresses
        ) TO '{path.as_posix()}' (FORMAT PARQUET)
        """
    )


class GoldenTests(unittest.TestCase):
    def test_project_a_golden_scores_normalized_pair_baselines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            labels = root / "labels.csv"
            _write_project_a_parquet(parquet)
            labels.write_text(
                "\n".join(
                    [
                        "id,base_id,website_truth_choice,website_truth_value,phone_truth_choice,phone_truth_value,address_truth_choice,address_truth_value,category_truth_choice,category_truth_value,name_truth_choice,name_truth_value",
                        "1,b1,current,,same,,same,,current,,same,",
                        "2,b2,base,,same,,same,,same,,base,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate_project_a_golden(parquet, labels, baselines=["current", "base", "hybrid", "smart_hybrid", "agreement_only"])

        self.assertEqual(report["label_rows"], 2)
        self.assertEqual(report["baselines"]["current"]["metrics"]["phone"]["accuracy"], 1.0)
        self.assertEqual(report["baselines"]["base"]["metrics"]["phone"]["accuracy"], 1.0)
        self.assertLess(report["baselines"]["current"]["metrics"]["website"]["accuracy"], 1.0)
        self.assertGreaterEqual(report["baselines"]["hybrid"]["metrics"]["website"]["accuracy"], 0.5)
        self.assertGreaterEqual(report["baselines"]["smart_hybrid"]["metrics"]["website"]["accuracy"], 0.5)
        self.assertEqual(report["baselines"]["current"]["conflict_metrics"]["website"]["total"], 2)
        self.assertLess(report["baselines"]["current"]["conflict_metrics"]["website"]["accuracy"], 1.0)
        self.assertEqual(report["baselines"]["agreement_only"]["conflict_metrics"]["website"]["abstention_rate"], 1.0)
        self.assertEqual(report["baselines"]["agreement_only"]["conflict_metrics"]["website"]["high_confidence_wrong_rate"], 0.0)

    def test_conflict_review_rows_export_attribute_level_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            labels = root / "labels.csv"
            output = root / "conflicts.csv"
            _write_project_a_parquet(parquet)
            labels.write_text(
                "\n".join(
                    [
                        "id,base_id,website_truth_choice,website_truth_value,phone_truth_choice,phone_truth_value,address_truth_choice,address_truth_value,category_truth_choice,category_truth_value,name_truth_choice,name_truth_value",
                        "1,b1,current,,same,,same,,current,,same,",
                        "2,b2,base,,same,,same,,same,,base,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_project_a_conflict_review_rows(parquet, labels, baseline="hybrid")
            csv_path = write_conflict_csv(rows, output)

            self.assertTrue(csv_path.exists())
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["needs_evidence"], True)
            self.assertIn("current_value", csv_path.read_text(encoding="utf-8"))

    def test_agreement_labels_seed_repeatable_same_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            labels = root / "agreement.csv"
            _write_project_a_parquet(parquet)

            rows = build_project_a_agreement_labels(parquet, limit=10, min_attributes=2)
            write_label_csv(rows, labels)
            report = evaluate_project_a_golden(parquet, labels, baselines=["hybrid"])

        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["phone_truth_choice"], "same")
        self.assertEqual(report["baselines"]["hybrid"]["metrics"]["phone"]["accuracy"], 1.0)

    def test_james_golden_import_maps_values_to_choices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            james = root / "james.csv"
            _write_project_a_parquet(parquet)
            james.write_text(
                "\n".join(
                    [
                        "sample_idx,names,addresses,categories,websites,phones,emails,socials,brand",
                        """0,"{'primary':'Cafe'}","[{'freeform':'1 Main St'}]","{'primary':'bakery'}","['https://example.com']","['+18315551212']",,,"{'names':{}}\"""",
                        """1,"{'primary':'Shop Base'}","[{'freeform':'2 Main St'}]","{'primary':'retail'}","['https://base.example.com']","['4085550000']",,,"{'names':{}}\"""",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_project_a_labels_from_james_golden(parquet, james)
            labels = root / "labels.csv"
            write_label_csv(rows, labels)
            report = evaluate_project_a_golden(parquet, labels, baselines=["hybrid"])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["website_truth_choice"], "current")
        self.assertEqual(rows[0]["phone_truth_choice"], "same")
        self.assertEqual(rows[1]["name_truth_choice"], "base")
        self.assertEqual(report["baselines"]["hybrid"]["metrics"]["website"]["total"], 2)

    def test_david_final_labels_import_attribute_level_choices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            labels = root / "labels.csv"
            _write_project_a_parquet(parquet)
            david = Path(__file__).resolve().parent / "fixtures" / "david_final_labels_sample.csv"

            rows = build_project_a_labels_from_david_finalized(david, split_name="fixture")
            write_label_csv(rows, labels)
            report = evaluate_project_a_golden(parquet, labels, baselines=["hybrid", "agreement_only"])

        self.assertEqual(len(rows), 2)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["1"]["website_truth_choice"], "current")
        self.assertEqual(by_id["1"]["phone_truth_choice"], "same")
        self.assertEqual(by_id["2"]["website_truth_choice"], "base")
        self.assertEqual(by_id["2"]["name_truth_choice"], "base")
        self.assertEqual(by_id["2"]["category_truth_choice"], "same")
        self.assertEqual(by_id["2"]["address_truth_choice"], "")
        self.assertEqual(report["baselines"]["hybrid"]["metrics"]["website"]["total"], 2)
        self.assertIn("precision", report["baselines"]["hybrid"]["metrics"]["website"])
        self.assertIn("recall", report["baselines"]["hybrid"]["metrics"]["website"])
        self.assertIn("f1", report["baselines"]["hybrid"]["metrics"]["website"])

    def test_build_evaluation_rows_keeps_truth_source_for_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            labels = root / "labels.csv"
            _write_project_a_parquet(parquet)
            with labels.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["id", "base_id", "website_truth_choice", "website_truth_value"],
                )
                writer.writeheader()
                writer.writerow({"id": "1", "base_id": "b1", "website_truth_choice": "", "website_truth_value": "https://example.com"})

            rows = build_project_a_evaluation_rows(parquet, labels, "current")

            self.assertEqual(len(load_label_rows(labels)), 1)
        self.assertEqual(rows[0]["website_truth_source"], "explicit")
        self.assertEqual(rows[0]["website_prediction"], "https://example.com")
        self.assertTrue(rows[0]["website_pair_differs"])

    def test_smart_hybrid_prefers_official_website_and_name_over_higher_confidence_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parquet = root / "project_a_samples.parquet"
            labels = root / "labels.csv"
            _write_project_a_parquet(parquet)
            labels.write_text(
                "\n".join(
                    [
                        "id,base_id,website_truth_choice,website_truth_value,name_truth_choice,name_truth_value",
                        "2,b2,current,,current,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = evaluate_project_a_golden(parquet, labels, baselines=["hybrid", "smart_hybrid"])

        self.assertEqual(report["baselines"]["hybrid"]["metrics"]["website"]["accuracy"], 0.0)
        self.assertEqual(report["baselines"]["smart_hybrid"]["metrics"]["website"]["accuracy"], 1.0)
        self.assertEqual(report["baselines"]["hybrid"]["metrics"]["name"]["accuracy"], 0.0)
        self.assertEqual(report["baselines"]["smart_hybrid"]["metrics"]["name"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
