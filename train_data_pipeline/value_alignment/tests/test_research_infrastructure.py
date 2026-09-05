from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from value_alignment.evaluation.evaluate_capabilities import extract_capability_metrics
from value_alignment.evaluation.analyze_hypo_advantage import correlation_summary
from value_alignment.evaluation.reference_mismatch_metrics import summarize_margins
from value_alignment.evaluation.statistics_utils import (
    benjamini_hochberg,
    bootstrap_mean_interval,
)
from value_alignment.experiment_utils import tagged_run_dir, write_run_manifest


class StatisticsUtilsTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_contains_the_sample_mean(self) -> None:
        first = bootstrap_mean_interval([0.0, 1.0, 2.0], replicates=500, seed=7)
        second = bootstrap_mean_interval([0.0, 1.0, 2.0], replicates=500, seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 1.0)
        self.assertGreaterEqual(first[1], 1.0)

    def test_benjamini_hochberg_matches_known_example(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002, None])

        self.assertAlmostEqual(adjusted[0], 0.02)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertAlmostEqual(adjusted[3], 0.008)
        self.assertIsNone(adjusted[4])

    def test_hypo_advantage_correlations_report_monotonic_association(self) -> None:
        rows = [
            {"mismatch": 0.1, "advantage": 1.0},
            {"mismatch": 0.2, "advantage": 2.0},
            {"mismatch": 0.3, "advantage": 3.0},
        ]

        result = correlation_summary(rows, "mismatch", "advantage")

        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["spearman_rho"], 1.0)


class RunManifestTest(unittest.TestCase):
    def test_tagged_paths_preserve_legacy_layout_and_validate_tags(self) -> None:
        base = Path("checkpoints/model/value")
        self.assertEqual(tagged_run_dir(base, ""), base)
        self.assertEqual(tagged_run_dir(base, "seed42"), base / "runs" / "seed42")
        with self.assertRaises(ValueError):
            tagged_run_dir(base, "seed 42")

    def test_manifest_records_input_hash_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            data_path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            output_dir = root / "run"

            write_run_manifest(
                output_dir,
                {"seed": 42},
                {"train_data": data_path},
                metadata={"method": "dpo"},
            )
            write_run_manifest(
                output_dir,
                {"seed": 42},
                {"train_data": data_path},
                metadata={"method": "dpo"},
                status="completed",
            )
            payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["attempt"], 1)
            self.assertEqual(payload["input_files"]["train_data"]["line_count"], 2)
            self.assertEqual(len(payload["input_files"]["train_data"]["sha256"]), 64)
            self.assertIn("completed_at_utc", payload)

            write_run_manifest(
                output_dir,
                {"seed": 43},
                {"train_data": data_path},
                metadata={"method": "dpo"},
            )
            restarted = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(restarted["attempt"], 2)
            self.assertNotIn("completed_at_utc", restarted)


class ReferenceMismatchMetricTest(unittest.TestCase):
    def test_target_and_anchor_mismatch_rates_are_separate(self) -> None:
        rows = [
            {
                "source_split": "train",
                "value": "Security",
                "is_target": True,
                "reference_margin": -2.0,
                "mean_token_reference_margin": -0.2,
                "reference_mismatch": True,
                "hypo_removed_pessimistic_bonus": 2.0,
            },
            {
                "source_split": "train",
                "value": "Security",
                "is_target": True,
                "reference_margin": 1.0,
                "mean_token_reference_margin": 0.1,
                "reference_mismatch": False,
                "hypo_removed_pessimistic_bonus": 0.0,
            },
            {
                "source_split": "train",
                "value": "Power",
                "is_target": False,
                "reference_margin": 0.5,
                "mean_token_reference_margin": 0.05,
                "reference_mismatch": False,
                "hypo_removed_pessimistic_bonus": 0.0,
            },
        ]

        summary = summarize_margins(rows)["by_split"]["train"]

        self.assertEqual(summary["target"]["reference_mismatch_rate"], 0.5)
        self.assertEqual(summary["anchors"]["reference_mismatch_rate"], 0.0)
        self.assertEqual(summary["target"]["mean_hypo_removed_pessimistic_bonus"], 1.0)


class CapabilityMetricTest(unittest.TestCase):
    def test_extracts_mmlu_and_flexible_gsm8k_metrics(self) -> None:
        payload = {
            "groups": {"mmlu": {"acc,none": 0.71}},
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.72,
                    "exact_match,flexible-extract": 0.81,
                }
            },
        }

        metrics = extract_capability_metrics(payload)

        self.assertEqual(metrics["mmlu_accuracy"], 0.71)
        self.assertEqual(metrics["gsm8k_flexible_extract_accuracy"], 0.81)
        self.assertEqual(metrics["gsm8k_strict_match_accuracy"], 0.72)


if __name__ == "__main__":
    unittest.main()
