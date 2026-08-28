from __future__ import annotations

import random
import unittest

from value_alignment.evaluation.aita_metrics import probability_gain, summarize
from value_alignment.evaluation.compare_kvs_results import compare_results
from value_alignment.prepare_aita_eval import convert_examples as convert_aita_examples
from value_alignment.prepare_kvs_dpo import convert_split as convert_kvs_split
from value_alignment.prepare_kvs_sft import make_sft_rows
from value_alignment.survey_data import extract_rating, iter_survey_variants, majority_rating
from value_alignment.value_taxonomy import BASIC_VALUES, basic_value_for_fine


def kvs_item(fine_value: str, positive: str = "Positive", negative: str = "Opposing") -> dict:
    return {
        "sentence": positive,
        "negative_sentence": negative,
        "category": "X",
        "level1": ["Goal"],
        "level2": [fine_value],
        "level3": ["Higher"],
        "level4": ["Focus"],
    }


class ValueTaxonomyTest(unittest.TestCase):
    def test_refined_values_map_to_ten_basic_values(self) -> None:
        self.assertEqual(len(BASIC_VALUES), 10)
        self.assertEqual(basic_value_for_fine("Self_direction_action"), "Self_direction")
        self.assertEqual(basic_value_for_fine("Face"), "Power")
        self.assertEqual(basic_value_for_fine("Humility"), "Tradition")
        self.assertEqual(basic_value_for_fine("Universalism_objectivity"), "Universalism")


class SurveyDataTest(unittest.TestCase):
    def test_each_description_expands_to_27_prompt_variants(self) -> None:
        tasks = ["Task A", "Task B", "Task C"]
        templates = [f"Template {index}: {{rating}}" for index in range(9)]
        rows = list(iter_survey_variants(kvs_item("Security_personal"), "test", 0, tasks, templates))

        self.assertEqual(len(rows), 27)
        self.assertEqual(len({row["id"] for row in rows}), 27)
        self.assertTrue(all(row["value"] == "Security" for row in rows))
        self.assertTrue(all("{rating}" not in row["prompt"] for row in rows))

    def test_rating_parser_and_majority_vote(self) -> None:
        self.assertEqual(extract_rating("Rating: 5"), 5)
        self.assertIsNone(extract_rating("Rating unavailable"))
        self.assertEqual(majority_rating([2, 2, 5]), 2)
        self.assertEqual(majority_rating([2, 4]), 2)


class KvsPreferenceDataTest(unittest.TestCase):
    def test_target_pair_is_reversed_and_non_target_pair_is_anchored(self) -> None:
        target = kvs_item("Security_personal", "Stay safe", "Take the risk")
        non_target = kvs_item("Benevolence_caring", "Help others", "Ignore others")
        rows = convert_kvs_split(
            [target, non_target],
            "train",
            "Security",
            max_per_value=0,
            rng=random.Random(42),
        )
        by_value = {row["value"]: row for row in rows}

        self.assertEqual(by_value["Security"]["chosen"], "Take the risk")
        self.assertEqual(by_value["Security"]["rejected"], "Stay safe")
        self.assertTrue(by_value["Security"]["is_target"])
        self.assertEqual(by_value["Benevolence"]["chosen"], "Help others")
        self.assertEqual(by_value["Benevolence"]["rejected"], "Ignore others")
        self.assertFalse(by_value["Benevolence"]["is_target"])


class KvsSftDataTest(unittest.TestCase):
    def test_target_rating_is_one_and_non_target_keeps_model_baseline(self) -> None:
        rows = make_sft_rows(
            [
                kvs_item("Security_personal", "Stay safe", "Take the risk"),
                kvs_item("Benevolence_caring", "Help others", "Ignore others"),
            ],
            "train",
            ["Task"],
            ["Rating: {rating}"],
            {"kvs-train-0000": 5, "kvs-train-0001": 4},
            target_value="Security",
            target_rating=1,
        )
        by_value = {row["value"]: row for row in rows}

        self.assertEqual(by_value["Security"]["rating"], 1)
        self.assertEqual(by_value["Security"]["response"], "1")
        self.assertEqual(by_value["Benevolence"]["rating"], 4)
        self.assertEqual(by_value["Benevolence"]["response"], "4")


class AitaEvaluationDataTest(unittest.TestCase):
    def test_prompt_is_neutral_and_refined_value_is_aggregated(self) -> None:
        raw = {
            "Security_personal": [
                {
                    "post": "I returned a lost wallet. AITA?",
                    "high_standard_stance": "NTA",
                    "low_standard_stance": "YTA",
                    "justification": {"value_assignment": "Example"},
                }
            ]
        }
        rows = convert_aita_examples(raw, ["all"], 0, random.Random(42))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "Security")
        self.assertEqual(rows[0]["fine_value"], "Security_personal")
        self.assertNotIn("Security", rows[0]["prompt"])
        self.assertNotIn("high-standard", rows[0]["prompt"].lower())
        self.assertNotIn("chosen", rows[0])

    def test_identical_high_and_low_labels_are_rejected(self) -> None:
        raw = {
            "Achievement": [
                {
                    "post": "Example",
                    "high_standard_stance": "NTA",
                    "low_standard_stance": "NTA",
                }
            ]
        }
        with self.assertRaises(ValueError):
            convert_aita_examples(raw, ["all"], 0, random.Random(42))


class AitaMetricTest(unittest.TestCase):
    def test_probability_gain_matches_paper_formula(self) -> None:
        base = {"NTA": 0.6, "Neutral": 0.1, "YTA": 0.3}
        conditioned = {"NTA": 0.4, "Neutral": 0.2, "YTA": 0.4}

        self.assertAlmostEqual(probability_gain(base, conditioned, "NTA", "YTA"), 0.35)
        self.assertAlmostEqual(
            probability_gain(base, conditioned, "NTA", "YTA", unused_stance_weight=0.0),
            0.3,
        )

    def test_summary_reports_percentage_points_and_strict_result(self) -> None:
        result = summarize(
            [
                {"value": "Security", "probability_gain": 0.1, "strict_probability_gain": 0.08},
                {"value": "Security", "probability_gain": 0.2, "strict_probability_gain": 0.12},
            ]
        )

        self.assertAlmostEqual(result["overall"]["mean_probability_gain"], 0.15)
        self.assertAlmostEqual(result["overall"]["mean_probability_gain_percentage_points"], 15.0)
        self.assertAlmostEqual(result["overall"]["strict_mean_probability_gain"], 0.1)


class KvsMetricTest(unittest.TestCase):
    def test_target_drop_and_other_value_absolute_fluctuation_are_paired(self) -> None:
        base = [
            {"id": "a", "source_id": "x", "value": "Security", "rating": 5},
            {"id": "b", "source_id": "y", "value": "Security", "rating": 4},
            {"id": "c", "source_id": "z", "value": "Benevolence", "rating": 3},
            {"id": "d", "source_id": "w", "value": "Power", "rating": 3},
        ]
        conditioned = [
            {"id": "a", "source_id": "x", "value": "Security", "rating": 2},
            {"id": "b", "source_id": "y", "value": "Security", "rating": 3},
            {"id": "c", "source_id": "z", "value": "Benevolence", "rating": 4},
            {"id": "d", "source_id": "w", "value": "Power", "rating": 1},
        ]
        result = compare_results(base, conditioned, "Security")

        self.assertAlmostEqual(result["target_value_rating_drop"], 2.0)
        self.assertAlmostEqual(result["other_values_mean_absolute_fluctuation"], 1.5)

    def test_three_stochastic_runs_are_summarized_before_aggregation(self) -> None:
        base = [
            {"id": "a", "value": "Security", "ratings": [5, 5, 5]},
            {"id": "b", "value": "Power", "ratings": [3, 3, 3]},
        ]
        conditioned = [
            {"id": "a", "value": "Security", "ratings": [4, 3, 2]},
            {"id": "b", "value": "Power", "ratings": [3, 4, 2]},
        ]
        result = compare_results(base, conditioned, "Security")

        self.assertEqual(result["num_runs"], 3)
        self.assertEqual(
            [row["target_value_rating_drop"] for row in result["runs"]],
            [1.0, 2.0, 3.0],
        )
        self.assertAlmostEqual(result["target_value_rating_drop"], 2.0)
        self.assertAlmostEqual(result["target_value_rating_drop_sample_std"], 1.0)
        self.assertAlmostEqual(result["other_values_mean_absolute_fluctuation"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
