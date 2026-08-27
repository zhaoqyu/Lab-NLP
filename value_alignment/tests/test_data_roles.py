from __future__ import annotations

import random
import unittest

from value_alignment.evaluation.aita_metrics import summarize, value_score
from value_alignment.prepare_aita_eval import convert_examples as convert_aita_examples
from value_alignment.prepare_kvs_dpo import convert_split as convert_kvs_split


class KvsTrainingDataTest(unittest.TestCase):
    def test_positive_statement_is_chosen_and_contrast_is_rejected(self) -> None:
        item = {
            "sentence": "Trying original ideas is important.",
            "negative_sentence": "I prefer never departing from familiar ideas.",
            "category": "SDT",
            "level1": ["Be creative"],
            "level2": ["Self_direction_thought"],
            "level3": ["Openness_to_change"],
            "level4": ["Personal focus"],
        }

        rows = convert_kvs_split(
            [item],
            "train",
            {"Self_direction_thought"},
            max_per_value=0,
            rng=random.Random(42),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen"], item["sentence"])
        self.assertEqual(rows[0]["rejected"], item["negative_sentence"])
        self.assertEqual(rows[0]["source_split"], "train")
        self.assertEqual(rows[0]["target_value"], "Self_direction_thought")

    def test_value_filter_excludes_other_kvs_values(self) -> None:
        base = {
            "sentence": "Positive",
            "negative_sentence": "Contrast",
            "category": "X",
            "level1": ["Goal"],
            "level3": ["Higher"],
            "level4": ["Focus"],
        }
        items = [
            {**base, "level2": ["Value_A"]},
            {**base, "level2": ["Value_B"]},
        ]

        rows = convert_kvs_split(items, "eval", {"Value_B"}, 0, random.Random(42))

        self.assertEqual([row["value"] for row in rows], ["Value_B"])


class AitaEvaluationDataTest(unittest.TestCase):
    def test_prompt_is_neutral_and_keeps_scoring_labels_as_metadata(self) -> None:
        raw = {
            "Secret_value_name": [
                {
                    "post": "I returned a lost wallet. AITA?",
                    "high_standard_stance": "NTA",
                    "low_standard_stance": "YTA",
                    "justification": {"value_assignment": "Returning it reflects the measured value."},
                }
            ]
        }

        rows = convert_aita_examples(raw, ["all"], 0, random.Random(42))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertNotIn("Secret_value_name", row["prompt"])
        self.assertNotIn("high-standard", row["prompt"].lower())
        self.assertEqual(row["high_standard_stance"], "NTA")
        self.assertEqual(row["low_standard_stance"], "YTA")
        self.assertNotIn("chosen", row)
        self.assertNotIn("rejected", row)

    def test_identical_high_and_low_labels_are_rejected(self) -> None:
        raw = {
            "Value_A": [
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
    def test_value_score_is_pairwise_high_probability(self) -> None:
        probabilities = {"NTA": 0.6, "YTA": 0.3, "Neutral": 0.1}

        self.assertAlmostEqual(value_score(probabilities, "NTA", "YTA"), 2 / 3)

    def test_summary_reports_micro_and_macro_score_change(self) -> None:
        rows = [
            {
                "value": "Value_A",
                "base_value_score": 0.4,
                "trained_value_score": 0.6,
                "value_score_change": 0.2,
                "base_correct": 0,
                "trained_correct": 1,
                "base_pairwise_correct": 0,
                "trained_pairwise_correct": 1,
                "probability_gain": 0.1,
            },
            {
                "value": "Value_B",
                "base_value_score": 0.7,
                "trained_value_score": 0.6,
                "value_score_change": -0.1,
                "base_correct": 1,
                "trained_correct": 1,
                "base_pairwise_correct": 1,
                "trained_pairwise_correct": 1,
                "probability_gain": -0.05,
            },
        ]

        result = summarize(rows)

        self.assertAlmostEqual(result["overall"]["mean_value_score_change"], 0.05)
        self.assertAlmostEqual(result["overall"]["macro_mean_value_score_change"], 0.05)


if __name__ == "__main__":
    unittest.main()
