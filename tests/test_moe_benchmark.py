import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.moe_benchmark import (
    compare_evaluations,
    encode_completion_pair,
    exact_paired_binary_pvalue,
    extract_python_code,
    is_syntactically_valid_python,
    load_benchmark,
    multiple_choice_variants,
    normalize_numeric_answer,
    write_json,
)
from evaluation.prepare_independent_moe_benchmark import (
    gsm8k_final_answer,
    make_mmlu_prompt,
    prepare_elementary_math,
)
from evaluation.domain_expert_benchmark import (
    compare_python_syntax,
    compare_runtime_results,
    select_domain_rows,
    subset_evaluation,
    theoretical_single_expert_compute,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = [ord(character) % 31 + 2 for character in text]
        return ([1] if add_special_tokens else []) + ids


class MoeBenchmarkTests(unittest.TestCase):
    def test_standalone_expert_compute_matches_75_percent_width(self):
        result = theoretical_single_expert_compute(640, 2048, 1536, 18)
        self.assertEqual(result["dense_layer_mac_per_token"], 3_932_160)
        self.assertEqual(result["expert_layer_mac_per_token"], 2_949_120)
        self.assertAlmostEqual(result["modified_layer_mac_reduction_fraction"], 0.25)
        self.assertAlmostEqual(result["all_mlp_mac_reduction_fraction"], 0.25 / 18)
        self.assertEqual(result["router_mac_per_token"], 0)

    def test_domain_selection_and_evaluation_subset(self):
        rows = [
            {"benchmark_id": "a", "domain_id": 0},
            {"benchmark_id": "b", "domain_id": 1},
        ]
        self.assertEqual(select_domain_rows(rows, [1]), [rows[1]])
        evaluation = {
            "model_variant": "base",
            "config": {"x": 1},
            "domains": {"0": {"n_samples": 1}, "1": {"n_samples": 1}},
            "per_example": rows,
        }
        subset = subset_evaluation(evaluation, [1])
        self.assertEqual(list(subset["domains"]), ["1"])
        self.assertEqual(subset["per_example"], [rows[1]])

    def test_standalone_runtime_comparison_requires_matched_tokens(self):
        def runtime(seconds, tokens=10, memory=100):
            return {
                "results": [{
                    "batch_size": 1,
                    "valid_tokens": tokens,
                    "median_seconds": seconds,
                    "tokens_per_second": tokens / seconds,
                    "cuda_baseline_allocated_bytes": memory,
                    "cuda_peak_allocated_bytes": memory + 20,
                }]
            }
        row = compare_runtime_results(runtime(2.0), runtime(1.0, memory=90))[0]
        self.assertEqual(row["speedup_base_over_expert"], 2.0)
        self.assertEqual(row["expert_latency_change_fraction"], -0.5)
        self.assertEqual(row["cuda_baseline_allocated_delta_bytes"], -10)
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            compare_runtime_results(runtime(2.0), runtime(1.0, tokens=9))

    def test_python_syntax_comparison_is_paired(self):
        def generation(values):
            return {
                "per_example": [
                    {"benchmark_id": str(i), "python_syntax_valid": value}
                    for i, value in enumerate(values)
                ]
            }
        result = compare_python_syntax(
            generation([True, True, False]), generation([True, False, True])
        )
        self.assertEqual(result["base_to_expert_valid"], 1)
        self.assertEqual(result["base_to_expert_invalid"], 1)
        self.assertEqual(result["exact_paired_pvalue"], 1.0)

    def test_completion_encoding_preserves_answer_when_prompt_is_truncated(self):
        encoded = encode_completion_pair(TinyTokenizer(), "abcdefghij", "XY", max_length=8)
        self.assertEqual(encoded["completion_tokens"], 2)
        self.assertEqual(encoded["prompt_tokens"], 6)
        self.assertTrue(encoded["truncated"])
        self.assertEqual(len(encoded["input_ids"]), 8)

    def test_numeric_and_python_output_normalization(self):
        self.assertEqual(normalize_numeric_answer("Reasoning... final: 1,234.0"), "1234")
        self.assertIsNone(normalize_numeric_answer("no numeric answer"))
        self.assertEqual(
            extract_python_code("```python\ndef answer():\n    return 1\n```"),
            "def answer():\n    return 1",
        )

    def test_python_extractor_handles_prompt_opening_fence_completion(self):
        generated = (
            "def add(a, b):\n    return a + b\n```\n"
            "This implementation adds two values.\n```python\n# later example\n```"
        )
        self.assertEqual(extract_python_code(generated), "def add(a, b):\n    return a + b")
        self.assertTrue(is_syntactically_valid_python(extract_python_code(generated)))
        self.assertFalse(is_syntactically_valid_python(""))
        self.assertFalse(is_syntactically_valid_python("```"))
        commented = "# helper\ndef value():\n    return 2\n```\nignored"
        self.assertEqual(
            extract_python_code(commented), "# helper\ndef value():\n    return 2"
        )

    def test_multiple_choice_order_runs_are_deterministic_and_semantic(self):
        row = {
            "benchmark_id": "mc-1",
            "prompt": "old",
            "choices": [" A", " B", " C", " D"],
            "answer_index": 2,
            "mc_instruction": "Choose.",
            "mc_stem_label": "Question",
            "mc_stem": "Which?",
            "mc_options": ["zero", "one", "two", "three"],
        }
        first = multiple_choice_variants(row, n_runs=4, seed=7)
        second = multiple_choice_variants(row, n_runs=4, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["option_order"], [0, 1, 2, 3])
        self.assertEqual(len({tuple(run["option_order"]) for run in first}), 4)
        for run in first:
            self.assertEqual(run["option_order"][run["answer_index"]], 2)
            self.assertTrue(run["prompt"].endswith("Answer:"))

    def test_binary_order_runs_are_capped_at_two(self):
        row = {
            "benchmark_id": "mc-2",
            "prompt": "old",
            "choices": [" A", " B"],
            "answer_index": 0,
            "mc_stem": "Statement",
            "mc_options": ["yes", "no"],
        }
        self.assertEqual(len(multiple_choice_variants(row, n_runs=8, seed=3)), 2)

    def test_elementary_math_is_balanced_unique_and_valid_latex(self):
        rows = prepare_elementary_math(12, seed=123)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["benchmark_id"] for row in rows}), 12)
        counts = {
            operation: sum(row["source_config"] == operation for row in rows)
            for operation in ("addition", "subtraction", "multiplication", "division")
        }
        self.assertEqual(set(counts.values()), {3})
        latex_rows = [
            row for row in rows if row["source_config"] in {"multiplication", "division"}
        ]
        self.assertTrue(all("\\\\" not in row["mc_stem"] for row in latex_rows))
        excluded = {rows[0]["benchmark_id"]}
        replacement = prepare_elementary_math(12, seed=123, excluded_ids=excluded)
        self.assertFalse(excluded & {row["benchmark_id"] for row in replacement})

    def test_exact_paired_accuracy_test(self):
        self.assertEqual(exact_paired_binary_pvalue([True, False], [True, False]), 1.0)
        self.assertAlmostEqual(
            exact_paired_binary_pvalue([False] * 8, [True] * 8),
            2.0 / (2**8),
        )

    def test_source_format_helpers(self):
        self.assertEqual(gsm8k_final_answer("work\n#### -12"), "-12")
        prompt = make_mmlu_prompt("Q?", ["one", "two", "three", "four"])
        self.assertIn("A. one", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_benchmark_loader_rejects_duplicate_ids(self):
        row = {
            "benchmark_id": "same",
            "domain_id": 0,
            "domain_name": "d",
            "task_type": "multiple_choice",
            "prompt": "p",
            "choices": [" a", " b"],
            "reference_completion": " a",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not unique"):
                load_benchmark(path)

    def test_paired_comparison_uses_item_level_differences(self):
        def evaluation(name, nll, correct):
            return {
                "model_variant": name,
                "per_example": [
                    {
                        "benchmark_id": f"x{i}",
                        "domain_id": 0,
                        "domain_name": "d",
                        "task_type": "multiple_choice",
                        "gold_mean_nll": value,
                        "correct": correctness,
                        "mc_variant_accuracy": float(correctness),
                    }
                    for i, (value, correctness) in enumerate(zip(nll, correct))
                ],
            }

        base = evaluation("base", [2.0, 4.0], [True, False])
        moe = evaluation("moe", [1.0, 5.0], [True, True])
        result = compare_evaluations(base, moe, n_bootstrap=20, seed=1)
        self.assertAlmostEqual(result["domains"]["0"]["mean_delta_gold_nll_moe_minus_base"], 0.0)
        self.assertAlmostEqual(result["domains"]["0"]["accuracy_delta_moe_minus_base"], 0.5)
        self.assertTrue(np.isfinite(result["domains"]["0"]["delta_gold_nll_ci95"]).all())

    def test_json_writer_rejects_nan_without_leaving_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaisesRegex(ValueError, r"\$\.domains\[0\]\.nll"):
                write_json(path, {"domains": [{"nll": float("nan")}]})
            self.assertFalse(path.exists())
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())


if __name__ == "__main__":
    unittest.main()
