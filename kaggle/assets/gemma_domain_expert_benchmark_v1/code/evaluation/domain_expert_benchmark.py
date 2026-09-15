"""Utilities for evaluating standalone domain experts against a base model."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from evaluation.moe_benchmark import compare_evaluations, exact_paired_binary_pvalue


def benchmark_domain_ids(rows: Sequence[Dict[str, Any]]) -> list[int]:
    domain_ids = sorted({int(row["domain_id"]) for row in rows})
    if not domain_ids:
        raise ValueError("Benchmark contains no domains")
    return domain_ids


def select_domain_rows(
    rows: Sequence[Dict[str, Any]], domain_ids: Iterable[int]
) -> list[Dict[str, Any]]:
    selected_ids = {int(domain_id) for domain_id in domain_ids}
    selected = [row for row in rows if int(row["domain_id"]) in selected_ids]
    present = {int(row["domain_id"]) for row in selected}
    missing = selected_ids - present
    if missing:
        raise ValueError(f"Benchmark has no rows for domains: {sorted(missing)}")
    return selected


def subset_evaluation(
    evaluation: Dict[str, Any], domain_ids: Iterable[int]
) -> Dict[str, Any]:
    """Select domains from an evaluation without recomputing the base model."""
    selected_ids = {int(domain_id) for domain_id in domain_ids}
    per_example = [
        row
        for row in evaluation["per_example"]
        if int(row["domain_id"]) in selected_ids
    ]
    present = {int(row["domain_id"]) for row in per_example}
    missing = selected_ids - present
    if missing:
        raise ValueError(f"Evaluation has no rows for domains: {sorted(missing)}")
    return {
        "model_variant": evaluation["model_variant"],
        "config": dict(evaluation.get("config", {})),
        "domains": {
            str(domain_id): evaluation["domains"][str(domain_id)]
            for domain_id in sorted(selected_ids)
        },
        "per_example": per_example,
    }


def compare_standalone_expert(
    base_evaluation: Dict[str, Any],
    expert_evaluation: Dict[str, Any],
    expert_domain_id: int,
    n_bootstrap: int = 2000,
    seed: int = 20260915,
) -> Dict[str, Any]:
    """Make a paired base comparison and record the expert's intended domain."""
    result = compare_evaluations(
        base_evaluation,
        expert_evaluation,
        n_bootstrap=n_bootstrap,
        seed=seed + 1009 * int(expert_domain_id),
    )
    result["format"] = "gemma_standalone_domain_expert_comparison_v1"
    result["expert_domain_id"] = int(expert_domain_id)
    result["evaluation_scope_domain_ids"] = sorted(
        {int(row["domain_id"]) for row in expert_evaluation["per_example"]}
    )
    return result


def theoretical_single_expert_compute(
    d_model: int,
    dense_width: int,
    expert_width: int,
    n_mlp_layers: int,
) -> Dict[str, Any]:
    """Report transparent MAC counts for one always-active compact gated MLP."""
    if min(d_model, dense_width, expert_width, n_mlp_layers) <= 0:
        raise ValueError("Model dimensions and layer count must be positive")
    if expert_width > dense_width:
        raise ValueError("Expert width cannot exceed dense width")
    dense_mac = 3 * int(d_model) * int(dense_width)
    expert_mac = 3 * int(d_model) * int(expert_width)
    saved_fraction = 1.0 - expert_mac / dense_mac
    return {
        "format": "gemma_standalone_expert_compute_v1",
        "d_model": int(d_model),
        "dense_width": int(dense_width),
        "expert_width": int(expert_width),
        "n_mlp_layers": int(n_mlp_layers),
        "dense_layer_mac_per_token": dense_mac,
        "expert_layer_mac_per_token": expert_mac,
        "modified_layer_mac_reduction_fraction": saved_fraction,
        "all_mlp_mac_reduction_fraction": saved_fraction / int(n_mlp_layers),
        "router_mac_per_token": 0,
        "scope_note": (
            "Counts the three linear projections of a gated MLP. It excludes "
            "attention, normalization, embeddings, LM head, memory traffic, and kernels."
        ),
    }


def compare_runtime_results(
    base_runtime: Dict[str, Any], expert_runtime: Dict[str, Any]
) -> list[Dict[str, Any]]:
    """Join matched prefill measurements without claiming statistical significance."""
    base_rows = {int(row["batch_size"]): row for row in base_runtime["results"]}
    expert_rows = {int(row["batch_size"]): row for row in expert_runtime["results"]}
    if base_rows.keys() != expert_rows.keys():
        raise ValueError("Base and expert runtime results use different batch sizes")
    result = []
    for batch_size in sorted(base_rows):
        base = base_rows[batch_size]
        expert = expert_rows[batch_size]
        if int(base["valid_tokens"]) != int(expert["valid_tokens"]):
            raise ValueError(f"Runtime token mismatch for batch size {batch_size}")
        base_seconds = float(base["median_seconds"])
        expert_seconds = float(expert["median_seconds"])
        result.append(
            {
                "batch_size": batch_size,
                "valid_tokens": int(base["valid_tokens"]),
                "base_median_seconds": base_seconds,
                "expert_median_seconds": expert_seconds,
                "speedup_base_over_expert": base_seconds / expert_seconds,
                "expert_latency_change_fraction": expert_seconds / base_seconds - 1.0,
                "base_tokens_per_second": float(base["tokens_per_second"]),
                "expert_tokens_per_second": float(expert["tokens_per_second"]),
                "cuda_baseline_allocated_delta_bytes": _optional_difference(
                    expert.get("cuda_baseline_allocated_bytes"),
                    base.get("cuda_baseline_allocated_bytes"),
                ),
                "cuda_peak_allocated_delta_bytes": _optional_difference(
                    expert.get("cuda_peak_allocated_bytes"),
                    base.get("cuda_peak_allocated_bytes"),
                ),
                "warning": (
                    "Matched implementation benchmark; raw repeat timings are not retained, "
                    "so no latency confidence interval is available."
                ),
            }
        )
    return result


def compare_python_syntax(
    base_generation: Dict[str, Any], expert_generation: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare syntax validity on paired generations without executing code."""
    base = {row["benchmark_id"]: row for row in base_generation["per_example"]}
    expert = {row["benchmark_id"]: row for row in expert_generation["per_example"]}
    if base.keys() != expert.keys():
        raise ValueError("Base and expert generations contain different benchmark IDs")
    ids = sorted(base)
    base_valid = [bool(base[key]["python_syntax_valid"]) for key in ids]
    expert_valid = [bool(expert[key]["python_syntax_valid"]) for key in ids]
    base_to_expert_valid = sum(
        not left and right for left, right in zip(base_valid, expert_valid)
    )
    base_to_expert_invalid = sum(
        left and not right for left, right in zip(base_valid, expert_valid)
    )
    return {
        "n_samples": len(ids),
        "base_python_syntax_valid": sum(base_valid) / len(ids),
        "expert_python_syntax_valid": sum(expert_valid) / len(ids),
        "base_to_expert_valid": int(base_to_expert_valid),
        "base_to_expert_invalid": int(base_to_expert_invalid),
        "exact_paired_pvalue": exact_paired_binary_pvalue(base_valid, expert_valid),
        "warning": "Syntax validity is not functional pass@1; generated code was not executed.",
    }


def _optional_difference(left: Any, right: Any) -> int | None:
    if left is None or right is None:
        return None
    return int(left) - int(right)


def summarize_own_domain_comparisons(
    comparisons: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    rows = []
    for comparison in comparisons:
        domain_id = int(comparison["expert_domain_id"])
        domain = comparison["domains"][str(domain_id)]
        rows.append(
            {
                "domain_id": domain_id,
                "domain_name": domain["domain_name"],
                "n_samples": int(domain["n_samples"]),
                "mean_delta_gold_nll_expert_minus_base": float(
                    domain["mean_delta_gold_nll_moe_minus_base"]
                ),
                "delta_gold_nll_ci95": list(domain["delta_gold_nll_ci95"]),
                "base_multiple_choice_accuracy": domain["base_multiple_choice_accuracy"],
                "expert_multiple_choice_accuracy": domain["moe_multiple_choice_accuracy"],
                "accuracy_delta_expert_minus_base": domain["accuracy_delta_moe_minus_base"],
                "exact_paired_accuracy_pvalue": domain["exact_paired_accuracy_pvalue"],
                "base_variant_accuracy": domain["base_multiple_choice_variant_accuracy"],
                "expert_variant_accuracy": domain["moe_multiple_choice_variant_accuracy"],
                "variant_accuracy_delta_expert_minus_base": domain[
                    "variant_accuracy_delta_moe_minus_base"
                ],
            }
        )
    return sorted(rows, key=lambda row: row["domain_id"])
