"""Evaluation utilities for paired base-vs-MoE domain benchmarks."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


MC_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    required = {
        "benchmark_id",
        "domain_id",
        "domain_name",
        "task_type",
        "prompt",
        "choices",
        "reference_completion",
    }
    if not rows:
        raise ValueError(f"Benchmark is empty: {path}")
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Benchmark row {index} misses fields: {sorted(missing)}")
    ids = [str(row["benchmark_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark IDs are not unique")
    return rows


def find_routed_mlp(model) -> Optional[torch.nn.Module]:
    routed = [
        module
        for module in model.modules()
        if hasattr(module, "routing_stats")
        and hasattr(module, "reset_routing_stats")
        and hasattr(module, "set_routing_token_mask")
    ]
    if len(routed) > 1:
        raise ValueError(f"Expected at most one routed MLP, found {len(routed)}")
    return routed[0] if routed else None


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def render_multiple_choice_prompt(row: Dict[str, Any], option_order: Sequence[int]) -> str:
    """Render an auditable answer-order variant from structured MC metadata."""
    options = [str(option) for option in row.get("mc_options", [])]
    if not options:
        if list(option_order) != list(range(len(row["choices"]))):
            raise ValueError(
                f"Benchmark {row['benchmark_id']} lacks mc_options required for answer-order variants"
            )
        return str(row["prompt"])
    if sorted(int(index) for index in option_order) != list(range(len(options))):
        raise ValueError(f"Invalid option permutation for {row['benchmark_id']}: {option_order}")
    if len(options) > len(MC_LABELS):
        raise ValueError(f"Too many multiple-choice options: {len(options)}")
    instruction = str(row.get("mc_instruction", "")).strip()
    stem_label = str(row.get("mc_stem_label", "Question")).strip()
    stem = str(row.get("mc_stem", "")).strip()
    parts = []
    if instruction:
        parts.append(instruction)
    parts.append(f"{stem_label}: {stem}")
    rendered = "\n".join(
        f"{MC_LABELS[position]}. {options[source_index]}"
        for position, source_index in enumerate(option_order)
    )
    parts.append(f"Choices:\n{rendered}")
    parts.append("Answer:")
    return "\n\n".join(parts)


def multiple_choice_variants(
    row: Dict[str, Any],
    n_runs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Create deterministic, unique answer-order runs for one MC item.

    Repeating an unchanged likelihood evaluation would be exactly identical.
    These runs instead test sensitivity to answer position.  The identity
    order is always run zero and inference is capped by n! unique orders.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least one")
    n_options = len(row["choices"])
    if n_options < 2:
        raise ValueError(f"Multiple-choice row {row['benchmark_id']} has fewer than two choices")
    identity = tuple(range(n_options))
    if not row.get("mc_options"):
        return [
            {
                "run_index": 0,
                "option_order": list(identity),
                "prompt": str(row["prompt"]),
                "choices": list(row["choices"]),
                "answer_index": int(row["answer_index"]),
            }
        ]
    permutations = list(itertools.permutations(range(n_options)))
    alternatives = [order for order in permutations if order != identity]
    digest = hashlib.sha256(f"{seed}:{row['benchmark_id']}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    rng.shuffle(alternatives)
    orders = [identity, *alternatives[: max(0, min(n_runs, len(permutations)) - 1)]]
    variants = []
    original_answer = int(row["answer_index"])
    for run_index, order in enumerate(orders):
        new_answer = order.index(original_answer)
        variants.append(
            {
                "run_index": run_index,
                "option_order": list(order),
                "prompt": render_multiple_choice_prompt(row, order),
                "choices": [f" {MC_LABELS[index]}" for index in range(n_options)],
                "answer_index": int(new_answer),
            }
        )
    return variants


def encode_completion_pair(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
) -> Dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("Completion tokenized to an empty sequence")
    if len(completion_ids) >= max_length:
        raise ValueError(
            f"Completion has {len(completion_ids)} tokens and does not fit max_length={max_length}"
        )
    maximum_prompt = max_length - len(completion_ids)
    truncated = len(prompt_ids) > maximum_prompt
    if truncated:
        # Preserve both task framing and the final answer cue. For long legal
        # vignettes this is preferable to silently dropping only one end.
        head = maximum_prompt // 2
        tail = maximum_prompt - head
        prompt_ids = prompt_ids[:head] + prompt_ids[-tail:]
    return {
        "input_ids": prompt_ids + completion_ids,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "truncated": truncated,
    }


@torch.inference_mode()
def score_completion_pairs(
    model,
    tokenizer,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = 8,
    max_length: int = 1024,
    completion_logit_budget: int = 128,
) -> List[Dict[str, Any]]:
    if batch_size < 1 or max_length < 2 or completion_logit_budget < 1:
        raise ValueError(
            "batch_size and completion_logit_budget must be positive; max_length must be at least 2"
        )
    encoded = [
        encode_completion_pair(tokenizer, prompt, completion, max_length)
        for prompt, completion in pairs
    ]
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]["input_ids"]))
    results: List[Optional[Dict[str, Any]]] = [None] * len(encoded)
    device = model_device(model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
    routed_mlp = find_routed_mlp(model)

    batches = []
    cursor = 0
    while cursor < len(order):
        batch_indices = []
        maximum_completion = 0
        while cursor < len(order) and len(batch_indices) < batch_size:
            candidate = order[cursor]
            candidate_completion = int(encoded[candidate]["completion_tokens"])
            proposed_maximum = max(maximum_completion, candidate_completion)
            if batch_indices and (len(batch_indices) + 1) * proposed_maximum > completion_logit_budget:
                break
            batch_indices.append(candidate)
            maximum_completion = proposed_maximum
            cursor += 1
        batches.append(batch_indices)

    for batch_indices in batches:
        maximum = max(len(encoded[index]["input_ids"]) for index in batch_indices)
        maximum_completion = max(
            int(encoded[index]["completion_tokens"]) for index in batch_indices
        )
        input_ids = torch.full(
            (len(batch_indices), maximum), int(pad_id), dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, source_index in enumerate(batch_indices):
            ids = torch.tensor(encoded[source_index]["input_ids"], dtype=torch.long, device=device)
            offset = maximum - len(ids)
            input_ids[row_index, offset:] = ids
            attention_mask[row_index, offset:] = 1
        if routed_mlp is not None:
            routed_mlp.set_routing_token_mask(attention_mask)
        # Right-align all sequences, then request only the answer-adjacent
        # logits. Gemma's vocabulary is large, so materializing [B,S,V] for
        # the full prompt would otherwise dominate GPU memory.
        logits_to_keep = maximum_completion + 1
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            use_cache=False,
        ).logits
        for row_index, source_index in enumerate(batch_indices):
            item = encoded[source_index]
            prompt_tokens = int(item["prompt_tokens"])
            completion_tokens = int(item["completion_tokens"])
            local_start = maximum_completion - completion_tokens
            answer_logits = logits[
                row_index, local_start : local_start + completion_tokens
            ].float()
            targets = input_ids[row_index, maximum - completion_tokens : maximum]
            token_nll = F.cross_entropy(answer_logits, targets, reduction="none")
            if not bool(torch.isfinite(token_nll).all()):
                parameter_dtype = next(model.parameters()).dtype
                raise FloatingPointError(
                    "Non-finite completion NLL detected for flattened pair "
                    f"{source_index} (model dtype={parameter_dtype}, "
                    f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}). "
                    "Use float32 on T4/P100 or bfloat16 on a GPU with native BF16 support."
                )
            results[source_index] = {
                "sum_nll": float(token_nll.sum().item()),
                "mean_nll": float(token_nll.mean().item()),
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "truncated": bool(item["truncated"]),
            }
    if any(result is None for result in results):
        raise RuntimeError("Some completion scores were not produced")
    return [result for result in results if result is not None]


@torch.inference_mode()
def collect_prompt_routing(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    routed_mlp = find_routed_mlp(model)
    if routed_mlp is None:
        return None
    routed_mlp.reset_routing_stats()
    device = model_device(model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            batch = tokenizer(
                list(prompts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            routed_mlp.set_routing_token_mask(batch["attention_mask"])
            model(**batch, logits_to_keep=1, use_cache=False)
    finally:
        tokenizer.padding_side = old_padding_side
    return routed_mlp.routing_stats()


@torch.inference_mode()
def benchmark_prefill_runtime(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_sizes: Sequence[int] = (1, 4, 8),
    max_length: int = 512,
    warmup: int = 3,
    repeats: int = 20,
) -> Dict[str, Any]:
    """Measure prefill latency/throughput with synchronization and warm-up.

    This is a small implementation benchmark, not a hardware-independent FLOP
    estimate. The same prompts and settings must be used for base and MoE.
    """
    if not prompts or warmup < 0 or repeats < 2 or max_length < 2:
        raise ValueError("Need prompts, repeats >= 2, warmup >= 0, and max_length >= 2")
    device = model_device(model)
    routed_mlp = find_routed_mlp(model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    rows = []
    try:
        for requested_batch_size in batch_sizes:
            batch_size = int(requested_batch_size)
            if batch_size < 1:
                raise ValueError("batch_sizes must contain positive integers")
            selected = [str(prompts[index % len(prompts)]) for index in range(batch_size)]
            encoded = tokenizer(
                selected,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            valid_tokens = int(encoded["attention_mask"].sum().item())

            def forward_once() -> None:
                if routed_mlp is not None:
                    routed_mlp.set_routing_token_mask(encoded["attention_mask"])
                model(**encoded, logits_to_keep=1, use_cache=False)

            for _ in range(warmup):
                forward_once()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline_allocated = int(torch.cuda.memory_allocated(device))
            else:
                baseline_allocated = None
            elapsed = []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                forward_once()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter() - started)
            values = np.asarray(elapsed, dtype=np.float64)
            median = float(np.median(values))
            peak_allocated = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": int(encoded["input_ids"].shape[1]),
                    "valid_tokens": valid_tokens,
                    "warmup": warmup,
                    "repeats": repeats,
                    "median_seconds": median,
                    "p25_seconds": float(np.quantile(values, 0.25)),
                    "p75_seconds": float(np.quantile(values, 0.75)),
                    "tokens_per_second": float(valid_tokens / median),
                    "cuda_baseline_allocated_bytes": baseline_allocated,
                    "cuda_peak_allocated_bytes": peak_allocated,
                    "cuda_incremental_peak_bytes": (
                        int(max(0, peak_allocated - baseline_allocated))
                        if peak_allocated is not None and baseline_allocated is not None
                        else None
                    ),
                }
            )
    finally:
        tokenizer.padding_side = old_padding_side
        if routed_mlp is not None:
            routed_mlp.reset_routing_stats()
    return {
        "format": "gemma_moe_prefill_runtime_v1",
        "device": str(device),
        "max_length": max_length,
        "results": rows,
        "warning": (
            "Hardware- and implementation-specific prefill measurement; "
            "not an autoregressive decode or energy benchmark."
        ),
    }


def summarize_domain_results(
    domain_rows: Sequence[Dict[str, Any]],
    routing: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    nll_sum = float(sum(row["gold_sum_nll"] for row in domain_rows))
    token_count = int(sum(row["gold_completion_tokens"] for row in domain_rows))
    mean_item_nll = float(np.mean([row["gold_mean_nll"] for row in domain_rows]))
    multiple_choice = [row for row in domain_rows if row["task_type"] == "multiple_choice"]
    return {
        "domain_id": int(domain_rows[0]["domain_id"]),
        "domain_name": domain_rows[0]["domain_name"],
        "n_samples": len(domain_rows),
        "reference_completion_tokens": token_count,
        "reference_token_weighted_nll": nll_sum / token_count,
        "reference_token_weighted_ppl": math.exp(min(nll_sum / token_count, 50.0)),
        "reference_mean_item_nll": mean_item_nll,
        "multiple_choice_samples": len(multiple_choice),
        "multiple_choice_accuracy": (
            float(np.mean([row["correct"] for row in multiple_choice]))
            if multiple_choice
            else None
        ),
        "multiple_choice_variant_accuracy": (
            float(np.mean([row["mc_variant_accuracy"] for row in multiple_choice]))
            if multiple_choice
            else None
        ),
        "multiple_choice_mean_order_agreement": (
            float(np.mean([row["mc_semantic_prediction_agreement"] for row in multiple_choice]))
            if multiple_choice
            else None
        ),
        "multiple_choice_mean_runs": (
            float(np.mean([len(row["mc_variants"]) for row in multiple_choice]))
            if multiple_choice
            else None
        ),
        "prompt_truncation_fraction": float(
            np.mean([row["prompt_truncated"] for row in domain_rows])
        ),
        "routing": routing,
    }


def evaluate_model(
    model,
    tokenizer,
    benchmark_rows: Sequence[Dict[str, Any]],
    model_variant: str,
    batch_size: int = 8,
    max_length: int = 1024,
    mc_order_runs: int = 3,
    mc_order_seed: int = 20260916,
) -> Dict[str, Any]:
    by_domain: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        by_domain[int(row["domain_id"])].append(row)
    all_results: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for domain_id in sorted(by_domain):
        source_rows = by_domain[domain_id]
        pairs: List[Tuple[str, str]] = []
        spans: List[List[Tuple[int, int, Dict[str, Any]]]] = []
        for row in source_rows:
            row_spans = []
            if row["task_type"] == "multiple_choice":
                variants = multiple_choice_variants(row, mc_order_runs, mc_order_seed)
            else:
                variants = [
                    {
                        "run_index": 0,
                        "option_order": [],
                        "prompt": str(row["prompt"]),
                        "choices": [str(row["reference_completion"])],
                        "answer_index": 0,
                    }
                ]
            for variant in variants:
                begin = len(pairs)
                pairs.extend(
                    (str(variant["prompt"]), str(completion))
                    for completion in variant["choices"]
                )
                row_spans.append((begin, len(pairs), variant))
            spans.append(row_spans)
        scores = score_completion_pairs(
            model, tokenizer, pairs, batch_size=batch_size, max_length=max_length
        )
        domain_results = []
        for row, row_spans in zip(source_rows, spans):
            if row["task_type"] == "multiple_choice":
                variant_results = []
                semantic_predictions = []
                for begin, end, variant in row_spans:
                    candidate_scores = scores[begin:end]
                    prediction = int(
                        np.argmin([candidate["mean_nll"] for candidate in candidate_scores])
                    )
                    gold_index = int(variant["answer_index"])
                    option_order = list(variant["option_order"])
                    semantic_prediction = int(option_order[prediction])
                    semantic_predictions.append(semantic_prediction)
                    variant_results.append(
                        {
                            "run_index": int(variant["run_index"]),
                            "option_order": option_order,
                            "predicted_index": prediction,
                            "semantic_predicted_index": semantic_prediction,
                            "answer_index": gold_index,
                            "correct": bool(prediction == gold_index),
                            "candidate_mean_nll": [
                                candidate["mean_nll"] for candidate in candidate_scores
                            ],
                        }
                    )
                primary = variant_results[0]
                primary_begin, primary_end, _ = row_spans[0]
                candidate_scores = scores[primary_begin:primary_end]
                prediction = int(primary["predicted_index"])
                gold_index = int(primary["answer_index"])
                counts = np.bincount(semantic_predictions, minlength=len(row["choices"]))
                modal_count = int(counts.max())
            else:
                begin, end, _ = row_spans[0]
                candidate_scores = scores[begin:end]
                prediction = None
                gold_index = 0
                variant_results = []
                modal_count = 0
            gold = candidate_scores[gold_index]
            result = {
                "benchmark_id": row["benchmark_id"],
                "model_variant": model_variant,
                "domain_id": int(row["domain_id"]),
                "domain_name": row["domain_name"],
                "task_type": row["task_type"],
                "source": row["source"],
                "source_config": row["source_config"],
                "gold_sum_nll": gold["sum_nll"],
                "gold_mean_nll": gold["mean_nll"],
                "gold_completion_tokens": gold["completion_tokens"],
                "prompt_tokens": gold["prompt_tokens"],
                "prompt_truncated": gold["truncated"],
                "predicted_index": prediction,
                "answer_index": row["answer_index"],
                "correct": (
                    bool(prediction == int(row["answer_index"]))
                    if prediction is not None
                    else None
                ),
                "candidate_mean_nll": [candidate["mean_nll"] for candidate in candidate_scores],
                "mc_variants": variant_results,
                "mc_variant_accuracy": (
                    float(np.mean([variant["correct"] for variant in variant_results]))
                    if variant_results
                    else None
                ),
                "mc_semantic_prediction_agreement": (
                    float(modal_count / len(variant_results)) if variant_results else None
                ),
            }
            domain_results.append(result)
            all_results.append(result)
        routing = collect_prompt_routing(
            model,
            tokenizer,
            [str(row["prompt"]) for row in source_rows],
            batch_size=batch_size,
            max_length=max_length,
        )
        summaries[str(domain_id)] = summarize_domain_results(domain_results, routing)
    return {
        "model_variant": model_variant,
        "config": {
            "batch_size": batch_size,
            "max_length": max_length,
            "mc_order_runs_requested": mc_order_runs,
            "mc_order_seed": mc_order_seed,
            "mc_order_runs_note": (
                "Unique answer-order variants; binary tasks have at most two runs. "
                "Unchanged deterministic reruns are intentionally not repeated."
            ),
        },
        "domains": summaries,
        "per_example": all_results,
    }


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> List[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        means[index] = float(np.mean(values[rng.integers(0, len(values), len(values))]))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def exact_paired_binary_pvalue(base_correct: Sequence[bool], moe_correct: Sequence[bool]) -> float:
    """Two-sided exact McNemar/binomial p-value over discordant pairs."""
    if len(base_correct) != len(moe_correct):
        raise ValueError("Paired binary outcomes must have the same length")
    base_loses = sum(bool(base) and not bool(moe) for base, moe in zip(base_correct, moe_correct))
    moe_gains = sum(not bool(base) and bool(moe) for base, moe in zip(base_correct, moe_correct))
    discordant = base_loses + moe_gains
    if discordant == 0:
        return 1.0
    smaller = min(base_loses, moe_gains)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * lower_tail))


def compare_evaluations(
    base: Dict[str, Any],
    moe: Dict[str, Any],
    n_bootstrap: int = 2000,
    seed: int = 20260914,
) -> Dict[str, Any]:
    base_rows = {row["benchmark_id"]: row for row in base["per_example"]}
    moe_rows = {row["benchmark_id"]: row for row in moe["per_example"]}
    if base_rows.keys() != moe_rows.keys():
        raise ValueError("Base and MoE evaluations contain different benchmark IDs")
    comparisons = []
    for benchmark_id in sorted(base_rows):
        base_row = base_rows[benchmark_id]
        moe_row = moe_rows[benchmark_id]
        comparisons.append(
            {
                "benchmark_id": benchmark_id,
                "domain_id": int(base_row["domain_id"]),
                "domain_name": base_row["domain_name"],
                "task_type": base_row["task_type"],
                "base_gold_mean_nll": base_row["gold_mean_nll"],
                "moe_gold_mean_nll": moe_row["gold_mean_nll"],
                "delta_gold_mean_nll": moe_row["gold_mean_nll"] - base_row["gold_mean_nll"],
                "base_correct": base_row["correct"],
                "moe_correct": moe_row["correct"],
            }
        )
    domains: Dict[str, Dict[str, Any]] = {}
    for domain_id in sorted({row["domain_id"] for row in comparisons}):
        rows = [row for row in comparisons if row["domain_id"] == domain_id]
        nll_delta = np.asarray([row["delta_gold_mean_nll"] for row in rows])
        mc_rows = [row for row in rows if row["base_correct"] is not None]
        accuracy_delta = (
            np.asarray(
                [float(row["moe_correct"]) - float(row["base_correct"]) for row in mc_rows]
            )
            if mc_rows
            else np.asarray([])
        )
        variant_delta = (
            np.asarray(
                [
                    float(moe_rows[row["benchmark_id"]].get("mc_variant_accuracy"))
                    - float(base_rows[row["benchmark_id"]].get("mc_variant_accuracy"))
                    for row in mc_rows
                ]
            )
            if mc_rows and all(
                base_rows[row["benchmark_id"]].get("mc_variant_accuracy") is not None
                and moe_rows[row["benchmark_id"]].get("mc_variant_accuracy") is not None
                for row in mc_rows
            )
            else np.asarray([])
        )
        base_correct = [bool(row["base_correct"]) for row in mc_rows]
        moe_correct = [bool(row["moe_correct"]) for row in mc_rows]
        base_loses = sum(base and not moe for base, moe in zip(base_correct, moe_correct))
        moe_gains = sum(not base and moe for base, moe in zip(base_correct, moe_correct))
        domains[str(domain_id)] = {
            "domain_id": domain_id,
            "domain_name": rows[0]["domain_name"],
            "n_samples": len(rows),
            "mean_delta_gold_nll_moe_minus_base": float(np.mean(nll_delta)),
            "delta_gold_nll_ci95": bootstrap_mean_ci(
                nll_delta, n_bootstrap, seed + 1009 * domain_id
            ),
            "moe_lower_nll_fraction": float(np.mean(nll_delta < 0.0)),
            "base_multiple_choice_accuracy": (
                float(np.mean([row["base_correct"] for row in mc_rows])) if mc_rows else None
            ),
            "moe_multiple_choice_accuracy": (
                float(np.mean([row["moe_correct"] for row in mc_rows])) if mc_rows else None
            ),
            "accuracy_delta_moe_minus_base": (
                float(np.mean(accuracy_delta)) if mc_rows else None
            ),
            "accuracy_delta_ci95": (
                bootstrap_mean_ci(accuracy_delta, n_bootstrap, seed + 7919 * domain_id)
                if mc_rows
                else None
            ),
            "base_to_moe_correct": int(moe_gains) if mc_rows else None,
            "base_to_moe_incorrect": int(base_loses) if mc_rows else None,
            "exact_paired_accuracy_pvalue": (
                exact_paired_binary_pvalue(base_correct, moe_correct) if mc_rows else None
            ),
            "base_multiple_choice_variant_accuracy": (
                float(np.mean([base_rows[row["benchmark_id"]]["mc_variant_accuracy"] for row in mc_rows]))
                if variant_delta.size
                else None
            ),
            "moe_multiple_choice_variant_accuracy": (
                float(np.mean([moe_rows[row["benchmark_id"]]["mc_variant_accuracy"] for row in mc_rows]))
                if variant_delta.size
                else None
            ),
            "variant_accuracy_delta_moe_minus_base": (
                float(np.mean(variant_delta)) if variant_delta.size else None
            ),
            "variant_accuracy_delta_ci95": (
                bootstrap_mean_ci(variant_delta, n_bootstrap, seed + 12347 * domain_id)
                if variant_delta.size
                else None
            ),
        }
    all_delta = np.asarray([row["delta_gold_mean_nll"] for row in comparisons])
    return {
        "format": "gemma_moe_independent_benchmark_comparison_v1",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "overall": {
            "n_samples": len(comparisons),
            "mean_delta_gold_nll_moe_minus_base": float(np.mean(all_delta)),
            "delta_gold_nll_ci95": bootstrap_mean_ci(all_delta, n_bootstrap, seed),
            "moe_lower_nll_fraction": float(np.mean(all_delta < 0.0)),
        },
        "domains": domains,
        "per_example": comparisons,
    }


def normalize_numeric_answer(value: str) -> Optional[str]:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", value)
    if not matches:
        return None
    answer = matches[-1].replace(",", "")
    try:
        numeric = float(answer)
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".12g")
    except ValueError:
        return answer


def extract_python_code(value: str) -> str:
    """Extract the first intended code fragment without confusing closing fences.

    MBPP prompts end in an opening Markdown Python fence, so decoded model text
    normally starts with raw code and then a *closing* fence.  A regex looking
    for the first complete fence pair incorrectly treated that closing marker
    as a new opener and returned later prose (or an empty block).
    """
    text = value.strip()
    if not text:
        return ""
    raw_prefix = text.split("```", 1)[0].strip()
    python_start = re.compile(r"^(?:#|@|async\s+def\s|def\s|class\s|from\s|import\s)")
    raw_prefix_is_program = False
    if raw_prefix:
        try:
            raw_prefix_is_program = bool(ast.parse(raw_prefix).body)
        except (SyntaxError, TypeError, ValueError):
            pass
    if raw_prefix and (python_start.match(raw_prefix) or raw_prefix_is_program):
        return raw_prefix

    fenced_blocks = re.findall(
        r"```\s*(python|py)?\s*\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Prefer an explicitly Python-labelled, non-empty block, then any
    # non-empty fenced block. Never return an empty match.
    for required_label in (True, False):
        for label, content in fenced_blocks:
            candidate = content.strip()
            if candidate and (not required_label or bool(label)):
                return candidate
    return text.strip("`").strip()


def is_syntactically_valid_python(value: str) -> bool:
    """Require both a parseable program and at least one AST statement."""
    if not value.strip():
        return False
    try:
        return bool(ast.parse(value).body)
    except (SyntaxError, TypeError, ValueError):
        return False


@torch.inference_mode()
def generate_task_outputs(
    model,
    tokenizer,
    benchmark_rows: Sequence[Dict[str, Any]],
    model_variant: str,
    batch_size: int = 8,
    max_prompt_length: int = 768,
    max_new_tokens: int = 160,
) -> Dict[str, Any]:
    rows = [
        row
        for row in benchmark_rows
        if row["task_type"] in {"numeric_generation", "python_generation"}
    ]
    device = model_device(model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    generated_rows = []
    try:
        for start in range(0, len(rows), batch_size):
            source_rows = rows[start : start + batch_size]
            batch = tokenizer(
                [str(row["prompt"]) for row in source_rows],
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output_ids = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
            new_ids = output_ids[:, batch["input_ids"].shape[1] :]
            outputs = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for row, output in zip(source_rows, outputs):
                if row["task_type"] == "numeric_generation":
                    predicted = normalize_numeric_answer(output)
                    expected = normalize_numeric_answer(str(row["reference_answer"]))
                    metric = {"numeric_exact_match": predicted == expected}
                else:
                    code = extract_python_code(output)
                    syntax_valid = is_syntactically_valid_python(code)
                    predicted = code
                    expected = None
                    metric = {"python_syntax_valid": syntax_valid}
                generated_rows.append(
                    {
                        "benchmark_id": row["benchmark_id"],
                        "model_variant": model_variant,
                        "domain_id": int(row["domain_id"]),
                        "domain_name": row["domain_name"],
                        "task_type": row["task_type"],
                        "generated_text": output,
                        "normalized_prediction": predicted,
                        "normalized_reference": expected,
                        **metric,
                    }
                )
    finally:
        tokenizer.padding_side = old_padding_side
    summary = {}
    for task_type in ("numeric_generation", "python_generation"):
        task_rows = [row for row in generated_rows if row["task_type"] == task_type]
        metric_name = (
            "numeric_exact_match" if task_type == "numeric_generation" else "python_syntax_valid"
        )
        summary[task_type] = {
            "n_samples": len(task_rows),
            metric_name: float(np.mean([row[metric_name] for row in task_rows])) if task_rows else None,
        }
    return {"model_variant": model_variant, "summary": summary, "per_example": generated_rows}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    def non_finite_paths(value: Any, prefix: str = "$") -> List[str]:
        if isinstance(value, float):
            return [prefix] if not math.isfinite(value) else []
        if isinstance(value, dict):
            result = []
            for key, child in value.items():
                result.extend(non_finite_paths(child, f"{prefix}.{key}"))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for index, child in enumerate(value):
                result.extend(non_finite_paths(child, f"{prefix}[{index}]"))
            return result
        return []

    invalid = non_finite_paths(payload)
    if invalid:
        preview = ", ".join(invalid[:10])
        suffix = " ..." if len(invalid) > 10 else ""
        raise ValueError(
            f"Refusing to write non-finite benchmark values at: {preview}{suffix}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
