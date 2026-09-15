from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from domain_mapping.domain_mlp_activation_mapping import describe_mlp, get_mlp_module
from domain_mapping.topographic_mlp_sae_mapping import (
    DEFAULT_DATA_PATH,
    DEFAULT_LAYER_NUM,
    MODEL_NAME,
    TOKENIZER_NAME,
    pearson_by_neuron,
    resolve_domain_dir,
)


@dataclass
class DomainMapping:
    domain_id: int
    domain_name: str
    activation_path: Path
    correlation_path: Path
    cluster_column: Optional[int] = None


def load_mapping_summary(mapping_dir: Path, selected_domain_ids: Optional[Sequence[int]]) -> List[DomainMapping]:
    summary_path = mapping_dir / "summary.json"
    selected = set(selected_domain_ids) if selected_domain_ids else None

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        mappings = []
        for domain in payload.get("domains", []):
            domain_id = int(domain["domain_id"])
            if selected is not None and domain_id not in selected:
                continue
            activation_path = Path(domain["activations_path"])
            correlation_path = Path(domain["correlations_path"])
            if not activation_path.exists():
                relocated = mapping_dir / activation_path.name
                if relocated.exists():
                    activation_path = relocated
            if not correlation_path.exists():
                relocated = mapping_dir / correlation_path.name
                if relocated.exists():
                    correlation_path = relocated
            mappings.append(
                DomainMapping(
                    domain_id=domain_id,
                    domain_name=str(domain.get("domain_name", f"domain_{domain_id}")),
                    activation_path=activation_path,
                    correlation_path=correlation_path,
                    cluster_column=(
                        int(domain["cluster_column"])
                        if domain.get("cluster_column") is not None
                        else None
                    ),
                )
            )
        if mappings:
            return mappings

    mappings = []
    for correlation_path in sorted(mapping_dir.glob("domain_*_mlp_neuron_correlations.csv")):
        stem_parts = correlation_path.stem.split("_")
        try:
            domain_id = int(stem_parts[1])
        except (IndexError, ValueError):
            continue
        if selected is not None and domain_id not in selected:
            continue
        mappings.append(
            DomainMapping(
                domain_id=domain_id,
                domain_name=f"domain_{domain_id}",
                activation_path=mapping_dir / f"domain_{domain_id}_mlp_activations.npz",
                correlation_path=correlation_path,
            )
        )

    if not mappings:
        raise FileNotFoundError(f"No domain MLP mapping files found in {mapping_dir}")
    return mappings


def load_neuron_scores(correlation_path: Path, metric: str, n_inner: int) -> np.ndarray:
    scores = np.full(n_inner, np.nan, dtype=np.float64)
    domain_name = None

    with correlation_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if metric not in (reader.fieldnames or []):
            raise ValueError(f"Metric '{metric}' not found in {correlation_path}")
        for row in reader:
            neuron_id = int(row["neuron_id"])
            if neuron_id < 0 or neuron_id >= n_inner:
                raise ValueError(f"Neuron id {neuron_id} in {correlation_path} is outside n_inner={n_inner}")
            value = row.get(metric, "")
            if value not in ("", None):
                scores[neuron_id] = float(value)
            domain_name = row.get("domain_name") or domain_name

    if domain_name:
        return scores
    return scores


def estimate_per_domain_null_tau(
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    n_permutations: int,
    percentile: float,
    seed: int,
    block_size: int = 32,
    permutation_batch_size: int = 16,
) -> Dict[str, object]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if np.std(cluster_activations) == 0:
        raise ValueError(
            "Domain signal has zero variance; cannot establish an empirical tau "
            "from shuffled null correlations."
        )

    rng = np.random.default_rng(seed)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be positive")
    null_max_abs_correlations = []

    # Pearson sufficient statistics for X do not change across permutations.
    # Batched X.T @ Y computes the same null as the scalar implementation but
    # lets optimized BLAS process many permutations in one pass.
    x = np.asarray(mlp_activations, dtype=np.float32)
    y = np.asarray(cluster_activations, dtype=np.float32).reshape(-1)
    n_rows = x.shape[0]
    sum_x = x.sum(axis=0, dtype=np.float64)
    sum_x2 = np.square(x, dtype=np.float64).sum(axis=0)
    x_ss = np.maximum(sum_x2 - np.square(sum_x) / n_rows, 0.0)
    block_indices = [
        np.arange(start, min(start + block_size, n_rows), dtype=np.int64)
        for start in range(0, n_rows, block_size)
    ]

    completed = 0
    while completed < n_permutations:
        batch_count = min(permutation_batch_size, n_permutations - completed)
        shuffled = np.empty((n_rows, batch_count), dtype=np.float32)
        for column in range(batch_count):
            order = rng.permutation(len(block_indices))
            shuffled[:, column] = y[np.concatenate([block_indices[int(i)] for i in order])]
        sum_y = shuffled.sum(axis=0, dtype=np.float64)
        sum_y2 = np.square(shuffled, dtype=np.float64).sum(axis=0)
        y_ss = np.maximum(sum_y2 - np.square(sum_y) / n_rows, 0.0)
        mean_x = np.asarray(sum_x / n_rows, dtype=np.float32)
        mean_y = np.asarray(sum_y / n_rows, dtype=np.float32)
        numerator = np.zeros((x.shape[1], batch_count), dtype=np.float64)
        # Center before the float32 BLAS operation to avoid subtracting two
        # large, nearly equal raw-moment terms. Accumulate chunk results in
        # float64 without materialising a second full [tokens, neurons] array.
        for row_start in range(0, n_rows, 4096):
            row_end = min(row_start + 4096, n_rows)
            x_centered = x[row_start:row_end] - mean_x
            y_centered = shuffled[row_start:row_end] - mean_y
            numerator += np.asarray(x_centered.T @ y_centered, dtype=np.float64)
        denominator = np.sqrt(np.outer(x_ss, y_ss))
        correlations = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        for column in range(batch_count):
            finite = np.abs(correlations[:, column][np.isfinite(correlations[:, column])])
            if finite.size:
                # Max-statistic controls the family-wise probability of
                # selecting any neuron under the shuffled null.
                null_max_abs_correlations.append(float(finite.max()))
        completed += batch_count

    finite_null = np.asarray(null_max_abs_correlations, dtype=np.float64)
    if finite_null.size == 0:
        raise ValueError("Null baseline produced no finite correlations")

    tau = float(np.percentile(finite_null, percentile))
    return {
        "tau": tau,
        "null_percentile": float(percentile),
        "n_permutations": int(n_permutations),
        "null_statistic": "max_abs_correlation_across_neurons",
        "null_block_size": int(block_size),
        "permutation_batch_size": int(permutation_batch_size),
        "null_max_mean": float(finite_null.mean()),
        "null_max_std": float(finite_null.std()),
        "null_max": float(finite_null.max()),
    }


def estimate_quantile_tau(scores: np.ndarray, percentile: float) -> Dict[str, object]:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("Cannot estimate quantile tau: no finite scores")
    tau = float(np.percentile(finite_scores, percentile))
    return {
        "tau": tau,
        "score_percentile": float(percentile),
    }


def build_keep_mask(scores: np.ndarray, tau: float, min_keep_fraction: float) -> np.ndarray:
    if not 0 <= min_keep_fraction <= 1:
        raise ValueError("min_keep_fraction must be between 0 and 1")

    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    keep_mask = finite_scores >= tau
    min_keep = int(math.ceil(min_keep_fraction * scores.shape[0]))

    if min_keep > 0 and int(keep_mask.sum()) < min_keep:
        ranked = np.argsort(finite_scores)[::-1]
        ranked = ranked[np.isfinite(finite_scores[ranked])]
        if ranked.size == 0:
            raise ValueError("Cannot enforce min_keep_fraction: no finite neuron scores")
        keep_mask[:] = False
        keep_mask[ranked[:min_keep]] = True

    return keep_mask


def build_empty_signal_mask(n_inner: int, policy: str) -> np.ndarray:
    if policy == "keep-all":
        return np.ones(n_inner, dtype=bool)
    if policy == "prune-all":
        return np.zeros(n_inner, dtype=bool)
    raise ValueError(f"Unsupported empty signal policy for mask construction: {policy}")


def validate_keep_mask(mapping: DomainMapping, keep_mask: np.ndarray, allow_empty_experts: bool) -> None:
    if keep_mask.any():
        return
    message = (
        f"Domain {mapping.domain_id} ({mapping.domain_name}) would export an empty expert "
        "with 0 kept neurons. This is usually a failed domain/mapping, not a usable "
        "specialist. Pass --allow-empty-experts only for diagnostic artifacts."
    )
    if allow_empty_experts:
        print(f"WARNING: {message}")
        return
    raise ValueError(message)


def _zero_by_inner_dim(weight, prune_mask: np.ndarray, preferred_axis: int) -> str:
    import torch

    mask = torch.as_tensor(prune_mask, dtype=torch.bool, device=weight.device)
    n_inner = int(mask.numel())

    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(weight.shape)}")

    with torch.no_grad():
        if preferred_axis == 0 and weight.shape[0] == n_inner:
            weight[mask, :] = 0
            return "rows"
        if preferred_axis == 1 and weight.shape[1] == n_inner:
            weight[:, mask] = 0
            return "columns"
        if weight.shape[0] == n_inner:
            weight[mask, :] = 0
            return "rows"
        if weight.shape[1] == n_inner:
            weight[:, mask] = 0
            return "columns"

    raise ValueError(f"Could not align prune mask of length {n_inner} with weight shape {tuple(weight.shape)}")


def apply_mlp_mask(mlp_module, keep_mask: np.ndarray, zero_bias: bool) -> Dict[str, object]:
    import torch

    spec = describe_mlp(mlp_module)
    prune_mask = ~keep_mask.astype(bool)
    projection_axes: Dict[str, str] = {}
    bias_zeroed: List[str] = []
    for projection_name in spec["input_projections"]:
        projection = getattr(mlp_module, projection_name)
        projection_axes[projection_name] = _zero_by_inner_dim(
            projection.weight, prune_mask, preferred_axis=0
        )
        bias = getattr(projection, "bias", None)
        if zero_bias and bias is not None and bias.shape[0] == keep_mask.shape[0]:
            with torch.no_grad():
                bias[torch.as_tensor(prune_mask, dtype=torch.bool, device=bias.device)] = 0
            bias_zeroed.append(str(projection_name))

    output_name = str(spec["output_projection"])
    output_projection = getattr(mlp_module, output_name)
    projection_axes[output_name] = _zero_by_inner_dim(
        output_projection.weight, prune_mask, preferred_axis=1
    )

    for parameter in mlp_module.parameters():
        parameter.requires_grad_(False)

    return {
        "mlp_family": spec["family"],
        "projection_zero_axes": projection_axes,
        "bias_zeroed": bias_zeroed,
        "n_inner": int(keep_mask.shape[0]),
        "n_kept": int(keep_mask.sum()),
        "n_pruned": int((~keep_mask).sum()),
        "sparsity_fraction": float((~keep_mask).sum() / keep_mask.shape[0]),
    }


def tensor_state_dict_to_cpu(module) -> Dict[str, object]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_activation_artifact(activation_path: Path) -> Dict[str, object]:
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_path}")
    with np.load(activation_path, allow_pickle=False) as payload:
        return {
            "mlp_activations": np.asarray(payload["mlp_activations"], dtype=np.float32),
            "cluster_activations": np.asarray(
                payload["sae_cluster_activations"], dtype=np.float32
            ),
            "layer_num": int(payload["layer_num"]) if "layer_num" in payload else None,
            "mlp_family": str(payload["mlp_family"]) if "mlp_family" in payload else None,
        }


def select_domain_arrays(
    artifact: Dict[str, object], cluster_column: Optional[int]
) -> Dict[str, object]:
    cluster_activations = artifact["cluster_activations"]
    if cluster_activations.ndim == 2:
        if cluster_column is None:
            raise ValueError("Shared activation artifact requires cluster_column")
        if not 0 <= cluster_column < cluster_activations.shape[1]:
            raise IndexError(
                f"cluster_column={cluster_column} is outside shared signal width "
                f"{cluster_activations.shape[1]}"
            )
        cluster_activations = cluster_activations[:, cluster_column]
    elif cluster_activations.ndim != 1:
        raise ValueError(
            "sae_cluster_activations must be a vector or [tokens, domains] matrix"
        )
    return {
        # Pearson correlation below accumulates float64 sufficient statistics
        # in chunks. Eager float64 copies only double peak RAM.
        "mlp_activations": artifact["mlp_activations"],
        "cluster_activations": cluster_activations,
        "layer_num": artifact["layer_num"],
        "mlp_family": artifact["mlp_family"],
    }


def load_domain_arrays(
    activation_path: Path,
    cluster_column: Optional[int] = None,
) -> Dict[str, object]:
    return select_domain_arrays(load_activation_artifact(activation_path), cluster_column)


def validate_selectivity_gate(
    selectivity_path: Path,
    mappings: Sequence[DomainMapping],
    allow_failed_selectivity: bool,
) -> Dict[str, object]:
    if not selectivity_path.exists():
        raise FileNotFoundError(
            f"SAE domain-selectivity result not found: {selectivity_path}. "
            "Run validate_domain_sae_selectivity.py before pruning."
        )
    with selectivity_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = {int(row["domain_id"]): row for row in payload.get("domains", [])}
    missing = [mapping.domain_id for mapping in mappings if mapping.domain_id not in results]
    failed = [
        mapping.domain_id
        for mapping in mappings
        if mapping.domain_id in results and not results[mapping.domain_id].get("passes_prespecified_gate")
    ]
    if missing or failed:
        message = f"SAE selectivity gate has missing domains={missing}, failed domains={failed}"
        if not allow_failed_selectivity:
            raise ValueError(message + ". Fix domain triage or use the diagnostic override.")
        print(f"WARNING: {message}")
    return payload


def export_domain_expert(
    output_dir: Path,
    mapping: DomainMapping,
    mlp_module,
    keep_mask: np.ndarray,
    tau_info: Dict[str, object],
    score_metric: str,
    model_name: str,
    layer_num: int,
    zero_stats: Dict[str, object],
    seed: int,
) -> Dict[str, object]:
    import torch

    kept_neuron_ids = np.flatnonzero(keep_mask).astype(int).tolist()
    pruned_neuron_ids = np.flatnonzero(~keep_mask).astype(int).tolist()

    expert_path = output_dir / f"domain_{mapping.domain_id}_mlp_expert.pt"
    mask_path = output_dir / f"domain_{mapping.domain_id}_mask.npy"
    metadata_path = output_dir / f"domain_{mapping.domain_id}_pruning.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, keep_mask.astype(bool))
    artifact = {
        "format": "mlp_only_sparse_expert",
        "model_name": model_name,
        "layer_num": layer_num,
        "domain_id": mapping.domain_id,
        "domain_name": mapping.domain_name,
        "score_metric": score_metric,
        "mlp_family": zero_stats.get("mlp_family"),
        "tau_info": tau_info,
        "keep_mask": torch.as_tensor(keep_mask.astype(bool)),
        "mlp_state_dict": tensor_state_dict_to_cpu(mlp_module),
    }
    torch.save(artifact, expert_path)

    metadata = {
        "format": "mlp_only_sparse_expert",
        "model_name": model_name,
        "layer_num": layer_num,
        "domain_id": mapping.domain_id,
        "domain_name": mapping.domain_name,
        "score_metric": score_metric,
        "mlp_family": zero_stats.get("mlp_family"),
        "seed": seed,
        "tau_info": tau_info,
        "zero_stats": zero_stats,
        "kept_neuron_ids": kept_neuron_ids,
        "pruned_neuron_ids": pruned_neuron_ids,
        "expert_path": str(expert_path),
        "mask_path": str(mask_path),
        "source_activation_path": str(mapping.activation_path),
        "source_correlation_path": str(mapping.correlation_path),
    }
    save_json(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune GPT-Neo, GPT-NeoX, or Gemma MLP weights into domain experts."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--domain-dir", default=None, help="Directory containing domain_mlp_mapping/.")
    parser.add_argument("--mapping-dir", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--domain-ids", type=int, nargs="*", default=None)
    parser.add_argument("--metric", choices=["abs_pearson_r", "mutual_info"], default="abs_pearson_r")
    parser.add_argument("--tau-method", choices=["per-domain-null", "quantile"], default="per-domain-null")
    parser.add_argument("--null-permutations", type=int, default=100)
    parser.add_argument("--null-percentile", type=float, default=95.0)
    parser.add_argument(
        "--null-block-size",
        type=int,
        default=32,
        help="Contiguous token block size preserved by the max-stat null permutation.",
    )
    parser.add_argument("--null-batch-size", type=int, default=16)
    parser.add_argument("--quantile-percentile", type=float, default=90.0)
    parser.add_argument(
        "--min-keep-fraction",
        type=float,
        default=0.0,
        help="Optional non-statistical floor. Default 0 does not force false-positive neurons.",
    )
    parser.add_argument(
        "--empty-signal-policy",
        choices=["error", "keep-all", "prune-all"],
        default="error",
        help=(
            "What to do when a domain has zero SAE-cluster signal and no finite "
            "correlations. The default refuses to invent tau."
        ),
    )
    parser.add_argument("--zero-bias", action="store_true")
    parser.add_argument(
        "--allow-empty-experts",
        action="store_true",
        help="Allow exporting a diagnostic expert with zero kept neurons.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-mapping-metadata-mismatch",
        action="store_true",
        help="Permit a model/layer/MLP-family mismatch only for diagnostics.",
    )
    parser.add_argument("--selectivity-results", default=None)
    parser.add_argument(
        "--allow-failed-selectivity",
        action="store_true",
        help="Diagnostic override for missing or failed domain-selectivity gates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if min(args.null_permutations, args.null_block_size, args.null_batch_size) < 1:
        raise ValueError("Null permutations, block size, and batch size must be positive")
    for name, value in (
        ("null_percentile", args.null_percentile),
        ("quantile_percentile", args.quantile_percentile),
    ):
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 100")
    if not 0.0 <= args.min_keep_fraction <= 1.0:
        raise ValueError("--min-keep-fraction must be between 0 and 1")
    if args.metric == "mutual_info" and args.tau_method == "per-domain-null":
        raise ValueError(
            "--tau-method per-domain-null currently estimates an absolute-Pearson "
            "null and cannot threshold mutual-information scores in different units. "
            "Use --metric abs_pearson_r or the explicitly heuristic quantile method."
        )

    from transformers import AutoModelForCausalLM

    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir else domain_dir / "domain_mlp_mapping"
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "domain_mlp_experts"

    if not mapping_dir.exists():
        raise FileNotFoundError(
            f"Mapping directory not found: {mapping_dir}. Run domain_mlp_activation_mapping.py first."
        )

    mappings = load_mapping_summary(mapping_dir, selected_domain_ids=args.domain_ids)
    selectivity_path = Path(args.selectivity_results) if args.selectivity_results else (
        mapping_dir / "selectivity" / "selectivity_results.json"
    )
    selectivity = validate_selectivity_gate(
        selectivity_path, mappings, allow_failed_selectivity=args.allow_failed_selectivity
    )

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    mlp_module = get_mlp_module(model, args.layer_num)
    mlp_spec = describe_mlp(mlp_module)
    base_mlp_state = tensor_state_dict_to_cpu(mlp_module)

    mapping_summary_path = mapping_dir / "summary.json"
    if mapping_summary_path.exists():
        with mapping_summary_path.open("r", encoding="utf-8") as f:
            mapping_summary = json.load(f)
        mismatches = []
        if mapping_summary.get("model_name") not in (None, args.model_name):
            mismatches.append(
                f"mapping model={mapping_summary.get('model_name')!r}, requested={args.model_name!r}"
            )
        if mapping_summary.get("layer_num") not in (None, args.layer_num):
            mismatches.append(
                f"mapping layer={mapping_summary.get('layer_num')}, requested={args.layer_num}"
            )
        if mapping_summary.get("mlp_family") not in (None, mlp_spec["family"]):
            mismatches.append(
                f"mapping MLP={mapping_summary.get('mlp_family')!r}, model={mlp_spec['family']!r}"
            )
        if mismatches:
            message = "Incompatible mapping metadata: " + "; ".join(mismatches)
            if args.allow_mapping_metadata_mismatch:
                print(f"WARNING: {message}")
            else:
                raise ValueError(
                    message + ". Use --allow-mapping-metadata-mismatch only for diagnostics."
                )

    summary_domains = []
    activation_cache: Dict[Path, Dict[str, object]] = {}
    for index, mapping in enumerate(mappings):
        cache_key = mapping.activation_path.resolve()
        if cache_key not in activation_cache:
            activation_cache[cache_key] = load_activation_artifact(mapping.activation_path)
        arrays = select_domain_arrays(activation_cache[cache_key], mapping.cluster_column)
        mlp_activations = arrays["mlp_activations"]
        cluster_activations = arrays["cluster_activations"]
        artifact_layer = arrays.get("layer_num")
        artifact_family = arrays.get("mlp_family")
        if artifact_layer not in (None, args.layer_num) or artifact_family not in (
            None,
            mlp_spec["family"],
        ):
            message = (
                f"Activation artifact for domain {mapping.domain_id} has layer={artifact_layer}, "
                f"MLP={artifact_family!r}; expected layer={args.layer_num}, "
                f"MLP={mlp_spec['family']!r}"
            )
            if args.allow_mapping_metadata_mismatch:
                print(f"WARNING: {message}")
            else:
                raise ValueError(message)
        n_inner = int(mlp_activations.shape[1])
        output_projection = getattr(mlp_module, str(mlp_spec["output_projection"]))
        if int(output_projection.weight.shape[1]) != n_inner:
            raise ValueError(
                f"Mapping n_inner={n_inner} does not match {mlp_spec['output_projection']} "
                f"input width {int(output_projection.weight.shape[1])}"
            )

        scores = load_neuron_scores(mapping.correlation_path, metric=args.metric, n_inner=n_inner)
        empty_signal = np.std(cluster_activations) == 0 or not np.isfinite(scores).any()
        if empty_signal:
            if args.empty_signal_policy == "error":
                raise ValueError(
                    f"Domain {mapping.domain_id} has zero domain signal or no finite "
                    "neuron correlations. Re-run with --empty-signal-policy keep-all "
                    "or prune-all if you want an explicit degenerate expert artifact."
                )
            tau_info = {
                "tau": None,
                "method": args.tau_method,
                "empty_signal": True,
                "empty_signal_policy": args.empty_signal_policy,
                "reason": "zero cluster variance or no finite neuron correlations",
                "min_keep_fraction": float(args.min_keep_fraction),
            }
            keep_mask = build_empty_signal_mask(n_inner=n_inner, policy=args.empty_signal_policy)
        elif args.tau_method == "per-domain-null":
            tau_info = estimate_per_domain_null_tau(
                mlp_activations=mlp_activations,
                cluster_activations=cluster_activations,
                n_permutations=args.null_permutations,
                percentile=args.null_percentile,
                seed=args.seed + mapping.domain_id,
                block_size=args.null_block_size,
                permutation_batch_size=args.null_batch_size,
            )
            keep_mask = build_keep_mask(
                scores=scores,
                tau=float(tau_info["tau"]),
                min_keep_fraction=args.min_keep_fraction,
            )
        elif args.tau_method == "quantile":
            tau_info = estimate_quantile_tau(scores=scores, percentile=args.quantile_percentile)
            keep_mask = build_keep_mask(
                scores=scores,
                tau=float(tau_info["tau"]),
                min_keep_fraction=args.min_keep_fraction,
            )
        else:
            raise ValueError(f"Unsupported tau method: {args.tau_method}")

        validate_keep_mask(
            mapping=mapping,
            keep_mask=keep_mask,
            allow_empty_experts=args.allow_empty_experts,
        )

        mlp_module.load_state_dict(base_mlp_state)
        zero_stats = apply_mlp_mask(mlp_module, keep_mask=keep_mask, zero_bias=args.zero_bias)
        exported_tau_info = {
            **tau_info,
            "method": args.tau_method,
            "min_keep_fraction": float(args.min_keep_fraction),
        }
        metadata = export_domain_expert(
            output_dir=output_dir,
            mapping=mapping,
            mlp_module=mlp_module,
            keep_mask=keep_mask,
            tau_info=exported_tau_info,
            score_metric=args.metric,
            model_name=args.model_name,
            layer_num=args.layer_num,
            zero_stats=zero_stats,
            seed=args.seed + index,
        )
        tau_value = tau_info.get("tau")
        summary_domains.append(
            {
                "domain_id": mapping.domain_id,
                "domain_name": mapping.domain_name,
                "tau": None if tau_value is None else float(tau_value),
                "empty_signal": bool(empty_signal),
                "n_inner": zero_stats["n_inner"],
                "n_kept": zero_stats["n_kept"],
                "n_pruned": zero_stats["n_pruned"],
                "sparsity_fraction": zero_stats["sparsity_fraction"],
                "expert_path": metadata["expert_path"],
                "mask_path": metadata["mask_path"],
                "diagnostic_empty_expert": bool(not keep_mask.any()),
            }
        )
        tau_text = "None" if tau_value is None else f"{float(tau_value):.6f}"
        print(
            f"Domain {mapping.domain_id}: tau={tau_text}, "
            f"kept={zero_stats['n_kept']}/{zero_stats['n_inner']}"
        )

    summary = {
        "format": "mlp_domain_pruning_summary",
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "layer_num": args.layer_num,
        "mapping_dir": str(mapping_dir),
        "output_dir": str(output_dir),
        "score_metric": args.metric,
        "tau_method": args.tau_method,
        "null_permutations": args.null_permutations,
        "null_percentile": args.null_percentile,
        "null_block_size": args.null_block_size,
        "null_batch_size": args.null_batch_size,
        "min_keep_fraction": args.min_keep_fraction,
        "mlp_family": mlp_spec["family"],
        "empty_signal_policy": args.empty_signal_policy,
        "zero_bias": args.zero_bias,
        "allow_empty_experts": args.allow_empty_experts,
        "allow_mapping_metadata_mismatch": args.allow_mapping_metadata_mismatch,
        "selectivity_results": str(selectivity_path),
        "selectivity_all_domains_pass": bool(selectivity.get("all_domains_pass")),
        "allow_failed_selectivity": args.allow_failed_selectivity,
        "domains": summary_domains,
    }
    save_json(output_dir / "pruning_summary.json", summary)
    print(f"Domain MLP experts written to: {output_dir}")


if __name__ == "__main__":
    main()
