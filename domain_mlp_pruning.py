from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from domain_mlp_activation_mapping import get_mlp_module
from topographic_mlp_sae_mapping import (
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
            mappings.append(
                DomainMapping(
                    domain_id=domain_id,
                    domain_name=str(domain.get("domain_name", f"domain_{domain_id}")),
                    activation_path=Path(domain["activations_path"]),
                    correlation_path=Path(domain["correlations_path"]),
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
) -> Dict[str, object]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if np.std(cluster_activations) == 0:
        raise ValueError(
            "Domain signal has zero variance; cannot establish an empirical tau "
            "from shuffled null correlations."
        )

    rng = np.random.default_rng(seed)
    null_abs_correlations = []

    for _ in range(n_permutations):
        shuffled_cluster = rng.permutation(cluster_activations)
        null_r = pearson_by_neuron(mlp_activations, shuffled_cluster)
        null_abs_correlations.append(np.abs(null_r))

    null_abs = np.concatenate(null_abs_correlations)
    finite_null = null_abs[np.isfinite(null_abs)]
    if finite_null.size == 0:
        raise ValueError("Null baseline produced no finite correlations")

    tau = float(np.percentile(finite_null, percentile))
    return {
        "tau": tau,
        "null_percentile": float(percentile),
        "n_permutations": int(n_permutations),
        "null_mean": float(finite_null.mean()),
        "null_std": float(finite_null.std()),
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

    if not hasattr(mlp_module, "c_fc") or not hasattr(mlp_module, "c_proj"):
        raise ValueError("MLP module must expose c_fc and c_proj to apply a neuron mask")

    prune_mask = ~keep_mask.astype(bool)
    c_fc_axis = _zero_by_inner_dim(mlp_module.c_fc.weight, prune_mask, preferred_axis=0)
    c_proj_axis = _zero_by_inner_dim(mlp_module.c_proj.weight, prune_mask, preferred_axis=1)

    bias_zeroed = False
    if zero_bias and getattr(mlp_module.c_fc, "bias", None) is not None:
        bias = mlp_module.c_fc.bias
        if bias.shape[0] == keep_mask.shape[0]:
            with torch.no_grad():
                bias[torch.as_tensor(prune_mask, dtype=torch.bool, device=bias.device)] = 0
            bias_zeroed = True

    for parameter in mlp_module.parameters():
        parameter.requires_grad_(False)

    return {
        "c_fc_zero_axis": c_fc_axis,
        "c_proj_zero_axis": c_proj_axis,
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


def load_domain_arrays(activation_path: Path) -> Dict[str, np.ndarray]:
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_path}")
    with np.load(activation_path) as payload:
        return {
            "mlp_activations": payload["mlp_activations"].astype(np.float64),
            "cluster_activations": payload["sae_cluster_activations"].astype(np.float64),
        }


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
        description="Prune TinyStories GPT-Neo MLP weights into sparse domain experts."
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
    parser.add_argument("--quantile-percentile", type=float, default=90.0)
    parser.add_argument("--min-keep-fraction", type=float, default=0.05)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM

    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir else domain_dir / "domain_mlp_mapping"
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "domain_mlp_experts"

    if not mapping_dir.exists():
        raise FileNotFoundError(
            f"Mapping directory not found: {mapping_dir}. Run domain_mlp_activation_mapping.py first."
        )

    mappings = load_mapping_summary(mapping_dir, selected_domain_ids=args.domain_ids)

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    mlp_module = get_mlp_module(model, args.layer_num)
    base_mlp_state = tensor_state_dict_to_cpu(mlp_module)

    summary_domains = []
    for index, mapping in enumerate(mappings):
        arrays = load_domain_arrays(mapping.activation_path)
        mlp_activations = arrays["mlp_activations"]
        cluster_activations = arrays["cluster_activations"]
        n_inner = int(mlp_activations.shape[1])

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
        "min_keep_fraction": args.min_keep_fraction,
        "empty_signal_policy": args.empty_signal_policy,
        "zero_bias": args.zero_bias,
        "allow_empty_experts": args.allow_empty_experts,
        "domains": summary_domains,
    }
    save_json(output_dir / "pruning_summary.json", summary)
    print(f"Domain MLP experts written to: {output_dir}")


if __name__ == "__main__":
    main()
