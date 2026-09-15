from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MODEL_NAME = "roneneldan/TinyStories-1M"
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
DEFAULT_DATA_PATH = "data/tinystories_dataset"
DEFAULT_LAYER_NUM = 4


@dataclass
class DomainSpec:
    domain_id: int
    name: str
    feature_ids: List[int]
    cluster_label: Optional[int] = None


def default_checkpoint_path(data_path: str, layer_num: int) -> str:
    return str(Path(data_path) / f"models/checkpoints/topk_sae_layer_{layer_num}_best.pt")


def load_sae_checkpoint(checkpoint_path: str, device):
    import torch

    from sae_pipeline.autoencoder_training import TopKSAE

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    sae = TopKSAE(
        d_model=int(config.get("d_model", 64)),
        expansion_factor=int(config.get("expansion_factor", 64)),
        k=int(config.get("k", 8)),
    )
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def resolve_domain_dir(domain_dir_arg: Optional[str], data_path: str) -> Path:
    if domain_dir_arg:
        return Path(domain_dir_arg)

    candidates = [
        Path(data_path) / "analysis/domain_triage",
        Path("other/domain_triage"),
    ]
    for candidate in candidates:
        if (candidate / "domains.json").exists() and (candidate / "domain_validation").exists():
            return candidate
    return candidates[0]


def load_domains(domains_path: Path, selected_domain_ids: Optional[Sequence[int]]) -> List[DomainSpec]:
    with domains_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    selected = set(selected_domain_ids) if selected_domain_ids else None
    domains = []
    for domain in payload.get("domains", []):
        domain_id = int(domain["domain_id"])
        if selected is not None and domain_id not in selected:
            continue

        feature_ids = [int(feature_id) for feature_id in domain.get("feature_ids", [])]
        if not feature_ids:
            raise ValueError(f"Domain {domain_id} has no feature_ids in {domains_path}")

        domains.append(
            DomainSpec(
                domain_id=domain_id,
                name=str(domain.get("name", f"domain_{domain_id}")),
                feature_ids=feature_ids,
                cluster_label=domain.get("cluster_label"),
            )
        )

    if not domains:
        raise ValueError(f"No domains selected from {domains_path}")
    return domains


def iter_domain_samples(jsonl_path: Path, max_texts: Optional[int]) -> Iterable[Tuple[int, str, Dict]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for sample_idx, line in enumerate(f):
            if max_texts is not None and sample_idx >= max_texts:
                break
            if not line.strip():
                continue

            payload = json.loads(line)
            text = payload.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            yield sample_idx, text, payload


def pearson_by_neuron(
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    chunk_rows: int = 8192,
) -> np.ndarray:
    import numpy as np

    if mlp_activations.ndim != 2 or cluster_activations.ndim != 1:
        raise ValueError("Expected MLP activations [tokens, neurons] and domain signal [tokens]")
    if mlp_activations.shape[0] != cluster_activations.shape[0]:
        raise ValueError("MLP activations and domain signal have different token counts")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    n_rows, n_neurons = mlp_activations.shape
    if n_rows < 2:
        return np.full(n_neurons, np.nan, dtype=np.float64)

    # Accumulate sufficient statistics in float64 by bounded row chunks.  The
    # previous implementation materialised an additional float64 [N, D] copy
    # and a second centered array, which can exceed Kaggle memory for Gemma.
    sum_x = np.zeros(n_neurons, dtype=np.float64)
    sum_x2 = np.zeros(n_neurons, dtype=np.float64)
    sum_xy = np.zeros(n_neurons, dtype=np.float64)
    sum_y = 0.0
    sum_y2 = 0.0
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        x = np.asarray(mlp_activations[start:end], dtype=np.float64)
        y = np.asarray(cluster_activations[start:end], dtype=np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += np.square(x).sum(axis=0)
        sum_xy += x.T @ y
        sum_y += float(y.sum())
        sum_y2 += float(np.square(y).sum())

    numerator = sum_xy - (sum_x * sum_y / n_rows)
    x_ss = np.maximum(sum_x2 - np.square(sum_x) / n_rows, 0.0)
    y_ss = max(sum_y2 - (sum_y * sum_y / n_rows), 0.0)
    denominator = np.sqrt(x_ss * y_ss)
    return np.divide(
        numerator,
        denominator,
        out=np.full(n_neurons, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def mutual_info_by_neuron(
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    n_neighbors: int,
    seed: int,
) -> np.ndarray:
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression

    if mlp_activations.shape[0] <= 1 or np.std(cluster_activations) == 0:
        return np.full(mlp_activations.shape[1], np.nan, dtype=np.float64)

    effective_neighbors = max(1, min(n_neighbors, mlp_activations.shape[0] - 1))
    return mutual_info_regression(
        mlp_activations,
        cluster_activations,
        discrete_features=False,
        n_neighbors=effective_neighbors,
        random_state=seed,
    )


def aggregate_cluster_activations(
    sparse_acts: torch.Tensor,
    feature_ids: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    cluster_acts = sparse_acts.index_select(dim=-1, index=feature_ids)
    if aggregation == "sum":
        return cluster_acts.sum(dim=-1)
    if aggregation == "mean":
        return cluster_acts.mean(dim=-1)
    if aggregation == "max":
        return cluster_acts.max(dim=-1).values
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def make_batch_positions(input_ids: torch.Tensor, sample_indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch

    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len)
    sample_ids = torch.tensor(sample_indices, device=input_ids.device).unsqueeze(1).expand(batch_size, seq_len)
    return sample_ids, positions


def collect_domain_activations(
    domain: DomainSpec,
    samples_path: Path,
    model,
    tokenizer,
    sae,
    device: torch.device,
    layer_num: int,
    batch_size: int,
    max_length: int,
    max_texts: Optional[int],
    cluster_aggregation: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], int]:
    import numpy as np
    import torch
    from tqdm import tqdm

    mlp_batches = []
    cluster_batches = []
    token_id_batches = []
    sample_idx_batches = []
    position_batches = []

    feature_ids = torch.tensor(domain.feature_ids, dtype=torch.long, device=device)
    samples = list(iter_domain_samples(samples_path, max_texts=max_texts))

    model.eval()
    sae.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(samples), batch_size), desc=f"Domain {domain.domain_id}"):
            batch_samples = samples[start : start + batch_size]
            sample_indices = [sample_idx for sample_idx, _, _ in batch_samples]
            texts = [text for _, text, _ in batch_samples]

            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            if layer_num + 1 >= len(outputs.hidden_states):
                raise IndexError(
                    f"Layer {layer_num} is unavailable: model returned "
                    f"{len(outputs.hidden_states) - 1} hidden layers."
                )

            # This is the same activation source used to train the SAE in this project.
            layer_activations = outputs.hidden_states[layer_num + 1].float()
            attention_mask = encoded["attention_mask"].bool()
            flat_layer_activations = layer_activations[attention_mask]

            _, sparse_acts = sae(flat_layer_activations)
            cluster_activations = aggregate_cluster_activations(
                sparse_acts=sparse_acts,
                feature_ids=feature_ids,
                aggregation=cluster_aggregation,
            )

            sample_ids, positions = make_batch_positions(encoded["input_ids"], sample_indices)
            mlp_batches.append(flat_layer_activations.detach().cpu().numpy().astype(np.float32))
            cluster_batches.append(cluster_activations.detach().cpu().numpy().astype(np.float32))
            token_id_batches.append(encoded["input_ids"][attention_mask].detach().cpu().numpy().astype(np.int64))
            sample_idx_batches.append(sample_ids[attention_mask].detach().cpu().numpy().astype(np.int64))
            position_batches.append(positions[attention_mask].detach().cpu().numpy().astype(np.int64))

    if not mlp_batches:
        raise ValueError(f"No usable validation samples found in {samples_path}")

    metadata = {
        "token_ids": np.concatenate(token_id_batches, axis=0),
        "sample_indices": np.concatenate(sample_idx_batches, axis=0),
        "positions": np.concatenate(position_batches, axis=0),
    }
    return (
        np.concatenate(mlp_batches, axis=0),
        np.concatenate(cluster_batches, axis=0),
        metadata,
        len(samples),
    )


def save_domain_activations(
    output_path: Path,
    domain: DomainSpec,
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    metadata: Dict[str, np.ndarray],
    layer_num: int,
    cluster_aggregation: str,
    compressed: bool,
) -> None:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(
        output_path,
        mlp_activations=mlp_activations,
        sae_cluster_activations=cluster_activations,
        token_ids=metadata["token_ids"],
        sample_indices=metadata["sample_indices"],
        positions=metadata["positions"],
        feature_ids=np.asarray(domain.feature_ids, dtype=np.int64),
        domain_id=np.asarray(domain.domain_id, dtype=np.int64),
        layer_num=np.asarray(layer_num, dtype=np.int64),
        cluster_aggregation=np.asarray(cluster_aggregation),
    )


def build_correlation_rows(
    domain: DomainSpec,
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    metrics: Sequence[str],
    n_neighbors: int,
    seed: int,
) -> List[Dict[str, object]]:
    pearson = pearson_by_neuron(mlp_activations, cluster_activations) if "pearson" in metrics else None
    mutual_info = (
        mutual_info_by_neuron(mlp_activations, cluster_activations, n_neighbors=n_neighbors, seed=seed)
        if "mutual_info" in metrics
        else None
    )

    rows = []
    neuron_means = mlp_activations.mean(axis=0)
    neuron_stds = mlp_activations.std(axis=0)
    for neuron_id in range(mlp_activations.shape[1]):
        rows.append(
            {
                "domain_id": domain.domain_id,
                "domain_name": domain.name,
                "cluster_label": "" if domain.cluster_label is None else domain.cluster_label,
                "n_cluster_features": len(domain.feature_ids),
                "n_tokens": int(mlp_activations.shape[0]),
                "neuron_id": neuron_id,
                "pearson_r": "" if pearson is None else float(pearson[neuron_id]),
                "abs_pearson_r": "" if pearson is None else float(abs(pearson[neuron_id])),
                "mutual_info": "" if mutual_info is None else float(mutual_info[neuron_id]),
                "mlp_mean": float(neuron_means[neuron_id]),
                "mlp_std": float(neuron_stds[neuron_id]),
                "cluster_mean": float(cluster_activations.mean()),
                "cluster_std": float(cluster_activations.std()),
            }
        )
    return rows


def write_correlation_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain_id",
        "domain_name",
        "cluster_label",
        "n_cluster_features",
        "n_tokens",
        "neuron_id",
        "pearson_r",
        "abs_pearson_r",
        "mutual_info",
        "mlp_mean",
        "mlp_std",
        "cluster_mean",
        "cluster_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_key_for_metric(row: Dict[str, object], metric: str) -> float:
    value = row.get(metric, "")
    if value == "":
        return float("-inf")
    return abs(float(value)) if metric == "pearson_r" else float(value)


def summarize_domain(
    domain: DomainSpec,
    n_texts: int,
    activation_path: Path,
    correlation_path: Path,
    rows: Sequence[Dict[str, object]],
    cluster_activations: np.ndarray,
    top_n: int,
) -> Dict[str, object]:
    import numpy as np

    top_by_pearson = sorted(rows, key=lambda row: sort_key_for_metric(row, "pearson_r"), reverse=True)[:top_n]
    top_by_mi = sorted(rows, key=lambda row: sort_key_for_metric(row, "mutual_info"), reverse=True)[:top_n]

    return {
        "domain_id": domain.domain_id,
        "domain_name": domain.name,
        "cluster_label": domain.cluster_label,
        "n_texts": n_texts,
        "n_tokens": int(cluster_activations.shape[0]),
        "n_cluster_features": len(domain.feature_ids),
        "cluster_activation_mean": float(cluster_activations.mean()),
        "cluster_activation_std": float(cluster_activations.std()),
        "cluster_activation_max": float(cluster_activations.max()),
        "cluster_activation_nonzero_fraction": float(np.mean(cluster_activations > 0)),
        "activations_path": str(activation_path),
        "correlations_path": str(correlation_path),
        "top_neurons_by_abs_pearson": [
            {
                "neuron_id": int(row["neuron_id"]),
                "pearson_r": row["pearson_r"],
                "abs_pearson_r": row["abs_pearson_r"],
            }
            for row in top_by_pearson
            if row.get("pearson_r", "") != ""
        ],
        "top_neurons_by_mutual_info": [
            {
                "neuron_id": int(row["neuron_id"]),
                "mutual_info": row["mutual_info"],
            }
            for row in top_by_mi
            if row.get("mutual_info", "") != ""
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map topographic relations between physical layer-neuron activations "
            "and aggregated SAE domain-cluster activations."
        )
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--domain-dir", default=None, help="Directory containing domains.json and domain_validation/.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--domain-ids", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-texts-per-domain", type=int, default=None)
    parser.add_argument("--cluster-aggregation", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--metrics", nargs="+", choices=["pearson", "mutual_info"], default=["pearson", "mutual_info"])
    parser.add_argument("--mi-n-neighbors", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-compress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common.utils import find_device

    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    domains_path = domain_dir / "domains.json"
    validation_dir = domain_dir / "domain_validation"
    checkpoint_path = Path(args.checkpoint or default_checkpoint_path(args.data_path, args.layer_num))
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "topographic_mlp_sae_mapping"

    if not domains_path.exists():
        raise FileNotFoundError(f"domains.json not found: {domains_path}")
    if not validation_dir.exists():
        raise FileNotFoundError(f"Domain validation directory not found: {validation_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAE checkpoint not found: {checkpoint_path}. Pass --checkpoint if it lives elsewhere."
        )

    device = find_device()
    print(f"Using device: {device}")
    print(f"Domain directory: {domain_dir}")
    print(f"Loading SAE checkpoint: {checkpoint_path}")
    sae = load_sae_checkpoint(str(checkpoint_path), device)

    domains = load_domains(domains_path, selected_domain_ids=args.domain_ids)
    max_feature_id = max(max(domain.feature_ids) for domain in domains)
    if max_feature_id >= sae.d_sae:
        raise ValueError(f"Domain feature id {max_feature_id} exceeds SAE size {sae.d_sae}")

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    all_rows: List[Dict[str, object]] = []
    summaries = []

    for domain in domains:
        samples_path = validation_dir / f"domain_{domain.domain_id}.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"Validation set for domain {domain.domain_id} not found: {samples_path}")

        mlp_activations, cluster_activations, metadata, n_texts = collect_domain_activations(
            domain=domain,
            samples_path=samples_path,
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            device=device,
            layer_num=args.layer_num,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_texts=args.max_texts_per_domain,
            cluster_aggregation=args.cluster_aggregation,
        )

        activation_path = output_dir / f"domain_{domain.domain_id}_activations.npz"
        save_domain_activations(
            output_path=activation_path,
            domain=domain,
            mlp_activations=mlp_activations,
            cluster_activations=cluster_activations,
            metadata=metadata,
            layer_num=args.layer_num,
            cluster_aggregation=args.cluster_aggregation,
            compressed=not args.no_compress,
        )

        rows = build_correlation_rows(
            domain=domain,
            mlp_activations=mlp_activations,
            cluster_activations=cluster_activations,
            metrics=args.metrics,
            n_neighbors=args.mi_n_neighbors,
            seed=args.seed,
        )
        correlation_path = output_dir / f"domain_{domain.domain_id}_neuron_correlations.csv"
        write_correlation_csv(correlation_path, rows)
        all_rows.extend(rows)

        summaries.append(
            summarize_domain(
                domain=domain,
                n_texts=n_texts,
                activation_path=activation_path,
                correlation_path=correlation_path,
                rows=rows,
                cluster_activations=cluster_activations,
                top_n=args.top_n,
            )
        )

    combined_correlations_path = output_dir / "all_neuron_correlations.csv"
    write_correlation_csv(combined_correlations_path, all_rows)

    summary = {
        "domain_dir": str(domain_dir),
        "domains_path": str(domains_path),
        "validation_dir": str(validation_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "layer_num": args.layer_num,
        "activation_source": "outputs.hidden_states[layer_num + 1]",
        "cluster_aggregation": args.cluster_aggregation,
        "metrics": args.metrics,
        "combined_correlations_path": str(combined_correlations_path),
        "domains": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Topographic mapping written to: {output_dir}")


if __name__ == "__main__":
    main()
