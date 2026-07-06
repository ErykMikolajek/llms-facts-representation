from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from topographic_mlp_sae_mapping import (
    DEFAULT_DATA_PATH,
    DEFAULT_LAYER_NUM,
    MODEL_NAME,
    TOKENIZER_NAME,
    DomainSpec,
    aggregate_cluster_activations,
    build_correlation_rows,
    default_checkpoint_path,
    iter_domain_samples,
    load_domains,
    load_sae_checkpoint,
    make_batch_positions,
    resolve_domain_dir,
    summarize_domain,
    write_correlation_csv,
)


def get_mlp_module(model, layer_num: int):
    try:
        return model.transformer.h[layer_num].mlp
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            "Could not locate GPT-Neo MLP module at "
            f"model.transformer.h[{layer_num}].mlp"
        ) from exc


def register_mlp_post_activation_hook(mlp_module) -> Tuple[Dict[str, object], object]:
    """Capture post-activation MLP neurons via the input to c_proj."""
    if not hasattr(mlp_module, "c_proj"):
        available = ", ".join(name for name, _ in mlp_module.named_children())
        raise ValueError(
            "MLP module does not expose c_proj; cannot capture post-activation neurons. "
            f"Available children: {available or '<none>'}"
        )

    capture: Dict[str, object] = {}

    def hook(_module, inputs) -> None:
        if not inputs:
            raise ValueError("c_proj forward pre-hook received no inputs")
        capture["activations"] = inputs[0].detach()

    handle = mlp_module.c_proj.register_forward_pre_hook(hook)
    return capture, handle


def collect_domain_mlp_activations(
    domain: DomainSpec,
    samples_path: Path,
    model,
    tokenizer,
    sae,
    device,
    layer_num: int,
    batch_size: int,
    max_length: int,
    max_texts: Optional[int],
    cluster_aggregation: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], int]:
    import torch
    from tqdm import tqdm

    mlp_module = get_mlp_module(model, layer_num)
    capture, handle = register_mlp_post_activation_hook(mlp_module)

    mlp_batches = []
    cluster_batches = []
    token_id_batches = []
    sample_idx_batches = []
    position_batches = []

    feature_ids = torch.tensor(domain.feature_ids, dtype=torch.long, device=device)
    samples = list(iter_domain_samples(samples_path, max_texts=max_texts))

    model.eval()
    sae.eval()
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(samples), batch_size), desc=f"Domain {domain.domain_id} MLP"):
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

                capture.pop("activations", None)
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                if "activations" not in capture:
                    raise RuntimeError("MLP hook did not capture activations during model forward pass")
                if layer_num + 1 >= len(outputs.hidden_states):
                    raise IndexError(
                        f"Layer {layer_num} is unavailable: model returned "
                        f"{len(outputs.hidden_states) - 1} hidden layers."
                    )

                attention_mask = encoded["attention_mask"].bool()
                mlp_activations = capture["activations"].float()
                layer_activations = outputs.hidden_states[layer_num + 1].float()

                flat_mlp_activations = mlp_activations[attention_mask]
                flat_layer_activations = layer_activations[attention_mask]

                _, sparse_acts = sae(flat_layer_activations)
                cluster_activations = aggregate_cluster_activations(
                    sparse_acts=sparse_acts,
                    feature_ids=feature_ids,
                    aggregation=cluster_aggregation,
                )

                sample_ids, positions = make_batch_positions(encoded["input_ids"], sample_indices)
                mlp_batches.append(flat_mlp_activations.detach().cpu().numpy().astype(np.float32))
                cluster_batches.append(cluster_activations.detach().cpu().numpy().astype(np.float32))
                token_id_batches.append(encoded["input_ids"][attention_mask].detach().cpu().numpy().astype(np.int64))
                sample_idx_batches.append(sample_ids[attention_mask].detach().cpu().numpy().astype(np.int64))
                position_batches.append(positions[attention_mask].detach().cpu().numpy().astype(np.int64))
    finally:
        handle.remove()

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


def save_domain_mlp_activations(
    output_path: Path,
    domain: DomainSpec,
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    metadata: Dict[str, np.ndarray],
    layer_num: int,
    cluster_aggregation: str,
    compressed: bool,
) -> None:
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
        activation_source=np.asarray("mlp.c_proj.input_post_activation"),
        n_inner=np.asarray(mlp_activations.shape[1], dtype=np.int64),
    )


def summarize_domain_signal(cluster_activations: np.ndarray) -> Dict[str, object]:
    return {
        "cluster_activation_mean": float(cluster_activations.mean()),
        "cluster_activation_std": float(cluster_activations.std()),
        "cluster_activation_max": float(cluster_activations.max()),
        "cluster_activation_nonzero_fraction": float(np.mean(cluster_activations > 0)),
    }


def validate_domain_signal(
    domain: DomainSpec,
    cluster_activations: np.ndarray,
    min_nonzero_fraction: float,
    allow_zero_domain_signal: bool,
) -> Dict[str, object]:
    diagnostics = summarize_domain_signal(cluster_activations)
    if diagnostics["cluster_activation_nonzero_fraction"] >= min_nonzero_fraction:
        return diagnostics

    message = (
        f"Domain {domain.domain_id} ({domain.name}) has too little SAE-cluster signal: "
        f"nonzero_fraction={diagnostics['cluster_activation_nonzero_fraction']:.6f}, "
        f"required={min_nonzero_fraction:.6f}. Re-run domain triage or pass "
        "--allow-zero-domain-signal for a diagnostic mapping artifact."
    )
    if allow_zero_domain_signal:
        print(f"WARNING: {message}")
        diagnostics["warning"] = message
        return diagnostics
    raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map real MLP post-activation neurons to aggregated SAE domain-cluster "
            "activations for domain-wise pruning."
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
    parser.add_argument("--min-cluster-nonzero-fraction", type=float, default=1e-6)
    parser.add_argument(
        "--allow-zero-domain-signal",
        action="store_true",
        help="Write diagnostic mapping artifacts even when the SAE domain signal is empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils import find_device

    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    domains_path = domain_dir / "domains.json"
    validation_dir = domain_dir / "domain_validation"
    checkpoint_path = Path(args.checkpoint or default_checkpoint_path(args.data_path, args.layer_num))
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "domain_mlp_mapping"

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

        mlp_activations, cluster_activations, metadata, n_texts = collect_domain_mlp_activations(
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
        signal_diagnostics = validate_domain_signal(
            domain=domain,
            cluster_activations=cluster_activations,
            min_nonzero_fraction=args.min_cluster_nonzero_fraction,
            allow_zero_domain_signal=args.allow_zero_domain_signal,
        )

        activation_path = output_dir / f"domain_{domain.domain_id}_mlp_activations.npz"
        save_domain_mlp_activations(
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
        correlation_path = output_dir / f"domain_{domain.domain_id}_mlp_neuron_correlations.csv"
        write_correlation_csv(correlation_path, rows)
        all_rows.extend(rows)

        domain_summary = summarize_domain(
            domain=domain,
            n_texts=n_texts,
            activation_path=activation_path,
            correlation_path=correlation_path,
            rows=rows,
            cluster_activations=cluster_activations,
            top_n=args.top_n,
        )
        domain_summary["n_inner"] = int(mlp_activations.shape[1])
        domain_summary["signal_diagnostics"] = signal_diagnostics
        summaries.append(domain_summary)

    combined_correlations_path = output_dir / "all_mlp_neuron_correlations.csv"
    write_correlation_csv(combined_correlations_path, all_rows)

    summary = {
        "domain_dir": str(domain_dir),
        "domains_path": str(domains_path),
        "validation_dir": str(validation_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "layer_num": args.layer_num,
        "activation_source": "mlp.c_proj.input_post_activation",
        "cluster_aggregation": args.cluster_aggregation,
        "metrics": args.metrics,
        "min_cluster_nonzero_fraction": args.min_cluster_nonzero_fraction,
        "allow_zero_domain_signal": args.allow_zero_domain_signal,
        "combined_correlations_path": str(combined_correlations_path),
        "domains": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Domain MLP mapping written to: {output_dir}")


if __name__ == "__main__":
    main()
