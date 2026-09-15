from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sae_pipeline.activations_collecting import _hidden_from_layer_output, get_backbone, get_transformer_layer

from domain_mapping.topographic_mlp_sae_mapping import (
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
    """Return the MLP for GPT-Neo, GPT-NeoX or Gemma-style decoders."""
    layer = get_transformer_layer(model, layer_num)
    if not hasattr(layer, "mlp"):
        raise ValueError(
            f"Transformer layer {layer_num} ({type(layer).__name__}) does not expose an MLP"
        )
    return layer.mlp


def set_mlp_module(model, layer_num: int, mlp_module) -> None:
    layer = get_transformer_layer(model, layer_num)
    if not hasattr(layer, "mlp"):
        raise ValueError(
            f"Transformer layer {layer_num} ({type(layer).__name__}) does not expose an MLP"
        )
    layer.mlp = mlp_module


def describe_mlp(mlp_module) -> Dict[str, Any]:
    """Describe projection names and the physical intermediate-neuron axis."""
    if hasattr(mlp_module, "c_fc") and hasattr(mlp_module, "c_proj"):
        return {
            "family": "gpt_neo",
            "input_projections": ["c_fc"],
            "output_projection": "c_proj",
            "activation_source": "mlp.c_proj.input_post_activation",
        }
    if hasattr(mlp_module, "dense_h_to_4h") and hasattr(mlp_module, "dense_4h_to_h"):
        return {
            "family": "gpt_neox",
            "input_projections": ["dense_h_to_4h"],
            "output_projection": "dense_4h_to_h",
            "activation_source": "mlp.dense_4h_to_h.input_post_activation",
        }
    if all(hasattr(mlp_module, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return {
            "family": "gated_mlp",
            "input_projections": ["gate_proj", "up_proj"],
            "output_projection": "down_proj",
            "activation_source": "mlp.down_proj.input_post_gating",
        }
    available = ", ".join(name for name, _ in mlp_module.named_children())
    raise ValueError(
        "Unsupported MLP projection layout. Expected GPT-Neo, GPT-NeoX, or "
        f"Gemma/Llama-style projections; available children: {available or '<none>'}"
    )


def register_mlp_post_activation_hook(mlp_module) -> Tuple[Dict[str, object], object]:
    """Capture the vector entering the MLP output projection.

    For Gemma this is the post-gating product ``act(gate_proj(x)) * up_proj(x)``;
    for GPT-Neo/NeoX it is the ordinary post-activation intermediate vector.
    """
    spec = describe_mlp(mlp_module)
    output_projection = getattr(mlp_module, str(spec["output_projection"]))

    capture: Dict[str, object] = {}

    def hook(_module, inputs) -> None:
        if not inputs:
            raise ValueError("c_proj forward pre-hook received no inputs")
        capture["activations"] = inputs[0].detach()

    handle = output_projection.register_forward_pre_hook(hook)
    return capture, handle


def get_sae_width(sae) -> int:
    width = getattr(sae, "d_sae", None)
    if width is None:
        width = getattr(getattr(sae, "cfg", None), "d_sae", None)
    if width is None:
        raise ValueError("Loaded SAE does not expose d_sae")
    return int(width)


def get_sae_input_dim(sae) -> int:
    d_in = getattr(sae, "d_model", None)
    if d_in is None:
        d_in = getattr(getattr(sae, "cfg", None), "d_in", None)
    if d_in is None:
        raise ValueError("Loaded SAE does not expose d_model/d_in")
    return int(d_in)


def encode_sae_features(sae, activations):
    """Normalize the incompatible TopKSAE and SAELens inference APIs."""
    if callable(getattr(sae, "encode", None)):
        encoded = sae.encode(activations)
        return encoded[0] if isinstance(encoded, tuple) else encoded
    output = sae(activations)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("Local SAE forward pass must return (reconstruction, feature_acts)")
    return output[1]


def load_sae_lens(release: str, sae_id: str, device):
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise ImportError("SAELens is required for --sae-release/--sae-id") from exc
    if release == "gemma-scope-2-270m-pt-resid_post":
        release = "gemma-scope-2-270m-pt-res"
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(device))
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval()


def infer_sae_lens_source(domains_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer Gemma Scope identifiers recorded by semantic-domain triage."""
    with domains_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("matrix_metadata", {}).get("source_format", {})
    if not isinstance(summary, dict):
        return None, None
    release = summary.get("sae_release")
    sae_id = summary.get("sae_id")
    return (
        str(release) if release else None,
        str(sae_id) if sae_id else None,
    )


def validate_downstream_approval(domains_path: Path, allow_unapproved: bool) -> None:
    with domains_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if bool(payload.get("downstream_approved", False)):
        return
    message = (
        f"Semantic domains in {domains_path} are not approved for downstream use. "
        "Inspect domain_report.md and validation contexts, then rerun triage with "
        "--approve-domains-for-downstream."
    )
    if allow_unapproved:
        print(f"WARNING: {message} Proceeding with a diagnostic mapping only.")
        return
    raise ValueError(message + " Use --allow-unapproved-domains only for diagnostics.")


def validate_sae_resid_post_compatibility(sae, model, layer_num: int) -> None:
    hidden_size = int(model.config.hidden_size)
    if get_sae_input_dim(sae) != hidden_size:
        raise ValueError(
            f"SAE input dimension {get_sae_input_dim(sae)} does not match model "
            f"hidden_size={hidden_size}"
        )
    cfg = getattr(sae, "cfg", None)
    metadata = getattr(cfg, "metadata", None)
    hook_name = getattr(metadata, "hook_name", None)
    if hook_name:
        expected = f"blocks.{layer_num}.hook_resid_post"
        if str(hook_name) != expected:
            raise ValueError(
                f"SAE hook {hook_name!r} does not match requested resid_post layer {layer_num} "
                f"({expected!r})"
            )


def collect_domain_mlp_activations(
    domain: DomainSpec,
    samples_paths: Sequence[Path],
    model,
    tokenizer,
    sae,
    device,
    layer_num: int,
    batch_size: int,
    max_length: int,
    max_texts: Optional[int],
    max_tokens: Optional[int],
    cluster_aggregation: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], int]:
    import torch
    from tqdm import tqdm

    mlp_module = get_mlp_module(model, layer_num)
    layer_module = get_transformer_layer(model, layer_num)
    backbone = get_backbone(model)
    mlp_spec = describe_mlp(mlp_module)
    capture, handle = register_mlp_post_activation_hook(mlp_module)
    layer_capture: Dict[str, object] = {}

    def layer_hook(_module, _inputs, output) -> None:
        layer_capture["activations"] = _hidden_from_layer_output(output).detach()

    layer_handle = layer_module.register_forward_hook(layer_hook)

    mlp_batches = []
    cluster_batches = []
    token_id_batches = []
    sample_idx_batches = []
    position_batches = []

    feature_ids = torch.tensor(domain.feature_ids, dtype=torch.long, device=device)
    samples = []
    for path in samples_paths:
        for _, text, payload in iter_domain_samples(path, max_texts=max_texts):
            samples.append((len(samples), text, payload))
    # Every domain must see the same pooled token subset. Domain-dependent
    # shuffling made cross-domain scores incomparable when max_tokens truncated
    # the corpus.
    random.Random(seed).shuffle(samples)
    collected_tokens = 0

    model.eval()
    sae.eval()
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(samples), batch_size), desc=f"Domain {domain.domain_id} MLP"):
                batch_samples = samples[start : start + batch_size]
                sample_indices = [sample_idx for sample_idx, _, _ in batch_samples]
                batch_domain_ids = [
                    int(payload.get("domain_id", -1))
                    for _, _, payload in batch_samples
                ]
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
                layer_capture.pop("activations", None)
                backbone(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                if "activations" not in capture:
                    raise RuntimeError("MLP hook did not capture activations during model forward pass")
                if "activations" not in layer_capture:
                    raise RuntimeError("Layer hook did not capture resid_post activations")

                attention_mask = encoded["attention_mask"].bool()
                mlp_activations = capture["activations"].float()
                layer_activations = layer_capture["activations"].float()

                flat_mlp_activations = mlp_activations[attention_mask]
                flat_layer_activations = layer_activations[attention_mask]

                sparse_acts = encode_sae_features(sae, flat_layer_activations)
                cluster_activations = aggregate_cluster_activations(
                    sparse_acts=sparse_acts,
                    feature_ids=feature_ids,
                    aggregation=cluster_aggregation,
                )

                sample_ids, positions = make_batch_positions(encoded["input_ids"], sample_indices)
                remaining = None if max_tokens is None else max_tokens - collected_tokens
                if remaining is not None and remaining <= 0:
                    break
                take = flat_mlp_activations.shape[0] if remaining is None else min(
                    int(remaining), int(flat_mlp_activations.shape[0])
                )
                flat_mlp_activations = flat_mlp_activations[:take]
                cluster_activations = cluster_activations[:take]
                flat_token_ids = encoded["input_ids"][attention_mask][:take]
                flat_sample_ids = sample_ids[attention_mask][:take]
                flat_positions = positions[attention_mask][:take]
                mlp_batches.append(flat_mlp_activations.detach().cpu().numpy().astype(np.float32))
                cluster_batches.append(cluster_activations.detach().cpu().numpy().astype(np.float32))
                token_id_batches.append(flat_token_ids.detach().cpu().numpy().astype(np.int64))
                sample_idx_batches.append(flat_sample_ids.detach().cpu().numpy().astype(np.int64))
                position_batches.append(flat_positions.detach().cpu().numpy().astype(np.int64))
                collected_tokens += int(take)
                if max_tokens is not None and collected_tokens >= max_tokens:
                    break
    finally:
        handle.remove()
        layer_handle.remove()

    if not mlp_batches:
        raise ValueError(
            "No usable mapping samples found in: "
            + ", ".join(str(path) for path in samples_paths)
        )

    metadata = {
        "token_ids": np.concatenate(token_id_batches, axis=0),
        "sample_indices": np.concatenate(sample_idx_batches, axis=0),
        "positions": np.concatenate(position_batches, axis=0),
    }
    return (
        np.concatenate(mlp_batches, axis=0),
        np.concatenate(cluster_batches, axis=0),
        metadata,
        int(len(np.unique(metadata["sample_indices"]))),
    )


def collect_pooled_domain_mlp_activations(
    domains: Sequence[DomainSpec],
    samples_paths: Sequence[Path],
    model,
    tokenizer,
    sae,
    device,
    layer_num: int,
    batch_size: int,
    max_length: int,
    max_texts: Optional[int],
    max_tokens: Optional[int],
    cluster_aggregation: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], int]:
    """Collect one shared MLP matrix and one SAE-signal column per domain.

    The pooled-domain design intentionally exposes every domain score to the
    same tokens.  Computing the frozen backbone once avoids repeated forwards
    and six identical copies of the physical MLP activation matrix.
    """
    import torch
    from tqdm import tqdm

    if not domains:
        raise ValueError("At least one domain is required for pooled mapping")
    mlp_module = get_mlp_module(model, layer_num)
    layer_module = get_transformer_layer(model, layer_num)
    backbone = get_backbone(model)
    capture, handle = register_mlp_post_activation_hook(mlp_module)
    layer_capture: Dict[str, object] = {}

    def layer_hook(_module, _inputs, output) -> None:
        layer_capture["activations"] = _hidden_from_layer_output(output).detach()

    layer_handle = layer_module.register_forward_hook(layer_hook)
    feature_ids_by_domain = [
        torch.tensor(domain.feature_ids, dtype=torch.long, device=device)
        for domain in domains
    ]
    mlp_batches = []
    cluster_matrix_batches = []
    token_id_batches = []
    sample_idx_batches = []
    position_batches = []
    source_domain_id_batches = []
    samples = []
    for path in samples_paths:
        for _, text, payload in iter_domain_samples(path, max_texts=max_texts):
            samples.append((len(samples), text, payload))
    random.Random(seed).shuffle(samples)
    collected_tokens = 0

    model.eval()
    sae.eval()
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(samples), batch_size), desc="Pooled domain MLP"):
                batch_samples = samples[start : start + batch_size]
                sample_indices = [sample_idx for sample_idx, _, _ in batch_samples]
                batch_domain_ids = [
                    int(payload.get("domain_id", -1))
                    for _, _, payload in batch_samples
                ]
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
                layer_capture.pop("activations", None)
                backbone(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
                if "activations" not in capture or "activations" not in layer_capture:
                    raise RuntimeError("MLP or resid_post hook did not capture pooled activations")

                attention_mask = encoded["attention_mask"].bool()
                flat_mlp = capture["activations"].float()[attention_mask]
                flat_resid = layer_capture["activations"].float()[attention_mask]
                sparse_acts = encode_sae_features(sae, flat_resid)
                cluster_matrix = torch.stack(
                    [
                        aggregate_cluster_activations(
                            sparse_acts=sparse_acts,
                            feature_ids=feature_ids,
                            aggregation=cluster_aggregation,
                        )
                        for feature_ids in feature_ids_by_domain
                    ],
                    dim=-1,
                )

                sample_ids, positions = make_batch_positions(
                    encoded["input_ids"], sample_indices
                )
                source_domain_ids = torch.tensor(
                    batch_domain_ids, dtype=torch.long, device=device
                ).unsqueeze(1).expand_as(encoded["input_ids"])
                remaining = None if max_tokens is None else max_tokens - collected_tokens
                if remaining is not None and remaining <= 0:
                    break
                take = flat_mlp.shape[0] if remaining is None else min(
                    int(remaining), int(flat_mlp.shape[0])
                )
                mlp_batches.append(flat_mlp[:take].cpu().numpy().astype(np.float32))
                cluster_matrix_batches.append(
                    cluster_matrix[:take].cpu().numpy().astype(np.float32)
                )
                token_id_batches.append(
                    encoded["input_ids"][attention_mask][:take].cpu().numpy().astype(np.int64)
                )
                sample_idx_batches.append(
                    sample_ids[attention_mask][:take].cpu().numpy().astype(np.int64)
                )
                position_batches.append(
                    positions[attention_mask][:take].cpu().numpy().astype(np.int64)
                )
                source_domain_id_batches.append(
                    source_domain_ids[attention_mask][:take].cpu().numpy().astype(np.int64)
                )
                collected_tokens += int(take)
                if max_tokens is not None and collected_tokens >= max_tokens:
                    break
    finally:
        handle.remove()
        layer_handle.remove()

    if not mlp_batches:
        raise ValueError(
            "No usable pooled mapping samples found in: "
            + ", ".join(str(path) for path in samples_paths)
        )
    metadata = {
        "token_ids": np.concatenate(token_id_batches, axis=0),
        "sample_indices": np.concatenate(sample_idx_batches, axis=0),
        "positions": np.concatenate(position_batches, axis=0),
        "source_domain_ids": np.concatenate(source_domain_id_batches, axis=0),
    }
    return (
        np.concatenate(mlp_batches, axis=0),
        np.concatenate(cluster_matrix_batches, axis=0),
        metadata,
        int(len(np.unique(metadata["sample_indices"]))),
    )


def save_domain_mlp_activations(
    output_path: Path,
    domain: DomainSpec,
    mlp_activations: np.ndarray,
    cluster_activations: np.ndarray,
    metadata: Dict[str, np.ndarray],
    layer_num: int,
    cluster_aggregation: str,
    activation_source: str,
    mlp_family: str,
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
        activation_source=np.asarray(activation_source),
        mlp_family=np.asarray(mlp_family),
        n_inner=np.asarray(mlp_activations.shape[1], dtype=np.int64),
    )


def save_pooled_domain_mlp_activations(
    output_path: Path,
    domains: Sequence[DomainSpec],
    mlp_activations: np.ndarray,
    cluster_activation_matrix: np.ndarray,
    metadata: Dict[str, np.ndarray],
    layer_num: int,
    cluster_aggregation: str,
    activation_source: str,
    mlp_family: str,
    compressed: bool,
) -> None:
    if cluster_activation_matrix.shape != (mlp_activations.shape[0], len(domains)):
        raise ValueError("Pooled cluster matrix must have shape [tokens, domains]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(
        output_path,
        mlp_activations=mlp_activations,
        sae_cluster_activations=cluster_activation_matrix,
        token_ids=metadata["token_ids"],
        sample_indices=metadata["sample_indices"],
        positions=metadata["positions"],
        source_domain_ids=metadata["source_domain_ids"],
        domain_ids=np.asarray([domain.domain_id for domain in domains], dtype=np.int64),
        layer_num=np.asarray(layer_num, dtype=np.int64),
        cluster_aggregation=np.asarray(cluster_aggregation),
        activation_source=np.asarray(activation_source),
        mlp_family=np.asarray(mlp_family),
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
    parser.add_argument("--sae-release", default=None, help="SAELens release, e.g. Gemma Scope 2.")
    parser.add_argument("--sae-id", default=None, help="SAELens SAE id; requires --sae-release.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--domain-ids", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-texts-per-domain", type=int, default=None)
    parser.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=100_000,
        help="Bound in-memory token activations per domain; samples are shuffled first.",
    )
    parser.add_argument(
        "--mapping-corpus",
        choices=["pooled-domains", "target-domain"],
        default="pooled-domains",
        help=(
            "Use the same pooled domain corpus for every domain signal (recommended), "
            "or only already-selected texts from the target domain."
        ),
    )
    parser.add_argument("--cluster-aggregation", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--metrics", nargs="+", choices=["pearson", "mutual_info"], default=["pearson"])
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
    parser.add_argument("--allow-unapproved-domains", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if min(args.batch_size, args.max_length, args.mi_n_neighbors, args.top_n) < 1:
        raise ValueError("Batch, length, neighbour, and top-N arguments must be positive")
    if args.max_texts_per_domain is not None and args.max_texts_per_domain < 1:
        raise ValueError("--max-texts-per-domain must be positive")
    if args.max_tokens_per_domain is not None and args.max_tokens_per_domain < 2:
        raise ValueError("--max-tokens-per-domain must be at least 2")
    if not 0.0 <= args.min_cluster_nonzero_fraction <= 1.0:
        raise ValueError("--min-cluster-nonzero-fraction must be between 0 and 1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common.utils import find_device

    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    domains_path = domain_dir / "domains.json"
    validation_dir = domain_dir / "domain_validation"
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "domain_mlp_mapping"

    if not domains_path.exists():
        raise FileNotFoundError(f"domains.json not found: {domains_path}")
    if not validation_dir.exists():
        raise FileNotFoundError(f"Domain validation directory not found: {validation_dir}")
    validate_downstream_approval(domains_path, args.allow_unapproved_domains)
    device = find_device()
    print(f"Using device: {device}")
    print(f"Domain directory: {domain_dir}")

    inferred_release, inferred_sae_id = infer_sae_lens_source(domains_path)
    sae_release = args.sae_release or inferred_release
    sae_id = args.sae_id or inferred_sae_id
    checkpoint_path: Optional[Path] = Path(args.checkpoint) if args.checkpoint else None
    if bool(sae_release) != bool(sae_id):
        raise ValueError("--sae-release and --sae-id must be provided together")
    if checkpoint_path is not None and sae_release is not None:
        raise ValueError("Choose either --checkpoint or --sae-release/--sae-id, not both")
    if checkpoint_path is None and sae_release is None:
        default_path = Path(default_checkpoint_path(args.data_path, args.layer_num))
        if not default_path.exists():
            raise FileNotFoundError(
                "No SAE source found. Pass --checkpoint or --sae-release/--sae-id; "
                f"the local default does not exist: {default_path}"
            )
        checkpoint_path = default_path

    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
        print(f"Loading SAE checkpoint: {checkpoint_path}")
        sae = load_sae_checkpoint(str(checkpoint_path), device)
        sae_source = str(checkpoint_path)
    else:
        print(f"Loading SAELens SAE: release={sae_release}, sae_id={sae_id}")
        sae = load_sae_lens(str(sae_release), str(sae_id), device)
        sae_source = f"{sae_release}:{sae_id}"

    domains = load_domains(domains_path, selected_domain_ids=args.domain_ids)
    max_feature_id = max(max(domain.feature_ids) for domain in domains)
    if max_feature_id >= get_sae_width(sae):
        raise ValueError(f"Domain feature id {max_feature_id} exceeds SAE size {get_sae_width(sae)}")

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    validate_sae_resid_post_compatibility(sae, model, args.layer_num)
    mlp_spec = describe_mlp(get_mlp_module(model, args.layer_num))

    all_rows: List[Dict[str, object]] = []
    summaries = []
    validation_paths = {
        domain.domain_id: validation_dir / f"domain_{domain.domain_id}.jsonl"
        for domain in domains
    }
    for domain_id, path in validation_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Validation set for domain {domain_id} not found: {path}")

    if args.mapping_corpus == "pooled-domains":
        samples_paths = list(validation_paths.values())
        (
            mlp_activations,
            cluster_activation_matrix,
            metadata,
            n_texts,
        ) = collect_pooled_domain_mlp_activations(
            domains=domains,
            samples_paths=samples_paths,
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            device=device,
            layer_num=args.layer_num,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_texts=args.max_texts_per_domain,
            max_tokens=args.max_tokens_per_domain,
            cluster_aggregation=args.cluster_aggregation,
            seed=args.seed,
        )
        activation_path = output_dir / "pooled_mlp_activations.npz"
        save_pooled_domain_mlp_activations(
            output_path=activation_path,
            domains=domains,
            mlp_activations=mlp_activations,
            cluster_activation_matrix=cluster_activation_matrix,
            metadata=metadata,
            layer_num=args.layer_num,
            cluster_aggregation=args.cluster_aggregation,
            activation_source=str(mlp_spec["activation_source"]),
            mlp_family=str(mlp_spec["family"]),
            compressed=not args.no_compress,
        )
        for cluster_column, domain in enumerate(domains):
            cluster_activations = cluster_activation_matrix[:, cluster_column]
            signal_diagnostics = validate_domain_signal(
                domain=domain,
                cluster_activations=cluster_activations,
                min_nonzero_fraction=args.min_cluster_nonzero_fraction,
                allow_zero_domain_signal=args.allow_zero_domain_signal,
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
            domain_summary["cluster_column"] = int(cluster_column)
            domain_summary["n_inner"] = int(mlp_activations.shape[1])
            domain_summary["signal_diagnostics"] = signal_diagnostics
            domain_summary["mapping_corpus_paths"] = [str(path) for path in samples_paths]
            summaries.append(domain_summary)
    else:
        for domain in domains:
            samples_paths = [validation_paths[domain.domain_id]]
            mlp_activations, cluster_activations, metadata, n_texts = collect_domain_mlp_activations(
                domain=domain,
                samples_paths=samples_paths,
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                device=device,
                layer_num=args.layer_num,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_texts=args.max_texts_per_domain,
                max_tokens=args.max_tokens_per_domain,
                cluster_aggregation=args.cluster_aggregation,
                seed=args.seed,
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
                activation_source=str(mlp_spec["activation_source"]),
                mlp_family=str(mlp_spec["family"]),
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
            domain_summary["cluster_column"] = None
            domain_summary["n_inner"] = int(mlp_activations.shape[1])
            domain_summary["signal_diagnostics"] = signal_diagnostics
            domain_summary["mapping_corpus_paths"] = [str(path) for path in samples_paths]
            summaries.append(domain_summary)

    combined_correlations_path = output_dir / "all_mlp_neuron_correlations.csv"
    write_correlation_csv(combined_correlations_path, all_rows)

    summary = {
        "domain_dir": str(domain_dir),
        "domains_path": str(domains_path),
        "validation_dir": str(validation_dir),
        "sae_source": sae_source,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "layer_num": args.layer_num,
        "activation_source": mlp_spec["activation_source"],
        "mlp_family": mlp_spec["family"],
        "cluster_aggregation": args.cluster_aggregation,
        "metrics": args.metrics,
        "min_cluster_nonzero_fraction": args.min_cluster_nonzero_fraction,
        "allow_zero_domain_signal": args.allow_zero_domain_signal,
        "allow_unapproved_domains": args.allow_unapproved_domains,
        "max_tokens_per_domain": args.max_tokens_per_domain,
        "mapping_corpus": args.mapping_corpus,
        "combined_correlations_path": str(combined_correlations_path),
        "domains": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Domain MLP mapping written to: {output_dir}")


if __name__ == "__main__":
    main()
