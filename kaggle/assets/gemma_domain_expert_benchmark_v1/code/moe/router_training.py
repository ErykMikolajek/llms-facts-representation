from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_pipeline.activations_collecting import get_backbone
from domain_mapping.domain_mlp_activation_mapping import get_mlp_module
from domain_mapping.topographic_mlp_sae_mapping import (
    DEFAULT_DATA_PATH,
    DEFAULT_LAYER_NUM,
    MODEL_NAME,
    TOKENIZER_NAME,
    iter_domain_samples,
    load_domains,
    resolve_domain_dir,
)
from common.utils import find_device


@dataclass(frozen=True)
class RouterSample:
    text: str
    domain_id: int
    domain_name: str
    class_idx: int
    group_id: str = ""
    diagnostic_only: bool = False


class DomainTextDataset(Dataset):
    def __init__(self, samples: Sequence[RouterSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RouterSample:
        return self.samples[idx]


class LinearDomainRouter(nn.Module):
    def __init__(self, d_model: int, n_domains: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_domains)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


def collate_router_samples(samples: Sequence[RouterSample]) -> Dict[str, object]:
    return {
        "texts": [sample.text for sample in samples],
        "domain_ids": [sample.domain_id for sample in samples],
        "domain_names": [sample.domain_name for sample in samples],
        "class_indices": torch.tensor([sample.class_idx for sample in samples], dtype=torch.long),
    }


def load_router_samples(
    domain_dir: Path,
    selected_domain_ids: Optional[Sequence[int]],
    max_texts_per_domain: Optional[int],
    min_samples_per_domain: int,
    allow_diagnostic_contexts: bool = False,
) -> Tuple[List[RouterSample], List[int], List[str]]:
    domains_path = domain_dir / "domains.json"
    validation_dir = domain_dir / "domain_validation"
    if not domains_path.exists():
        raise FileNotFoundError(f"domains.json not found: {domains_path}")
    if not validation_dir.exists():
        raise FileNotFoundError(f"Domain validation directory not found: {validation_dir}")

    domains = load_domains(domains_path, selected_domain_ids=selected_domain_ids)
    domain_ids = [domain.domain_id for domain in domains]
    domain_names = [domain.name for domain in domains]
    class_by_domain_id = {domain_id: class_idx for class_idx, domain_id in enumerate(domain_ids)}

    provisional: List[Tuple[RouterSample, str]] = []
    skipped_diagnostic = 0
    for domain in domains:
        samples_path = validation_dir / f"domain_{domain.domain_id}.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"Validation set for domain {domain.domain_id} not found: {samples_path}")

        for sample_idx, text, payload in iter_domain_samples(samples_path, max_texts=max_texts_per_domain):
            diagnostic_only = bool(payload.get("diagnostic_only", False))
            if diagnostic_only and not allow_diagnostic_contexts:
                skipped_diagnostic += 1
                continue
            normalized_text = " ".join(text.lower().split())
            digest = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()
            source = str(payload.get("source", "domain_validation"))
            source_group = payload.get("source_group")
            if not source_group:
                source_text_idx = payload.get("source_text_idx", "")
                source_group = (
                    f"{source}:{source_text_idx}"
                    if source_text_idx not in (None, "")
                    else f"text:{digest}"
                )
            provisional.append(
                (
                    RouterSample(
                        text=text,
                        domain_id=domain.domain_id,
                        domain_name=domain.name,
                        class_idx=class_by_domain_id[domain.domain_id],
                        group_id=str(source_group),
                        diagnostic_only=diagnostic_only,
                    ),
                    digest,
                )
            )

    text_owners: Dict[str, set[int]] = {}
    group_owners: Dict[str, set[int]] = {}
    for sample, digest in provisional:
        text_owners.setdefault(digest, set()).add(sample.domain_id)
        group_owners.setdefault(sample.group_id, set()).add(sample.domain_id)
    ambiguous_texts = {digest for digest, owners in text_owners.items() if len(owners) > 1}
    ambiguous_groups = {group for group, owners in group_owners.items() if len(owners) > 1}

    samples = []
    seen_domain_texts = set()
    for sample, digest in provisional:
        key = (sample.domain_id, digest)
        if digest in ambiguous_texts or sample.group_id in ambiguous_groups or key in seen_domain_texts:
            continue
        seen_domain_texts.add(key)
        samples.append(sample)

    if ambiguous_texts or ambiguous_groups:
        print(
            "Excluded ambiguous router supervision: "
            f"texts={len(ambiguous_texts)}, source_groups={len(ambiguous_groups)}"
        )
    counts_by_domain = {
        domain_id: sum(sample.domain_id == domain_id for sample in samples)
        for domain_id in domain_ids
    }

    too_small = {
        domain_id: count
        for domain_id, count in counts_by_domain.items()
        if count < min_samples_per_domain
    }
    if too_small:
        raise ValueError(
            "Not enough router samples for some domains: "
            f"{too_small}. Skipped diagnostic contexts: {skipped_diagnostic}. "
            "Supply separately labelled development texts; use "
            "--allow-diagnostic-contexts only for a circular smoke test."
        )
    if len(domain_ids) < 2:
        raise ValueError("Router training requires at least two domains.")
    if not samples:
        raise ValueError(f"No router samples found in {validation_dir}")

    return samples, domain_ids, domain_names


def split_samples_by_domain(
    samples: Sequence[RouterSample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[RouterSample], List[RouterSample]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    rng = random.Random(seed)
    by_domain: Dict[int, Dict[str, List[RouterSample]]] = {}
    for sample in samples:
        group_id = sample.group_id or f"ungrouped:{id(sample)}"
        by_domain.setdefault(sample.domain_id, {}).setdefault(group_id, []).append(sample)

    train_samples: List[RouterSample] = []
    val_samples: List[RouterSample] = []
    for domain_id, grouped_samples in by_domain.items():
        group_ids = list(grouped_samples)
        rng.shuffle(group_ids)
        n_val = max(1, int(round(len(group_ids) * val_fraction)))
        if n_val >= len(group_ids):
            n_val = len(group_ids) - 1
        if n_val < 1:
            raise ValueError(
                f"Domain {domain_id} does not have enough independent source groups "
                "for a validation split"
            )
        val_groups = set(group_ids[:n_val])
        for group_id, group_samples in grouped_samples.items():
            (val_samples if group_id in val_groups else train_samples).extend(group_samples)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def register_mlp_input_hook(mlp_module) -> Tuple[Dict[str, torch.Tensor], object]:
    capture: Dict[str, torch.Tensor] = {}

    def hook(_module, inputs) -> None:
        if not inputs:
            raise ValueError("MLP forward pre-hook received no inputs")
        capture["hidden_states"] = inputs[0].detach()

    handle = mlp_module.register_forward_pre_hook(hook)
    return capture, handle


def extract_router_inputs(
    model,
    tokenizer,
    capture: Dict[str, torch.Tensor],
    texts: Sequence[str],
    class_indices: torch.Tensor,
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    capture.pop("hidden_states", None)
    with torch.no_grad():
        get_backbone(model)(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
    if "hidden_states" not in capture:
        raise RuntimeError("MLP hook did not capture router inputs during model forward pass")
    hidden_states = capture["hidden_states"].float()
    attention_mask = encoded["attention_mask"].bool()
    class_indices = class_indices.to(device)
    token_labels = class_indices.unsqueeze(1).expand_as(encoded["input_ids"])
    return (
        hidden_states[attention_mask],
        token_labels[attention_mask],
        hidden_states,
        attention_mask,
    )


def cache_router_features(
    model,
    tokenizer,
    capture: Dict[str, torch.Tensor],
    samples: Sequence[RouterSample],
    device: torch.device,
    batch_size: int,
    max_length: int,
    description: str,
) -> Dict[str, torch.Tensor]:
    """Run the frozen backbone once and keep compact CPU token features."""
    feature_batches = []
    label_batches = []
    sample_id_batches = []
    weight_batches = []
    model.eval()
    for start in tqdm(range(0, len(samples), batch_size), desc=description):
        batch_samples = samples[start : start + batch_size]
        texts = [sample.text for sample in batch_samples]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        capture.pop("hidden_states", None)
        with torch.no_grad():
            get_backbone(model)(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        if "hidden_states" not in capture:
            raise RuntimeError("MLP hook did not capture router inputs")
        hidden_states = capture["hidden_states"]
        attention_mask = encoded["attention_mask"].bool()
        token_counts = attention_mask.sum(dim=1).clamp_min(1)
        class_indices = torch.tensor(
            [sample.class_idx for sample in batch_samples], dtype=torch.long, device=device
        )
        global_sample_ids = torch.arange(
            start, start + len(batch_samples), dtype=torch.long, device=device
        )
        labels = class_indices.unsqueeze(1).expand_as(encoded["input_ids"])
        sample_ids = global_sample_ids.unsqueeze(1).expand_as(encoded["input_ids"])
        # Each text contributes total weight one, independent of its token
        # length. This prevents long MATH solutions from dominating DBpedia.
        weights = (1.0 / token_counts.float()).unsqueeze(1).expand_as(hidden_states[..., 0])
        feature_batches.append(hidden_states[attention_mask].detach().cpu().to(torch.float16))
        label_batches.append(labels[attention_mask].detach().cpu())
        sample_id_batches.append(sample_ids[attention_mask].detach().cpu())
        weight_batches.append(weights[attention_mask].detach().cpu())
    if not feature_batches:
        raise ValueError(f"No router features extracted for {description}")
    return {
        "features": torch.cat(feature_batches),
        "labels": torch.cat(label_batches),
        "sample_ids": torch.cat(sample_id_batches),
        "weights": torch.cat(weight_batches),
    }


def run_cached_router_epoch(
    router: LinearDomainRouter,
    cached: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    is_train = optimizer is not None
    router.train(is_train)
    dataset = TensorDataset(
        cached["features"], cached["labels"], cached["weights"]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    total_weighted_nll = 0.0
    total_weight = 0.0
    confusion = np.zeros((router.linear.out_features, router.linear.out_features), dtype=np.int64)
    for features, labels, weights in loader:
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device)
        weights = weights.to(device)
        logits = router(features)
        token_losses = F.cross_entropy(logits, labels, reduction="none")
        loss = torch.sum(token_losses * weights) / torch.sum(weights)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            update_confusion(
                confusion, labels.detach().cpu().numpy(), predictions.detach().cpu().numpy()
            )
            batch_weight = float(weights.sum().item())
            total_weighted_nll += float(loss.item()) * batch_weight
            total_weight += batch_weight
    return {
        "loss": total_weighted_nll / total_weight if total_weight else math.nan,
        "confusion": confusion,
        "total_tokens": int(len(dataset)),
        "total_text_weight": total_weight,
    }


def predict_cached_router(
    router: LinearDomainRouter,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = []
    router.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(
                device=device, dtype=torch.float32
            )
            probabilities.append(torch.softmax(router(batch), dim=-1).cpu().numpy())
    matrix = np.concatenate(probabilities, axis=0)
    return matrix.max(axis=1), matrix.argmax(axis=1)


def sample_level_router_metrics(
    router: LinearDomainRouter,
    cached: Dict[str, torch.Tensor],
    n_samples: int,
    domain_ids: Sequence[int],
    domain_names: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    _, predictions = predict_cached_router(router, cached["features"], device, batch_size)
    labels = cached["labels"].numpy()
    sample_ids = cached["sample_ids"].numpy()
    n_domains = len(domain_ids)
    vote_counts = np.zeros((n_samples, n_domains), dtype=np.int64)
    np.add.at(vote_counts, (sample_ids, predictions), 1)
    sample_predictions = vote_counts.argmax(axis=1)
    sample_targets = np.empty(n_samples, dtype=np.int64)
    for sample_id in range(n_samples):
        target_values = np.unique(labels[sample_ids == sample_id])
        if len(target_values) != 1:
            raise ValueError(f"Cached sample {sample_id} has inconsistent labels")
        sample_targets[sample_id] = target_values[0]
    confusion = np.zeros((n_domains, n_domains), dtype=np.int64)
    update_confusion(confusion, sample_targets, sample_predictions)
    return metrics_from_confusion(confusion, domain_ids, domain_names)


def calibrate_confidence_threshold(
    router: LinearDomainRouter,
    general_features: torch.Tensor,
    domain_cached: Dict[str, torch.Tensor],
    target_general_route_rate: float,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    if not 0.0 <= target_general_route_rate <= 1.0:
        raise ValueError("target_general_route_rate must be between 0 and 1")
    general_confidence, general_predictions = predict_cached_router(
        router, general_features, device, batch_size
    )
    domain_confidence, domain_predictions = predict_cached_router(
        router, domain_cached["features"], device, batch_size
    )
    if target_general_route_rate == 1.0:
        threshold = 0.0
    else:
        quantile = np.quantile(
            general_confidence, 1.0 - target_general_route_rate, method="higher"
        )
        threshold = float(np.nextafter(quantile, np.inf))
        threshold = min(1.0, threshold)
    general_routed = general_confidence >= threshold
    domain_routed = domain_confidence >= threshold
    domain_labels = domain_cached["labels"].numpy()
    return {
        "method": "general-development empirical confidence quantile",
        "confidence_threshold": threshold,
        "target_general_route_rate": float(target_general_route_rate),
        "general_token_route_rate": float(np.mean(general_routed)),
        "general_tokens": int(len(general_confidence)),
        "domain_token_coverage": float(np.mean(domain_routed)),
        "domain_routed_accuracy": (
            float(np.mean(domain_predictions[domain_routed] == domain_labels[domain_routed]))
            if np.any(domain_routed)
            else None
        ),
        "domain_tokens": int(len(domain_confidence)),
        "general_predicted_class_histogram": np.bincount(
            general_predictions, minlength=router.linear.out_features
        ).astype(int).tolist(),
    }


def load_general_calibration_samples(path: Path, max_texts: Optional[int]) -> List[RouterSample]:
    if not path.exists():
        raise FileNotFoundError(f"General calibration CSV not found: {path}")
    samples = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a text column")
        for row in reader:
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                samples.append(RouterSample(text, -1, "general", 0, "general"))
                if max_texts is not None and len(samples) >= max_texts:
                    break
    if not samples:
        raise ValueError(f"No general calibration texts found in {path}")
    return samples


def update_confusion(confusion: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> None:
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1


def metrics_from_confusion(confusion: np.ndarray, domain_ids: Sequence[int], domain_names: Sequence[str]) -> Dict[str, object]:
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = correct / total if total else 0.0

    per_domain = []
    f1_values = []
    for class_idx, (domain_id, domain_name) in enumerate(zip(domain_ids, domain_names)):
        tp = float(confusion[class_idx, class_idx])
        fp = float(confusion[:, class_idx].sum() - tp)
        fn = float(confusion[class_idx, :].sum() - tp)
        support = int(confusion[class_idx, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_values.append(f1)
        per_domain.append(
            {
                "domain_id": int(domain_id),
                "domain_name": domain_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "micro_f1": accuracy,
        "per_domain": per_domain,
        "confusion_matrix": confusion.astype(int).tolist(),
        "total_tokens": total,
    }


def run_router_epoch(
    model,
    tokenizer,
    router: LinearDomainRouter,
    dataloader: DataLoader,
    capture: Dict[str, torch.Tensor],
    device: torch.device,
    max_length: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    is_train = optimizer is not None
    router.train(is_train)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    confusion = np.zeros((router.linear.out_features, router.linear.out_features), dtype=np.int64)

    for batch in tqdm(dataloader, desc="Router train" if is_train else "Router eval"):
        features, labels, _, _ = extract_router_inputs(
            model=model,
            tokenizer=tokenizer,
            capture=capture,
            texts=batch["texts"],
            class_indices=batch["class_indices"],
            device=device,
            max_length=max_length,
        )
        logits = router(features)
        loss = F.cross_entropy(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            update_confusion(
                confusion,
                labels.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
            )
            n_tokens = int(labels.numel())
            total_nll += float(loss.item()) * n_tokens
            total_tokens += n_tokens

    return {
        "loss": total_nll / total_tokens if total_tokens else math.nan,
        "confusion": confusion,
        "total_tokens": total_tokens,
    }


def write_router_report(path: Path, metrics: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_epoch = metrics.get("best_epoch")
    best_val = metrics.get("best_val", {})
    with path.open("w", encoding="utf-8") as f:
        f.write("# Router Training Report\n\n")
        f.write(f"- Best epoch: {best_epoch}\n")
        f.write(f"- Validation loss: {best_val.get('loss', 'n/a')}\n")
        f.write(f"- Validation accuracy: {best_val.get('accuracy', 'n/a')}\n")
        f.write(f"- Validation macro F1: {best_val.get('macro_f1', 'n/a')}\n\n")
        sample_metrics = metrics.get("sample_level_validation", {})
        calibration = metrics.get("confidence_calibration", {})
        f.write(f"- Sample-level validation accuracy: {sample_metrics.get('accuracy', 'n/a')}\n")
        f.write(f"- Sample-level validation macro F1: {sample_metrics.get('macro_f1', 'n/a')}\n")
        f.write(f"- Calibrated confidence threshold: {calibration.get('confidence_threshold', 'n/a')}\n")
        f.write(f"- General token route rate: {calibration.get('general_token_route_rate', 'n/a')}\n")
        f.write(f"- Domain token coverage: {calibration.get('domain_token_coverage', 'n/a')}\n\n")
        f.write("## Per-domain validation metrics\n\n")
        f.write("| domain_id | domain_name | precision | recall | f1 | support |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in best_val.get("per_domain", []):
            f.write(
                f"| {row['domain_id']} | {row['domain_name']} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['support']} |\n"
            )


def save_router_checkpoint(
    path: Path,
    router: LinearDomainRouter,
    domain_ids: Sequence[int],
    domain_names: Sequence[str],
    model_name: str,
    tokenizer_name: str,
    layer_num: int,
    metrics: Dict[str, object],
    confidence_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "linear_domain_router",
            "router_state_dict": router.state_dict(),
            "domain_ids": [int(domain_id) for domain_id in domain_ids],
            "domain_names": list(domain_names),
            "d_model": int(router.linear.in_features),
            "n_domains": int(router.linear.out_features),
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "layer_num": int(layer_num),
            "metrics": metrics,
            "confidence_threshold": float(confidence_threshold),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear domain router on MLP inputs.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--domain-dir", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--domain-ids", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-texts-per-domain", type=int, default=None)
    parser.add_argument("--min-samples-per-domain", type=int, default=10)
    parser.add_argument(
        "--allow-diagnostic-contexts",
        action="store_true",
        help=(
            "Allow router training on contexts reused from feature discovery. "
            "This is circular and must not be reported as independent validation."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--general-calibration-csv", default=None)
    parser.add_argument("--max-general-calibration-texts", type=int, default=200)
    parser.add_argument("--target-general-route-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.feature_batch_size,
        args.epochs,
        args.max_length,
        args.min_samples_per_domain,
    ) < 1:
        raise ValueError("Batch size, epochs, max length, and minimum samples must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if not 0.0 <= args.target_general_route_rate <= 1.0:
        raise ValueError("--target-general-route-rate must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "router"

    samples, domain_ids, domain_names = load_router_samples(
        domain_dir=domain_dir,
        selected_domain_ids=args.domain_ids,
        max_texts_per_domain=args.max_texts_per_domain,
        min_samples_per_domain=args.min_samples_per_domain,
        allow_diagnostic_contexts=args.allow_diagnostic_contexts,
    )
    train_samples, val_samples = split_samples_by_domain(samples, val_fraction=args.val_fraction, seed=args.seed)
    train_groups = {sample.group_id for sample in train_samples}
    val_groups = {sample.group_id for sample in val_samples}
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Group-aware router split leaked {len(overlap)} source groups")

    device = find_device()
    print(f"Using device: {device}")
    print(f"Domain directory: {domain_dir}")
    print(f"Router samples: train={len(train_samples)}, val={len(val_samples)}")
    general_calibration_path = Path(args.general_calibration_csv) if args.general_calibration_csv else (
        domain_dir / "development_general.csv"
    )
    general_samples = load_general_calibration_samples(
        general_calibration_path, args.max_general_calibration_texts
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mlp_module = get_mlp_module(model, args.layer_num)
    capture, handle = register_mlp_input_hook(mlp_module)
    try:
        router = LinearDomainRouter(d_model=int(model.config.hidden_size), n_domains=len(domain_ids)).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        train_cached = cache_router_features(
            model, tokenizer, capture, train_samples, device, args.batch_size,
            args.max_length, "Cache router train features"
        )
        val_cached = cache_router_features(
            model, tokenizer, capture, val_samples, device, args.batch_size,
            args.max_length, "Cache router validation features"
        )
        general_cached = cache_router_features(
            model, tokenizer, capture, general_samples, device, args.batch_size,
            args.max_length, "Cache general calibration features"
        )

        history = []
        best_metrics: Optional[Dict[str, object]] = None
        best_state_dict = None
        best_epoch = -1
        best_macro_f1 = float("-inf")

        for epoch in range(1, args.epochs + 1):
            train_raw = run_cached_router_epoch(
                router=router,
                cached=train_cached,
                device=device,
                batch_size=args.feature_batch_size,
                optimizer=optimizer,
            )
            val_raw = run_cached_router_epoch(
                router=router,
                cached=val_cached,
                device=device,
                batch_size=args.feature_batch_size,
                optimizer=None,
            )

            train_metrics = {
                "loss": train_raw["loss"],
                **metrics_from_confusion(train_raw["confusion"], domain_ids, domain_names),
            }
            val_metrics = {
                "loss": val_raw["loss"],
                **metrics_from_confusion(val_raw["confusion"], domain_ids, domain_names),
            }
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if float(val_metrics["macro_f1"]) > best_macro_f1:
                best_macro_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                best_metrics = val_metrics
                best_state_dict = {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}

        if best_state_dict is not None:
            router.load_state_dict(best_state_dict)
        sample_validation = sample_level_router_metrics(
            router, val_cached, len(val_samples), domain_ids, domain_names,
            device, args.feature_batch_size
        )
        calibration = calibrate_confidence_threshold(
            router=router,
            general_features=general_cached["features"],
            domain_cached=val_cached,
            target_general_route_rate=args.target_general_route_rate,
            device=device,
            batch_size=args.feature_batch_size,
        )
        metrics = {
            "domain_dir": str(domain_dir),
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "layer_num": args.layer_num,
            "domain_ids": domain_ids,
            "domain_names": domain_names,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_source_groups": len(train_groups),
            "val_source_groups": len(val_groups),
            "diagnostic_context_samples": sum(sample.diagnostic_only for sample in samples),
            "independent_of_feature_discovery": not any(sample.diagnostic_only for sample in samples),
            "development_validation_only": True,
            "holdout_used": False,
            "allow_diagnostic_contexts": bool(args.allow_diagnostic_contexts),
            "best_epoch": best_epoch,
            "best_val": best_metrics,
            "sample_level_validation": sample_validation,
            "confidence_calibration": calibration,
            "general_calibration_csv": str(general_calibration_path),
            "history": history,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        save_router_checkpoint(
            path=output_dir / "router.pt",
            router=router,
            domain_ids=domain_ids,
            domain_names=domain_names,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            layer_num=args.layer_num,
            metrics=metrics,
            confidence_threshold=float(calibration["confidence_threshold"]),
        )
        with (output_dir / "router_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        write_router_report(output_dir / "router_report.md", metrics)
        print(f"Router artifacts written to: {output_dir}")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
