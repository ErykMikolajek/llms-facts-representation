from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from domain_mlp_activation_mapping import get_mlp_module
from topographic_mlp_sae_mapping import (
    DEFAULT_DATA_PATH,
    DEFAULT_LAYER_NUM,
    MODEL_NAME,
    TOKENIZER_NAME,
    iter_domain_samples,
    load_domains,
    resolve_domain_dir,
)
from utils import find_device


@dataclass(frozen=True)
class RouterSample:
    text: str
    domain_id: int
    domain_name: str
    class_idx: int


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

    samples: List[RouterSample] = []
    counts_by_domain: Dict[int, int] = {}
    for domain in domains:
        samples_path = validation_dir / f"domain_{domain.domain_id}.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"Validation set for domain {domain.domain_id} not found: {samples_path}")

        domain_count = 0
        for _, text, _ in iter_domain_samples(samples_path, max_texts=max_texts_per_domain):
            samples.append(
                RouterSample(
                    text=text,
                    domain_id=domain.domain_id,
                    domain_name=domain.name,
                    class_idx=class_by_domain_id[domain.domain_id],
                )
            )
            domain_count += 1
        counts_by_domain[domain.domain_id] = domain_count

    too_small = {
        domain_id: count
        for domain_id, count in counts_by_domain.items()
        if count < min_samples_per_domain
    }
    if too_small:
        raise ValueError(
            "Not enough router samples for some domains: "
            f"{too_small}. Lower --min-samples-per-domain only for diagnostics."
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
    by_domain: Dict[int, List[RouterSample]] = {}
    for sample in samples:
        by_domain.setdefault(sample.domain_id, []).append(sample)

    train_samples: List[RouterSample] = []
    val_samples: List[RouterSample] = []
    for domain_id, domain_samples in by_domain.items():
        shuffled = list(domain_samples)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        if n_val >= len(shuffled):
            n_val = len(shuffled) - 1
        if n_val < 1:
            raise ValueError(f"Domain {domain_id} does not have enough samples for a validation split")
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

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
        model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-texts-per-domain", type=int, default=None)
    parser.add_argument("--min-samples-per-domain", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    )
    train_samples, val_samples = split_samples_by_domain(samples, val_fraction=args.val_fraction, seed=args.seed)

    device = find_device()
    print(f"Using device: {device}")
    print(f"Domain directory: {domain_dir}")
    print(f"Router samples: train={len(train_samples)}, val={len(val_samples)}")

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

        train_loader = DataLoader(
            DomainTextDataset(train_samples),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_router_samples,
        )
        val_loader = DataLoader(
            DomainTextDataset(val_samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_router_samples,
        )

        history = []
        best_metrics: Optional[Dict[str, object]] = None
        best_state_dict = None
        best_epoch = -1
        best_macro_f1 = float("-inf")

        for epoch in range(1, args.epochs + 1):
            train_raw = run_router_epoch(
                model=model,
                tokenizer=tokenizer,
                router=router,
                dataloader=train_loader,
                capture=capture,
                device=device,
                max_length=args.max_length,
                optimizer=optimizer,
            )
            val_raw = run_router_epoch(
                model=model,
                tokenizer=tokenizer,
                router=router,
                dataloader=val_loader,
                capture=capture,
                device=device,
                max_length=args.max_length,
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
        metrics = {
            "domain_dir": str(domain_dir),
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "layer_num": args.layer_num,
            "domain_ids": domain_ids,
            "domain_names": domain_names,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "best_epoch": best_epoch,
            "best_val": best_metrics,
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
        )
        with (output_dir / "router_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        write_router_report(output_dir / "router_report.md", metrics)
        print(f"Router artifacts written to: {output_dir}")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
