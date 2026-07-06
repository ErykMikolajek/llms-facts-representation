from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_assembly import build_hard_routed_moe, build_single_expert_model
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


def iter_texts_from_domain_jsonl(path: Path, max_texts: Optional[int]) -> Iterable[str]:
    for _, text, _ in iter_domain_samples(path, max_texts=max_texts):
        yield text


def iter_texts_from_validation_csv(path: Path, max_texts: Optional[int]) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        yielded = 0
        for row in reader:
            if max_texts is not None and yielded >= max_texts:
                break
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            yielded += 1
            yield text


def batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def setup_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, tokenizer: AutoTokenizer, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model


def compute_causal_lm_ppl(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Dict[str, float]:
    if not texts:
        return {
            "loss": math.nan,
            "ppl": math.nan,
            "n_texts": 0,
            "n_loss_tokens": 0,
        }

    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for batch_texts in tqdm(list(batched(texts, batch_size)), desc="PPL batches"):
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            n_loss_tokens = int((labels[:, 1:] != -100).sum().item())
            if n_loss_tokens == 0:
                continue

            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
                return_dict=True,
            )
            total_nll += float(outputs.loss.item()) * n_loss_tokens
            total_tokens += n_loss_tokens

    mean_loss = total_nll / total_tokens if total_tokens else math.nan
    ppl = math.inf if mean_loss > 700 else math.exp(mean_loss)
    return {
        "loss": mean_loss,
        "ppl": ppl,
        "n_texts": len(texts),
        "n_loss_tokens": total_tokens,
    }


def load_eval_texts(
    domain_dir: Path,
    validation_csv: Path,
    selected_domain_ids: Optional[Sequence[int]],
    max_texts_per_domain: Optional[int],
    max_general_texts: Optional[int],
) -> Dict[str, object]:
    domains_path = domain_dir / "domains.json"
    validation_dir = domain_dir / "domain_validation"
    domains = load_domains(domains_path, selected_domain_ids=selected_domain_ids)

    domain_texts = {}
    for domain in domains:
        samples_path = validation_dir / f"domain_{domain.domain_id}.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"Domain validation set not found: {samples_path}")
        domain_texts[str(domain.domain_id)] = {
            "domain_id": domain.domain_id,
            "domain_name": domain.name,
            "texts": list(iter_texts_from_domain_jsonl(samples_path, max_texts=max_texts_per_domain)),
        }

    if not validation_csv.exists():
        raise FileNotFoundError(f"General validation CSV not found: {validation_csv}")
    general_texts = list(iter_texts_from_validation_csv(validation_csv, max_texts=max_general_texts))
    return {
        "domains": domain_texts,
        "general": general_texts,
    }


def evaluate_base_model(
    model_name: str,
    tokenizer,
    device: torch.device,
    eval_texts: Dict[str, object],
    batch_size: int,
    max_length: int,
) -> Dict[str, object]:
    model = load_base_model(model_name, tokenizer, device)
    results = {
        "general": compute_causal_lm_ppl(model, tokenizer, eval_texts["general"], device, batch_size, max_length),
        "domains": {},
    }
    for domain_id, payload in eval_texts["domains"].items():
        results["domains"][domain_id] = {
            "domain_id": payload["domain_id"],
            "domain_name": payload["domain_name"],
            **compute_causal_lm_ppl(model, tokenizer, payload["texts"], device, batch_size, max_length),
        }
    return results


def evaluate_single_experts(
    model_name: str,
    tokenizer,
    device: torch.device,
    eval_texts: Dict[str, object],
    experts_dir: Path,
    layer_num: int,
    batch_size: int,
    max_length: int,
    evaluate_general: bool,
) -> Dict[str, object]:
    results = {}
    for domain_id, payload in eval_texts["domains"].items():
        expert_path = experts_dir / f"domain_{domain_id}_mlp_expert.pt"
        model = load_base_model(model_name, tokenizer, device)
        build_single_expert_model(
            model=model,
            expert_path=expert_path,
            layer_num=layer_num,
            model_name=model_name,
        )
        domain_metrics = compute_causal_lm_ppl(model, tokenizer, payload["texts"], device, batch_size, max_length)
        expert_results = {
            "domain_id": payload["domain_id"],
            "domain_name": payload["domain_name"],
            "expert_path": str(expert_path),
            "own_domain": domain_metrics,
        }
        if evaluate_general:
            expert_results["general"] = compute_causal_lm_ppl(
                model,
                tokenizer,
                eval_texts["general"],
                device,
                batch_size,
                max_length,
            )
        results[domain_id] = expert_results
    return results


def evaluate_moe_model(
    model_name: str,
    tokenizer,
    device: torch.device,
    eval_texts: Dict[str, object],
    experts_dir: Path,
    router_path: Path,
    layer_num: int,
    batch_size: int,
    max_length: int,
    allow_diagnostic_experts: bool,
) -> Dict[str, object]:
    model = load_base_model(model_name, tokenizer, device)
    build_hard_routed_moe(
        model=model,
        experts_dir=experts_dir,
        router_path=router_path,
        layer_num=layer_num,
        model_name=model_name,
        allow_diagnostic_experts=allow_diagnostic_experts,
    )
    results = {
        "general": compute_causal_lm_ppl(model, tokenizer, eval_texts["general"], device, batch_size, max_length),
        "domains": {},
        "router_path": str(router_path),
    }
    for domain_id, payload in eval_texts["domains"].items():
        results["domains"][domain_id] = {
            "domain_id": payload["domain_id"],
            "domain_name": payload["domain_name"],
            **compute_causal_lm_ppl(model, tokenizer, payload["texts"], device, batch_size, max_length),
        }
    return results


def build_summary(results: Dict[str, object], max_general_ppl_regression: float) -> Dict[str, object]:
    summary = {
        "general_ppl_regression": None,
        "general_regression_within_threshold": None,
        "domain_comparisons": {},
    }
    base_general = results.get("base", {}).get("general", {}).get("ppl")
    moe_general = results.get("moe", {}).get("general", {}).get("ppl")
    if base_general and moe_general and math.isfinite(base_general) and math.isfinite(moe_general):
        regression = (moe_general - base_general) / base_general
        summary["general_ppl_regression"] = regression
        summary["general_regression_within_threshold"] = regression <= max_general_ppl_regression

    base_domains = results.get("base", {}).get("domains", {})
    moe_domains = results.get("moe", {}).get("domains", {})
    single_experts = results.get("single_experts", {})
    for domain_id, base_metrics in base_domains.items():
        base_ppl = base_metrics.get("ppl")
        moe_ppl = moe_domains.get(domain_id, {}).get("ppl")
        expert_ppl = single_experts.get(domain_id, {}).get("own_domain", {}).get("ppl")
        summary["domain_comparisons"][domain_id] = {
            "domain_name": base_metrics.get("domain_name"),
            "base_ppl": base_ppl,
            "single_expert_ppl": expert_ppl,
            "moe_ppl": moe_ppl,
            "single_expert_improves_base": (
                bool(expert_ppl < base_ppl)
                if base_ppl and expert_ppl and math.isfinite(base_ppl) and math.isfinite(expert_ppl)
                else None
            ),
            "moe_improves_base": (
                bool(moe_ppl < base_ppl)
                if base_ppl and moe_ppl and math.isfinite(base_ppl) and math.isfinite(moe_ppl)
                else None
            ),
        }
    return summary


def write_ppl_report(path: Path, results: Dict[str, object], summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# MoE Perplexity Report\n\n")
        f.write("## General Validation\n\n")
        f.write("| model | loss | ppl | n_texts | n_loss_tokens |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for model_key in ["base", "moe"]:
            metrics = results.get(model_key, {}).get("general", {})
            f.write(
                f"| {model_key} | {metrics.get('loss')} | {metrics.get('ppl')} | "
                f"{metrics.get('n_texts')} | {metrics.get('n_loss_tokens')} |\n"
            )

        f.write("\n## Domain Validation\n\n")
        f.write("| domain_id | domain_name | base_ppl | single_expert_ppl | moe_ppl | expert_improves | moe_improves |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for domain_id, row in summary.get("domain_comparisons", {}).items():
            f.write(
                f"| {domain_id} | {row.get('domain_name')} | {row.get('base_ppl')} | "
                f"{row.get('single_expert_ppl')} | {row.get('moe_ppl')} | "
                f"{row.get('single_expert_improves_base')} | {row.get('moe_improves_base')} |\n"
            )

        f.write("\n## Summary\n\n")
        f.write(f"- General PPL regression: {summary.get('general_ppl_regression')}\n")
        f.write(f"- Within threshold: {summary.get('general_regression_within_threshold')}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPL for base, single-expert, and hard-routed MoE models.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--domain-dir", default=None)
    parser.add_argument("--experts-dir", default=None)
    parser.add_argument("--router-path", default=None)
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--domain-ids", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-texts-per-domain", type=int, default=None)
    parser.add_argument("--max-general-texts", type=int, default=None)
    parser.add_argument("--evaluate-experts-on-general", action="store_true")
    parser.add_argument("--allow-diagnostic-experts", action="store_true")
    parser.add_argument("--max-general-ppl-regression", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    experts_dir = Path(args.experts_dir) if args.experts_dir else domain_dir / "domain_mlp_experts"
    router_path = Path(args.router_path) if args.router_path else domain_dir / "router/router.pt"
    validation_csv = Path(args.validation_csv) if args.validation_csv else Path(args.data_path) / "validation.csv"
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "moe_validation"

    device = find_device()
    print(f"Using device: {device}")
    tokenizer = setup_tokenizer(args.tokenizer_name)
    eval_texts = load_eval_texts(
        domain_dir=domain_dir,
        validation_csv=validation_csv,
        selected_domain_ids=args.domain_ids,
        max_texts_per_domain=args.max_texts_per_domain,
        max_general_texts=args.max_general_texts,
    )

    results = {
        "config": {
            "domain_dir": str(domain_dir),
            "experts_dir": str(experts_dir),
            "router_path": str(router_path),
            "validation_csv": str(validation_csv),
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "layer_num": args.layer_num,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        }
    }

    print("Evaluating base model...")
    results["base"] = evaluate_base_model(
        model_name=args.model_name,
        tokenizer=tokenizer,
        device=device,
        eval_texts=eval_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("Evaluating single experts...")
    results["single_experts"] = evaluate_single_experts(
        model_name=args.model_name,
        tokenizer=tokenizer,
        device=device,
        eval_texts=eval_texts,
        experts_dir=experts_dir,
        layer_num=args.layer_num,
        batch_size=args.batch_size,
        max_length=args.max_length,
        evaluate_general=args.evaluate_experts_on_general,
    )

    print("Evaluating hard-routed MoE...")
    results["moe"] = evaluate_moe_model(
        model_name=args.model_name,
        tokenizer=tokenizer,
        device=device,
        eval_texts=eval_texts,
        experts_dir=experts_dir,
        router_path=router_path,
        layer_num=args.layer_num,
        batch_size=args.batch_size,
        max_length=args.max_length,
        allow_diagnostic_experts=args.allow_diagnostic_experts,
    )

    summary = build_summary(results, max_general_ppl_regression=args.max_general_ppl_regression)
    results["summary"] = summary

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppl_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_ppl_report(output_dir / "ppl_report.md", results, summary)
    print(f"PPL validation artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
