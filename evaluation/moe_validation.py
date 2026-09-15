from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe.moe_assembly import (
    CompactPrunedMLP,
    build_hard_routed_moe,
    build_single_expert_model,
)
from domain_mapping.domain_mlp_activation_mapping import describe_mlp, get_mlp_module, set_mlp_module
from domain_mapping.domain_mlp_pruning import apply_mlp_mask
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


def load_labeled_domain_eval_csv(
    path: Path,
    domains,
    max_texts_per_domain: Optional[int],
) -> Dict[str, Dict[str, object]]:
    """Load a manually curated holdout with required ``domain_id,text`` columns."""
    by_id = {
        int(domain.domain_id): {
            "domain_id": int(domain.domain_id),
            "domain_name": domain.name,
            "texts": [],
        }
        for domain in domains
    }
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"domain_id", "text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: domain_id,text")
        for row in reader:
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                domain_id = int(row.get("domain_id", ""))
            except ValueError as exc:
                raise ValueError(f"Invalid domain_id in {path}: {row.get('domain_id')!r}") from exc
            if domain_id not in by_id:
                continue
            texts = by_id[domain_id]["texts"]
            if max_texts_per_domain is None or len(texts) < max_texts_per_domain:
                texts.append(text)

    missing = [domain_id for domain_id, payload in by_id.items() if not payload["texts"]]
    if missing:
        raise ValueError(f"Independent domain evaluation CSV has no texts for domains: {missing}")
    return {str(domain_id): payload for domain_id, payload in by_id.items()}


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
            # With left padding, Hugging Face's shifted causal-LM loss would
            # otherwise ask a padding-position logit to predict the first real
            # token. Exclude that first valid token for both padding sides.
            nonempty = encoded["attention_mask"].sum(dim=1) > 0
            first_valid = encoded["attention_mask"].long().argmax(dim=1)
            rows = torch.arange(labels.shape[0], device=labels.device)[nonempty]
            labels[rows, first_valid[nonempty]] = -100
            n_loss_tokens = int((labels[:, 1:] != -100).sum().item())
            if n_loss_tokens == 0:
                continue

            # HardRoutedMLP cannot infer padding from its hidden-state tensor.
            # Supply the attention mask explicitly so routing telemetry counts
            # only real tokens. Computation and language-model loss are
            # unchanged; this affects statistics only.
            for module in model.modules():
                setter = getattr(module, "set_routing_token_mask", None)
                if callable(setter):
                    setter(encoded["attention_mask"])

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
    domain_eval_csv: Optional[Path] = None,
) -> Dict[str, object]:
    domains_path = domain_dir / "domains.json"
    validation_dir = domain_dir / "domain_validation"
    domains = load_domains(domains_path, selected_domain_ids=selected_domain_ids)

    if domain_eval_csv is not None:
        if not domain_eval_csv.exists():
            raise FileNotFoundError(f"Independent domain evaluation CSV not found: {domain_eval_csv}")
        domain_texts = load_labeled_domain_eval_csv(
            domain_eval_csv, domains, max_texts_per_domain=max_texts_per_domain
        )
        domain_source = str(domain_eval_csv)
    else:
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
        domain_source = str(validation_dir)

    if not validation_csv.exists():
        raise FileNotFoundError(f"General validation CSV not found: {validation_csv}")
    general_texts = list(iter_texts_from_validation_csv(validation_csv, max_texts=max_general_texts))
    return {
        "domains": domain_texts,
        "general": general_texts,
        "domain_source": domain_source,
    }


def validate_evaluation_independence(
    domain_dir: Path,
    general_eval_csv: Path,
    domain_eval_csv: Optional[Path],
    allow_development_eval: bool,
) -> None:
    problems = []
    if domain_eval_csv is None:
        problems.append("no --domain-eval-csv was supplied; domain_validation would be reused")

    diagnostic_markers = {"1", "true", "yes", "y"}

    def csv_declares_diagnostic_rows(path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "diagnostic_only" not in (reader.fieldnames or []):
                return False
            return any(
                str(row.get("diagnostic_only", "")).strip().lower()
                in diagnostic_markers
                for row in reader
            )

    validation_dir = (domain_dir / "domain_validation").resolve()
    try:
        general_eval_csv.resolve().relative_to(validation_dir)
    except ValueError:
        pass
    else:
        problems.append("general evaluation CSV comes from domain_validation")
    if csv_declares_diagnostic_rows(general_eval_csv):
        problems.append("general evaluation CSV declares diagnostic_only rows")

    if domain_eval_csv is not None:
        try:
            domain_eval_csv.resolve().relative_to(validation_dir)
        except ValueError:
            pass
        else:
            problems.append("domain evaluation CSV comes from domain_validation")
        if csv_declares_diagnostic_rows(domain_eval_csv):
            problems.append("domain evaluation CSV declares diagnostic_only rows")

    domains_path = domain_dir / "domains.json"
    if domains_path.exists():
        with domains_path.open("r", encoding="utf-8") as f:
            triage = json.load(f)
        discovery_validation = triage.get("validation_source")
        if discovery_validation and discovery_validation != "skipped":
            if Path(str(discovery_validation)).resolve() == general_eval_csv.resolve():
                problems.append("general evaluation CSV is the same source used during domain triage")
            if (
                domain_eval_csv is not None
                and Path(str(discovery_validation)).resolve() == domain_eval_csv.resolve()
            ):
                problems.append("domain evaluation CSV is the same development source used for mapping/router work")

    if not problems:
        return
    message = "Non-independent MoE evaluation: " + "; ".join(problems) + "."
    if allow_development_eval:
        print(f"WARNING: {message} Results are diagnostic only.")
    else:
        raise ValueError(message + " Pass --allow-development-eval only for a smoke test.")


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


def mlp_neuron_magnitude_scores(mlp_module) -> np.ndarray:
    """A simple weight-magnitude control, not a domain-specific importance score."""
    spec = describe_mlp(mlp_module)
    input_energy = None
    for name in spec["input_projections"]:
        weight = getattr(mlp_module, str(name)).weight.detach().float()
        energy = torch.sum(torch.square(weight), dim=1)
        input_energy = energy if input_energy is None else input_energy + energy
    output_weight = getattr(mlp_module, str(spec["output_projection"])).weight.detach().float()
    output_energy = torch.sum(torch.square(output_weight), dim=0)
    scores = torch.sqrt(torch.clamp(input_energy * output_energy, min=0.0))
    return scores.cpu().numpy().astype(np.float64)


def make_matched_control_masks(
    mlp_module,
    selected_mask: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    selected_mask = np.asarray(selected_mask, dtype=bool)
    n_inner = len(selected_mask)
    n_keep = int(selected_mask.sum())
    if not 0 < n_keep <= n_inner:
        raise ValueError("Control masks require a non-empty selected mask")
    rng = np.random.default_rng(seed)
    random_mask = np.zeros(n_inner, dtype=bool)
    random_mask[rng.choice(n_inner, size=n_keep, replace=False)] = True
    magnitude = mlp_neuron_magnitude_scores(mlp_module)
    order = np.argsort(magnitude, kind="stable")[::-1]
    magnitude_mask = np.zeros(n_inner, dtype=bool)
    magnitude_mask[order[:n_keep]] = True
    return {"random_matched": random_mask, "magnitude_matched": magnitude_mask}


def build_masked_control_model(model, keep_mask: np.ndarray, layer_num: int):
    mlp_module = get_mlp_module(model, layer_num)
    apply_mlp_mask(mlp_module, keep_mask, zero_bias=True)
    compact = CompactPrunedMLP(
        mlp_module, torch.as_tensor(keep_mask, dtype=torch.bool)
    ).to(next(model.parameters()).device).eval()
    set_mlp_module(model, layer_num, compact)
    return model


def evaluate_matched_controls(
    model_name: str,
    tokenizer,
    device: torch.device,
    eval_texts: Dict[str, object],
    experts_dir: Path,
    layer_num: int,
    batch_size: int,
    max_length: int,
    seed: int,
) -> Dict[str, object]:
    results: Dict[str, object] = {"random_matched": {}, "magnitude_matched": {}}
    for domain_id, payload in eval_texts["domains"].items():
        selected_mask = np.load(experts_dir / f"domain_{domain_id}_mask.npy").astype(bool)
        # Scores and random masks are frozen before seeing this domain's loss.
        template = load_base_model(model_name, tokenizer, device)
        masks = make_matched_control_masks(
            get_mlp_module(template, layer_num), selected_mask, seed + int(domain_id)
        )
        del template
        for method, keep_mask in masks.items():
            model = load_base_model(model_name, tokenizer, device)
            build_masked_control_model(model, keep_mask, layer_num)
            metrics = compute_causal_lm_ppl(
                model, tokenizer, payload["texts"], device, batch_size, max_length
            )
            results[method][domain_id] = {
                "domain_id": payload["domain_id"],
                "domain_name": payload["domain_name"],
                "n_kept": int(keep_mask.sum()),
                **metrics,
            }
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
    confidence_threshold: Optional[float],
) -> Dict[str, object]:
    model = load_base_model(model_name, tokenizer, device)
    build_hard_routed_moe(
        model=model,
        experts_dir=experts_dir,
        router_path=router_path,
        layer_num=layer_num,
        model_name=model_name,
        allow_diagnostic_experts=allow_diagnostic_experts,
        confidence_threshold=confidence_threshold,
    )
    routed_mlp = get_mlp_module(model, layer_num)
    resolved_threshold = float(routed_mlp.confidence_threshold)
    routed_mlp.reset_routing_stats()
    general_metrics = compute_causal_lm_ppl(
        model, tokenizer, eval_texts["general"], device, batch_size, max_length
    )
    general_metrics["routing"] = routed_mlp.routing_stats()
    results = {
        "general": general_metrics,
        "domains": {},
        "router_path": str(router_path),
        "router_confidence_threshold": resolved_threshold,
    }
    for domain_id, payload in eval_texts["domains"].items():
        routed_mlp.reset_routing_stats()
        metrics = compute_causal_lm_ppl(
            model, tokenizer, payload["texts"], device, batch_size, max_length
        )
        metrics["routing"] = routed_mlp.routing_stats()
        results["domains"][domain_id] = {
            "domain_id": payload["domain_id"],
            "domain_name": payload["domain_name"],
            **metrics,
        }
    return results


def build_summary(
    results: Dict[str, object],
    max_general_ppl_regression: float,
    max_domain_ppl_regression: float,
) -> Dict[str, object]:
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
        random_ppl = results.get("controls", {}).get("random_matched", {}).get(domain_id, {}).get("ppl")
        magnitude_ppl = results.get("controls", {}).get("magnitude_matched", {}).get(domain_id, {}).get("ppl")
        expert_regression = (
            (expert_ppl - base_ppl) / base_ppl
            if base_ppl and expert_ppl and math.isfinite(base_ppl) and math.isfinite(expert_ppl)
            else None
        )
        moe_regression = (
            (moe_ppl - base_ppl) / base_ppl
            if base_ppl and moe_ppl and math.isfinite(base_ppl) and math.isfinite(moe_ppl)
            else None
        )
        summary["domain_comparisons"][domain_id] = {
            "domain_name": base_metrics.get("domain_name"),
            "base_ppl": base_ppl,
            "single_expert_ppl": expert_ppl,
            "moe_ppl": moe_ppl,
            "random_matched_ppl": random_ppl,
            "magnitude_matched_ppl": magnitude_ppl,
            "single_expert_relative_ppl_change": expert_regression,
            "moe_relative_ppl_change": moe_regression,
            "single_expert_within_regression_threshold": (
                expert_regression <= max_domain_ppl_regression
                if expert_regression is not None else None
            ),
            "moe_within_regression_threshold": (
                moe_regression <= max_domain_ppl_regression
                if moe_regression is not None else None
            ),
            "selected_beats_random_matched": (
                bool(expert_ppl < random_ppl)
                if expert_ppl and random_ppl and math.isfinite(expert_ppl) and math.isfinite(random_ppl)
                else None
            ),
            "selected_beats_magnitude_matched": (
                bool(expert_ppl < magnitude_ppl)
                if expert_ppl and magnitude_ppl and math.isfinite(expert_ppl) and math.isfinite(magnitude_ppl)
                else None
            ),
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
        f.write("| domain_id | domain_name | base_ppl | selected | random | magnitude | MoE | selected Δ | MoE Δ |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for domain_id, row in summary.get("domain_comparisons", {}).items():
            f.write(
                f"| {domain_id} | {row.get('domain_name')} | {row.get('base_ppl')} | "
                f"{row.get('single_expert_ppl')} | {row.get('random_matched_ppl')} | "
                f"{row.get('magnitude_matched_ppl')} | {row.get('moe_ppl')} | "
                f"{row.get('single_expert_relative_ppl_change')} | "
                f"{row.get('moe_relative_ppl_change')} |\n"
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
    parser.add_argument(
        "--domain-eval-csv",
        default=None,
        help="Independent manually labelled CSV with domain_id,text columns.",
    )
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
    parser.add_argument("--max-domain-ppl-regression", type=float, default=0.10)
    parser.add_argument("--skip-matched-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=["base", "single", "controls", "moe"],
        default=["base", "single", "controls", "moe"],
        help="Evaluation components to run; useful for telemetry-only reruns.",
    )
    parser.add_argument(
        "--router-confidence-threshold",
        type=float,
        default=None,
        help="Override the development-calibrated threshold stored in router.pt.",
    )
    parser.add_argument(
        "--allow-development-eval",
        action="store_true",
        help="Allow circular development data for smoke tests; never report it as holdout performance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.max_length) < 1:
        raise ValueError("--batch-size and --max-length must be positive")
    if args.router_confidence_threshold is not None and not 0.0 <= args.router_confidence_threshold <= 1.0:
        raise ValueError("--router-confidence-threshold must be between 0 and 1")
    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    experts_dir = Path(args.experts_dir) if args.experts_dir else domain_dir / "domain_mlp_experts"
    router_path = Path(args.router_path) if args.router_path else domain_dir / "router/router.pt"
    validation_csv = Path(args.validation_csv) if args.validation_csv else Path(args.data_path) / "validation.csv"
    domain_eval_csv = Path(args.domain_eval_csv) if args.domain_eval_csv else None
    output_dir = Path(args.output_dir) if args.output_dir else domain_dir / "moe_validation"

    validate_evaluation_independence(
        domain_dir=domain_dir,
        general_eval_csv=validation_csv,
        domain_eval_csv=domain_eval_csv,
        allow_development_eval=args.allow_development_eval,
    )

    device = find_device()
    print(f"Using device: {device}")
    tokenizer = setup_tokenizer(args.tokenizer_name)
    eval_texts = load_eval_texts(
        domain_dir=domain_dir,
        validation_csv=validation_csv,
        selected_domain_ids=args.domain_ids,
        max_texts_per_domain=args.max_texts_per_domain,
        max_general_texts=args.max_general_texts,
        domain_eval_csv=domain_eval_csv,
    )

    results = {
        "config": {
            "domain_dir": str(domain_dir),
            "experts_dir": str(experts_dir),
            "router_path": str(router_path),
            "validation_csv": str(validation_csv),
            "domain_eval_csv": str(domain_eval_csv) if domain_eval_csv else None,
            "independent_domain_evaluation_declared": domain_eval_csv is not None,
            "allow_development_eval": bool(args.allow_development_eval),
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "layer_num": args.layer_num,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "router_confidence_threshold": args.router_confidence_threshold,
            "components": args.components,
        }
    }

    if "base" in args.components:
        print("Evaluating base model...")
        results["base"] = evaluate_base_model(
            model_name=args.model_name,
            tokenizer=tokenizer,
            device=device,
            eval_texts=eval_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    if "single" in args.components:
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

    if "controls" in args.components and not args.skip_matched_controls:
        print("Evaluating matched random and magnitude controls...")
        results["controls"] = evaluate_matched_controls(
            model_name=args.model_name,
            tokenizer=tokenizer,
            device=device,
            eval_texts=eval_texts,
            experts_dir=experts_dir,
            layer_num=args.layer_num,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
        )

    if "moe" in args.components:
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
            confidence_threshold=args.router_confidence_threshold,
        )

    summary = build_summary(
        results,
        max_general_ppl_regression=args.max_general_ppl_regression,
        max_domain_ppl_regression=args.max_domain_ppl_regression,
    )
    results["summary"] = summary

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppl_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_ppl_report(output_dir / "ppl_report.md", results, summary)
    print(f"PPL validation artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
