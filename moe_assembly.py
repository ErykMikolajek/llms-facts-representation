from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from domain_mlp_activation_mapping import get_mlp_module
from router_training import LinearDomainRouter
from topographic_mlp_sae_mapping import DEFAULT_DATA_PATH, DEFAULT_LAYER_NUM, MODEL_NAME, resolve_domain_dir
from utils import find_device


def get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def set_mlp_module(model, layer_num: int, mlp_module: nn.Module) -> None:
    try:
        model.transformer.h[layer_num].mlp = mlp_module
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"Could not set model.transformer.h[{layer_num}].mlp") from exc


def load_mlp_expert(expert_path: Path, layer_num: Optional[int] = None, model_name: Optional[str] = None) -> Dict[str, object]:
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert artifact not found: {expert_path}")
    try:
        artifact = torch.load(expert_path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(expert_path, map_location="cpu")

    if artifact.get("format") != "mlp_only_sparse_expert":
        raise ValueError(f"Unsupported expert format in {expert_path}: {artifact.get('format')}")
    if layer_num is not None and int(artifact.get("layer_num")) != int(layer_num):
        raise ValueError(
            f"Expert {expert_path} is for layer {artifact.get('layer_num')}, expected {layer_num}"
        )
    if model_name is not None and str(artifact.get("model_name")) != model_name:
        raise ValueError(
            f"Expert {expert_path} is for model {artifact.get('model_name')}, expected {model_name}"
        )
    if "mlp_state_dict" not in artifact:
        raise ValueError(f"Expert {expert_path} does not contain mlp_state_dict")
    return artifact


def load_pruning_summary(experts_dir: Path) -> Dict[int, Dict[str, object]]:
    summary_path = experts_dir / "pruning_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(domain["domain_id"]): domain for domain in payload.get("domains", [])}


def clone_mlp_with_state(base_mlp: nn.Module, state_dict: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    expert = copy.deepcopy(base_mlp)
    expert.load_state_dict(state_dict)
    expert.to(device)
    expert.eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    return expert


def load_router_checkpoint(router_path: Path, device: torch.device) -> Tuple[LinearDomainRouter, Dict[str, object]]:
    if not router_path.exists():
        raise FileNotFoundError(f"Router checkpoint not found: {router_path}")
    try:
        checkpoint = torch.load(router_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(router_path, map_location="cpu")
    if checkpoint.get("format") != "linear_domain_router":
        raise ValueError(f"Unsupported router format in {router_path}: {checkpoint.get('format')}")

    router = LinearDomainRouter(
        d_model=int(checkpoint["d_model"]),
        n_domains=int(checkpoint["n_domains"]),
    )
    router.load_state_dict(checkpoint["router_state_dict"])
    router.to(device)
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    return router, checkpoint


def load_expert_bank(
    experts_dir: Path,
    base_mlp: nn.Module,
    domain_ids: Sequence[int],
    layer_num: int,
    model_name: str,
    device: torch.device,
    allow_diagnostic_experts: bool = False,
) -> nn.ModuleDict:
    pruning_summary = load_pruning_summary(experts_dir)
    expert_bank = nn.ModuleDict()

    for class_idx, domain_id in enumerate(domain_ids):
        summary = pruning_summary.get(int(domain_id), {})
        if summary.get("diagnostic_empty_expert") and not allow_diagnostic_experts:
            raise ValueError(
                f"Expert for domain {domain_id} is marked diagnostic_empty_expert. "
                "Pass allow_diagnostic_experts=True only for diagnostics."
            )

        expert_path = experts_dir / f"domain_{domain_id}_mlp_expert.pt"
        artifact = load_mlp_expert(expert_path, layer_num=layer_num, model_name=model_name)
        expert = clone_mlp_with_state(
            base_mlp=base_mlp,
            state_dict=artifact["mlp_state_dict"],
            device=device,
        )
        expert_bank[str(class_idx)] = expert

    return expert_bank


class HardRoutedMLP(nn.Module):
    def __init__(
        self,
        router: LinearDomainRouter,
        experts: nn.ModuleDict,
        domain_ids: Sequence[int],
        domain_names: Sequence[str],
    ):
        super().__init__()
        if len(experts) != len(domain_ids):
            raise ValueError(f"Expected {len(domain_ids)} experts, got {len(experts)}")
        self.router = router
        self.experts = experts
        self.domain_ids = [int(domain_id) for domain_id in domain_ids]
        self.domain_names = list(domain_names)
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_expert_ids: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(hidden_states.float())
        expert_ids = torch.argmax(router_logits, dim=-1)

        output = torch.zeros_like(hidden_states)
        for class_idx, expert in self.experts.items():
            class_idx_int = int(class_idx)
            token_mask = expert_ids == class_idx_int
            if not token_mask.any():
                continue
            expert_output = expert(hidden_states)
            output[token_mask] = expert_output[token_mask]

        self.last_router_logits = router_logits.detach()
        self.last_expert_ids = expert_ids.detach()
        return output


def validate_router_model_compatibility(model, router_checkpoint: Dict[str, object], layer_num: int, model_name: str) -> None:
    if int(router_checkpoint["d_model"]) != int(model.config.hidden_size):
        raise ValueError(
            f"Router d_model={router_checkpoint['d_model']} does not match "
            f"model hidden_size={model.config.hidden_size}"
        )
    if int(router_checkpoint["layer_num"]) != int(layer_num):
        raise ValueError(
            f"Router layer_num={router_checkpoint['layer_num']} does not match requested layer {layer_num}"
        )
    if str(router_checkpoint["model_name"]) != model_name:
        raise ValueError(
            f"Router model_name={router_checkpoint['model_name']} does not match requested model {model_name}"
        )


def build_single_expert_model(
    model,
    expert_path: Path,
    layer_num: int,
    model_name: str = MODEL_NAME,
):
    device = get_module_device(model)
    base_mlp = get_mlp_module(model, layer_num)
    artifact = load_mlp_expert(expert_path, layer_num=layer_num, model_name=model_name)
    expert = clone_mlp_with_state(base_mlp, artifact["mlp_state_dict"], device=device)
    set_mlp_module(model, layer_num, expert)
    return model


def build_hard_routed_moe(
    model,
    experts_dir: Path,
    router_path: Path,
    layer_num: int,
    model_name: str = MODEL_NAME,
    allow_diagnostic_experts: bool = False,
):
    device = get_module_device(model)
    router, router_checkpoint = load_router_checkpoint(router_path, device=device)
    validate_router_model_compatibility(
        model=model,
        router_checkpoint=router_checkpoint,
        layer_num=layer_num,
        model_name=model_name,
    )
    domain_ids = [int(domain_id) for domain_id in router_checkpoint["domain_ids"]]
    domain_names = list(router_checkpoint["domain_names"])
    base_mlp = get_mlp_module(model, layer_num)
    expert_bank = load_expert_bank(
        experts_dir=experts_dir,
        base_mlp=base_mlp,
        domain_ids=domain_ids,
        layer_num=layer_num,
        model_name=model_name,
        device=device,
        allow_diagnostic_experts=allow_diagnostic_experts,
    )
    routed_mlp = HardRoutedMLP(
        router=router,
        experts=expert_bank,
        domain_ids=domain_ids,
        domain_names=domain_names,
    )
    routed_mlp.to(device)
    routed_mlp.eval()
    set_mlp_module(model, layer_num, routed_mlp)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble TinyStories MLP experts into single-expert or MoE models.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--domain-dir", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--mode", choices=["single-expert", "moe"], required=True)
    parser.add_argument("--expert-path", default=None)
    parser.add_argument("--experts-dir", default=None)
    parser.add_argument("--router-path", default=None)
    parser.add_argument("--allow-diagnostic-experts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
    experts_dir = Path(args.experts_dir) if args.experts_dir else domain_dir / "domain_mlp_experts"
    router_path = Path(args.router_path) if args.router_path else domain_dir / "router/router.pt"

    device = find_device()
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    if args.mode == "single-expert":
        if not args.expert_path:
            raise ValueError("--expert-path is required for --mode single-expert")
        build_single_expert_model(
            model=model,
            expert_path=Path(args.expert_path),
            layer_num=args.layer_num,
            model_name=args.model_name,
        )
        print(f"Loaded single expert: {args.expert_path}")
    else:
        build_hard_routed_moe(
            model=model,
            experts_dir=experts_dir,
            router_path=router_path,
            layer_num=args.layer_num,
            model_name=args.model_name,
            allow_diagnostic_experts=args.allow_diagnostic_experts,
        )
        print(f"Assembled hard-routed MoE from {experts_dir} and {router_path}")


if __name__ == "__main__":
    main()
