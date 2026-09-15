from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from domain_mapping.domain_mlp_activation_mapping import describe_mlp, get_mlp_module, set_mlp_module
from moe.router_training import LinearDomainRouter
from domain_mapping.topographic_mlp_sae_mapping import DEFAULT_DATA_PATH, DEFAULT_LAYER_NUM, MODEL_NAME, resolve_domain_dir
from common.utils import find_device


def get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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


def load_mask_expert_bank(
    experts_dir: Path,
    base_mlp: nn.Module,
    domain_ids: Sequence[int],
    layer_num: int,
    model_name: str,
    device: torch.device,
    allow_diagnostic_experts: bool = False,
) -> nn.ModuleDict:
    """Reconstruct compact experts from the base MLP and boolean masks only."""
    summary_path = experts_dir / "pruning_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Pruning summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model_name") != model_name or int(payload.get("layer_num", -1)) != layer_num:
        raise ValueError(
            "Mask expert metadata does not match the requested base model and layer"
        )
    pruning_summary = {
        int(domain["domain_id"]): domain for domain in payload.get("domains", [])
    }
    expert_bank = nn.ModuleDict()
    for class_idx, domain_id in enumerate(domain_ids):
        domain_summary = pruning_summary.get(int(domain_id))
        if domain_summary is None:
            raise ValueError(f"Pruning summary has no domain {domain_id}")
        if domain_summary.get("diagnostic_empty_expert") and not allow_diagnostic_experts:
            raise ValueError(
                f"Expert for domain {domain_id} is marked diagnostic_empty_expert"
            )
        mask_path = experts_dir / f"domain_{domain_id}_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Expert mask not found: {mask_path}")
        keep_array = np.load(mask_path, allow_pickle=False)
        if keep_array.ndim != 1 or keep_array.dtype != np.bool_:
            raise ValueError(f"Expert mask must be a one-dimensional boolean array: {mask_path}")
        n_kept = int(keep_array.sum())
        if n_kept != int(domain_summary.get("n_kept", -1)):
            raise ValueError(
                f"Expert mask {mask_path} keeps {n_kept} neurons, but the summary declares "
                f"{domain_summary.get('n_kept')}"
            )
        if n_kept == 0 and not allow_diagnostic_experts:
            raise ValueError(f"Expert mask for domain {domain_id} is empty")
        expert_bank[str(class_idx)] = CompactPrunedMLP(
            base_mlp, torch.from_numpy(keep_array)
        ).to(device).eval()
    return expert_bank


def clone_mlp_with_state(base_mlp: nn.Module, state_dict: Dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    expert = copy.deepcopy(base_mlp)
    expert.load_state_dict(state_dict)
    expert.to(device)
    expert.eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    return expert


def _sliced_linear(
    source: nn.Linear,
    row_indices: Optional[torch.Tensor] = None,
    column_indices: Optional[torch.Tensor] = None,
) -> nn.Linear:
    weight = source.weight.detach()
    if row_indices is not None:
        weight = weight.index_select(0, row_indices.to(weight.device))
    if column_indices is not None:
        weight = weight.index_select(1, column_indices.to(weight.device))
    bias = source.bias.detach() if source.bias is not None else None
    if bias is not None and row_indices is not None:
        bias = bias.index_select(0, row_indices.to(bias.device))
    result = nn.Linear(
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        result.weight.copy_(weight)
        if bias is not None:
            result.bias.copy_(bias)
    return result


class CompactPrunedMLP(nn.Module):
    """Structurally compact MLP reconstructed from a masked dense expert."""

    def __init__(self, dense_mlp: nn.Module, keep_mask: torch.Tensor):
        super().__init__()
        spec = describe_mlp(dense_mlp)
        keep_indices = torch.nonzero(
            keep_mask.to(dtype=torch.bool, device="cpu"), as_tuple=False
        ).flatten()
        input_name = str(spec["input_projections"][0])
        n_inner = int(getattr(dense_mlp, input_name).weight.shape[0])
        if keep_mask.numel() != n_inner:
            raise ValueError(
                f"keep_mask has {keep_mask.numel()} entries, but {input_name} has {n_inner} neurons"
            )
        if keep_indices.numel() == 0:
            raise ValueError("Cannot materialize a compact expert with zero neurons")
        self.family = str(spec["family"])
        self.input_projections = nn.ModuleDict(
            {
                str(name): _sliced_linear(getattr(dense_mlp, str(name)), row_indices=keep_indices)
                for name in spec["input_projections"]
            }
        )
        output_name = str(spec["output_projection"])
        self.output_projection = _sliced_linear(
            getattr(dense_mlp, output_name), column_indices=keep_indices
        )
        if self.family == "gated_mlp":
            self.activation = copy.deepcopy(dense_mlp.act_fn)
            self.dropout = nn.Identity()
        else:
            self.activation = copy.deepcopy(dense_mlp.act)
            self.dropout = copy.deepcopy(getattr(dense_mlp, "dropout", nn.Identity()))
        self.n_kept = int(keep_indices.numel())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.family == "gated_mlp":
            gated = self.activation(self.input_projections["gate_proj"](hidden_states))
            up = self.input_projections["up_proj"](hidden_states)
            return self.output_projection(gated * up)
        input_name = "c_fc" if self.family == "gpt_neo" else "dense_h_to_4h"
        hidden_states = self.activation(self.input_projections[input_name](hidden_states))
        hidden_states = self.output_projection(hidden_states)
        return self.dropout(hidden_states)


def compact_mlp_from_artifact(dense_mlp: nn.Module, artifact: Dict[str, object]) -> nn.Module:
    keep_mask = artifact.get("keep_mask")
    if not isinstance(keep_mask, torch.Tensor):
        raise ValueError("Expert artifact does not contain a tensor keep_mask")
    artifact_family = artifact.get("mlp_family")
    actual_family = describe_mlp(dense_mlp)["family"]
    if artifact_family not in (None, actual_family):
        raise ValueError(
            f"Expert MLP family {artifact_family!r} does not match base MLP {actual_family!r}"
        )
    return CompactPrunedMLP(dense_mlp, keep_mask=keep_mask)


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
        keep_mask = artifact.get("keep_mask")
        if (
            isinstance(keep_mask, torch.Tensor)
            and not bool(keep_mask.any())
            and not allow_diagnostic_experts
        ):
            raise ValueError(
                f"Expert for domain {domain_id} has an empty keep_mask. "
                "Pass allow_diagnostic_experts=True only for diagnostics."
            )
        expert = clone_mlp_with_state(
            base_mlp=base_mlp,
            state_dict=artifact["mlp_state_dict"],
            device=device,
        )
        if bool(artifact["keep_mask"].any()):
            expert = compact_mlp_from_artifact(expert, artifact).to(device).eval()
        expert_bank[str(class_idx)] = expert

    return expert_bank


class HardRoutedMLP(nn.Module):
    def __init__(
        self,
        router: LinearDomainRouter,
        experts: nn.ModuleDict,
        domain_ids: Sequence[int],
        domain_names: Sequence[str],
        fallback_expert: nn.Module,
        confidence_threshold: float = 0.6,
    ):
        super().__init__()
        if len(experts) != len(domain_ids):
            raise ValueError(f"Expected {len(domain_ids)} experts, got {len(experts)}")
        self.router = router
        self.experts = experts
        self.domain_ids = [int(domain_id) for domain_id in domain_ids]
        self.domain_names = list(domain_names)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.fallback_expert = fallback_expert
        self.confidence_threshold = float(confidence_threshold)
        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_expert_ids: Optional[torch.Tensor] = None
        self._routing_token_mask: Optional[torch.Tensor] = None
        self._routing_counts = torch.zeros(len(domain_ids) + 1, dtype=torch.long)

    def set_routing_token_mask(self, token_mask: torch.Tensor) -> None:
        """Supply the attention mask used only for valid-token telemetry."""
        self._routing_token_mask = token_mask.detach().to(dtype=torch.bool)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, original_shape[-1])
        router_logits = self.router(flat_hidden.float())
        probabilities = torch.softmax(router_logits, dim=-1)
        confidence, expert_ids = torch.max(probabilities, dim=-1)
        fallback_mask = confidence < self.confidence_threshold
        if self._routing_token_mask is None:
            valid_token_mask = torch.ones_like(fallback_mask, dtype=torch.bool)
        else:
            valid_token_mask = self._routing_token_mask.to(hidden_states.device).reshape(-1)
            self._routing_token_mask = None
            if valid_token_mask.numel() != fallback_mask.numel():
                raise ValueError(
                    "Routing telemetry mask shape does not match the MLP token shape"
                )

        output = torch.empty_like(flat_hidden)
        if fallback_mask.any():
            output[fallback_mask] = self.fallback_expert(flat_hidden[fallback_mask])
        for class_idx, expert in self.experts.items():
            class_idx_int = int(class_idx)
            token_mask = (expert_ids == class_idx_int) & ~fallback_mask
            if not token_mask.any():
                continue
            # Evaluate an expert only on the tokens assigned to it.  The old
            # implementation ran every expert over the full [B,S,D] tensor and
            # discarded almost all outputs.
            output[token_mask] = expert(flat_hidden[token_mask])

        counts = torch.bincount(
            expert_ids[~fallback_mask & valid_token_mask], minlength=len(self.domain_ids)
        )
        self._routing_counts[:-1] += counts.detach().cpu()
        self._routing_counts[-1] += int((fallback_mask & valid_token_mask).sum().item())
        self.last_router_logits = router_logits.detach().cpu()
        routed_ids = expert_ids.detach().clone()
        routed_ids[fallback_mask] = -1
        routed_ids[~valid_token_mask] = -2
        self.last_expert_ids = routed_ids.reshape(original_shape[:-1]).cpu()
        return output.reshape(original_shape)

    def reset_routing_stats(self) -> None:
        self._routing_counts.zero_()

    def routing_stats(self) -> Dict[str, object]:
        counts = self._routing_counts.tolist()
        total = int(sum(counts))
        return {
            "confidence_threshold": self.confidence_threshold,
            "total_tokens": total,
            "experts": [
                {
                    "class_idx": class_idx,
                    "domain_id": self.domain_ids[class_idx],
                    "domain_name": self.domain_names[class_idx],
                    "tokens": int(counts[class_idx]),
                    "fraction": float(counts[class_idx] / total) if total else 0.0,
                }
                for class_idx in range(len(self.domain_ids))
            ],
            "fallback_tokens": int(counts[-1]),
            "fallback_fraction": float(counts[-1] / total) if total else 0.0,
        }


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
    if bool(artifact["keep_mask"].any()):
        expert = compact_mlp_from_artifact(expert, artifact).to(device).eval()
    set_mlp_module(model, layer_num, expert)
    return model


def build_single_mask_expert_model(
    model,
    experts_dir: Path,
    domain_id: int,
    layer_num: int,
    model_name: str = MODEL_NAME,
    allow_diagnostic_experts: bool = False,
):
    """Replace one MLP with one compact expert reconstructed from a base mask.

    Unlike the routed MoE, this model has no router and no dense fallback: the
    selected domain expert processes every token. This is the appropriate
    construction for measuring a standalone expert's quality and runtime.
    """
    device = get_module_device(model)
    base_mlp = get_mlp_module(model, layer_num)
    bank = load_mask_expert_bank(
        experts_dir=experts_dir,
        base_mlp=base_mlp,
        domain_ids=[int(domain_id)],
        layer_num=layer_num,
        model_name=model_name,
        device=device,
        allow_diagnostic_experts=allow_diagnostic_experts,
    )
    expert = bank["0"]
    set_mlp_module(model, layer_num, expert)
    return model


def build_hard_routed_moe(
    model,
    experts_dir: Path,
    router_path: Path,
    layer_num: int,
    model_name: str = MODEL_NAME,
    allow_diagnostic_experts: bool = False,
    confidence_threshold: Optional[float] = None,
    expert_source: str = "artifact",
):
    device = get_module_device(model)
    router, router_checkpoint = load_router_checkpoint(router_path, device=device)
    if confidence_threshold is None:
        confidence_threshold = float(router_checkpoint.get("confidence_threshold", 0.6))
    validate_router_model_compatibility(
        model=model,
        router_checkpoint=router_checkpoint,
        layer_num=layer_num,
        model_name=model_name,
    )
    domain_ids = [int(domain_id) for domain_id in router_checkpoint["domain_ids"]]
    domain_names = list(router_checkpoint["domain_names"])
    base_mlp = get_mlp_module(model, layer_num)
    if expert_source == "artifact":
        expert_bank = load_expert_bank(
            experts_dir=experts_dir,
            base_mlp=base_mlp,
            domain_ids=domain_ids,
            layer_num=layer_num,
            model_name=model_name,
            device=device,
            allow_diagnostic_experts=allow_diagnostic_experts,
        )
    elif expert_source == "base_masks":
        expert_bank = load_mask_expert_bank(
            experts_dir=experts_dir,
            base_mlp=base_mlp,
            domain_ids=domain_ids,
            layer_num=layer_num,
            model_name=model_name,
            device=device,
            allow_diagnostic_experts=allow_diagnostic_experts,
        )
    else:
        raise ValueError(f"Unsupported expert_source: {expert_source!r}")
    routed_mlp = HardRoutedMLP(
        router=router,
        experts=expert_bank,
        domain_ids=domain_ids,
        domain_names=domain_names,
        fallback_expert=base_mlp,
        confidence_threshold=confidence_threshold,
    )
    routed_mlp.to(device)
    routed_mlp.eval()
    set_mlp_module(model, layer_num, routed_mlp)
    return model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_moe_bundle(bundle_path: Path) -> Tuple[Dict[str, object], Path]:
    """Load a bundle manifest and verify every runtime dependency by SHA-256."""
    bundle_path = bundle_path.resolve()
    with bundle_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)
    if bundle.get("format") != "gemma_domain_moe_bundle_v1":
        raise ValueError(f"Unsupported MoE bundle format: {bundle.get('format')}")

    workspace_root = next(
        (
            parent
            for parent in [bundle_path.parent, *bundle_path.parents]
            if (parent / "moe" / "moe_assembly.py").exists()
        ),
        None,
    )
    if workspace_root is None:
        raise ValueError("Could not locate the workspace root containing moe/moe_assembly.py")
    for record in bundle.get("runtime_files", []):
        path = workspace_root / str(record["path"])
        if not path.exists():
            raise FileNotFoundError(f"Bundle dependency not found: {path}")
        expected_size = int(record["bytes"])
        if path.stat().st_size != expected_size:
            raise ValueError(f"Bundle dependency size changed: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"Bundle dependency hash changed: {path}")
    return bundle, workspace_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble domain MLP experts into a hard-routed MoE model.")
    parser.add_argument(
        "--bundle",
        default=None,
        help="Verified gemma_domain_moe_bundle_v1 manifest; supplies all MoE paths and settings.",
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--domain-dir", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--mode", choices=["single-expert", "moe"], default=None)
    parser.add_argument("--expert-path", default=None)
    parser.add_argument("--experts-dir", default=None)
    parser.add_argument("--router-path", default=None)
    parser.add_argument("--allow-diagnostic-experts", action="store_true")
    parser.add_argument(
        "--router-confidence-threshold",
        type=float,
        default=None,
        help="Override the development-calibrated threshold stored in router.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.router_confidence_threshold is not None and not 0.0 <= args.router_confidence_threshold <= 1.0:
        raise ValueError("--router-confidence-threshold must be between 0 and 1")
    expert_source = "artifact"
    if args.bundle:
        if args.mode not in (None, "moe"):
            raise ValueError("--bundle can only assemble --mode moe")
        if args.router_confidence_threshold is not None:
            raise ValueError(
                "A verified --bundle does not permit a router-threshold override; "
                "use the explicit non-bundle CLI for a new diagnostic configuration"
            )
        bundle, workspace_root = load_moe_bundle(Path(args.bundle))
        runtime = bundle["runtime"]
        model_identity = str(runtime["model_name"])
        model_load_path = str(workspace_root / model_identity)
        layer_num = int(runtime["layer_num"])
        experts_dir = workspace_root / str(runtime["experts_dir"])
        router_path = workspace_root / str(runtime["router_path"])
        confidence_threshold = float(runtime["confidence_threshold"])
        expert_source = str(runtime["expert_source"])
        mode = "moe"
    else:
        if args.mode is None:
            raise ValueError("--mode is required unless --bundle is supplied")
        domain_dir = resolve_domain_dir(args.domain_dir, args.data_path)
        experts_dir = Path(args.experts_dir) if args.experts_dir else domain_dir / "domain_mlp_experts"
        router_path = Path(args.router_path) if args.router_path else domain_dir / "router/router.pt"
        model_identity = args.model_name
        model_load_path = args.model_name
        layer_num = args.layer_num
        confidence_threshold = args.router_confidence_threshold
        mode = args.mode

    device = find_device()
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(model_load_path).to(device)
    model.eval()

    if mode == "single-expert":
        if not args.expert_path:
            raise ValueError("--expert-path is required for --mode single-expert")
        build_single_expert_model(
            model=model,
            expert_path=Path(args.expert_path),
            layer_num=layer_num,
            model_name=model_identity,
        )
        print(f"Loaded single expert: {args.expert_path}")
    else:
        build_hard_routed_moe(
            model=model,
            experts_dir=experts_dir,
            router_path=router_path,
            layer_num=layer_num,
            model_name=model_identity,
            allow_diagnostic_experts=args.allow_diagnostic_experts,
            confidence_threshold=confidence_threshold,
            expert_source=expert_source,
        )
        print(
            f"Assembled hard-routed MoE from {experts_dir} and {router_path} "
            f"(expert_source={expert_source})"
        )


if __name__ == "__main__":
    main()
