from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_record(path: Path, workspace_root: Path) -> Dict[str, object]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        relative = path.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError(f"Bundle dependency is outside the workspace: {path}") from error
    return {
        "path": str(relative),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a verified mask-only runtime bundle for the Gemma domain MoE."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    experiment_dir = Path(args.experiment_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else experiment_dir / "moe_bundle.json"
    )
    experts_dir = experiment_dir / "domain_mlp_experts_floor75"
    router_path = experiment_dir / "router" / "router.pt"

    domains_payload = load_json(experiment_dir / "domains.json")
    pruning = load_json(experts_dir / "pruning_summary.json")
    router_metrics = load_json(experiment_dir / "router" / "router_metrics.json")
    holdout = load_json(
        experiment_dir / "moe_validation_holdout_frozen" / "ppl_results.json"
    )
    domain_rows = {int(row["domain_id"]): row for row in pruning["domains"]}

    domains: List[Dict[str, object]] = []
    mask_paths: List[Path] = []
    for domain in domains_payload["domains"]:
        domain_id = int(domain["domain_id"])
        if domain_id not in domain_rows:
            raise ValueError(f"Pruning metadata has no selected domain {domain_id}")
        pruning_row = domain_rows[domain_id]
        mask_path = experts_dir / f"domain_{domain_id}_mask.npy"
        keep_mask = np.load(mask_path, allow_pickle=False)
        if keep_mask.ndim != 1 or keep_mask.dtype != np.bool_:
            raise ValueError(f"Invalid boolean expert mask: {mask_path}")
        n_kept = int(keep_mask.sum())
        if n_kept != int(pruning_row["n_kept"]):
            raise ValueError(
                f"Mask/summary mismatch for domain {domain_id}: "
                f"{n_kept} != {pruning_row['n_kept']}"
            )
        domains.append(
            {
                "class_idx": len(domains),
                "domain_id": domain_id,
                "domain_name": domain["name"],
                "n_inner": int(pruning_row["n_inner"]),
                "n_kept": n_kept,
                "n_pruned": int(pruning_row["n_pruned"]),
                "mask": relative_record(mask_path, workspace_root),
            }
        )
        mask_paths.append(mask_path)

    if [row["domain_id"] for row in domains] != list(router_metrics["domain_ids"]):
        raise ValueError("Domain order differs between domains.json and the router")
    if [row["domain_name"] for row in domains] != list(router_metrics["domain_names"]):
        raise ValueError("Domain names differ between domains.json and the router")

    model_files = [
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    ]
    support_files = [
        experts_dir / "pruning_summary.json",
        router_path,
        workspace_root / "moe" / "moe_assembly.py",
    ]
    runtime_paths = [*model_files, *support_files, *mask_paths]
    calibration = router_metrics["confidence_calibration"]
    protocol_path = experiment_dir / "holdout_protocol_frozen.json"
    result_path = experiment_dir / "moe_validation_holdout_frozen" / "ppl_results.json"
    telemetry_path = experiment_dir / "moe_routing_telemetry_corrected" / "ppl_results.json"
    audit_path = experiment_dir / "post_holdout_telemetry_correction.json"

    runtime_model_name = str(model_dir.relative_to(workspace_root))
    runtime_experts_dir = str(experts_dir.relative_to(workspace_root))
    runtime_router_path = str(router_path.relative_to(workspace_root))
    bundle = {
        "format": "gemma_domain_moe_bundle_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_base": "workspace_root containing moe/moe_assembly.py",
        "status": "evaluated proof of concept; holdout is consumed and must not be used for tuning",
        "runtime": {
            "model_name": runtime_model_name,
            "layer_num": int(pruning["layer_num"]),
            "experts_dir": runtime_experts_dir,
            "router_path": runtime_router_path,
            "expert_source": "base_masks",
            "confidence_threshold": float(calibration["confidence_threshold"]),
            "routing": "hard top-1 with dense base-MLP fallback below threshold",
            "assembly_command": f".venv/bin/python -m moe.moe_assembly --bundle {output.relative_to(workspace_root)}",
        },
        "domains": domains,
        "runtime_files": [relative_record(path, workspace_root) for path in runtime_paths],
        "provenance": {
            "frozen_protocol": relative_record(protocol_path, workspace_root),
            "holdout_results": relative_record(result_path, workspace_root),
            "corrected_routing_telemetry": relative_record(telemetry_path, workspace_root),
            "post_holdout_correction_audit": relative_record(audit_path, workspace_root),
        },
        "evaluation_summary": holdout["summary"],
        "storage_note": (
            "Experts are reconstructed from the immutable base layer and boolean masks. "
            "The legacy full masked state_dict files are not runtime dependencies."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Verified MoE bundle written to: {output}")


if __name__ == "__main__":
    main()
