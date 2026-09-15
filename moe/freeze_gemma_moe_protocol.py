from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def records(paths: Iterable[Path]) -> Dict[str, Dict[str, object]]:
    return {str(path): file_record(path) for path in sorted(set(paths))}


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the Gemma MoE holdout protocol.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    model_dir = Path(args.model_dir)
    output = Path(args.output) if args.output else experiment_dir / "holdout_protocol_frozen.json"
    experts_dir = experiment_dir / "domain_mlp_experts_floor75"
    router_dir = experiment_dir / "router"
    mapping_dir = experiment_dir / "domain_mlp_mapping"

    dataset_manifest = load_json(experiment_dir / "dataset_manifest.json")
    pruning = load_json(experts_dir / "pruning_summary.json")
    router = load_json(router_dir / "router_metrics.json")
    selectivity = load_json(mapping_dir / "selectivity" / "selectivity_results.json")
    aggressive_dev = load_json(
        experiment_dir / "moe_validation_development_smoke" / "ppl_results.json"
    )["summary"]
    chosen_dev = load_json(
        experiment_dir / "moe_validation_development_floor75" / "ppl_results.json"
    )["summary"]

    holdout_paths = [experiment_dir / "holdout_domains.csv", experiment_dir / "holdout_general.csv"]
    for path in holdout_paths:
        expected = dataset_manifest["files"][str(path)]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Holdout hash changed for {path}: {actual} != {expected}")
    if not selectivity.get("all_domains_pass"):
        raise ValueError("Cannot freeze: not all SAE domains passed selectivity")
    kept = {int(row["n_kept"]) for row in pruning["domains"]}
    if kept != {1536}:
        raise ValueError(f"Expected the chosen 75% operational masks, got widths {sorted(kept)}")
    if router.get("holdout_used") is not False:
        raise ValueError("Router metadata does not explicitly state holdout_used=false")

    source_paths = [
        Path(name)
        for name in [
            "domain_mapping/domain_mlp_activation_mapping.py",
            "domain_mapping/validate_domain_sae_selectivity.py",
            "domain_mapping/domain_mlp_pruning.py",
            "moe/router_training.py",
            "moe/moe_assembly.py",
            "evaluation/moe_validation.py",
            "domain_mapping/prepare_domain_datasets.py",
            "domain_triage/prepare_gemma_domain_experiment.py",
        ]
    ]
    artifact_paths = [
        experiment_dir / "domains.json",
        experiment_dir / "dataset_manifest.json",
        mapping_dir / "summary.json",
        mapping_dir / "selectivity" / "selectivity_results.json",
        experts_dir / "pruning_summary.json",
        router_dir / "router.pt",
        router_dir / "router_metrics.json",
        *sorted(experts_dir.glob("domain_*_mask.npy")),
        *holdout_paths,
    ]
    model_paths = [
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    ]
    calibration = router["confidence_calibration"]
    protocol = {
        "format": "gemma_moe_holdout_protocol_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_status_at_freeze": "sealed; hashes verified; no metric used for selection",
        "model": {
            "path": str(model_dir),
            "layer_replaced": 9,
            "architecture": "Gemma3ForCausalLM; layer-9 gated MLP",
            "files": records(model_paths),
        },
        "domains": [
            "prawo i orzecznictwo",
            "biomedycyna",
            "sport",
            "polityka i wiadomości",
            "matematyka / zapis LaTeX",
            "Python",
        ],
        "development_decision": {
            "statistical_masks": {
                "null": "100 block permutations; block=32; 95th percentile of max |r|",
                "observed_keep_range": [1272, 1408],
                "development_summary": aggressive_dev,
            },
            "chosen_operational_masks": {
                "rule": "statistical mask plus min_keep_fraction=0.75",
                "kept_per_expert": 1536,
                "pruned_per_expert": 512,
                "reason": "all six single experts and the routed MoE stayed within the prespecified 10% development PPL-regression gate",
                "development_summary": chosen_dev,
            },
        },
        "router": {
            "type": "linear token router on layer-9 MLP inputs",
            "checkpoint": str(router_dir / "router.pt"),
            "confidence_threshold": calibration["confidence_threshold"],
            "threshold_calibration": calibration,
            "fallback": "original dense layer-9 MLP",
        },
        "final_evaluation": {
            "domain_csv": str(experiment_dir / "holdout_domains.csv"),
            "general_csv": str(experiment_dir / "holdout_general.csv"),
            "domain_texts": 540,
            "general_texts": 100,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "primary_comparisons": ["base", "single_semantic_expert", "hard_routed_moe"],
            "matched_controls": [
                "one seeded random mask with identical width per domain",
                "weight-magnitude mask with identical width per domain",
            ],
            "general_ppl_regression_gate": 0.10,
            "domain_ppl_regression_gate": 0.10,
            "no_post_holdout_tuning": True,
        },
        "known_limitations_fixed_before_evaluation": [
            "Domain labels are also correlated with source and genre; this PoC cannot fully separate topic from dataset style.",
            "Only one transformer layer is converted to MoE.",
            "Perplexity tests knowledge preservation, not factual QA accuracy.",
            "One random-mask replicate is a lightweight control, not a stable Monte Carlo estimate.",
        ],
        "dataset_integrity": dataset_manifest["integrity"],
        "source_code_files": records(source_paths),
        "artifact_files": records(artifact_paths),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False)
    print(f"Frozen protocol written to: {output}")


if __name__ == "__main__":
    main()
