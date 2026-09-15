"""Package benchmark v3 and standalone domain-expert code for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


CODE_FILES = (
    "common/__init__.py",
    "common/utils.py",
    "sae_pipeline/__init__.py",
    "sae_pipeline/activations_collecting.py",
    "domain_mapping/__init__.py",
    "domain_mapping/domain_mlp_activation_mapping.py",
    "domain_mapping/topographic_mlp_sae_mapping.py",
    "moe/__init__.py",
    "moe/router_training.py",
    "moe/moe_assembly.py",
    "evaluation/__init__.py",
    "evaluation/moe_benchmark.py",
    "evaluation/domain_expert_benchmark.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_relative_files(
    relative_paths: Iterable[str], source_root: Path, destination: Path
) -> list[Path]:
    copied = []
    for relative_name in relative_paths:
        source = source_root / relative_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def copy_files(files: Iterable[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package standalone domain-expert benchmark assets.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--output-dir", default="kaggle/assets/gemma_domain_expert_benchmark_v1"
    )
    parser.add_argument(
        "--kaggle-dataset-id",
        default="erykmikoajek/gemma-domain-expert-benchmark-assets-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    experiment_dir = Path(args.experiment_dir).resolve()
    benchmark_dir = (
        Path(args.benchmark_dir).resolve()
        if args.benchmark_dir
        else experiment_dir / "independent_benchmark_v3_sealed"
    )
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    experts_dir = experiment_dir / "domain_mlp_experts_floor75"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty asset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    copied.extend(copy_files(
        [benchmark_dir / "benchmark.jsonl", benchmark_dir / "benchmark_manifest.json"],
        output_dir / "benchmark",
    ))
    copied.extend(copy_files(
        [
            experts_dir / "pruning_summary.json",
            *[experts_dir / f"domain_{domain_id}_mask.npy" for domain_id in range(6)],
        ],
        output_dir / "experts",
    ))
    copied.extend(copy_relative_files(CODE_FILES, workspace_root, output_dir / "code"))

    benchmark_manifest = json.loads(
        (benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    pruning = json.loads((experts_dir / "pruning_summary.json").read_text(encoding="utf-8"))
    base_files = []
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        base_files.append({
            "filename": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    domains = sorted(pruning["domains"], key=lambda row: int(row["domain_id"]))
    manifest = {
        "format": "gemma_domain_expert_kaggle_assets_v1",
        "kaggle_model_source": "google/gemma-3/transformers/gemma-3-270m/2",
        "model_identity": str(pruning["model_name"]),
        "layer_num": int(pruning["layer_num"]),
        "domain_ids": [int(row["domain_id"]) for row in domains],
        "domain_names": [str(row["domain_name"]) for row in domains],
        "expert_widths": [int(row["n_kept"]) for row in domains],
        "expert_source": "base_masks",
        "benchmark_format": benchmark_manifest["format"],
        "benchmark_sha256": benchmark_manifest["benchmark_file"]["sha256"],
        "base_model_files": base_files,
        "runtime_files": [file_record(path, output_dir) for path in sorted(copied)],
        "methodology": {
            "default_scope": "each standalone expert on its own domain",
            "cross_domain_optional": True,
            "router_present": False,
            "dense_fallback_present": False,
            "benchmark_status": "previously consumed by MoE; standalone analysis is post-hoc",
        },
        "security": {
            "benchmark_answers_present": True,
            "training_on_assets_forbidden": True,
            "generated_python_execution_default": False,
        },
    }
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "Standalone Gemma domain-expert benchmark assets. Upload as a PRIVATE "
        "Kaggle Dataset and attach Gemma 3 270M. Do not train or tune on benchmark v3.\n",
        encoding="utf-8",
    )
    source_lines = [
        f"- `{source['repository']}`, revision `{source['revision']}`, "
        f"license {source['license']}."
        for source in benchmark_manifest["sources"]
    ]
    (output_dir / "ATTRIBUTION.md").write_text(
        "# Benchmark data attribution\n\n"
        + "\n".join(source_lines)
        + "\n\nThis evaluation package must not be used for training or model selection.\n",
        encoding="utf-8",
    )
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps({
            "id": args.kaggle_dataset_id,
            "title": "Gemma standalone domain expert benchmark assets v1",
            "licenses": [{"name": "other"}],
        }, indent=2),
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    print(f"Kaggle assets: {output_dir}")
    print(f"Upload archive: {archive}")


if __name__ == "__main__":
    main()
