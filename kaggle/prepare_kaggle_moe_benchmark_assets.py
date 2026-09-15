"""Package the independent benchmark and mask-only MoE artifacts for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


DATASET_METADATA_FILENAME = "dataset-metadata.json"

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
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> Dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_files(files: Iterable[Path], destination: Path) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in files:
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def copy_relative_files(
    relative_paths: Iterable[str], source_root: Path, destination: Path
) -> List[Path]:
    copied = []
    for relative_name in relative_paths:
        source = source_root / relative_name
        if not source.exists():
            raise FileNotFoundError(source)
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle assets for Gemma MoE evaluation.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="Defaults to <experiment-dir>/independent_benchmark_v3_sealed.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", default="kaggle/assets/gemma_moe_benchmark_v3")
    parser.add_argument(
        "--kaggle-dataset-id",
        default="erykmikoajek/gemma-moe-benchmark-assets-v3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    experiment_dir = Path(args.experiment_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    benchmark_dir = (
        Path(args.benchmark_dir).resolve()
        if args.benchmark_dir
        else experiment_dir / "independent_benchmark_v3_sealed"
    )
    experts_dir = experiment_dir / "domain_mlp_experts_floor75"
    router_dir = experiment_dir / "router"

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty asset directory: {output_dir}. "
            "Choose a new --output-dir after reviewing the old package."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    copied.extend(
        copy_files(
            [benchmark_dir / "benchmark.jsonl", benchmark_dir / "benchmark_manifest.json"],
            output_dir / "benchmark",
        )
    )
    copied.extend(
        copy_files(
            [
                experts_dir / "pruning_summary.json",
                *[experts_dir / f"domain_{domain_id}_mask.npy" for domain_id in range(6)],
            ],
            output_dir / "experts",
        )
    )
    copied.extend(copy_files([router_dir / "router.pt"], output_dir / "router"))
    copied.extend(copy_relative_files(CODE_FILES, workspace_root, output_dir / "code"))

    with (router_dir / "router_metrics.json").open("r", encoding="utf-8") as handle:
        router_metrics = json.load(handle)
    with (benchmark_dir / "benchmark_manifest.json").open("r", encoding="utf-8") as handle:
        benchmark_manifest = json.load(handle)
    with (experts_dir / "pruning_summary.json").open("r", encoding="utf-8") as handle:
        pruning = json.load(handle)

    base_files = []
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        base_files.append(
            {"filename": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "format": "gemma_moe_kaggle_assets_v1",
        "kaggle_model_source": "google/gemma-3/transformers/gemma-3-270m/2",
        "model_identity": str(pruning["model_name"]),
        "layer_num": int(pruning["layer_num"]),
        "expert_source": "base_masks",
        "confidence_threshold": float(
            router_metrics["confidence_calibration"]["confidence_threshold"]
        ),
        "domain_ids": list(router_metrics["domain_ids"]),
        "domain_names": list(router_metrics["domain_names"]),
        "benchmark_format": benchmark_manifest["format"],
        "benchmark_sha256": benchmark_manifest["benchmark_file"]["sha256"],
        "base_model_files": base_files,
        "runtime_files": [file_record(path, output_dir) for path in sorted(copied)],
        "security": {
            "benchmark_answers_present": True,
            "training_on_assets_forbidden": True,
            "generated_python_execution_default": False,
        },
    }
    manifest_path = output_dir / "asset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)

    readme = output_dir / "README.txt"
    readme.write_text(
        "Gemma MoE independent benchmark assets\n"
        "=====================================\n\n"
        "Upload this directory as a PRIVATE Kaggle Dataset and attach it to "
        "kaggle_gemma_moe_benchmark.ipynb. Also attach Kaggle Model "
        "google/gemma-3/transformers/gemma-3-270m/2. Do not use benchmark.jsonl "
        "for training, prompt selection, threshold calibration, or pruning.\n",
        encoding="utf-8",
    )
    attribution = output_dir / "ATTRIBUTION.md"
    source_lines = [
        f"- `{source['repository']}`, revision `{source['revision']}`, "
        f"license {source['license']}."
        for source in benchmark_manifest["sources"]
    ]
    attribution.write_text(
        "# Benchmark data attribution\n\n"
        + "\n".join(source_lines)
        + "\n\nSynthetic elementary arithmetic was generated locally and released as CC0; "
        "the remaining entries retain the licenses listed above.\n\n"
        "This package contains evaluation subsets and must not be used for training.\n",
        encoding="utf-8",
    )
    dataset_metadata = {
        "id": args.kaggle_dataset_id,
        "title": "Gemma MoE benchmark assets v3",
        "licenses": [{"name": "other"}],
    }
    with (output_dir / DATASET_METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(dataset_metadata, handle, ensure_ascii=False, indent=2)
    archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
    print(f"Kaggle assets: {output_dir}")
    print(f"Upload archive: {archive}")


if __name__ == "__main__":
    main()
