"""Freeze the manually approved Gemma domain selection for downstream work."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


SELECTED_DOMAINS = (
    {"domain_id": 0, "cluster_label": 20, "name": "prawo i orzecznictwo", "slug": "legal"},
    {"domain_id": 1, "cluster_label": 17, "name": "biomedycyna", "slug": "biomedical"},
    {"domain_id": 2, "cluster_label": 16, "name": "sport", "slug": "sports"},
    {"domain_id": 3, "cluster_label": 22, "name": "polityka i wiadomości", "slug": "politics_news"},
    {"domain_id": 4, "cluster_label": 15, "name": "matematyka / zapis LaTeX", "slug": "math_latex"},
    {"domain_id": 5, "cluster_label": 5, "name": "Python", "slug": "python"},
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Zatwierdzone domeny Gemma 3 270M\n\n")
        handle.write(
            "Wybór został jawnie zatwierdzony po analizie raportu stabilności. "
            "Zatwierdzenie pozwala rozpocząć walidację na nowych danych i mapowanie, "
            "ale nie jest dowodem przyczynowej lokalizacji wiedzy.\n\n"
        )
        handle.write("| domain_id | cluster | nazwa | liczba cech | średni Jaccard |\n")
        handle.write("| ---: | ---: | --- | ---: | ---: |\n")
        for domain in payload["domains"]:
            handle.write(
                f"| {domain['domain_id']} | {domain['cluster_label']} | "
                f"{domain['name']} | {domain['size']} | "
                f"{domain['stability']['mean_best_jaccard']:.3f} |\n"
            )
        handle.write("\nEkspert ogólny nie jest siódmym klastrem SAE. W MoE będzie nim "
                     "oryginalny, gęsty MLP używany jako fallback routera.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="models/gemma-3-270m")
    parser.add_argument("--tokenizer-name", default="models/gemma-3-270m")
    parser.add_argument("--layer-num", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.stability_results)
    output_dir = Path(args.output_dir)
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    candidates = {
        int(candidate["cluster_label"]): candidate
        for candidate in source.get("candidates", [])
    }

    domains = []
    all_feature_ids = set()
    for selection in SELECTED_DOMAINS:
        cluster_label = int(selection["cluster_label"])
        if cluster_label not in candidates:
            raise ValueError(f"Selected cluster {cluster_label} is absent from {source_path}")
        candidate = candidates[cluster_label]
        feature_ids = [int(value) for value in candidate["feature_ids"]]
        overlap = all_feature_ids.intersection(feature_ids)
        if overlap:
            raise ValueError(
                f"Selected HDBSCAN clusters unexpectedly overlap on {len(overlap)} features"
            )
        all_feature_ids.update(feature_ids)
        domains.append(
            {
                **selection,
                "size": int(candidate["size"]),
                "cohesion": float(candidate["cohesion"]),
                "selection_score": float(candidate["stability_adjusted_score"]),
                "top_tokens": candidate["top_tokens"],
                "feature_ids": feature_ids,
                "stability": {
                    "mean_best_jaccard": float(candidate["mean_best_jaccard"]),
                    "min_best_jaccard": float(candidate["min_best_jaccard"]),
                    "matches": candidate["matches"],
                },
            }
        )

    input_metadata = source.get("input", {})
    payload = {
        "format": "semantic_domain_triage_v3_manual_selection",
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "layer_num": int(args.layer_num),
        "domains": domains,
        "selected_feature_count": len(all_feature_ids),
        "matrix_metadata": input_metadata.get("representation_metadata", {}),
        "selection_provenance": {
            "source_results": str(source_path),
            "source_results_sha256": sha256_file(source_path),
            "source_feature_analysis": input_metadata.get("feature_analysis"),
            "source_feature_analysis_sha256": input_metadata.get("feature_analysis_sha256"),
            "criterion": "explicit_user_selection_after_stability_and_context_review",
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "manual_review_required": True,
        "downstream_approved": True,
        "approval": {
            "method": "explicit_user_message",
            "scope": "six named domains; independent validation still required",
        },
        "validation_source": "independent_development_dataset_to_be_prepared",
        "general_fallback": {
            "type": "original_dense_mlp",
            "is_sae_cluster": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "domains.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir / "domain_report.md", payload)
    print(f"Approved six-domain manifest written to: {output_dir}")


if __name__ == "__main__":
    main()
