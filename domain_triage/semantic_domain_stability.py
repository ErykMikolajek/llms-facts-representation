"""Stability audit for semantic-domain clustering of SAE feature vectors.

The script builds the feature-token matrix exactly once from a completed
feature analysis, then repeats SVD + HDBSCAN under several defensible
perturbations.  Baseline clusters are matched to the most similar cluster in
each alternative run by Jaccard overlap.  The output is a candidate report for
human review; it does not approve domains for pruning or router training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from transformers import AutoTokenizer

from domain_triage.semantic_domain_triage import (
    DomainInfo,
    build_validation_sets_from_feature_analysis,
    cluster_features,
    compute_cluster_candidates,
    load_precomputed_feature_analysis,
    reduce_feature_matrix,
    validate_feature_analysis_contract,
    write_json,
)
from common.utils import tokenizer_vocab_fingerprint


DEFAULT_RUNS = (
    {"name": "baseline", "svd_components": 50, "seed": 0, "min_cluster_size": 40, "min_samples": 10, "selection_method": "leaf"},
    {"name": "svd_30", "svd_components": 30, "seed": 0, "min_cluster_size": 40, "min_samples": 10, "selection_method": "leaf"},
    {"name": "svd_80", "svd_components": 80, "seed": 0, "min_cluster_size": 40, "min_samples": 10, "selection_method": "leaf"},
    {"name": "seed_1", "svd_components": 50, "seed": 1, "min_cluster_size": 40, "min_samples": 10, "selection_method": "leaf"},
    {"name": "seed_2", "svd_components": 50, "seed": 2, "min_cluster_size": 40, "min_samples": 10, "selection_method": "leaf"},
    {"name": "clusters_20", "svd_components": 50, "seed": 0, "min_cluster_size": 20, "min_samples": 10, "selection_method": "leaf"},
    {"name": "clusters_80", "svd_components": 50, "seed": 0, "min_cluster_size": 80, "min_samples": 20, "selection_method": "leaf"},
)


def load_context_feature_analysis(
    analysis_path: Path,
    max_features: int,
    min_document_frequency: int,
    max_document_fraction: float,
    top_terms_per_feature: int,
):
    """Represent each SAE feature by words in its strongest activation contexts."""
    with analysis_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_records = payload.get("features", []) if isinstance(payload, dict) else payload
    if isinstance(raw_records, dict):
        raw_records = list(raw_records.values())
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"No feature records found in {analysis_path}")
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    declared_total = summary.get("total_features")
    max_feature_id = max(int(record["feature_id"]) for record in raw_records)
    total_features = int(declared_total) if declared_total is not None else max_feature_id + 1
    documents = [""] * total_features
    observed = 0
    for record in raw_records:
        feature_id = int(record["feature_id"])
        is_observed = bool(
            record.get(
                "observed_in_analysis",
                int(record.get("total_activations", 0)) > 0,
            )
        )
        if not is_observed:
            continue
        observed += 1
        contexts = []
        for example in record.get("top_examples", []):
            text = str(example.get("full_context", ""))
            text = text.replace("[[", " ").replace("]]", " ")
            if text.strip():
                contexts.append(text)
        documents[feature_id] = "\n".join(contexts)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"(?u)\b[^\W\d_][\w'-]{1,}\b",
        min_df=min_document_frequency,
        max_df=max_document_fraction,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents).tocsr()
    terms = vectorizer.get_feature_names_out()
    feature_top_tokens: List[List[Dict[str, float]]] = [
        [] for _ in range(total_features)
    ]
    for feature_id in np.flatnonzero(matrix.getnnz(axis=1) > 0):
        row = matrix.getrow(int(feature_id))
        order = np.argsort(row.data)[::-1][:top_terms_per_feature]
        feature_top_tokens[int(feature_id)] = [
            {
                "token": str(terms[int(row.indices[position])]),
                "normalized": str(terms[int(row.indices[position])]),
                "token_id": int(row.indices[position]),
                "logit": float(row.data[position]),
                "idf": 1.0,
                "source": "top_activation_context_tfidf",
            }
            for position in order
        ]
    metadata = {
        "source": str(analysis_path),
        "source_format": summary,
        "total_features": total_features,
        "observed_features": observed,
        "nonempty_features": int(np.count_nonzero(matrix.getnnz(axis=1))),
        "context_vocabulary_size": int(len(terms)),
        "max_features": int(max_features),
        "min_document_frequency": int(min_document_frequency),
        "max_document_fraction": float(max_document_fraction),
        "top_terms_per_feature": int(top_terms_per_feature),
    }
    vocabulary = {str(term): int(index) for index, term in enumerate(terms)}
    return matrix, feature_top_tokens, metadata, vocabulary


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def label_sets(labels: np.ndarray, feature_ids: np.ndarray) -> Dict[int, set[int]]:
    result: Dict[int, set[int]] = {}
    for label in np.unique(labels):
        label_int = int(label)
        if label_int < 0:
            continue
        result[label_int] = set(feature_ids[labels == label_int].astype(int).tolist())
    return result


def best_jaccard(reference: set[int], alternatives: Dict[int, set[int]]) -> Dict[str, Any]:
    best_label = None
    best_score = 0.0
    best_intersection = 0
    for label, members in alternatives.items():
        intersection = len(reference & members)
        union = len(reference | members)
        score = intersection / union if union else 0.0
        if score > best_score:
            best_label = int(label)
            best_score = float(score)
            best_intersection = int(intersection)
    return {
        "matched_cluster_label": best_label,
        "jaccard": best_score,
        "intersection": best_intersection,
    }


def run_clustering_grid(feature_matrix, runs: Sequence[Dict[str, Any]]):
    results = []
    expected_ids = None
    for config in runs:
        print(
            f"Stability run {config['name']}: SVD={config['svd_components']}, "
            f"seed={config['seed']}, min_cluster_size={config['min_cluster_size']}, "
            f"min_samples={config['min_samples']}, method={config['selection_method']}"
        )
        feature_ids, reduced = reduce_feature_matrix(
            feature_matrix,
            int(config["svd_components"]),
            random_state=int(config["seed"]),
        )
        if expected_ids is None:
            expected_ids = feature_ids
        elif not np.array_equal(expected_ids, feature_ids):
            raise RuntimeError("The non-empty feature set changed between stability runs")
        labels = cluster_features(
            reduced_vectors=reduced,
            min_cluster_size=int(config["min_cluster_size"]),
            min_samples=int(config["min_samples"]),
            cluster_selection_method=str(config["selection_method"]),
        )
        results.append(
            {
                "config": dict(config),
                "feature_ids": feature_ids,
                "reduced": reduced,
                "labels": labels,
                "cluster_count": int(len(set(labels.tolist()) - {-1})),
                "noise_features": int(np.sum(labels < 0)),
                "noise_fraction": float(np.mean(labels < 0)),
            }
        )
    return results


def stability_for_baseline_clusters(results) -> Dict[int, Dict[str, Any]]:
    baseline = results[0]
    baseline_sets = label_sets(baseline["labels"], baseline["feature_ids"])
    alternative_sets = [
        (result["config"]["name"], label_sets(result["labels"], result["feature_ids"]))
        for result in results[1:]
    ]
    output = {}
    for label, members in baseline_sets.items():
        matches = []
        for run_name, clusters in alternative_sets:
            match = best_jaccard(members, clusters)
            match["run"] = run_name
            matches.append(match)
        scores = [float(match["jaccard"]) for match in matches]
        output[int(label)] = {
            "matches": matches,
            "mean_best_jaccard": float(np.mean(scores)) if scores else 1.0,
            "min_best_jaccard": float(np.min(scores)) if scores else 1.0,
            "runs_jaccard_at_least_0_5": int(sum(score >= 0.5 for score in scores)),
            "runs_jaccard_at_least_0_7": int(sum(score >= 0.7 for score in scores)),
        }
    return output


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Semantic-domain stability report\n\n")
        handle.write(
            "> This report is a clustering diagnostic. Candidate names are lexical "
            "proxies and no domain is approved for downstream use.\n\n"
        )
        handle.write("## Input\n\n")
        for key, value in payload["input"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## Stability runs\n\n")
        handle.write(
            "| run | SVD | seed | min cluster | min samples | method | "
            "clusters | noise | ARI (all) | ARI (jointly clustered) |\n"
        )
        handle.write(
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n"
        )
        for run in payload["runs"]:
            handle.write(
                f"| {run['name']} | {run['svd_components']} | {run['seed']} | "
                f"{run['min_cluster_size']} | {run['min_samples']} | {run['selection_method']} | "
                f"{run['cluster_count']} | {run['noise_fraction']:.3f} | "
                f"{run['adjusted_rand_all_features']:.3f} | "
                f"{run['adjusted_rand_jointly_clustered']:.3f} |\n"
            )
        handle.write("\n## Baseline candidates ranked by stability-adjusted score\n\n")
        handle.write("| rank | cluster | size | cohesion | mean Jaccard | min Jaccard | score | tokens |\n")
        handle.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for rank, candidate in enumerate(payload["candidates"], 1):
            tokens = ", ".join(token["normalized"] for token in candidate["top_tokens"][:12])
            handle.write(
                f"| {rank} | {candidate['cluster_label']} | {candidate['size']} | "
                f"{candidate['cohesion']:.3f} | {candidate['mean_best_jaccard']:.3f} | "
                f"{candidate['min_best_jaccard']:.3f} | "
                f"{candidate['stability_adjusted_score']:.3f} | {tokens} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit stability of SAE semantic-domain clustering before manual selection."
    )
    parser.add_argument("--feature-analysis", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--layer-num", type=int, default=9)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-m", type=int, default=128)
    parser.add_argument(
        "--feature-representation",
        choices=["promoted", "contexts"],
        default="promoted",
        help=(
            "Use decoder-to-unembedding promoted tokens or words from the saved "
            "top activation contexts. Both are diagnostics until manual review."
        ),
    )
    parser.add_argument("--context-max-features", type=int, default=50_000)
    parser.add_argument("--context-min-df", type=int, default=3)
    parser.add_argument("--context-max-df", type=float, default=0.20)
    parser.add_argument("--context-top-terms-per-feature", type=int, default=64)
    parser.add_argument("--top-tokens-per-cluster", type=int, default=80)
    parser.add_argument("--min-token-feature-support", type=int, default=2)
    parser.add_argument("--min-token-feature-fraction", type=float, default=0.01)
    parser.add_argument("--max-cluster-fraction", type=float, default=0.25)
    parser.add_argument("--min-distinct-tokens", type=int, default=8)
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument("--contexts-per-candidate", type=int, default=5)
    parser.add_argument("--keep-common-function-tokens", action="store_true")
    parser.add_argument("--no-token-boundary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.top_m,
        args.top_tokens_per_cluster,
        args.min_token_feature_support,
        args.min_distinct_tokens,
        args.shortlist_size,
        args.contexts_per_candidate,
        args.context_max_features,
        args.context_min_df,
        args.context_top_terms_per_feature,
    ) < 1:
        raise ValueError("Positive stability-audit limits are required")
    if not 0.0 < args.max_cluster_fraction <= 1.0:
        raise ValueError("--max-cluster-fraction must be in (0, 1]")
    if not 0.0 <= args.min_token_feature_fraction <= 1.0:
        raise ValueError("--min-token-feature-fraction must be in [0, 1]")
    if not 0.0 < args.context_max_df <= 1.0:
        raise ValueError("--context-max-df must be in (0, 1]")

    analysis_path = Path(args.feature_analysis)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    vocabulary = None
    if args.feature_representation == "promoted":
        print("Building the shared TF-IDF feature-token matrix...")
        matrix, feature_top_tokens, metadata = load_precomputed_feature_analysis(
            analysis_path=analysis_path,
            tokenizer=tokenizer,
            min_logit=0.0,
            keep_common_function_tokens=args.keep_common_function_tokens,
            min_token_chars=2,
            require_token_boundary=not args.no_token_boundary,
            observed_features_only=True,
            use_tfidf=True,
            top_m=args.top_m,
            feature_token_source="promoted",
        )
        validate_feature_analysis_contract(
            matrix_metadata=metadata,
            layer_num=args.layer_num,
            tokenizer=tokenizer,
            requested_top_m=args.top_m,
            allow_mismatch=False,
            feature_token_source="promoted",
        )
    else:
        print("Building the shared TF-IDF activation-context matrix...")
        matrix, feature_top_tokens, metadata, vocabulary = load_context_feature_analysis(
            analysis_path=analysis_path,
            max_features=args.context_max_features,
            min_document_frequency=args.context_min_df,
            max_document_fraction=args.context_max_df,
            top_terms_per_feature=args.context_top_terms_per_feature,
        )
        source_summary = metadata.get("source_format", {})
        if source_summary.get("layer_num") not in (None, args.layer_num):
            raise ValueError(
                f"Context analysis layer={source_summary.get('layer_num')} does not "
                f"match requested layer={args.layer_num}"
            )
    save_npz(output_dir / "feature_token_matrix.npz", matrix)
    if vocabulary is not None:
        write_json(output_dir / "context_vocabulary.json", vocabulary)

    results = run_clustering_grid(matrix, DEFAULT_RUNS)
    baseline = results[0]
    candidates = compute_cluster_candidates(
        labels=baseline["labels"],
        valid_feature_ids=baseline["feature_ids"],
        feature_matrix=matrix,
        reduced_vectors=baseline["reduced"],
        feature_top_tokens=feature_top_tokens,
        top_tokens_per_domain=args.top_tokens_per_cluster,
        min_token_feature_support=args.min_token_feature_support,
        min_token_feature_fraction=args.min_token_feature_fraction,
    )
    stability = stability_for_baseline_clusters(results)
    max_size = max(1, int(len(baseline["feature_ids"]) * args.max_cluster_fraction))
    exported_candidates = []
    for candidate in candidates:
        label = int(candidate["cluster_label"])
        cluster_stability = stability[label]
        base_score = float(candidate["cohesion"] * math.log1p(candidate["size"]))
        exported = {
            "cluster_label": label,
            "size": int(candidate["size"]),
            "cohesion": float(candidate["cohesion"]),
            "top_tokens": candidate["top_tokens"],
            "feature_ids": candidate["feature_ids"],
            **cluster_stability,
            "base_score": base_score,
            "stability_adjusted_score": float(
                base_score * cluster_stability["mean_best_jaccard"]
            ),
            "eligible": bool(
                candidate["size"] <= max_size
                and len(candidate["top_tokens"]) >= args.min_distinct_tokens
            ),
        }
        exported_candidates.append(exported)
    exported_candidates.sort(
        key=lambda item: (
            not item["eligible"],
            -item["stability_adjusted_score"],
            -item["mean_best_jaccard"],
            item["cluster_label"],
        )
    )

    baseline_labels = baseline["labels"]
    run_rows = []
    for result in results:
        if result is baseline:
            ari_all = 1.0
            ari_jointly_clustered = 1.0
            jointly_clustered_features = int(np.sum(baseline_labels >= 0))
        else:
            shared_clustered = (baseline_labels >= 0) & (result["labels"] >= 0)
            ari_all = float(adjusted_rand_score(baseline_labels, result["labels"]))
            ari_jointly_clustered = (
                float(adjusted_rand_score(
                    baseline_labels[shared_clustered], result["labels"][shared_clustered]
                ))
                if int(shared_clustered.sum()) >= 2
                else float("nan")
            )
            jointly_clustered_features = int(shared_clustered.sum())
        run_rows.append(
            {
                **result["config"],
                "cluster_count": result["cluster_count"],
                "noise_features": result["noise_features"],
                "noise_fraction": result["noise_fraction"],
                "adjusted_rand_all_features": ari_all,
                "adjusted_rand_jointly_clustered": ari_jointly_clustered,
                "jointly_clustered_features": jointly_clustered_features,
            }
        )
        np.save(
            output_dir / f"labels_{result['config']['name']}.npy",
            result["labels"].astype(np.int32),
        )
    np.save(output_dir / "valid_feature_ids.npy", baseline["feature_ids"].astype(np.int32))

    payload = {
        "format": "semantic_domain_stability_v1",
        "input": {
            "feature_analysis": str(analysis_path),
            "feature_analysis_sha256": sha256_file(analysis_path),
            "tokenizer_name": args.tokenizer_name,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_vocab_sha256": tokenizer_vocab_fingerprint(tokenizer),
            "layer_num": args.layer_num,
            "top_m": args.top_m if args.feature_representation == "promoted" else None,
            "feature_token_source": args.feature_representation,
            "tfidf": True,
            "require_token_boundary": (
                not args.no_token_boundary
                if args.feature_representation == "promoted"
                else None
            ),
            "observed_features_only": True,
            "nonempty_features": int(np.count_nonzero(matrix.getnnz(axis=1))),
            "representation_metadata": metadata,
        },
        "runs": run_rows,
        "baseline_run": DEFAULT_RUNS[0]["name"],
        "max_eligible_cluster_size": max_size,
        "candidates": exported_candidates,
        "manual_review_required": True,
        "downstream_approved": False,
    }
    write_json(output_dir / "stability_results.json", payload)
    write_report(output_dir / "stability_report.md", payload)

    shortlist = [candidate for candidate in exported_candidates if candidate["eligible"]][
        : args.shortlist_size
    ]
    shortlist_domains: List[DomainInfo] = []
    for domain_id, candidate in enumerate(shortlist):
        name_terms = [token["normalized"] for token in candidate["top_tokens"][:3]]
        shortlist_domains.append(
            DomainInfo(
                domain_id=domain_id,
                cluster_label=int(candidate["cluster_label"]),
                name=" / ".join(name_terms) or f"cluster_{candidate['cluster_label']}",
                size=int(candidate["size"]),
                cohesion=float(candidate["cohesion"]),
                selection_score=float(candidate["stability_adjusted_score"]),
                top_tokens=candidate["top_tokens"],
                feature_ids=candidate["feature_ids"],
            )
        )
    write_json(
        output_dir / "shortlist.json",
        {
            "manual_review_required": True,
            "domains": [asdict(domain) for domain in shortlist_domains],
        },
    )
    context_summary = build_validation_sets_from_feature_analysis(
        domains=shortlist_domains,
        analysis_path=analysis_path,
        output_dir=output_dir / "candidate_contexts",
        samples_per_domain=args.contexts_per_candidate,
    )
    write_json(output_dir / "candidate_contexts_summary.json", context_summary)
    print(f"Stability audit written to: {output_dir}")


if __name__ == "__main__":
    main()
