"""Validate candidate domain clusters in a second feature representation.

The primary clustering result supplies fixed SAE-feature memberships.  This
script measures their centroid cohesion in a secondary sparse matrix and
compares it with same-size random feature sets.  It is a diagnostic only: a
small p-value does not turn a lexical cluster into an approved semantic domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def centroid_cohesion(matrix: sparse.csr_matrix, feature_ids: Iterable[int]) -> float:
    ids = np.asarray(list(feature_ids), dtype=np.int64)
    if ids.size == 0:
        return float("nan")
    members = matrix[ids]
    centroid = normalize(
        sparse.csr_matrix(members.sum(axis=0)), norm="l2", axis=1
    )
    return float(members.dot(centroid.T).mean())


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Return FDR-adjusted p-values while preserving the input order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_min = 1.0
    count = len(values)
    for rank_index in range(count - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running_min = min(running_min, float(values[original_index]) * count / rank)
        adjusted[original_index] = min(running_min, 1.0)
    return adjusted.tolist()


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Cross-view validation of domain candidates\n\n")
        handle.write(
            "> Candidate memberships are fixed from activation-context clustering. "
            "This report tests cohesion in a secondary representation and does not "
            "approve domains for downstream use.\n\n"
        )
        handle.write("## Method\n\n")
        handle.write(
            f"- secondary representation: {payload['secondary']['label']}\n"
            f"- random sets per candidate: {payload['null_model']['permutations']}\n"
            f"- random seed: {payload['null_model']['seed']}\n"
            "- population: features with a non-zero vector in the secondary matrix\n"
            "- test: one-sided empirical probability of random cohesion >= observed cohesion\n"
            "- multiple testing: Benjamini-Hochberg false-discovery-rate correction\n\n"
        )
        handle.write(
            "| cluster | primary size | evaluated | missing | observed | null mean | "
            "z | p | q (BH) |\n"
        )
        handle.write(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for candidate in payload["candidates"]:
            handle.write(
                f"| {candidate['cluster_label']} | {candidate['primary_size']} | "
                f"{candidate['evaluated_features']} | {candidate['missing_secondary_features']} | "
                f"{candidate['observed_cohesion']:.4f} | "
                f"{candidate['null_mean']:.4f} | {candidate['z_score']:.2f} | "
                f"{candidate['empirical_p_greater_equal']:.4f} | "
                f"{candidate['fdr_bh_q_value']:.4f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test fixed semantic-domain clusters in a secondary feature view."
    )
    parser.add_argument("--primary-results", required=True)
    parser.add_argument("--secondary-matrix", required=True)
    parser.add_argument("--secondary-label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")

    primary_path = Path(args.primary_results)
    secondary_path = Path(args.secondary_matrix)
    with primary_path.open("r", encoding="utf-8") as handle:
        primary = json.load(handle)
    matrix = sparse.load_npz(secondary_path).tocsr()
    population = np.flatnonzero(matrix.getnnz(axis=1) > 0).astype(np.int64)
    population_set = set(population.tolist())
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    for candidate in primary["candidates"]:
        primary_ids = [int(value) for value in candidate["feature_ids"]]
        evaluated_ids = [value for value in primary_ids if value in population_set]
        if len(evaluated_ids) > len(population):
            raise ValueError("Candidate is larger than the secondary population")
        observed = centroid_cohesion(matrix, evaluated_ids)
        null_values = np.asarray(
            [
                centroid_cohesion(
                    matrix,
                    rng.choice(population, size=len(evaluated_ids), replace=False),
                )
                for _ in range(args.permutations)
            ],
            dtype=np.float64,
        )
        null_mean = float(null_values.mean())
        null_std = float(null_values.std(ddof=1))
        z_score = (
            float((observed - null_mean) / null_std)
            if null_std > 0.0
            else float("nan")
        )
        empirical_p = float(
            (1 + int(np.sum(null_values >= observed))) / (args.permutations + 1)
        )
        rows.append(
            {
                "cluster_label": int(candidate["cluster_label"]),
                "primary_size": len(primary_ids),
                "evaluated_features": len(evaluated_ids),
                "missing_secondary_features": len(primary_ids) - len(evaluated_ids),
                "observed_cohesion": observed,
                "null_mean": null_mean,
                "null_std": null_std,
                "z_score": z_score,
                "empirical_p_greater_equal": empirical_p,
            }
        )

    q_values = benjamini_hochberg(
        [row["empirical_p_greater_equal"] for row in rows]
    )
    for row, q_value in zip(rows, q_values):
        row["fdr_bh_q_value"] = float(q_value)

    payload = {
        "format": "semantic_domain_cross_view_v1",
        "primary": {
            "results": str(primary_path),
            "sha256": sha256_file(primary_path),
            "representation": primary.get("input", {}).get("feature_token_source"),
        },
        "secondary": {
            "matrix": str(secondary_path),
            "sha256": sha256_file(secondary_path),
            "label": args.secondary_label,
            "shape": list(matrix.shape),
            "nonempty_features": int(len(population)),
        },
        "null_model": {
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "sample_without_replacement": True,
            "population": "nonempty_secondary_rows",
            "alternative": "greater_or_equal_cohesion",
            "multiple_testing": "Benjamini-Hochberg FDR",
        },
        "candidates": rows,
        "manual_review_required": True,
        "downstream_approved": False,
    }
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_report, payload)
    print(f"Cross-view validation written to: {output_json}")


if __name__ == "__main__":
    main()
