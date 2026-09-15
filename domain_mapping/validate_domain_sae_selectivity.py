from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def aggregate_token_signals_by_sample(
    sample_indices: np.ndarray,
    source_domain_ids: np.ndarray,
    signals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique sample ids, one source label per sample, and sample means."""
    sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    source_domain_ids = np.asarray(source_domain_ids, dtype=np.int64).reshape(-1)
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, None]
    if not (len(sample_indices) == len(source_domain_ids) == signals.shape[0]):
        raise ValueError("Token metadata and SAE signals have different row counts")
    if len(sample_indices) == 0:
        raise ValueError("The pooled mapping contains no tokens")

    unique_samples, inverse = np.unique(sample_indices, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sample_signals = np.column_stack(
        [
            np.bincount(inverse, weights=signals[:, column]) / counts
            for column in range(signals.shape[1])
        ]
    )
    sample_domains = np.empty(len(unique_samples), dtype=np.int64)
    for sample_row in range(len(unique_samples)):
        labels = np.unique(source_domain_ids[inverse == sample_row])
        if len(labels) != 1:
            raise ValueError(
                f"Sample {int(unique_samples[sample_row])} has inconsistent domain labels: "
                f"{labels.tolist()}"
            )
        sample_domains[sample_row] = labels[0]
    return unique_samples, sample_domains, sample_signals


def standardized_mean_difference(positive: np.ndarray, negative: np.ndarray) -> float:
    pooled_variance = (float(np.var(positive, ddof=1)) + float(np.var(negative, ddof=1))) / 2.0
    if pooled_variance <= 0.0:
        difference = float(np.mean(positive) - np.mean(negative))
        # Keep reports strict-JSON serializable even for perfectly constant groups.
        # In that degenerate case the usual standardized effect size is undefined;
        # return the signed, epsilon-regularized limit instead of +/- infinity.
        if difference == 0.0:
            return 0.0
        pooled_variance = float(np.finfo(np.float64).eps)
    return float((np.mean(positive) - np.mean(negative)) / np.sqrt(pooled_variance))


def bootstrap_binary_metrics(
    positive: np.ndarray,
    negative: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, List[float]]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_bootstrap, dtype=np.float64)
    mean_differences = np.empty(n_bootstrap, dtype=np.float64)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int8), np.zeros(len(negative), dtype=np.int8)]
    )
    for index in range(n_bootstrap):
        pos = positive[rng.integers(0, len(positive), size=len(positive))]
        neg = negative[rng.integers(0, len(negative), size=len(negative))]
        scores = np.concatenate([pos, neg])
        aucs[index] = roc_auc_score(labels, scores)
        mean_differences[index] = float(np.mean(pos) - np.mean(neg))
    return {
        "auc_ci95": [float(value) for value in np.quantile(aucs, [0.025, 0.975])],
        "mean_difference_ci95": [
            float(value) for value in np.quantile(mean_differences, [0.025, 0.975])
        ],
    }


def evaluate_selectivity(
    sample_domains: np.ndarray,
    sample_signals: np.ndarray,
    domain_ids: Sequence[int],
    domain_names: Sequence[str],
    n_bootstrap: int,
    seed: int,
    minimum_auc: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if sample_signals.shape[1] != len(domain_ids) or len(domain_ids) != len(domain_names):
        raise ValueError("Domain metadata does not match the SAE signal matrix")
    rows: List[Dict[str, object]] = []
    pairwise: List[Dict[str, object]] = []
    for column, (domain_id, domain_name) in enumerate(zip(domain_ids, domain_names)):
        scores = sample_signals[:, column]
        own_mask = sample_domains == domain_id
        other_mask = ~own_mask
        if int(own_mask.sum()) < 2 or int(other_mask.sum()) < 2:
            raise ValueError(f"Too few samples to validate domain {domain_id}")
        labels = own_mask.astype(np.int8)
        positive = scores[own_mask]
        negative = scores[other_mask]
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))
        intervals = bootstrap_binary_metrics(
            positive, negative, n_bootstrap=n_bootstrap, seed=seed + 1009 * column
        )
        row: Dict[str, object] = {
            "domain_id": int(domain_id),
            "domain_name": domain_name,
            "cluster_column": int(column),
            "n_own_samples": int(own_mask.sum()),
            "n_other_samples": int(other_mask.sum()),
            "own_mean": float(np.mean(positive)),
            "other_mean": float(np.mean(negative)),
            "mean_difference": float(np.mean(positive) - np.mean(negative)),
            "standardized_mean_difference": standardized_mean_difference(positive, negative),
            "roc_auc": auc,
            "average_precision": ap,
            **intervals,
        }
        row["passes_prespecified_gate"] = bool(
            auc >= minimum_auc and intervals["auc_ci95"][0] > 0.5
        )
        rows.append(row)

        for comparison_id, comparison_name in zip(domain_ids, domain_names):
            if comparison_id == domain_id:
                continue
            comparison_mask = sample_domains == comparison_id
            pair_scores = np.concatenate([positive, scores[comparison_mask]])
            pair_labels = np.concatenate(
                [
                    np.ones(len(positive), dtype=np.int8),
                    np.zeros(int(comparison_mask.sum()), dtype=np.int8),
                ]
            )
            pairwise.append(
                {
                    "target_domain_id": int(domain_id),
                    "target_domain_name": domain_name,
                    "comparison_domain_id": int(comparison_id),
                    "comparison_domain_name": comparison_name,
                    "roc_auc": float(roc_auc_score(pair_labels, pair_scores)),
                }
            )
    return rows, pairwise


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether selected SAE clusters distinguish independent domain texts."
    )
    parser.add_argument("--mapping-dir", required=True)
    parser.add_argument("--domains-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-auc", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    if not 0.5 <= args.minimum_auc <= 1.0:
        raise ValueError("--minimum-auc must be between 0.5 and 1")

    mapping_dir = Path(args.mapping_dir)
    with (mapping_dir / "summary.json").open("r", encoding="utf-8") as handle:
        mapping_summary = json.load(handle)
    domains_path = Path(args.domains_path or mapping_summary["domains_path"])
    with domains_path.open("r", encoding="utf-8") as handle:
        domains_payload = json.load(handle)
    names_by_id = {
        int(domain["domain_id"]): str(domain["name"])
        for domain in domains_payload["domains"]
    }

    mapping_domains = mapping_summary["domains"]
    if any(domain.get("cluster_column") is None for domain in mapping_domains):
        raise ValueError("Selectivity validation requires --mapping-corpus pooled-domains")
    activation_paths = {
        str(domain.get("activations_path", domain.get("activation_path", "")))
        for domain in mapping_domains
    }
    activation_paths.discard("")
    if len(activation_paths) != 1:
        raise ValueError("Expected one shared pooled activation artifact")
    activation_path = Path(next(iter(activation_paths)))
    with np.load(activation_path, allow_pickle=False) as payload:
        required = {"sae_cluster_activations", "sample_indices", "source_domain_ids", "domain_ids"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Pooled artifact lacks required arrays: {sorted(missing)}")
        signal_domain_ids = [int(value) for value in payload["domain_ids"].tolist()]
        _, sample_domains, sample_signals = aggregate_token_signals_by_sample(
            payload["sample_indices"],
            payload["source_domain_ids"],
            payload["sae_cluster_activations"],
        )
    domain_names = [names_by_id[domain_id] for domain_id in signal_domain_ids]
    unknown = sorted(set(sample_domains.tolist()).difference(signal_domain_ids))
    if unknown:
        raise ValueError(f"Pooled samples contain unknown domain ids: {unknown}")

    rows, pairwise = evaluate_selectivity(
        sample_domains=sample_domains,
        sample_signals=sample_signals,
        domain_ids=signal_domain_ids,
        domain_names=domain_names,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        minimum_auc=args.minimum_auc,
    )
    output_dir = Path(args.output_dir) if args.output_dir else mapping_dir / "selectivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "domain_selectivity.csv", rows)
    write_csv(output_dir / "pairwise_auc.csv", pairwise)
    result = {
        "method": "sample-level mean SAE-cluster activation; one-vs-rest ROC-AUC",
        "development_only": True,
        "holdout_used": False,
        "mapping_artifact": str(activation_path),
        "n_samples": int(len(sample_domains)),
        "samples_per_domain": {
            str(domain_id): int(np.sum(sample_domains == domain_id))
            for domain_id in signal_domain_ids
        },
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "prespecified_gate": {
            "minimum_auc": args.minimum_auc,
            "auc_ci95_lower_strictly_above": 0.5,
        },
        "all_domains_pass": bool(all(row["passes_prespecified_gate"] for row in rows)),
        "domains": rows,
    }
    with (output_dir / "selectivity_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)

    report_lines = [
        "# Development-set SAE domain selectivity",
        "",
        "This is an independent-corpus development check. The final holdout split was not used.",
        "Signals were averaged over tokens within each text, so long documents do not dominate.",
        "",
        "| Domain | n | ROC-AUC | 95% bootstrap CI | AP | SMD | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        ci = row["auc_ci95"]
        report_lines.append(
            f"| {row['domain_name']} | {row['n_own_samples']} | {row['roc_auc']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {row['average_precision']:.3f} | "
            f"{row['standardized_mean_difference']:.3f} | "
            f"{'PASS' if row['passes_prespecified_gate'] else 'FAIL'} |"
        )
    report_lines.extend(
        [
            "",
            f"Overall gate: **{'PASS' if result['all_domains_pass'] else 'FAIL'}**.",
            "A failed domain should be revised or excluded before its MLP expert is interpreted as domain-specific.",
            "",
        ]
    )
    (output_dir / "selectivity_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Selectivity results written to: {output_dir}")
    print(f"All domains pass: {result['all_domains_pass']}")


if __name__ == "__main__":
    main()
