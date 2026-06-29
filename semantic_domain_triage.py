import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoencoder_training import TopKSAE
from dataset_sequencing import _split_into_sentences
from utils import find_device


MODEL_NAME = "roneneldan/TinyStories-1M"
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
DEFAULT_DATA_PATH = "data/tinystories_dataset"
DEFAULT_LAYER_NUM = 4

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


@dataclass
class DomainInfo:
    domain_id: int
    cluster_label: int
    name: str
    size: int
    cohesion: float
    selection_score: float
    top_tokens: List[Dict[str, float]]
    feature_ids: List[int]


def require_hdbscan():
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: hdbscan. Install project requirements first, "
            "for example: pip install -r requirements.txt"
        ) from exc
    return hdbscan


def normalize_token_text(token: str) -> str:
    token = token.replace("\n", " ")
    token = re.sub(r"\s+", " ", token).strip().lower()
    token = token.strip("\"'`.,:;!?()[]{}")
    return token


def is_semantic_token(token: str, token_id: int, tokenizer: AutoTokenizer) -> bool:
    if token_id in set(tokenizer.all_special_ids):
        return False
    normalized = normalize_token_text(token)
    if len(normalized) < 2:
        return False
    if not re.search(r"[a-zA-Z]", normalized):
        return False
    if normalized.startswith("<") and normalized.endswith(">"):
        return False
    return True


def load_sae_checkpoint(checkpoint_path: str, device: torch.device) -> TopKSAE:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    sae = TopKSAE(
        d_model=int(config.get("d_model", 64)),
        expansion_factor=int(config.get("expansion_factor", 64)),
        k=int(config.get("k", 8)),
    )
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def compute_top_promoted_tokens(
    sae: TopKSAE,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    top_m: int,
    logit_batch_size: int,
    min_logit: float,
) -> Tuple[sparse.csr_matrix, List[List[Dict[str, float]]]]:
    """Compute sparse logit-lens vectors from SAE decoder directions."""

    model.to(device)
    model.eval()
    sae.to(device)
    sae.eval()

    W_U = model.lm_head.weight.detach().to(device).float()
    W_dec = sae.W_dec.detach().to(device).float()

    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    feature_top_tokens: List[List[Dict[str, float]]] = [[] for _ in range(sae.d_sae)]

    with torch.no_grad():
        for start in tqdm(range(0, sae.d_sae, logit_batch_size), desc="Logit lens"):
            end = min(start + logit_batch_size, sae.d_sae)
            logits = W_dec[start:end] @ W_U.T
            top_values, top_ids = torch.topk(logits, k=top_m, dim=-1)

            top_values_np = top_values.cpu().numpy()
            top_ids_np = top_ids.cpu().numpy()

            for local_idx, (token_ids, token_values) in enumerate(zip(top_ids_np, top_values_np)):
                feature_id = start + local_idx
                for token_id, score in zip(token_ids, token_values):
                    score = float(score)
                    if score < min_logit:
                        continue

                    token_text = tokenizer.decode([int(token_id)])
                    if not is_semantic_token(token_text, int(token_id), tokenizer):
                        continue

                    normalized = normalize_token_text(token_text)
                    feature_top_tokens[feature_id].append(
                        {
                            "token": token_text,
                            "normalized": normalized,
                            "token_id": int(token_id),
                            "logit": score,
                        }
                    )
                    row_indices.append(feature_id)
                    col_indices.append(int(token_id))
                    values.append(max(score, 0.0))

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(sae.d_sae, model.config.vocab_size),
        dtype=np.float32,
    )
    matrix = normalize(matrix, norm="l2", axis=1)
    return matrix, feature_top_tokens


def reduce_feature_matrix(
    feature_matrix: sparse.csr_matrix,
    svd_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_feature_ids = np.flatnonzero(feature_matrix.getnnz(axis=1) > 0)
    if len(valid_feature_ids) < 2:
        raise ValueError("Not enough non-empty feature vectors for clustering.")

    valid_matrix = feature_matrix[valid_feature_ids]
    n_components = min(svd_components, valid_matrix.shape[0] - 1, valid_matrix.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Feature matrix is too small for SVD reduction.")

    svd = TruncatedSVD(n_components=n_components, random_state=0)
    reduced = svd.fit_transform(valid_matrix)
    reduced = normalize(reduced, norm="l2", axis=1)
    return valid_feature_ids, np.asarray(reduced, dtype=np.float32)


def aggregate_cluster_tokens(
    feature_ids: Sequence[int],
    feature_top_tokens: Sequence[Sequence[Dict[str, float]]],
    limit: int,
) -> List[Dict[str, float]]:
    token_scores: Dict[str, float] = defaultdict(float)
    token_counts: Counter[str] = Counter()
    display_text: Dict[str, str] = {}

    for feature_id in feature_ids:
        for token in feature_top_tokens[feature_id]:
            normalized = str(token["normalized"])
            token_scores[normalized] += float(token["logit"])
            token_counts[normalized] += 1
            display_text.setdefault(normalized, str(token["token"]))

    ranked = sorted(token_scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {
            "token": display_text[token],
            "normalized": token,
            "score": float(score),
            "feature_count": int(token_counts[token]),
        }
        for token, score in ranked
    ]


def cluster_features(
    reduced_vectors: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int],
) -> np.ndarray:
    hdbscan = require_hdbscan()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(reduced_vectors)


def compute_cluster_candidates(
    labels: np.ndarray,
    valid_feature_ids: np.ndarray,
    reduced_vectors: np.ndarray,
    feature_top_tokens: Sequence[Sequence[Dict[str, float]]],
    top_tokens_per_domain: int,
) -> List[Dict]:
    candidates = []
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue

        member_positions = np.flatnonzero(labels == label)
        if len(member_positions) == 0:
            continue

        member_vectors = reduced_vectors[member_positions]
        centroid = normalize(member_vectors.mean(axis=0, keepdims=True), norm="l2")[0]
        cohesion = float(cosine_similarity(member_vectors, centroid.reshape(1, -1)).mean())
        feature_ids = valid_feature_ids[member_positions].astype(int).tolist()
        top_tokens = aggregate_cluster_tokens(feature_ids, feature_top_tokens, top_tokens_per_domain)

        candidates.append(
            {
                "cluster_label": int(label),
                "feature_ids": feature_ids,
                "size": int(len(feature_ids)),
                "centroid": centroid,
                "cohesion": cohesion,
                "top_tokens": top_tokens,
            }
        )

    return candidates


def select_orthogonal_domains(candidates: List[Dict], n_domains: int) -> Tuple[List[DomainInfo], np.ndarray]:
    if not candidates:
        return [], np.empty((0, 0), dtype=np.float32)

    remaining = candidates.copy()
    selected: List[Dict] = []

    first = max(remaining, key=lambda item: item["cohesion"] * math.log1p(item["size"]))
    first["selection_score"] = float(first["cohesion"] * math.log1p(first["size"]))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < n_domains:
        selected_centroids = np.vstack([item["centroid"] for item in selected])

        def score_candidate(item: Dict) -> float:
            sims = cosine_similarity(item["centroid"].reshape(1, -1), selected_centroids)[0]
            min_distance = float(1.0 - np.max(sims))
            return min_distance * item["cohesion"] * math.log1p(item["size"])

        next_item = max(remaining, key=score_candidate)
        next_item["selection_score"] = float(score_candidate(next_item))
        selected.append(next_item)
        remaining.remove(next_item)

    selected_centroids = np.vstack([item["centroid"] for item in selected])
    similarity_matrix = cosine_similarity(selected_centroids)

    domains = []
    for domain_id, item in enumerate(selected):
        name_terms = [tok["normalized"] for tok in item["top_tokens"][:3]]
        name = " / ".join(name_terms) if name_terms else f"cluster_{item['cluster_label']}"
        domains.append(
            DomainInfo(
                domain_id=domain_id,
                cluster_label=int(item["cluster_label"]),
                name=name,
                size=int(item["size"]),
                cohesion=float(item["cohesion"]),
                selection_score=float(item["selection_score"]),
                top_tokens=item["top_tokens"],
                feature_ids=item["feature_ids"],
            )
        )

    return domains, similarity_matrix


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_feature_top_tokens(path: Path, feature_top_tokens: Sequence[Sequence[Dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for feature_id, tokens in enumerate(feature_top_tokens):
            f.write(json.dumps({"feature_id": feature_id, "top_tokens": tokens}, ensure_ascii=False) + "\n")


def write_assignments(
    path: Path,
    labels: np.ndarray,
    valid_feature_ids: np.ndarray,
    total_features: int,
    domains: Sequence[DomainInfo],
) -> None:
    cluster_by_feature = {int(feature_id): int(label) for feature_id, label in zip(valid_feature_ids, labels)}
    domain_by_cluster = {domain.cluster_label: domain.domain_id for domain in domains}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature_id", "cluster_label", "domain_id", "is_noise"],
        )
        writer.writeheader()
        for feature_id in range(total_features):
            cluster_label = cluster_by_feature.get(feature_id, -1)
            writer.writerow(
                {
                    "feature_id": feature_id,
                    "cluster_label": cluster_label,
                    "domain_id": domain_by_cluster.get(cluster_label, ""),
                    "is_noise": cluster_label == -1,
                }
            )


def write_domain_report(path: Path, domains: Sequence[DomainInfo], similarity_matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Semantic Domain Triage Report\n\n")
        f.write(f"Selected domains: {len(domains)}\n\n")

        for domain in domains:
            f.write(f"## Domain {domain.domain_id}: {domain.name}\n\n")
            f.write(f"- Cluster label: {domain.cluster_label}\n")
            f.write(f"- Features: {domain.size}\n")
            f.write(f"- Cohesion: {domain.cohesion:.4f}\n")
            f.write(f"- Selection score: {domain.selection_score:.4f}\n")
            f.write("- Top promoted tokens: ")
            f.write(", ".join(token["normalized"] for token in domain.top_tokens[:20]))
            f.write("\n\n")

        if len(domains) > 1:
            f.write("## Domain Cosine Similarity\n\n")
            header = ["domain"] + [str(domain.domain_id) for domain in domains]
            f.write("| " + " | ".join(header) + " |\n")
            f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
            for row_idx, domain in enumerate(domains):
                values = [f"{similarity_matrix[row_idx, col_idx]:.3f}" for col_idx in range(len(domains))]
                f.write("| " + " | ".join([str(domain.domain_id)] + values) + " |\n")


def iter_validation_fragments(csv_path: Path, max_texts: Optional[int]) -> Iterable[Tuple[int, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for text_idx, row in enumerate(reader):
            if max_texts is not None and text_idx >= max_texts:
                break
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            for sentence in _split_into_sentences(text):
                yield text_idx, sentence


def build_domain_lexicons(domains: Sequence[DomainInfo], tokens_per_domain: int) -> List[Dict[str, float]]:
    lexicons: List[Dict[str, float]] = []
    for domain in domains:
        lexicon: Dict[str, float] = {}
        for token in domain.top_tokens[:tokens_per_domain]:
            normalized = str(token["normalized"])
            if not normalized:
                continue
            lexicon[normalized] = max(lexicon.get(normalized, 0.0), float(token["score"]))
        if lexicon:
            max_score = max(lexicon.values())
            lexicon = {token: score / max_score for token, score in lexicon.items()}
        lexicons.append(lexicon)
    return lexicons


def score_fragment(text: str, lexicon: Dict[str, float]) -> float:
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    if not words:
        return 0.0

    word_counts = Counter(words)
    score = 0.0
    text_lower = text.lower()

    for term, weight in lexicon.items():
        if " " in term:
            occurrences = text_lower.count(term)
        else:
            occurrences = word_counts.get(term, 0)
        score += weight * occurrences

    return score / math.sqrt(len(words))


def assign_fragment(
    text: str,
    lexicons: Sequence[Dict[str, float]],
    min_domain_score: float,
    exclusive_margin: float,
) -> Optional[Tuple[int, List[float]]]:
    scores = [score_fragment(text, lexicon) for lexicon in lexicons]
    if not scores:
        return None

    order = np.argsort(scores)[::-1]
    best_idx = int(order[0])
    best_score = float(scores[best_idx])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0

    if best_score < min_domain_score:
        return None
    if best_score - second_score < exclusive_margin:
        return None
    return best_idx, scores


def generate_domain_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    domain: DomainInfo,
    needed: int,
    seed: int,
) -> List[Dict]:
    if needed <= 0:
        return []

    torch.manual_seed(seed + domain.domain_id)
    model.to(device)
    model.eval()

    keywords = ", ".join(token["normalized"] for token in domain.top_tokens[:8])
    prompt = f"Write a very short children's story about: {keywords}.\nStory:"

    samples = []
    for sample_idx in range(needed):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_new_tokens=80,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        text = text.replace(prompt, "").strip()
        samples.append(
            {
                "domain_id": domain.domain_id,
                "domain_name": domain.name,
                "text": text,
                "source": "generated",
                "source_text_idx": "",
                "score": "",
            }
        )
    return samples


def build_validation_sets(
    domains: Sequence[DomainInfo],
    csv_path: Path,
    output_dir: Path,
    samples_per_domain: int,
    tokens_per_domain: int,
    min_domain_score: float,
    exclusive_margin: float,
    max_validation_texts: Optional[int],
    seed: int,
    generate_if_needed: bool,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Dict:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    lexicons = build_domain_lexicons(domains, tokens_per_domain)
    samples_by_domain: Dict[int, List[Dict]] = {domain.domain_id: [] for domain in domains}
    rejected_low_score = 0
    rejected_not_exclusive = 0
    seen_texts = set()

    for source_text_idx, fragment in tqdm(
        iter_validation_fragments(csv_path, max_validation_texts),
        desc="Filtering validation fragments",
    ):
        key = fragment.strip().lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)

        scores = [score_fragment(fragment, lexicon) for lexicon in lexicons]
        if not scores or max(scores) < min_domain_score:
            rejected_low_score += 1
            continue

        order = np.argsort(scores)[::-1]
        best_idx = int(order[0])
        second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0
        if float(scores[best_idx]) - second_score < exclusive_margin:
            rejected_not_exclusive += 1
            continue

        domain = domains[best_idx]
        samples_by_domain[domain.domain_id].append(
            {
                "domain_id": domain.domain_id,
                "domain_name": domain.name,
                "text": fragment,
                "source": "tinystories_validation",
                "source_text_idx": source_text_idx,
                "score": float(scores[best_idx]),
            }
        )

    balanced_samples = []
    for domain in domains:
        samples = samples_by_domain[domain.domain_id]
        random.shuffle(samples)
        selected = samples[:samples_per_domain]

        if generate_if_needed and len(selected) < samples_per_domain:
            selected.extend(
                generate_domain_samples(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    domain=domain,
                    needed=samples_per_domain - len(selected),
                    seed=seed,
                )
            )

        samples_by_domain[domain.domain_id] = selected
        balanced_samples.extend(selected)

        domain_path = output_dir / f"domain_{domain.domain_id}.jsonl"
        with domain_path.open("w", encoding="utf-8") as f:
            for sample in selected:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    csv_path_out = output_dir / "all_domains_balanced.csv"
    with csv_path_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["domain_id", "domain_name", "text", "source", "source_text_idx", "score"],
        )
        writer.writeheader()
        writer.writerows(balanced_samples)

    summary = {
        "samples_per_domain_requested": samples_per_domain,
        "domains": [
            {
                "domain_id": domain.domain_id,
                "domain_name": domain.name,
                "samples": len(samples_by_domain[domain.domain_id]),
                "mean_text_length": float(
                    np.mean([len(sample["text"]) for sample in samples_by_domain[domain.domain_id]])
                )
                if samples_by_domain[domain.domain_id]
                else 0.0,
            }
            for domain in domains
        ],
        "rejected_low_score": rejected_low_score,
        "rejected_not_exclusive": rejected_not_exclusive,
    }
    write_json(output_dir / "validation_summary.json", summary)
    return summary


def default_checkpoint_path(data_path: str, layer_num: int) -> str:
    return os.path.join(data_path, f"models/checkpoints/topk_sae_layer_{layer_num}_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster SAE logit-lens vectors into semantic domains and build validation sets."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-domains", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=128)
    parser.add_argument("--top-tokens-per-domain", type=int, default=80)
    parser.add_argument("--logit-batch-size", type=int, default=128)
    parser.add_argument("--min-logit", type=float, default=0.0)
    parser.add_argument("--svd-components", type=int, default=50)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--samples-per-domain", type=int, default=100)
    parser.add_argument("--validation-tokens-per-domain", type=int, default=50)
    parser.add_argument("--min-domain-score", type=float, default=0.15)
    parser.add_argument("--exclusive-margin", type=float, default=0.05)
    parser.add_argument("--max-validation-texts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generate-if-needed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    checkpoint_path = args.checkpoint or default_checkpoint_path(args.data_path, args.layer_num)
    output_dir = Path(args.output_dir) if args.output_dir else data_path / "analysis/domain_triage"
    validation_csv = Path(args.validation_csv) if args.validation_csv else data_path / "validation.csv"

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
    if not validation_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {validation_csv}")

    device = find_device()
    print(f"Using device: {device}")
    print(f"Loading SAE checkpoint: {checkpoint_path}")
    sae = load_sae_checkpoint(checkpoint_path, device)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    feature_matrix, feature_top_tokens = compute_top_promoted_tokens(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_m=args.top_m,
        logit_batch_size=args.logit_batch_size,
        min_logit=args.min_logit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "feature_token_matrix.npz", feature_matrix)
    write_feature_top_tokens(output_dir / "logit_lens_top_tokens.jsonl", feature_top_tokens)

    valid_feature_ids, reduced_vectors = reduce_feature_matrix(feature_matrix, args.svd_components)
    labels = cluster_features(
        reduced_vectors=reduced_vectors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    candidates = compute_cluster_candidates(
        labels=labels,
        valid_feature_ids=valid_feature_ids,
        reduced_vectors=reduced_vectors,
        feature_top_tokens=feature_top_tokens,
        top_tokens_per_domain=args.top_tokens_per_domain,
    )
    domains, similarity_matrix = select_orthogonal_domains(candidates, args.n_domains)

    write_json(
        output_dir / "domains.json",
        {
            "domains": [asdict(domain) for domain in domains],
            "domain_similarity": similarity_matrix.tolist(),
            "n_clusters": len(candidates),
            "noise_features": int(np.sum(labels == -1)),
            "valid_features": int(len(valid_feature_ids)),
        },
    )
    write_assignments(
        output_dir / "feature_domain_assignments.csv",
        labels=labels,
        valid_feature_ids=valid_feature_ids,
        total_features=sae.d_sae,
        domains=domains,
    )
    write_domain_report(output_dir / "domain_report.md", domains, similarity_matrix)

    print(f"Selected {len(domains)} domains. Building validation sets...")
    validation_summary = build_validation_sets(
        domains=domains,
        csv_path=validation_csv,
        output_dir=output_dir / "domain_validation",
        samples_per_domain=args.samples_per_domain,
        tokens_per_domain=args.validation_tokens_per_domain,
        min_domain_score=args.min_domain_score,
        exclusive_margin=args.exclusive_margin,
        max_validation_texts=args.max_validation_texts,
        seed=args.seed,
        generate_if_needed=args.generate_if_needed,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print(f"Domain triage outputs written to: {output_dir}")
    print(json.dumps(validation_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
