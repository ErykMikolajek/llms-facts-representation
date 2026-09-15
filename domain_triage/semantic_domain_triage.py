import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_pipeline.autoencoder_training import TopKSAE
from sae_pipeline.dataset_sequencing import _split_into_sentences
from common.utils import find_device, tokenizer_vocab_fingerprint


MODEL_NAME = "roneneldan/TinyStories-1M"
TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
DEFAULT_DATA_PATH = "data/tinystories_dataset"
DEFAULT_LAYER_NUM = 4

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
COMMON_FUNCTION_TOKENS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "himself",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "let",
    "like",
    "many",
    "me",
    "mine",
    "more",
    "my",
    "myself",
    "next",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "ours",
    "ourselves",
    "out",
    "she",
    "so",
    "some",
    "that",
    "the",
    "their",
    "them",
    "theirs",
    "themselves",
    "then",
    "there",
    "they",
    "this",
    "to",
    "too",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
    "yourself",
    "yours",
    "yourselves",
    "oneself",
    "which",
    "whom",
    "whose",
}
COMMON_FUNCTION_TOKENS.update(ENGLISH_STOP_WORDS)


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


def is_semantic_token(
    token: str,
    token_id: int,
    tokenizer: AutoTokenizer,
    keep_common_function_tokens: bool = False,
    min_token_chars: int = 2,
    require_token_boundary: bool = False,
) -> bool:
    if token_id in set(tokenizer.all_special_ids):
        return False
    normalized = normalize_token_text(token)
    if len(normalized) < min_token_chars:
        return False
    if not any(character.isalpha() for character in normalized):
        return False
    if normalized.startswith("<") and normalized.endswith(">"):
        return False
    if not keep_common_function_tokens and normalized in COMMON_FUNCTION_TOKENS:
        return False
    if require_token_boundary and normalized.isalpha():
        raw_token = tokenizer.convert_ids_to_tokens(int(token_id))
        raw_token = str(raw_token) if raw_token is not None else ""
        backend_model = getattr(getattr(tokenizer, "backend_tokenizer", None), "model", None)
        is_wordpiece = "wordpiece" in type(backend_model).__name__.lower()
        has_boundary = (
            token[:1].isspace()
            or raw_token.startswith(("Ġ", "▁"))
            or (raw_token and not raw_token.startswith("##") and is_wordpiece)
        )
        if not has_boundary:
            # GPT BPE uses Ġ, SentencePiece uses ▁, and WordPiece uses ## to
            # distinguish continuations. Decoded whitespace alone is not
            # reliable across tokenizer implementations.
            return False
    return True


def resolve_single_token_id(tokenizer: AutoTokenizer, token: str) -> Optional[int]:
    """Resolve a decoded token string back to its vocabulary id.

    The feature-analysis JSON stores decoded strings rather than token ids.
    ``convert_tokens_to_ids`` is not sufficient for GPT-2-style tokenizers
    because decoded strings may contain a leading space, so we require a
    round-trip through ``encode``/``decode``.
    """
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    token_id = int(token_ids[0])
    if tokenizer.decode([token_id]) != token:
        return None
    return token_id


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
    keep_common_function_tokens: bool,
    min_token_chars: int = 2,
    require_token_boundary: bool = False,
    use_tfidf: bool = True,
) -> Tuple[sparse.csr_matrix, List[List[Dict[str, float]]]]:
    """Compute sparse logit-lens vectors from SAE decoder directions."""

    model.to(device)
    model.eval()
    sae.to(device)
    sae.eval()

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose output embeddings for logit lens")
    W_U = output_embeddings.weight.detach().to(device).float()
    W_dec = sae.W_dec.detach().to(device).float()

    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    feature_top_tokens: List[List[Dict[str, float]]] = [[] for _ in range(sae.d_sae)]

    with torch.no_grad():
        for start in tqdm(range(0, sae.d_sae, logit_batch_size), desc="Logit lens"):
            end = min(start + logit_batch_size, sae.d_sae)
            logits = W_dec[start:end] @ W_U.T
            top_values, top_ids = torch.topk(logits, k=min(top_m, logits.shape[-1]), dim=-1)

            top_values_np = top_values.cpu().numpy()
            top_ids_np = top_ids.cpu().numpy()

            for local_idx, (token_ids, token_values) in enumerate(zip(top_ids_np, top_values_np)):
                feature_id = start + local_idx
                for token_id, score in zip(token_ids, token_values):
                    score = float(score)
                    if score <= 0.0 or score < min_logit:
                        continue

                    token_text = tokenizer.decode([int(token_id)])
                    if not is_semantic_token(
                        token_text,
                        int(token_id),
                        tokenizer,
                        keep_common_function_tokens=keep_common_function_tokens,
                        min_token_chars=min_token_chars,
                        require_token_boundary=require_token_boundary,
                    ):
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
        shape=(sae.d_sae, W_U.shape[0]),
        dtype=np.float32,
    )
    if use_tfidf:
        document_frequency = np.asarray(matrix.getnnz(axis=0)).ravel()
        inverse_document_frequency = np.log(
            (1.0 + sae.d_sae) / (1.0 + document_frequency)
        ) + 1.0
        matrix = matrix.multiply(inverse_document_frequency)
        for tokens in feature_top_tokens:
            for token in tokens:
                token["idf"] = float(inverse_document_frequency[int(token["token_id"])])
    matrix = normalize(matrix, norm="l2", axis=1)
    return matrix, feature_top_tokens


def load_precomputed_feature_analysis(
    analysis_path: Path,
    tokenizer: AutoTokenizer,
    min_logit: float,
    keep_common_function_tokens: bool,
    min_token_chars: int,
    require_token_boundary: bool,
    observed_features_only: bool,
    use_tfidf: bool,
    top_m: int = 128,
    feature_token_source: str = "promoted",
) -> Tuple[sparse.csr_matrix, List[List[Dict[str, float]]], Dict[str, object]]:
    """Build the triage matrix from an existing ``features_analysis.json``.

    This is the offline path used for the Pythia artifacts shipped with the
    project.  The JSON contains the already-computed logit-lens top tokens,
    so the base model weights are not needed a second time.
    """
    with analysis_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_records = payload.get("features") if isinstance(payload, dict) else payload
    if isinstance(raw_records, dict):
        feature_records = list(raw_records.values())
    else:
        feature_records = raw_records
    if not isinstance(feature_records, list) or not feature_records:
        raise ValueError(f"No feature records found in {analysis_path}")

    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    metadata_payload = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    declared_total = summary.get("total_features", metadata_payload.get("num_features"))
    max_feature_id = max(int(record["feature_id"]) for record in feature_records)
    total_features = int(declared_total) if declared_total is not None else max_feature_id + 1
    if total_features <= max_feature_id:
        raise ValueError(
            f"Declared total_features={total_features} does not include feature id {max_feature_id}"
        )
    row_indices: List[int] = []
    col_indices: List[int] = []
    values: List[float] = []
    feature_top_tokens: List[List[Dict[str, float]]] = [
        [] for _ in range(total_features)
    ]
    included_features = 0
    observed_features = 0
    skipped_unresolvable_tokens = 0
    tokenizer_size = len(tokenizer)
    token_id_cache: Dict[str, Optional[int]] = {}
    semantic_token_cache: Dict[Tuple[str, int], bool] = {}

    special_ids = set(tokenizer.all_special_ids)
    for record in feature_records:
        feature_id = int(record["feature_id"])
        observed = bool(
            record.get(
                "observed_in_analysis",
                int(record.get("total_activations", 0)) > 0 or not bool(record.get("is_dead", True)),
            )
        )
        observed_features += int(observed)
        if observed_features_only and not observed:
            continue
        included_features += 1

        promoted_tokens = record.get("top_promoted_tokens", [])
        trigger_tokens = record.get("common_trigger_tokens", [])
        if feature_token_source == "promoted":
            raw_tokens = promoted_tokens
        elif feature_token_source == "trigger":
            raw_tokens = trigger_tokens
        elif feature_token_source == "fallback":
            raw_tokens = promoted_tokens or trigger_tokens
        else:
            raise ValueError(
                "feature_token_source must be promoted, trigger, or fallback"
            )

        for raw_token in raw_tokens[:top_m]:
            if isinstance(raw_token, dict):
                token_text = str(raw_token.get("token", ""))
                if "logit" in raw_token or "score" in raw_token:
                    score = float(raw_token.get("logit", raw_token.get("score", 0.0)))
                else:
                    score = math.log1p(float(raw_token.get("count", 0.0)))
                raw_token_id = raw_token.get("token_id")
            elif isinstance(raw_token, (list, tuple)) and len(raw_token) >= 2:
                token_text = str(raw_token[0])
                score = float(raw_token[1])
                raw_token_id = None
            else:
                continue
            if score <= 0.0 or score < min_logit:
                continue

            if raw_token_id is not None:
                token_id = int(raw_token_id)
            else:
                if token_text not in token_id_cache:
                    token_id_cache[token_text] = resolve_single_token_id(tokenizer, token_text)
                token_id = token_id_cache[token_text]
            if token_id is None:
                skipped_unresolvable_tokens += 1
                continue
            if token_id < 0 or token_id >= tokenizer_size:
                skipped_unresolvable_tokens += 1
                continue
            if token_id in special_ids:
                continue
            semantic_key = (token_text, token_id)
            if semantic_key not in semantic_token_cache:
                semantic_token_cache[semantic_key] = is_semantic_token(
                    token_text,
                    token_id,
                    tokenizer,
                    keep_common_function_tokens=keep_common_function_tokens,
                    min_token_chars=min_token_chars,
                    require_token_boundary=require_token_boundary,
                )
            if not semantic_token_cache[semantic_key]:
                continue
            normalized = normalize_token_text(token_text)
            feature_top_tokens[feature_id].append(
                {
                    "token": token_text,
                    "normalized": normalized,
                    "token_id": token_id,
                    "logit": score,
                    "source": (
                        "precomputed_logit_lens"
                        if raw_tokens is promoted_tokens
                        else "observed_trigger_token"
                    ),
                }
            )
            row_indices.append(feature_id)
            col_indices.append(token_id)
            values.append(max(score, 0.0))

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(total_features, tokenizer_size),
        dtype=np.float32,
    )
    if use_tfidf:
        document_frequency = np.asarray(matrix.getnnz(axis=0)).ravel()
        inverse_document_frequency = np.log(
            (1.0 + max(included_features, 1)) / (1.0 + document_frequency)
        ) + 1.0
        matrix = matrix.multiply(inverse_document_frequency)
        for tokens in feature_top_tokens:
            for token in tokens:
                token["idf"] = float(inverse_document_frequency[int(token["token_id"])])
    matrix = normalize(matrix, norm="l2", axis=1)

    metadata = {
        "source": str(analysis_path),
        "source_format": summary,
        "total_features": int(total_features),
        "included_features": int(included_features),
        "observed_features": int(observed_features),
        "observed_features_only": bool(observed_features_only),
        "nonempty_features": int(np.count_nonzero(matrix.getnnz(axis=1))),
        "skipped_unresolvable_tokens": int(skipped_unresolvable_tokens),
        "use_tfidf": bool(use_tfidf),
        "feature_token_source": feature_token_source,
        "top_m": int(top_m),
    }
    return matrix, feature_top_tokens, metadata


def load_observed_feature_ids(analysis_path: Path, total_features: int) -> np.ndarray:
    """Return feature ids observed in a saved activation analysis."""
    with analysis_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("features", []) if isinstance(payload, dict) else payload
    if isinstance(records, dict):
        records = list(records.values())
    observed = np.zeros(total_features, dtype=bool)
    for record in records:
        if not isinstance(record, dict):
            continue
        is_observed = bool(
            record.get(
                "observed_in_analysis",
                int(record.get("total_activations", 0)) > 0 or not bool(record.get("is_dead", True)),
            )
        )
        if not is_observed:
            continue
        feature_id = int(record["feature_id"])
        if 0 <= feature_id < total_features:
            observed[feature_id] = True
    return np.flatnonzero(observed)


def reduce_feature_matrix(
    feature_matrix: sparse.csr_matrix,
    svd_components: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    valid_feature_ids = np.flatnonzero(feature_matrix.getnnz(axis=1) > 0)
    if len(valid_feature_ids) < 2:
        raise ValueError("Not enough non-empty feature vectors for clustering.")

    valid_matrix = feature_matrix[valid_feature_ids]
    n_components = min(svd_components, valid_matrix.shape[0] - 1, valid_matrix.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Feature matrix is too small for SVD reduction.")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(valid_matrix)
    reduced = normalize(reduced, norm="l2", axis=1)
    return valid_feature_ids, np.asarray(reduced, dtype=np.float32)


def aggregate_cluster_tokens(
    feature_ids: Sequence[int],
    feature_top_tokens: Sequence[Sequence[Dict[str, float]]],
    limit: int,
    min_feature_support: int = 1,
    min_feature_fraction: float = 0.0,
) -> List[Dict[str, float]]:
    token_scores: Dict[str, float] = defaultdict(float)
    token_counts: Counter[str] = Counter()
    display_text: Dict[str, str] = {}

    for feature_id in feature_ids:
        for token in feature_top_tokens[feature_id]:
            normalized = str(token["normalized"])
            token_scores[normalized] += float(token["logit"]) * float(token.get("idf", 1.0))
            token_counts[normalized] += 1
            display_text.setdefault(normalized, str(token["token"]))

    required_support = max(
        min_feature_support,
        int(math.ceil(len(feature_ids) * min_feature_fraction)),
    )
    ranked = sorted(
        (
            item
            for item in token_scores.items()
            if token_counts[item[0]] >= required_support
        ),
        key=lambda item: (-item[1], item[0]),
    )[:limit]
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
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    hdbscan = require_hdbscan()
    if cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
    )
    return clusterer.fit_predict(reduced_vectors)


def compute_cluster_candidates(
    labels: np.ndarray,
    valid_feature_ids: np.ndarray,
    feature_matrix: sparse.csr_matrix,
    reduced_vectors: np.ndarray,
    feature_top_tokens: Sequence[Sequence[Dict[str, float]]],
    top_tokens_per_domain: int,
    min_token_feature_support: int = 1,
    min_token_feature_fraction: float = 0.0,
) -> List[Dict]:
    candidates = []
    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue

        member_positions = np.flatnonzero(labels == label)
        if len(member_positions) == 0:
            continue

        feature_ids = valid_feature_ids[member_positions].astype(int).tolist()
        member_vectors = feature_matrix[feature_ids]
        centroid = normalize(
            sparse.csr_matrix(member_vectors.sum(axis=0)), norm="l2", axis=1
        )
        latent_centroid = normalize(
            reduced_vectors[member_positions].mean(axis=0, keepdims=True),
            norm="l2",
            axis=1,
        )[0]
        cohesion = float((member_vectors @ centroid.T).mean())
        top_tokens = aggregate_cluster_tokens(
            feature_ids,
            feature_top_tokens,
            top_tokens_per_domain,
            min_feature_support=min_token_feature_support,
            min_feature_fraction=min_token_feature_fraction,
        )

        candidates.append(
            {
                "cluster_label": int(label),
                "feature_ids": feature_ids,
                "size": int(len(feature_ids)),
                "centroid": centroid,
                "latent_centroid": np.asarray(latent_centroid, dtype=np.float32),
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
        selected_centroids = sparse.vstack([item["centroid"] for item in selected])
        selected_latent_centroids = np.vstack(
            [item["latent_centroid"] for item in selected]
        )

        def score_candidate(item: Dict) -> float:
            lexical_sims = cosine_similarity(item["centroid"], selected_centroids)[0]
            latent_sims = cosine_similarity(
                item["latent_centroid"].reshape(1, -1), selected_latent_centroids
            )[0]
            # Top-k vocabulary rows can have no exact token overlap despite
            # being close in the SVD space. Use the stronger non-negative
            # signal; negative SVD cosines are reduction artefacts here, not
            # evidence of semantic opposition.
            sims = np.maximum(lexical_sims, np.maximum(latent_sims, 0.0))
            min_distance = float(1.0 - np.max(sims))
            return min_distance * item["cohesion"] * math.log1p(item["size"])

        next_item = max(remaining, key=score_candidate)
        next_item["selection_score"] = float(score_candidate(next_item))
        selected.append(next_item)
        remaining.remove(next_item)

    selected_centroids = sparse.vstack([item["centroid"] for item in selected])
    selected_latent_centroids = np.vstack(
        [item["latent_centroid"] for item in selected]
    )
    lexical_similarity = cosine_similarity(selected_centroids)
    latent_similarity = np.maximum(
        cosine_similarity(selected_latent_centroids),
        0.0,
    )
    similarity_matrix = np.maximum(lexical_similarity, latent_similarity)

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


def summarize_domain_quality(
    domains: Sequence[DomainInfo],
    n_clusters: int,
    noise_features: int,
    valid_features: int,
    labels: Optional[np.ndarray] = None,
    reduced_vectors: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    total_top_tokens = 0
    common_function_tokens = 0
    small_domains = 0

    for domain in domains:
        if domain.size < 20:
            small_domains += 1
        for token in domain.top_tokens[:20]:
            total_top_tokens += 1
            if str(token["normalized"]) in COMMON_FUNCTION_TOKENS:
                common_function_tokens += 1

    noise_fraction = noise_features / valid_features if valid_features else 0.0
    function_token_fraction = common_function_tokens / total_top_tokens if total_top_tokens else 0.0

    silhouette = None
    if labels is not None and reduced_vectors is not None:
        clustered = labels >= 0
        clustered_labels = labels[clustered]
        if clustered.sum() >= 3 and len(set(clustered_labels.tolist())) >= 2:
            silhouette = float(
                silhouette_score(
                    reduced_vectors[clustered],
                    clustered_labels,
                    sample_size=min(5000, int(clustered.sum())),
                    random_state=0,
                )
            )

    return {
        "selected_domains": len(domains),
        "n_clusters": int(n_clusters),
        "noise_features": int(noise_features),
        "valid_features": int(valid_features),
        "noise_fraction": float(noise_fraction),
        "small_domains": int(small_domains),
        "top_token_function_fraction": float(function_token_fraction),
        "clustered_feature_silhouette": silhouette,
    }


def validate_domain_selection(
    domains: Sequence[DomainInfo],
    min_required_domains: int,
    allow_underfilled_domains: bool,
    diagnostics: Dict[str, object],
) -> None:
    if len(domains) >= min_required_domains:
        return

    message = (
        f"Only {len(domains)} semantic domains were selected, but at least "
        f"{min_required_domains} are required. Diagnostics: {diagnostics}. "
        "Try lowering --min-cluster-size/--min-samples, increasing --top-m, "
        "or inspecting logit_lens_top_tokens.jsonl for over-filtered features."
    )
    if allow_underfilled_domains:
        print(f"WARNING: {message}")
        return
    raise ValueError(message)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def approve_existing_domains(output_dir: Path) -> None:
    """Record a human gate without rerunning deterministic clustering."""
    domains_path = output_dir / "domains.json"
    report_path = output_dir / "domain_report.md"
    if not domains_path.exists() or not report_path.exists():
        raise FileNotFoundError(
            f"Expected existing domains.json and domain_report.md in {output_dir}"
        )
    with domains_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not payload.get("domains"):
        raise ValueError(f"No domains are available to approve in {domains_path}")
    payload["manual_review_required"] = True
    payload["downstream_approved"] = True
    payload["approval"] = {
        "method": "explicit_cli_after_manual_review",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(domains_path, payload)
    report = report_path.read_text(encoding="utf-8")
    report = re.sub(
        r"Downstream approved: (?:True|False)",
        "Downstream approved: True",
        report,
        count=1,
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Approved existing semantic domains for downstream use: {domains_path}")


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


def write_domain_report(
    path: Path,
    domains: Sequence[DomainInfo],
    similarity_matrix: np.ndarray,
    diagnostics: Optional[Dict[str, object]] = None,
    downstream_approved: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Semantic Domain Triage Report\n\n")
        f.write(f"Selected domains: {len(domains)}\n\n")
        f.write(f"Downstream approved: {downstream_approved}\n\n")
        f.write(
            "> Automatic names below are token-based proxies. They require manual "
            "inspection of validation contexts before being treated as semantic labels.\n\n"
        )

        if diagnostics:
            f.write("## Diagnostics\n\n")
            for key, value in diagnostics.items():
                if isinstance(value, float):
                    f.write(f"- {key}: {value:.4f}\n")
                else:
                    f.write(f"- {key}: {value}\n")
            f.write("\n")

        for domain in domains:
            f.write(f"## Domain {domain.domain_id}: {domain.name}\n\n")
            f.write(f"- Cluster label: {domain.cluster_label}\n")
            f.write(f"- Features: {domain.size}\n")
            f.write(f"- Cohesion: {domain.cohesion:.4f}\n")
            f.write(f"- Selection score: {domain.selection_score:.4f}\n")
            f.write("- Top representation tokens (feature support): ")
            f.write(
                ", ".join(
                    f"{token['normalized']} ({token['feature_count']})"
                    for token in domain.top_tokens[:20]
                )
            )
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
    prompt = f"Write a short, coherent passage centered on: {keywords}.\nPassage:"

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
                "source": "synthetic_model_generated",
                "source_text_idx": "",
                "source_group": f"synthetic:{domain.domain_id}:{sample_idx}",
                "score": "",
                "diagnostic_only": True,
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
                "source": "external_validation_csv",
                "source_text_idx": source_text_idx,
                "source_group": f"validation_csv:{source_text_idx}",
                "score": float(scores[best_idx]),
                "diagnostic_only": False,
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
            fieldnames=[
                "domain_id",
                "domain_name",
                "text",
                "source",
                "source_text_idx",
                "source_group",
                "score",
                "diagnostic_only",
            ],
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
        "synthetic_samples": int(
            sum(sample.get("diagnostic_only", False) for sample in balanced_samples)
        ),
        "independent_evaluation": False,
        "warning": (
            "Labels were assigned using lexicons derived from the discovered domains; "
            "this set is suitable for mapping/router development, not final independent evaluation."
        ),
    }
    write_json(output_dir / "validation_summary.json", summary)
    return summary


def build_validation_sets_from_feature_analysis(
    domains: Sequence[DomainInfo],
    analysis_path: Path,
    output_dir: Path,
    samples_per_domain: int,
) -> Dict:
    """Build domain examples from contexts retained by feature analysis.

    Pythia's local artifacts contain token sequences and feature contexts but
    not the original Pile validation CSV.  Reusing those contexts keeps the
    validation set tied to observed SAE activations and avoids inventing
    labels from a hand-written keyword list.
    """
    with analysis_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw_records = payload.get("features", []) if isinstance(payload, dict) else payload
    if isinstance(raw_records, dict):
        raw_records = list(raw_records.values())
    records = {
        int(record["feature_id"]): record
        for record in raw_records
        if isinstance(record, dict) and "feature_id" in record
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_by_domain: Dict[int, Dict[int, List[Dict]]] = {}
    owners_by_text: Dict[str, set[int]] = defaultdict(set)
    for domain in domains:
        lexicon = build_domain_lexicons([domain], tokens_per_domain=50)[0]
        by_feature: Dict[int, List[Dict]] = {}
        for feature_id in domain.feature_ids:
            examples = []
            for example in records.get(int(feature_id), {}).get("top_examples", []):
                context = str(example.get("full_context", "")).replace("[[", "").replace("]]", "")
                normalized_context = re.sub(r"\s+", " ", context).strip().lower()
                if not normalized_context:
                    continue
                domain_score = score_fragment(context, lexicon)
                if domain_score <= 0.0:
                    continue
                sequence_id = example.get("sequence_id", "")
                candidate = {
                    "domain_id": domain.domain_id,
                    "domain_name": domain.name,
                    "text": context.strip(),
                    "source": "feature_analysis_context",
                    "source_text_idx": sequence_id,
                    "source_group": f"sequence:{sequence_id}" if sequence_id != "" else normalized_context,
                    "score": float(example.get("activation_score", 0.0)),
                    "domain_lexical_score": float(domain_score),
                    "source_feature_id": int(feature_id),
                    "diagnostic_only": True,
                    "_dedup_key": normalized_context,
                }
                examples.append(candidate)
                owners_by_text[normalized_context].add(domain.domain_id)
            if examples:
                by_feature[int(feature_id)] = sorted(
                    examples,
                    key=lambda item: (
                        float(item["domain_lexical_score"]),
                        float(item["score"]),
                    ),
                    reverse=True,
                )
        candidates_by_domain[domain.domain_id] = by_feature

    # A context can be a top example for features in several clusters. Giving
    # the same text contradictory labels leaks the construction heuristic into
    # the router, so ambiguous contexts are excluded rather than assigned by
    # arbitrary domain iteration order or incomparable SAE activation scales.
    ambiguous_keys = {key for key, owners in owners_by_text.items() if len(owners) > 1}
    all_samples: List[Dict] = []
    summary_domains: List[Dict] = []
    for domain in domains:
        by_feature = {
            feature_id: [
                candidate
                for candidate in candidates
                if candidate["_dedup_key"] not in ambiguous_keys
            ]
            for feature_id, candidates in candidates_by_domain[domain.domain_id].items()
        }
        by_feature = {feature_id: candidates for feature_id, candidates in by_feature.items() if candidates}
        available_contexts = sum(len(items) for items in by_feature.values())

        selected: List[Dict] = []
        seen = set()
        feature_order = sorted(
            by_feature,
            key=lambda feature_id: (
                by_feature[feature_id][0]["domain_lexical_score"],
                by_feature[feature_id][0]["score"],
            ),
            reverse=True,
        )
        # Round-robin over features prevents a single highly active feature
        # from filling an entire domain's validation set.
        while len(selected) < samples_per_domain and feature_order:
            next_order = []
            for feature_id in feature_order:
                candidates = by_feature[feature_id]
                if not candidates:
                    continue
                sample = candidates.pop(0)
                key = sample["_dedup_key"]
                if key not in seen:
                    seen.add(key)
                    # ``sample`` is also referenced by ``candidates_by_domain``
                    # for diagnostics below.  Do not mutate that source record.
                    sample = dict(sample)
                    sample.pop("_dedup_key", None)
                    selected.append(sample)
                    if len(selected) >= samples_per_domain:
                        break
                if candidates:
                    next_order.append(feature_id)
            feature_order = next_order

        domain_path = output_dir / f"domain_{domain.domain_id}.jsonl"
        with domain_path.open("w", encoding="utf-8") as f:
            for sample in selected:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        all_samples.extend(selected)
        summary_domains.append(
            {
                "domain_id": domain.domain_id,
                "domain_name": domain.name,
                "samples": len(selected),
                "available_contexts": available_contexts,
                "ambiguous_contexts_excluded": int(
                    sum(
                        candidate["_dedup_key"] in ambiguous_keys
                        for candidates in candidates_by_domain[domain.domain_id].values()
                        for candidate in candidates
                    )
                ),
            }
        )

    csv_path = output_dir / "all_domains_balanced.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "domain_id",
            "domain_name",
            "text",
            "source",
            "source_text_idx",
            "score",
            "domain_lexical_score",
            "source_feature_id",
            "source_group",
            "diagnostic_only",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)

    summary = {
        "source": str(analysis_path),
        "strategy": "observed_feature_contexts_with_domain_lexicon_filter",
        "independent_evaluation": False,
        "diagnostic_only": True,
        "warning": (
            "These contexts were selected and lexically filtered using the same feature "
            "analysis used to define the domains. They may be used for debugging/mapping, "
            "not as an independent estimate of router or MoE generalization."
        ),
        "ambiguous_unique_contexts_excluded": int(len(ambiguous_keys)),
        "samples_per_domain_requested": int(samples_per_domain),
        "domains": summary_domains,
    }
    write_json(output_dir / "validation_summary.json", summary)
    return summary


def default_checkpoint_path(data_path: str, layer_num: int) -> str:
    return os.path.join(data_path, f"models/checkpoints/topk_sae_layer_{layer_num}_best.pt")


def validate_feature_analysis_contract(
    matrix_metadata: Dict[str, object],
    layer_num: int,
    tokenizer,
    requested_top_m: int,
    allow_mismatch: bool,
    feature_token_source: str = "promoted",
) -> None:
    summary = matrix_metadata.get("source_format", {})
    if not isinstance(summary, dict):
        return
    mismatches = []
    source_layer = summary.get("layer_num")
    if source_layer is not None and int(source_layer) != int(layer_num):
        mismatches.append(f"analysis layer={source_layer}, requested layer={layer_num}")
    source_site = summary.get("site")
    if source_site is not None and str(source_site) != "resid_post":
        mismatches.append(f"analysis site={source_site!r}, expected 'resid_post'")
    if feature_token_source == "promoted":
        logit_lens = summary.get("logit_lens")
        if logit_lens == "disabled":
            mismatches.append("feature analysis has logit_lens='disabled'")
        source_logit_top_k = summary.get("logit_top_k")
        if source_logit_top_k is not None and int(source_logit_top_k) < requested_top_m:
            mismatches.append(
                f"analysis logit_top_k={source_logit_top_k}, requested top_m={requested_top_m}"
            )
    source_vocab_size = summary.get("tokenizer_vocab_size")
    if source_vocab_size is not None and int(source_vocab_size) != len(tokenizer):
        mismatches.append(
            f"analysis tokenizer size={source_vocab_size}, loaded tokenizer size={len(tokenizer)}"
        )
    source_vocab_hash = summary.get("tokenizer_vocab_sha256")
    if source_vocab_hash is not None:
        loaded_vocab_hash = tokenizer_vocab_fingerprint(tokenizer)
        if str(source_vocab_hash) != loaded_vocab_hash:
            mismatches.append("analysis and loaded tokenizer vocabularies have different hashes")
    if not mismatches:
        return
    message = "Incompatible feature-analysis metadata: " + "; ".join(mismatches)
    if allow_mismatch:
        print(f"WARNING: {message}")
    else:
        raise ValueError(message + ". Pass --allow-analysis-metadata-mismatch only for diagnostics.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster SAE logit-lens vectors into semantic domains and build validation sets."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--feature-analysis",
        default=None,
        help="Use an existing feature(s)_analysis.json instead of recomputing logit lens.",
    )
    parser.add_argument(
        "--context-analysis",
        default=None,
        help="features_analysis.json used only to build fallback validation contexts.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-domains", type=int, default=5)
    parser.add_argument(
        "--select-cluster-labels",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Restrict downstream domains to manually reviewed HDBSCAN cluster labels. "
            "All candidates are always written to cluster_candidates.json."
        ),
    )
    parser.add_argument("--top-m", type=int, default=128)
    parser.add_argument(
        "--feature-token-source",
        choices=["promoted", "trigger", "fallback"],
        default="promoted",
        help=(
            "Feature representation from saved analysis. 'trigger' is a diagnostic "
            "lexical proxy for partial checkpoints without a completed logit lens."
        ),
    )
    parser.add_argument("--top-tokens-per-domain", type=int, default=80)
    parser.add_argument(
        "--min-token-feature-support",
        type=int,
        default=2,
        help="A domain label token must occur among top logits of at least this many member features.",
    )
    parser.add_argument(
        "--min-distinct-domain-tokens",
        type=int,
        default=5,
        help="Reject clusters without enough supported tokens for a meaningful label.",
    )
    parser.add_argument(
        "--min-token-feature-fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of cluster features that must promote a label token.",
    )
    parser.add_argument("--logit-batch-size", type=int, default=128)
    parser.add_argument("--min-logit", type=float, default=0.0)
    parser.add_argument("--svd-components", type=int, default=50)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument(
        "--hdbscan-selection-method",
        choices=["eom", "leaf"],
        default="eom",
        help=(
            "EOM favors broad persistent clusters; leaf returns finer terminal "
            "clusters and is often more useful for semantic-domain triage."
        ),
    )
    parser.add_argument(
        "--max-domain-fraction",
        type=float,
        default=0.35,
        help="Exclude broad HDBSCAN clusters above this fraction of valid features.",
    )
    parser.add_argument("--min-token-chars", type=int, default=2)
    parser.add_argument(
        "--require-token-boundary",
        action="store_true",
        help="Drop continuation-only alphabetic BPE fragments from domain tokens.",
    )
    parser.add_argument(
        "--no-tfidf",
        action="store_true",
        help="Do not downweight tokens shared by many SAE features.",
    )
    parser.add_argument(
        "--include-unobserved-features",
        action="store_true",
        help="Include features absent from the saved activation analysis.",
    )
    parser.add_argument(
        "--allow-analysis-metadata-mismatch",
        action="store_true",
        help="Allow a layer/site mismatch only for diagnostic triage.",
    )
    parser.add_argument("--validation-csv", default=None)
    parser.add_argument("--samples-per-domain", type=int, default=100)
    parser.add_argument("--validation-tokens-per-domain", type=int, default=50)
    parser.add_argument("--min-domain-score", type=float, default=0.15)
    parser.add_argument("--exclusive-margin", type=float, default=0.05)
    parser.add_argument("--max-validation-texts", type=int, default=None)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not build text validation sets; useful when the available context analysis belongs to another SAE run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generate-if-needed", action="store_true")
    parser.add_argument("--min-required-domains", type=int, default=3)
    parser.add_argument(
        "--allow-underfilled-domains",
        action="store_true",
        help="Write outputs even when fewer than --min-required-domains domains are selected.",
    )
    parser.add_argument(
        "--keep-common-function-tokens",
        action="store_true",
        help="Keep stopwords and common function words in logit-lens clustering.",
    )
    parser.add_argument(
        "--approve-domains-for-downstream",
        action="store_true",
        help=(
            "Record an explicit human-review gate allowing MLP mapping/pruning. "
            "Use only after inspecting names, contexts, overlap, and diagnostics."
        ),
    )
    parser.add_argument(
        "--approve-existing-domains",
        action="store_true",
        help="Approve domains already present in --output-dir without rerunning triage.",
    )
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    positive = {
        "n_domains": args.n_domains,
        "top_m": args.top_m,
        "top_tokens_per_domain": args.top_tokens_per_domain,
        "min_token_feature_support": args.min_token_feature_support,
        "min_distinct_domain_tokens": args.min_distinct_domain_tokens,
        "logit_batch_size": args.logit_batch_size,
        "svd_components": args.svd_components,
        "min_cluster_size": args.min_cluster_size,
        "min_token_chars": args.min_token_chars,
        "samples_per_domain": args.samples_per_domain,
        "validation_tokens_per_domain": args.validation_tokens_per_domain,
        "min_required_domains": args.min_required_domains,
    }
    invalid = {name: value for name, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args.min_samples is not None and args.min_samples < 1:
        raise ValueError("--min-samples must be positive")
    if not 0.0 < args.max_domain_fraction <= 1.0:
        raise ValueError("--max-domain-fraction must be in (0, 1]")
    if not 0.0 <= args.min_token_feature_fraction <= 1.0:
        raise ValueError("--min-token-feature-fraction must be between 0 and 1")
    if args.min_required_domains > args.n_domains:
        raise ValueError("--min-required-domains cannot exceed --n-domains")
    if args.max_validation_texts is not None and args.max_validation_texts < 1:
        raise ValueError("--max-validation-texts must be positive")


def main() -> None:
    args = parse_args()
    validate_cli_args(args)

    data_path = Path(args.data_path)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(default_checkpoint_path(args.data_path, args.layer_num))
    )
    output_dir = Path(args.output_dir) if args.output_dir else data_path / "analysis/domain_triage"
    if args.approve_existing_domains:
        approve_existing_domains(output_dir)
        return
    validation_csv = Path(args.validation_csv) if args.validation_csv else data_path / "validation.csv"
    feature_analysis_path = Path(args.feature_analysis) if args.feature_analysis else None
    if args.context_analysis:
        context_analysis_path = Path(args.context_analysis)
    elif feature_analysis_path is not None:
        context_analysis_path = feature_analysis_path
    else:
        plural = data_path / "analysis/features_analysis.json"
        singular = data_path / "analysis/feature_analysis.json"
        context_analysis_path = plural if plural.exists() else singular

    if feature_analysis_path is None and not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
    if feature_analysis_path is not None and not feature_analysis_path.exists():
        raise FileNotFoundError(f"Feature analysis JSON not found: {feature_analysis_path}")
    if (
        not args.skip_validation
        and not validation_csv.exists()
        and not context_analysis_path.exists()
    ):
        raise FileNotFoundError(f"Validation CSV not found: {validation_csv}")

    device = find_device()
    print(f"Using device: {device}")
    sae = None
    if feature_analysis_path is None:
        print(f"Loading SAE checkpoint: {checkpoint_path}")
        sae = load_sae_checkpoint(str(checkpoint_path), device)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = None
    matrix_metadata: Dict[str, object] = {
        "source": "live_model_logit_lens",
        "observed_features_only": False,
        "use_tfidf": not args.no_tfidf,
    }
    if feature_analysis_path is not None:
        print(f"Loading precomputed feature analysis: {feature_analysis_path}")
        feature_matrix, feature_top_tokens, matrix_metadata = load_precomputed_feature_analysis(
            analysis_path=feature_analysis_path,
            tokenizer=tokenizer,
            min_logit=args.min_logit,
            keep_common_function_tokens=args.keep_common_function_tokens,
            min_token_chars=args.min_token_chars,
            require_token_boundary=args.require_token_boundary,
            observed_features_only=not args.include_unobserved_features,
            use_tfidf=not args.no_tfidf,
            top_m=args.top_m,
            feature_token_source=args.feature_token_source,
        )
        validate_feature_analysis_contract(
            matrix_metadata=matrix_metadata,
            layer_num=args.layer_num,
            tokenizer=tokenizer,
            requested_top_m=args.top_m,
            allow_mismatch=args.allow_analysis_metadata_mismatch,
            feature_token_source=args.feature_token_source,
        )
    else:
        assert sae is not None
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
            keep_common_function_tokens=args.keep_common_function_tokens,
            min_token_chars=args.min_token_chars,
            require_token_boundary=args.require_token_boundary,
            use_tfidf=not args.no_tfidf,
        )
        matrix_metadata.update(
            {
                "total_features": int(sae.d_sae),
                "nonempty_features": int(np.count_nonzero(feature_matrix.getnnz(axis=1))),
                "top_m": int(args.top_m),
            }
        )

        if (
            not args.include_unobserved_features
            and not args.skip_validation
            and context_analysis_path.exists()
        ):
            observed_ids = load_observed_feature_ids(context_analysis_path, sae.d_sae)
            observed_mask = np.zeros(sae.d_sae, dtype=bool)
            observed_mask[observed_ids] = True
            feature_matrix = feature_matrix.multiply(observed_mask[:, None]).tocsr()
            feature_matrix.eliminate_zeros()
            matrix_metadata.update(
                {
                    "observed_features_only": True,
                    "observed_features": int(len(observed_ids)),
                    "observation_source": str(context_analysis_path),
                    "nonempty_features": int(
                        np.count_nonzero(feature_matrix.getnnz(axis=1))
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "feature_token_matrix.npz", feature_matrix)
    write_feature_top_tokens(output_dir / "logit_lens_top_tokens.jsonl", feature_top_tokens)

    valid_feature_ids, reduced_vectors = reduce_feature_matrix(
        feature_matrix,
        args.svd_components,
        random_state=args.seed,
    )
    labels = cluster_features(
        reduced_vectors=reduced_vectors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.hdbscan_selection_method,
    )
    candidates = compute_cluster_candidates(
        labels=labels,
        valid_feature_ids=valid_feature_ids,
        feature_matrix=feature_matrix,
        reduced_vectors=reduced_vectors,
        feature_top_tokens=feature_top_tokens,
        top_tokens_per_domain=args.top_tokens_per_domain,
        min_token_feature_support=args.min_token_feature_support,
        min_token_feature_fraction=args.min_token_feature_fraction,
    )
    max_domain_size = max(1, int(len(valid_feature_ids) * args.max_domain_fraction))
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["size"] <= max_domain_size
        and len(candidate["top_tokens"]) >= args.min_distinct_domain_tokens
    ]
    excluded_broad_clusters = sum(
        candidate["size"] > max_domain_size for candidate in candidates
    )
    excluded_low_token_support_clusters = sum(
        candidate["size"] <= max_domain_size
        and len(candidate["top_tokens"]) < args.min_distinct_domain_tokens
        for candidate in candidates
    )
    eligible_cluster_labels = {
        int(candidate["cluster_label"]) for candidate in eligible_candidates
    }
    write_json(
        output_dir / "cluster_candidates.json",
        {
            "format": "semantic_domain_cluster_candidates_v1",
            "feature_token_source": args.feature_token_source,
            "candidates": [
                {
                    "cluster_label": int(candidate["cluster_label"]),
                    "size": int(candidate["size"]),
                    "cohesion": float(candidate["cohesion"]),
                    "top_tokens": candidate["top_tokens"],
                    "feature_ids": candidate["feature_ids"],
                    "eligible": int(candidate["cluster_label"]) in eligible_cluster_labels,
                }
                for candidate in candidates
            ],
        },
    )
    if args.select_cluster_labels is not None:
        requested_labels = list(dict.fromkeys(args.select_cluster_labels))
        available_labels = {
            int(candidate["cluster_label"]) for candidate in eligible_candidates
        }
        missing_labels = sorted(set(requested_labels) - available_labels)
        if missing_labels:
            raise ValueError(
                "Requested cluster labels are missing or ineligible: "
                f"{missing_labels}. Inspect cluster_candidates.json."
            )
        eligible_candidates = [
            candidate
            for candidate in eligible_candidates
            if int(candidate["cluster_label"]) in requested_labels
        ]
        if len(eligible_candidates) > args.n_domains:
            raise ValueError(
                f"Selected {len(eligible_candidates)} cluster labels but --n-domains="
                f"{args.n_domains}."
            )
    domains, similarity_matrix = select_orthogonal_domains(
        eligible_candidates,
        args.n_domains,
    )
    noise_features = int(np.sum(labels == -1))
    diagnostics = summarize_domain_quality(
        domains=domains,
        n_clusters=len(candidates),
        noise_features=noise_features,
        valid_features=len(valid_feature_ids),
        labels=labels,
        reduced_vectors=reduced_vectors,
    )
    diagnostics.update(
        {
            "candidate_clusters": int(len(candidates)),
            "eligible_clusters": int(len(eligible_candidates)),
            "excluded_broad_clusters": int(excluded_broad_clusters),
            "excluded_low_token_support_clusters": int(excluded_low_token_support_clusters),
            "max_domain_size": int(max_domain_size),
            "manually_selected_cluster_labels": (
                list(dict.fromkeys(args.select_cluster_labels))
                if args.select_cluster_labels is not None
                else None
            ),
        }
    )
    validate_domain_selection(
        domains=domains,
        min_required_domains=args.min_required_domains,
        allow_underfilled_domains=args.allow_underfilled_domains,
        diagnostics=diagnostics,
    )

    if args.generate_if_needed and model is None:
        print(f"Loading model for explicitly requested synthetic samples: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device).eval()

    write_json(
        output_dir / "domains.json",
        {
            "format": "semantic_domain_triage_v2",
            "domains": [asdict(domain) for domain in domains],
            "domain_similarity": similarity_matrix.tolist(),
            "n_clusters": len(candidates),
            "noise_features": noise_features,
            "valid_features": int(len(valid_feature_ids)),
            "candidate_clusters": int(len(candidates)),
            "eligible_clusters": int(len(eligible_candidates)),
            "excluded_broad_clusters": int(excluded_broad_clusters),
            "excluded_low_token_support_clusters": int(excluded_low_token_support_clusters),
            "max_domain_size": int(max_domain_size),
            "diagnostics": diagnostics,
            "filters": {
                "keep_common_function_tokens": bool(args.keep_common_function_tokens),
                "min_token_chars": int(args.min_token_chars),
                "require_token_boundary": bool(args.require_token_boundary),
                "use_tfidf": not args.no_tfidf,
                "include_unobserved_features": bool(args.include_unobserved_features),
                "max_domain_fraction": float(args.max_domain_fraction),
                "min_token_feature_support": int(args.min_token_feature_support),
                "min_token_feature_fraction": float(args.min_token_feature_fraction),
                "min_distinct_domain_tokens": int(args.min_distinct_domain_tokens),
                "min_required_domains": int(args.min_required_domains),
                "allow_underfilled_domains": bool(args.allow_underfilled_domains),
                "feature_token_source": args.feature_token_source,
                "select_cluster_labels": args.select_cluster_labels,
                "hdbscan_selection_method": args.hdbscan_selection_method,
            },
            "matrix_metadata": matrix_metadata,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "layer_num": int(args.layer_num),
            "feature_count": int(feature_matrix.shape[0]),
            "sae_checkpoint": str(checkpoint_path) if feature_analysis_path is None else None,
            "feature_analysis": str(feature_analysis_path) if feature_analysis_path else None,
            "manual_review_required": True,
            "downstream_approved": bool(args.approve_domains_for_downstream),
            "validation_source": (
                "skipped"
                if args.skip_validation
                else str(validation_csv)
                if validation_csv.exists()
                else str(context_analysis_path)
            ),
        },
    )
    write_assignments(
        output_dir / "feature_domain_assignments.csv",
        labels=labels,
        valid_feature_ids=valid_feature_ids,
        total_features=int(feature_matrix.shape[0]),
        domains=domains,
    )
    write_domain_report(
        output_dir / "domain_report.md",
        domains,
        similarity_matrix,
        diagnostics,
        downstream_approved=args.approve_domains_for_downstream,
    )

    print(f"Selected {len(domains)} domains.")
    if args.skip_validation:
        validation_summary = {
            "strategy": "skipped",
            "reason": "The available context analysis was not used because it belongs to a different SAE/data run.",
            "domains": [int(domain.domain_id) for domain in domains],
        }
        write_json(output_dir / "validation_summary.json", validation_summary)
    elif validation_csv.exists():
        print("Building validation sets from validation.csv...")
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
    elif context_analysis_path.exists():
        print("Building validation sets from saved feature contexts...")
        validation_summary = build_validation_sets_from_feature_analysis(
            domains=domains,
            analysis_path=context_analysis_path,
            output_dir=output_dir / "domain_validation",
            samples_per_domain=args.samples_per_domain,
        )
    else:
        raise FileNotFoundError(f"Validation CSV not found: {validation_csv}")

    # Keep a root-level summary as a convenient, unambiguous pointer to the
    # validation result written inside ``domain_validation``.
    write_json(output_dir / "validation_summary.json", validation_summary)
    print(f"Domain triage outputs written to: {output_dir}")
    print(json.dumps(validation_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
