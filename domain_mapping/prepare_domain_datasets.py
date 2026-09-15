"""Prepare independent development and holdout corpora for the six Gemma domains.

The script uses source-provided train/test boundaries whenever available,
records immutable Hugging Face revisions, deduplicates normalized text, and
keeps source groups disjoint.  Python samples come from the local CPython 3.11
standard library and are split by file before functions are extracted.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import random
import re
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


@dataclass(frozen=True)
class DomainDefinition:
    domain_id: int
    name: str
    slug: str


def normalized_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split()).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).lower().encode("utf-8")).hexdigest()


def stable_text_window(text: str, key: str, max_chars: int) -> str:
    text = text.replace("\x00", " ").strip()
    if len(text) <= max_chars:
        return text
    width = max_chars
    maximum_start = len(text) - width
    start = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    start %= maximum_start + 1
    if start > 0:
        boundary = text.find(" ", start, min(len(text), start + 160))
        if boundary >= 0:
            start = boundary + 1
    return text[start : start + width].strip()


def make_sample(
    domain: DomainDefinition,
    text: str,
    source: str,
    revision: str,
    split: str,
    record_id: str,
    license_name: str,
    max_chars: int,
    license_note: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    text = stable_text_window(text, f"{source}:{split}:{record_id}", max_chars=max_chars)
    if len(normalized_text(text)) < 120:
        return None
    digest = content_hash(text)
    return {
        "domain_id": domain.domain_id,
        "domain_name": domain.name,
        "text": text,
        "source": source,
        "source_revision": revision,
        "source_split": split,
        "source_record_id": str(record_id),
        "source_group": f"{source}:{split}:{record_id}",
        "license": license_name,
        "license_note": license_note or "",
        "content_sha256": digest,
        "diagnostic_only": False,
    }


def take_from_stream(
    dataset,
    count: int,
    seed: int,
    transform: Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]],
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    shuffle_buffer: int = 10_000,
) -> List[Dict[str, Any]]:
    shuffled = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    samples = []
    seen = set()
    for stream_index, row in enumerate(shuffled):
        if predicate is not None and not predicate(row):
            continue
        sample = transform(row, stream_index)
        if sample is None or sample["content_sha256"] in seen:
            continue
        seen.add(sample["content_sha256"])
        samples.append(sample)
        if len(samples) >= count:
            break
    if len(samples) < count:
        raise ValueError(f"Only {len(samples)} usable rows found; requested {count}")
    return samples


def resolve_revisions(repository_ids: Sequence[str]) -> Dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    return {
        repository_id: str(api.dataset_info(repository_id).sha)
        for repository_id in repository_ids
    }


def load_stream(repository: str, config: str, split: str, revision: str):
    from datasets import load_dataset

    return load_dataset(
        repository,
        config,
        split=split,
        revision=revision,
        streaming=True,
    )


def prepare_legal(
    domain: DomainDefinition,
    count: int,
    split: str,
    revision: str,
    seed: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    repository = "coastalcph/lex_glue"
    dataset = load_stream(repository, "scotus", split, revision)

    def transform(row, _index):
        raw = str(row.get("text", ""))
        record_id = content_hash(raw)
        return make_sample(
            domain, raw, repository, revision, f"scotus/{split}", record_id,
            "cc-by-4.0", max_chars,
        )

    return take_from_stream(dataset, count, seed, transform)


def pubmed_text(row: Dict[str, Any]) -> str:
    context = row.get("context", {})
    paragraphs = context.get("contexts", []) if isinstance(context, dict) else []
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    parts = [str(value) for value in paragraphs if str(value).strip()]
    question = str(row.get("question", "")).strip()
    answer = str(row.get("long_answer", "")).strip()
    if question:
        parts.insert(0, f"Research question: {question}")
    if answer:
        parts.append(f"Conclusion: {answer}")
    return "\n".join(parts)


def prepare_biomedical(
    domain: DomainDefinition,
    count: int,
    config: str,
    revision: str,
    seed: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    repository = "qiaojin/PubMedQA"
    dataset = load_stream(repository, config, "train", revision)

    def transform(row, index):
        raw = pubmed_text(row)
        record_id = str(row.get("pubid", index))
        return make_sample(
            domain,
            raw,
            repository,
            revision,
            f"{config}/train",
            record_id,
            "mit (dataset card)",
            max_chars,
            license_note=(
                "Dataset card declares MIT; underlying PubMed abstract text may retain "
                "publisher rights. Store locally for research and do not redistribute blindly."
            ),
        )

    return take_from_stream(dataset, count, seed, transform)


def dbpedia_label_index(dataset, wanted_name: str) -> int:
    feature = dataset.features.get("label")
    names = list(getattr(feature, "names", []) or [])
    normalized = {str(name).lower(): index for index, name in enumerate(names)}
    key = wanted_name.lower()
    if key not in normalized:
        raise ValueError(f"DBpedia label {wanted_name!r} absent from labels: {names}")
    return int(normalized[key])


def prepare_dbpedia_domain(
    domain: DomainDefinition,
    wanted_label: str,
    count: int,
    split: str,
    revision: str,
    seed: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    repository = "fancyzhx/dbpedia_14"
    dataset = load_stream(repository, "dbpedia_14", split, revision)
    label_id = dbpedia_label_index(dataset, wanted_label)

    def transform(row, index):
        title = str(row.get("title", "")).strip()
        body = str(row.get("content", "")).strip()
        raw = f"{title}. {body}" if title else body
        record_id = content_hash(raw) if raw else str(index)
        return make_sample(
            domain,
            raw,
            repository,
            revision,
            f"{wanted_label}/{split}",
            record_id,
            "cc-by-sa-3.0 (DBpedia source)",
            max_chars,
        )

    return take_from_stream(
        dataset,
        count,
        seed,
        transform,
        predicate=lambda row: int(row.get("label", -1)) == label_id,
    )


def prepare_math(
    domain: DomainDefinition,
    count: int,
    split: str,
    revision: str,
    seed: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    repository = "EleutherAI/hendrycks_math"
    quota = int(math.ceil(count / len(MATH_CONFIGS)))
    samples = []
    for config_index, config in enumerate(MATH_CONFIGS):
        dataset = load_stream(repository, config, split, revision)

        def transform(row, index, config=config):
            problem = str(row.get("problem", "")).strip()
            solution = str(row.get("solution", "")).strip()
            raw = f"Problem:\n{problem}\n\nSolution:\n{solution}"
            record_id = content_hash(f"{config}:{problem}") if problem else str(index)
            return make_sample(
                domain,
                raw,
                repository,
                revision,
                f"{config}/{split}",
                record_id,
                "mit (dataset card)",
                max_chars,
                license_note="Use pinned revision; the repository has a public licensing discussion.",
            )

        samples.extend(
            take_from_stream(
                dataset,
                quota,
                seed + config_index,
                transform,
                shuffle_buffer=3000,
            )
        )
    random.Random(seed).shuffle(samples)
    unique = {}
    for sample in samples:
        unique.setdefault(sample["content_sha256"], sample)
    if len(unique) < count:
        raise ValueError("Not enough unique MATH examples after combining configurations")
    return list(unique.values())[:count]


def python_function_samples(
    domain: DomainDefinition,
    split: str,
    count: int,
    seed: int,
    max_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    excluded_parts = {"site-packages", "test", "tests", "idlelib", "ensurepip", "encodings"}
    files = [
        path
        for path in stdlib.rglob("*.py")
        if not excluded_parts.intersection(path.relative_to(stdlib).parts)
        and not path.name.startswith("_sysconfigdata")
    ]
    split_files = []
    for path in files:
        relative = path.relative_to(stdlib).as_posix()
        bucket = int(hashlib.sha256(relative.encode("utf-8")).hexdigest()[:8], 16) % 5
        if (split == "development" and bucket != 0) or (split == "holdout" and bucket == 0):
            split_files.append(path)
    random.Random(seed).shuffle(split_files)

    samples = []
    seen = set()
    for path in split_files:
        relative = path.relative_to(stdlib).as_posix()
        try:
            source_text = path.read_text(encoding="utf-8")
            tree = ast.parse(source_text)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and hasattr(node, "end_lineno")
        ]
        nodes.sort(key=lambda node: (int(node.lineno), str(node.name)))
        for node in nodes[:4]:
            lines = source_text.splitlines()
            snippet = "\n".join(lines[int(node.lineno) - 1 : int(node.end_lineno)])
            record_id = f"{relative}:{node.lineno}:{node.name}"
            sample = make_sample(
                domain,
                snippet,
                "CPython-standard-library",
                platform.python_version(),
                split,
                record_id,
                "Python-Software-Foundation-License-2.0",
                max_chars,
            )
            if sample is None or sample["content_sha256"] in seen:
                continue
            sample["source_group"] = f"CPython-standard-library:{relative}"
            seen.add(sample["content_sha256"])
            samples.append(sample)
            if len(samples) >= count:
                manifest = {
                    "stdlib_root": str(stdlib),
                    "python_version": platform.python_version(),
                    "candidate_files": len(files),
                    "split_files": len(split_files),
                    "split_rule": "sha256(relative_path) modulo 5; bucket 0 is holdout",
                }
                return samples, manifest
    raise ValueError(f"Only {len(samples)} Python functions found for {split}")


def prepare_general(
    count: int,
    split: str,
    revision: str,
    seed: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    repository = "Salesforce/wikitext"
    dataset = load_stream(repository, "wikitext-2-raw-v1", split, revision)
    general = DomainDefinition(-1, "ogólne", "general")

    def transform(row, index):
        raw = str(row.get("text", ""))
        record_id = content_hash(raw) if raw.strip() else str(index)
        return make_sample(
            general,
            raw,
            repository,
            revision,
            f"wikitext-2-raw-v1/{split}",
            record_id,
            "cc-by-sa-3.0 and gfdl",
            max_chars,
        )

    return take_from_stream(dataset, count, seed, transform, shuffle_buffer=5000)


def load_domains(path: Path) -> Dict[str, DomainDefinition]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    domains = {
        str(domain["slug"]): DomainDefinition(
            int(domain["domain_id"]), str(domain["name"]), str(domain["slug"])
        )
        for domain in payload.get("domains", [])
    }
    required = {"legal", "biomedical", "sports", "politics_news", "math_latex", "python"}
    if set(domains) != required:
        raise ValueError(f"Expected exactly {sorted(required)}, got {sorted(domains)}")
    if not payload.get("downstream_approved", False):
        raise ValueError("Domain manifest is not approved for downstream use")
    return domains


def validate_disjoint_splits(
    development: Sequence[Dict[str, Any]],
    holdout: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    development_hashes = {sample["content_sha256"] for sample in development}
    holdout_hashes = {sample["content_sha256"] for sample in holdout}
    development_groups = {sample["source_group"] for sample in development}
    holdout_groups = {sample["source_group"] for sample in holdout}
    hash_overlap = development_hashes & holdout_hashes
    group_overlap = development_groups & holdout_groups
    if hash_overlap or group_overlap:
        raise ValueError(
            f"Development/holdout leakage: text_hashes={len(hash_overlap)}, "
            f"source_groups={len(group_overlap)}"
        )
    return {
        "development_unique_hashes": len(development_hashes),
        "holdout_unique_hashes": len(holdout_hashes),
        "content_hash_overlap": 0,
        "source_group_overlap": 0,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain_id", "domain_name", "text", "source", "source_revision",
        "source_split", "source_record_id", "source_group", "license",
        "license_note", "content_sha256", "diagnostic_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--development-per-domain", type=int, default=180)
    parser.add_argument("--holdout-per-domain", type=int, default=90)
    parser.add_argument("--development-general", type=int, default=200)
    parser.add_argument("--holdout-general", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.development_per_domain,
        args.holdout_per_domain,
        args.development_general,
        args.holdout_general,
        args.max_chars,
    ) < 1:
        raise ValueError("All dataset sizes and --max-chars must be positive")
    domains_path = Path(args.domains)
    output_dir = Path(args.output_dir)
    domains = load_domains(domains_path)
    repositories = (
        "coastalcph/lex_glue",
        "qiaojin/PubMedQA",
        "fancyzhx/dbpedia_14",
        "EleutherAI/hendrycks_math",
        "Salesforce/wikitext",
    )
    revisions = resolve_revisions(repositories)
    n_dev = args.development_per_domain
    n_holdout = args.holdout_per_domain
    seed = args.seed
    max_chars = args.max_chars

    development: Dict[str, List[Dict[str, Any]]] = {}
    holdout: Dict[str, List[Dict[str, Any]]] = {}
    development["legal"] = prepare_legal(
        domains["legal"], n_dev, "train", revisions["coastalcph/lex_glue"], seed, max_chars
    )
    holdout["legal"] = prepare_legal(
        domains["legal"], n_holdout, "test", revisions["coastalcph/lex_glue"], seed + 1, max_chars
    )
    development["biomedical"] = prepare_biomedical(
        domains["biomedical"], n_dev, "pqa_artificial",
        revisions["qiaojin/PubMedQA"], seed + 2, max_chars,
    )
    holdout["biomedical"] = prepare_biomedical(
        domains["biomedical"], n_holdout, "pqa_labeled",
        revisions["qiaojin/PubMedQA"], seed + 3, max_chars,
    )
    development["sports"] = prepare_dbpedia_domain(
        domains["sports"], "Athlete", n_dev, "train",
        revisions["fancyzhx/dbpedia_14"], seed + 4, max_chars,
    )
    holdout["sports"] = prepare_dbpedia_domain(
        domains["sports"], "Athlete", n_holdout, "test",
        revisions["fancyzhx/dbpedia_14"], seed + 5, max_chars,
    )
    development["politics_news"] = prepare_dbpedia_domain(
        domains["politics_news"], "OfficeHolder", n_dev, "train",
        revisions["fancyzhx/dbpedia_14"], seed + 6, max_chars,
    )
    holdout["politics_news"] = prepare_dbpedia_domain(
        domains["politics_news"], "OfficeHolder", n_holdout, "test",
        revisions["fancyzhx/dbpedia_14"], seed + 7, max_chars,
    )
    development["math_latex"] = prepare_math(
        domains["math_latex"], n_dev, "train",
        revisions["EleutherAI/hendrycks_math"], seed + 8, max_chars,
    )
    holdout["math_latex"] = prepare_math(
        domains["math_latex"], n_holdout, "test",
        revisions["EleutherAI/hendrycks_math"], seed + 9, max_chars,
    )
    development["python"], python_dev_meta = python_function_samples(
        domains["python"], "development", n_dev, seed + 10, max_chars
    )
    holdout["python"], python_holdout_meta = python_function_samples(
        domains["python"], "holdout", n_holdout, seed + 11, max_chars
    )
    development_general = prepare_general(
        args.development_general, "validation", revisions["Salesforce/wikitext"],
        seed + 12, max_chars,
    )
    holdout_general = prepare_general(
        args.holdout_general, "test", revisions["Salesforce/wikitext"],
        seed + 13, max_chars,
    )

    all_development = [row for slug in domains for row in development[slug]]
    all_holdout = [row for slug in domains for row in holdout[slug]]
    integrity = validate_disjoint_splits(
        all_development + development_general,
        all_holdout + holdout_general,
    )
    development_dir = output_dir / "domain_validation"
    for slug, domain in domains.items():
        write_jsonl(development_dir / f"domain_{domain.domain_id}.jsonl", development[slug])
    write_csv(output_dir / "development_domains.csv", all_development)
    write_csv(output_dir / "holdout_domains.csv", all_holdout)
    write_csv(output_dir / "development_general.csv", development_general)
    write_csv(output_dir / "holdout_general.csv", holdout_general)

    output_files = [
        output_dir / "development_domains.csv",
        output_dir / "holdout_domains.csv",
        output_dir / "development_general.csv",
        output_dir / "holdout_general.csv",
        *[development_dir / f"domain_{domain.domain_id}.jsonl" for domain in domains.values()],
    ]
    manifest = {
        "format": "gemma_domain_dataset_manifest_v1",
        "domains_path": str(domains_path),
        "seed": seed,
        "max_chars": max_chars,
        "development_per_domain": n_dev,
        "holdout_per_domain": n_holdout,
        "development_general": len(development_general),
        "holdout_general": len(holdout_general),
        "resolved_huggingface_revisions": revisions,
        "python_development": python_dev_meta,
        "python_holdout": python_holdout_meta,
        "integrity": integrity,
        "domain_counts": {
            slug: {"development": len(development[slug]), "holdout": len(holdout[slug])}
            for slug in domains
        },
        "files": {
            str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        },
        "holdout_policy": (
            "Holdout files must not be used to select clusters, thresholds, pruning "
            "budgets, router epochs, or confidence thresholds."
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Independent development/holdout datasets written to: {output_dir}")


if __name__ == "__main__":
    main()
