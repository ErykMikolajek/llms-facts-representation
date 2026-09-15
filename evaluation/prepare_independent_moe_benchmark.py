"""Build a source-independent six-domain benchmark for the Gemma MoE.

The selected test sources were not used for domain discovery, router training,
pruning, confidence calibration, or the first holdout evaluation. Revisions are
pinned and the resulting artifact records hashes and provenance. Public-data
independence does not imply absence from the base model's pretraining corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REVISIONS = {
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "EleutherAI/bigbench": "f975b5fd41084e0f90085b98292430e8d56609dc",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google-research-datasets/mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
}

DOMAINS = {
    0: "prawo i orzecznictwo",
    1: "biomedycyna",
    2: "sport",
    3: "polityka i wiadomości",
    4: "matematyka / zapis LaTeX",
    5: "Python",
}

MMLU_LETTERS = ("A", "B", "C", "D")
BIOMEDICAL_CONFIGS = (
    "anatomy",
    "clinical_knowledge",
    "medical_genetics",
    "professional_medicine",
)
POLITICS_CONFIGS = (
    "high_school_government_and_politics",
    "security_studies",
    "us_foreign_policy",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split()).strip().lower()


def deterministic_sample(rows: Sequence[Any], count: int, seed: int) -> List[Any]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows, but source contains only {len(rows)}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:count]]


def make_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    rendered = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(MMLU_LETTERS, choices)
    )
    return f"Question: {question.strip()}\n\nChoices:\n{rendered}\n\nAnswer:"


def source_benchmark_id(
    domain_id: int,
    source: str,
    source_config: str,
    source_split: str,
    source_record_id: str,
) -> str:
    key = f"{source}:{source_config}:{source_split}:{source_record_id}"
    return f"d{domain_id}-{sha256_text(key)[:16]}"


def benchmark_row(
    *,
    domain_id: int,
    source: str,
    source_revision: str,
    source_config: str,
    source_split: str,
    source_record_id: str,
    task_type: str,
    prompt: str,
    choices: Sequence[str],
    answer_index: int | None,
    reference_answer: str,
    reference_completion: str,
    tests: Sequence[str] = (),
    test_setup_code: str = "",
    license_name: str,
    mc_instruction: str = "",
    mc_stem_label: str = "",
    mc_stem: str = "",
    mc_options: Sequence[str] = (),
) -> Dict[str, Any]:
    canonical_source = json.dumps(
        {
            "prompt": prompt,
            "choices": list(choices),
            "reference_answer": reference_answer,
            "tests": list(tests),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "benchmark_id": source_benchmark_id(
            domain_id, source, source_config, source_split, str(source_record_id)
        ),
        "domain_id": domain_id,
        "domain_name": DOMAINS[domain_id],
        "source": source,
        "source_revision": source_revision,
        "source_config": source_config,
        "source_split": source_split,
        "source_record_id": str(source_record_id),
        "task_type": task_type,
        "prompt": prompt,
        "choices": list(choices),
        "answer_index": answer_index,
        "reference_answer": reference_answer,
        "reference_completion": reference_completion,
        "tests": list(tests),
        "test_setup_code": test_setup_code,
        "license": license_name,
        "content_sha256": sha256_text(normalized_text(canonical_source)),
        "mc_instruction": mc_instruction,
        "mc_stem_label": mc_stem_label,
        "mc_stem": mc_stem,
        "mc_options": list(mc_options),
    }


def load_mmlu(config: str):
    from datasets import load_dataset

    return load_dataset(
        "cais/mmlu",
        config,
        split="test",
        revision=REVISIONS["cais/mmlu"],
    )


def prepare_mmlu_domain(
    domain_id: int,
    config: str,
    count: int,
    seed: int,
    excluded_ids: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    dataset = load_mmlu(config)
    excluded = set(excluded_ids)
    available = [
        row
        for row in dataset
        if source_benchmark_id(
            domain_id,
            "cais/mmlu",
            config,
            "test",
            sha256_text(str(row["question"]))[:20],
        )
        not in excluded
    ]
    selected = deterministic_sample(available, count=count, seed=seed)
    result = []
    for row in selected:
        choices = [str(choice).strip() for choice in row["choices"]]
        answer_index = int(row["answer"])
        result.append(
            benchmark_row(
                domain_id=domain_id,
                source="cais/mmlu",
                source_revision=REVISIONS["cais/mmlu"],
                source_config=config,
                source_split="test",
                source_record_id=sha256_text(str(row["question"]))[:20],
                task_type="multiple_choice",
                prompt=make_mmlu_prompt(str(row["question"]), choices),
                choices=[f" {letter}" for letter in MMLU_LETTERS],
                answer_index=answer_index,
                reference_answer=MMLU_LETTERS[answer_index],
                reference_completion=f" {MMLU_LETTERS[answer_index]}",
                license_name="MIT",
                mc_stem_label="Question",
                mc_stem=str(row["question"]),
                mc_options=choices,
            )
        )
    return result


def prepare_biomedical(
    count: int, seed: int, excluded_ids: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    base, remainder = divmod(count, len(BIOMEDICAL_CONFIGS))
    result: List[Dict[str, Any]] = []
    for index, config in enumerate(BIOMEDICAL_CONFIGS):
        quota = base + int(index < remainder)
        result.extend(
            prepare_mmlu_domain(
                1, config, quota, seed + 101 * index, excluded_ids=excluded_ids
            )
        )
    random.Random(seed).shuffle(result)
    return result


def prepare_politics(
    count: int, seed: int, excluded_ids: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    """Build a fresh political-domain mixture after earlier MMLU items were consumed."""
    base, remainder = divmod(count, len(POLITICS_CONFIGS))
    result: List[Dict[str, Any]] = []
    for index, config in enumerate(POLITICS_CONFIGS):
        quota = base + int(index < remainder)
        result.extend(
            prepare_mmlu_domain(
                3, config, quota, seed + 101 * index, excluded_ids=excluded_ids
            )
        )
    random.Random(seed).shuffle(result)
    return result


def prepare_sports(
    count: int,
    seed: int,
    excluded_ids: Sequence[str] = (),
    excluded_statements: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "EleutherAI/bigbench",
        repo_type="dataset",
        filename="sports_understanding.json",
        revision=REVISIONS["EleutherAI/bigbench"],
    )
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    buckets = {"plausible": [], "implausible": []}
    excluded = set(excluded_ids)
    excluded_text = set(excluded_statements)
    seen_text = set()
    for index, example in enumerate(payload["examples"]):
        if source_benchmark_id(
            2, "EleutherAI/bigbench", "sports_understanding", "benchmark", str(index)
        ) in excluded:
            continue
        statement_key = normalized_text(str(example["input"]))
        if statement_key in excluded_text or statement_key in seen_text:
            continue
        seen_text.add(statement_key)
        target = max(example["target_scores"], key=example["target_scores"].get)
        buckets[target].append((index, example))
    plausible_count = count // 2
    quotas = {"plausible": plausible_count, "implausible": count - plausible_count}
    chosen = []
    for offset, target in enumerate(("plausible", "implausible")):
        chosen.extend(deterministic_sample(buckets[target], quotas[target], seed + offset))
    random.Random(seed).shuffle(chosen)

    option_texts = ["plausible", "implausible"]
    choices = [" A", " B"]
    result = []
    for index, example in chosen:
        target = max(example["target_scores"], key=example["target_scores"].get)
        answer_index = 0 if target == "plausible" else 1
        instruction = "Determine whether the following sports statement is plausible or implausible."
        statement = str(example["input"]).strip()
        prompt = (
            f"{instruction}\n\nStatement: {statement}\n\nChoices:\n"
            "A. plausible\nB. implausible\n\nAnswer:"
        )
        result.append(
            benchmark_row(
                domain_id=2,
                source="EleutherAI/bigbench",
                source_revision=REVISIONS["EleutherAI/bigbench"],
                source_config="sports_understanding",
                source_split="benchmark",
                source_record_id=str(index),
                task_type="multiple_choice",
                prompt=prompt,
                choices=choices,
                answer_index=answer_index,
                reference_answer=MMLU_LETTERS[answer_index],
                reference_completion=choices[answer_index],
                license_name="Apache-2.0",
                mc_instruction=instruction,
                mc_stem_label="Statement",
                mc_stem=statement,
                mc_options=option_texts,
            )
        )
    return result


def gsm8k_final_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer, flags=re.DOTALL)
    if not match:
        raise ValueError(f"GSM8K answer lacks the expected #### delimiter: {answer!r}")
    return match.group(1).strip()


def prepare_gsm8k_math(count: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=REVISIONS["openai/gsm8k"],
    )
    selected = deterministic_sample(dataset, count, seed)
    result = []
    for index, row in enumerate(selected):
        answer = gsm8k_final_answer(str(row["answer"]))
        question = str(row["question"]).strip()
        result.append(
            benchmark_row(
                domain_id=4,
                source="openai/gsm8k",
                source_revision=REVISIONS["openai/gsm8k"],
                source_config="main",
                source_split="test",
                source_record_id=sha256_text(question)[:20],
                task_type="numeric_generation",
                prompt=f"Solve the problem. Give only the final numeric answer.\n\nProblem: {question}\n\nFinal answer:",
                choices=[],
                answer_index=None,
                reference_answer=answer,
                reference_completion=f" {answer}",
                license_name="MIT",
            )
        )
    return result


def prepare_elementary_math(
    count: int, seed: int, excluded_ids: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    """Create deterministic, balanced arithmetic MC items suitable for a 270M model."""
    rng = random.Random(seed)
    operations = ("addition", "subtraction", "multiplication", "division")
    rows = []
    seen = set()
    excluded = set(excluded_ids)
    while len(rows) < count:
        operation = operations[len(rows) % len(operations)]
        if operation == "addition":
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            answer, expression = a + b, rf"{a} + {b}"
        elif operation == "subtraction":
            a, b = rng.randint(0, 60), rng.randint(0, 40)
            if b > a:
                a, b = b, a
            answer, expression = a - b, rf"{a} - {b}"
        elif operation == "multiplication":
            a, b = rng.randint(0, 12), rng.randint(0, 12)
            answer, expression = a * b, rf"{a} \times {b}"
        else:
            b, answer = rng.randint(1, 12), rng.randint(0, 12)
            a, expression = b * answer, rf"{b * answer} \div {b}"
        key = (operation, expression)
        if key in seen:
            continue
        seen.add(key)
        source_record_id = f"{operation}:{expression}"
        if source_benchmark_id(
            4,
            "synthetic-elementary-arithmetic",
            operation,
            "generated-test",
            source_record_id,
        ) in excluded:
            continue
        distractors = {answer - 2, answer - 1, answer + 1, answer + 2, abs(answer - 3)}
        distractors.discard(answer)
        selected_distractors = sorted(distractors, key=lambda value: (abs(value - answer), value))[:3]
        while len(selected_distractors) < 3:
            candidate = answer + 3 + len(selected_distractors)
            if candidate != answer and candidate not in selected_distractors:
                selected_distractors.append(candidate)
        options = [answer, *selected_distractors]
        rng.shuffle(options)
        answer_index = options.index(answer)
        option_texts = [str(value) for value in options]
        instruction = "Compute the following elementary expression."
        stem = rf"\({expression}\)"
        prompt = make_mmlu_prompt(stem, option_texts).replace("Question:", f"{instruction}\n\nExpression:", 1)
        rows.append(
            benchmark_row(
                domain_id=4,
                source="synthetic-elementary-arithmetic",
                source_revision="v1",
                source_config=operation,
                source_split="generated-test",
                source_record_id=source_record_id,
                task_type="multiple_choice",
                prompt=prompt,
                choices=[f" {letter}" for letter in MMLU_LETTERS],
                answer_index=answer_index,
                reference_answer=MMLU_LETTERS[answer_index],
                reference_completion=f" {MMLU_LETTERS[answer_index]}",
                license_name="CC0-1.0",
                mc_instruction=instruction,
                mc_stem_label="Expression",
                mc_stem=stem,
                mc_options=option_texts,
            )
        )
    return rows


def prepare_python(
    count: int, seed: int, excluded_ids: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        split="test",
        revision=REVISIONS["google-research-datasets/mbpp"],
    )
    excluded = set(excluded_ids)
    available = [
        row
        for row in dataset
        if source_benchmark_id(
            5, "google-research-datasets/mbpp", "full", "test", str(row["task_id"])
        )
        not in excluded
    ]
    selected = deterministic_sample(available, count, seed)
    result = []
    for row in selected:
        task = str(row["text"]).strip()
        code = str(row["code"]).replace("\r\n", "\n").strip()
        result.append(
            benchmark_row(
                domain_id=5,
                source="google-research-datasets/mbpp",
                source_revision=REVISIONS["google-research-datasets/mbpp"],
                source_config="full",
                source_split="test",
                source_record_id=str(row["task_id"]),
                task_type="python_generation",
                prompt=f"Write a Python solution for the following task. Return only code.\n\nTask: {task}\n\n```python\n",
                choices=[],
                answer_index=None,
                reference_answer=code,
                reference_completion=f"{code}\n```",
                tests=[str(test) for test in row["test_list"]],
                test_setup_code=str(row.get("test_setup_code", "")),
                license_name="CC-BY-4.0",
            )
        )
    return result


def prior_content_hashes(experiment_dir: Path) -> set[str]:
    result: set[str] = set()
    for name in (
        "development_domains.csv",
        "development_general.csv",
        "holdout_domains.csv",
        "holdout_general.csv",
    ):
        path = experiment_dir / name
        with path.open("r", encoding="utf-8", newline="") as handle:
            result.update(str(row["content_sha256"]) for row in csv.DictReader(handle))
    return result


def consumed_benchmark_ids(experiment_dir: Path) -> set[str]:
    """Exclude every item already exposed in an earlier benchmark run."""
    result: set[str] = set()
    for path in sorted(experiment_dir.glob("independent_benchmark*/benchmark.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    result.add(str(json.loads(line)["benchmark_id"]))
    return result


def consumed_sports_statements(experiment_dir: Path) -> set[str]:
    """Recover semantic sports inputs so duplicate BIG-bench records stay excluded."""
    result: set[str] = set()
    pattern = re.compile(r"Statement:\s*(.*?)(?:\n\nChoices:|\n\nAnswer:)", re.DOTALL)
    for path in sorted(experiment_dir.glob("independent_benchmark*/benchmark.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("source_config") != "sports_understanding":
                    continue
                stem = str(row.get("mc_stem", "")).strip()
                if not stem:
                    match = pattern.search(str(row.get("prompt", "")))
                    stem = match.group(1).strip() if match else ""
                if stem:
                    result.add(normalized_text(stem))
    return result


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the independent Gemma MoE benchmark.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples-per-domain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--math-benchmark",
        choices=["elementary", "gsm8k"],
        default="elementary",
        help="Elementary balanced arithmetic is recommended for Gemma 270M.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_domain < 4:
        raise ValueError("--samples-per-domain must be at least 4")
    experiment_dir = Path(args.experiment_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_dir / "independent_benchmark_v3_sealed"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing benchmark: {output_dir}. "
            "A consumed benchmark must be replaced with a new directory and seed."
        )
    count = args.samples_per_domain
    excluded_ids = consumed_benchmark_ids(experiment_dir)
    excluded_sports = consumed_sports_statements(experiment_dir)
    math_rows = (
        prepare_elementary_math(count, args.seed + 50, excluded_ids=excluded_ids)
        if args.math_benchmark == "elementary"
        else prepare_gsm8k_math(count, args.seed + 50)
    )
    rows = [
        *prepare_mmlu_domain(
            0, "professional_law", count, args.seed + 10, excluded_ids=excluded_ids
        ),
        *prepare_biomedical(count, args.seed + 20, excluded_ids=excluded_ids),
        *prepare_sports(
            count,
            args.seed + 30,
            excluded_ids=excluded_ids,
            excluded_statements=excluded_sports,
        ),
        *prepare_politics(count, args.seed + 40, excluded_ids=excluded_ids),
        *math_rows,
        *prepare_python(count, args.seed + 60, excluded_ids=excluded_ids),
    ]
    benchmark_ids = [row["benchmark_id"] for row in rows]
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("Duplicate benchmark IDs")
    consumed_overlap = sorted(excluded_ids & set(benchmark_ids))
    if consumed_overlap:
        raise ValueError(f"Benchmark reuses {len(consumed_overlap)} consumed benchmark IDs")
    content_hashes = [row["content_sha256"] for row in rows]
    if len(set(content_hashes)) != len(content_hashes):
        duplicate_hashes = {
            digest for digest, frequency in Counter(content_hashes).items() if frequency > 1
        }
        duplicate_records = [
            (row["domain_name"], row["source_config"], row["source_record_id"])
            for row in rows
            if row["content_sha256"] in duplicate_hashes
        ]
        raise ValueError(f"Duplicate normalized benchmark contents: {duplicate_records}")

    prior_sources = {
        "coastalcph/lex_glue",
        "qiaojin/PubMedQA",
        "fancyzhx/dbpedia_14",
        "EleutherAI/hendrycks_math",
        "local-cpython-stdlib",
        "Salesforce/wikitext",
    }
    new_sources = {row["source"] for row in rows}
    source_overlap = sorted(prior_sources & new_sources)
    prior_hashes = prior_content_hashes(experiment_dir)
    exact_overlap = sorted(prior_hashes & set(content_hashes))
    if source_overlap or exact_overlap:
        raise ValueError(
            f"Benchmark is not independent: source_overlap={source_overlap}, "
            f"content_overlap={len(exact_overlap)}"
        )

    rows.sort(key=lambda row: (int(row["domain_id"]), str(row["benchmark_id"])))
    benchmark_path = output_dir / "benchmark.jsonl"
    write_jsonl(benchmark_path, rows)
    counts = Counter(int(row["domain_id"]) for row in rows)
    task_counts = Counter(str(row["task_type"]) for row in rows)
    manifest = {
        "format": "gemma_moe_independent_benchmark_v3",
        "seed": args.seed,
        "sealed_status": (
            "evaluation-only; do not tune masks, router, threshold, prompts, or hyperparameters "
            "against these answers"
        ),
        "independence_definition": (
            "No source repository or exact normalized content hash was used in domain discovery, "
            "router training, pruning, confidence calibration, or the first holdout. Public benchmark "
            "contamination in the base model pretraining remains unknown."
        ),
        "samples_per_domain": count,
        "excluded_consumed_benchmark_ids": len(excluded_ids),
        "excluded_consumed_sports_statements": len(excluded_sports),
        "consumed_benchmark_id_overlap": len(consumed_overlap),
        "math_benchmark": args.math_benchmark,
        "total_samples": len(rows),
        "domain_counts": {str(key): value for key, value in sorted(counts.items())},
        "task_type_counts": dict(sorted(task_counts.items())),
        "sources": [
            {
                "repository": repository,
                "revision": next(
                    row["source_revision"] for row in rows if row["source"] == repository
                ),
                "license": next(row["license"] for row in rows if row["source"] == repository),
            }
            for repository in sorted(new_sources)
        ],
        "source_overlap_with_previous_pipeline": source_overlap,
        "exact_content_overlap_with_previous_pipeline": len(exact_overlap),
        "benchmark_file": {
            "path": str(benchmark_path),
            "bytes": benchmark_path.stat().st_size,
            "sha256": sha256_file(benchmark_path),
        },
        "primary_metrics": {
            "all_domains": "paired change in teacher-forced reference-completion NLL",
            "multiple_choice": (
                "zero-shot normalized candidate-log-likelihood accuracy with deterministic "
                "answer-order robustness runs and item-level bootstrap"
            ),
            "math": (
                "balanced elementary arithmetic multiple choice with LaTeX notation"
                if args.math_benchmark == "elementary"
                else "GSM8K numeric exact match from deterministic generation plus reference NLL"
            ),
            "mbpp": (
                "reference-code NLL and generated-code syntax rate; pass@1 is intentionally disabled "
                "because Kaggle is not a security sandbox for arbitrary generated code"
            ),
        },
    }
    manifest_path = output_dir / "benchmark_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Prepared {len(rows)} benchmark samples in {benchmark_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
