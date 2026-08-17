from __future__ import annotations

import os
import json
import re
import random
import io
from pathlib import Path
import numpy as np
from typing import Iterable, List, Iterator, Tuple, Optional, Sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import find_n_proc, find_device

try:
    from syntok.segmenter import process as _syntok_process
except ImportError:  # Keep CLI help and minimal smoke tests usable.
    _syntok_process = None


def _count_tokens(batch, tokenizer: AutoTokenizer):
    enc = tokenizer(
        batch["text"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return {"n_tokens": [len(ids) for ids in enc["input_ids"]]}


def _split_into_sentences(text: str) -> List[str]:
    if _syntok_process is None:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]
    sentences = []
    for paragraph in _syntok_process(text or ""):
        for sentence in paragraph:
            s = " ".join(tok.value for tok in sentence).strip()
            if s:
                sentences.append(s)
    return sentences


def _encode_sentences_batch(sentences: List[str], tokenizer: AutoTokenizer, seq_length: int) -> List[List[int]]:
    if not sentences:
        return []
    enc = tokenizer(
        sentences,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False
    )
    ids_list = enc["input_ids"]
    out = []
    for ids in ids_list:
        if len(ids) == 0:
            continue
        if len(ids) <= seq_length:
            out.append(ids)
        else:
            for i in range(0, len(ids), seq_length):
                out.append(ids[i:i+seq_length])
    return out


def _pack_and_write_from_sentence_ids(iter_ids: Iterator[List[int]],
                                     seq_length: int,
                                     min_seq_len: int,
                                     total_tokens_in_dataset: Optional[int],
                                     output_dir: str,
                                     verbose: str = "high",
                                     verbose_interval: int = 1000):
    if verbose not in {"low", "high"}:
        raise ValueError("verbose must be 'low' or 'high'")
    if verbose_interval < 1:
        raise ValueError("verbose_interval must be positive")
    buffer = []
    seq_count = 0
    short_count = 0
    total_tokens = 0
    os.makedirs(os.path.join(output_dir, 'sequenced'), exist_ok=True)
    out_path = os.path.join(output_dir, "sequenced/TEMP_tokenized_not_padded.jsonl")

    pbar = tqdm(total=total_tokens_in_dataset, unit="tok") if verbose == "high" else None
    last_report_tokens = 0
    with pbar if pbar is not None else _null_context() as progress:
        with open(out_path, "w") as out:
            for ids in iter_ids:
                if len(ids) == seq_length and not buffer:
                    out.write(json.dumps({"input_ids": ids}) + "\n")
                    seq_count += 1
                    total_tokens += len(ids)
                    if progress is not None:
                        progress.update(len(ids))
                    elif total_tokens - last_report_tokens >= verbose_interval:
                        print(
                            f"Sequencing progress: approximately {total_tokens:,} tokens; "
                            f"{seq_count:,} sequences"
                        )
                        last_report_tokens = total_tokens
                    continue
    
                if len(buffer) + len(ids) <= seq_length:
                    buffer.extend(ids)
                    if progress is not None:
                        progress.update(len(ids))
                else:
                    if len(buffer) >= min_seq_len:
                        out.write(json.dumps({"input_ids": buffer}) + "\n")
                        seq_count += 1
                        total_tokens += len(buffer)
                    else:
                        short_count += 1
                    buffer = ids.copy()
                    if progress is not None:
                        progress.update(len(ids))

                if progress is None and total_tokens - last_report_tokens >= verbose_interval:
                    print(f"Sequencing progress: approximately {total_tokens:,} tokens; {seq_count:,} sequences")
                    last_report_tokens = total_tokens
    
            if buffer:
                if len(buffer) >= min_seq_len:
                    out.write(json.dumps({"input_ids": buffer}) + "\n")
                    seq_count += 1
                    total_tokens += len(buffer)
                else:
                    short_count += 1

    if progress is None:
        print(f"Sequencing completed: {total_tokens:,} tokens; {seq_count:,} sequences")

    avg_len = total_tokens / seq_count if seq_count else 0
    return {"seq_count": seq_count, "short_count": short_count, "avg_len": avg_len, "total_tokens": total_tokens}


class _null_context:
    """Tiny context manager used to keep the streaming writer branch simple."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _sentences_generator_from_texts(texts: Iterable[str]) -> Iterator[str]:
    for t in texts:
        for s in _split_into_sentences(t):
            yield s


def _token_ids_generator_from_texts(texts: Iterable[str],
                                   batch_sentences: int,
                                   tokenizer: AutoTokenizer,
                                   seq_length: int,
                                   max_tokens: Optional[int] = None) -> Iterator[List[int]]:
    batch = []
    emitted_tokens = 0
    sent_gen = _sentences_generator_from_texts(texts)
    for sent in sent_gen:
        batch.append(sent)
        if len(batch) >= batch_sentences:
            ids_lists = _encode_sentences_batch(batch, tokenizer, seq_length)
            for ids in ids_lists:
                if max_tokens is not None and emitted_tokens >= max_tokens:
                    return
                if max_tokens is not None:
                    ids = ids[:max_tokens - emitted_tokens]
                if not ids:
                    return
                yield ids
                emitted_tokens += len(ids)
            batch = []
    if batch:
        ids_lists = _encode_sentences_batch(batch, tokenizer, seq_length)
        for ids in ids_lists:
            if max_tokens is not None and emitted_tokens >= max_tokens:
                return
            if max_tokens is not None:
                ids = ids[:max_tokens - emitted_tokens]
            if not ids:
                return
            yield ids
            emitted_tokens += len(ids)


def _jsonl_paths(input_path: str | os.PathLike[str],
                 file_pattern: str = "*.jsonl*",
                 max_files: Optional[int] = None) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(
            p for p in path.rglob(file_pattern)
            if p.is_file() and p.suffix in {".jsonl", ".zst", ".gz"}
        )
    else:
        raise FileNotFoundError(f"JSONL input path not found: {path}")
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No JSONL/JSONL.ZST files found under {path}")
    return paths


def _iter_jsonl_records(path: Path) -> Iterator[dict]:
    """Read plain or zstd/gzip JSONL without loading a shard into RAM."""
    if path.name.endswith(".zst"):
        import zstandard as zstd
        with path.open("rb") as raw:
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line_number, line in enumerate(text_stream, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    elif path.name.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as text_stream:
            for line_number, line in enumerate(text_stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    else:
        with path.open("r", encoding="utf-8") as text_stream:
            for line_number, line in enumerate(text_stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def _extract_jsonl_text(record: object) -> str:
    """Extract text from flat Pile JSONL or a ``{"root": {...}}`` wrapper."""
    if not isinstance(record, dict):
        return ""
    text = record.get("text")
    if isinstance(text, str):
        return text
    root = record.get("root")
    if isinstance(root, dict) and isinstance(root.get("text"), str):
        return root["text"]
    return ""


def _iter_jsonl_texts(paths: Sequence[Path], max_documents: Optional[int] = None,
                      source_counts: Optional[dict] = None) -> Iterator[str]:
    documents = 0
    for path in paths:
        file_documents = 0
        for record in _iter_jsonl_records(path):
            if max_documents is not None and documents >= max_documents:
                return
            text = _extract_jsonl_text(record)
            if not text.strip():
                continue
            documents += 1
            file_documents += 1
            yield text
        if source_counts is not None:
            source_counts[path.name] = file_documents


def prepare_sequences_from_jsonl(
    input_path: str,
    output_dir: str,
    tokenizer: AutoTokenizer,
    seq_length: int,
    min_seq_length: int = 10,
    batch_sentences: int = 32,
    max_documents: Optional[int] = None,
    max_tokens: Optional[int] = None,
    file_pattern: str = "*.jsonl*",
    max_files: Optional[int] = None,
    pad_token_id: int = 0,
    verbose: str = "high",
    verbose_interval: int = 1000,
) -> dict:
    """Convert Pile-style JSONL shards directly to token memmaps.

    The source files remain read-only in ``/kaggle/input`` (or another input
    directory). Only token arrays, masks and a manifest are created under
    ``output_dir``.
    """
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    paths = _jsonl_paths(input_path, file_pattern=file_pattern, max_files=max_files)
    source_counts: dict = {}
    texts = _iter_jsonl_texts(paths, max_documents=max_documents, source_counts=source_counts)
    ids_gen = _token_ids_generator_from_texts(
        texts=texts,
        batch_sentences=batch_sentences,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_tokens=max_tokens,
    )
    stats = _pack_and_write_from_sentence_ids(
        ids_gen,
        seq_length=seq_length,
        min_seq_len=min_seq_length,
        total_tokens_in_dataset=None,
        output_dir=output_dir,
        verbose=verbose,
        verbose_interval=verbose_interval,
    )
    sequence_count = _pad_sequences(
        output_dir,
        seq_length,
        pad_token_id=pad_token_id,
        verbose=verbose,
        verbose_interval=verbose_interval,
    )

    temp_path = Path(output_dir) / "sequenced/TEMP_tokenized_not_padded.jsonl"
    temp_path.unlink(missing_ok=True)
    manifest = {
        "format": "pile_jsonl_sequence_manifest_v1",
        "jsonl_text_fields": ["text", "root.text"],
        "source_path": str(Path(input_path).resolve()),
        "source_files": [p.name for p in paths],
        "source_document_counts": source_counts,
        "file_pattern": file_pattern,
        "max_files": max_files,
        "max_documents": max_documents,
        "max_tokens": max_tokens,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "seq_length": seq_length,
        "pad_token_id": pad_token_id,
        "stats": {**stats, "sequence_count": sequence_count},
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("DONE:", manifest["stats"])
    return manifest


def _pad_sequences(
    output_dir: str,
    seq_length: int,
    pad_token_id: int = 0,
    verbose: str = "high",
    verbose_interval: int = 1000,
):
    """Materialise tokens and an explicit validity mask without a RAM-sized list."""
    padded_output_file = os.path.join(output_dir, "sequenced/tokens_seqs_padded.npy")
    attention_mask_file = os.path.join(output_dir, "sequenced/attention_mask.npy")
    tokenized_not_padded_input_file = os.path.join(output_dir, "sequenced/TEMP_tokenized_not_padded.jsonl")

    with open(tokenized_not_padded_input_file, "r") as in_f:
        n_rows = sum(1 for line in in_f if line.strip())
    if n_rows == 0:
        raise ValueError("Sequencing produced no usable sequences")

    tokens_seq_padded = np.lib.format.open_memmap(
        padded_output_file, mode="w+", dtype=np.int32, shape=(n_rows, seq_length)
    )
    attention_mask = np.lib.format.open_memmap(
        attention_mask_file, mode="w+", dtype=np.uint8, shape=(n_rows, seq_length)
    )
    tokens_seq_padded[:] = pad_token_id
    attention_mask[:] = 0

    if verbose not in {"low", "high"}:
        raise ValueError("verbose must be 'low' or 'high'")
    if verbose_interval < 1:
        raise ValueError("verbose_interval must be positive")
    with open(tokenized_not_padded_input_file, 'r') as in_f:
        input_lines = (
            tqdm(in_f, total=n_rows, desc="Padding sequences", unit="seq")
            if verbose == "high" else in_f
        )
        for row_idx, line in enumerate(input_lines):
            ids = np.asarray(json.loads(line)['input_ids'], dtype=np.int32)
            if ids.size > seq_length:
                raise ValueError(f"Sequence {row_idx} has {ids.size} tokens, expected <= {seq_length}")
            tokens_seq_padded[row_idx, :ids.size] = ids
            attention_mask[row_idx, :ids.size] = 1
            if verbose == "low" and (row_idx + 1) % verbose_interval == 0:
                print(f"Padding progress: {row_idx + 1:,}/{n_rows:,} sequences")

    tokens_seq_padded.flush()
    attention_mask.flush()
    del tokens_seq_padded, attention_mask
    print(f"Saved tokens to {padded_output_file} and mask to {attention_mask_file} ({n_rows} sequences)")
    return n_rows


def prepare_sequences(data_path: str, tokenizer: AutoTokenizer, seq_length: int, min_seq_length: int,
                      batch_sentences: int, ds_fraction: float = 1.0, interactive: bool = True,
                      input_file: str = None, seed: int = 0, pad_token_id: int = 0,
                      verbose: str = "high", verbose_interval: int = 1000) -> Dataset:
    try:
        files = os.listdir(data_path)
    except Exception as e:
        raise Exception(f"Error listing files in {data_path}: {e}")

    candidates = []
    for file in files:
        if file.endswith(".csv"):
            candidates.append(file)

    if input_file is not None:
        path = input_file if os.path.isabs(input_file) else os.path.join(data_path, input_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input CSV not found: {path}")
    elif len(candidates) > 1 and interactive:
        from pick import pick
        option, index = pick(candidates, title="Select file to process", indicator="=>")
        path = os.path.join(data_path, option)
    else:
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        print(f"Using {candidates[0]}")
        path = os.path.join(data_path, candidates[0])

    try:
        from datasets import Dataset
        dataset = Dataset.from_csv(path)
    except Exception as e:
        raise Exception(f"Error loading dataset from {path}: {e}")

    dataset = dataset.filter(
        lambda x: isinstance(x["text"], str) and x["text"].strip()
    )
    if not 0 < ds_fraction <= 1.0:
        raise ValueError("ds_fraction must be in (0, 1]")
    if ds_fraction < 1.0:
        rng = random.Random(seed)
        sample_size = max(1, int(len(dataset) * ds_fraction))
        dataset = dataset.select(rng.sample(range(len(dataset)), sample_size))
    
    print("Dataset loaded")

    can_reuse_dataset_info = False
    dataset_info_path = os.path.join(data_path, 'dataset_info.json')
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        if (dataset_info.get("fraction_used") != ds_fraction or
                dataset_info.get("seed") != seed or
                dataset_info.get("source_file") != os.path.abspath(path) or
                dataset_info.get("seq_length") != seq_length):
            can_reuse_dataset_info = False
        else:
            total_tokens_in_dataset = dataset_info["total_tokens_in_dataset"]
            can_reuse_dataset_info = True
            print(f"Reusing dataset info for fraction {ds_fraction}")

    if not can_reuse_dataset_info:
        num_proc = find_n_proc()
        counted = dataset.map(_count_tokens, batched=True, num_proc=num_proc, fn_kwargs={"tokenizer": tokenizer})
        total_tokens_in_dataset = sum(counted["n_tokens"])
        with open(dataset_info_path, 'w') as f:
            json.dump({
                "fraction_used": ds_fraction,
                "seed": seed,
                "source_file": os.path.abspath(path),
                "seq_length": seq_length,
                "total_tokens_in_dataset": total_tokens_in_dataset,
            }, f)

    print(f"Total tokens found in the dataset: {total_tokens_in_dataset}")


    ids_gen = _token_ids_generator_from_texts(dataset["text"], batch_sentences, tokenizer, seq_length)
    stats = _pack_and_write_from_sentence_ids(
        ids_gen,
        seq_length,
        min_seq_length,
        total_tokens_in_dataset,
        data_path,
        verbose=verbose,
        verbose_interval=verbose_interval,
    )
    sequence_count = _pad_sequences(
        data_path,
        seq_length,
        pad_token_id=pad_token_id,
        verbose=verbose,
        verbose_interval=verbose_interval,
    )

    temp_path = os.path.join(data_path, "sequenced/TEMP_tokenized_not_padded.jsonl")
    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass

    print("DONE:", {**stats, "sequence_count": sequence_count})
    return stats
