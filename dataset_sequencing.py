import os
import json
from pick import pick
import random
import numpy as np
from typing import Iterable, List, Iterator, Tuple
from syntok.segmenter import process
from tqdm import tqdm
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import find_n_proc, find_device


def _count_tokens(batch, tokenizer: AutoTokenizer):
    enc = tokenizer(
        batch["text"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return {"n_tokens": [len(ids) for ids in enc["input_ids"]]}


def _split_into_sentences(text: str) -> List[str]:
    sentences = []
    for paragraph in process(text or ""):
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
                                     total_tokens_in_dataset: int,
                                     output_dir: str):
    buffer = []
    seq_count = 0
    short_count = 0
    total_tokens = 0
    os.makedirs(os.path.join(output_dir, 'sequenced'), exist_ok=True)
    out_path = os.path.join(output_dir, "sequenced/TEMP_tokenized_not_padded.jsonl")

    with tqdm(total=total_tokens_in_dataset, unit="tok") as pbar:
        with open(out_path, "w") as out:
            for ids in iter_ids:
                if len(ids) == seq_length and not buffer:
                    out.write(json.dumps({"input_ids": ids}) + "\n")
                    seq_count += 1
                    total_tokens += len(ids)
                    continue
    
                if len(buffer) + len(ids) <= seq_length:
                    buffer.extend(ids)
                    pbar.update(len(ids))
                else:
                    if len(buffer) >= min_seq_len:
                        out.write(json.dumps({"input_ids": buffer}) + "\n")
                        seq_count += 1
                        total_tokens += len(buffer)
                    else:
                        short_count += 1
                    buffer = ids.copy()
    
            if buffer:
                if len(buffer) >= min_seq_len:
                    out.write(json.dumps({"input_ids": buffer}) + "\n")
                    seq_count += 1
                    total_tokens += len(buffer)
                else:
                    short_count += 1

    avg_len = total_tokens / seq_count if seq_count else 0
    return {"seq_count": seq_count, "short_count": short_count, "avg_len": avg_len, "total_tokens": total_tokens}


def _sentences_generator_from_texts(texts: Iterable[str]) -> Iterator[str]:
    for t in texts:
        for s in _split_into_sentences(t):
            yield s


def _token_ids_generator_from_texts(texts: Iterable[str],
                                   batch_sentences: int,
                                   tokenizer: AutoTokenizer,
                                   seq_length: int) -> Iterator[List[int]]:
    batch = []
    sent_gen = _sentences_generator_from_texts(texts)
    for sent in sent_gen:
        batch.append(sent)
        if len(batch) >= batch_sentences:
            ids_lists = _encode_sentences_batch(batch, tokenizer, seq_length)
            for ids in ids_lists:
                yield ids
            batch = []
    if batch:
        ids_lists = _encode_sentences_batch(batch, tokenizer, seq_length)
        for ids in ids_lists:
            yield ids


def _pad_sequences(output_dir: str, seq_length: int):
    rows = []
    padded_ouptut_file = os.path.join(output_dir, "sequenced/tokens_seqs_padded.npy")
    tokenized_not_padded_input_file = os.path.join(output_dir, "sequenced/TEMP_tokenized_not_padded.jsonl")

    tokens_seq_padded = np.empty((0, seq_length))
    with open(tokenized_not_padded_input_file, 'r') as in_f:
        for line in tqdm(in_f, desc="Padding sequences", unit="seq"):
            in_dict = json.loads(line)
            tmp = np.array(json.loads(line)['input_ids'], dtype=np.int32)
            padded = np.pad(tmp, (0, seq_length - tmp.size))
            rows.append(padded)

    tokens_seq_padded = np.stack(rows)
    print(f"Saving padded sequences to {padded_ouptut_file} ({tokens_seq_padded.size})")
    np.save(padded_ouptut_file, tokens_seq_padded)


def prepare_sequences(data_path: str, tokenizer: AutoTokenizer, seq_length: int, min_seq_length: int, batch_sentences: int, ds_fraction: float = 1.0, interactive: bool = True) -> Dataset:
    try:
        files = os.listdir(data_path)
    except Exception as e:
        raise Exception(f"Error listing files in {data_path}: {e}")

    candidates = []
    for file in files:
        if file.endswith(".csv"):
            candidates.append(file)

    if len(candidates) > 1 and interactive:
        option, index = pick(candidates, title="Select file to process", indicator="=>")
        path = os.path.join(data_path, option)
    else:
        print(f"Using {candidates[0]}")
        path = os.path.join(data_path, candidates[0])

    try:
        dataset = Dataset.from_csv(path)
    except Exception as e:
        raise Exception(f"Error loading dataset from {path}: {e}")

    dataset = dataset.filter(
        lambda x: isinstance(x["text"], str) and x["text"].strip()
    )
    if ds_fraction < 1.0:
        dataset = dataset.select(random.sample(range(len(dataset)), int(len(dataset) * ds_fraction)))
    
    print("Dataset loaded")

    can_reuse_dataset_info = False
    dataset_info_path = os.path.join(data_path, 'dataset_info.json')
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        if dataset_info["fraction_used"] != ds_fraction:
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
                "total_tokens_in_dataset": total_tokens_in_dataset,
            }, f)

    print(f"Total tokens found in the dataset: {total_tokens_in_dataset}")


    ids_gen = _token_ids_generator_from_texts(dataset["text"], batch_sentences, tokenizer, seq_length)
    stats = _pack_and_write_from_sentence_ids(ids_gen, seq_length, min_seq_length, total_tokens_in_dataset, data_path)
    _pad_sequences(data_path, seq_length)

    print("DONE:", stats)

