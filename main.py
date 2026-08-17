"""CLI for storage-bounded Top-K SAE training.

The MoE-related scripts remain separate experiments.  This entry point only
owns the data preparation and SAE training path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

import autoencoder_training
import dataset_sequencing
from utils import find_device


PROFILES = {
    "tiny": {
        "model_name": "roneneldan/TinyStories-1M",
        "tokenizer_name": "EleutherAI/gpt-neo-125M",
        "data_path": "data/tinystories_dataset",
        "layer_num": 4,
        "seq_length": 256,
        "ds_fraction": 0.01,
        "model_batch_size": 4,
        "chunk_sequences": 32,
        "d_model": 64,
        "expansion_factor": 64,
        "k": 8,
        "batch_size_sae": 256,
        "num_epochs": 5,
    },
    "local-50gb": {
        "model_name": "EleutherAI/pythia-160m",
        "tokenizer_name": "EleutherAI/pythia-160m",
        "data_path": "data/pythia_160m_sae",
        "layer_num": 6,
        "seq_length": 256,
        "ds_fraction": 1.0,
        "model_batch_size": 4,
        "chunk_sequences": 32,
        "d_model": None,
        "expansion_factor": 16,
        "k": 64,
        "batch_size_sae": 1024,
        "num_epochs": 2,
    },
    "colab": {
        "model_name": "EleutherAI/pythia-160m",
        "tokenizer_name": "EleutherAI/pythia-160m",
        "data_path": "data/pythia_160m_sae",
        "layer_num": 6,
        "seq_length": 256,
        "ds_fraction": 1.0,
        "model_batch_size": 2,
        "chunk_sequences": 16,
        "d_model": None,
        "expansion_factor": 8,
        "k": 64,
        "batch_size_sae": 512,
        "num_epochs": 1,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Top-K SAE without materialising all activations.")
    parser.add_argument("--stage", choices=["sequence", "train-sae", "analyze", "all"], default="all")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="local-50gb")
    parser.add_argument("--model-name")
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--data-path")
    parser.add_argument("--input-file", default=None, help="CSV filename/path relative to data-path")
    parser.add_argument("--input-path", default=None, help="Directory/file containing Pile-style JSONL or JSONL.ZST shards")
    parser.add_argument("--file-pattern", default="*.jsonl*", help="Glob for JSONL shards inside input-path")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--layer-num", type=int)
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--ds-fraction", type=float)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument(
        "--analysis-sampling",
        choices=["head", "tail", "uniform"],
        default="head",
        help="Sampling mode for stage analyze when --max-sequences is set",
    )
    parser.add_argument("--min-seq-length", type=int, default=10)
    parser.add_argument("--batch-sentences", type=int, default=32)
    parser.add_argument("--model-batch-size", type=int)
    parser.add_argument("--chunk-sequences", type=int)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--expansion-factor", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--batch-size-sae", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--save-every-n-chunks", type=int, default=1)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--verbose",
        choices=["low", "high"],
        default="high",
        help="Progress output mode. 'low' prints periodic summaries without tqdm refreshes.",
    )
    parser.add_argument(
        "--verbose-interval",
        type=int,
        default=1000,
        help="Print interval in optimizer steps/chunks when --verbose low is used.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--no-interactive", action="store_true")
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    profile = PROFILES[args.profile].copy()
    for key, value in vars(args).items():
        if value is not None and key in profile:
            profile[key] = value
    profile.update({
        "stage": args.stage,
        "profile": args.profile,
        "input_file": args.input_file,
        "input_path": args.input_path,
        "file_pattern": args.file_pattern,
        "max_files": args.max_files,
        "max_documents": args.max_documents,
        "max_tokens": args.max_tokens,
        "max_sequences": args.max_sequences,
        "analysis_sampling": args.analysis_sampling,
        "min_seq_length": args.min_seq_length,
        "batch_sentences": args.batch_sentences,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "seed": args.seed,
        "checkpoint_path": args.checkpoint_path,
        "save_every_n_chunks": args.save_every_n_chunks,
        "keep_last_checkpoints": args.keep_last_checkpoints,
        "max_history": args.max_history,
        "max_steps": args.max_steps,
        "verbose": args.verbose,
        "verbose_interval": args.verbose_interval,
        "resume": args.resume,
        "no_interactive": args.no_interactive,
    })
    return argparse.Namespace(**profile)


def _load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    args.pad_token_id = int(pad_token_id)
    return tokenizer


def _prepare_sequences(args: argparse.Namespace, tokenizer) -> None:
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    if args.input_path is not None:
        dataset_sequencing.prepare_sequences_from_jsonl(
            input_path=args.input_path,
            output_dir=args.data_path,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            min_seq_length=args.min_seq_length,
            batch_sentences=args.batch_sentences,
            max_documents=args.max_documents,
            max_tokens=args.max_tokens,
            file_pattern=args.file_pattern,
            max_files=args.max_files,
            pad_token_id=args.pad_token_id,
            verbose=args.verbose,
            verbose_interval=args.verbose_interval,
        )
        return
    dataset_sequencing.prepare_sequences(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        min_seq_length=args.min_seq_length,
        batch_sentences=args.batch_sentences,
        ds_fraction=args.ds_fraction,
        interactive=not args.no_interactive,
        input_file=args.input_file,
        seed=args.seed,
        pad_token_id=args.pad_token_id,
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
    )


def _train_sae(args: argparse.Namespace, tokenizer) -> None:
    print(f"Using device: {find_device()}")
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    autoencoder_training.train_autoencoder_streaming(
        model=model,
        data_path=args.data_path,
        seq_length=args.seq_length,
        layer_num=args.layer_num,
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        k=args.k,
        batch_size_sae=args.batch_size_sae,
        model_batch_size=args.model_batch_size,
        chunk_sequences=args.chunk_sequences,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        save_every_n_chunks=args.save_every_n_chunks,
        keep_last_checkpoints=args.keep_last_checkpoints,
        max_history=args.max_history,
        max_sequences=args.max_sequences,
        max_steps=args.max_steps,
        model_name=args.model_name,
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
    )


def _analyze_sae(args: argparse.Namespace, tokenizer) -> None:
    import torch
    import features_analysis
    from autoencoder_training import TopKSAE

    checkpoint = Path(args.checkpoint_path) if args.checkpoint_path else (
        Path(args.data_path) / "models" / "checkpoints" / f"topk_sae_layer_{args.layer_num}_best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config", {})
    sae = TopKSAE(
        d_model=int(config["d_model"]),
        expansion_factor=int(config["expansion_factor"]),
        k=int(config["k"]),
    )
    sae.load_state_dict(payload["model_state_dict"])
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    features_analysis.analyze_sae(
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        path_dir=args.data_path,
        layer_num=args.layer_num,
        batch_size=args.model_batch_size,
        chunk_sequences=args.chunk_sequences,
        max_sequences=args.max_sequences,
        sampling=args.analysis_sampling,
        training_usage_counts=payload.get("usage_counts"),
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = _resolve_args(_parser().parse_args())
    print(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str))

    tokenizer = _load_tokenizer(args)
    if args.stage in {"sequence", "all"}:
        _prepare_sequences(args, tokenizer)
    if args.stage in {"train-sae", "all"}:
        _train_sae(args, tokenizer)
    if args.stage == "analyze":
        _analyze_sae(args, tokenizer)


if __name__ == "__main__":
    main()
