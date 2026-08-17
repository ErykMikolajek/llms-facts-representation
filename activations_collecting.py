"""Memory-bounded extraction of one transformer-layer activation stream.

The original implementation materialised every layer's hidden states and then
copied a temporary memmap into a second, equally large ``.npy`` file.  SAE
training only needs one layer at a time, so this module exposes an iterator of
bounded activation chunks instead.  A caller can consume a chunk immediately
and resume from the next sequence without keeping the complete activation
dataset on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import find_device


def get_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Return the transformer body without the language-model head."""
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    if hasattr(model, "transformer"):
        return model.transformer
    raise ValueError(
        "Unsupported model architecture: expected a GPT-NeoX ``gpt_neox`` "
        "or GPT-Neo ``transformer`` backbone."
    )


def get_transformer_layers(model: torch.nn.Module):
    backbone = get_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    raise ValueError("Could not locate transformer layers on the model backbone")


def get_transformer_layer(model: torch.nn.Module, layer_num: int) -> torch.nn.Module:
    layers = get_transformer_layers(model)
    if layer_num < 0 or layer_num >= len(layers):
        raise ValueError(f"layer_num={layer_num} is outside [0, {len(layers) - 1}]")
    return layers[layer_num]


def _hidden_from_layer_output(output) -> torch.Tensor:
    """Normalize GPT-Neo/GPT-NeoX block outputs to ``[B, S, D]``."""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise RuntimeError(
            "Layer hook did not receive a [batch, sequence, hidden] tensor; "
            f"got {type(output)!r} with shape {getattr(output, 'shape', None)}"
        )
    return output


@dataclass(frozen=True)
class TokenStore:
    tokens_path: Path
    mask_path: Path
    n_sequences: int
    seq_length: int
    d_model: Optional[int] = None


class TokenDataset(Dataset):
    """Memory-mapped token sequences and their valid-token masks."""

    def __init__(self, tokens_np_path: str | os.PathLike[str], seq_length: int,
                 mask_np_path: str | os.PathLike[str] | None = None):
        self.arr = np.load(tokens_np_path, mmap_mode="r")
        if self.arr.ndim != 2 or self.arr.shape[1] != seq_length:
            raise ValueError(
                f"Expected token array [N, {seq_length}], got {self.arr.shape}"
            )
        self.mask = None
        if mask_np_path is not None and Path(mask_np_path).exists():
            self.mask = np.load(mask_np_path, mmap_mode="r")
            if self.mask.shape != self.arr.shape:
                raise ValueError(
                    f"Attention mask shape {self.mask.shape} does not match tokens {self.arr.shape}"
                )

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.from_numpy(np.array(self.arr[idx], dtype=np.int64, copy=True)).long()
        if self.mask is None:
            # Compatibility with the old artifact format. New sequencing always
            # writes an explicit mask because token id 0 may be a real token.
            mask = tokens.ne(0)
        else:
            mask = torch.from_numpy(np.array(self.mask[idx], dtype=np.int64, copy=True)).bool()
        return tokens, mask


def open_token_store(data_path: str | os.PathLike[str], seq_length: int) -> TokenStore:
    root = Path(data_path) / "sequenced"
    tokens_path = root / "tokens_seqs_padded.npy"
    mask_path = root / "attention_mask.npy"
    dataset = TokenDataset(tokens_path, seq_length, mask_path)
    return TokenStore(
        tokens_path=tokens_path,
        mask_path=mask_path,
        n_sequences=len(dataset),
        seq_length=seq_length,
    )


def iter_activation_chunks(
    model: AutoModelForCausalLM,
    data_path: str | os.PathLike[str],
    seq_length: int,
    layer_num: int,
    batch_size: int = 4,
    chunk_sequences: int = 32,
    start_sequence: int = 0,
    max_sequences: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_dtype: np.dtype = np.float16,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield ``(sequence_start, sequence_end, activations)`` chunks.

    ``activations`` is flattened to ``[valid_tokens, d_model]``.  Padding is
    removed before the array reaches the SAE, while the model still receives a
    correct attention mask.  At most ``chunk_sequences * seq_length`` vectors
    are retained at once.
    """
    if batch_size < 1 or chunk_sequences < 1:
        raise ValueError("batch_size and chunk_sequences must be positive")

    store = open_token_store(data_path, seq_length)
    end_sequence = store.n_sequences
    if max_sequences is not None:
        end_sequence = min(end_sequence, start_sequence + max_sequences)
    if not 0 <= start_sequence <= end_sequence:
        raise ValueError(f"Invalid sequence range {start_sequence}:{end_sequence}")

    tokens = np.load(store.tokens_path, mmap_mode="r")
    masks = np.load(store.mask_path, mmap_mode="r") if store.mask_path.exists() else None
    device = device or find_device()
    model = model.to(device)
    model.eval()
    backbone = get_backbone(model)
    layer = get_transformer_layer(model, layer_num)
    hidden_capture: dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inputs, output):
        hidden_capture["value"] = _hidden_from_layer_output(output)

    handle = layer.register_forward_hook(capture_hook)
    try:
        for chunk_start in range(start_sequence, end_sequence, chunk_sequences):
            # Keep inference_mode scoped to model execution only.  The caller
            # must be able to attach autograd graphs to the SAE loss after the
            # generator yields; yielding from inside inference_mode would
            # silently disable gradients in the training loop.
            with torch.inference_mode():
                chunk_end = min(chunk_start + chunk_sequences, end_sequence)
                chunk_parts = []
                for batch_start in range(chunk_start, chunk_end, batch_size):
                    batch_end = min(batch_start + batch_size, chunk_end)
                    input_ids = torch.from_numpy(
                        np.array(tokens[batch_start:batch_end], copy=True)
                    ).long().to(device)
                    if masks is None:
                        attention_mask = input_ids.ne(0)
                    else:
                        attention_mask = torch.from_numpy(
                            np.array(masks[batch_start:batch_end], copy=True)
                        ).bool().to(device)

                    hidden_capture.pop("value", None)
                    # Calling the backbone avoids constructing the full LM
                    # vocabulary logits and avoids output_hidden_states=True.
                    try:
                        backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True,
                        )
                    except TypeError:
                        backbone(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )
                    if "value" not in hidden_capture:
                        raise RuntimeError("Target layer hook did not capture activations")

                    hidden = hidden_capture["value"].float()
                    valid = attention_mask.to(hidden.device)
                    chunk_parts.append(hidden[valid].cpu().numpy().astype(output_dtype, copy=False))

                if chunk_parts:
                    activations = np.concatenate(chunk_parts, axis=0)
                else:
                    activations = np.empty((0, int(model.config.hidden_size)), dtype=output_dtype)
            yield chunk_start, chunk_end, activations
    finally:
        handle.remove()


def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_path: str,
    seq_length: int,
    layer_num: int,
    checkpoint_freq: int = 10000,
    batch_size: int = 4,
    output_path: str | None = None,
    chunk_sequences: int = 32,
    max_sequences: Optional[int] = None,
    keep_full_file: bool = False,
):
    """Legacy-compatible collector.

    By default it writes no full activation artifact and returns a summary.
    ``keep_full_file=True`` is retained only for old experiments and should
    not be used for Pythia-scale runs.
    """
    del tokenizer, checkpoint_freq
    device = find_device()
    store = open_token_store(data_path, seq_length)
    hidden_size = int(model.config.hidden_size)
    if output_path is None:
        output_path = str(Path(data_path) / f"activations/activations_layer_{layer_num}.npy")

    full_chunks = [] if keep_full_file else None
    total_vectors = 0
    last_end = 0
    for start, end, activations in tqdm(
        iter_activation_chunks(
            model=model,
            data_path=data_path,
            seq_length=seq_length,
            layer_num=layer_num,
            batch_size=batch_size,
            chunk_sequences=chunk_sequences,
            max_sequences=max_sequences,
            device=device,
        ),
        desc="Collecting activation chunks",
    ):
        total_vectors += int(activations.shape[0])
        last_end = end
        if full_chunks is not None:
            full_chunks.append(activations)

    if full_chunks is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.concatenate(full_chunks, axis=0))

    return {
        "sequence_count": int(last_end),
        "activation_vectors": int(total_vectors),
        "d_model": hidden_size,
        "output_path": None if full_chunks is None else str(output_path),
        "storage_mode": "full_file" if full_chunks is not None else "streaming",
        "token_store": str(store.tokens_path),
    }
