"""Streaming analysis of a pretrained Gemma Scope 2 residual-stream SAE.

This module deliberately does not train an SAE.  It loads a Gemma 3 model and
the matching pretrained Gemma Scope 2 SAE through SAELens, streams the already
prepared token memmaps, and writes:

* ``analysis/feature_cards.jsonl`` - one card for every observed feature,
* ``analysis/feature_analysis.json`` - summary plus all feature records,
* ``analysis/feature_analysis.txt`` - a human-readable report.

The first pilot targets Gemma 3 270M and the ``resid_post`` site.  The code is
kept separate from the Pythia training/checkpoint path because SAELens SAEs
use JumpReLU and their inference API is not compatible with ``TopKSAE``.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_pipeline import dataset_sequencing
from sae_pipeline.activations_collecting import _hidden_from_layer_output, get_backbone, get_transformer_layer
from common.utils import find_device, tokenizer_vocab_fingerprint


DEFAULT_MODEL_NAME = "google/gemma-3-270m"
DEFAULT_TOKENIZER_NAME = DEFAULT_MODEL_NAME
# SAELens' release key is ``...-res``; it maps to the ``resid_post/`` folder
# in google/gemma-scope-2-270m-pt.  ``...-resid_post`` was used in an older
# draft of this pilot and is kept as a CLI compatibility alias below.
DEFAULT_RELEASE = "gemma-scope-2-270m-pt-res"
DEFAULT_SAE_ID = "layer_9_width_16k_l0_medium"
DEFAULT_LAYER_NUM = 9
DEFAULT_SEQ_LENGTH = 1024


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_torch_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_config_matches(
    loaded: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Compare resume settings while allowing Kaggle input mount relocation."""
    loaded = dict(loaded)
    expected = dict(expected)
    for key in ("tokens_path", "attention_mask_path"):
        loaded_path = loaded.pop(key, None)
        expected_path = expected.pop(key, None)
        if loaded_path is None or expected_path is None:
            if loaded_path != expected_path:
                return False
        elif Path(loaded_path).name != Path(expected_path).name:
            return False
    return loaded == expected


def _load_sae(release: str, sae_id: str, device: torch.device):
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise ImportError(
            "Gemma Scope analysis requires sae-lens. Install it with "
            "`pip install -U sae-lens`."
        ) from exc

    if release == "gemma-scope-2-270m-pt-resid_post":
        release = "gemma-scope-2-270m-pt-res"

    loaded = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=str(device),
    )
    # SAELens <=5 returned (sae, cfg, sparsity); v6 returns the SAE directly.
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval()


def _sae_metadata(sae) -> dict[str, Any]:
    cfg = getattr(sae, "cfg", None)
    metadata = getattr(cfg, "metadata", None)
    d_in = getattr(cfg, "d_in", None)
    d_sae = getattr(cfg, "d_sae", None)
    return {
        "d_in": int(d_in) if d_in is not None else -1,
        "d_sae": int(d_sae) if d_sae is not None else -1,
        "model_name": getattr(metadata, "model_name", None),
        "hook_name": getattr(metadata, "hook_name", None),
        "hf_hook_name": getattr(metadata, "hf_hook_name", None),
        "normalize_activations": getattr(cfg, "normalize_activations", None),
        "apply_b_dec_to_input": getattr(cfg, "apply_b_dec_to_input", None),
    }


def _validate_sae_model_contract(sae, model, layer_num: int) -> dict[str, Any]:
    metadata = _sae_metadata(sae)
    hidden_size = int(model.config.hidden_size)
    if metadata["d_in"] != hidden_size:
        raise ValueError(
            f"SAE d_in={metadata['d_in']} does not match model hidden_size={hidden_size}"
        )
    expected_hook = f"blocks.{layer_num}.hook_resid_post"
    if metadata["hook_name"] and metadata["hook_name"] != expected_hook:
        raise ValueError(
            f"SAE hook {metadata['hook_name']!r} does not match requested layer/site "
            f"{expected_hook!r}"
        )
    # This also verifies that the requested layer exists even when a completed
    # checkpoint causes the activation loop to be skipped.
    get_transformer_layer(model, layer_num)
    return metadata


def _model_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"Unknown dtype: {requested}")
    if device.type == "cuda":
        # T4 (compute capability 7.5), the default accelerator in the Kaggle
        # workflow, does not provide native bfloat16 arithmetic.
        supports_bfloat16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if supports_bfloat16 else torch.float16
    return torch.float32


def _load_model(model_name: str, device: torch.device, dtype_name: str):
    dtype = _model_dtype(device, dtype_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
    except TypeError:
        # Compatibility with newer Transformers versions which renamed the
        # loading keyword while retaining the older one in some model classes.
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    return model.to(device).eval(), dtype


def _analysis_ranges(
    sequence_count: int,
    max_sequences: Optional[int],
    chunk_sequences: int,
    sampling: str,
) -> list[tuple[int, int]]:
    if max_sequences is None or max_sequences >= sequence_count:
        return [(0, sequence_count)]
    if max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    if sampling == "head":
        return [(0, max_sequences)]
    if sampling == "tail":
        return [(sequence_count - max_sequences, sequence_count)]
    if sampling != "uniform":
        raise ValueError("sampling must be head, tail or uniform")

    if chunk_sequences < 1:
        raise ValueError("chunk_sequences must be positive")
    window_count = math.ceil(max_sequences / chunk_sequences)
    sizes = [chunk_sequences] * (window_count - 1)
    sizes.append(max_sequences - sum(sizes))
    if window_count == 1:
        start = (sequence_count - max_sequences) // 2
        return [(start, start + max_sequences)]

    # Spread all rows that are not sampled between the windows. This produces
    # exactly max_sequences distinct rows, including samples close to N.
    total_gap = sequence_count - max_sequences
    base_gap, extra_gaps = divmod(total_gap, window_count - 1)
    ranges = []
    start = 0
    for index, size in enumerate(sizes):
        end = start + size
        ranges.append((start, end))
        if index < window_count - 1:
            start = end + base_gap + (1 if index < extra_gaps else 0)
    return ranges


def _prepare_sequences_if_needed(
    data_path: Path,
    input_path: Optional[str],
    tokenizer,
    seq_length: int,
    max_documents: Optional[int],
    max_tokens: Optional[int],
    verbose: str,
    verbose_interval: int,
    tokens_path: Optional[str] = None,
    attention_mask_path: Optional[str] = None,
    require_prepared: bool = False,
) -> tuple[Path, Path]:
    resolved_tokens_path = (
        Path(tokens_path)
        if tokens_path is not None
        else data_path / "sequenced" / "tokens_seqs_padded.npy"
    )
    resolved_mask_path = (
        Path(attention_mask_path)
        if attention_mask_path is not None
        else data_path / "sequenced" / "attention_mask.npy"
    )
    if resolved_tokens_path.exists() and resolved_mask_path.exists():
        return resolved_tokens_path, resolved_mask_path
    if require_prepared or tokens_path is not None or attention_mask_path is not None:
        missing = [
            str(path)
            for path in (resolved_tokens_path, resolved_mask_path)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Gotowe sekwencje zostały wymagane, ale brakuje plików: "
            + ", ".join(missing)
        )
    if input_path is None:
        raise FileNotFoundError(
            f"Missing prepared sequences under {data_path}; provide --input-path "
            "to the The Pile JSONL source."
        )

    data_path.mkdir(parents=True, exist_ok=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    dataset_sequencing.prepare_sequences_from_jsonl(
        input_path=input_path,
        output_dir=str(data_path),
        tokenizer=tokenizer,
        seq_length=seq_length,
        min_seq_length=10,
        batch_sentences=32,
        max_documents=max_documents,
        max_tokens=max_tokens,
        file_pattern="*.jsonl*",
        pad_token_id=int(pad_token_id),
        verbose=verbose,
        verbose_interval=verbose_interval,
    )
    return resolved_tokens_path, resolved_mask_path


def iter_resid_post_batches(
    model: torch.nn.Module,
    data_path: str | os.PathLike[str],
    seq_length: int,
    layer_num: int,
    model_batch_size: int,
    ranges: list[tuple[int, int]],
    device: torch.device,
    tokens_path: Optional[str | os.PathLike[str]] = None,
    attention_mask_path: Optional[str | os.PathLike[str]] = None,
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield model hidden states while preserving sequence/token alignment."""
    if model_batch_size < 1:
        raise ValueError("model_batch_size must be positive")

    root = Path(data_path) / "sequenced"
    tokens_source = Path(tokens_path) if tokens_path is not None else root / "tokens_seqs_padded.npy"
    mask_path = Path(attention_mask_path) if attention_mask_path is not None else root / "attention_mask.npy"
    tokens_mm = np.load(tokens_source, mmap_mode="r")
    masks_mm = np.load(mask_path, mmap_mode="r") if mask_path.exists() else None
    backbone = get_backbone(model)
    layer = get_transformer_layer(model, layer_num)
    capture: dict[str, torch.Tensor] = {}

    def capture_hook(_module, _inputs, output):
        capture["value"] = _hidden_from_layer_output(output)

    handle = layer.register_forward_hook(capture_hook)
    try:
        for range_start, range_end in ranges:
            for batch_start in range(range_start, range_end, model_batch_size):
                batch_end = min(batch_start + model_batch_size, range_end)
                input_ids = torch.from_numpy(
                    np.array(tokens_mm[batch_start:batch_end], copy=True)
                ).long().to(device)
                if masks_mm is None:
                    attention_mask = input_ids.ne(0)
                else:
                    attention_mask = torch.from_numpy(
                        np.array(masks_mm[batch_start:batch_end], copy=True)
                    ).bool().to(device)

                capture.pop("value", None)
                with torch.inference_mode():
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
                if "value" not in capture:
                    raise RuntimeError(
                        f"Could not capture resid_post at layer {layer_num}."
                    )

                # Move only one model batch to CPU.  The SAE encoder will use
                # smaller microbatches on the selected device below.
                hidden = capture["value"].detach().to("cpu")
                yield (
                    batch_start,
                    batch_end,
                    hidden,
                    torch.from_numpy(np.array(tokens_mm[batch_start:batch_end], copy=True)).long(),
                    torch.from_numpy(
                        np.array(
                            masks_mm[batch_start:batch_end], copy=True
                        ) if masks_mm is not None else
                        np.array(tokens_mm[batch_start:batch_end], copy=True) != 0
                    ).bool(),
                )
    finally:
        handle.remove()


class FeatureAccumulator:
    """Memory-bounded feature/token/context aggregation."""

    def __init__(
        self,
        sae,
        tokenizer,
        device: torch.device,
        microbatch_tokens: int,
        top_k_examples: int,
        top_k_tokens: int,
        max_token_entries: int,
        context_size: int,
        activation_threshold: float,
    ):
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device
        self.microbatch_tokens = max(1, int(microbatch_tokens))
        self.top_k_examples = max(1, int(top_k_examples))
        self.top_k_tokens = max(1, int(top_k_tokens))
        self.max_token_entries = max(self.top_k_tokens, int(max_token_entries))
        self.context_size = max(0, int(context_size))
        self.activation_threshold = float(activation_threshold)
        sae_width = getattr(sae, "d_sae", None)
        if sae_width is None:
            sae_cfg = getattr(sae, "cfg", None)
            sae_width = (
                sae_cfg.get("d_sae") if isinstance(sae_cfg, dict)
                else getattr(sae_cfg, "d_sae", None)
            )
        if sae_width is None:
            raise ValueError("Loaded SAE does not expose d_sae")
        self.num_features = int(sae_width)
        self.counts = np.zeros(self.num_features, dtype=np.int64)
        self.sums = np.zeros(self.num_features, dtype=np.float64)
        self.maxima = np.zeros(self.num_features, dtype=np.float32)
        self.token_counts: dict[int, Counter[int]] = {}
        self.example_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = {}
        self._serial = 0
        self.total_tokens = 0
        self._decoded_tokens: dict[int, str] = {}

    def _decode_token(self, token_id: int) -> str:
        if token_id not in self._decoded_tokens:
            self._decoded_tokens[token_id] = self.tokenizer.decode(
                [int(token_id)], skip_special_tokens=False
            )
        return self._decoded_tokens[token_id]

    def _update_bounded_counter(self, feature_id: int, token_id: int) -> None:
        counter = self.token_counts.setdefault(feature_id, Counter())
        if token_id in counter or len(counter) < self.max_token_entries:
            counter[token_id] += 1
            return
        # Space-saving update: preserve a bounded approximation to the most
        # frequent trigger tokens instead of allocating feature x vocabulary.
        min_token, min_count = min(counter.items(), key=lambda item: item[1])
        del counter[min_token]
        counter[token_id] = min_count + 1

    def _context(self, row_tokens: np.ndarray, position: int, valid_length: int):
        start = max(0, position - self.context_size)
        end = min(valid_length, position + self.context_size + 1)
        before = self.tokenizer.decode(row_tokens[start:position].tolist(), skip_special_tokens=False)
        trigger = self._decode_token(int(row_tokens[position]))
        after = self.tokenizer.decode(row_tokens[position + 1:end].tolist(), skip_special_tokens=False)
        return before, after, f"{before}[[{trigger}]]{after}"

    def _update_example(
        self,
        feature_id: int,
        score: float,
        row_tokens: np.ndarray,
        position: int,
        valid_length: int,
        sequence_id: int,
    ) -> None:
        heap = self.example_heaps.setdefault(feature_id, [])
        if len(heap) >= self.top_k_examples and score <= heap[0][0]:
            return
        before, after, full = self._context(row_tokens, position, valid_length)
        token_id = int(row_tokens[position])
        example = {
            "activation_score": float(score),
            "trigger_token": self._decode_token(token_id),
            "trigger_token_id": token_id,
            "position_in_seq": int(position),
            "context_before": before,
            "context_after": after,
            "full_context": full,
            "sequence_id": int(sequence_id),
        }
        self._serial += 1
        item = (float(score), self._serial, example)
        if len(heap) < self.top_k_examples:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    def process_batch(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        sequence_start: int,
    ) -> None:
        if hidden.ndim != 3 or tokens.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("Expected hidden [B,S,D], tokens [B,S], mask [B,S]")
        if hidden.shape[:2] != tokens.shape or tokens.shape != attention_mask.shape:
            raise ValueError("Hidden, token and mask shapes do not match")

        tokens_np = tokens.numpy()
        mask_np = attention_mask.numpy().astype(bool, copy=False)
        valid_rows, valid_positions = np.nonzero(mask_np)
        if len(valid_rows) == 0:
            return
        valid_lengths = mask_np.sum(axis=1).astype(np.int64)
        flat_hidden = hidden[attention_mask].float()
        flat_tokens = tokens_np[valid_rows, valid_positions]
        self.total_tokens += int(flat_tokens.shape[0])

        self.sae.eval()
        for start in range(0, flat_hidden.shape[0], self.microbatch_tokens):
            end = min(start + self.microbatch_tokens, flat_hidden.shape[0])
            with torch.inference_mode():
                encoded = self.sae.encode(flat_hidden[start:end].to(self.device))
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
                encoded = encoded.detach().float()
                active_rows, active_features = torch.nonzero(
                    encoded > self.activation_threshold, as_tuple=True
                )
                active_values = encoded[active_rows, active_features]

            rows_np = active_rows.cpu().numpy()
            features_np = active_features.cpu().numpy()
            values_np = active_values.cpu().numpy()
            for local_row, feature_id, score in zip(rows_np, features_np, values_np):
                flat_index = start + int(local_row)
                row = int(valid_rows[flat_index])
                position = int(valid_positions[flat_index])
                feature_id = int(feature_id)
                score = float(score)
                token_id = int(flat_tokens[flat_index])
                self.counts[feature_id] += 1
                self.sums[feature_id] += score
                self.maxima[feature_id] = max(self.maxima[feature_id], score)
                self._update_bounded_counter(feature_id, token_id)
                self._update_example(
                    feature_id,
                    score,
                    tokens_np[row],
                    position,
                    int(valid_lengths[row]),
                    sequence_start + row,
                )

    def _feature_card(self, feature_id: int, logit_lens=None) -> dict[str, Any]:
        count = int(self.counts[feature_id])
        counter = self.token_counts.get(feature_id, Counter())
        common_tokens = [
            {
                "token_id": int(token_id),
                "token": self._decode_token(int(token_id)),
                "count": int(token_count),
            }
            for token_id, token_count in counter.most_common(self.top_k_tokens)
        ]
        examples = [
            item[2] for item in sorted(
                self.example_heaps.get(feature_id, []),
                key=lambda item: (-item[0], item[1]),
            )
        ]
        card = {
            "feature_id": int(feature_id),
            "observed_in_analysis": bool(count > 0),
            "total_activations": count,
            "activation_frequency": float(count / self.total_tokens) if self.total_tokens else 0.0,
            "mean_activation": float(self.sums[feature_id] / count) if count else 0.0,
            "max_activation": float(self.maxima[feature_id]),
            "common_trigger_tokens": common_tokens,
            "top_examples": examples,
        }
        if logit_lens is not None:
            values = logit_lens.get(feature_id, {})
            card["top_promoted_tokens"] = values.get("promoted", [])
            card["top_suppressed_tokens"] = values.get("suppressed", [])
        return card

    def active_feature_ids(self) -> np.ndarray:
        return np.flatnonzero(self.counts > 0)

    def state_dict(self) -> dict[str, Any]:
        """Return the bounded analysis state needed for checkpoint/resume."""
        return {
            "num_features": int(self.num_features),
            "counts": self.counts,
            "sums": self.sums,
            "maxima": self.maxima,
            "token_counts": {
                int(feature_id): {int(token_id): int(count) for token_id, count in counter.items()}
                for feature_id, counter in self.token_counts.items()
            },
            "example_heaps": self.example_heaps,
            "serial": int(self._serial),
            "total_tokens": int(self.total_tokens),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("num_features", -1)) != self.num_features:
            raise ValueError(
                "Checkpoint SAE width does not match the loaded SAE: "
                f"{state.get('num_features')} != {self.num_features}"
            )
        self.counts = np.asarray(state["counts"], dtype=np.int64).copy()
        self.sums = np.asarray(state["sums"], dtype=np.float64).copy()
        self.maxima = np.asarray(state["maxima"], dtype=np.float32).copy()
        self.token_counts = {
            int(feature_id): Counter({int(token_id): int(count) for token_id, count in values.items()})
            for feature_id, values in state.get("token_counts", {}).items()
        }
        self.example_heaps = {
            int(feature_id): [tuple(item) for item in values]
            for feature_id, values in state.get("example_heaps", {}).items()
        }
        self._serial = int(state.get("serial", 0))
        self.total_tokens = int(state.get("total_tokens", 0))
        self._decoded_tokens.clear()

    def cards(self, include_unobserved: bool = True, logit_lens=None):
        ids = range(self.num_features) if include_unobserved else self.active_feature_ids()
        return [self._feature_card(int(feature_id), logit_lens) for feature_id in ids]


def _compute_logit_lens(
    model,
    sae,
    active_ids: np.ndarray,
    tokenizer,
    device: torch.device,
    top_k: int,
    block_size: int,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    """Compute the same decoder-to-unembedding approximation as the Pythia report."""
    if len(active_ids) == 0:
        return {}
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose output embeddings")
    if top_k < 1 or block_size < 1:
        raise ValueError("top_k and block_size must be positive")

    # Gemma Scope residual decoders live in model hidden space.  This is an
    # approximate logit lens because Gemma's final RMSNorm is nonlinear, but
    # it matches the interpretation produced by the existing Pythia report.
    W_U = output_embeddings.weight.detach().to(device=device, dtype=torch.float32)
    W_dec = sae.W_dec.detach().to(device=device, dtype=torch.float32)
    result: dict[int, dict[str, list[dict[str, Any]]]] = {}
    active_ids = np.asarray(active_ids, dtype=np.int64)
    with torch.inference_mode():
        for offset in range(0, len(active_ids), block_size):
            ids = active_ids[offset:offset + block_size]
            decoder = W_dec[torch.from_numpy(ids).to(device)]
            logits = decoder @ W_U.T
            k = min(top_k, logits.shape[-1])
            values, token_ids = torch.topk(logits, k=k, dim=-1)
            suppressed_values, suppressed_ids = torch.topk(
                logits, k=k, dim=-1, largest=False
            )
            for row, feature_id in enumerate(ids.tolist()):
                result[int(feature_id)] = {
                    "promoted": [
                        {
                            "token_id": int(token_id.item()),
                            "token": tokenizer.decode([int(token_id.item())], skip_special_tokens=False),
                            "logit": float(value.item()),
                        }
                        for value, token_id in zip(values[row], token_ids[row])
                    ],
                    "suppressed": [
                        {
                            "token_id": int(token_id.item()),
                            "token": tokenizer.decode([int(token_id.item())], skip_special_tokens=False),
                            "logit": float(value.item()),
                        }
                        for value, token_id in zip(suppressed_values[row], suppressed_ids[row])
                    ],
                }
            del decoder, logits, values, token_ids, suppressed_values, suppressed_ids
    return result


def _write_text_report(path: Path, summary: dict[str, Any], cards: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("GEMMA SCOPE 2 FEATURE ANALYSIS\n")
        handle.write("=" * 80 + "\n\nSUMMARY\n-------\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
        for card in sorted(cards, key=lambda item: (-item["activation_frequency"], item["feature_id"])):
            if not card["observed_in_analysis"]:
                continue
            handle.write("\n" + "=" * 80 + f"\nFEATURE #{card['feature_id']}\n" + "=" * 80 + "\n")
            handle.write(f"Total activations: {card['total_activations']:,}\n")
            handle.write(f"Activation frequency: {card['activation_frequency']:.6%}\n")
            handle.write(f"Mean activation: {card['mean_activation']:.6f}\n")
            handle.write(f"Max activation: {card['max_activation']:.6f}\n\n")
            handle.write("MOST COMMON TRIGGER TOKENS\n---------------------------\n")
            for token in card["common_trigger_tokens"]:
                handle.write(f"  {token['count']:>8,}  {token['token']!r} (id={token['token_id']})\n")
            if card.get("top_promoted_tokens"):
                handle.write("\nPROMOTED TOKENS (APPROXIMATE LOGIT LENS)\n-----------------------------------------\n")
                for token in card["top_promoted_tokens"]:
                    handle.write(f"  {token['logit']:+.4f}  {token['token']!r} (id={token['token_id']})\n")
            if card.get("top_suppressed_tokens"):
                handle.write("\nSUPPRESSED TOKENS (APPROXIMATE LOGIT LENS)\n-----------------------------------------\n")
                for token in card["top_suppressed_tokens"]:
                    handle.write(f"  {token['logit']:+.4f}  {token['token']!r} (id={token['token_id']})\n")
            handle.write("\nTOP ACTIVATION EXAMPLES\n-----------------------\n")
            for index, example in enumerate(card["top_examples"], 1):
                context = example["full_context"].replace("\n", " ").strip()
                handle.write(
                    f"\n[{index}] activation={example['activation_score']:.6f}, "
                    f"token={example['trigger_token']!r}, sequence={example['sequence_id']}, "
                    f"position={example['position_in_seq']}\n  {context}\n"
                )


def finalize_analysis_checkpoint(
    checkpoint_path: str,
    data_path: str,
    tokenizer_name: str,
    model_name: Optional[str] = None,
    release: Optional[str] = None,
    sae_id: Optional[str] = None,
    dtype: str = "auto",
    compute_logit_lens: bool = False,
    logit_top_k: int = 128,
    logit_block_size: int = 64,
) -> dict[str, Any]:
    """Materialize feature cards from a bounded analysis checkpoint.

    A pre-empted Kaggle run can contain millions of processed tokens and all
    retained contexts even though it never reached the final reporting step.
    Finalization must not pretend that such a checkpoint is complete, and it
    must not require the original token memmaps unless activation streaming is
    resumed.  The optional logit lens still loads the matching model and SAE.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Analysis checkpoint not found: {checkpoint_file}")
    checkpoint = _load_torch_checkpoint(checkpoint_file)
    if checkpoint.get("format") != "gemma_scope_analysis_checkpoint_v1":
        raise ValueError(
            "Unsupported Gemma analysis checkpoint format: "
            f"{checkpoint.get('format')!r}"
        )
    config = checkpoint.get("config", {})
    state = checkpoint.get("accumulator")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain config and accumulator dictionaries")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    num_features = int(state.get("num_features", -1))
    counts = np.asarray(state.get("counts"), dtype=np.int64)
    sums = np.asarray(state.get("sums"), dtype=np.float64)
    maxima = np.asarray(state.get("maxima"), dtype=np.float32)
    if num_features < 1 or any(array.shape != (num_features,) for array in (counts, sums, maxima)):
        raise ValueError("Checkpoint accumulator arrays do not match num_features")
    total_tokens = int(state.get("total_tokens", 0))
    token_counts = state.get("token_counts", {})
    example_heaps = state.get("example_heaps", {})
    top_k_tokens = int(config.get("top_k_tokens", 10))

    promoted: dict[int, dict[str, list[dict[str, Any]]]] = {}
    sae_metadata: dict[str, Any] = {
        "d_in": None,
        "d_sae": num_features,
        "model_name": None,
        "hook_name": None,
        "hf_hook_name": None,
        "normalize_activations": None,
        "apply_b_dec_to_input": None,
    }
    runtime_model_name = model_name or str(config.get("model_name", DEFAULT_MODEL_NAME))
    resolved_release = release or str(config.get("release", DEFAULT_RELEASE))
    resolved_sae_id = sae_id or str(config.get("sae_id", DEFAULT_SAE_ID))
    if compute_logit_lens:
        device = find_device()
        model, _ = _load_model(runtime_model_name, device, dtype)
        sae = _load_sae(resolved_release, resolved_sae_id, device)
        layer_num = int(config.get("layer_num", DEFAULT_LAYER_NUM))
        sae_metadata = _validate_sae_model_contract(sae, model, layer_num)
        if int(sae_metadata["d_sae"]) != num_features:
            raise ValueError(
                f"Checkpoint has {num_features} features, loaded SAE has "
                f"{sae_metadata['d_sae']}"
            )
        promoted = _compute_logit_lens(
            model=model,
            sae=sae,
            active_ids=np.flatnonzero(counts > 0),
            tokenizer=tokenizer,
            device=device,
            top_k=logit_top_k,
            block_size=logit_block_size,
        )

    decoded_tokens: dict[int, str] = {}

    def decode_token(token_id: int) -> str:
        if token_id not in decoded_tokens:
            decoded_tokens[token_id] = tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
        return decoded_tokens[token_id]

    cards: list[dict[str, Any]] = []
    for feature_id in range(num_features):
        count = int(counts[feature_id])
        raw_counter = token_counts.get(feature_id, token_counts.get(str(feature_id), {}))
        counter = Counter({int(token_id): int(value) for token_id, value in raw_counter.items()})
        raw_heap = example_heaps.get(feature_id, example_heaps.get(str(feature_id), []))
        examples = [
            item[2]
            for item in sorted(raw_heap, key=lambda item: (-float(item[0]), int(item[1])))
        ]
        card = {
            "feature_id": feature_id,
            "observed_in_analysis": bool(count > 0),
            "total_activations": count,
            "activation_frequency": float(count / total_tokens) if total_tokens else 0.0,
            "mean_activation": float(sums[feature_id] / count) if count else 0.0,
            "max_activation": float(maxima[feature_id]),
            "common_trigger_tokens": [
                {
                    "token_id": int(token_id),
                    "token": decode_token(int(token_id)),
                    "count": int(token_count),
                }
                for token_id, token_count in counter.most_common(top_k_tokens)
            ],
            "top_examples": examples,
        }
        if compute_logit_lens:
            values = promoted.get(feature_id, {})
            card["top_promoted_tokens"] = values.get("promoted", [])
            card["top_suppressed_tokens"] = values.get("suppressed", [])
        cards.append(card)

    checkpoint_completed = bool(checkpoint.get("completed", False))
    summary = {
        "model_name": str(config.get("model_name", runtime_model_name)),
        "runtime_model_name": runtime_model_name,
        "tokenizer_name": tokenizer_name,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "tokenizer_vocab_sha256": tokenizer_vocab_fingerprint(tokenizer),
        "sae_release": resolved_release,
        "sae_id": resolved_sae_id,
        "site": "resid_post",
        "layer_num": int(config.get("layer_num", DEFAULT_LAYER_NUM)),
        "seq_length": int(config.get("seq_length", DEFAULT_SEQ_LENGTH)),
        "sequence_count_total": None,
        "sequence_count_analyzed": None,
        "last_sequence_cursor": int(checkpoint.get("next_sequence", 0)),
        "processed_batches": int(checkpoint.get("processed_batches", 0)),
        "total_tokens_processed": total_tokens,
        "total_features": num_features,
        "observed_features": int(np.count_nonzero(counts)),
        "unobserved_features": int(num_features - np.count_nonzero(counts)),
        "activation_threshold": float(config.get("activation_threshold", 0.0)),
        "logit_lens": (
            "approximate_resid_decoder_to_unembedding"
            if compute_logit_lens
            else "disabled"
        ),
        "logit_top_k": int(logit_top_k) if compute_logit_lens else 0,
        "analysis_completed": checkpoint_completed,
        "finalized_from_checkpoint": True,
        "finalized_from_partial_checkpoint": not checkpoint_completed,
        "source_checkpoint": str(checkpoint_file),
        "sae_metadata": _json_safe(sae_metadata),
    }
    output_dir = Path(data_path) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = output_dir / "feature_cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as handle:
        for card in cards:
            if card["observed_in_analysis"]:
                handle.write(json.dumps(_json_safe(card), ensure_ascii=False) + "\n")
    json_path = output_dir / "feature_analysis.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe({"summary": summary, "features": cards}),
            handle,
            indent=2,
            ensure_ascii=False,
        )
    text_path = output_dir / "feature_analysis.txt"
    _write_text_report(text_path, summary, cards)
    print(
        "Finalized "
        f"{'complete' if checkpoint_completed else 'partial'} checkpoint into {json_path}"
    )
    return {
        "summary": summary,
        "cards_path": str(cards_path),
        "json_path": str(json_path),
        "text_path": str(text_path),
    }


def analyze(
    model_name: str = DEFAULT_MODEL_NAME,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    data_path: str = "data/gemma_scope2_270m_pilecc",
    input_path: Optional[str] = None,
    tokens_path: Optional[str] = None,
    attention_mask_path: Optional[str] = None,
    require_prepared: bool = False,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    layer_num: int = DEFAULT_LAYER_NUM,
    model_batch_size: int = 2,
    chunk_sequences: int = 8,
    sae_batch_size: int = 256,
    max_sequences: Optional[int] = None,
    sampling: str = "uniform",
    max_documents: Optional[int] = None,
    max_tokens: Optional[int] = None,
    dtype: str = "auto",
    top_k_examples: int = 15,
    top_k_tokens: int = 10,
    max_token_entries: int = 256,
    context_size: int = 25,
    activation_threshold: float = 0.0,
    compute_logit_lens: bool = True,
    logit_top_k: int = 128,
    logit_block_size: int = 64,
    verbose: str = "low",
    verbose_interval: int = 1000,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    checkpoint_every_batches: int = 25,
) -> dict[str, Any]:
    if checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive")
    if min(seq_length, model_batch_size, chunk_sequences, sae_batch_size) < 1:
        raise ValueError("Sequence and batch sizes must be positive")
    if verbose_interval < 1:
        raise ValueError("verbose_interval must be positive")
    if layer_num < 0:
        raise ValueError("layer_num must be non-negative")
    if min(top_k_examples, top_k_tokens, max_token_entries, context_size) < 1:
        raise ValueError("Feature-card limits and context_size must be positive")
    if compute_logit_lens and min(logit_top_k, logit_block_size) < 1:
        raise ValueError("logit_top_k and logit_block_size must be positive")
    for name, value in (
        ("max_sequences", max_sequences),
        ("max_documents", max_documents),
        ("max_tokens", max_tokens),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive when supplied")
    device = find_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    resolved_tokens_path, resolved_mask_path = _prepare_sequences_if_needed(
        Path(data_path), input_path, tokenizer, seq_length,
        max_documents, max_tokens, verbose, verbose_interval,
        tokens_path=tokens_path,
        attention_mask_path=attention_mask_path,
        require_prepared=require_prepared,
    )
    model, loaded_dtype = _load_model(model_name, device, dtype)
    sae = _load_sae(release, sae_id, device)
    sae_metadata = _validate_sae_model_contract(sae, model, layer_num)

    tokens_mm = np.load(resolved_tokens_path, mmap_mode="r")
    masks_mm = np.load(resolved_mask_path, mmap_mode="r")
    if tokens_mm.ndim != 2 or tokens_mm.shape[1] != seq_length:
        raise ValueError(
            f"Expected prepared tokens [N, {seq_length}], got {tokens_mm.shape}"
        )
    if masks_mm.shape != tokens_mm.shape:
        raise ValueError(
            f"Attention mask shape {masks_mm.shape} does not match tokens "
            f"{tokens_mm.shape}"
        )
    sequence_count = int(tokens_mm.shape[0])
    if sequence_count < 1:
        raise ValueError("Prepared token array contains no sequences")
    ranges = _analysis_ranges(sequence_count, max_sequences, chunk_sequences, sampling)
    sampled_sequences = sum(end - start for start, end in ranges)
    print(
        f"Gemma Scope 2 analysis: model={model_name}, release={release}, sae_id={sae_id}\n"
        f"device={device}, dtype={loaded_dtype}, sequences={sampled_sequences:,}/{sequence_count:,}, "
        f"ranges={len(ranges)}"
    )

    accumulator = FeatureAccumulator(
        sae=sae,
        tokenizer=tokenizer,
        device=device,
        microbatch_tokens=sae_batch_size,
        top_k_examples=top_k_examples,
        top_k_tokens=top_k_tokens,
        max_token_entries=max_token_entries,
        context_size=context_size,
        activation_threshold=activation_threshold,
    )
    checkpoint_file = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(data_path) / "analysis" / "analysis_checkpoint.pt"
    )
    checkpoint_config = {
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "release": release,
        "sae_id": sae_id,
        "tokens_path": str(resolved_tokens_path),
        "attention_mask_path": str(resolved_mask_path),
        "seq_length": int(seq_length),
        "layer_num": int(layer_num),
        "model_batch_size": int(model_batch_size),
        "chunk_sequences": int(chunk_sequences),
        "sae_batch_size": int(sae_batch_size),
        "max_sequences": max_sequences,
        "sampling": sampling,
        "top_k_examples": int(top_k_examples),
        "top_k_tokens": int(top_k_tokens),
        "max_token_entries": int(max_token_entries),
        "context_size": int(context_size),
        "activation_threshold": float(activation_threshold),
    }
    if max_sequences is not None and sampling == "uniform":
        checkpoint_config["sampling_layout"] = "disjoint-v1"
    resume_range_index = 0
    resume_sequence = 0
    batch_count = 0
    if resume and checkpoint_file.is_file():
        checkpoint = _load_torch_checkpoint(checkpoint_file)
        if checkpoint.get("format") != "gemma_scope_analysis_checkpoint_v1":
            raise ValueError(
                "Unsupported Gemma analysis checkpoint format: "
                f"{checkpoint.get('format')!r}"
            )
        if not _checkpoint_config_matches(
            checkpoint.get("config", {}), checkpoint_config
        ):
            raise ValueError(
                "Analysis checkpoint configuration does not match the current run. "
                "Use --no-resume or delete analysis_checkpoint.pt to start over."
            )
        accumulator.load_state_dict(checkpoint["accumulator"])
        resume_range_index = int(checkpoint.get("next_range_index", 0))
        resume_sequence = int(checkpoint.get("next_sequence", 0))
        batch_count = int(checkpoint.get("processed_batches", 0))
        if not 0 <= resume_range_index <= len(ranges):
            raise ValueError(
                f"Invalid checkpoint range cursor: {resume_range_index}/{len(ranges)}"
            )
        if resume_sequence < 0:
            raise ValueError(f"Invalid checkpoint sequence cursor: {resume_sequence}")
        print(
            f"Resuming analysis from range {resume_range_index}/{len(ranges)}, "
            f"sequence {resume_sequence:,}; processed batches={batch_count:,}."
        )
    total_batches = sum(
        math.ceil((end - start) / model_batch_size) for start, end in ranges
    )
    progress = (
        tqdm(
            total=total_batches,
            initial=min(batch_count, total_batches),
            desc="Gemma Scope analysis",
            unit="batch",
        )
        if verbose == "high" else None
    )
    for range_index, (range_start, range_end) in enumerate(ranges):
        if range_index < resume_range_index:
            continue
        start_sequence = range_start
        if range_index == resume_range_index:
            start_sequence = max(range_start, resume_sequence)
        for start, end, hidden, tokens, mask in iter_resid_post_batches(
            model=model,
            data_path=data_path,
            seq_length=seq_length,
            layer_num=layer_num,
            model_batch_size=model_batch_size,
            ranges=[(start_sequence, range_end)],
            device=device,
            tokens_path=resolved_tokens_path,
            attention_mask_path=resolved_mask_path,
        ):
            accumulator.process_batch(hidden, tokens, mask, start)
            batch_count += 1
            next_range_index = range_index
            next_sequence = end
            if end >= range_end:
                next_range_index = range_index + 1
                next_sequence = 0
            if progress is not None:
                progress.update(1)
            elif batch_count % verbose_interval == 0 or batch_count == total_batches:
                print(
                    f"Analysis progress: batch {batch_count:,}/{total_batches:,}; "
                    f"observed features={len(accumulator.active_feature_ids()):,}"
                )
            if (
                batch_count % checkpoint_every_batches == 0
                or next_range_index >= len(ranges)
            ):
                _atomic_torch_save(
                    {
                        "format": "gemma_scope_analysis_checkpoint_v1",
                        "config": checkpoint_config,
                        "accumulator": accumulator.state_dict(),
                        "next_range_index": int(next_range_index),
                        "next_sequence": int(next_sequence),
                        "processed_batches": int(batch_count),
                        "completed": False,
                    },
                    checkpoint_file,
                )
    if progress is not None:
        progress.close()

    promoted = {}
    if compute_logit_lens:
        print("Computing approximate decoder-to-unembedding logit lens...")
        promoted = _compute_logit_lens(
            model, sae, accumulator.active_feature_ids(), tokenizer,
            device, logit_top_k, logit_block_size,
        )
    cards = accumulator.cards(include_unobserved=True, logit_lens=promoted)
    active_count = int(len(accumulator.active_feature_ids()))
    summary = {
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "tokenizer_vocab_sha256": tokenizer_vocab_fingerprint(tokenizer),
        "sae_release": release,
        "sae_id": sae_id,
        "site": "resid_post",
        "layer_num": int(layer_num),
        "seq_length": int(seq_length),
        "sequence_count_total": sequence_count,
        "sequence_count_analyzed": int(sampled_sequences),
        "total_tokens_processed": int(accumulator.total_tokens),
        "total_features": int(accumulator.num_features),
        "observed_features": active_count,
        "unobserved_features": int(accumulator.num_features - active_count),
        "activation_threshold": float(activation_threshold),
        "logit_lens": "approximate_resid_decoder_to_unembedding" if compute_logit_lens else "disabled",
        "logit_top_k": int(logit_top_k) if compute_logit_lens else 0,
        "sae_metadata": _json_safe(sae_metadata),
    }
    output_dir = Path(data_path) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = output_dir / "feature_cards.jsonl"
    with cards_path.open("w", encoding="utf-8") as handle:
        for card in cards:
            if card["observed_in_analysis"]:
                handle.write(json.dumps(_json_safe(card), ensure_ascii=False) + "\n")
    json_path = output_dir / "feature_analysis.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe({"summary": summary, "features": cards}), handle, indent=2, ensure_ascii=False)
    text_path = output_dir / "feature_analysis.txt"
    _write_text_report(text_path, summary, cards)
    _atomic_torch_save(
        {
            "format": "gemma_scope_analysis_checkpoint_v1",
            "config": checkpoint_config,
            "accumulator": accumulator.state_dict(),
            "next_range_index": int(len(ranges)),
            "next_sequence": 0,
            "processed_batches": int(batch_count),
            "completed": True,
            "summary": summary,
        },
        checkpoint_file,
    )
    print(f"Feature cards saved to: {cards_path}")
    print(f"JSON analysis saved to: {json_path}")
    print(f"Text analysis saved to: {text_path}")
    return {"summary": summary, "cards_path": str(cards_path), "json_path": str(json_path), "text_path": str(text_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a pretrained Gemma Scope 2 resid-post SAE.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    parser.add_argument("--data-path", default="data/gemma_scope2_270m_pilecc")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--tokens-path", default=None)
    parser.add_argument("--attention-mask-path", default=None)
    parser.add_argument(
        "--require-prepared-sequences",
        action="store_true",
        help="Fail if the supplied prepared sequence files are missing; never tokenize JSONL.",
    )
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--layer-num", type=int, default=DEFAULT_LAYER_NUM)
    parser.add_argument("--model-batch-size", type=int, default=2)
    parser.add_argument("--chunk-sequences", type=int, default=8)
    parser.add_argument("--sae-batch-size", type=int, default=256)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sampling", choices=["head", "tail", "uniform"], default="uniform")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--top-k-examples", type=int, default=15)
    parser.add_argument("--top-k-tokens", type=int, default=10)
    parser.add_argument("--max-token-entries", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=25)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--skip-logit-lens", action="store_true")
    parser.add_argument(
        "--logit-top-k",
        type=int,
        default=128,
        help="Keep enough decoder-to-vocabulary tokens for downstream semantic triage.",
    )
    parser.add_argument("--logit-block-size", type=int, default=64)
    parser.add_argument("--verbose", choices=["low", "high"], default="low")
    parser.add_argument("--verbose-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--finalize-checkpoint-only",
        action="store_true",
        help=(
            "Write feature reports from an existing checkpoint without requiring "
            "the original token memmaps or resuming activation collection."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every-batches", type=int, default=25)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.finalize_checkpoint_only:
        checkpoint_path = args.checkpoint_path or str(
            Path(args.data_path) / "analysis" / "analysis_checkpoint.pt"
        )
        finalize_analysis_checkpoint(
            checkpoint_path=checkpoint_path,
            data_path=args.data_path,
            tokenizer_name=args.tokenizer_name,
            model_name=args.model_name,
            release=args.release,
            sae_id=args.sae_id,
            dtype=args.dtype,
            compute_logit_lens=not args.skip_logit_lens,
            logit_top_k=args.logit_top_k,
            logit_block_size=args.logit_block_size,
        )
        return
    analyze(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        release=args.release,
        sae_id=args.sae_id,
        data_path=args.data_path,
        input_path=args.input_path,
        tokens_path=args.tokens_path,
        attention_mask_path=args.attention_mask_path,
        require_prepared=args.require_prepared_sequences,
        seq_length=args.seq_length,
        layer_num=args.layer_num,
        model_batch_size=args.model_batch_size,
        chunk_sequences=args.chunk_sequences,
        sae_batch_size=args.sae_batch_size,
        max_sequences=args.max_sequences,
        sampling=args.sampling,
        max_documents=args.max_documents,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        top_k_examples=args.top_k_examples,
        top_k_tokens=args.top_k_tokens,
        max_token_entries=args.max_token_entries,
        context_size=args.context_size,
        activation_threshold=args.activation_threshold,
        compute_logit_lens=not args.skip_logit_lens,
        logit_top_k=args.logit_top_k,
        logit_block_size=args.logit_block_size,
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
        checkpoint_path=args.checkpoint_path,
        resume=not args.no_resume,
        checkpoint_every_batches=args.checkpoint_every_batches,
    )


if __name__ == "__main__":
    main()
