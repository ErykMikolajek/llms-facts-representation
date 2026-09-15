from __future__ import annotations  # TODO: refactor the code not to use this

import os
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import heapq
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_pipeline.autoencoder_training import TopKSAE
from common.utils import find_device


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _analysis_checkpoint_config_matches(loaded: Dict, expected: Dict) -> bool:
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


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _save_features_to_json(
    analyzer: "ComprehensiveFeatureAnalyzer",
    output_path: str,
    include_dead_features: bool = True,
):
    payload = {
        "summary": _json_safe(analyzer.get_summary_stats()),
        "features": [
            _json_safe(asdict(analysis))
            for analysis in sorted(
                analyzer.feature_analyses.values(),
                key=lambda x: x.feature_id,
            )
            if include_dead_features or not analysis.is_dead
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Structured analysis saved to: {output_path}")
    return output_path


def _save_features_to_text(
    analyzer: "ComprehensiveFeatureAnalyzer",
    output_path: str,
    include_dead_features: bool = False,
    max_examples_per_feature: int = 10
):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SAE FEATURES ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        summary = analyzer.get_summary_stats()
        f.write("SUMMARY\n")
        f.write("-"*40 + "\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.6f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        sorted_features = sorted(
            analyzer.feature_analyses.values(),
            key=lambda x: x.activation_frequency,
            reverse=True
        )
        
        for analysis in sorted_features:
            if analysis.is_dead and not include_dead_features:
                continue
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"FEATURE #{analysis.feature_id}\n")
            f.write("="*80 + "\n\n")
            
            if analysis.is_dead:
                f.write("STATUS: DEAD DURING TRAINING\n\n")
                continue

            if not analysis.observed_in_analysis:
                f.write("STATUS: NOT OBSERVED IN ANALYSIS SAMPLE\n\n")
            
            f.write("ACTIVATION STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"  Total activations:      {analysis.total_activations:,}\n")
            f.write(f"  Activation frequency:    {analysis.activation_frequency:.4%}\n")
            f.write(f"  Mean activation:     {analysis.mean_activation:.4f}\n")
            f.write(f"  Max activation:  {analysis.max_activation:.4f}\n\n")
            
            if analysis.common_trigger_tokens:
                f.write("MOST COMMON TRIGGER TOKENS\n")
                f.write("-"*40 + "\n")
                for i, (token, count) in enumerate(analysis.common_trigger_tokens[:8], 1):
                    token_display = repr(token) if token.strip() != token or not token else f"'{token}'"
                    f.write(f"  {i}. {token_display:20} (occurrences: {count})\n")
                f.write("\n")
            
            if analysis.top_promoted_tokens:
                f.write("PROMOTED TOKENS (Logit Lens)\n")
                f.write("-"*40 + "\n")
                f.write("  (Tokens that this feature wants to generate)\n")
                for i, (token, score) in enumerate(analysis.top_promoted_tokens[:8], 1):
                    token_display = repr(token) if token.strip() != token or not token else f"'{token}'"
                    f.write(f"  {i}. {token_display:20} (logit: {score:+.3f})\n")
                f.write("\n")
            
            if analysis.top_examples:
                f.write("TOP EXAMPLES OF ACTIVATIONS (from strongest to weakest)\n")
                f.write("-"*40 + "\n")
                
                for i, ex in enumerate(analysis.top_examples[:max_examples_per_feature], 1):
                    f.write(f"\n  [{i}] Activation: {ex.activation_score:.4f}\n")
                    f.write(f"      Trigger token: {repr(ex.trigger_token)}\n")
                    f.write(f"      Sequence position: {ex.position_in_seq}\n")
                    f.write(f"      Sequence ID: {ex.sequence_id}\n")
                    f.write(f"      Context:\n")
                    
                    context_lines = ex.full_context.replace('\n', ' ').strip()
                    max_line_len = 70
                    for j in range(0, len(context_lines), max_line_len):
                        f.write(f"        {context_lines[j:j+max_line_len]}\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Analysis saved to: {output_path}")
    return output_path


@dataclass
class FeatureExample:
    activation_score: float = 0.0
    trigger_token: str = ""
    trigger_token_id: int = 0
    position_in_seq: int = 0
    context_before: str = ""
    context_after: str = ""
    full_context: str = ""
    sequence_id: int = 0

@dataclass
class FeatureAnalysis:
    feature_id: int
    
    # activation statistics
    total_activations: int = 0
    mean_activation: float = 0.0
    max_activation: float = 0.0
    activation_frequency: float = 0.0
    
    top_examples: List[FeatureExample] = field(default_factory=list)
    
    top_promoted_tokens: List[Tuple[str, float]] = field(default_factory=list)
    top_suppressed_tokens: List[Tuple[str, float]] = field(default_factory=list)
    
    common_trigger_tokens: List[Tuple[str, int]] = field(default_factory=list)

    # A feature may be alive during training but absent from a sampled
    # analysis slice. Keep these scopes separate from the legacy is_dead flag.
    observed_in_analysis: bool = False
    observed_in_training: Optional[bool] = None
    
    is_dead: bool = False
    
    interpretation: str = ""


class ComprehensiveFeatureAnalyzer:
    def __init__(
        self,
        sae_model,
        llm_model,
        tokenizer,
        device,
        context_size: int = 30,
        top_k_examples: int = 20,
        activation_threshold: float = 0.05,
        top_k_logits: int = 15,
        training_usage_counts: Optional[np.ndarray] = None,
    ):
        self.sae = sae_model
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.device = device
        self.context_size = context_size
        self.top_k_examples = top_k_examples
        self.activation_threshold = activation_threshold
        self.top_k_logits = top_k_logits
        self.num_features = sae_model.d_sae

        self.training_usage_counts = None
        if training_usage_counts is not None:
            counts = np.asarray(training_usage_counts).reshape(-1)
            if counts.size != self.num_features:
                raise ValueError(
                    "training_usage_counts length does not match SAE feature count: "
                    f"{counts.size} != {self.num_features}"
                )
            self.training_usage_counts = counts
        
        self.feature_analyses: Dict[int, FeatureAnalysis] = {
            i: FeatureAnalysis(feature_id=i) for i in range(self.num_features)
        }
        
        self._activation_sums = defaultdict(float)
        self._activation_counts = defaultdict(int)
        self._trigger_token_counts = defaultdict(lambda: defaultdict(int))
        self._total_tokens_processed = 0
        
        self._example_heaps: Dict[int, list] = {i: [] for i in range(self.num_features)}
        self._example_serial = 0

    def state_dict(self) -> Dict:
        """Return all bounded statistics needed to resume analysis."""
        return {
            "num_features": int(self.num_features),
            "feature_analyses": {
                int(feature_id): asdict(analysis)
                for feature_id, analysis in self.feature_analyses.items()
            },
            "activation_sums": dict(self._activation_sums),
            "activation_counts": dict(self._activation_counts),
            "trigger_token_counts": {
                int(feature_id): dict(token_counts)
                for feature_id, token_counts in self._trigger_token_counts.items()
            },
            "total_tokens_processed": int(self._total_tokens_processed),
            "example_serial": int(self._example_serial),
            "example_heaps": {
                int(feature_id): [
                    (float(score), int(serial), asdict(example))
                    for score, serial, example in heap
                ]
                for feature_id, heap in self._example_heaps.items()
                if heap
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state.get("num_features", -1)) != self.num_features:
            raise ValueError(
                "Checkpoint SAE width does not match the loaded SAE: "
                f"{state.get('num_features')} != {self.num_features}"
            )
        self.feature_analyses = {}
        for feature_id, payload in state.get("feature_analyses", {}).items():
            payload = dict(payload)
            payload["top_examples"] = [
                FeatureExample(**example) if isinstance(example, dict) else example
                for example in payload.get("top_examples", [])
            ]
            payload["top_promoted_tokens"] = [tuple(item) for item in payload.get("top_promoted_tokens", [])]
            payload["top_suppressed_tokens"] = [tuple(item) for item in payload.get("top_suppressed_tokens", [])]
            payload["common_trigger_tokens"] = [tuple(item) for item in payload.get("common_trigger_tokens", [])]
            self.feature_analyses[int(feature_id)] = FeatureAnalysis(**payload)
        for feature_id in range(self.num_features):
            self.feature_analyses.setdefault(feature_id, FeatureAnalysis(feature_id=feature_id))
        self._activation_sums = defaultdict(float, {
            int(feature_id): float(value)
            for feature_id, value in state.get("activation_sums", {}).items()
        })
        self._activation_counts = defaultdict(int, {
            int(feature_id): int(value)
            for feature_id, value in state.get("activation_counts", {}).items()
        })
        self._trigger_token_counts = defaultdict(lambda: defaultdict(int))
        for feature_id, token_counts in state.get("trigger_token_counts", {}).items():
            self._trigger_token_counts[int(feature_id)] = defaultdict(
                int,
                {str(token): int(count) for token, count in token_counts.items()},
            )
        self._total_tokens_processed = int(state.get("total_tokens_processed", 0))
        self._example_heaps = {i: [] for i in range(self.num_features)}
        self._example_serial = int(state.get("example_serial", 0))
        for feature_id, heap in state.get("example_heaps", {}).items():
            self._example_heaps[int(feature_id)] = [
                (
                    float(score),
                    int(serial),
                    FeatureExample(**example) if isinstance(example, dict) else example,
                )
                for score, serial, example in heap
            ]
            if self._example_heaps[int(feature_id)]:
                self._example_serial = max(
                    self._example_serial,
                    max(int(item[1]) for item in self._example_heaps[int(feature_id)]),
                )

    def migrate_flattened_context_examples(
        self,
        tokens: np.ndarray,
        masks: Optional[np.ndarray],
        chunk_sequences: int,
    ) -> int:
        """Repair context coordinates written by the original streaming code.

        Legacy checkpoints treated all valid tokens in a multi-sequence chunk
        as one pseudo-sequence. Activation statistics were correct, but
        ``sequence_id``, ``position_in_seq`` and decoded contexts crossed real
        sequence boundaries. The legacy pair (chunk start, flat token index)
        contains enough information to reconstruct the correct coordinates.
        """
        chunk_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def remap(example: FeatureExample) -> None:
            chunk_start = int(example.sequence_id)
            flat_position = int(example.position_in_seq)
            if chunk_start not in chunk_cache:
                chunk_end = min(chunk_start + chunk_sequences, int(tokens.shape[0]))
                token_chunk = np.asarray(tokens[chunk_start:chunk_end])
                mask_chunk = (
                    np.asarray(masks[chunk_start:chunk_end]).astype(bool)
                    if masks is not None
                    else token_chunk != 0
                )
                coordinates = np.argwhere(mask_chunk)
                chunk_cache[chunk_start] = (token_chunk, mask_chunk, coordinates)
            token_chunk, mask_chunk, coordinates = chunk_cache[chunk_start]
            if not 0 <= flat_position < len(coordinates):
                raise ValueError(
                    "Legacy context position is outside its token chunk: "
                    f"chunk={chunk_start}, position={flat_position}, "
                    f"valid_tokens={len(coordinates)}"
                )
            row_index, original_position = map(int, coordinates[flat_position])
            valid_row = mask_chunk[row_index]
            context_tokens = torch.from_numpy(
                np.array(token_chunk[row_index][valid_row], dtype=np.int64, copy=True)
            )
            context_position = int(valid_row[:original_position].sum())
            context_before, context_after, full_context = self._get_context(
                context_tokens,
                context_position,
                int(context_tokens.numel()),
            )
            trigger_token_id = int(token_chunk[row_index, original_position])
            example.trigger_token_id = trigger_token_id
            example.trigger_token = self.tokenizer.decode([trigger_token_id])
            example.position_in_seq = original_position
            example.sequence_id = chunk_start + row_index
            example.context_before = context_before
            example.context_after = context_after
            example.full_context = full_context

        migrated = 0
        for heap in self._example_heaps.values():
            for _score, _serial, example in heap:
                remap(example)
                migrated += 1
        for analysis in self.feature_analyses.values():
            for example in analysis.top_examples:
                remap(example)
                migrated += 1
        return migrated
    
    def _get_context(
        self, 
        tokens: torch.Tensor, 
        position: int, 
        seq_len: int
    ) -> Tuple[str, str, str]:
        """Extracts context before, after and full context with the highlighted token."""
        
        start_before = max(0, position - self.context_size)
        end_after = min(seq_len, position + self.context_size + 1)
        
        tokens_before = tokens[start_before:position]
        trigger_token = tokens[position:position+1]
        tokens_after = tokens[position+1:end_after]
        
        context_before = self.tokenizer.decode(tokens_before, skip_special_tokens=False)
        trigger_str = self.tokenizer.decode(trigger_token, skip_special_tokens=False)
        context_after = self.tokenizer.decode(tokens_after, skip_special_tokens=False)
        
        full_context = f"{context_before}[[{trigger_str}]]{context_after}"
        
        return context_before, context_after, full_context
    
    def process_batch(
        self,
        activations: torch.Tensor,
        tokens: torch.Tensor,
        batch_start_idx: int,
        sequence_tokens: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None,
    ):
        """Processes a batch of activations and collects statistics."""

        self.sae.eval()
        streaming_coordinates = None
        source_sequence_tokens = None
        source_sequence_mask = None
        if activations.ndim == 2:
            # Activations remain flattened for bounded SAE inference, while the
            # optional source tensors preserve true sequence boundaries for
            # examples and coordinates.
            if (sequence_tokens is None) != (sequence_mask is None):
                raise ValueError(
                    "sequence_tokens and sequence_mask must be supplied together"
                )
            if sequence_tokens is not None and sequence_mask is not None:
                source_sequence_tokens = sequence_tokens.cpu()
                source_sequence_mask = sequence_mask.cpu().bool()
                if source_sequence_tokens.ndim != 2:
                    raise ValueError("sequence_tokens must have shape [sequences, length]")
                if source_sequence_mask.shape != source_sequence_tokens.shape:
                    raise ValueError(
                        "sequence_mask shape must match sequence_tokens shape"
                    )
                streaming_coordinates = source_sequence_mask.nonzero(as_tuple=False)
                if streaming_coordinates.shape[0] != activations.shape[0]:
                    raise ValueError(
                        "Number of valid source tokens does not match flattened "
                        f"activations: {streaming_coordinates.shape[0]} != "
                        f"{activations.shape[0]}"
                    )
            activations = activations.unsqueeze(0)
            tokens = tokens.reshape(1, -1)
        batch_size, seq_len, _ = activations.shape
        
        with torch.no_grad():
            activations = activations.to(self.device).float()
            
            _, sparse_acts = self.sae(activations)  # [batch, seq, d_sae]
            sparse_acts = sparse_acts.cpu()
            tokens = tokens.cpu()
        
        for feat_idx in range(self.num_features):
            feat_acts = sparse_acts[..., feat_idx]  # [batch, seq]
            
            active_mask = feat_acts > self.activation_threshold
            
            if not active_mask.any():
                continue
            
            active_values = feat_acts[active_mask]
            self._activation_sums[feat_idx] += active_values.sum().item()
            self._activation_counts[feat_idx] += active_values.numel()
            
            current_max = active_values.max().item()
            if current_max > self.feature_analyses[feat_idx].max_activation:
                self.feature_analyses[feat_idx].max_activation = current_max
            
            # Iterate only over activations that passed the threshold. Scanning
            # every token position separately for every SAE feature made the
            # full-corpus path unnecessarily expensive.
            for active_position in active_mask.nonzero(as_tuple=False):
                batch_i, seq_i = map(int, active_position.tolist())
                score = feat_acts[batch_i, seq_i].item()

                trigger_token_id = tokens[batch_i, seq_i].item()
                trigger_token_str = self.tokenizer.decode([trigger_token_id])
                self._trigger_token_counts[feat_idx][trigger_token_str] += 1

                heap = self._example_heaps[feat_idx]

                if streaming_coordinates is not None:
                    source_row, source_position = map(
                        int, streaming_coordinates[seq_i].tolist()
                    )
                    assert source_sequence_tokens is not None
                    assert source_sequence_mask is not None
                    valid_source_row = source_sequence_mask[source_row]
                    context_tokens = source_sequence_tokens[source_row][valid_source_row]
                    context_position = int(
                        valid_source_row[:source_position].sum().item()
                    )
                    example_position = source_position
                    example_sequence_id = batch_start_idx + source_row
                else:
                    context_tokens = tokens[batch_i]
                    context_position = seq_i
                    example_position = seq_i
                    example_sequence_id = batch_start_idx + batch_i

                if len(heap) < self.top_k_examples:
                    context_before, context_after, full_context = self._get_context(
                        context_tokens,
                        context_position,
                        int(context_tokens.numel()),
                    )
                    example = FeatureExample(
                        activation_score=score,
                        trigger_token=trigger_token_str,
                        trigger_token_id=trigger_token_id,
                        position_in_seq=example_position,
                        context_before=context_before,
                        context_after=context_after,
                        full_context=full_context,
                        sequence_id=example_sequence_id,
                    )
                    self._example_serial += 1
                    heapq.heappush(heap, (score, self._example_serial, example))

                elif score > heap[0][0]:
                    context_before, context_after, full_context = self._get_context(
                        context_tokens,
                        context_position,
                        int(context_tokens.numel()),
                    )
                    example = FeatureExample(
                        activation_score=score,
                        trigger_token=trigger_token_str,
                        trigger_token_id=trigger_token_id,
                        position_in_seq=example_position,
                        context_before=context_before,
                        context_after=context_after,
                        full_context=full_context,
                        sequence_id=example_sequence_id,
                    )
                    self._example_serial += 1
                    heapq.heapreplace(heap, (score, self._example_serial, example))
        
        self._total_tokens_processed += batch_size * seq_len
    
    def compute_logit_lens(self, logit_batch_size: int = 256):
        """Computes logit lens for all features - which tokens are promoted."""
        
        print("Calculating logit lens for all features...")
        
        output_embeddings = self.llm.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("Model does not expose output embeddings for logit lens")
        # The Pythia checkpoint may load the unembedding matrix in float16,
        # while the SAE checkpoint is normally float32. Matmul requires both
        # operands to have the same dtype; use float32 for stable logit lens.
        W_U = output_embeddings.weight.detach().to(
            self.device, dtype=torch.float32
        )  # [vocab_size, d_model]

        W_dec = self.sae.W_dec.detach().to(
            self.device, dtype=torch.float32
        )  # [d_sae, d_model]
        
        if logit_batch_size < 1:
            raise ValueError("logit_batch_size must be positive")

        # Compute feature logits in blocks. A full Pythia SAE x vocabulary
        # matrix can easily occupy multiple gigabytes.
        with torch.no_grad():
            for start in range(0, self.num_features, logit_batch_size):
                end = min(start + logit_batch_size, self.num_features)
                feature_logits = W_dec[start:end] @ W_U.T
                for local_idx, logits in enumerate(feature_logits):
                    feat_idx = start + local_idx
                    top_vals, top_ids = torch.topk(logits, self.top_k_logits)
                    self.feature_analyses[feat_idx].top_promoted_tokens = [
                        (self.tokenizer.decode([tid.item()]), val.item())
                        for tid, val in zip(top_ids, top_vals)
                    ]

                    bottom_vals, bottom_ids = torch.topk(
                        logits, self.top_k_logits, largest=False
                    )
                    self.feature_analyses[feat_idx].top_suppressed_tokens = [
                        (self.tokenizer.decode([tid.item()]), val.item())
                        for tid, val in zip(bottom_ids, bottom_vals)
                    ]
        
    
    def finalize_analysis(self):
        """Finalizes the analysis - calculates final statistics."""
        
        print("Finalizing analysis...")
        
        for feat_idx in range(self.num_features):
            analysis = self.feature_analyses[feat_idx]
            analysis.observed_in_analysis = self._activation_counts[feat_idx] > 0
            
            if self._activation_counts[feat_idx] > 0:
                analysis.total_activations = self._activation_counts[feat_idx]
                analysis.mean_activation = (
                    self._activation_sums[feat_idx] / self._activation_counts[feat_idx]
                )
                analysis.activation_frequency = (
                    self._activation_counts[feat_idx] / self._total_tokens_processed
                )
            else:
                # Absence from this analysis sample is not enough to call a
                # feature dead. Training usage_counts, when available below,
                # is the authoritative scope for that label.
                analysis.is_dead = False

            if self.training_usage_counts is not None:
                analysis.observed_in_training = bool(
                    self.training_usage_counts[feat_idx] > 0
                )
                analysis.is_dead = not analysis.observed_in_training
            
            heap = self._example_heaps[feat_idx]
            examples = [item[2] for item in sorted(heap, key=lambda x: -x[0])]
            analysis.top_examples = examples
            
            trigger_counts = self._trigger_token_counts[feat_idx]
            common_triggers = sorted(
                trigger_counts.items(), 
                key=lambda x: -x[1]
            )[:10]
            analysis.common_trigger_tokens = common_triggers
        
        dead_count = sum(1 for a in self.feature_analyses.values() if a.is_dead)
        alive_count = self.num_features - dead_count
        observed_count = sum(
            1 for a in self.feature_analyses.values()
            if a.observed_in_analysis
        )
        
        print(f"Analysis completed.")
        print(f"  - Total features: {self.num_features}")
        print(f"  - Training-active features: {alive_count}")
        print(f"  - Training-dead features: {dead_count}")
        print(f"  - Observed in analysis sample: {observed_count}")
        print(f"  - Not observed in analysis sample: {self.num_features - observed_count}")
        print(f"  - Processed tokens: {self._total_tokens_processed:,}")
    
    def get_summary_stats(self) -> Dict:
        """Returns summary statistics."""
        
        active_features = [a for a in self.feature_analyses.values() if not a.is_dead]
        observed_features = [
            a for a in self.feature_analyses.values()
            if a.observed_in_analysis
        ]
        
        if not active_features:
            return {"error": "No active features"}
        
        return {
            "total_features": self.num_features,
            "active_features": len(active_features),
            "dead_features": self.num_features - len(active_features),
            "training_active_features": len(active_features),
            "training_dead_features": self.num_features - len(active_features),
            "observed_in_analysis_features": len(observed_features),
            "unobserved_in_analysis_features": self.num_features - len(observed_features),
            "mean_activation_frequency": np.mean([a.activation_frequency for a in active_features]),
            "max_activation_frequency": max(a.activation_frequency for a in active_features),
            "min_activation_frequency": min(a.activation_frequency for a in active_features),
            "total_tokens_processed": self._total_tokens_processed
        }


def _analysis_ranges(
    sequence_count: int,
    max_sequences: Optional[int],
    chunk_sequences: int,
    sampling: str,
) -> List[Tuple[int, int]]:
    """Build contiguous windows for head, tail or stratified uniform sampling."""
    if max_sequences is None or max_sequences >= sequence_count:
        return [(0, sequence_count)]
    if max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    if sampling == "head":
        return [(0, max_sequences)]
    if sampling == "tail":
        return [(sequence_count - max_sequences, sequence_count)]
    if sampling != "uniform":
        raise ValueError(f"Unknown analysis sampling mode: {sampling}")

    window_count = (max_sequences + chunk_sequences - 1) // chunk_sequences
    sizes = [chunk_sequences] * (window_count - 1)
    sizes.append(max_sequences - sum(sizes))
    if window_count == 1:
        start = (sequence_count - max_sequences) // 2
        return [(start, start + max_sequences)]

    # Distribute all unsampled sequences as gaps between windows. This yields
    # exactly max_sequences distinct rows; the previous linspace-based
    # implementation could create overlapping windows when the requested
    # sample was close to the full dataset size.
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


def analyze_sae(
    sae: TopKSAE,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    path_dir: str,
    layer_num: int,
    top_k: int = 10,
    context_size: int = 25,
    top_k_examples: int = 15,
    activation_threshold: float = 0.03,
    batch_size: int = 4,
    chunk_sequences: int = 32,
    logit_batch_size: int = 256,
    max_sequences: Optional[int] = None,
    sampling: str = "head",
    training_usage_counts: Optional[np.ndarray] = None,
    verbose: str = "high",
    verbose_interval: int = 1000,
    tokens_path: str | os.PathLike[str] | None = None,
    attention_mask_path: str | os.PathLike[str] | None = None,
    analysis_checkpoint_path: str | os.PathLike[str] | None = None,
    resume: bool = True,
    checkpoint_every_chunks: int = 10,
):
    if verbose not in {"low", "high"}:
        raise ValueError("verbose must be 'low' or 'high'")
    if verbose_interval < 1:
        raise ValueError("verbose_interval must be positive")
    if checkpoint_every_chunks < 1:
        raise ValueError("checkpoint_every_chunks must be positive")
    device = find_device()
    sae.to(device)
    sae.eval()

    analyzer = ComprehensiveFeatureAnalyzer(
        sae_model=sae,
        llm_model=model,
        tokenizer=tokenizer,
        device=device,
        context_size=context_size,
        top_k_examples=top_k_examples,
        activation_threshold=activation_threshold,
        top_k_logits=top_k,
        training_usage_counts=training_usage_counts,
    )

    print(f"Number of features to analyze: {analyzer.num_features}")
    print("Starting analysis of activations...")

    tokens_path = (
        Path(tokens_path)
        if tokens_path is not None
        else Path(path_dir) / "sequenced/tokens_seqs_padded.npy"
    )
    mask_path = (
        Path(attention_mask_path)
        if attention_mask_path is not None
        else Path(path_dir) / "sequenced/attention_mask.npy"
    )
    tokens_mm = np.load(tokens_path, mmap_mode="r")
    masks_mm = np.load(mask_path, mmap_mode="r") if mask_path.exists() else None
    if tokens_mm.ndim != 2:
        raise ValueError(f"Expected token array [N, S], got {tokens_mm.shape}")
    if masks_mm is not None and masks_mm.shape != tokens_mm.shape:
        raise ValueError(
            f"Attention mask shape {masks_mm.shape} does not match tokens "
            f"{tokens_mm.shape}"
        )
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    sequence_count = tokens_mm.shape[0]
    ranges = _analysis_ranges(
        sequence_count=sequence_count,
        max_sequences=max_sequences,
        chunk_sequences=chunk_sequences,
        sampling=sampling,
    )
    sampled_sequences = sum(end - start for start, end in ranges)
    print(
        f"Sequences to analyze: {sampled_sequences:,}/{sequence_count:,}; "
        f"sampling={sampling}"
    )

    checkpoint_file = (
        Path(analysis_checkpoint_path)
        if analysis_checkpoint_path is not None
        else Path(path_dir) / "analysis" / "analysis_checkpoint.pt"
    )
    checkpoint_config = {
        "tokens_path": str(tokens_path),
        "attention_mask_path": str(mask_path),
        "layer_num": int(layer_num),
        "seq_length": int(tokens_mm.shape[1]),
        "batch_size": int(batch_size),
        "chunk_sequences": int(chunk_sequences),
        "top_k": int(top_k),
        "context_size": int(context_size),
        "top_k_examples": int(top_k_examples),
        "activation_threshold": float(activation_threshold),
        "max_sequences": max_sequences,
        "sampling": sampling,
        "num_features": int(analyzer.num_features),
        "context_mapping": "sequence-aware-v1",
    }
    if max_sequences is not None and sampling == "uniform":
        checkpoint_config["sampling_layout"] = "disjoint-v1"
    resume_range_index = 0
    resume_sequence = 0
    chunk_count = 0
    if resume and checkpoint_file.is_file():
        checkpoint = _load_torch_checkpoint(checkpoint_file)
        if checkpoint.get("format") != "pythia_feature_analysis_checkpoint_v1":
            raise ValueError(
                "Unsupported analysis checkpoint format: "
                f"{checkpoint.get('format')!r}"
            )
        loaded_config = checkpoint.get("config", {})
        legacy_config = dict(checkpoint_config)
        legacy_config.pop("context_mapping")
        is_legacy_context_checkpoint = _analysis_checkpoint_config_matches(
            loaded_config, legacy_config
        )
        if not _analysis_checkpoint_config_matches(
            loaded_config, checkpoint_config
        ) and not is_legacy_context_checkpoint:
            raise ValueError(
                "Analysis checkpoint configuration does not match the current run. "
                "Use --no-analysis-resume or delete analysis_checkpoint.pt to start over."
            )
        analyzer.load_state_dict(checkpoint["analyzer"])
        if is_legacy_context_checkpoint:
            migrated = analyzer.migrate_flattened_context_examples(
                tokens=tokens_mm,
                masks=masks_mm,
                chunk_sequences=chunk_sequences,
            )
            print(
                f"Migrated {migrated:,} legacy context examples to true "
                "sequence coordinates."
            )
        resume_range_index = int(checkpoint.get("next_range_index", 0))
        resume_sequence = int(checkpoint.get("next_sequence", 0))
        chunk_count = int(checkpoint.get("processed_chunks", 0))
        if not 0 <= resume_range_index <= len(ranges):
            raise ValueError(
                f"Invalid checkpoint range cursor: {resume_range_index}/{len(ranges)}"
            )
        print(
            f"Resuming analysis from range {resume_range_index}/{len(ranges)}, "
            f"sequence {resume_sequence:,}; processed chunks={chunk_count:,}."
        )

    from sae_pipeline.activations_collecting import iter_activation_chunks

    total_chunks = sum(
        (end - start + chunk_sequences - 1) // chunk_sequences
        for start, end in ranges
    )
    if resume and checkpoint_file.is_file():
        cursor_chunk_count = sum(
            (end - start + chunk_sequences - 1) // chunk_sequences
            for start, end in ranges[:resume_range_index]
        )
        if resume_range_index < len(ranges):
            current_start, current_end = ranges[resume_range_index]
            cursor_sequence = min(max(resume_sequence, current_start), current_end)
            cursor_chunk_count += (
                cursor_sequence - current_start + chunk_sequences - 1
            ) // chunk_sequences
        if chunk_count != cursor_chunk_count:
            print(
                "Corrected legacy processed_chunks from "
                f"{chunk_count:,} to {cursor_chunk_count:,} using the resume cursor."
            )
            chunk_count = cursor_chunk_count
    progress = None
    if verbose == "high":
        progress = tqdm(
            total=total_chunks,
            initial=min(chunk_count, total_chunks),
            desc="Analysis of streaming activations",
        )
    try:
        for range_index, (range_start, range_end) in enumerate(ranges):
            if range_index < resume_range_index:
                continue
            start_sequence = range_start
            if range_index == resume_range_index:
                start_sequence = max(range_start, resume_sequence)
            iterator = iter_activation_chunks(
                model=model,
                data_path=path_dir,
                seq_length=tokens_mm.shape[1],
                layer_num=layer_num,
                batch_size=batch_size,
                chunk_sequences=chunk_sequences,
                start_sequence=start_sequence,
                max_sequences=range_end - start_sequence,
                device=device,
                tokens_path=tokens_path,
                mask_path=mask_path,
            )
            for sequence_start, sequence_end, acts in iterator:
                token_chunk = np.array(
                    tokens_mm[sequence_start:sequence_end], copy=True
                )
                if masks_mm is None:
                    valid = token_chunk != 0
                else:
                    valid = np.array(
                        masks_mm[sequence_start:sequence_end], copy=True
                    ).astype(bool)
                flat_tokens = token_chunk[valid]
                analyzer.process_batch(
                    activations=torch.from_numpy(acts).float(),
                    tokens=torch.from_numpy(flat_tokens).long(),
                    batch_start_idx=sequence_start,
                    sequence_tokens=torch.from_numpy(token_chunk).long(),
                    sequence_mask=torch.from_numpy(valid).bool(),
                )
                chunk_count += 1
                next_range_index = range_index
                next_sequence = sequence_end
                if sequence_end >= range_end:
                    next_range_index = range_index + 1
                    next_sequence = 0
                if progress is not None:
                    progress.update(1)

                if verbose == "high" and chunk_count % 200 == 0:
                    active = sum(
                        1 for a in analyzer.feature_analyses.values()
                        if analyzer._activation_counts[a.feature_id] > 0
                    )
                    tqdm.write(
                        f"  Chunk {chunk_count}: {active} observed features"
                    )
                elif (
                    verbose == "low"
                    and (chunk_count % verbose_interval == 0 or chunk_count == total_chunks)
                ):
                    active = sum(
                        1 for a in analyzer.feature_analyses.values()
                        if analyzer._activation_counts[a.feature_id] > 0
                    )
                    print(
                        f"Analysis progress: chunk {chunk_count:,}/{total_chunks:,}; "
                        f"{active:,} observed features"
                    )
                if (
                    chunk_count % checkpoint_every_chunks == 0
                    or next_range_index >= len(ranges)
                ):
                    _atomic_torch_save(
                        {
                            "format": "pythia_feature_analysis_checkpoint_v1",
                            "config": checkpoint_config,
                            "analyzer": analyzer.state_dict(),
                            "next_range_index": int(next_range_index),
                            "next_sequence": int(next_sequence),
                            "processed_chunks": int(chunk_count),
                            "completed": False,
                        },
                        checkpoint_file,
                    )
    finally:
        if progress is not None:
            progress.close()

    print("\nProcessing completed.")

    model.to(device)
    analyzer.compute_logit_lens(logit_batch_size=logit_batch_size)

    analyzer.finalize_analysis()

    summary = analyzer.get_summary_stats()
    print("\n" + "="*60)
    print("SUMMARY OF SAE FEATURES ANALYSIS")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    output_dir = os.path.join(path_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    _save_features_to_text(
        analyzer,
        os.path.join(output_dir, "features_analysis.txt"),
        include_dead_features=False,
        max_examples_per_feature=25,
    )
    _save_features_to_json(
        analyzer,
        os.path.join(output_dir, "features_analysis.json"),
        include_dead_features=True,
    )
    _atomic_torch_save(
        {
            "format": "pythia_feature_analysis_checkpoint_v1",
            "config": checkpoint_config,
            "analyzer": analyzer.state_dict(),
            "next_range_index": int(len(ranges)),
            "next_sequence": 0,
            "processed_chunks": int(chunk_count),
            "completed": True,
            "summary": summary,
        },
        checkpoint_file,
    )


def finalize_analysis_checkpoint(
    sae: TopKSAE,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    path_dir: str,
    layer_num: int,
    analysis_checkpoint_path: str | os.PathLike[str],
    training_usage_counts: Optional[np.ndarray] = None,
    logit_batch_size: int = 256,
):
    """Write reports from a saved analysis checkpoint without new inference."""
    del layer_num  # retained in the public signature for CLI symmetry
    checkpoint_file = Path(analysis_checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Analysis checkpoint not found: {checkpoint_file}")
    checkpoint = _load_torch_checkpoint(checkpoint_file)
    if checkpoint.get("format") != "pythia_feature_analysis_checkpoint_v1":
        raise ValueError(
            "Unsupported analysis checkpoint format: "
            f"{checkpoint.get('format')!r}"
        )

    config = checkpoint.get("config", {})
    device = find_device()
    sae.to(device).eval()
    analyzer = ComprehensiveFeatureAnalyzer(
        sae_model=sae,
        llm_model=model,
        tokenizer=tokenizer,
        device=device,
        context_size=int(config.get("context_size", 25)),
        top_k_examples=int(config.get("top_k_examples", 15)),
        activation_threshold=float(config.get("activation_threshold", 0.03)),
        top_k_logits=int(config.get("top_k", 10)),
        training_usage_counts=training_usage_counts,
    )
    analyzer.load_state_dict(checkpoint["analyzer"])
    context_mapping = config.get("context_mapping")
    if context_mapping is None:
        legacy_tokens_path = Path(config.get("tokens_path", ""))
        legacy_mask_path = Path(config.get("attention_mask_path", ""))
        if not legacy_tokens_path.is_file():
            raise FileNotFoundError(
                "Legacy checkpoint contexts require token sequences for migration. "
                f"Not found: {legacy_tokens_path}. Resume a regular analysis run "
                "once before using finalize-analysis."
            )
        legacy_tokens = np.load(legacy_tokens_path, mmap_mode="r")
        legacy_masks = (
            np.load(legacy_mask_path, mmap_mode="r")
            if legacy_mask_path.is_file()
            else None
        )
        migrated = analyzer.migrate_flattened_context_examples(
            tokens=legacy_tokens,
            masks=legacy_masks,
            chunk_sequences=int(config.get("chunk_sequences", 32)),
        )
        config = dict(config)
        config["context_mapping"] = "sequence-aware-v1"
        checkpoint["config"] = config
        checkpoint["analyzer"] = analyzer.state_dict()
        _atomic_torch_save(checkpoint, checkpoint_file)
        print(f"Migrated {migrated:,} legacy context examples before finalization.")
    elif context_mapping != "sequence-aware-v1":
        raise ValueError(f"Unsupported checkpoint context mapping: {context_mapping!r}")
    print(
        "Finalizing reports from checkpoint only; no new token/model inference "
        f"will be performed. Processed chunks: {checkpoint.get('processed_chunks', 0):,}."
    )
    analyzer.compute_logit_lens(logit_batch_size=logit_batch_size)
    analyzer.finalize_analysis()

    output_dir = Path(path_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "features_analysis.txt"
    json_path = output_dir / "features_analysis.json"
    _save_features_to_text(
        analyzer,
        str(text_path),
        include_dead_features=False,
        max_examples_per_feature=25,
    )
    _save_features_to_json(analyzer, str(json_path), include_dead_features=True)
    print(f"Partial checkpoint text report saved to: {text_path}")
    print(f"Partial checkpoint JSON report saved to: {json_path}")
    return {
        "summary": analyzer.get_summary_stats(),
        "checkpoint_completed": bool(checkpoint.get("completed", False)),
        "processed_chunks": int(checkpoint.get("processed_chunks", 0)),
        "json_path": str(json_path),
        "text_path": str(text_path),
    }
