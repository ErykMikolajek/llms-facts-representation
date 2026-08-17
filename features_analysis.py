from __future__ import annotations  # TODO: refactor the code not to use this

import os
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import heapq
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoencoder_training import TopKSAE
from utils import find_device

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
        batch_start_idx: int
    ):
        """Processes a batch of activations and collects statistics."""
        
        self.sae.eval()
        if activations.ndim == 2:
            # Streaming extraction supplies valid tokens as [N, D]. Treat one
            # bounded chunk as a pseudo-sequence; this keeps analysis bounded
            # without requiring a complete activation file. Context examples
            # remain useful within the chunk and are never used for training.
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
            
            for batch_i in range(batch_size):
                for seq_i in range(seq_len):
                    score = feat_acts[batch_i, seq_i].item()
                    
                    if score <= self.activation_threshold:
                        continue
                    
                    trigger_token_id = tokens[batch_i, seq_i].item()
                    trigger_token_str = self.tokenizer.decode([trigger_token_id])
                    self._trigger_token_counts[feat_idx][trigger_token_str] += 1
                    
                    heap = self._example_heaps[feat_idx]
                    
                    if len(heap) < self.top_k_examples:
                        context_before, context_after, full_context = self._get_context(
                            tokens[batch_i], seq_i, seq_len
                        )
                        example = FeatureExample(
                            activation_score=score,
                            trigger_token=trigger_token_str,
                            trigger_token_id=trigger_token_id,
                            position_in_seq=seq_i,
                            context_before=context_before,
                            context_after=context_after,
                            full_context=full_context,
                            sequence_id=batch_start_idx + batch_i
                        )
                        heapq.heappush(heap, (score, id(example), example))
                        
                    elif score > heap[0][0]:
                        context_before, context_after, full_context = self._get_context(
                            tokens[batch_i], seq_i, seq_len
                        )
                        example = FeatureExample(
                            activation_score=score,
                            trigger_token=trigger_token_str,
                            trigger_token_id=trigger_token_id,
                            position_in_seq=seq_i,
                            context_before=context_before,
                            context_after=context_after,
                            full_context=full_context,
                            sequence_id=batch_start_idx + batch_i
                        )
                        heapq.heapreplace(heap, (score, id(example), example))
        
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
    starts = np.linspace(
        0,
        max(0, sequence_count - chunk_sequences),
        window_count,
        dtype=int,
    )
    ranges = []
    remaining = max_sequences
    for start in starts:
        size = min(chunk_sequences, remaining)
        start = int(start)
        end = min(sequence_count, start + size)
        ranges.append((start, end))
        remaining -= end - start
        if remaining <= 0:
            break
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
):
    if verbose not in {"low", "high"}:
        raise ValueError("verbose must be 'low' or 'high'")
    if verbose_interval < 1:
        raise ValueError("verbose_interval must be positive")
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

    tokens_path = os.path.join(path_dir, "sequenced/tokens_seqs_padded.npy")
    mask_path = os.path.join(path_dir, "sequenced/attention_mask.npy")
    tokens_mm = np.load(tokens_path, mmap_mode="r")
    masks_mm = np.load(mask_path, mmap_mode="r") if os.path.exists(mask_path) else None
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

    from activations_collecting import iter_activation_chunks

    total_chunks = sum(
        (end - start + chunk_sequences - 1) // chunk_sequences
        for start, end in ranges
    )
    chunk_count = 0
    progress = None
    if verbose == "high":
        progress = tqdm(
            total=total_chunks,
            desc="Analysis of streaming activations",
        )
    try:
        for range_start, range_end in ranges:
            iterator = iter_activation_chunks(
                model=model,
                data_path=path_dir,
                seq_length=tokens_mm.shape[1],
                layer_num=layer_num,
                batch_size=batch_size,
                chunk_sequences=chunk_sequences,
                start_sequence=range_start,
                max_sequences=range_end - range_start,
                device=device,
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
                )
                chunk_count += 1
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
