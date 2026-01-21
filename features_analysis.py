from __future__ import annotations  # TODO: refactor the code not to use this

import torch
from utils import find_device
from transformers import AutoModelForCausalLM, AutoTokenizer
from autoencoder_training import TopKSAE
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import heapq
from torch.utils.data import Dataset
import numpy as np

# TODO: create this dataset while collecting activations
def _create_activations_text_dataset(model: AutoModelForCausalLM, device: torch.device, layer_num: int, num_samples: int = 5000):
    tokenized_seqs_validation = np.load('data/tinystories_dataset/sequenced/tokens_seqs_padded.npy')
    if num_samples != -1:
        tokenized_seqs_validation = tokenized_seqs_validation[:num_samples]
    BATCH_SIZE = 4

    tokenized_seqs_validation_dataloader = DataLoader(tokenized_seqs_validation, batch_size=BATCH_SIZE, shuffle=False)
    tokenized_seqs_validation_helper_arr = []
    activations_validation_helper_arr = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(tokenized_seqs_validation_dataloader)):
            tokenized_seqs_validation_helper_arr.extend(batch)
            input_ids = batch.to(device)

            outputs = model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            layer_activations = outputs.hidden_states[LAYER_NUM + 1] 
            layer_activations_np = layer_activations.cpu().numpy().astype(np.float16)
            
            activations_validation_helper_arr.extend(layer_activations_np)
    
    class ActivationsTextDataset(Dataset):
        def __init__(self, activations_arr, tokenized_seqs_arr):
            self.activations = np.array(activations_arr)
            self.tokenized_seqs = np.array(tokenized_seqs_arr)
            self.num_samples, self.seq_len, self.d_model = self.activations.shape
            self.total_vectors = self.num_samples * self.seq_len
        
        def __len__(self):
            return min(len(self.activations), len(self.tokenized_seqs))
        
        def __getitem__(self, idx):
            return self.activations[idx], self.tokenized_seqs[idx]

    return ActivationsTextDataset(activations_validation_helper_arr, tokenized_seqs_validation_helper_arr)


def _save_features_to_text(
    analyzer: ComprehensiveFeatureAnalyzer,
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
                f.write("STATUS: DEAD FEATURE (no activations)\n\n")
                continue
            
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
        top_k_logits: int = 15
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
    
    def compute_logit_lens(self):
        """Computes logit lens for all features - which tokens are promoted."""
        
        print("Calculating logit lens for all features...")
        
        W_U = self.llm.lm_head.weight.detach().to(self.device)  # [vocab_size, d_model]
        
        W_dec = self.sae.W_dec.detach().to(self.device)  # [d_sae, d_model]
        
        # [d_sae, d_model] @ [d_model, vocab_size] -> [d_sae, vocab_size]
        feature_logits = W_dec @ W_U.T
        
        for feat_idx in range(self.num_features):
            logits = feature_logits[feat_idx]
            
            top_vals, top_ids = torch.topk(logits, self.top_k_logits)
            promoted = [
                (self.tokenizer.decode([tid.item()]), val.item()) 
                for tid, val in zip(top_ids, top_vals)
            ]
            self.feature_analyses[feat_idx].top_promoted_tokens = promoted
            
            bottom_vals, bottom_ids = torch.topk(logits, self.top_k_logits, largest=False)
            suppressed = [
                (self.tokenizer.decode([tid.item()]), val.item()) 
                for tid, val in zip(bottom_ids, bottom_vals)
            ]
            self.feature_analyses[feat_idx].top_suppressed_tokens = suppressed
        
    
    def finalize_analysis(self):
        """Finalizes the analysis - calculates final statistics."""
        
        print("Finalizing analysis...")
        
        for feat_idx in range(self.num_features):
            analysis = self.feature_analyses[feat_idx]
            
            if self._activation_counts[feat_idx] > 0:
                analysis.total_activations = self._activation_counts[feat_idx]
                analysis.mean_activation = (
                    self._activation_sums[feat_idx] / self._activation_counts[feat_idx]
                )
                analysis.activation_frequency = (
                    self._activation_counts[feat_idx] / self._total_tokens_processed
                )
            else:
                analysis.is_dead = True
            
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
        
        print(f"Analysis completed.")
        print(f"  - Total features: {self.num_features}")
        print(f"  - Active features: {alive_count}")
        print(f"  - Dead features: {dead_count}")
        print(f"  - Processed tokens: {self._total_tokens_processed:,}")
    
    def get_summary_stats(self) -> Dict:
        """Returns summary statistics."""
        
        active_features = [a for a in self.feature_analyses.values() if not a.is_dead]
        
        if not active_features:
            return {"error": "No active features"}
        
        return {
            "total_features": self.num_features,
            "active_features": len(active_features),
            "dead_features": self.num_features - len(active_features),
            "mean_activation_frequency": np.mean([a.activation_frequency for a in active_features]),
            "max_activation_frequency": max(a.activation_frequency for a in active_features),
            "min_activation_frequency": min(a.activation_frequency for a in active_features),
            "total_tokens_processed": self._total_tokens_processed
        }


def analyze_sae(sae: TopKSAE, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, path_dir: str, layer_num: int, top_k: int = 10, context_size: int = 25, top_k_examples: int = 15, activation_threshold: float = 0.03, batch_size: int = 4):
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
        top_k_logits=top_k
    )

    print(f"Number of features to analyze: {analyzer.num_features}")
    print("Starting analysis of activations...")

    activations_text_dataset = _create_activations_text_dataset(model, device, layer_num, num_samples=-1)
    activations_loader = DataLoader(activations_text_dataset, batch_size=4, shuffle=False)
    print(f"Dataset: {len(activations_loader)} batches")

    for batch_idx, (acts, tokens) in enumerate(tqdm(activations_loader, desc="Analysis of features")):
        analyzer.process_batch(
            activations=acts,
            tokens=tokens,
            batch_start_idx=batch_idx * batch_size
        )
        
        if (batch_idx + 1) % 200 == 0:
            active = sum(1 for a in analyzer.feature_analyses.values() 
                        if analyzer._activation_counts[a.feature_id] > 0)
            tqdm.write(f"  Batch {batch_idx+1}: {active} active features")

    print("\nProcessing completed.")

    model.to(device)
    analyzer.compute_logit_lens()

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

    _save_features_to_text(analyzer, os.path.join(path_dir, 'analysis/features_analysis.txt'), include_dead_features=False, max_examples_per_feature=25)