# Project Summary: LLMs Facts Representation

> **Master thesis** focused on extracting, identifying, and visualizing factual knowledge encoded in the internal structure of Large Language Models (LLMs) using Sparse Autoencoders (SAEs).

---

## 1. Project Goal

The goal of this project is to investigate how factual knowledge is represented inside the hidden layers of a language model. The approach relies on **mechanistic interpretability** — specifically, training a **Top-K Sparse Autoencoder (SAE)** on the internal activations of a small language model and then analyzing the learned features to understand what concepts and tokens each feature encodes.

---

## 2. Model & Dataset

| Component | Details |
|-----------|---------|
| **Target LLM** | `roneneldan/TinyStories-1M` — a 1M-parameter causal language model trained on the TinyStories dataset |
| **Tokenizer** | `EleutherAI/gpt-neo-125M` — used to tokenize texts (vocabulary shared with GPT-Neo) |
| **Dataset** | TinyStories dataset (`data/tinystories_dataset/`), containing simple children's stories; both `train.csv` (~1.9 GB) and `validation.csv` (~19 MB) are present |
| **Layer analyzed** | Layer **4** (configurable via `LAYER_NUM` in `main.py`) |
| **Sequence length** | 256 tokens per sequence |
| **Hidden size** | 64 (model's `hidden_size`) |

---

## 3. Pipeline Architecture

The project implements an end-to-end pipeline orchestrated by `main.py`. It consists of **four sequential stages**:

```
Dataset Sequencing → Activation Collection → Autoencoder Training → Feature Analysis
```

### 3.1. Stage 1 — Dataset Sequencing (`dataset_sequencing.py`)

**Purpose:** Convert raw text from the CSV dataset into fixed-length token sequences suitable for feeding into the LLM.

**What it does:**
- **Loads** the dataset from a CSV file (with interactive file selection via `pick` if multiple files exist).
- **Filters** out empty/non-string entries.
- **Subsamples** the dataset based on `ds_fraction` (default: 1% for experimentation, configurable up to 100%).
- **Counts tokens** across the dataset using parallelized tokenization (`_count_tokens`) to estimate total token count; caches this info in `dataset_info.json` to avoid recomputation.
- **Sentence splitting** via `syntok.segmenter` — each document is split into individual sentences before tokenization.
- **Tokenizes sentences** in batches using the GPT-Neo tokenizer (`_encode_sentences_batch`), splitting overlong sentences into chunks of `seq_length`.
- **Packs** tokenized sentences into sequences of up to `seq_length` tokens (`_pack_and_write_from_sentence_ids`), discarding sequences shorter than `min_seq_length` (default: 10 tokens). Outputs an intermediate JSONL file.
- **Pads** all sequences to exactly `seq_length` with zeros and saves the final result as a NumPy array: `sequenced/tokens_seqs_padded.npy`.

**Key outputs:**
- `sequenced/tokens_seqs_padded.npy` — 2D NumPy array of shape `(num_sequences, 256)` with padded token IDs.
- `dataset_info.json` — cached metadata (fraction used, total token count).

---

### 3.2. Stage 2 — Activation Collection (`activations_collecting.py`)

**Purpose:** Run the TinyStories model on all token sequences and extract the hidden-state activations from a specific transformer layer.

**What it does:**
- Uses `TokenDataset` to load the padded sequences from `tokens_seqs_padded.npy` with memory-mapped access (`mmap_mode="r"`).
- Feeds sequences through the model in batches (`batch_size=4`) with `output_hidden_states=True`.
- Extracts activations from layer `layer_num + 1` (accounting for the embedding layer at index 0).
- Stores activations incrementally in a **memory-mapped temporary file** (`TEMP_activations_layer_{layer_num}.dat`) in `float16` precision to handle large datasets.
- Implements **checkpointing** — saves progress every `checkpoint_freq` batches (default: 10,000) to a progress file, enabling resumption after interruption.
- After completion, converts the memmap file to a standard `.npy` file and cleans up the temporary file.

**Key outputs:**
- `activations/activations_layer_4.npy` — 3D NumPy array of shape `(num_sequences, 256, 64)` containing float16 activations.

**Memory consideration:** For 10% of the TinyStories dataset, the activation file is approximately **6 GB** (as seen in `other/tinystories_activations/`).

---

### 3.3. Stage 3 — Autoencoder Training (`autoencoder_training.py`)

**Purpose:** Train a Top-K Sparse Autoencoder on the collected activations to learn a sparse, interpretable dictionary of features.

**Architecture — `TopKSAE` class:**
- **Encoder:** Linear layer `(d_model → d_sae)` where `d_sae = d_model × expansion_factor`. Default config: `64 → 4096` (expansion factor = 64).
- **Activation:** ReLU applied before Top-K selection.
- **Top-K Sparsity:** Only the `k` largest activations are kept (default `k=8`); all others are zeroed out, producing a sparse feature vector.
- **Decoder:** Weight matrix `W_dec` of shape `(d_sae, d_model)` with a bias term, used to reconstruct the input.
- **Pre-decoder bias trick:** Input is centered by subtracting `b_dec` before encoding (stabilization technique from Anthropic/OpenAI research).
- **Decoder normalization:** After each gradient step, decoder column norms are normalized to unit Euclidean norm to prevent "scale hacking".

**Training details:**
- **Dataset:** `ActivationsDataset` class treats the 3D activation array as individual token-level vectors, flattening across samples and sequence positions.
- **Optimizer:** Adam with learning rate `1e-3`.
- **Scheduler:** Cosine annealing (from `learning_rate` down to `1e-5`).
- **Loss:** MSE between original and reconstructed activations.
- **Epochs:** 5 (default).
- **Batch size:** 256 for SAE training.
- **Checkpointing:** Saves model every `save_every_n_steps` (default: 1000), plus saves best model based on average epoch loss.
- **Visualization:** Plots training loss curve (raw + moving average) and learning rate schedule via `matplotlib`.
- **Evaluation:** Computes MSE, average active feature count, and explained variance on a random batch.

**Key outputs:**
- `models/checkpoints/topk_sae_layer_4.pt` — final model checkpoint (includes optimizer state and training history).
- `models/checkpoints/topk_sae_layer_4_best.pt` — best model by validation loss.
- Periodic step-level checkpoints.

**Previous experiment results** (in `other/models/`):
- `topk_sae_tinystories.pt` (~23 MB) — full checkpoint from a prior experiment.
- `topk_sae_tinystories_best.pt` (~2.1 MB) — best model from a prior experiment.

---

### 3.4. Stage 4 — Feature Analysis (`features_analysis.py`)

**Purpose:** Analyze every learned SAE feature to understand what it represents — which tokens activate it, what tokens it promotes in the model's output, and example contexts.

**Components:**

#### Data Structures
- `FeatureExample` — stores a single activation instance: score, trigger token, position, surrounding context.
- `FeatureAnalysis` — comprehensive per-feature analysis: activation statistics, top examples, promoted/suppressed tokens (logit lens), trigger token frequency, dead feature flag.

#### `ComprehensiveFeatureAnalyzer` class
Core analysis engine that processes all activations through the SAE and aggregates per-feature statistics:

1. **`process_batch()`** — For each activation batch:
   - Runs activations through the SAE to get sparse feature activations.
   - For each of the 4096 features, identifies positions where activation exceeds the threshold (`activation_threshold=0.03`).
   - Maintains a **min-heap** of top-k examples per feature (highest activation scores) with full context (surrounding tokens decoded to text, trigger token highlighted with `[[...]]`).
   - Counts trigger token frequencies per feature.
   - Accumulates activation sums and counts for mean/frequency calculations.

2. **`compute_logit_lens()`** — Logit lens analysis:
   - Multiplies the SAE's decoder weight matrix `W_dec` by the LLM's unembedding matrix `W_U^T` to get `(d_sae × vocab_size)` logit contributions.
   - For each feature, identifies the **top 15 promoted tokens** (highest logit values) and **top 15 suppressed tokens** (lowest logit values).
   - This reveals what each feature "wants to predict" in the model's output.

3. **`finalize_analysis()`** — Computes final statistics:
   - Calculates activation frequency, mean activation, marks dead features (no activations).
   - Sorts examples by activation score.
   - Ranks trigger tokens by frequency.

4. **`_save_features_to_text()`** — Outputs a human-readable text report with:
   - Global summary statistics (active/dead features, frequency ranges).
   - Per-feature sections: activation stats, top trigger tokens, promoted tokens (logit lens), and up to 25 contextual examples.

**Key outputs** (from prior experiments in `other/`):
- `sae_features_analysis.txt` (~1.2 MB) — human-readable feature analysis report.
- `sae_features_analysis.json` (~3.3 MB) — structured JSON analysis data.

---

## 4. Utility Module (`utils.py`)

Simple helper functions:
- `find_device()` — returns the best available device: CUDA → MPS (Apple Silicon) → CPU.
- `find_n_proc()` — returns `cpu_count() - 1` (minimum 1) for parallel data loading.

---

## 5. Experiment Notebooks (in `other/`)

Two Jupyter notebooks document exploratory experiments:
- **`experiments.ipynb`** (~1.1 MB) — primary experiment notebook.
- **`tinystories_experiments.ipynb`** (~176 KB) — TinyStories-specific experiments.

These notebooks likely contain interactive exploration of the trained SAE features, visualization of results, and iterative experimentation that informed the final pipeline design.

---

## 6. Project Structure

```
llms-facts-representation/
├── main.py                      # Pipeline orchestrator
├── dataset_sequencing.py        # Stage 1: text → token sequences
├── activations_collecting.py    # Stage 2: model → activation extraction
├── autoencoder_training.py      # Stage 3: Top-K SAE training
├── features_analysis.py         # Stage 4: feature interpretation
├── utils.py                     # Device & parallelism helpers
├── README.md                    # Brief project description
├── .gitignore                   # Excludes data/, models/, other/
│
├── data/
│   └── tinystories_dataset/
│       ├── train.csv            # ~1.9 GB raw training data
│       ├── validation.csv       # ~19 MB raw validation data
│       ├── sequenced/           # Tokenized & padded sequences
│       ├── activations/         # Extracted LLM activations
│       └── models/              # Trained SAE checkpoints
│
└── other/                       # Prior experiment artifacts
    ├── experiments.ipynb
    ├── tinystories_experiments.ipynb
    ├── sae_features_analysis.txt
    ├── sae_features_analysis.json
    ├── models/
    │   ├── topk_sae_tinystories.pt
    │   └── topk_sae_tinystories_best.pt
    └── tinystories_activations/
        ├── activations_layer_4_tinystories_10perc.npy        # ~6 GB
        └── activations_layer_4_tinystories_10perc_standard.npy  # ~6 GB
```

---

## 7. Version History (Git)

| Commit | Message |
|--------|---------|
| `170a4ea` | Initial commit |
| `472c08c` | First experiments |
| `7732115` | Did some experiments *(current HEAD, `main` branch)* |

---

## 8. Default Pipeline Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| Model | `roneneldan/TinyStories-1M` | `main.py` |
| Tokenizer | `EleutherAI/gpt-neo-125M` | `main.py` |
| Sequence length | 256 | `main.py` |
| Layer number | 4 | `main.py` |
| Dataset fraction | 1% | `main.py` |
| LLM batch size | 4 | `main.py` |
| SAE input dim (`d_model`) | 64 | `main.py` |
| SAE expansion factor | 64 (→ 4096 features) | `main.py` |
| Top-K sparsity | 8 | `main.py` |
| SAE batch size | 256 | `main.py` |
| SAE epochs | 5 | `main.py` |
| Learning rate | 1e-3 | `main.py` |
| LR scheduler | Cosine annealing (min: 1e-5) | `autoencoder_training.py` |
| Activation threshold | 0.03 | `features_analysis.py` |

---

## 9. Dependencies

The project relies on the following key Python libraries (inferred from imports):
- **PyTorch** — model inference, SAE training, GPU/MPS acceleration
- **Hugging Face Transformers** — loading pretrained LLM and tokenizer
- **Hugging Face Datasets** — dataset loading and processing
- **NumPy** — array operations, memory-mapped file handling
- **Matplotlib** — training visualization
- **syntok** — sentence segmentation
- **pick** — interactive CLI file selection
- **tqdm** — progress bars

---

## 10. Key Technical Decisions

1. **Top-K SAE over vanilla SAE:** Uses Top-K sparsity constraint instead of L1 regularization, which provides more direct control over the number of active features per input and avoids the need to tune a sparsity penalty coefficient.

2. **Pre-decoder bias trick:** Following Anthropic/OpenAI research, the decoder bias is subtracted from the input before encoding, which helps center the representation and stabilize training.

3. **Decoder weight normalization:** After every training step, decoder column norms are normalized to unit length to prevent the model from "cheating" by scaling decoder weights while shrinking encoder outputs.

4. **Float16 activations:** Activations are stored in half-precision to manage the ~6 GB file sizes.

5. **Memory-mapped files:** Both activation collection and loading use NumPy memory-mapped arrays (`mmap_mode`) to handle datasets that don't fit in RAM.

6. **Resumable activation collection:** Checkpoint-based progress tracking allows collection to be interrupted and resumed, essential for the multi-hour collection process.

7. **Logit lens analysis:** By projecting SAE decoder directions through the LLM's unembedding matrix, each feature can be interpreted in terms of the output tokens it promotes — a powerful interpretability technique.

---

## 11. Summary of Work Done

1. **Set up the research environment** — selected TinyStories-1M as the target model (small enough for rapid experimentation, yet complex enough to learn meaningful representations).

2. **Built a complete data preprocessing pipeline** — tokenization, sentence segmentation, sequence packing, and padding to create fixed-length inputs.

3. **Implemented activation extraction** — with checkpointing and memory-mapped storage for scalability.

4. **Designed and trained a Top-K Sparse Autoencoder** — with 4096 features (64× expansion), cosine LR schedule, decoder normalization, and comprehensive checkpointing.

5. **Built a comprehensive feature analysis framework** — including:
   - Per-feature activation statistics (frequency, mean, max).
   - Top activation examples with full textual context.
   - Trigger token frequency analysis.
   - Logit lens analysis (promoted/suppressed tokens).
   - Export to both human-readable text and structured JSON.

6. **Ran experiments** — trained SAE on 10% of TinyStories data, generated ~1.2 MB of feature analysis text and ~3.3 MB of structured JSON results. Experiment notebooks document exploratory analysis of the results.

7. **Iterated on the pipeline** — refactored from notebook-based experimentation (in `other/`) into a modular, reusable Python pipeline with clear separation of concerns.
