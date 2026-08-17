# llms-facts-representation

Repozytorium do pracy magisterskiej nad reprezentacją faktów i konceptów w małym modelu językowym. Aktualnym celem jest solidny, storage-bounded trening **Top-K Sparse Autoencodera**. Późniejsze etapy semantic triage, pruning i MoE pozostają odłożonym eksperymentem.

Głównym celem jest `EleutherAI/pythia-160m` (GPT-NeoX), z warstwą `6`, `hidden_size=768` i MLP `768 -> 3072 -> 768`. Profil `tiny` zachowuje wcześniejszą konfigurację TinyStories do smoke testów.

Szczegółowy opis architektury, przepływów i artefaktów znajduje się w [`architecture.md`](architecture.md).

## Aktualny pipeline SAE

- sekwencjonowanie zapisuje tokeny oraz jawny `attention_mask` przez memmap,
- kolektor używa hooka jednej warstwy i nie uruchamia `output_hidden_states=True`,
- aktywacje są spłaszczane do ograniczonych chunków i zużywane natychmiast przez SAE,
- nie powstaje pełny plik aktywacji dla całego datasetu,
- checkpoint SAE przechowuje optimizer, scheduler, RNG, historię oraz `epoch/next_sequence`,
- trening można wznowić po zakończonym chunku,
- analiza cech również czyta aktywacje strumieniowo i liczy logit lens blokami.

Pruning i budowanie MoE nie są obecnie częścią aktywnego celu.

## Struktura Kodu

- `main.py` - CLI etapów `sequence`, `train-sae`, `analyze` i `all`.
- `dataset_sequencing.py` - CSV albo Pile-style JSONL/JSONL.ZST do memmapów tokenów i maski.
- `activations_collecting.py` - bounded one-layer activation chunks dla GPT-Neo/GPT-NeoX.
- `autoencoder_training.py` - `TopKSAE` oraz resumable streaming trainer.
- `features_analysis.py` - raport tekstowy i JSON z analizą cech SAE.
- `semantic_domain_triage.py` - logit lens, HDBSCAN, domeny i walidacje domenowe.
- `topographic_mlp_sae_mapping.py` - korelacja residual hidden state z domenami SAE, etap eksploracyjny.
- `domain_mlp_activation_mapping.py` - właściwe mapowanie neuronów MLP przez hook na wejściu `mlp.c_proj`.
- `domain_mlp_pruning.py` - progi `tau`, maski neuronów i eksport MLP-only ekspertów.
- `router_training.py` - trening liniowego routera domen na wejściu do MLP.
- `moe_assembly.py` - składanie pojedynczego eksperta albo hard-routed MoE.
- `moe_validation.py` - walidacja causal LM loss/perplexity dla bazy, ekspertów i MoE.
- `utils.py` - pomocniczy wybór urządzenia i liczby procesów.

Katalogi `data/`, `models/` i `other/` są ignorowane przez Git, bo zawierają duże dane, checkpointy i artefakty eksperymentalne.

## Instalacja

```bash
pip install -r requirements.txt
```

Wymagane biblioteki obejmują `torch`, `transformers`, `datasets`, `numpy`, `scipy`, `scikit-learn`, `hdbscan`, `pandas`, `syntok`, `zstandard`, `pick` i `tqdm`.

## Pipeline SAE

Smoke test TinyStories:

```bash
python3 main.py --profile tiny --stage all --no-interactive
```

Pythia-160m, lokalny profil:

```bash
python3 main.py --profile local-50gb --stage all \
  --input-file train.csv --max-sequences 20000 --no-interactive
```

### Kaggle: Pythia-160m i shardy The Pile

`main.py` czyta katalog z wieloma plikami `*.jsonl`, `*.jsonl.zst` albo
`*.jsonl.gz` strumieniowo. Źródła w `/kaggle/input` pozostają niezmienione;
pod `--data-path` powstają tokeny, maska, manifest i checkpointy.

Gotowy, uporządkowany notebook dla wytrenowanego modelu Pile-CC znajduje się w
[`kaggle_pythia160m_sae.ipynb`](kaggle_pythia160m_sae.ipynb). Na jego początku
ustawia się `RUN_MODE = "TRAIN"` albo `RUN_MODE = "ANALYZE"`. Analiza ma
domyślnie `ANALYSIS_MAX_SEQUENCES = None`, czyli przechodzi po całym zbiorze
sekwencji z `/kaggle/working/pythia160m_sae_pilecc` i ładuje checkpoint
`/kaggle/input/datasets/erykmikoajek/trained-sae-models/topk_sae_layer_6_best.pt`.

Argument `--verbose low --verbose-interval 1000` wyłącza odświeżanie `tqdm` w
logach Kaggle. Zamiast tego trening wypisuje postęp co 1000 kroków, a analiza
co 1000 chunków. `--verbose high` zachowuje pełne paski postępu.

```python
from pathlib import Path
import subprocess
import sys

CODE_DIR = Path("/kaggle/input/datasets/erykmikoajek/sae-training-and-moeffication")
PILE_DIR = Path("/kaggle/input/datasets/dschettler8845/the-pile-github-files-part-01")
OUT_DIR = Path("/kaggle/working/pythia160m_sae")
MAIN = CODE_DIR / "main.py"

def run_cli(*args):
    command = [sys.executable, str(MAIN), *map(str, args)]
    print(" ".join(command))
    subprocess.run(command, check=True)
```

Pilot sekwencjonowania:

```python
run_cli(
    "--stage", "sequence", "--profile", "colab",
    "--model-name", "EleutherAI/pythia-160m",
    "--tokenizer-name", "EleutherAI/pythia-160m",
    "--data-path", OUT_DIR, "--input-path", PILE_DIR,
    "--file-pattern", "*.jsonl*", "--max-files", 1,
    "--max-tokens", 5_000_000, "--seq-length", 256,
    "--no-interactive",
)
```

Po sprawdzeniu pilota uruchom trening kontrolny z `--max-steps 50`, a potem
powtórz komendę bez tego limitu. Ten sam katalog `OUT_DIR` pozwala wznowić
checkpoint od ostatniego zakończonego chunka:

```python
run_cli(
    "--stage", "train-sae", "--profile", "colab",
    "--model-name", "EleutherAI/pythia-160m",
    "--tokenizer-name", "EleutherAI/pythia-160m",
    "--data-path", OUT_DIR, "--layer-num", 6, "--seq-length", 256,
    "--model-batch-size", 2, "--chunk-sequences", 16,
    "--batch-size-sae", 512, "--expansion-factor", 8, "--k", 64,
    "--num-epochs", 1, "--max-steps", 50, "--no-interactive",
)
```

Na właściwym przebiegu usuń `--max-steps`; zachowaj `OUT_DIR` i konfigurację.
Profil `colab` jest bezpieczniejszym punktem startowym na pojedynczej sesji,
a `local-50gb` daje większy słownik SAE.

Jeżeli Kaggle przydzieli Tesla P100, a PyTorch zgłasza brak kernela dla
`sm_60`, uruchom notebook na T4/L4/A100 albo wymuś CPU przez
`SAE_DEVICE=cpu`. `utils.find_device()` wykrywa teraz tę niezgodność i nie
próbuje wykonywać nieobsługiwanych kerneli CUDA.

Po przerwaniu powtórzenie tej samej komendy wznowi checkpoint SAE. Aby zacząć
od nowa, użyj `--no-resume` i osobnego katalogu/checkpointu.

Profile są zdefiniowane w `main.py`:

- `tiny`: TinyStories, `64 -> 4096`, `k=8`,
- `local-50gb`: Pythia-160m, `d_sae=12288`, `k=64`, warstwa `6`,
- `colab`: mniejszy wariant Pythia z krótszym treningiem.

Główne artefakty:

- `sequenced/tokens_seqs_padded.npy`,
- `sequenced/attention_mask.npy`,
- `models/checkpoints/topk_sae_layer_<layer>.pt`,
- `models/checkpoints/topk_sae_layer_<layer>_best.pt`,
- `analysis/features_analysis.txt` i `.json` po opcjonalnym etapie `analyze`.

Kolektor nie zapisuje `activations/activations_layer_<layer>.npy`; parametr
`keep_full_file=True` w funkcji kompatybilności jest przeznaczony wyłącznie do
odtwarzania starych eksperymentów.

## Semantic Domain Triage

Po wytrenowaniu SAE można zbudować domeny semantyczne z logit lens:

```bash
python3 semantic_domain_triage.py \
  --data-path data/tinystories_dataset \
  --checkpoint data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt \
  --n-domains 5 \
  --min-required-domains 3 \
  --samples-per-domain 100
```

Skrypt:

- liczy `W_dec @ W_U.T` dla cech SAE,
- filtruje tokeny techniczne i częste tokeny funkcyjne,
- zapisuje `feature_token_matrix.npz` oraz `logit_lens_top_tokens.jsonl`,
- redukuje macierz przez SVD,
- klastruje cechy przez HDBSCAN,
- wybiera ortogonalne domeny,
- buduje `domain_validation/domain_<id>.jsonl`.

Domyślnie skrypt wymaga przynajmniej 3 domen. Jeśli HDBSCAN znajdzie mniej, run kończy się błędem z diagnostyką. Do diagnostycznego zapisu słabych wyników można użyć:

```bash
python3 semantic_domain_triage.py \
  --checkpoint data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt \
  --allow-underfilled-domains
```

Wyniki trafiają do `data/tinystories_dataset/analysis/domain_triage/`, chyba że podasz `--output-dir`.

## MLP Mapping

Etap pruningowy wymaga prawdziwych neuronów MLP, nie 64-wymiarowego residual stream. Do tego służy:

```bash
python3 domain_mlp_activation_mapping.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --checkpoint data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt \
  --layer-num 4
```

Skrypt zakłada hook na `model.transformer.h[layer_num].mlp.c_proj` i przechwytuje wejście do `c_proj`, czyli post-aktywacje wewnętrznych neuronów MLP. Dla TinyStories-1M jest to zwykle `n_inner=256`.

Domyślnie mapping odrzuca domeny, których zagregowany sygnał SAE jest pusty. Do zapisu artefaktów diagnostycznych można jawnie dodać:

```bash
python3 domain_mlp_activation_mapping.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --allow-zero-domain-signal
```

## Pruning Ekspertów

Po mappingu można zbudować sparse MLP-only ekspertów:

```bash
python3 domain_mlp_pruning.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --tau-method per-domain-null \
  --null-permutations 100 \
  --null-percentile 95
```

Pruning:

- czyta `domain_mlp_mapping/*.npz` i korelacje CSV,
- liczy próg `tau` metodą null baseline albo kwantylem,
- buduje `keep_mask`,
- zeruje prunowane neurony w `c_fc` i `c_proj`,
- zapisuje `domain_<id>_mlp_expert.pt`, `domain_<id>_mask.npy` i `domain_<id>_pruning.json`.

Puste eksperci są domyślnie blokowani. Jeśli celowo chcesz wyeksportować zdegenerowany artefakt diagnostyczny, użyj:

```bash
python3 domain_mlp_pruning.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --empty-signal-policy prune-all \
  --allow-empty-experts
```

## Router Training

Po uzyskaniu niepustych ekspertów można wytrenować router domenowy. Router jest liniowy (`nn.Linear(d_model, n_domains)`) i trenuje się na wejściu do MLP warstwy 4, przechwyconym przez `forward_pre_hook` na `model.transformer.h[layer_num].mlp`.

```bash
python3 router_training.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --layer-num 4 \
  --batch-size 8 \
  --epochs 10
```

Wyjścia:

- `data/tinystories_dataset/analysis/domain_triage/router/router.pt`,
- `data/tinystories_dataset/analysis/domain_triage/router/router_metrics.json`,
- `data/tinystories_dataset/analysis/domain_triage/router/router_report.md`.

Do szybkiego smoke testu:

```bash
python3 router_training.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --max-texts-per-domain 5 \
  --epochs 1 \
  --batch-size 2
```

## MoE Assembly

`moe_assembly.py` udostępnia funkcje importowalne i prosty CLI do sprawdzenia, czy da się złożyć model:

```bash
python3 moe_assembly.py \
  --mode moe \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --layer-num 4
```

Obsługiwane są dwa tryby:

- `single-expert` - podstawia jeden `domain_<id>_mlp_expert.pt` za MLP warstwy 4.
- `moe` - ładuje `router/router.pt` i `domain_mlp_experts/`, a następnie zastępuje MLP wrapperem `HardRoutedMLP`.

Pierwsza wersja `HardRoutedMLP` jest poprawnościowa: używa pełnych MLP z wyzerowanymi wagami i wybiera output per token przez maski. Nie daje jeszcze realnego speed-upu.

## PPL Validation

Pełne porównanie PPL:

```bash
python3 moe_validation.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --experts-dir data/tinystories_dataset/analysis/domain_triage/domain_mlp_experts \
  --router-path data/tinystories_dataset/analysis/domain_triage/router/router.pt \
  --validation-csv data/tinystories_dataset/validation.csv \
  --layer-num 4 \
  --output-dir data/tinystories_dataset/analysis/domain_triage/moe_validation
```

Wyjścia:

- `moe_validation/ppl_results.json`,
- `moe_validation/ppl_report.md`.

Smoke test na małej próbce:

```bash
python3 moe_validation.py \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --max-texts-per-domain 5 \
  --max-general-texts 20 \
  --batch-size 2
```

## Co Jest Jeszcze Do Zrobienia

Najważniejsze dalsze prace po implementacji pierwszej wersji MoE:

1. Uruchomić pełny pipeline na 3-5 niepustych domenach i zapisać rzeczywiste wyniki PPL.
2. Dodać testy jednostkowe dla routera, `HardRoutedMLP` i obliczania PPL.
3. Zoptymalizować `HardRoutedMLP`, żeby nie liczył wszystkich ekspertów na całym batchu.
4. Zmaterializować rzadkie podgrafy MLP, jeśli celem stanie się realny speed-up, a nie tylko eksperyment interpretowalności.

Te etapy powinny powstać dopiero po uzyskaniu 3-5 niepustych, sensownych domen semantycznych.
