# llms-facts-representation

Repozytorium do pracy magisterskiej nad reprezentacją faktów i konceptów w małym modelu językowym. Obejmuje storage-bounded trening **Top-K Sparse Autoencodera** oraz eksperymentalną ścieżkę Gemma Scope 2 → semantic triage → pruning → hard-routed MoE.

Głównym celem jest `EleutherAI/pythia-160m` (GPT-NeoX), z warstwą `6`, `hidden_size=768` i MLP `768 -> 3072 -> 768`. Profil `tiny` zachowuje wcześniejszą konfigurację TinyStories do smoke testów.

Szczegółowy opis architektury, przepływów i artefaktów znajduje się w
[`documentation/architecture.md`](documentation/architecture.md).
Osobna, przeglądalna dokumentacja każdego modułu eksperymentalnego wraz z
diagramem pipeline'u znajduje się w [`documentation/`](documentation/README.md).

## Aktualny pipeline SAE

- sekwencjonowanie zapisuje tokeny oraz jawny `attention_mask` przez memmap,
- kolektor używa hooka jednej warstwy i nie uruchamia `output_hidden_states=True`,
- aktywacje są spłaszczane do ograniczonych chunków i zużywane natychmiast przez SAE,
- nie powstaje pełny plik aktywacji dla całego datasetu,
- checkpoint SAE przechowuje optimizer, scheduler, RNG, historię oraz `epoch/next_sequence`,
- trening można wznowić po zakończonym chunku,
- analiza cech również czyta aktywacje strumieniowo i liczy logit lens blokami.

Pipeline Gemmy został wykonany do etapu niezależnej walidacji. Kryteria
badawcze i kolejność uruchomienia opisuje
[`reports/GEMMA_MOE_RESEARCH_AUDIT.md`](reports/GEMMA_MOE_RESEARCH_AUDIT.md).

## Struktura repozytorium

Każdy katalog odpowiada etapowi eksperymentu:

```text
sae_pipeline/    sekwencjonowanie, aktywacje, trening i analiza SAE
domain_triage/   odkrywanie, stabilność i wybór domen
domain_mapping/  zbiory domenowe, mapowanie SAE–MLP, selektywność i pruning
moe/             router, składanie MoE, zamrożenie protokołu i bundle
evaluation/      holdout PPL oraz niezależny benchmark kompetencyjny
kaggle/          CLI, notebooki, buildery i wersjonowane assety
tests/           testy jednostkowe i regresyjne
reports/         raporty badawcze
results/         zaimportowane wyniki eksperymentów
documentation/   źródła MkDocs i dokumenty architektoniczne
common/          współdzielone funkcje techniczne
```

## Struktura kodu

- `sae_pipeline/main.py` - CLI etapów `sequence`, `train-sae`, `analyze` i `all`.
- `sae_pipeline/dataset_sequencing.py` - CSV albo Pile-style JSONL/JSONL.ZST do memmapów tokenów i maski.
- `sae_pipeline/activations_collecting.py` - bounded one-layer activation chunks dla GPT-Neo/GPT-NeoX i Gemmy 3.
- `sae_pipeline/autoencoder_training.py` - `TopKSAE` oraz resumable streaming trainer.
- `sae_pipeline/features_analysis.py` - raport tekstowy i JSON z analizą cech SAE.
- `sae_pipeline/gemma_scope_analysis.py` - streamingowa analiza gotowego SAE Gemma Scope 2
  dla Gemma 3 270M (`resid_post`), bez treningu.
- `domain_triage/semantic_domain_triage.py` - logit lens, HDBSCAN, domeny i walidacje domenowe.
- `domain_mapping/topographic_mlp_sae_mapping.py` - korelacja residual hidden state z domenami SAE, etap eksploracyjny.
- `domain_mapping/domain_mlp_activation_mapping.py` - mapowanie fizycznych neuronów MLP przez hook wejścia projekcji wyjściowej (`c_proj`, `dense_4h_to_h` albo `down_proj`).
- `domain_mapping/domain_mlp_pruning.py` - progi `tau`, maski neuronów i eksport MLP-only ekspertów.
- `moe/router_training.py` - trening liniowego routera domen na wejściu do MLP.
- `moe/moe_assembly.py` - składanie pojedynczego eksperta albo hard-routed MoE.
- `evaluation/moe_validation.py` - walidacja causal LM loss/perplexity dla bazy, ekspertów i MoE.
- `common/utils.py` - pomocniczy wybór urządzenia i liczby procesów.

Katalogi `data/`, `models/` i `other/` są ignorowane przez Git, bo zawierają duże dane, checkpointy i artefakty eksperymentalne.

Katalog `praca_tex/source_code/` jest historycznym snapshotem do materiałów
pracy, nie źródłem wykonywalnym. Po zamknięciu eksperymentu należy wygenerować
go ponownie z plików w pakietach etapowych; bieżącego pipeline'u nie należy
uruchamiać z tej kopii.

## Instalacja

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -c constraints.txt
```

Projekt wymaga Pythona 3.11 lub nowszego; Python 3.11 jest obecnie najbardziej
konserwatywnym wyborem dla lokalnego stosu PyTorch, SAELens i Kaggle CLI.
`constraints.txt` utrwala zestaw wersji zweryfikowany w lokalnym `.venv`.

Wymagane biblioteki obejmują `torch`, `transformers`, `sae-lens`, `datasets`, `numpy`, `scipy`, `scikit-learn`, `hdbscan`, `pandas`, `syntok`, `zstandard`, `pick` i `tqdm`.

## Pipeline SAE

Smoke test TinyStories:

```bash
python3 -m sae_pipeline.main --profile tiny --stage all --no-interactive
```

Pythia-160m, lokalny profil:

```bash
python3 -m sae_pipeline.main --profile local-50gb --stage all \
  --input-file train.csv --max-sequences 20000 --no-interactive
```

### Kaggle: Pythia-160m i shardy The Pile

`sae_pipeline/main.py` czyta katalog z wieloma plikami `*.jsonl`, `*.jsonl.zst` albo
`*.jsonl.gz` strumieniowo. Źródła w `/kaggle/input` pozostają niezmienione;
pod `--data-path` powstają tokeny, maska, manifest i checkpointy.

Gotowy, uporządkowany notebook dla wytrenowanego modelu na 50-milionowej
próbce tokenów z pliku `00.jsonl` znajduje się w
[`kaggle/notebooks/kaggle_pythia160m_sae.ipynb`](kaggle/notebooks/kaggle_pythia160m_sae.ipynb). Na jego początku
ustawia się `RUN_MODE = "TRAIN"` albo `RUN_MODE = "ANALYZE"`. Analiza ma
domyślnie `ANALYSIS_MAX_SEQUENCES = None`, czyli przechodzi po całym zbiorze
sekwencji z `/kaggle/working/pythia160m_sae_pilecc` i ładuje checkpoint
`topk_sae_layer_6_best_pilecc.pt` z datasetu `trained-sae-models`. Nie jest to
analiza całego wieloshardowego korpusu The Pile: manifest lokalnego artefaktu
wskazuje limit `50_000_000` tokenów i wyłącznie shard `00.jsonl`.

Notebook ma również `USE_PREPARED_SEQUENCES = True`. W tym trybie korzysta
bezpośrednio z `tokens_seqs_padded_pythia.npy` i `attention_mask_pythia.npy` w
`/kaggle/input/datasets/erykmikoajek/trained-sae-models/`, zarówno podczas
treningu, jak i analizy. Nie uruchamia wtedy sekwencjonowania The Pile ani nie
tworzy `TEMP_tokenized_not_padded.jsonl`. Ustawienie `False` przywraca
opcjonalne automatyczne sekwencjonowanie do `RESULTS_DIR`.

Analiza zapisuje checkpoint w
`RESULTS_DIR/analysis/analysis_checkpoint_pythia.pt`. `RESUME_ANALYSIS = True`
wznawia pracę od ostatniego zapisu, a `CHECKPOINT_EVERY_CHUNKS` steruje
częstotliwością checkpointów. `RESUME_ANALYSIS = False` rozpoczyna analizę od
zera.

Do obejrzenia aktualnego, częściowego stanu bez dalszego przetwarzania ustaw
`FINALIZE_CHECKPOINT_ONLY = True`. Zostaną wtedy wygenerowane raporty na bazie
dotychczasowych danych zapisanych w checkpointcie.

Jeśli checkpoint ma zostać przeniesiony do nowej sesji, można opublikować go
jako plik Kaggle Dataset i ustawić w notebooku `CHECKPOINT_INPUT_PATH` na jego
ścieżkę w `/kaggle/input`. Notebook skopiuje go wtedy do
`CHECKPOINT_WORKING_PATH`, ponieważ `/kaggle/input` jest tylko do odczytu.

Kolejne runy Pythii i Gemmy oraz przełączanie między trzema kontami automatyzuje
[`kaggle/cli/kaggle_pipeline.py`](kaggle/cli/kaggle_pipeline.py). Instrukcja
konfiguracji credentials oraz wyboru `--workflow pythia|gemma|moe_benchmark` znajduje się w
[`kaggle/cli/README.md`](kaggle/cli/README.md).

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
MAIN = CODE_DIR / "sae_pipeline/main.py"

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

Profile są zdefiniowane w `sae_pipeline/main.py`:

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

### Kaggle: Gemma Scope 2 dla Gemma 3 270M

Pilot analizy gotowego SAE znajduje się w
[`kaggle/notebooks/kaggle_gemma_scope2_270m.ipynb`](kaggle/notebooks/kaggle_gemma_scope2_270m.ipynb). Notebook
korzysta z tego samego źródła The Pile co notebook Pythii i uruchamia wyłącznie
analizę, bez treningu SAE.

Domyślna konfiguracja to:

- model `google/gemma-3-270m`,
- release SAELens `gemma-scope-2-270m-pt-res` (mapowany na katalog `resid_post`),
- warstwa `9`,
- SAE `layer_9_width_16k_l0_medium`,
- sekwencje długości `1024` tokenów.

Analiza Gemmy zapisuje stan w
`RESULTS_DIR/analysis/analysis_checkpoint_gemma_scope2_270m.pt`.
Po ponownym uruchomieniu notebook automatycznie odczytuje ten plik i pomija
już przetworzone batchy. Częstotliwość zapisu można zmienić przez
`CHECKPOINT_EVERY_BATCHES`, a pełny restart wymusić przez
`RESUME_ANALYSIS = False`.

Przy uruchamianiu nowej sesji checkpoint można wcześniej umieścić w Kaggle
Dataset i wskazać przez `CHECKPOINT_INPUT_PATH`. Notebook skopiuje go do
zapisywalnego
`/kaggle/working/.../analysis/analysis_checkpoint_gemma_scope2_270m.pt`; nie należy
ustawiać analizatora bezpośrednio na plik w `/kaggle/input`, bo checkpoint jest
aktualizowany podczas pracy.

Model bazowy jest podpinany do notebooka jako Kaggle Model
`google/gemma-3/transformers/gemma-3-270m/2` i ładowany bezpośrednio z
`/kaggle/input`. Każde konto uruchamiające analizę musi wcześniej zaakceptować
warunki Gemmy na stronie modelu w Kaggle. Notebook nie wymaga sekretu
`HF_TOKEN`; pipeline sprawdza dostęp do modelu przed wysłaniem nowej wersji.

Domyślnie `USE_PREPARED_SEQUENCES = True`: notebook wykorzystuje gotowe pliki
`tokens_seqs_padded_gemma.npy` i `attention_mask_gemma.npy` z datasetu
`/kaggle/input/datasets/erykmikoajek/trained-sae-models/` i przekazuje je
bezpośrednio do analizatora. Dzięki temu nie są kopiowane do `/kaggle/working`
ani ponownie liczone; opcję można wyłączyć, ustawiając `False`, aby przygotować
sekwencje bezpośrednio z The Pile.

Analizator ładuje SAE przez `sae-lens`, zachowuje granice sekwencji podczas
zbierania kontekstów i zapisuje:

- `analysis/feature_cards.jsonl`,
- `analysis/feature_analysis.json`,
- `analysis/feature_analysis.txt`.

Wyniki logit-lens są oznaczone jako przybliżenie: decoder resid-post jest
rzutowany przez macierz unembeddingu, bez pełnego uwzględnienia końcowej
RMSNorm Gemmy. Oficjalny model card Gemma Scope 2 opisuje release, strukturę
folderów i rekomendowane szerokości SAE.

Po zakończeniu analizy dalszą ścieżkę dla Gemmy — wraz z obowiązkowym
rozdzieleniem danych discovery/development/holdout — opisuje
[`reports/GEMMA_MOE_RESEARCH_AUDIT.md`](reports/GEMMA_MOE_RESEARCH_AUDIT.md).

Gotowe porównanie domenowego MoE z modelem bazowym na niezależnych źródłach
opisuje [`reports/GEMMA_MOE_INDEPENDENT_BENCHMARK.md`](reports/GEMMA_MOE_INDEPENDENT_BENCHMARK.md).
Notebook `kaggle/notebooks/kaggle_gemma_moe_benchmark.ipynb` korzysta obecnie z prywatnej paczki
`kaggle/assets/gemma_moe_benchmark_v3.zip`. Follow-up ma 600 nowych przykładów,
wyklucza rekordy v1/v2, zastępuje GSM8K arytmetyką elementarną, testuje kilka
kolejności odpowiedzi MC i mierzy implementacyjny koszt prefill.

Osobna analiza modeli dziedzinowych jest dostępna w
`evaluation/domain_expert_benchmark.py` oraz notebooku
`kaggle/notebooks/kaggle_gemma_domain_expert_benchmark.ipynb`. Każdy model
zastępuje MLP warstwy 9 jednym ekspertem działającym na wszystkich tokenach,
bez routera i fallbacku. Notebook porównuje jakość na własnej domenie, liczbę
parametrów, teoretyczne MAC oraz zmierzone latency, throughput i VRAM. Pełna
macierz cross-domain jest opcjonalna.

## Semantic Domain Triage

Po wytrenowaniu SAE można zbudować domeny semantyczne z logit lens. Dla
Pythia-160m użyj warstwy 6 i checkpointu zgodnego z tym modelem:

```bash
python3 -m domain_triage.semantic_domain_triage \
  --data-path data/pythia160m_sae_github_data \
  --checkpoint data/pythia160m_sae_github_data/models/checkpoints/topk_sae_layer_6_best_github_data.pt \
  --model-name EleutherAI/pythia-160m \
  --tokenizer-name EleutherAI/pythia-160m \
  --layer-num 6 \
  --output-dir data/pythia160m_sae_github_data/analysis/domain_triage \
  --top-m 128 \
  --logit-batch-size 256 \
  --svd-components 20 \
  --min-cluster-size 20 \
  --min-token-chars 4 \
  --require-token-boundary \
  --max-domain-fraction 0.35 \
  --samples-per-domain 25 \
  --context-analysis data/pythia160m_sae_github_data/analysis/features_analysis.json
```

Jeżeli wagi bazowego modelu nie są dostępne, parametr `--feature-analysis`
pozwala użyć zapisanych wyników logit lens z `features_analysis.json` bez
ponownego ładowania modelu. Gdy Pythia nie ma osobnego `validation.csv`,
`--context-analysis` buduje zestawy walidacyjne z kontekstów zachowanych przez
analizę cech. Są to próbki diagnostyczne związane z aktywacjami SAE, a nie
niezależny benchmark.

Dla checkpointu Pile-CC można wykonać świeży triage z lokalnie dostępnymi
wagami modelu następująco:

```bash
HF_HUB_OFFLINE=1 SAE_DEVICE=cpu python3 -m domain_triage.semantic_domain_triage \
  --data-path data/pythia160m_sae_pilecc \
  --checkpoint data/pythia160m_sae_pilecc/models/checkpoints/topk_sae_layer_6_best_pilecc.pt \
  --model-name EleutherAI/pythia-160m \
  --tokenizer-name EleutherAI/pythia-160m \
  --layer-num 6 \
  --output-dir data/pythia160m_sae_pilecc/analysis/domain_triage \
  --n-domains 5 --top-m 32 --svd-components 10 --min-cluster-size 10 \
  --min-token-chars 4 --require-token-boundary --max-domain-fraction 0.05 \
  --include-unobserved-features --skip-validation
```

`--skip-validation` należy stosować, gdy zapisany `features_analysis.json` nie
pochodzi z tego samego checkpointu i tego samego zbioru sekwencji. W takim
przypadku automatyczne nazwy domen są tylko proxy tokenowymi i wymagają świeżej
walidacji kontekstowej.

Wariant TinyStories pozostaje dostępny:

```bash
python3 -m domain_triage.semantic_domain_triage \
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

Wektor domenowy jest dodatkowo ważony przez TF–IDF, aby tokeny wspólne dla
wielu cech nie tworzyły pozornie semantycznych klastrów. Szerokie klastry są
pomijane przez `--max-domain-fraction`, a filtr `--require-token-boundary`
ogranicza wpływ kontynuacji BPE. Automatycznie wybrane nazwy są skrótami z
top tokenów i wymagają kontroli na podstawie `domain_report.md` oraz
kontekstów walidacyjnych przed użyciem w mapowaniu/pruningu.

Triage domyślnie zapisuje `downstream_approved=false`. Po ręcznej kontroli
istniejącego raportu bramkę można otworzyć bez ponownej klasteryzacji:

```bash
python3 -m domain_triage.semantic_domain_triage \
  --data-path data/gemma_scope2_270m_pilecc \
  --approve-existing-domains
```

Domyślnie skrypt wymaga przynajmniej 3 domen. Jeśli HDBSCAN znajdzie mniej, run kończy się błędem z diagnostyką. Do diagnostycznego zapisu słabych wyników można użyć:

```bash
python3 -m domain_triage.semantic_domain_triage \
  --checkpoint data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt \
  --allow-underfilled-domains
```

Wyniki trafiają do `data/tinystories_dataset/analysis/domain_triage/`, chyba że podasz `--output-dir`.

## MLP Mapping

Etap pruningowy wymaga prawdziwych neuronów MLP, nie 64-wymiarowego residual stream. Do tego służy:

```bash
python3 -m domain_mapping.domain_mlp_activation_mapping \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --checkpoint data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt \
  --layer-num 4
```

Skrypt rozpoznaje układy GPT-Neo, GPT-NeoX i gated-MLP Gemmy/Llamy. Przechwytuje
wektor wchodzący do projekcji wyjściowej, czyli odpowiednio zwykłą
post-aktywację albo iloczyn `act(gate_proj(x)) * up_proj(x)`. Domyślnie każda
domena jest mapowana na tej samej połączonej puli tekstów i z tym samym
wylosowanym podzbiorem tokenów.

Domyślnie mapping odrzuca domeny, których zagregowany sygnał SAE jest pusty. Do zapisu artefaktów diagnostycznych można jawnie dodać:

```bash
python3 -m domain_mapping.domain_mlp_activation_mapping \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --allow-zero-domain-signal
```

## Pruning Ekspertów

Po mappingu można zbudować sparse MLP-only ekspertów:

```bash
python3 -m domain_mapping.domain_mlp_pruning \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --tau-method per-domain-null \
  --null-permutations 100 \
  --null-percentile 95
```

Pruning:

- czyta `domain_mlp_mapping/*.npz` i korelacje CSV,
- liczy próg `tau` metodą null baseline albo kwantylem,
- buduje `keep_mask`,
- zeruje odpowiadające wiersze wszystkich projekcji wejściowych i kolumny
  projekcji wyjściowej MLP,
- zapisuje `domain_<id>_mlp_expert.pt`, `domain_<id>_mask.npy` i `domain_<id>_pruning.json`.

Eksperci z pustą maską są domyślnie blokowani. Jeśli celowo chcesz wyeksportować zdegenerowany artefakt diagnostyczny, użyj:

```bash
python3 -m domain_mapping.domain_mlp_pruning \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --empty-signal-policy prune-all \
  --allow-empty-experts
```

## Router Training

Po uzyskaniu niepustych ekspertów można wytrenować router domenowy. Router jest
liniowy (`nn.Linear(d_model, n_domains)`) i trenuje się na wejściu do MLP
wybranej warstwy. Podział train/validation zachowuje całe grupy źródłowych
sekwencji. Konteksty odzyskane z tej samej analizy SAE są domyślnie odrzucane;
`--allow-diagnostic-contexts` istnieje wyłącznie dla smoke testu.

```bash
python3 -m moe.router_training \
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
python3 -m moe.router_training \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --max-texts-per-domain 5 \
  --epochs 1 \
  --batch-size 2
```

## MoE Assembly

`moe/moe_assembly.py` udostępnia funkcje importowalne i prosty CLI do sprawdzenia, czy da się złożyć model:

```bash
python3 -m moe.moe_assembly \
  --mode moe \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --layer-num 4
```

Obsługiwane są dwa tryby:

- `single-expert` - podstawia jeden `domain_<id>_mlp_expert.pt` za MLP warstwy 4.
- `moe` - ładuje `router/router.pt` i `domain_mlp_experts/`, a następnie zastępuje MLP wrapperem `HardRoutedMLP`.

`HardRoutedMLP` wykonuje tylko ekspertów faktycznie wybranych dla tokenów.
Niska pewność routera kieruje token do bazowego MLP, a eksperci są w czasie
składania materializowani jako kompaktowe projekcje zawierające wyłącznie
zachowane neurony. Nie gwarantuje to speed-upu bez osobnego benchmarku
sprzętowego, ponieważ routing i nieregularne małe batche również mają koszt.

## PPL Validation

Pełne porównanie PPL:

```bash
python3 -m evaluation.moe_validation \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --experts-dir data/tinystories_dataset/analysis/domain_triage/domain_mlp_experts \
  --router-path data/tinystories_dataset/analysis/domain_triage/router/router.pt \
  --validation-csv data/tinystories_dataset/holdout_general.csv \
  --domain-eval-csv data/tinystories_dataset/holdout_domains.csv \
  --layer-num 4 \
  --output-dir data/tinystories_dataset/analysis/domain_triage/moe_validation
```

Wyjścia:

- `moe_validation/ppl_results.json`,
- `moe_validation/ppl_report.md`.

Smoke test na małej próbce:

```bash
python3 -m evaluation.moe_validation \
  --domain-dir data/tinystories_dataset/analysis/domain_triage \
  --max-texts-per-domain 5 \
  --max-general-texts 20 \
  --batch-size 2 \
  --allow-development-eval
```

## Co Jest Jeszcze Do Zrobienia

Najważniejsze dalsze prace po implementacji pierwszej wersji MoE:

1. Uruchomić pełny pipeline na 3-5 niepustych domenach i zapisać rzeczywiste wyniki PPL.
2. Przygotować ręcznie oznaczone, niezależne zbiory development i holdout.
3. Dodać baseline'y masek losowych i magnitude pruning oraz kilka seedów.
4. Zmierzyć czas, VRAM/RAM i liczbę parametrów kompaktowych ekspertów na GPU.

Te etapy powinny powstać dopiero po uzyskaniu 3-5 niepustych, sensownych domen semantycznych.
