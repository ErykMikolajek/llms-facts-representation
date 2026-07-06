# llms-facts-representation

Repozytorium do pracy magisterskiej nad reprezentacją faktów i konceptów w małym modelu językowym. Obecny kierunek badawczy to **SAE-guided MoEfication**: wykorzystanie Top-K Sparse Autoencodera do znalezienia semantycznych domen, zmapowania ich na fizyczne neurony MLP i wycięcia wyspecjalizowanych ekspertów domenowych.

Model bazowy to `roneneldan/TinyStories-1M`, tokenizer to `EleutherAI/gpt-neo-125M`, a główna analizowana warstwa to warstwa `4`.

Szczegółowy opis architektury, przepływów i artefaktów znajduje się w [`architecture.md`](architecture.md).

## Aktualny Status

Projekt ma działający rdzeń interpretowalności:

- preprocessing TinyStories do stałych sekwencji tokenów,
- ekstrakcję hidden state z warstwy 4,
- trening Top-K SAE z ekspansją `64 -> 4096`,
- analizę cech SAE z przykładami aktywacji i logit lens,
- klastrowanie cech SAE przez HDBSCAN w domeny semantyczne,
- budowę walidacji per domena,
- mapowanie prawdziwych post-aktywacji MLP na zagregowany sygnał domen SAE,
- pruning wag `c_fc` i `c_proj` do MLP-only ekspertów domenowych.

Projekt nie ma jeszcze pełnej architektury MoE:

- nie ma wytrenowanego routera liniowego,
- nie ma podmiany warstwy MLP na zespół ekspertów z hard-routingiem Top-1,
- nie ma walidacji perplexity dla modelu bazowego, pojedynczych ekspertów i złożonego MoE.

Ostatnio zaobserwowany lokalny stan artefaktów w `other/domain_triage/` jest częściowo niezgodny z wymaganiami: HDBSCAN wybrał tylko 2 domeny, jedna domena ma zerowy sygnał SAE na walidacji, a jej pruning prowadził do pustego eksperta. Kod został wzmocniony tak, aby takie przypadki były teraz błędem domyślnym, a nie cichym sukcesem.

## Struktura Kodu

- `main.py` - uruchamia podstawowy pipeline: sequencing, aktywacje, SAE, analiza cech.
- `dataset_sequencing.py` - CSV TinyStories do `tokens_seqs_padded.npy`.
- `activations_collecting.py` - forward modelu i zapis `outputs.hidden_states[layer_num + 1]`.
- `autoencoder_training.py` - implementacja i trening `TopKSAE`.
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

Wymagane biblioteki obejmują `torch`, `transformers`, `datasets`, `numpy`, `scipy`, `scikit-learn`, `hdbscan`, `pandas`, `syntok`, `pick` i `tqdm`.

## Pipeline Podstawowy

```bash
python3 main.py
```

Domyślna konfiguracja w `main.py`:

- model: `roneneldan/TinyStories-1M`,
- tokenizer: `EleutherAI/gpt-neo-125M`,
- dane: `data/tinystories_dataset`,
- długość sekwencji: `256`,
- warstwa: `4`,
- frakcja datasetu: `0.01`,
- Top-K SAE: `d_model=64`, `expansion_factor=64`, `k=8`.

Główne artefakty:

- `data/tinystories_dataset/sequenced/tokens_seqs_padded.npy`,
- `data/tinystories_dataset/activations/activations_layer_4.npy`,
- `data/tinystories_dataset/models/checkpoints/topk_sae_layer_4_best.pt`,
- `data/tinystories_dataset/analysis/features_analysis.txt`,
- `data/tinystories_dataset/analysis/features_analysis.json`.

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