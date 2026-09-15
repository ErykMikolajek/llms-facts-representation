# `sae_pipeline/main.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/sae_pipeline/main.py) · [następny etap: sekwencjonowanie](dataset_sequencing.md)

## Rola modułu

`sae_pipeline/main.py` jest głównym CLI aktywnego pipeline'u własnego Top-K SAE. Nie
implementuje algorytmu uczenia, lecz scala konfigurację profilu z argumentami
użytkownika i deleguje pracę do trzech modułów:

- [`sae_pipeline/dataset_sequencing.py`](dataset_sequencing.md) — przygotowanie tokenów;
- [`sae_pipeline/autoencoder_training.py`](autoencoder_training.md) — strumieniowy trening;
- [`sae_pipeline/features_analysis.py`](features_analysis.md) — analiza albo finalizacja
  częściowego checkpointu analizy.

Skrypt celowo nie uruchamia triage domen, pruningowania ani MoE. Te etapy są
oddzielnymi eksperymentami z własnymi CLI.

## Dostępne etapy

| `--stage` | Działanie |
| --- | --- |
| `sequence` | tylko przygotowanie tablic tokenów i maski |
| `train-sae` | tylko trening; zakłada istniejące sekwencje |
| `analyze` | tylko analiza wskazanego/najlepszego checkpointu SAE |
| `finalize-analysis` | tworzy raporty z istniejącego checkpointu analizy bez ponownej inferencji na danych |
| `all` | sekwencjonowanie, następnie trening; nazwa nie obejmuje analizy |

!!! note "Znaczenie `all`"
    `all` wykonuje `sequence` i `train-sae`, ale nie `analyze`. Wynika to
    bezpośrednio z warunków w `main()` i warto uwzględnić to w skryptach
    uruchomieniowych.

## Profile

| Profil | Model i warstwa | `D → F`, `k` | Batchowanie | Epoki |
| --- | --- | --- | --- | --- |
| `tiny` | TinyStories-1M, warstwa 4 | `64 → 4096`, `k=8` | model 4, chunk 32, SAE 256 | 5 |
| `local-50gb` | Pythia-160M, warstwa 6 | `768 → 12288`, `k=64` (wymiar modelu wykrywany) | model 4, chunk 32, SAE 1024 | 2 |
| `colab` | Pythia-160M, warstwa 6 | `768 → 6144`, `k=64` | model 2, chunk 16, SAE 512 | 1 |

Wartości podane jawnie w CLI nadpisują pola profilu. Pola niezwiązane z
profilem — np. ścieżki checkpointów i tryb logowania — są dołączane bezpośrednio.

## Przebieg krok po kroku

1. `main()` wyłącza domyślnie równoległość Hugging Face tokenizers przez
   `TOKENIZERS_PARALLELISM=false`, o ile użytkownik nie ustawił tej zmiennej.
2. `_parser()` parsuje argumenty, a `_resolve_args()` scala je z profilem.
3. Pełna efektywna konfiguracja zostaje wypisana jako JSON — to prosty ślad
   reprodukowalności runu.
4. `_load_tokenizer()` ładuje tokenizer. Jeśli nie ma tokenu paddingu, jako
   `pad_token` wybiera EOS (lub UNK), a identyfikator zapisuje w `args`.
5. Dla `sequence`/`all` `_prepare_sequences()` wybiera jedną z trzech ścieżek:
   gotowe pliki `.npy`, źródło JSONL lub wejście CSV.
6. Dla `train-sae`/`all` `_train_sae()` ładuje model bazowy i przekazuje całą
   konfigurację do resumowalnego trenera.
7. Dla `analyze` `_analyze_sae()` rekonstruuje architekturę SAE z `config`
   checkpointu, ładuje wagi i uruchamia streamingową analizę.
8. Dla `finalize-analysis` `_finalize_analysis()` rekonstruuje SAE i model,
   lecz nie czyta tokenów ani aktywacji; odtwarza raport z akumulatorów w
   checkpointcie analizy.

## Rozstrzyganie danych wejściowych

Priorytet `_prepare_sequences()` jest następujący:

1. Jeżeli podano choć jedną z `--tokens-path`/`--attention-mask-path`, obie
   muszą wskazywać istniejące pliki. Są używane bez kopiowania i bez tokenizacji.
2. `--require-prepared-sequences` bez obu ścieżek kończy run błędem. Chroni to
   Kaggle przed przypadkowym, kosztownym przetworzeniem źródłowego JSONL.
3. `--input-path` wybiera streaming Pile-style JSONL.
4. W przeciwnym razie wybierany jest CSV z `--input-file` lub katalogu danych.

## Checkpoint analizy

W `analyze` domyślny checkpoint SAE to
`<data-path>/models/checkpoints/topk_sae_layer_<L>_best.pt`. Można go nadpisać
przez `--checkpoint-path`. Pole `usage_counts` jest przekazywane do analizatora,
dzięki czemu „dead feature” oznacza nieużycie w treningu, a nie tylko brak
obserwacji w wycinku analitycznym.

`--analysis-checkpoint-path`, `--no-analysis-resume` i
`--analysis-checkpoint-every-chunks` sterują osobnym checkpointem statystyk
analizy. `finalize-analysis` wymaga jawnego podania obu checkpointów, aby nie
połączyć przypadkowo niespójnych artefaktów.

## Katalog funkcji

| Funkcja | Odpowiedzialność |
| --- | --- |
| `_parser()` | definiuje etapy, profile oraz argumenty danych, modelu, treningu, analizy i wznowienia |
| `_resolve_args(args)` | kopiuje wybrany profil, nadpisuje jego niepuste pola i buduje finalny `Namespace` |
| `_load_tokenizer(args)` | ładuje tokenizer, ustala bezpieczny token paddingu i zapisuje `pad_token_id` |
| `_prepare_sequences(args, tokenizer)` | waliduje gotowe memmapy albo deleguje sekwencjonowanie JSONL/CSV |
| `_train_sae(args, tokenizer)` | ładuje model i uruchamia `train_autoencoder_streaming()`; `tokenizer` jest zachowany dla symetrii, lecz nie jest używany lokalnie |
| `_analyze_sae(args, tokenizer)` | odtwarza `TopKSAE` z checkpointu i uruchamia `analyze_sae()` |
| `_finalize_analysis(args, tokenizer)` | tworzy raporty z już zgromadzonych statystyk bez nowej inferencji po korpusie |
| `main()` | wykonuje dispatch etapów w ustalonej kolejności |

## Najważniejsze grupy argumentów

- **Źródło:** `--data-path`, `--input-file`, `--input-path`, `--file-pattern`,
  `--max-files`, `--max-documents`, `--max-tokens`.
- **Gotowe sekwencje:** `--tokens-path`, `--attention-mask-path`,
  `--require-prepared-sequences`.
- **Model i próbka:** `--model-name`, `--tokenizer-name`, `--layer-num`,
  `--seq-length`, `--max-sequences`, `--analysis-sampling`.
- **SAE:** `--d-model`, `--expansion-factor`, `--k`, `--batch-size-sae`,
  `--num-epochs`, `--learning-rate`, `--min-learning-rate`.
- **Ograniczenie pamięci:** `--model-batch-size`, `--chunk-sequences`.
- **Wznowienie:** `--checkpoint-path`, `--resume`/`--no-resume`,
  `--save-every-n-chunks`, `--keep-last-checkpoints`, `--max-history`.
- **Run kontrolny:** `--max-steps` zatrzymuje trening na granicy chunka.

## Warunki poprawności i ograniczenia

- Konfiguracja checkpointu SAE jest później sprawdzana przez trener; zmiana
  parametrów przy wznowieniu nie jest dopuszczona.
- `finalize-analysis` nadal ładuje pełny model bazowy, ponieważ logit lens
  potrzebuje macierzy unembeddingu, mimo że nie wykonuje forward passów danych.
- CLI nie zapisuje automatycznie konfiguracji do oddzielnego pliku. Jest ona
  jednak zawarta w checkpointach i wypisana do standardowego wyjścia.
- Nazwa parametru `_train_sae(..., tokenizer)` sugeruje użycie tokenizera, ale
  aktualna implementacja nie odwołuje się do niego. Nie wpływa to na wynik.

