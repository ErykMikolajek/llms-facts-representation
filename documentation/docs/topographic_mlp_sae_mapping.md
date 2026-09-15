# `domain_mapping/topographic_mlp_sae_mapping.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/domain_mapping/topographic_mlp_sae_mapping.py) · [właściwe mapowanie neuronów MLP](domain_mlp_activation_mapping.md)

## Rola i status

Jest to eksploracyjny, historyczny etap mapujący zagregowany sygnał domeny SAE
na każdy wymiar hidden state po wybranym bloku. Udostępnia też wspólne helpery
używane przez nowsze moduły: ładowanie domen, agregację cech, korelację Pearsona,
mutual information i raportowanie.

!!! danger "Nazwa historyczna"
    `collect_domain_activations()` zapisuje
    `outputs.hidden_states[layer_num + 1]`, czyli **residual stream po bloku** o
    szerokości `D`. Zmienna i kolumna `mlp_activations` nie reprezentuje
    fizycznych neuronów pośrednich MLP o szerokości `H`. Tego wyniku nie należy
    używać do maskowania `c_fc/c_proj`. Do pruningowania służy
    [`domain_mapping/domain_mlp_activation_mapping.py`](domain_mlp_activation_mapping.md).

## Wejścia

- `domains.json` z listą feature IDs per domena;
- `domain_validation/domain_<id>.jsonl` z tekstami;
- lokalny checkpoint Top-K SAE;
- zgodny model i tokenizer.

`resolve_domain_dir()` preferuje `<data_path>/analysis/domain_triage`, potem
`other/domain_triage`, i zwraca pierwszą ścieżkę nawet, gdy jeszcze nie istnieje,
aby późniejszy błąd wskazywał oczekiwaną lokalizację.

## Przebieg zbierania aktywacji

1. Wszystkie teksty domeny są wczytywane do listy i batchowane.
2. Tokenizer stosuje padding i truncation do `max_length`.
3. Model jest wywoływany z `output_hidden_states=True`, a wybierany tensor to
   `hidden_states[layer_num+1]` (`+1`, ponieważ element 0 to embedding input).
4. Maska usuwa padding; powstaje `[N_valid,D]`.
5. TopKSAE koduje residual i zwraca `[N_valid,F]`.
6. Aktywacje cech należących do domeny są agregowane przez `sum`, `mean` albo
   `max`, tworząc skalarny sygnał `[N_valid]`.
7. Zapisywane są też token IDs, indeks próbki i pozycja tokenu.

## Miary zależności

### Pearson per wymiar

`pearson_by_neuron()` akumuluje w chunkach w float64 statystyki `Σx`, `Σx²`,
`Σxy`, `Σy`, `Σy²`, a następnie liczy:

```text
r_j = (Σ x_j y - Σx_j Σy/N) /
      sqrt((Σx_j² - (Σx_j)²/N)(Σy² - (Σy)²/N))
```

Brak wariancji daje `NaN`. Implementacja nie tworzy dodatkowej wycentrowanej
macierzy `[N,D]` i jest ponownie używana przez właściwy mapping MLP.

### Mutual information

`mutual_info_regression()` estymuje nieliniową zależność każdego wymiaru od
sygnału domeny metodą k-NN. `n_neighbors` jest ograniczane do `[1,N-1]`.
Stały sygnał albo mniej niż dwa wiersze daje same `NaN`.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `domain_<id>_activations.npz` | residual nazwany historycznie `mlp_activations`, sygnał SAE i koordynaty |
| `domain_<id>_neuron_correlations.csv` | Pearson/MI per wymiar `D` |
| `all_neuron_correlations.csv` | połączone domeny |
| `summary.json` | provenance, statystyki sygnału i top wymiary |

Domyślnie NPZ jest kompresowany. `--no-compress` zmniejsza koszt CPU kosztem
większego pliku.

## Katalog funkcji i klas

| Element | Odpowiedzialność |
| --- | --- |
| `DomainSpec` | ID, nazwa, feature IDs i opcjonalny label klastra |
| `default_checkpoint_path()` | standardowa ścieżka najlepszego TopKSAE |
| `load_sae_checkpoint()` | odtwarza lokalny SAE z configu i wag |
| `resolve_domain_dir()` | wybiera lokalizację artefaktów triage |
| `load_domains()` | filtruje domeny, waliduje niepuste listy cech |
| `iter_domain_samples()` | czyta poprawne teksty z JSONL z opcjonalnym limitem |
| `pearson_by_neuron()` | bounded Pearson dla każdej kolumny |
| `mutual_info_by_neuron()` | MI regression dla każdej kolumny |
| `aggregate_cluster_activations()` | redukuje wybrane cechy SAE do sygnału domeny |
| `make_batch_positions()` | tworzy macierze indeksów próbki i pozycji `[B,S]` |
| `collect_domain_activations()` | residual post-block + sygnał SAE + metadata |
| `save_domain_activations()` | zapisuje kontrakt NPZ |
| `build_correlation_rows()` | łączy miary i statystyki w rekord per wymiar |
| `write_correlation_csv()` | zapisuje stały schemat kolumn |
| `sort_key_for_metric()` | sortuje brakujące wyniki na końcu, Pearson po wartości bezwzględnej |
| `summarize_domain()` | top wymiary i rozkład sygnału domeny |
| `parse_args()` / `main()` | wykonuje mapping dla wybranych domen |

## Ograniczenia

- `output_hidden_states=True` materializuje stany wszystkich warstw i jest
  bardziej pamięciożerne niż hook używany w aktywnym pipeline.
- Wszystkie batch'e domeny są przechowywane w listach do końcowej konkatenacji;
  `max_texts_per_domain` jest jedynym limitem rozmiaru.
- Moduł obsługuje tylko lokalny `TopKSAE`, nie SAELens/Gemma Scope.
- Nazwy kolumn „neuron” są metodologicznie mylące dla residual dimensions.
  Summary zapisuje `activation_source`, dzięki czemu artefakt można rozpoznać.

