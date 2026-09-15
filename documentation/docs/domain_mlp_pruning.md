# `domain_mapping/domain_mlp_pruning.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/domain_mapping/domain_mlp_pruning.py) · [wejście: mapping](domain_mlp_activation_mapping.md) · [konsument: assembly](moe_assembly.md)

## Rola modułu

Moduł przekształca korelacje domena–neuron w osobne, MLP-only sparse experts.
Dla każdej domeny wybiera próg `tau`, buduje boolowską maskę `H` i zeruje
parametry prunowanych neuronów w kopii MLP modelu bazowego.

Ekspert zachowuje pełne gęste tensory wag z zerami. Jest funkcjonalnie rzadki,
ale sam zapis nie daje automatycznie przyspieszenia sparse-kernel. Konsument
`moe/moe_assembly.py` używa zapisanej `keep_mask`, aby podczas ładowania
zmaterializować mniejsze gęste projekcje o wymiarze `H_kept`.

## Ładowanie mappingu

`load_mapping_summary()` preferuje `domain_mlp_mapping/summary.json`. Jeżeli
zapisane ścieżki bezwzględne nie istnieją po przeniesieniu artefaktów, próbuje
pliku o tej samej nazwie w bieżącym katalogu mappingu. Bez summary skanuje
`domain_*_mlp_neuron_correlations.csv` i rekonstruuje odpowiadające nazwy NPZ.

`load_neuron_scores()` tworzy wektor długości `H`, domyślnie `NaN`, i wypełnia
pozycje według `neuron_id`. Chroni przed ID spoza osi fizycznego MLP.

Przed odczytem scores `validate_selectivity_gate()` wymaga
`selectivity/selectivity_results.json` i potwierdza, że wszystkie wybrane
domeny przeszły test selektywności SAE. `--allow-failed-selectivity` jest
wyłącznie diagnostycznym override i pozostaje zapisany w provenance.

## Metody wyznaczania `tau`

### `per-domain-null` — zalecane dla `abs_pearson_r`

1. Skalarny sygnał domeny jest dzielony na ciągłe bloki po
   `null_block_size` tokenów.
2. Kolejność bloków jest permutowana; kolejność tokenów wewnątrz bloku zostaje.
3. Dla każdego neuronu ponownie liczony jest Pearson z przetasowanym sygnałem.
4. Z każdej permutacji zachowywane jest **maksymalne** `|r|` po wszystkich
   neuronach.
5. `tau` jest wskazanym percentylem rozkładu tych maksimów.

Max-statistic kontroluje rodzinne prawdopodobieństwo wybrania dowolnego neuronu
pod nullem znacznie lepiej niż pooling wszystkich neuronów/permutacji. Bloki
zachowują krótką autokorelację tokenową, choć nie respektują jawnie granic
sekwencji zapisanych w NPZ.

### `quantile`

`tau` jest percentylem empirycznych, skończonych scores neuronów. Metoda
gwarantuje względny wybór w obrębie domeny, ale nie stanowi testu względem
nulla. Nadaje się także do `mutual_info`.

!!! danger "Niezgodna kombinacja parametrów"
    Kod pozwala połączyć `--metric mutual_info` z
    `--tau-method per-domain-null`, ale null tau jest zawsze liczony z
    bezwzględnych korelacji Pearsona. Porównywanie wartości MI z progiem w skali
    `|r|` nie ma poprawnej interpretacji. Dla obecnej implementacji należy użyć
    `abs_pearson_r + per-domain-null` albo `mutual_info + quantile`.

## Budowa maski

`build_keep_mask()` zastępuje niefinity scores przez `-∞` i zachowuje neurony
`score ≥ tau`. Opcjonalne `min_keep_fraction` wymusza minimalną liczbę top
neuronów nawet wtedy, gdy próg statystyczny wybrał mniej. Domyślne `0` nie
wymusza potencjalnych false positives.

Pusta maska jest blokowana, chyba że `--allow-empty-experts` jawnie dopuszcza
artefakt diagnostyczny.

## Pusty sygnał domeny

Jeśli sygnał ma zerową wariancję albo nie istnieje żaden finite score:

- `error` (domyślnie) przerywa pracę;
- `keep-all` eksportuje nieprzycięty MLP jako jawny artefakt degeneracji;
- `prune-all` buduje pusty ekspert i wymaga dodatkowo `--allow-empty-experts`.

W takim przypadku `tau=None`, a metadata zachowuje powód i politykę.

## Maskowanie architektur

Dla `prune_mask = ~keep_mask`:

- w każdej projekcji wejściowej zerowana jest oś odpowiadająca output features
  (`preferowana oś 0`);
- w projekcji wyjściowej zerowana jest oś input features (`preferowana oś 1`);
- helper sprawdza rzeczywiste kształty i może wybrać drugą oś, jeśli biblioteka
  przechowuje macierz w innej orientacji;
- dla gated MLP maskowane są jednocześnie `gate_proj` i `up_proj`, a następnie
  `down_proj`;
- `--zero-bias` zeruje biasy projekcji wejściowych na prunowanych pozycjach;
  bias projekcji wyjściowej nie ma osi `H` i pozostaje bez zmian;
- wszystkie parametry eksperta są zamrażane.

Przed każdą domeną bazowy `state_dict` MLP jest ładowany ponownie, więc maski
nie kumulują się między ekspertami.

## Przebieg `main()`

1. Rozwiązuje domain/mapping/output directories i ładuje listę domen.
2. Ładuje model bazowy tylko raz, wybiera MLP i zachowuje jego stan CPU.
3. Dla domeny ładuje duże tablice aktywacji jako float32, sprawdza `H` względem
   projekcji wyjściowej i odczytuje scores z CSV. Wspólna funkcja Pearsona
   akumuluje potrzebne statystyki w float64 bez tworzenia drugiej pełnej kopii.
4. Obsługuje pusty sygnał albo estymuje `tau` wybraną metodą.
5. Buduje i waliduje maskę.
6. Odtwarza bazowe MLP, zeruje parametry i eksportuje artefakty.
7. Zapisuje wspólne `pruning_summary.json`.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `domain_<id>_mask.npy` | bool `[H]`, `True` oznacza neuron zachowany |
| `domain_<id>_mlp_expert.pt` | format `mlp_only_sparse_expert`: MLP state, maska, model/layer/domain i tau |
| `domain_<id>_pruning.json` | czytelne ID zachowane/przycięte, statystyki i provenance |
| `pruning_summary.json` | konfiguracja oraz sparsity wszystkich domen |

Checkpoint eksperta zawiera wyłącznie MLP, nie cały model językowy. Assembly
musi odtworzyć zgodny model bazowy i podstawić stan do właściwej warstwy.

## Katalog funkcji i klas

| Element | Odpowiedzialność |
| --- | --- |
| `DomainMapping` | ścieżki NPZ/CSV i identyfikacja domeny |
| `load_mapping_summary()` | czyta summary albo rekonstruuje mapping z nazw plików |
| `load_neuron_scores()` | mapuje CSV do gęstego wektora scores `[H]` |
| `load_activation_artifact()` / `select_domain_arrays()` | wspólny pooled NPZ i wybór kolumny domeny |
| `validate_selectivity_gate()` | obowiązkowa bramka AUC/CI przed pruningiem |
| `estimate_per_domain_null_tau()` | blokowa permutacja i max-statistic threshold |
| `estimate_quantile_tau()` | percentyl finite scores |
| `build_keep_mask()` | próg i opcjonalny minimalny odsetek |
| `build_empty_signal_mask()` | jawna maska keep-all/prune-all |
| `validate_keep_mask()` | blokuje 0-neuron expert poza diagnostyką |
| `_zero_by_inner_dim()` | dopasowuje maskę do osi wagi i zeruje ją in-place |
| `apply_mlp_mask()` | maskuje wszystkie właściwe projekcje danej rodziny |
| `tensor_state_dict_to_cpu()` | bezpieczna kopia parametrów na CPU |
| `save_json()` | zapis czytelnych metadanych |
| `load_domain_arrays()` | ładuje MLP/sygnał z NPZ w float64 |
| `export_domain_expert()` | zapis maski, MLP checkpointu i metadanych |
| `parse_args()` / `main()` | wykonuje pruning per domena |

## Ograniczenia metodologiczne

- Tokeny nie są niezależnymi obserwacjami; blokowy null zachowuje tylko
  lokalne fragmenty i nie wykorzystuje `sample_indices`/`positions` z NPZ.
- Wybór `min_keep_fraction > 0` może nadpisać brak istotności statystycznej.
- Kwantyl zawsze wybiera relatywnie wysokie neurony nawet przy globalnie słabym
  sygnale; powinien być traktowany jako heurystyka sparsity.
- Sam gęsty artefakt z zerami nie redukuje rozmiaru pliku ani FLOP-ów
  standardowego `Linear`; właściwe składanie modelu materializuje mniejsze
  operatory, lecz uzysk trzeba potwierdzić osobnym benchmarkiem.

## Decyzja w eksperymencie Gemma

Czysty próg null zachował 1272–1408 z 2048 neuronów, lecz na development
pojedyncze eksperty biomedycyny i sportu przekroczyły 10% regresji PPL.
Przed holdoutem przyjęto `min_keep_fraction=0.75`: wszystkie maski mają 1536
neuronów, czyli pruning 25%. Floor jest kompromisem zachowania funkcji, nie
wynikiem dodatkowego testu istotności.
