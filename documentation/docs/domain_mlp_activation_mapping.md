# `domain_mapping/domain_mlp_activation_mapping.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/domain_mapping/domain_mlp_activation_mapping.py) · [domeny](semantic_domain_triage.md) · [następny etap: pruning](domain_mlp_pruning.md)

## Rola modułu

To kanoniczny etap wiążący domeny SAE z **fizycznymi neuronami pośrednimi
MLP**. Uruchamia się dopiero dla domen ręcznie dopuszczonych do dalszego
pipeline'u. Dla tych samych ważnych tokenów przechwytuje jednocześnie:

- `resid_post` bloku `[N,D]`, kodowany przez SAE do sygnału domeny;
- wektor wchodzący do projekcji wyjściowej MLP `[N,H]`, czyli post-aktywacje
  neuronów, które można bezpośrednio maskować w wagach.

Następnie liczy zależność każdej z `H` kolumn od skalarnego sygnału domeny.

## Obsługiwane rodziny MLP

| Rodzina | Projekcje wejściowe | Projekcja wyjściowa | Hookowany sygnał |
| --- | --- | --- | --- |
| GPT-Neo | `c_fc` | `c_proj` | wejście `c_proj`, po aktywacji |
| GPT-NeoX | `dense_h_to_4h` | `dense_4h_to_h` | wejście `dense_4h_to_h` |
| Gemma/Llama gated | `gate_proj`, `up_proj` | `down_proj` | `act(gate_proj(x)) × up_proj(x)` wchodzące do `down_proj` |

`describe_mlp()` zwraca nazwy projekcji i string `activation_source`, który jest
zapisywany w NPZ i summary. Nie zakłada na sztywno orientacji macierzy poza
semantyką projekcji; pruning później dopasowuje maskę po rozmiarze osi.

## Obsługiwane SAE

- lokalny `TopKSAE`, ładowany z checkpointu;
- SAE z SAELens, np. Gemma Scope 2, wskazany przez `--sae-release` i `--sae-id`.

Jeżeli `domains.json` powstał z analizy Gemma Scope, release i SAE ID mogą być
odczytane z `matrix_metadata.source_format`. Jawne argumenty mają priorytet.
Nie wolno jednocześnie podać lokalnego checkpointu i SAELens ID.

`encode_sae_features()` normalizuje API: dla SAELens wywołuje `encode()`, a dla
lokalnego SAE bierze drugi element `(reconstruction, feature_acts)`.

## Dwa hooki w jednym forward pass

1. Forward pre-hook na projekcji wyjściowej MLP przechwytuje post-aktywacje
   pośrednie przed ich zmieszaniem do residual stream.
2. Forward hook na całym bloku przechwytuje jego wyjście `resid_post`.
3. Uruchamiany jest sam backbone z `use_cache=False`, bez logitów słownika.
4. Oba tensory są filtrowane tą samą `attention_mask`, więc ich wiersze
   odpowiadają tym samym tokenom.
5. Residual jest kodowany przez SAE. Wybrane feature IDs domeny są redukowane
   przez `sum`, `mean` albo `max` do jednego sygnału.
6. MLP `[N,H]`, sygnał `[N]`, token IDs, sample IDs i pozycje są odcinane do
   `max_tokens_per_domain` i odkładane na CPU.
7. Oba hooki są usuwane w `finally`.

### Wspólny korpus mappingu

Domyślny `--mapping-corpus pooled-domains` łączy pliki walidacyjne wszystkich
wybranych domen. Każdy sygnał domenowy jest zatem korelowany z neuronami na
**tym samym przekroju tekstów**, zamiast wyłącznie na tekstach wcześniej
przypisanych do tej domeny. Ogranicza to selection bias i sprawia, że scores
między domenami są bardziej porównywalne.

Dla każdej domeny lista połączonych próbek powstaje w tej samej kolejności i
jest tasowana tym samym `random.Random(seed)`. Jeżeli
`max_tokens_per_domain` ucina korpus, wszystkie domeny widzą dzięki temu ten
sam deterministyczny podzbiór tokenów. `max_texts_per_domain` jest stosowane
osobno podczas czytania każdego pliku JSONL. Tryb
`--mapping-corpus target-domain` zachowuje starszy, selekcyjny wariant i służy
głównie do porównań diagnostycznych.

## Walidacja kontraktów

- `domains.json.downstream_approved` musi być prawdziwe. W przeciwnym razie
  mapping kończy się błędem; `--allow-unapproved-domains` jedynie ostrzega i
  oznacza świadome uruchomienie diagnostyczne.
- `get_sae_input_dim(sae)` musi odpowiadać `model.config.hidden_size`.
- Metadata SAELens, jeśli zawiera hook, musi wskazywać
  `blocks.<layer>.hook_resid_post`.
- Największy feature ID domen nie może przekroczyć szerokości SAE.
- Każda domena musi mieć plik walidacyjny.
- Po zebraniu danych `cluster_activation_nonzero_fraction` musi być co najmniej
  `min_cluster_nonzero_fraction` (domyślnie `1e-6`).

Pusty sygnał zwykle oznacza niezgodne domeny, SAE, warstwę albo dane.
`--allow-zero-domain-signal` zapisuje artefakt wyłącznie diagnostyczny i dodaje
warning; nie czyni go poprawnym wejściem do pruningowania.

## Korelacje

Moduł korzysta ze wspólnych `pearson_by_neuron()`,
`mutual_info_by_neuron()` i `build_correlation_rows()` z historycznego modułu.
Tym razem wejście ma rzeczywiście kształt `[N,H]`, więc `neuron_id` odpowiada
fizycznej osi pośredniej MLP.

Domyślnie liczony jest Pearson, ponieważ jest potrzebny do permutacyjnego progu
pruningu. MI można dołączyć przez `--metrics mutual_info`, ale jest znacznie
kosztowniejsze.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `pooled_mlp_activations.npz` | wspólne `[N,H]`, macierz sygnałów wszystkich domen, domain/sample/token IDs i metadata |
| `domain_<id>_mlp_activations.npz` | starszy wariant target-domain: `[N,H]`, pojedynczy sygnał SAE i koordynaty |
| `domain_<id>_mlp_neuron_correlations.csv` | miary i statystyki per neuron |
| `all_mlp_neuron_correlations.csv` | wszystkie domeny |
| `summary.json` | provenance SAE/modelu, korpus mappingu, ścieżki próbek, diagnostyka i top neurony |

`n_texts` jest liczbą unikalnych indeksów próbek reprezentowanych po limicie
tokenów, a nie zawsze liczbą wszystkich wczytanych tekstów.

## Katalog funkcji

| Funkcja | Odpowiedzialność |
| --- | --- |
| `get_mlp_module()` / `set_mlp_module()` | odczytuje lub podmienia `.mlp` danego bloku |
| `describe_mlp()` | rozpoznaje rodzinę, projekcje i fizyczną oś neuronów |
| `register_mlp_post_activation_hook()` | hookuje wejście projekcji wyjściowej MLP |
| `get_sae_width()` | normalizuje `d_sae` lokalnego SAE i SAELens |
| `get_sae_input_dim()` | normalizuje `d_model`/`cfg.d_in` |
| `encode_sae_features()` | wspólne API inferencji obu typów SAE |
| `load_sae_lens()` | ładuje pretrained SAE z kompatybilnością wersji API |
| `infer_sae_lens_source()` | odczytuje release/ID z provenance `domains.json` |
| `validate_downstream_approval()` | egzekwuje ręczne zatwierdzenie domen lub jawny override diagnostyczny |
| `validate_sae_resid_post_compatibility()` | sprawdza wymiar i hook layer/site |
| `collect_domain_mlp_activations()` | łączy wskazane JSONL, wykonuje dwa hooki, maskowanie, limit i dane per token |
| `collect_pooled_domain_mlp_activations()` | jeden forward i wspólna macierz MLP/sygnałów dla wszystkich domen |
| `save_domain_mlp_activations()` | zapisuje rozszerzony kontrakt NPZ |
| `save_pooled_domain_mlp_activations()` | zapisuje współdzielony artefakt bez duplikowania macierzy MLP |
| `summarize_domain_signal()` | średnia, std, max i frakcja niezerowa |
| `validate_domain_signal()` | blokuje pustą/zbyt rzadką domenę poza override |
| `parse_args()` / `main()` | rozwiązuje źródła SAE i wykonuje wszystkie domeny |

## Zużycie pamięci i ograniczenia

- Forward jest batchowany, lecz wszystkie wybrane tokeny domeny są na końcu
  przechowywane w RAM do korelacji. Domyślny limit 100 000 tokenów ogranicza
  macierz do około `100000 × H × 4` bajtów plus pozostałe tablice.
- Limit może przeciąć batch, a nawet tekst; metadata nadal jednoznacznie
  wskazuje zachowane tokeny.
- Przy `pooled-domains` liczba kandydatów przed limitem rośnie wraz z liczbą
  domen; jest to koszt większej porównywalności sygnałów.
- Próbki diagnostyczne pochodzące z analizy cech mogą zawyżać widoczną relację
  domena–neuron. Provenance walidacji trzeba uwzględnić w interpretacji.
- Korelacja token-po-tokenie ignoruje zależność obserwacji w tej samej
  sekwencji. Pruning częściowo reaguje na to blokową permutacją, ale nie jest to
  pełny model hierarchiczny.

## Wykonany mapping Gemmy

Finalny run użył agregacji `sum`, 351 tekstów i wspólnego limitu 60 000
tokenów. Powstała jedna macierz `[60000,2048]` oraz sześć kolumn sygnału SAE.
Usunięcie sześciu osobnych kopii macierzy zmniejszyło koszt RAM i dysku oraz
zagwarantowało identyczne obserwacje dla porównań domen. Sygnały zostały przed
pruningiem dodatkowo sprawdzone na poziomie tekstu przez
[`domain_mapping/validate_domain_sae_selectivity.py`](domain_selectivity.md).
