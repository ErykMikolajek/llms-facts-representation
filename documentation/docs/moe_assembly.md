# `moe/moe_assembly.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/moe/moe_assembly.py) · [eksperci](domain_mlp_pruning.md) · [router](router_training.md) · [walidacja](moe_validation.md)

## Rola modułu

Moduł składa model bazowy z MLP-only ekspertami w jednym z dwóch trybów:

- `single-expert` — jeden sparse MLP zastępuje MLP wybranej warstwy;
- `moe` — `HardRoutedMLP` wybiera per token eksperta domenowego albo bazowy
  MLP jako fallback przy niskiej pewności.

Złożenie zachodzi in-memory. CLI sprawdza, czy model da się poprawnie utworzyć,
ale nie zapisuje całego złożonego modelu na dysku.

Możliwy jest także start przez `--bundle`. Manifest zamraża wszystkie ścieżki,
hashe, próg i `expert_source`; w tym trybie override progu jest blokowany.

## Ładowanie artefaktów

### Ekspert

`load_mlp_expert()` wymaga formatu `mlp_only_sparse_expert`, zgodnego
`layer_num`, `model_name` i obecności `mlp_state_dict`. `clone_mlp_with_state()`
głęboko kopiuje architekturę bazowego MLP, ładuje stan eksperta, przenosi na
urządzenie, przełącza do eval i zamraża parametry.

Artefakt z pruningu nadal przechowuje pełny gęsty stan z wyzerowanymi
neuronami, ale podczas składania `compact_mlp_from_artifact()` materializuje z
niego mniejszy moduł. `keep_mask` wyznacza:

- zachowane wiersze każdej projekcji wejściowej MLP;
- zachowane kolumny projekcji wyjściowej;
- zredukowany wymiar pośredni `H_kept`.

`_sliced_linear()` kopiuje odpowiedni wycinek wagi i biasu do nowego
`nn.Linear`. `CompactPrunedMLP` odtwarza dokładną kolejność operacji dla
GPT-Neo, GPT-NeoX i gated MLP; dla wariantu bramkowanego zachowuje obie
projekcje `gate_proj`/`up_proj` i ich iloczyn. Pusta maska nie może zostać
zmaterializowana jako kompaktowy ekspert.

### Router

`load_router_checkpoint()` wymaga formatu `linear_domain_router`, rekonstruuje
`LinearDomainRouter(D,K)` i zamraża go. Dodatkowo
`validate_router_model_compatibility()` porównuje `D`, warstwę i nazwę modelu.

### Bank ekspertów

Eksperci są ładowani w kolejności `domain_ids` zapisanej przez router i
umieszczani w `ModuleDict` pod kluczami będącymi **indeksami klas** (`"0"`,
`"1"`, ...), nie domain IDs. Dzięki temu `argmax` routera bezpośrednio wskazuje
moduł.

Pusty ekspert jest odrzucany na podstawie `pruning_summary.json` albo samej
`keep_mask`, chyba że jawnie dopuszczono artefakty diagnostyczne. Niepuste
eksperckie MLP są zawsze konwertowane do postaci kompaktowej przed włożeniem
do banku.

`expert_source=base_masks` nie ładuje pełnych zerowanych `state_dict`. Funkcja
`load_mask_expert_bank()` czyta boolowskie `.npy`, waliduje je wobec
`pruning_summary.json` i wycina projekcje bezpośrednio z bieżącego bazowego
MLP. Jest to lekki wariant używany przez `moe_bundle.json`; zachowuje dokładnie
te same tensory co evaluated artifact path, o ile model bazowy i maski mają
zgodne hashe.

## `HardRoutedMLP.forward()` — dokładny przebieg

1. Hidden states o dowolnym prefiksie kształtu, zwykle `[B,S,D]`, są
   spłaszczane do `[N,D]`.
2. Router działa w float32 i zwraca `[N,K]`.
3. Softmax zamienia logity na prawdopodobieństwa; `max` daje confidence i
   indeks klasy dla każdego tokenu.
4. Tokeny z `confidence < confidence_threshold` tworzą maskę fallbacku.
5. Dla fallbacku wykonywany jest oryginalny, nieprzycięty MLP tylko na tych
   tokenach.
6. Dla każdej klasy ekspert jest wykonywany wyłącznie na tokenach przypisanych
   do niej i pewnych; wyniki trafiają do odpowiednich wierszy outputu.
7. Aktualizowane są liczniki routingu, logity ostatniego forwardu i mapa ID;
   fallback otrzymuje `last_expert_ids=-1`.
8. Output jest odtwarzany do oryginalnego kształtu.

Przed forwardem ewaluator może wywołać `set_routing_token_mask()`. Maska nie
zmienia predykcji ani dispatchu; usuwa wyłącznie padding z liczników. W
`last_expert_ids` padding ma wartość `-2`.

Implementacja łączy dwa źródła oszczędności: nie liczy każdego eksperta na
całym batchu, a niepuste eksperty mają fizycznie zmniejszony wymiar
`H → H_kept`. Teoretyczny koszt projekcji eksperta spada zatem wraz z liczbą
zachowanych neuronów. Rzeczywisty czas zależy jednak także od narzutu routera,
indeksowania boolowskiego, małych nieregularnych batchy i częstości fallbacku.

## Fallback pewności

Domyślny próg wynosi `0.6`. Jest to mechanizm bezpieczeństwa: router o płaskim
rozkładzie nie wymusza przypadkowego eksperta, tylko zachowuje zachowanie modelu
bazowego dla danego tokenu. `routing_stats()` raportuje liczbę i udział tokenów
per ekspert oraz fallback.

Finalny próg jest częścią checkpointu routera i bundle, a wrapper może go
otrzymać jawnie. Override bez bundle służy analizie development; nie wolno go
dostrajać na zużytym holdoucie.

## Tryb pojedynczego eksperta

`build_single_expert_model()` ładuje jeden stan, materializuje kompaktowy MLP
dla niepustej maski i bezpośrednio podmienia `layer.mlp`. Nie sprawdza
`pruning_summary.json`; waliduje natomiast format/model/warstwę. Dla pustej
maski pozostawia gęsty, całkowicie wyzerowany artefakt diagnostyczny, dlatego
provenance nadal trzeba kontrolować poza tą funkcją.

## Katalog klas i funkcji

| Element | Odpowiedzialność |
| --- | --- |
| `get_module_device()` | odczytuje urządzenie pierwszego parametru, fallback CPU |
| `load_mlp_expert()` | waliduje format, model, warstwę i state dict |
| `load_pruning_summary()` | mapuje domain ID do summary, opcjonalnie pusty wynik |
| `load_mask_expert_bank()` | kompaktowi eksperci bezpośrednio z base MLP i masek |
| `clone_mlp_with_state()` | klonuje bazową strukturę i ładuje zamrożone wagi |
| `_sliced_linear()` | tworzy mniejszy `nn.Linear` z wybranych wierszy/kolumn i biasu |
| `CompactPrunedMLP.__init__()` | waliduje maskę i rekonstruuje kompaktowe projekcje danej rodziny MLP |
| `CompactPrunedMLP.forward()` | zachowuje oryginalną kolejność aktywacji, projekcji i dropout |
| `compact_mlp_from_artifact()` | sprawdza rodzinę i materializuje ekspert z `keep_mask` |
| `load_router_checkpoint()` | odtwarza i zamraża liniowy router |
| `load_expert_bank()` | mapuje kolejność klas routera na ekspertów i blokuje puste |
| `HardRoutedMLP.__init__()` | waliduje liczność i próg, rejestruje router/ekspertów/fallback |
| `HardRoutedMLP.forward()` | pewność, selektywna inferencja i scalenie tokenów |
| `set_routing_token_mask()` | wyklucza padding tylko z telemetrii |
| `reset_routing_stats()` | zeruje skumulowane liczniki |
| `routing_stats()` | tworzy serializowalny rozkład routingu |
| `validate_router_model_compatibility()` | sprawdza `D`, layer i model name |
| `build_single_expert_model()` | podmienia MLP jednym ekspertem |
| `build_hard_routed_moe()` | ładuje router/bank/fallback i podmienia warstwę wrapperem |
| `sha256_file()` / `load_moe_bundle()` | weryfikacja manifestu i zależności runtime |
| `parse_args()` / `main()` | CLI smoke testu obu trybów |

## Ograniczenia

- Artefakty `.pt` mają nadal gęsty storage; redukcja pamięci i FLOP-ów pojawia
  się dopiero po materializacji `CompactPrunedMLP` w pamięci.
- Kod nie benchmarkuje latencji ani pamięci. Mniejsza liczba FLOP-ów nie
  gwarantuje przyspieszenia przy małych grupach tokenów i narzucie routingu.
- Hard routing przez `argmax` jest niedyferencjowalny; moduł służy inferencji,
  nie wspólnemu fine-tuningowi routera i ekspertów.
- Router i eksperci są całkowicie zamrożeni, a fallback zachowuje bazowe MLP.
- Liczniki routingu są stanem diagnostycznym na CPU i kumulują kolejne forwardy
  aż do `reset_routing_stats()`.
- `last_router_logits` może być duże dla długiego batcha; przechowywany jest
  tylko ostatni forward, ale kopia trafia na CPU.
