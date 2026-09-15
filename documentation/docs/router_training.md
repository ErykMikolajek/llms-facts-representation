# `moe/router_training.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/moe/router_training.py) · [źródło domen](semantic_domain_triage.md) · [złożenie MoE](moe_assembly.md)

## Rola modułu

Moduł uczy liniowy router domenowy na hidden states wchodzących do MLP wybranej
warstwy. Dla każdego tokenu router zwraca logity klas domen:

```text
router(h) = W_router h + b_router,  h ∈ R^D
```

Model językowy jest zamrożony. Trenowane są tylko parametry `nn.Linear(D,K)`,
gdzie `K` to liczba wybranych domen.

## Dane uczące i ochrona przed przeciekiem

Teksty pochodzą z `domain_validation/domain_<id>.jsonl`. Każdy tekst otrzymuje
klasę odpowiadającą domenie, a każdy ważny token tekstu dziedziczy tę samą
etykietę. `RouterSample` zachowuje również:

- `group_id` — wspólne źródło, np. `sequence:<id>`;
- `diagnostic_only` — informację, że tekst pochodzi z analizy cech, a nie
  niezależnego zbioru.

`load_router_samples()` normalizuje tekst i liczy SHA-1 wyłącznie jako
identyfikator deduplikacji. Rekordy `diagnostic_only=True` są domyślnie
pomijane; `--allow-diagnostic-contexts` włącza je wyłącznie do circular smoke
testu. Następnie funkcja usuwa:

- ten sam tekst występujący w więcej niż jednej domenie;
- grupę źródłową należącą do więcej niż jednej domeny;
- duplikaty tekstu wewnątrz tej samej domeny.

Każda domena musi mieć co najmniej `min_samples_per_domain` tekstów, a router
wymaga przynajmniej dwóch domen.

## Group-aware split

`split_samples_by_domain()` grupuje próbki osobno per domena i `group_id`.
Losuje całe grupy, nie pojedyncze teksty. Liczba grup validation to w
przybliżeniu `round(groups × val_fraction)`, ale co najmniej 1 i mniej niż
wszystkie. Domena z mniej niż dwiema niezależnymi grupami powoduje błąd.

Po podziale `main()` jeszcze raz sprawdza przecięcie zbiorów group IDs i
przerywa run przy jakimkolwiek przecieku.

!!! note "Co chroni grupowanie"
    Kilka kontekstów z tej samej sekwencji może być bardzo podobnych. Trzymanie
    ich po jednej stronie splitu zapobiega prostemu zapamiętaniu źródła. Nie
    usuwa jednak zależności wynikającej z tego, że domeny i diagnostyczne teksty
    mogły powstać z tej samej analizy SAE.

## Pozyskiwanie wejścia routera

`register_mlp_input_hook()` rejestruje forward pre-hook na całym module MLP.
Pierwszy argument wywołania MLP jest residual hidden state wchodzącym do MLP.

Dla batcha tekstów `extract_router_inputs()`:

1. tokenizuje z paddingiem i truncation;
2. uruchamia zamrożony model bez gradientu;
3. odczytuje hookowany tensor `[B,S,D]`;
4. rozszerza klasę tekstu do `[B,S]`;
5. maskuje padding i zwraca `[N_valid,D]` oraz `[N_valid]`.

W finalnej ścieżce `cache_router_features()` wykonuje backbone dokładnie raz
dla train, validation i danych ogólnych. Ważne tokeny są przenoszone na CPU w
FP16 wraz z etykietą, globalnym `sample_id` i wagą. Pełny hidden tensor nie
jest utrzymywany po zakończeniu ekstrakcji.

## Epoka treningowa

`run_cached_router_epoch()` działa na zbuforowanych cechach i przełącza się w
tryb treningowy, jeśli przekazano optimizer:

1. pobiera batch tokenowych cech FP16 z CPU i konwertuje go do float32;
2. oblicza logity liniowego routera dla wszystkich ważnych tokenów;
3. używa multiclass cross-entropy;
4. w treningu wykonuje zero grad, backward i krok AdamW;
5. aktualizuje macierz pomyłek per token;
6. waży każdy token przez `1/liczba_tokenów_tekstu`, więc każdy tekst wnosi
   łączną wagę jeden.

Metryki z macierzy:

- accuracy i micro-F1 (dla single-label multiclass są równe);
- precision, recall, F1 i support per domena;
- macro-F1 jako nieważona średnia F1 domen.

Po wyborze epoki `sample_level_router_metrics()` agreguje predykcje głosowaniem
tokenów i raportuje również accuracy/macro-F1 na poziomie tekstu.

## Kalibracja fallbacku

Router nie ma klasy ogólnej. `calibrate_confidence_threshold()` ocenia maksimum
softmax na osobnych tekstach ogólnych i wybiera próg ograniczający odsetek
pewnego routingu do `target_general_route_rate`. Jednocześnie raportuje
coverage i accuracy na danych domenowych. Próg jest zapisywany w checkpointcie
i przy braku jawnego override automatycznie używany przez assembly.

## Wybór modelu i artefakty

Po każdej epoce liczona jest walidacja. Najlepszy stan to pierwsza epoka z
najwyższym `val_macro_f1`; remis nie zastępuje wcześniejszego stanu. Na końcu
router jest cofany do najlepszych wag.

| Plik | Zawartość |
| --- | --- |
| `router/router.pt` | format `linear_domain_router`, wagi, kolejność domen, model/layer i metrics |
| `router/router_metrics.json` | pełna historia train/validation i provenance |
| `router/router_report.md` | skrót najlepszej epoki i metryki per domena |

Kolejność `domain_ids` w checkpointcie definiuje mapowanie `class_idx →
domain_id` i musi być zachowana podczas ładowania expert banku.

## Katalog klas i funkcji

| Element | Odpowiedzialność |
| --- | --- |
| `RouterSample` | tekst, domena, indeks klasy, source group i flaga diagnostyczna |
| `DomainTextDataset.__init__()` | kopiuje sekwencję `RouterSample` do listy |
| `DomainTextDataset.__len__()` / `__getitem__()` | udostępnia długość i pojedynczą próbkę dla DataLoadera |
| `LinearDomainRouter.__init__()` | tworzy pojedynczą warstwę `nn.Linear(D,K)` |
| `LinearDomainRouter.forward()` | zwraca logity domen dla ostatniego wymiaru hidden states |
| `collate_router_samples()` | listy tekstów/metadanych i tensor klas |
| `load_router_samples()` | ładowanie, deduplikacja, konflikt domen i minimalne liczności |
| `split_samples_by_domain()` | stratyfikowany podział całych source groups |
| `register_mlp_input_hook()` | przechwytuje wejście MLP |
| `extract_router_inputs()` | forward modelu i spłaszczenie ważnych tokenów |
| `cache_router_features()` | jednorazowy backbone, FP16 CPU, sample IDs i text-balanced weights |
| `run_cached_router_epoch()` | właściwy trening/eval na cache |
| `predict_cached_router()` | batched softmax dla zbuforowanych cech |
| `sample_level_router_metrics()` | głosowanie tokenów i metryki tekstów |
| `calibrate_confidence_threshold()` | próg route rate na tekstach ogólnych |
| `load_general_calibration_samples()` | odczyt osobnego development general |
| `update_confusion()` | inkrementuje macierz target–prediction |
| `metrics_from_confusion()` | accuracy, macro/micro F1 i metryki klas |
| `run_router_epoch()` | train/eval cross-entropy i akumulacja wyników |
| `write_router_report()` | raport Markdown najlepszej walidacji |
| `save_router_checkpoint()` | zapis kontraktu routera i kolejności domen |
| `parse_args()` / `main()` | pełny trening i selekcja najlepszego stanu |

## Ograniczenia metodologiczne

- Etykieta tekstu jest nadawana każdemu tokenowi, nawet tokenom neutralnym.
  Router uczy się więc rozpoznawać globalny kontekst lokalnych residuali, a nie
  „prawdziwą domenę” każdego pojedynczego tokenu.
- Każdy tekst ma równą łączną wagę, ale klasy z większą liczbą tekstów nadal
  mogą mieć większy wpływ; nie ma jawnych class weights.
- Cache FP16 ogranicza transfer i pamięć, lecz wprowadza małe zaokrąglenie
  względem treningu bezpośrednio na hidden states modelu.
- Trening nie ma schedulera, early stopping ani resume checkpointu.
- Pole `independent_evaluation` w metrics jest obliczane wyłącznie jako brak
  rekordów z `diagnostic_only=True`. Zewnętrzny CSV etykietowany leksykonem
  domen może mieć tę flagę fałszywie uspokajającą; ostateczną niezależność
  ustala osobny ręczny holdout używany przez `evaluation/moe_validation.py`.

## Wynik wykonanego treningu

Podział obejmował 860 tekstów train i 220 validation. Token accuracy/macro-F1
wyniosły 0,9602/0,9355, a sample accuracy/macro-F1 0,9864/0,9861. Próg
0,8246666193 dał 5% routingu na 200 tekstach ogólnych oraz 77,19% coverage i
99,83% trafności wśród routowanych tokenów domenowych.
