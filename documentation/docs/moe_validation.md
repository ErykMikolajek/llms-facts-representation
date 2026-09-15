# `evaluation/moe_validation.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/evaluation/moe_validation.py) · [złożenie modelu](moe_assembly.md)

## Rola modułu

Moduł porównuje causal language-model loss i perplexity dla:

1. niezmienionego modelu bazowego;
2. każdego pojedynczego eksperta na jego własnej domenie;
3. hard-routed MoE na zbiorze ogólnym i każdej domenie.

Dodatkowo zapisuje statystyki wyboru ekspertów/fallbacku MoE.
Domyślny pełny przebieg ocenia także dwie maski kontrolne o identycznej liczbie
zachowanych neuronów: losową oraz magnitudową.

## Zbiory ewaluacyjne i wymuszona niezależność

- **Domenowy holdout:** ręcznie etykietowany `--domain-eval-csv` z kolumnami
  `domain_id,text`. Nieznane lub niewybrane domain IDs są pomijane, puste
  teksty odrzucane, a brak choć jednej próbki dla dowolnej ocenianej domeny
  powoduje błąd.
- **Ogólny holdout:** `validation.csv` z kolumną `text`.

Ogólny CSV jest wymagany nawet wtedy, gdy użytkownik interesuje się głównie
domenami. Limity `max_texts_per_domain` i `max_general_texts` tworzą smoke testy.

`validate_evaluation_independence()` domyślnie blokuje ewaluację, jeżeli nie
podano osobnego domenowego holdoutu. Dla obu CSV sprawdza też, czy ścieżka nie
leży pod `domain_validation/` oraz czy opcjonalna kolumna `diagnostic_only` nie
zawiera wartości `1`, `true`, `yes` lub `y`. Dodatkowo porównuje rozwiązaną
ścieżkę ogólnego CSV z `domains.json.validation_source` i blokuje ponowne
użycie danych, które uczestniczyły w triage domen.

!!! danger "Override wyłącznie do developmentu"
    `--allow-development-eval` pozwala wrócić do
    `domain_validation/domain_<id>.jsonl` i/lub powtórnie użyć CSV z triage,
    ale drukuje ostrzeżenie. Taki wynik jest circular smoke testem, nie
    niezależną oceną generalizacji i nie powinien być raportowany jako holdout.

## Obliczanie causal LM loss

`compute_causal_lm_ppl()`:

1. tokenizuje batch z paddingiem i truncation;
2. kopiuje `input_ids` jako labels;
3. ustawia padding labels na `-100`;
4. znajduje pierwszy ważny token każdego tekstu i również ustawia jego label
   na `-100`;
5. liczy ważne cele po przesunięciu jako `(labels[:,1:] != -100).sum()`;
6. wywołuje model z labels i mnoży średni loss batcha przez liczbę celów;
7. agreguje NLL po tokenach, a na końcu liczy `exp(mean_loss)`.

Wyłączenie pierwszego ważnego labelu ma znaczenie przy lewym paddingu:
zapobiega sytuacji, w której logit na pozycji paddingu ma przewidywać pierwszy
rzeczywisty token. Przy prawym paddingu pierwszy label i tak nie jest celem po
standardowym przesunięciu, więc operacja jest bezpieczna.

Jeśli `mean_loss > 700`, PPL jest ustawiane na `inf`, aby uniknąć overflow.
Pusty zbiór zwraca `NaN` i zerowe liczności.

## Kolejność ewaluacji

### Model bazowy

Model jest ładowany raz, oceniany na general set i kolejno na każdej domenie.

### Pojedynczy eksperci

Dla każdej domeny model bazowy jest ładowany od nowa, a jej ekspert zastępuje
MLP. Domyślnie oceniany jest tylko na własnej domenie. Flaga
`--evaluate-experts-on-general` dodaje ocenę ogólną; kod nie wykonuje pełnej
macierzy cross-domain expert × domain.

### MoE

Model jest ładowany raz i składany z routerem, expert bankiem i fallbackiem.
Przed general set oraz każdą domeną liczniki routingu są zerowane. Do metrics
danego zbioru dołączany jest udział tokenów per expert i fallback.
Przed każdym forwardem `attention_mask` jest przekazywana do wrappera tylko w
celu telemetrii; padding nie wpływa na routing stats.

### Kontrole dopasowane budżetem

`make_matched_control_masks()` losuje `n_keep` unikalnych neuronów oraz wybiera
`n_keep` największych wyników magnitudowych. Dla gated MLP score łączy normy
obu projekcji wejściowych z normą odpowiedniej kolumny wyjściowej. Maski są
materializowane przez ten sam compact-MLP path co ekspert semantyczny.

`--components base single controls moe` pozwala uruchamiać podzbiory oceny,
np. telemetry-only rerun bez ponownego wykonywania pozostałych komponentów.

## Summary

General regression jest względną zmianą:

```text
(PPL_moe_general - PPL_base_general) / PPL_base_general
```

`general_regression_within_threshold` sprawdza, czy zmiana nie przekracza
`max_general_ppl_regression` (domyślnie 10%). Dla każdej domeny summary podaje
PPL trzech wariantów i boolowską informację, czy ekspert/MoE poprawia bazę.

Niższa PPL oznacza lepsze przewidywanie tylko wtedy, gdy wszystkie modele są
ocenione dokładnie na tych samych tokenach, co zapewnia wspólna funkcja
tokenizacji i maskowania.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `moe_validation/ppl_results.json` | config, wszystkie surowe metryki, routing i summary |
| `moe_validation/ppl_report.md` | tabela general i domen oraz próg regresji |

## Katalog funkcji

| Funkcja | Odpowiedzialność |
| --- | --- |
| `iter_texts_from_domain_jsonl()` | wyciąga teksty przez wspólny iterator domen |
| `iter_texts_from_validation_csv()` | strumieniuje niepuste `text` z limitem liczby zwróconych |
| `load_labeled_domain_eval_csv()` | ładuje ręcznie etykietowany holdout i wymusza pokrycie domen |
| `batched()` | dzieli sekwencję tekstów na listy batchy |
| `setup_tokenizer()` | ładuje tokenizer i ustala EOS jako padding, jeśli potrzebne |
| `load_base_model()` | ładuje model, ustala `pad_token_id`, przenosi i ustawia eval |
| `compute_causal_lm_ppl()` | token-weighted NLL i perplexity |
| `load_eval_texts()` | wybiera niezależny domenowy CSV lub diagnostyczne JSONL oraz ładuje general CSV |
| `validate_evaluation_independence()` | blokuje brak holdoutu, ścieżki/wiersze diagnostyczne i reuse źródła triage bez override |
| `evaluate_base_model()` | baza na general i wszystkich domenach |
| `evaluate_single_experts()` | świeży model per own-domain expert, opcjonalnie general |
| `mlp_neuron_magnitude_scores()` | heurystyczny score wag per fizyczny neuron |
| `make_matched_control_masks()` | losowa i magnitudowa maska o tym samym `n_keep` |
| `build_masked_control_model()` / `evaluate_matched_controls()` | materializacja i PPL kontroli |
| `evaluate_moe_model()` | złożenie MoE, PPL i osobne routing stats per zbiór |
| `build_summary()` | względna regresja general i porównania domenowe |
| `write_ppl_report()` | zapis skróconego Markdown |
| `parse_args()` / `main()` | pełna kolejność ewaluacji i zapis JSON |

## Ograniczenia metodologiczne i techniczne

- PPL holdoutu nie ma bootstrap confidence intervals; boolowskie
  „improves” może reagować na szum małej próbki.
- Teksty są ucinane do `max_length`; wyniki nie mierzą dalszych tokenów.
- PPL domenowa może być obciążona leksykalnym sposobem konstrukcji domen.
- Kontrola niezależności opiera się na ścieżkach i deklarowanym pochodzeniu;
  nie wykrywa duplikatów lub parafraz pomiędzy różnymi plikami CSV.
- Każdy pojedynczy ekspert wymaga ponownego załadowania pełnego modelu; kod nie
  zwalnia jawnie poprzedniego modelu ani cache urządzenia, co może być istotne
  dla wielu ekspertów na małym GPU.
- Próg pewności routera silnie wpływa na wynik. Powinien być raportowany i
  najlepiej analizowany jako sweep, nie dobierany po test set.
- Raport Markdown nie pokazuje rozkładów routingu; pełne dane są tylko w JSON.

## Wynik holdoutu i audyt poprawki

Przy limicie 10% wszystkie pojedyncze eksperty i MoE przeszły bramkę. MoE
zwiększył PPL domen o 1,80–5,89%, a ogólną o 0,39%. Ekspert semantyczny był
lepszy od kontroli magnitudowej 6/6 i od jednej losowej 3/6. Nie poprawił
modelu bazowego. Po wykryciu paddingu w licznikach powtórzono wyłącznie część
MoE z maską telemetrii; PPL wszystkich siedmiu zbiorów pozostały bitowo
identyczne. Szczegóły zawiera [opis eksperymentu](gemma_moe_experiment.md).
