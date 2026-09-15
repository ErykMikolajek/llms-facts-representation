# `domain_triage/semantic_domain_triage.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/domain_triage/semantic_domain_triage.py) · [analiza Pythii](features_analysis.md) · [analiza Gemmy](gemma_scope_analysis.md) · [następny etap](domain_mlp_activation_mapping.md)

## Rola modułu

Moduł grupuje cechy SAE w kandydatów na domeny semantyczne. Reprezentacją
cechy nie jest jej pełny wektor aktywacji w korpusie, lecz rzadki profil
tokenów: domyślnie dodatnich tokenów promowanych przez logit lens, a w trybie
diagnostycznym tokenów obserwowanych przy aktywacji cechy. Pipeline wykonuje:

```text
kierunki dekodera SAE
→ top promowane tokeny
→ filtrowana macierz cecha × słownik
→ opcjonalne TF–IDF i normalizacja L2
→ Truncated SVD
→ HDBSCAN w przestrzeni SVD
→ ocena w przestrzeni tokenowej i hybrydowy wybór tokeny/SVD
→ teksty walidacyjne/diagnostyczne
```

## Dwie ścieżki wejściowe

### Live logit lens

Gdy nie podano `--feature-analysis`, skrypt ładuje lokalny checkpoint Top-K
SAE oraz model bazowy i oblicza `W_dec @ W_Uᵀ`. Wymaga wag modelu, ale może
wybrać do `top_m` tokenów bez ograniczenia przez wcześniejszy raport.

### Zapisana analiza cech

`--feature-analysis` czyta `features_analysis.json` Pythii albo
`feature_analysis.json` Gemmy. Wagi modelu i sam SAE nie są wtedy potrzebne do
klasteryzacji. Tokeny mogą być zapisane jako słowniki z `token_id` lub jako
pary `(tekst, score)`. W drugim wariancie kod wymaga dokładnego round-tripu
`encode → jeden token → decode`, aby nie przypisać błędnego ID tokenom BPE.

Domyślnie w tej ścieżce uwzględniane są tylko cechy zaobserwowane w analizie;
`--include-unobserved-features` wyłącza filtr.

`--feature-token-source` określa reprezentację cechy:

- `promoted` (domyślnie) używa `top_promoted_tokens` z logit lens;
- `trigger` używa obserwowanych `common_trigger_tokens`, a ich count zamienia
  na wagę `log(1+count)`; jest to diagnostyczny fallback dla częściowego
  checkpointu bez policzonego logit lens;
- `fallback` wybiera dla każdej cechy promowane tokeny, a gdy ich brak —
  trigger tokens.

W każdym wariancie lista wejściowa jest ograniczana do pierwszych `top_m`
elementów. Tryb trigger opisuje współwystępowanie cechy z tokenami korpusu, a
nie kierunek dekodera względem słownika; wyników obu reprezentacji nie należy
interpretować jako tej samej wielkości.

## Filtrowanie tokenów

`is_semantic_token()` odrzuca:

- tokeny specjalne;
- elementy krótsze niż `min_token_chars` po normalizacji;
- elementy bez żadnej litery oraz zapis wyglądający jak `<special>`;
- ręcznie rozszerzoną listę słów funkcyjnych oraz pełny
  `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`, chyba że włączono
  `--keep-common-function-tokens`;
- opcjonalnie kontynuacje subwordów bez granicy słowa.

Detekcja granicy rozpoznaje prefiks `Ġ` GPT-BPE, `▁` SentencePiece i `##`
WordPiece, a pomocniczo także początkową spację tekstu zdekodowanego. Filtr jest
heurystyczny i zależny od implementacji tokenizera.

`normalize_token_text()` zamienia newline na spację, zwija whitespace,
konwertuje do lowercase i usuwa brzegową interpunkcję. Ta postać służy do
agregacji i automatycznej nazwy domeny; oryginalny tekst i token ID pozostają w
rekordach wyjściowych.

## Budowa macierzy `F × V`

Dla każdej cechy zachowywane są wyłącznie dodatnie scores spełniające
`score ≥ min_logit`. W trybie promoted jest to logit, a w trigger —
`log(1+count)`. Powstaje CSR, gdzie:

```text
M[feature_id, token_id] = dodatnia waga wybranego źródła tokenów
```

Przy TF–IDF kolumna tokenu jest mnożona przez:

```text
idf(t) = log((1 + liczba_cech) / (1 + df(t))) + 1
```

W live path mianownikiem globalnym jest szerokość SAE, a w ścieżce offline
liczba włączonych cech. Następnie każdy niepusty wiersz jest normalizowany L2.
Wartość `idf` jest dopisywana także do list top tokenów, aby agregacja domen
użyła zgodnej korekty.

## Redukcja i klasteryzacja

1. Puste wiersze są wyłączane, ale oryginalne `feature_id` są zachowane w
   `valid_feature_ids`.
2. `TruncatedSVD` wybiera
   `min(requested, n_features-1, n_vocab-1)` składowych, co najmniej 2, i ma
   stały `random_state=0`.
3. Wektory po SVD są ponownie normalizowane L2.
4. HDBSCAN pracuje z metryką euklidesową. Dla wektorów jednostkowych jest ona
   monotonicznie powiązana z podobieństwem cosinusowym.
5. Label `-1` oznacza szum i nie staje się domeną.

SVD służy do znalezienia labeli HDBSCAN oraz do pomocniczej oceny odległości
między klastrami. Kohezja i leksykalny centroid pozostają liczone w oryginalnej
przestrzeni tokenowej.

## Kandydaci i wybór domen

Dla każdego klastra kod wraca do znormalizowanej, rzadkiej macierzy `F×V`.
Sumuje oryginalne wektory tokenowe cech, normalizuje sumę L2 i otrzymuje
rzadki centroid `[1,V]`. **Cohesion** to średni iloczyn skalarny członków z tym
centroidem, czyli średnie podobieństwo cosinusowe w przestrzeni semantycznych
tokenów — bez zniekształcenia przez redukcję SVD.

Tokeny klastra są agregowane według zsumowanego `logit × idf`; raport zapisuje
też liczbę cech wspierających dany token. Token jest dopuszczany do etykiety
domeny dopiero, gdy jego wsparcie osiąga maksimum z dwóch progów:

```text
required_support = max(
    min_token_feature_support,
    ceil(cluster_size × min_token_feature_fraction)
)
```

Domyślne wartości to odpowiednio 2 cechy i 5% klastra. Zapobiega to nazywaniu
dużych domen tokenem wspieranym przez stałą, znikomą liczbę cech.

Klastry większe niż
`floor(valid_features × max_domain_fraction)` są odrzucane jako zbyt szerokie.
Odrzucany jest również klaster, który po filtrze wsparcia nie ma co najmniej
`min_distinct_domain_tokens` różnych tokenów (domyślnie 5), ponieważ nie daje
podstaw do sensownej etykiety.
Następnie:

1. pierwszy klaster maksymalizuje `cohesion × log(1+size)`;
2. dla pary klastrów podobieństwo jest maksimum z podobieństwa rzadkich
   centroidów tokenowych i nieujemnej części podobieństwa centroidów SVD;
3. każdy następny klaster maksymalizuje
   `(1 - max hybrid-similarity-to-selected) × cohesion × log(1+size)`;
4. pierwsze trzy zagregowane tokeny tworzą automatyczną nazwę.

Użycie maksimum sprawia, że klastry nie są uznawane za odległe tylko dlatego,
że ich przycięte listy top tokenów nie mają wspólnych pozycji. Ujemny cosinus
w przestrzeni SVD jest obcinany do zera, ponieważ kod traktuje go jako artefakt
redukcji, a nie dowód semantycznego przeciwieństwa.

Wszystkie klastry, także niewybrane, trafiają do `cluster_candidates.json` z
kohezją, tokenami, feature IDs i flagą `eligible`. Po ręcznej inspekcji
`--select-cluster-labels` może ograniczyć dalszy dobór do konkretnych labeli
HDBSCAN. Label musi być dostępny i spełniać filtry, a liczba ręcznie wskazanych
klastrów nie może przekroczyć `n_domains`. Kolejność nowych `domain_id` nadal
wynika z zachłannego selektora, nie z kolejności argumentów CLI.

To zachłanny wybór różnorodnych domen, a nie globalnie optymalny dobór.
Gdy liczba domen jest mniejsza niż `min_required_domains`, run kończy
się błędem, chyba że jawnie włączono tryb diagnostyczny
`--allow-underfilled-domains`.

Diagnostyka zawiera również silhouette score dla cech przypisanych do
klastrów (szum `-1` jest wyłączony). Miara jest liczona w przestrzeni SVD, na
co najwyżej 5000 punktach z `random_state=0`. Jest to opis separacji klastrów,
nie dowód ich poprawności semantycznej.

!!! warning "Nazwy domen"
    Nazwa `token1 / token2 / token3` jest proxy, nie zweryfikowaną etykietą
    semantyczną. Przed użyciem domen w interpretacji pracy trzeba obejrzeć
    `domain_report.md`, konteksty i reprezentatywne cechy.

## Ręczna bramka przed dalszym pipeline'em

Każdy nowy `domains.json` zawiera `manual_review_required=True`. Domyślnie ma
również `downstream_approved=False`, przez co właściwy mapping MLP odmawia
pracy. Po obejrzeniu raportu, top tokenów, kontekstów, podobieństw i
diagnostyki badacz może:

- zatwierdzić domeny podczas pełnego uruchomienia przez
  `--approve-domains-for-downstream`;
- albo zatwierdzić istniejące artefakty bez ponownej klasteryzacji przez
  `--approve-existing-domains` z tym samym `--output-dir`.

Drugi wariant ustawia `downstream_approved=True`, dodaje metodę i znacznik UTC
do pola `approval` oraz aktualizuje stan w `domain_report.md`. To zapis decyzji
człowieka, a nie automatyczny test jakości. `--allow-unapproved-domains` w
mappingu istnieje wyłącznie dla diagnostycznego smoke testu.

## Budowa zbiorów tekstowych

### Zewnętrzny `validation.csv` do developmentu

Dokumenty są dzielone na zdania. Dla każdej domeny top tokeny tworzą leksykon,
a wagi są dzielone przez maksymalną wagę domeny. Wynik fragmentu to:

```text
score_d(text) = Σ_t weight_d(t) × occurrences(t) / sqrt(number_of_words)
```

Fragment jest przyjmowany, gdy najlepszy wynik przekracza `min_domain_score`
i przewyższa drugi o co najmniej `exclusive_margin`. Duplikaty tekstu są
usuwane, próbki tasowane i ograniczane do `samples_per_domain`.

Teksty mogą pochodzić z korpusu niezależnego od analizy SAE, ale **etykiety są
nadal nadawane leksykonem utworzonym z odkrytych domen**. Z tego powodu summary
ustawia `independent_evaluation=False`; zbiór nadaje się do mappingu i treningu
routera, lecz nie jest końcowym, ręcznie etykietowanym holdoutem.

Jeżeli `--generate-if-needed` jest włączone, brakujące próbki są generowane z
promptu o top tokenach (domyślnie jako krótka historia dziecięca) z temperaturą
0.8 i top-p 0.9. Takie teksty mają `source="generated"` i należy analizować je
oddzielnie od danych naturalnych.

### Konteksty z analizy cech

Gdy CSV nie istnieje, `build_validation_sets_from_feature_analysis()` pobiera
najsilniejsze konteksty cech należących do klastra. Usuwa oznaczenia `[[...]]`,
odrzuca ten sam tekst, jeśli jest przypisany do kilku domen, i dobiera próbki
round-robin po cechach, aby jedna silna cecha nie zdominowała domeny.

Wszystkie takie rekordy mają `diagnostic_only=True`, a summary jawnie ustawia
`independent_evaluation=False`. `source_group=sequence:<id>` pozwala routerowi
utrzymać konteksty tej samej sekwencji po jednej stronie podziału.

!!! danger "Brak niezależności"
    Obie automatyczne strategie generują dane deweloperskie: zewnętrzny CSV ma
    etykiety wyznaczone leksykonem domen, a konteksty pochodzą dodatkowo z tej
    samej analizy, która zdefiniowała domeny. Do końcowej walidacji MoE potrzebny
    jest osobny, ręcznie etykietowany `--domain-eval-csv`.

## Przebieg `main()`

1. Rozwiązuje ścieżki checkpointu, outputu, CSV i analizy kontekstów.
2. Sprawdza, czy dostępne są minimalne artefakty dla wybranej ścieżki.
3. Ładuje tokenizer, a zależnie od trybu także SAE i model.
4. Buduje macierz cecha–token i jej metadane.
5. Dla zapisanego raportu sprawdza warstwę, site `resid_post`, rozmiar i SHA-256
   słownika tokenizera, a w trybie `promoted` także dostępność i wystarczające
   `top_k` logit lens; mismatch wymaga jawnego override diagnostycznego.
6. W live path może dodatkowo wyzerować wiersze cech niezaobserwowanych według
   analizy kontekstów.
7. Zapisuje macierz i top tokeny jeszcze przed SVD — diagnostyka pozostaje
   dostępna nawet, gdy późniejszy etap nie znajdzie wystarczających domen.
8. Redukuje i klastruje, po czym liczy kohezję w przestrzeni tokenowej, zapisuje
   wszystkich kandydatów, filtruje klastry szerokie lub ubogie we wspierane
   tokeny i wybiera domeny z hybrydową odległością tokeny/SVD.
9. Zapisuje `domains.json`, przypisania CSV i raport Markdown wraz ze stanem
   ręcznego zatwierdzenia.
10. Buduje walidację z CSV, diagnostyczne konteksty albo summary `skipped`.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `feature_token_matrix.npz` | znormalizowana rzadka macierz `F×V` |
| `logit_lens_top_tokens.jsonl` | przefiltrowane top tokeny każdej cechy |
| `cluster_candidates.json` | wszystkie klastry z jakością, feature IDs i statusem eligibility do ręcznej inspekcji |
| `domains.json` | domeny, cechy, podobieństwa, filtry, provenance, diagnostyka i bramka zatwierdzenia |
| `feature_domain_assignments.csv` | label HDBSCAN i wybrana domena dla każdego feature ID |
| `domain_report.md` | raport domen i macierz podobieństw |
| `domain_validation/domain_<id>.jsonl` | teksty per domena |
| `domain_validation/all_domains_balanced.csv` | połączone próbki |
| `validation_summary.json` | główny wskaźnik strategii i jakości próbek |

## Katalog funkcji

| Funkcja / klasa | Odpowiedzialność |
| --- | --- |
| `DomainInfo` | finalny rekord domeny i listy jej feature IDs |
| `require_hdbscan()` | leniwie importuje zależność z czytelnym błędem |
| `normalize_token_text()` | normalizuje tekst do agregacji |
| `is_semantic_token()` | filtruje techniczne, funkcyjne i kontynuacyjne tokeny |
| `resolve_single_token_id()` | bezpieczny round-trip tekstu do pojedynczego ID |
| `load_sae_checkpoint()` | odtwarza lokalny TopKSAE |
| `compute_top_promoted_tokens()` | live logit lens, filtr, CSR, TF–IDF i L2 |
| `load_precomputed_feature_analysis()` | buduje CSR z promowanych lub trigger tokens, stosując `top_m`, filtry i metadane |
| `load_observed_feature_ids()` | tworzy listę cech widzianych w zapisanej analizie |
| `reduce_feature_matrix()` | usuwa puste wiersze, Truncated SVD i normalizuje |
| `aggregate_cluster_tokens()` | sumuje ważone tokeny i wymusza bezwzględny/procentowy próg wsparcia |
| `cluster_features()` | uruchamia HDBSCAN |
| `compute_cluster_candidates()` | centroid tokenowy, centroid SVD, cohesion w `F×V` i top tokeny klastra |
| `select_orthogonal_domains()` | zachłannie wybiera domeny według hybrydowej odległości i kohezji |
| `summarize_domain_quality()` | noise fraction, małe domeny, udział słów funkcyjnych i silhouette SVD |
| `validate_domain_selection()` | blokuje niedostateczną liczbę domen poza trybem diagnostycznym |
| `approve_existing_domains()` | zapisuje po ręcznym przeglądzie zgodę na dalsze etapy bez rerunu |
| `write_json()`, `write_feature_top_tokens()`, `write_assignments()`, `write_domain_report()` | zapisują artefakty |
| `iter_validation_fragments()` | strumieniuje zdania z CSV |
| `build_domain_lexicons()` | tworzy i skaluje słowniki domen |
| `score_fragment()` | ważony lexical score z normalizacją długości |
| `assign_fragment()` | zwraca najlepszą domenę po progach; helper nie jest obecnie wywoływany przez `main()` |
| `generate_domain_samples()` | generuje brakujące teksty na żądanie |
| `build_validation_sets()` | selekcja/balansowanie walidacji CSV |
| `build_validation_sets_from_feature_analysis()` | diagnostyczne konteksty z ochroną przed konfliktem etykiet |
| `default_checkpoint_path()` | standardowa ścieżka najlepszego TopKSAE |
| `validate_feature_analysis_contract()` | sprawdza layer/site/logit top-k oraz fingerprint tokenizera |
| `validate_cli_args()` | sprawdza dodatniość, zakresy i relacje argumentów CLI |
| `parse_args()` / `main()` | CLI, tryb samego zatwierdzenia i pełna orkiestracja |

## Ograniczenia metodologiczne

- Logit lens mierzy kierunek dekodera względem słownika, nie rozkład aktywacji
  cechy w danych. Dwie cechy z podobnymi promowanymi tokenami mogą reagować na
  odmienne konteksty.
- `trigger` jest innym, korpusowym proxy; `fallback` może mieszać oba źródła w
  jednej macierzy i dodatkowo pomija ścisłą walidację metadanych logit lens.
- HDBSCAN i automatyczny wybór domen zależą od filtrów, `top_m`, wymiaru SVD i
  minimalnego rozmiaru klastra; stabilność powinna być sprawdzona w analizie
  czułości.
- CLI `--seed` nie steruje SVD, które ma stałe `random_state=0`; steruje głównie
  samplingiem/generowaniem walidacji.
- Leksykalna walidacja używa tych samych tokenów, które nazwały domeny, więc
  potwierdza zgodność tekstu ze słownikiem, a nie niezależnie odkrytą semantykę.

## Finalny triaż Gemmy

Właściwy przebieg korzystał z ukończonej analizy około 100 mln tokenów i
promowanych tokenów logit lens. Stabilność sprawdzono dla zmian wymiaru SVD,
seedu i minimalnego rozmiaru klastra oraz w reprezentacji kontekstowej;
procedurę opisuje [Stabilność i zgodność triażu](triage_stability.md).

Po ręcznym przeglądzie zaakceptowano klastry: 20 prawo (161 cech), 17
biomedycyna (396), 16 sport (117), 22 polityka (123), 15 matematyka/LaTeX
(322) i 5 Python (149). Finalne `domains.json` zostało przygotowane przez
kontrolowany skrypt, a nie przez automatyczne przemianowanie top-6 rankingu.
