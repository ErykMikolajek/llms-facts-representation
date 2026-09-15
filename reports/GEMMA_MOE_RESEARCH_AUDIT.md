# Audyt pipeline'u Gemma 3 → domeny semantyczne → pruning → MoE

## Status

Kod jest przygotowany do eksperymentu z `google/gemma-3-270m`, warstwą 9 i
SAE Gemma Scope 2 `resid_post`. Gotowość kodu nie jest wynikiem badawczym:
pełny przebieg wymaga dostępu do wag Gemmy, artefaktu
`feature_analysis.json` oraz trzech rozłącznych źródeł tekstów.

## Test snapshotu z 8 września 2026

Snapshot `analysis_checkpoint_gemma_scope2_270m.pt` był poprawny, lecz
nieukończony (`completed=false`). Zawierał statystyki po 31 068 456 tokenach,
16 050 batchach i 32 100 sekwencjach do bieżącego kursora. Zaobserwowano
16 312 z 16 384 cech. Nie zawierał gotowego logit lens ani źródłowych tablic
tokenów, dlatego został sfinalizowany bez wznawiania analizy:

```bash
python3 -m sae_pipeline.gemma_scope_analysis \
  --data-path data/gemma_scope2_270m_pilecc \
  --checkpoint-path data/gemma_scope2_270m_pilecc/analysis/analysis_checkpoint_gemma_scope2_270m.pt \
  --model-name models/gemma-3-270m \
  --tokenizer-name models/gemma-3-270m \
  --finalize-checkpoint-only \
  --skip-logit-lens
```

Testowy triage użył więc tokenów wyzwalających jako leksykalnego proxy, a nie
wektorów logit lens. HDBSCAN utworzył 47 klastrów z 9 105 niepustych wektorów;
5 166 cech uznano za szum (56,7%), a silhouette dla części sklasteryzowanej
wyniósł 0,627. Po przeglądzie kandydatów do smoke testu wybrano klastry:

- 37 — programowanie i typy danych (`int / string / bool`), 36 cech;
- 42 — miejsca geograficzne (`york / london / washington`), 44 cechy;
- 45 — biologia komórkowa (`cells / cell / protein`), 131 cech.

Mapowanie wykonano na identycznej puli 769 tokenów dla każdej domeny. Udział
pozycji z niezerowym sygnałem klastra SAE wyniósł odpowiednio 18,5%, 11,7% i
43,8%. Router trenowany na 60 kołowych kontekstach diagnostycznych uzyskał
macro-F1 0,863 na grupowym podziale 46/14; liczba ta nie jest miarą
generalizacji.

Kalibracja minimalnej liczby neuronów dała następujący wynik na tym samym,
kołowym zbiorze rozwojowym:

| zachowane neurony eksperta | pruning | regresja PPL MoE | fallback routera |
| --- | --- | --- | --- |
| 512/2048 (25%) | 75% | +19,6% | 33,8% |
| 1024/2048 (50%) | 50% | +8,9% | 33,8% |
| 1536/2048 (75%) | 25% | +2,1% | 33,8% |

Wariant 50% zachowanych neuronów jest wybranym artefaktem smoke testu, ponieważ
przechodzi techniczną bramkę maksymalnie 10% regresji PPL i nadal redukuje
połowę neuronów wykonywanego eksperta. Nie jest to wynik empiryczny do pracy:
dobór sparsity nastąpił na danych rozwojowych, a eksperci nie poprawili PPL
własnych domen względem modelu bazowego. Wymagany jest niezależny podział B/C.

Przecięcia masek wariantu 50% wynoszą 502, 512 i 540 neuronów; odpowiada to
49,0%, 50,0% i 52,7% pojedynczej maski. Dla dwóch niezależnych masek po 1024 z
2048 neuronów wartość oczekiwana wynosi 512. Potrójne przecięcie ma 266
neuronów wobec wartości oczekiwanej 256. Maski nie są więc kopiami, ale ten
smoke test nie dostarcza też mocnego dowodu struktury wspólnego rdzenia ani
ponadlosowej separacji ekspertów.

## Najważniejsze naprawione problemy

| Problem | Skutek przed poprawką | Obecne zabezpieczenie |
| --- | --- | --- |
| Downstream wymuszał lokalny `TopKSAE` | artefakt Gemma Scope nie mógł zasilić mapowania | obsługa SAELens i automatyczne odczytanie `sae_release`/`sae_id` z `domains.json` |
| MLP było na stałe związane z `transformer.h[].mlp.c_proj` | brak zgodności z Gemmą 3 | adapter dla `model.layers[]` i `gate_proj`, `up_proj`, `down_proj` |
| Maskowana była tylko pojedyncza projekcja wejściowa | ekspert gated-MLP zachowywał część prunowanego neuronu | spójne maskowanie obu projekcji wejściowych i kolumn `down_proj` |
| Każdy ekspert liczony był dla wszystkich tokenów | koszt rósł liniowo z liczbą ekspertów | obliczanie tylko tokenów przypisanych do eksperta i kompaktowe projekcje z zachowanych neuronów |
| Każdy token musiał trafić do domeny | router wymuszał arbitralnego eksperta poza rozkładem | próg ufności i fallback do bazowego, gęstego MLP |
| Próg null powstawał z puli wszystkich neuronów | pod nullem pozostawał oczekiwany odsetek fałszywych neuronów | blokowa permutacja i maksimum po neuronach dla każdej permutacji |
| Domyślne minimum 5% neuronów nadpisywało test statystyczny | ekspert nigdy nie mógł być naprawdę pusty | domyślny `min_keep_fraction=0`; pusty ekspert kończy run błędem |
| Korelacja tworzyła kilka pełnych kopii `float64` | ryzyko OOM | statystyki Pearsona liczone blokowo, dane przechowywane w `float32` |
| Różne domeny dostawały różny podzbiór puli tekstów | korelacje domen nie były porównywalne | identyczne losowanie i limit tokenów dla każdej domeny |
| Podział routera był losowany po fragmentach | konteksty jednej sekwencji trafiały do train i validation | podział grupowy po źródłowej sekwencji i usuwanie sprzecznych etykiet |
| Te same konteksty służyły do odkrycia domen i oceny | zawyżone metryki przez wyciek danych | oznaczenie `diagnostic_only` oraz blokady w routerze i końcowej ewaluacji |
| PPL przy lewym paddingu obejmowało pierwszy token tekstu | błędny mianownik i strata | pierwszy poprawny token każdego przykładu jest ignorowany |

## Poprawny podział danych

Nie należy nazywać katalogu `domain_validation` z triage'u niezależną
walidacją. To dane rozwojowe, ponieważ etykiety pochodzą z leksykonów domen
albo z top-kontekstów tych samych cech SAE.

Zalecany podział:

1. **Discovery A** — próbka Pile-CC do analizy SAE, logit lens i klasteryzacji.
2. **Development B** — osobna pula tekstów do mapowania neuronów oraz treningu
   routera. Etykiety domen należy skontrolować ręcznie; podział routera jest
   wykonywany po grupach źródłowych.
3. **Holdout C** — niewidziany wcześniej zbiór ogólny oraz ręcznie opisany plik
   `domain_id,text` do końcowej oceny. Nie wolno na nim dobierać `tau`, progu
   routera ani hiperparametrów HDBSCAN.

## Kolejność uruchomienia dla Gemmy 3

### 1. Analiza Gemma Scope 2

Notebook `kaggle/notebooks/kaggle_gemma_scope2_270m.ipynb` ustawia `LOGIT_TOP_K = 128`. Dla
nowego przebiegu uruchamia `sae_pipeline/gemma_scope_analysis.py` z domyślnym
`layer_9_width_16k_l0_medium`. Wynikiem wymaganym przez triage jest:

```text
data/gemma_scope2_270m_pilecc/analysis/feature_analysis.json
```

Model card pokazuje historyczną nazwę release'u zakończoną `-resid_post`,
natomiast rejestr SAELens 6.50 używa `gemma-scope-2-270m-pt-res`. Loader
akceptuje obie nazwy i normalizuje je do klucza bieżącego rejestru.

Pełny triage badawczy nadal wymaga sfinalizowania logit lens z co najmniej
`logit_top_k=128`. Tryb `--feature-token-source trigger` jest dopuszczalny
wyłącznie jako leksykalny smoke test częściowego checkpointu.

### 2. Triage bez zatwierdzenia

```bash
python3 -m domain_triage.semantic_domain_triage \
  --data-path data/gemma_scope2_270m_pilecc \
  --feature-analysis data/gemma_scope2_270m_pilecc/analysis/feature_analysis.json \
  --context-analysis data/gemma_scope2_270m_pilecc/analysis/feature_analysis.json \
  --validation-csv data/gemma_scope2_270m_pilecc/development.csv \
  --model-name google/gemma-3-270m \
  --tokenizer-name google/gemma-3-270m \
  --layer-num 9 \
  --n-domains 5 \
  --svd-components 50 \
  --min-cluster-size 20 \
  --require-token-boundary
```

Po runie należy sprawdzić `domain_report.md`, top tokeny i konteksty. Nazwy
generowane z tokenów są wyłącznie propozycjami etykiet. Dopiero po przeglądzie:

```bash
python3 -m domain_triage.semantic_domain_triage \
  --data-path data/gemma_scope2_270m_pilecc \
  --approve-existing-domains
```

`development.csv` jest osobną, niewykorzystaną w analizie pulą B. Triage nadaje
jej słabe etykiety przez leksykony domen; przed mapowaniem należy przejrzeć i
w razie potrzeby poprawić wygenerowane pliki `domain_validation`. Jeżeli pliku
B nie podano, konteksty z `feature_analysis.json` są jawnie diagnostyczne i
router domyślnie je odrzuci.

### 3. Mapowanie fizycznych neuronów gated-MLP

```bash
python3 -m domain_mapping.domain_mlp_activation_mapping \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage \
  --model-name google/gemma-3-270m \
  --tokenizer-name google/gemma-3-270m \
  --layer-num 9 \
  --mapping-corpus pooled-domains \
  --max-tokens-per-domain 100000 \
  --metrics pearson
```

Źródło SAE jest odczytywane z `domains.json`. Można je podać jawnie przez
`--sae-release gemma-scope-2-270m-pt-res --sae-id
layer_9_width_16k_l0_medium`.

### 4. Pruning

```bash
python3 -m domain_mapping.domain_mlp_pruning \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage \
  --model-name google/gemma-3-270m \
  --tokenizer-name google/gemma-3-270m \
  --layer-num 9 \
  --tau-method per-domain-null \
  --null-permutations 100 \
  --null-block-size 32 \
  --null-percentile 95 \
  --min-keep-fraction 0
```

Brak zachowanych neuronów jest wynikiem negatywnym dla danej domeny, a nie
błędem do automatycznego obejścia.

### 5. Router i złożenie MoE

```bash
python3 -m moe.router_training \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage \
  --model-name google/gemma-3-270m \
  --tokenizer-name google/gemma-3-270m \
  --layer-num 9 \
  --epochs 10

python3 -m moe.moe_assembly \
  --mode moe \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage \
  --model-name google/gemma-3-270m \
  --layer-num 9 \
  --router-confidence-threshold 0.6
```

Jeśli dostępne są tylko konteksty z analizy cech, router zatrzyma się. Flaga
`--allow-diagnostic-contexts` służy wyłącznie do sprawdzenia wykonania kodu.

### 6. Niezależna ewaluacja

```bash
python3 -m evaluation.moe_validation \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage \
  --model-name google/gemma-3-270m \
  --tokenizer-name google/gemma-3-270m \
  --layer-num 9 \
  --validation-csv data/gemma_scope2_270m_pilecc/holdout_general.csv \
  --domain-eval-csv data/gemma_scope2_270m_pilecc/holdout_domains.csv \
  --router-confidence-threshold 0.6
```

`holdout_domains.csv` musi mieć kolumny `domain_id,text`. Flaga
`--allow-development-eval` pozwala użyć danych z triage'u tylko jako smoke
test i nie może być podstawą wniosków w pracy.

## Nierozstrzygnięte problemy badawcze

- Logit lens `W_dec @ W_U^T` jest przybliżeniem i pomija nieliniowy końcowy
  RMSNorm. Domenę należy potwierdzić na kontekstach aktywacji oraz interwencją,
  nie tylko listą tokenów.
- Wynik HDBSCAN zależy od redukcji SVD i hiperparametrów. Należy raportować
  stabilność klastrów dla kilku seedów/próbek oraz porównać z klasteryzacją
  bez redukcji lub innym baseline'em.
- Wariant SAE 16k jest ekonomicznym pilotem. Przed wnioskiem ogólnym warto
  powtórzyć kluczowy eksperyment dla szerokości rekomendowanej w model card
  Gemma Scope 2 i sprawdzić stabilność domen między SAE.
- Korelacja neuronu MLP z aktywacją domeny nie dowodzi jego przyczynowej
  funkcji. Potrzebne są kontrole: maska losowa dopasowana rozmiarem, pruning
  po magnitudzie, ablacja cech SAE, kilka seedów i przedziały ufności.
- „Eksperci” są prunowanymi podmodułami wspólnego MLP, nie ekspertami uczonymi
  niezależnie. W tekście pracy należy używać określenia *domain-conditioned
  pruned MLP experts* i nie utożsamiać ich z klasycznym treningiem sparse MoE.
- Etykieta routera jest nadawana całemu tekstowi i kopiowana na tokeny, więc
  część tokenów ma zaszumioną etykietę. Należy raportować kalibrację, pokrycie
  fallbacku, macierz pomyłek i wyniki w funkcji progu ufności.
- Obecna modyfikacja dotyczy jednego MLP. Wnioski o lokalizacji wiedzy nie mogą
  być uogólnione na cały model bez kontroli innych warstw i miejsc hookowania.
- PPL mierzy jakość językową, ale nie bezpośrednio zachowanie faktów. Końcowa
  ocena powinna zawierać również zadania faktograficzne, skuteczność routingu,
  koszt obliczeniowy i liczbę fizycznie zachowanych parametrów.

## Minimalny warunek raportowania wyniku

Wynik można traktować jako empiryczny dopiero, gdy istnieją: co najmniej trzy
ręcznie zaakceptowane domeny, niepuste maski, oddzielne zbiory A/B/C, baseline'y
o dopasowanym budżecie sparsity, kilka seedów oraz końcowe metryki PPL i
faktograficzne na holdoucie C. Sam pomyślny smoke test pipeline'u tego warunku
nie spełnia.

## Źródła techniczne

- [Gemma 3 270M — model card](https://huggingface.co/google/gemma-3-270m)
- [Gemma Scope 2 270M — model card](https://huggingface.co/google/gemma-scope-2-270m-pt)
- [Implementacja `Gemma3MLP` w Transformers 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gemma3/modeling_gemma3.py)
