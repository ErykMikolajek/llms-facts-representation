# Audyt zmian, smoke testów i problemów badawczych

Ta strona zachowuje historię decyzji. Finalne artefakty są opisane na stronie
[wykonanego eksperymentu](gemma_moe_experiment.md), ale wcześniejszych wyników
nie należy usuwać z dokumentacji: wyjaśniają, dlaczego pipeline ma obecne
bramki i kontrakty.

## Pierwszy smoke test częściowego snapshotu

8 września 2026 r. dostępny checkpoint miał `completed=false`: 31 068 456
tokenów, 16 050 batchy, 32 100 sekwencji i 16 312/16 384 zaobserwowanych cech.
Nie miał gotowego logit lens ani źródłowych memmap, dlatego został
sfinalizowany diagnostycznie bez wznawiania forwardów. Triage musiał użyć
trigger tokens, nie kierunków dekodera.

HDBSCAN utworzył 47 klastrów z 9105 niepustych wektorów; 5166 cech było
szumem, a silhouette części sklasteryzowanej wyniósł 0,627. Do testu wybrano:

- klaster 37: programowanie i typy danych, 36 cech;
- klaster 42: miejsca geograficzne, 44 cechy;
- klaster 45: biologia komórkowa, 131 cech.

Mapping użył tej samej puli 769 tokenów. Router na 60 kołowych kontekstach
diagnostycznych osiągnął macro-F1 0,863; nie była to miara generalizacji.

| Zachowane neurony | Pruning | Regresja PPL MoE | Fallback |
| ---: | ---: | ---: | ---: |
| 512/2048 | 75% | +19,6% | 33,8% |
| 1024/2048 | 50% | +8,9% | 33,8% |
| 1536/2048 | 25% | +2,1% | 33,8% |

Wariant 50% służył tylko jako techniczny artefakt smoke testu. Dobór odbywał
się na kołowych danych development, eksperci nie poprawili bazowej PPL, a
maski miały przecięcia zgodne w przybliżeniu z przypadkiem. Wyniku nie należy
cytować jako finalnego eksperymentu.

## Najważniejsze naprawione błędy

| Problem wcześniejszego kodu | Ryzyko | Obecne rozwiązanie |
| --- | --- | --- |
| downstream wymuszał lokalny `TopKSAE` | brak obsługi Gemma Scope | zunifikowane API SAELens i odczyt release/ID z provenance |
| ścieżki MLP zakładały GPT `c_proj` | błędna warstwa dla Gemmy | adapter GPT-Neo/GPT-NeoX/Gemma-Llama gated |
| mapping nie gwarantował fizycznej osi MLP | pruning residual stream zamiast neuronów | hook wejścia `down_proj`, czyli post-gating `[N,H]` |
| domeny widziały różne tokeny | nieporównywalne korelacje | jeden pooled corpus i jedna macierz `[N,H]` |
| sześć kopii dużej macierzy MLP | zbędny RAM/dysk i ryzyko OOM | pojedynczy `pooled_mlp_activations.npz` |
| maskowana była jedna projekcja gated MLP | neuron pozostawał częściowo aktywny | `gate_proj`, `up_proj` i kolumna `down_proj` |
| eksperci byli liczeni dla wszystkich tokenów | koszt rósł × liczba ekspertów | selektywny hard dispatch tylko przypisanych wierszy |
| brak fallbacku poza domenami | arbitralny ekspert dla OOD | confidence threshold i gęsty base MLP |
| null poolował wszystkie neurony | wiele false positives | max-statistic per permutacja blokowa |
| minimum 5% nadpisywało pusty test | ukrywanie wyniku negatywnego | domyślny floor 0 i blokada pustego eksperta |
| brak bramki semantycznej | pruning nawet dla źle nazwanej domeny | sample-level AUC/CI przed pruningiem |
| korelacja robiła kopie float64 | OOM | float32 storage i blokowe statystyki float64 |
| losowy split fragmentów | source leakage | group-aware split całych źródeł |
| długie teksty dominowały router | bias MATH/Python | token weights sumujące się do 1 per tekst |
| backbone był liczony co epokę | wysoki koszt treningu | jednokrotny cache wejść FP16 CPU |
| próg routera był arbitralny | niekontrolowany routing ogólny | kwantyl na development general, target 5% |
| dane discovery służyły ewaluacji | circular validation | jawne A/B/C oraz osobny benchmark D |
| PPL przy lewym paddingu mogło liczyć pierwszy token | błędny mianownik | maskowanie pierwszego ważnego labelu |
| routing stats liczyły padding | mylący route/fallback rate | osobna maska telemetrii; PPL potwierdzone bitowo |
| pełne stany ekspertów były wymagane runtime | około 90 MiB duplikacji | `base_masks` i bundle z sześcioma maskami 13 056 B |

## Dobre praktyki dodane do protokołu

- dokładna walidacja model–SAE–warstwa–site i fingerprint tokenizera;
- ręczna bramka `downstream_approved` po inspekcji domen;
- stabilność klastrów dla wielu ustawień i cross-view z korektą FDR;
- rozłączność hashy oraz `source_group` między development i holdout;
- sample-level bootstrap zamiast uznawania tokenów za niezależne;
- losowa i magnitudowa kontrola o tym samym budżecie sparsity;
- zamrożenie konfiguracji i hashy przed holdoutem;
- manifest runtime weryfikujący każdą zależność;
- sealed benchmark, katalog bez prawa nadpisania i marker zużytego smoke testu;
- niewykonywanie niezaufanego kodu MBPP w notebooku.

## Nierozstrzygnięte problemy badawcze

1. Logit lens pomija zależną od wejścia końcową RMSNorm i nie jest dokładnym
   efektem interwencji.
2. Klasteryzacja pozostaje zależna od filtrów, SVD i hiperparametrów; wykonana
   analiza stabilności nie obejmuje wszystkich możliwych metod.
3. Źródło danych jest skorelowane z domeną. Wysokie AUC routera i SAE mogą
   częściowo oznaczać klasyfikację stylu datasetu.
4. Korelacja SAE–MLP nie dowodzi przyczynowej roli neuronu. Potrzebne są
   ablacje, activation patching lub sterowanie cechą.
5. Badana jest jedna warstwa, jeden SAE i mały model 270M; brak podstaw do
   szerokiej generalizacji.
6. Etykieta całego tekstu jest kopiowana na neutralne tokeny. Confidence
   fallback zmniejsza, ale nie usuwa label noise.
7. Jedna losowa maska per domena nie estymuje wariancji przypadku. Następny
   eksperyment powinien użyć wielu seedów lub rozkładu kontroli.
8. Pruning jednej warstwy o 25% nie oznacza 25% redukcji FLOPs modelu ani
   przyspieszenia ściennego; routing i małe nieregularne grupy mają narzut.
9. Mask-only MoE przechowuje sześć kompaktowych ekspertów i fallback w RAM,
   więc warunkowy koszt forwardu może maleć mimo wzrostu liczby parametrów.
10. PPL mierzy modelowanie języka, nie poprawność wiedzy. Accuracy, exact
    match, NLL i routing z niezależnego benchmarku trzeba interpretować razem.

## Precyzyjne nazewnictwo

W pracy zalecane określenie to *domain-conditioned pruned MLP experts*.
Eksperci są podmodułami powstałymi przez wybór neuronów wspólnego MLP, a nie
niezależnie trenowanymi ekspertami klasycznej architektury sparse MoE. Wynik
holdoutu uzasadnia stwierdzenie „zachowanie większości jakości przy
warunkowym pruningu jednej warstwy”, nie „poprawę wiedzy domenowej”.

