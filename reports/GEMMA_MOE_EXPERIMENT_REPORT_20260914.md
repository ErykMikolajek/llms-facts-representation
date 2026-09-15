# Gemma 3 270M: triaż domen, pruning i prototyp MoE

Data wykonania: 14.09.2026. Status: kompletny przebieg proof of concept z zamrożonym holdoutem.

## 1. Najważniejszy wniosek

Powstał działający, tokenowo routowany MoE dla warstwy 9 modelu Gemma 3 270M, z sześcioma kompaktowymi ekspertami MLP i oryginalnym gęstym MLP jako fallbackiem. Każdy ekspert usuwa 512 z 2048 neuronów warstwy (25%). Na zamrożonym holdoucie MoE zachował jakość w przyjętym limicie: wzrost perplexity wyniósł od 1,80% do 5,89% na domenach oraz 0,39% na korpusie ogólnym.

Nie uzyskano poprawy perplexity względem nieprzyciętego modelu. Wynik dowodzi zachowania znacznej części jakości przy warunkowym użyciu przyciętej warstwy, a nie wzrostu wiedzy lub jakości generacji. Maski semantyczne były lepsze od baseline'u magnitudowego w 6/6 domen, lecz od pojedynczej maski losowej tylko w 3/6. Jest to obiecujący PoC, ale jeszcze nie mocny dowód przewagi semantycznej selekcji neuronów.

## 2. Dane wejściowe i kontrakt analizy

Punktem wyjścia był ukończony `analysis_checkpoint_gemma_scope2_270m.pt` oraz odpowiadający mu `feature_analysis.json`. Analiza dotyczy:

- modelu Gemma 3 270M;
- warstwy 9 i strumienia rezydualnego `resid_post`;
- SAE Gemma Scope 2 `layer_9_width_16k_l0_medium` o 16 384 cechach;
- 99 999 989 tokenów w 103 513 sekwencjach;
- 16 343 zaobserwowanych cech SAE.

Kontrakt warstwy jest sprawdzany przed mapowaniem: wymiar wejścia SAE musi być równy `hidden_size=640`, identyfikator hooka musi wskazywać `blocks.9.hook_resid_post`, a indeksy cech nie mogą przekraczać szerokości SAE.

## 3. Wybrane domeny

| ID | Domena | Klaster | Cechy SAE | AUC selektywności | 95% CI |
|---:|---|---:|---:|---:|---:|
| 0 | prawo i orzecznictwo | 20 | 161 | 0,998 | [0,995; 1,000] |
| 1 | biomedycyna | 17 | 396 | 1,000 | [1,000; 1,000] |
| 2 | sport | 16 | 117 | 0,999 | [0,996; 1,000] |
| 3 | polityka i wiadomości | 22 | 123 | 0,957 | [0,934; 0,975] |
| 4 | matematyka / zapis LaTeX | 15 | 322 | 0,999 | [0,997; 1,000] |
| 5 | Python | 5 | 149 | 0,995 | [0,987; 1,000] |

Wszystkie klastry przeszły ustaloną wcześniej bramkę: ROC-AUC co najmniej 0,60 i dolna granica bootstrapowego 95% CI powyżej 0,50. Najtrudniejszą parą była polityka kontra prawo (pairwise AUC 0,808), co odpowiada podobieństwu języka instytucjonalnego.

## 4. Przygotowanie zbiorów

Zbiory zostały utworzone automatycznie z przypiętymi rewizjami źródeł. Dla każdej domeny zapisano 180 tekstów deweloperskich i 90 holdout; dodatkowo 200 tekstów ogólnych do kalibracji i 100 ogólnych do holdoutu. Łącznie jest 1280 unikalnych tekstów deweloperskich i 640 holdout. Przecięcie hashy treści i grup źródłowych między częściami wynosi zero.

Źródła:

- prawo: LexGLUE/SCOTUS, `train` → development, `test` → holdout;
- biomedycyna: PubMedQA `pqa_artificial` → development, `pqa_labeled` → holdout;
- sport i polityka: DBpedia14, klasy Athlete i OfficeHolder, oficjalne podziały `train`/`test`;
- matematyka/LaTeX: siedem konfiguracji MATH, problem wraz z rozwiązaniem, podziały `train`/`test`;
- Python: funkcje z lokalnej biblioteki standardowej CPython 3.11; pliki podzielone przed ekstrakcją funkcji regułą `sha256(relative_path) mod 5`;
- tekst ogólny: WikiText-2 raw, `validation` → development, `test` → holdout.

To usuwa bezpośredni przeciek treści, ale nie usuwa konfuzji domena–źródło. Wysoka klasyfikowalność może częściowo wynikać ze stylu zbioru, a nie wyłącznie z semantyki domeny.

## 5. Mapowanie SAE → neurony MLP

Na wspólnym, losowo przemieszanym korpusie deweloperskim wykonano jeden forward pass modelu dla 60 000 tokenów z 351 tekstów. W każdym tokenie zapisano:

1. aktywację `resid_post` warstwy 9 i zakodowano ją przez SAE;
2. sumę aktywacji cech każdego z sześciu klastrów;
3. rzeczywistą aktywację neuronu MLP na wejściu `down_proj`, czyli po bramkowaniu `act(gate_proj(x)) * up_proj(x)`;
4. identyfikator tekstu, pozycję tokenu i domenę źródłową.

Wspólna macierz `[60000, 2048]` gwarantuje, że wszystkie domeny są porównywane na dokładnie tych samych tokenach. Jednocześnie eliminuje sześć powtórzeń forward passu i sześć kopii 447-megabajtowej macierzy MLP.

Dla każdego neuronu policzono bezwzględną korelację Pearsona z sygnałem klastra. Użycie wartości bezwzględnej zachowuje zarówno dodatnie, jak i ujemne związki; ze względu na znakowane aktywacje po bramkowaniu znak korelacji nie jest prostą etykietą „wspierający/hamujący”.

## 6. Pruning

Próg statystyczny wyznaczono oddzielnie dla każdej domeny:

- 100 permutacji sygnału klastra;
- permutowane bloki po 32 tokeny, aby częściowo zachować lokalną autokorelację;
- dla każdej permutacji maksimum `|r|` wśród 2048 neuronów;
- próg `tau` jako 95. percentyl rozkładu tych maksimów.

Ta statystyka maksimum ogranicza rodzinny błąd pierwszego rodzaju. Czysto statystyczne maski zachowały 1272–1408 neuronów, czyli dawały sparsity 31,25–37,89%. Mały test deweloperski wykazał jednak regresję PPL +12,13% dla biomedycyny i +17,90% dla sportu w wariancie pojedynczego eksperta.

Przed otwarciem holdoutu wybrano więc wariant operacyjny: maska statystyczna z dolnym limitem `min_keep_fraction=0.75`. Każdy ekspert zachowuje 1536 neuronów i usuwa 512. Wszyscy pojedynczy eksperci i MoE mieścili się wtedy w 10-procentowej bramce deweloperskiej.

Maski nie są identyczne: średni pairwise Jaccard wynosi 0,611 (zakres 0,595–0,625), 445 neuronów występuje we wszystkich maskach, jeden w żadnej, a 1602 mają zmienne członkostwo. Przy wymuszonym zachowaniu 75% szerokości duże przecięcie jest oczekiwane; wartość Jaccarda jest tylko nieznacznie wyższa od około 0,60 oczekiwanego dla niezależnych masek o tej gęstości.

## 7. Router i składanie MoE

Router to warstwa liniowa `640 → 6`, uczona w sposób nadzorowany na wejściach MLP warstwy 9. Zamrożony backbone uruchomiono tylko raz, a cechy zapisano w FP16 na CPU. Funkcja straty jest ważona tak, aby każdy tekst wnosił łączną wagę jeden; długie rozwiązania matematyczne nie dominują więc krótkich rekordów DBpedia.

Podział deweloperski był grupowy: 860 tekstów treningowych i 220 walidacyjnych, bez wspólnych grup źródłowych. Wyniki:

- dokładność tokenowa 96,02%, macro-F1 93,55%;
- dokładność na poziomie tekstu 98,64%, macro-F1 98,61%;
- trudniejsze klasy: sport F1 0,838 i polityka F1 0,839;
- pozostałe klasy: F1 0,981–0,993.

Router nie ma jawnej klasy „ogólne”. Zamiast tego maksimum softmax skalibrowano na 200 tekstach ogólnych: próg 0,8246666 ogranicza routing ogólny do 5%. Tokeny poniżej progu korzystają z oryginalnego gęstego MLP. Na walidacji domenowej coverage wyniósł 77,19%, a trafność wśród routowanych tokenów 99,83%.

Podczas inferencji tylko warstwa 9 jest zastępowana przez `HardRoutedMLP`. Pozostałe 17 warstw są identyczne z modelem bazowym. Dla tokenu uruchamiany jest jeden kompaktowy ekspert albo fallback; nie oblicza się wszystkich sześciu ekspertów. Wariant uruchomieniowy odtwarza ekspertów bezpośrednio z bazowego MLP i sześciu masek boolowskich. Porównanie tensor po tensorze potwierdziło jego identyczność z ekspertami użytymi w ewaluacji, zapisanymi wcześniej jako pełne `state_dict`.

## 8. Zamrożony holdout — wyniki

Konfigurację, hashe kodu, modelu, masek, routera i danych zamrożono przed oceną w `holdout_protocol_frozen.json`. Użyto 90 tekstów na domenę, 100 tekstów ogólnych, `max_length=128`, `batch_size=4` i seed 20260914.

| Domena | Base PPL | Ekspert sem. | Losowy | Magnituda | MoE | Δ ekspert | Δ MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| prawo | 24,623 | 26,358 | 26,087 | 28,972 | 25,877 | +7,05% | +5,09% |
| biomedycyna | 17,367 | 18,752 | 18,466 | 21,381 | 18,389 | +7,98% | +5,89% |
| sport | 11,843 | 12,582 | 12,383 | 13,560 | 12,188 | +6,24% | +2,91% |
| polityka | 11,405 | 12,064 | 12,790 | 13,026 | 11,610 | +5,78% | +1,80% |
| matematyka/LaTeX | 6,591 | 6,963 | 7,090 | 7,693 | 6,865 | +5,64% | +4,16% |
| Python | 10,869 | 11,144 | 11,614 | 12,546 | 11,087 | +2,53% | +2,00% |

Makrośredni względny wzrost PPL wynosi 5,87% dla pojedynczych ekspertów, 7,23% dla masek losowych, 16,94% dla masek magnitudowych i 3,64% dla routowanego MoE. Ważona tokenami domenowa PPL zmienia się z 12,980 do 13,483.

Na ogólnym holdoucie PPL zmienia się z 32,858 do 32,985, czyli +0,39%. Router wysyła do ekspertów 5,02% prawdziwych tokenów ogólnych.

### Routing na holdoucie po usunięciu paddingu z liczników

| Domena | Do dowolnego eksperta | Do poprawnego eksperta | Błędny pewny routing | Efektywna szerokość MLP-9 |
|---|---:|---:|---:|---:|
| prawo | 73,60% | 73,50% | 0,10% | 81,60% |
| biomedycyna | 74,09% | 74,07% | 0,02% | 81,48% |
| sport | 47,69% | 47,31% | 0,38% | 88,08% |
| polityka | 44,69% | 44,13% | 0,57% | 88,83% |
| matematyka/LaTeX | 79,34% | 79,22% | 0,13% | 80,16% |
| Python | 86,98% | 86,96% | 0,02% | 78,25% |

„Efektywna szerokość” to `fallback_fraction × 100% + routed_fraction × 75%`. Dotyczy tylko MLP warstwy 9; nie jest redukcją FLOPs całego modelu. Implementacja z maskami zwiększa również łączny rozmiar parametrów, bo przechowuje sześciu ekspertów i gęsty fallback. Warunkowy koszt obliczeń jednej warstwy maleje, ale czas ścienny może nie spaść przez koszt routingu i nieregularne małe batche.

## 9. Baselines i interpretacja

Ekspert semantyczny pokonał maskę magnitudową w każdej domenie. Baseline magnitudowy jest jednak tylko heurystyką opartą na iloczynie energii wag wejściowych i wyjściowych; w bramkowanym MLP istnieją symetrie skali, więc sama magnituda nie jest mocnym estymatorem przyczynowej ważności.

W porównaniu z pojedynczą maską losową ekspert semantyczny był lepszy dla polityki, matematyki/LaTeX i Pythona, a nieznacznie gorszy dla prawa, biomedycyny i sportu. Jeden losowy seed nie wystarcza do testu istotności. Następny rzetelny eksperyment powinien użyć co najmniej 20–50 losowych masek na domenę albo bootstrapować różnice straty per tekst.

Wynik nie uzasadnia jeszcze określenia „model ekspercki poprawia domenę”. Poprawne sformułowanie brzmi: „domenowo wybrane, przycięte MLP zachowuje większość jakości bazowej, a confidence-gated MoE ogranicza regresję przez fallback”.

## 10. Poprawka telemetrii po holdoucie

Pierwszy przebieg liczył pozycje paddingu w statystykach routingu, choć padding był prawidłowo wyłączony z loss. Po otwarciu holdoutu poprawiono wyłącznie licznik: `HardRoutedMLP` otrzymuje maskę uwagi i pomija padding w telemetrii. Masek, routera, progu ani wag nie zmieniono. Ponowny przebieg MoE dał bitowo identyczne PPL dla wszystkich siedmiu zbiorów (różnica 0,0). Audyt z hashami przed i po znajduje się w `post_holdout_telemetry_correction.json`.

## 11. Ograniczenia badawcze

1. Domena jest częściowo spleciona ze źródłem danych i stylem dokumentu. Potrzebny jest drugi, źródłowo niezależny benchmark dla każdej domeny.
2. Perplexity mierzy zachowanie modelowania języka, nie poprawność faktów. Dla prawa, biomedycyny, matematyki i Pythona należy dodać zadania QA/code execution z jednoznaczną metryką.
3. Badana jest jedna warstwa i jeden SAE. Nie wiadomo, czy wyniki przenoszą się na inne warstwy lub większe modele.
4. Korelacja SAE–MLP nie jest miarą przyczynową. Potrzebne są ablacje, patching albo interwencje na wybranych neuronach.
5. Max-statistic używa bloków tokenów, nie pełnej hierarchii dokumentów; bootstrap/permutacja grupowa na poziomie tekstów byłaby mocniejsza.
6. Router jest uczony nadzorowanie z etykietą tekstu przypisaną wszystkim tokenom. Wczesne i ogólne tokeny mają z natury zaszumioną etykietę; fallback ogranicza, ale nie usuwa tego problemu.
7. Tylko jeden losowy baseline nie pozwala oszacować wariancji przypadku.
8. Dziedziczone artefakty ewaluacyjne zachowują pełne zamaskowane `state_dict` (około 90 MiB łącznie), aby umożliwić audyt historycznego przebiegu. Nowy manifest uruchomieniowy używa wyłącznie sześciu masek (13 056 bajtów łącznie) i wag modelu bazowego; nie zależy od pełnych kopii ekspertów.

## 12. Reprodukcja etapów

Kolejność jest obowiązkowa: przygotowanie domen → zbiory → mapowanie → selektywność → pruning → router → test deweloperski → zamrożenie → holdout.

```bash
.venv/bin/python -m domain_mapping.domain_mlp_activation_mapping \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --model-name models/gemma-3-270m --tokenizer-name models/gemma-3-270m \
  --layer-num 9 --mapping-corpus pooled-domains \
  --max-texts-per-domain 80 --max-tokens-per-domain 60000 \
  --batch-size 4 --max-length 256 --cluster-aggregation sum \
  --metrics pearson --seed 20260914

.venv/bin/python -m domain_mapping.validate_domain_sae_selectivity \
  --mapping-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/domain_mlp_mapping \
  --n-bootstrap 2000 --minimum-auc 0.60 --seed 20260914

.venv/bin/python -m domain_mapping.domain_mlp_pruning \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --mapping-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/domain_mlp_mapping \
  --model-name models/gemma-3-270m --tokenizer-name models/gemma-3-270m \
  --layer-num 9 --output-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/domain_mlp_experts_floor75 \
  --metric abs_pearson_r --tau-method per-domain-null \
  --null-permutations 100 --null-percentile 95 --null-block-size 32 \
  --min-keep-fraction 0.75 --zero-bias --seed 20260914

.venv/bin/python -m moe.router_training \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --model-name models/gemma-3-270m --tokenizer-name models/gemma-3-270m \
  --layer-num 9 --epochs 15 --learning-rate 0.001 --weight-decay 0.0001 \
  --batch-size 8 --feature-batch-size 8192 --max-length 256 \
  --val-fraction 0.2 --target-general-route-rate 0.05 --seed 20260914

.venv/bin/python -m evaluation.moe_validation \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --experts-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/domain_mlp_experts_floor75 \
  --router-path data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/router/router.pt \
  --validation-csv data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/holdout_general.csv \
  --domain-eval-csv data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/holdout_domains.csv \
  --model-name models/gemma-3-270m --tokenizer-name models/gemma-3-270m \
  --layer-num 9 --batch-size 4 --max-length 128 --seed 20260914
```

Do zweryfikowanego złożenia dokładnie ocenionego wariantu modelu wystarczy manifest runtime:

```bash
.venv/bin/python -m moe.moe_assembly \
  --bundle data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/moe_bundle.json
```

Loader sprawdza rozmiar i SHA-256 każdego pliku potrzebnego w runtime, a następnie odtwarza kompaktowych ekspertów z modelu bazowego i masek. Pełna jawna postać polecenia, przydatna przy eksperymentach z innymi artefaktami, to:

```bash
.venv/bin/python -m moe.moe_assembly \
  --mode moe \
  --domain-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --experts-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/domain_mlp_experts_floor75 \
  --router-path data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914/router/router.pt \
  --model-name models/gemma-3-270m --layer-num 9
```

Próg confidence jest domyślnie pobierany z `router.pt`; argument CLI służy wyłącznie do jawnego override'u.

Manifest można odtworzyć po kontrolowanej zmianie artefaktów poleceniem:

```bash
.venv/bin/python -m moe.prepare_gemma_moe_bundle \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --model-dir models/gemma-3-270m --workspace-root .
```

Nie należy uruchamiać tego polecenia wyłącznie po to, aby „zaakceptować” niezamierzoną zmianę hasha; najpierw trzeba ustalić, dlaczego zależność runtime się zmieniła.

## 13. Artefakty

- `domains.json` — zamrożone domeny i listy cech SAE;
- `dataset_manifest.json` — rewizje, liczności, hashe i kontrola przecieków;
- `domain_mlp_mapping/` — wspólne aktywacje oraz korelacje 6 × 2048;
- `domain_mlp_mapping/selectivity/` — AUC, CI i macierz pairwise;
- `domain_mlp_experts_null95/` — czysto statystyczny wariant badawczy;
- `domain_mlp_experts_floor75/` — wybrany wariant operacyjny;
- `router/router.pt` — router i skalibrowany próg;
- `holdout_protocol_frozen.json` — manifest sprzed otwarcia holdoutu;
- `moe_validation_holdout_frozen/` — PPL i baselines;
- `moe_routing_telemetry_corrected/` — skorygowane liczniki bez paddingu;
- `post_holdout_telemetry_correction.json` — audyt poprawki;
- `moe_bundle.json` — zweryfikowany manifest runtime: bazowa Gemma, sześć masek, router, próg i proweniencja;
- `moe/prepare_gemma_moe_bundle.py` — odtwarzalne generowanie manifestu i kontrola spójności masek.
