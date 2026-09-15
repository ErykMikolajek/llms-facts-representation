# Eksperyment Gemma 3 270M: stan wykonany

Ta strona opisuje **rzeczywiście wykonany** przebieg, a nie tylko możliwości
skryptów. Eksperyment zakończono 14 września 2026 r.; niezależny benchmark
kompetencji został następnie przygotowany i jest obecnie uruchomiony na
Kaggle. Jego wyników nie ma jeszcze w repozytorium.

## Punkt wyjścia

Analiza `analysis_checkpoint_gemma_scope2_270m.pt` dotyczy modelu Gemma 3
270M, warstwy 9 i site'u `resid_post`. Użyto SAE Gemma Scope 2
`gemma-scope-2-270m-pt-res/layer_9_width_16k_l0_medium`:

| Wielkość | Wartość |
| --- | ---: |
| przeanalizowane tokeny | 99 999 989 |
| sekwencje | 103 513 |
| szerokość SAE | 16 384 |
| zaobserwowane cechy | 16 343 |
| `hidden_size` / `intermediate_size` | 640 / 2048 |

Raport końcowy `feature_analysis.json` został użyty do triażu na podstawie
dodatnich tokenów logit lens. Nie należy mylić tego przebiegu z wcześniejszym
smoke testem częściowego checkpointu, który korzystał z trigger tokens.

## Od triażu do sześciu domen

Po analizie stabilności i zgodności dwóch reprezentacji ręcznie zaakceptowano
sześć klastrów:

| ID | Domena | Klaster | Cechy SAE | AUC one-vs-rest | 95% CI |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | prawo i orzecznictwo | 20 | 161 | 0,998 | [0,995; 1,000] |
| 1 | biomedycyna | 17 | 396 | 1,000 | [1,000; 1,000] |
| 2 | sport | 16 | 117 | 0,999 | [0,996; 1,000] |
| 3 | polityka i wiadomości | 22 | 123 | 0,957 | [0,934; 0,975] |
| 4 | matematyka / zapis LaTeX | 15 | 322 | 0,999 | [0,997; 1,000] |
| 5 | Python | 5 | 149 | 0,995 | [0,987; 1,000] |

Wszystkie domeny przeszły ustaloną bramkę: AUC co najmniej 0,60 i dolna
granica bootstrapowego CI powyżej 0,50. Najtrudniejszą parą była polityka
kontra prawo, AUC 0,808. Wysokie wyniki nie dowodzą jeszcze czystej semantyki:
część separacji może pochodzić ze stylu i źródła zbioru.

## Dane development i holdout

Automatycznie zbudowano 180 tekstów development i 90 holdout na domenę oraz
200/100 tekstów ogólnych. Łącznie daje to 1280 rekordów development i 640
holdout. Hashe treści i `source_group` nie przecinają się między podziałami.
Dokładne źródła i rewizje opisuje strona [Przygotowanie danych](domain_datasets.md).

## Mapowanie i selekcja neuronów

Wspólny korpus mappingu zawierał 60 000 tokenów z 351 przemieszanych tekstów.
Jeden forward zebrał macierz prawdziwych postaktywacji gated MLP
`[60000, 2048]`, sygnały sześciu klastrów SAE oraz identyfikatory tekstów,
tokenów i domen źródłowych. Sygnał domeny to suma aktywacji jej cech SAE;
ranking neuronu oparto na `abs(Pearson r)`.

Próg null wyznaczono przez 100 permutacji bloków po 32 tokeny. Dla każdej
permutacji brano maksimum korelacji po 2048 neuronach, a `tau` było jego 95.
percentylem. Czyste maski statystyczne zachowywały 1272–1408 neuronów, lecz
warianty biomedyczny i sportowy przekroczyły na development limit 10% regresji
PPL. **Przed otwarciem holdoutu** zamrożono zatem `min_keep_fraction=0.75`:
każdy ekspert zachowuje 1536 i usuwa 512 neuronów, czyli 25% warstwy.

Średni Jaccard masek wyniósł 0,611 przy wartości około 0,60 oczekiwanej dla
niezależnych masek o gęstości 75%. Maski nie są kopiami, ale samo to nie jest
mocnym dowodem odrębnych obwodów domenowych.

## Router i runtime MoE

Liniowy router `640 → 6` wytrenowano na raz zbuforowanych wejściach warstwy 9.
Każdy tekst ma łączną wagę 1, więc długie rekordy nie dominują straty. Podział
grupowy zawierał 860 tekstów train i 220 validation, bez wspólnych grup.

| Metryka | Wynik |
| --- | ---: |
| token accuracy / macro-F1 | 0,9602 / 0,9355 |
| sample accuracy / macro-F1 | 0,9864 / 0,9861 |
| próg confidence | 0,8246666193 |
| routing tekstu ogólnego na development | 5,00% |
| coverage domenowe / accuracy wśród routowanych | 77,19% / 99,83% |

`HardRoutedMLP` wykonuje dla tokenu dokładnie jeden kompaktowy ekspert albo
oryginalny gęsty MLP jako fallback. Tylko MLP warstwy 9 jest zastępowane;
pozostałe warstwy modelu są bez zmian. Runtime `base_masks` odtwarza eksperta
z bazowych wag i boolowskiej maski. Porównanie tensor po tensorze potwierdziło
zgodność z historycznymi pełnymi checkpointami ekspertów.

## Zamrożony holdout

Przed oceną zapisano `holdout_protocol_frozen.json` z hashami kodu, danych,
masek, routera i modelu. Użyto 90 tekstów na domenę, 100 ogólnych,
`max_length=128`, `batch_size=4` i seed 20260914.

| Domena | Base PPL | Ekspert | Losowy | Magnituda | MoE | Δ MoE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prawo | 24,623 | 26,358 | 26,087 | 28,972 | 25,877 | +5,09% |
| biomedycyna | 17,367 | 18,752 | 18,466 | 21,381 | 18,389 | +5,89% |
| sport | 11,843 | 12,582 | 12,383 | 13,560 | 12,188 | +2,91% |
| polityka | 11,405 | 12,064 | 12,790 | 13,026 | 11,610 | +1,80% |
| matematyka/LaTeX | 6,591 | 6,963 | 7,090 | 7,693 | 6,865 | +4,16% |
| Python | 10,869 | 11,144 | 11,614 | 12,546 | 11,087 | +2,00% |

Ogólna PPL wzrosła z 32,858 do 32,985, czyli o 0,39%. Selekcja semantyczna
pokonała baseline magnitudowy w 6/6 domen, ale pojedynczą maskę losową tylko w
3/6. Żaden ekspert ani MoE nie poprawił PPL modelu bazowego. Poprawna
interpretacja brzmi: jest to PoC zachowania jakości przy warunkowym użyciu
przyciętej jednej warstwy, nie dowód wzrostu wiedzy domenowej.

## Korekta telemetrii po holdoucie

Pierwszy przebieg poprawnie usuwał padding z loss, lecz liczył go w routing
stats. Po otwarciu holdoutu zmieniono wyłącznie telemetrię: wrapper otrzymuje
maskę uwagi i nie zlicza paddingu. Masek, wag, routera, progu ani wyboru modelu
nie zmieniono. Powtórzone PPL były bitowo identyczne; audyt zapisano w
`post_holdout_telemetry_correction.json`. Poprawiony routing do dowolnego
eksperta wyniósł odpowiednio 73,60%, 74,09%, 47,69%, 44,69%, 79,34% i 86,98%
dla sześciu domen oraz 5,02% dla danych ogólnych.

## Status końcowy

Holdout PPL jest **zużyty** i nie wolno na nim dalej stroić pipeline'u.
Niezależny benchmark v2 został wykonany i jest zużyty. Nie wykazał przewagi
kompetencyjnej MoE, a wykryty błąd ekstrakcji Pythona oraz efekt podłogi GSM8K
stały się podstawą follow-upu v3. Nowe 600 rekordów nie powtarza identyfikatorów
v1/v2; wyniki v2 pozostają osobnym, historycznym raportem i nie służą strojeniu.
