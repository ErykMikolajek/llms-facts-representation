# Zakres i status plików

## Reguła wyboru

Udokumentowano każdy kanoniczny plik `.py`, który:

- przygotowuje dane do eksperymentu,
- pozyskuje albo analizuje aktywacje modelu lub SAE,
- trenuje element metody,
- buduje, modyfikuje albo waliduje wariant modelu,
- albo stanowi bezpośrednią infrastrukturę uruchamiania tych etapów.

Zakres obejmuje kanoniczne moduły z pakietów etapowych, współdzielony
`common/utils.py` i `kaggle/cli/kaggle_pipeline.py`. Dokumentacja obejmuje też notebook
niezależnego benchmarku, ponieważ jest wykonywalną częścią protokołu, oraz
opisuje testy jako zabezpieczenia kontraktów.

## Pliki kanoniczne i kopie do pracy

Katalog `praca_tex/source_code/` zawiera kopie kodu przygotowane jako materiał
źródłowy do pracy. Część z nich jest bajtowo identyczna z plikami głównymi, a
część reprezentuje starszy stan implementacji. Nie są one wykonywane przez
główny pipeline i nie stanowią drugiego, niezależnego eksperymentu.

Dokumentacja opisuje zatem **wersje z pakietów etapowych jako źródło prawdy**.
Każda strona podaje nazwę odpowiadającego pliku; kopii w
`praca_tex/source_code/` nie nadano osobnych stron, ponieważ prowadziłoby to do
dwóch rozbieżnych opisów tej samej roli.

!!! danger "Synchronizacja kopii"
    Przed włączeniem listingu kodu do ostatecznej wersji pracy należy ponownie
    zsynchronizować `praca_tex/source_code/` z wersjami kanonicznymi. Różnice
    dotyczą między innymi nowszych zabezpieczeń metodologicznych i obsługi
    checkpointów.

## Pliki świadomie wyłączone

| Plik lub grupa | Powód |
| --- | --- |
| `tests/test_features_analysis.py` | test regresyjny; weryfikuje analizę, ale nie realizuje etapu eksperymentu |
| `tests/test_gemma_scope_analysis.py` | test jednostkowy kontraktu Gemma Scope |
| `tests/test_moe_pipeline.py` | testy kontraktów domen, mappingu, pruningu, routera, compact MLP, bundle i niezależności ewaluacji |
| `tests/test_moe_benchmark.py` | zgodność pamięciooszczędnego NLL, kodowania completion i metryk benchmarku |
| `tests/test_semantic_domain_cross_view.py` | test korekty BH i kohezji cross-view |
| `kaggle/cli/test_kaggle_pipeline.py` | test infrastruktury Kaggle |
| pozostałe notebooki `.ipynb` | notebooki SAE wywołują opisane moduły; notebooki benchmarku MoE i pojedynczych ekspertów są opisane osobno |
| kod zależności i środowiska wirtualnego | nie należy do repozytorium badawczego |

Testy wykorzystano do sprawdzenia intencji i warunków brzegowych opisywanych
funkcji, ale nie otrzymały osobnych stron modułowych.

## Status części eksperymentu

| Część | Status w repozytorium | Interpretacja |
| --- | --- | --- |
| sekwencjonowanie i memmapy | aktywna | kanoniczne przygotowanie danych |
| strumieniowanie aktywacji | aktywne | ścieżka storage-bounded |
| trening Top-K SAE | aktywny | główny własny eksperyment Pythia |
| analiza Top-K SAE | aktywna | obsługuje checkpoint/resume |
| analiza Gemma Scope 2 | aktywna | osobny tor bez treningu SAE |
| semantic domain triage | wykonany dla Gemmy | sześć ręcznie zaakceptowanych domen; nazwy nadal wymagają inspekcji |
| stabilność i cross-view | wykonane | analiza wrażliwości i zgodności reprezentacji, nie niezależna replikacja |
| dane development/holdout | wykonane | brak exact hash/source-group overlap; pozostaje confounding źródła |
| topographic mapping | eksploracyjny / historyczny | residual stream, nie fizyczne neurony MLP |
| właściwy mapping MLP | wykonany dla Gemmy | wspólna pula 60 tys. tokenów i rzeczywiste gated-MLP postactivations |
| selektywność SAE | wykonana | wszystkie 6 domen przeszło ustaloną bramkę AUC/CI |
| pruning ekspertów | wykonany | 1536/2048 neuronów per ekspert; próg null plus floor 75% |
| router i MoE | wykonane jako PoC | hard top-1, confidence fallback, mask-only runtime |
| walidacja PPL | holdout zużyty | MoE +0,39% general; brak poprawy jakości względem base |
| niezależny benchmark v2 | wykonany i zużyty | raport wyników w `results/gemma_moe_validation` |
| follow-up benchmark v3 | gotowy do Kaggle | 600 świeżych zadań, MC order variants i prefill runtime |
| benchmark single-expert | gotowy do Kaggle | każdy ekspert na własnej domenie; opcjonalna macierz cross-domain |
| automatyzacja Kaggle | operacyjna | zarządza wznowieniami, nie zmienia metody analizy |

## Data przeglądu

Opis sporządzono na podstawie stanu plików roboczych repozytorium z
15 września 2026 r. Dokumentacja opisuje również niezacommitowane zmiany obecne
w chwili przeglądu; przy późniejszej modyfikacji sygnatur lub formatów
artefaktów należy zaktualizować odpowiednią stronę.
