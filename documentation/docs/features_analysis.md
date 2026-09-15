# `sae_pipeline/features_analysis.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/sae_pipeline/features_analysis.py) · [checkpoint SAE](autoencoder_training.md) · [następny etap: domeny](semantic_domain_triage.md)

## Rola modułu

Moduł interpretuje wytrenowany Top-K SAE dla Pythii/TinyStories. Ponownie
uruchamia model nad wybranym fragmentem przygotowanych sekwencji i gromadzi dla
każdej cechy:

- liczbę, częstość, średnią i maksimum aktywacji powyżej progu;
- najczęstsze tokeny wyzwalające;
- ograniczoną liczbę najsilniejszych przykładów z kontekstem;
- tokeny promowane i tłumione przez przybliżony logit lens;
- oddzielne flagi obserwacji w analizie i użycia podczas treningu.

Analiza ma własny checkpoint i może zostać wznowiona bez utraty statystyk.

## Struktury danych

### `FeatureExample`

Przechowuje wartość aktywacji, tekst i ID tokenu wyzwalającego, pozycję,
kontekst przed/po, pełny kontekst z oznaczeniem `[[token]]` oraz globalny indeks
sekwencji.

### `FeatureAnalysis`

Jest finalną kartą cechy. `observed_in_analysis` mówi, czy cecha przekroczyła
`activation_threshold` w badanej próbce. `observed_in_training`, jeśli
dostępne są `usage_counts`, odnosi się do całego dotychczasowego treningu.
`is_dead` jest wyznaczane na podstawie treningu, a nie braku w próbce analizy.

### `ComprehensiveFeatureAnalyzer`

W trakcie streamingu utrzymuje:

- słowniki sum i liczników tylko dla aktywnych cech;
- zagnieżdżone liczniki tekstów tokenów;
- min-heap o pojemności `top_k_examples` per cecha;
- pełny zestaw pustych `FeatureAnalysis`, aby raport JSON zawierał także cechy
  niezaobserwowane.

## Przetwarzanie batcha — dokładny przebieg

1. Analizator akceptuje legacy tensor `[B,S,D]` albo streamingowy `[N,D]`.
2. Dla `[N,D]` opcjonalne `sequence_tokens` i `sequence_mask` zachowują
   prawdziwe granice sekwencji. `nonzero(mask)` tworzy mapę z płaskiego indeksu
   ważnego tokenu do `(wiersz, pozycja)`.
3. SAE działa bez gradientu w float32 i zwraca `sparse_acts`.
4. Dla każdej z `F` cech tworzona jest maska `activation > threshold`.
5. Dla pozycji spełniających próg aktualizowane są suma, licznik, maksimum i
   licznik zdekodowanego tokenu.
6. Jeżeli heap cechy nie jest pełny albo wynik jest większy od bieżącego
   minimum, dekodowany jest kontekst o promieniu `context_size`.
7. Dla streamingu kontekst jest pobierany z jednego prawdziwego wiersza, a
   pozycja w skompresowanym wektorze ważnych tokenów jest odtwarzana z maski.
8. Licznik przetworzonych tokenów zwiększa się o liczbę wejściowych wektorów;
   w kanonicznej ścieżce są to wyłącznie ważne tokeny.

Heap zawiera trójki `(score, serial, example)`. Monotoniczny `serial` rozstrzyga
remisy bez próby porównywania obiektów dataclass.

## Logit lens

Dla kierunku dekodera cechy `f` obliczane jest:

```text
logits_f = W_dec[f] @ W_U.T
```

gdzie `W_U` to waga `model.get_output_embeddings()`. Największe wartości są
interpretowane jako tokeny promowane przez kierunek, a najmniejsze jako
tłumione. Obie macierze są konwertowane do float32, a cechy liczone blokami
`logit_batch_size`, aby nie tworzyć pełnej macierzy `[F,V]`.

!!! warning "To jest przybliżenie"
    Kierunek cechy jest rzutowany bezpośrednio przez unembedding. Obliczenie nie
    przeprowadza kierunku przez pozostałe warstwy i nie modeluje wszystkich
    transformacji końcowych. Wynik służy jako wskazówka interpretacyjna, nie
    jako dokładna zmiana logitów po interwencji.

## Finalizacja statystyk

`finalize_analysis()` dla każdej cechy:

1. ustala obserwację w analizie z licznika progowego;
2. oblicza średnią i częstość względem wszystkich przetworzonych tokenów;
3. jeśli istnieją training usage counts, ustala `observed_in_training` i
   `is_dead`;
4. sortuje heap malejąco;
5. wybiera 10 najczęstszych tokenów wyzwalających.

Bez `usage_counts` cechy niezaobserwowane w analizie nie są automatycznie
oznaczane jako martwe. To celowe rozdzielenie dwóch zakresów obserwacji.

## Sampling sekwencji

`_analysis_ranges()` zwraca okna półotwarte:

- `head`: pierwsze `max_sequences`;
- `tail`: ostatnie `max_sequences`;
- `uniform`: rozłączne okna o rozmiarze `chunk_sequences`, rozstawione przez
  możliwie równe luki; łącznie dokładnie `max_sequences` unikalnych wierszy;
- brak limitu: całe `[0,N)`.

Jeśli uniform wymaga tylko jednego okna, jest ono centrowane w zbiorze.

## Główna funkcja `analyze_sae()`

1. Inicjalizuje analizator i otwiera tokeny/maskę jako memmapy.
2. Buduje zakresy próbkowania i konfigurację checkpointu.
3. Opcjonalnie ładuje checkpoint `pythia_feature_analysis_checkpoint_v1`.
   Ścieżki tokenów mogą zmienić katalog montowania Kaggle, ale ich nazwy plików
   muszą pozostać zgodne; reszta konfiguracji musi być identyczna.
4. Stary checkpoint bez `context_mapping` jest migrowany z płaskich indeksów
   chunka do prawdziwych współrzędnych sekwencji.
5. Dla każdego zakresu wywołuje `iter_activation_chunks()`, pobiera odpowiadający
   wycinek tokenów i maski oraz przekazuje płaskie aktywacje wraz z mapą sekwencji.
6. Po każdym skonfigurowanym interwale atomowo zapisuje akumulatory i kursor
   następnego zakresu/sekwencji.
7. Po przejściu danych wykonuje logit lens, finalizację i zapis raportów.
8. Checkpoint końcowy otrzymuje `completed=True`, summary i kursor za ostatnim zakresem.

## Finalizacja bez nowej inferencji

`finalize_analysis_checkpoint()` ładuje zapisany akumulator, ewentualnie
migruje stare współrzędne, oblicza logit lens z modelu bazowego i zapisuje
raporty. Nie uruchamia forward passów na tokenach. Raport może być częściowy,
co funkcja jawnie zwraca przez `checkpoint_completed` i `processed_chunks`.

## Artefakty

| Plik | Zawartość |
| --- | --- |
| `analysis/analysis_checkpoint.pt` | resumowalny akumulator i kursor |
| `analysis/features_analysis.json` | summary oraz wszystkie cechy, także martwe/niezaobserwowane |
| `analysis/features_analysis.txt` | raport czytelny dla człowieka; domyślnie bez cech martwych |

JSON zachowuje top promoted/suppressed tokens jako pary tekst–logit i pełne
konteksty. Jest wejściem offline dla semantic domain triage.

## Katalog funkcji i metod

| Element | Odpowiedzialność |
| --- | --- |
| `_atomic_torch_save()` | bezpieczny zapis checkpointu przez `.tmp` i `os.replace()` |
| `_load_torch_checkpoint()` | zgodne ładowanie pełnego obiektu PyTorch |
| `_analysis_checkpoint_config_matches()` | porównuje kontrakt resume, tolerując relokację mountu przy tej samej nazwie pliku |
| `_json_safe()` | konwertuje skalarne typy NumPy i zagnieżdżone struktury do JSON |
| `_save_features_to_json()` | serializuje summary i posortowane karty cech |
| `_save_features_to_text()` | tworzy raport tekstowy posortowany częstością |
| `ComprehensiveFeatureAnalyzer.__init__()` | waliduje liczniki treningowe i inicjalizuje bounded state |
| `state_dict()` / `load_state_dict()` | serializuje i odtwarza cały stan analizy, w tym heap'y |
| `migrate_flattened_context_examples()` | naprawia historyczne indeksy i konteksty przecinające granice sekwencji |
| `_get_context()` | dekoduje lewy/prawy kontekst i oznacza trigger |
| `process_batch()` | koduje SAE i aktualizuje statystyki/heap'y |
| `compute_logit_lens()` | blokowe `W_dec @ W_Uᵀ` dla wszystkich cech |
| `finalize_analysis()` | materializuje statystyki w kartach cech |
| `get_summary_stats()` | agreguje liczbę cech, częstości i liczbę tokenów |
| `_analysis_ranges()` | buduje rozłączne zakresy head/tail/uniform |
| `analyze_sae()` | pełny streaming, resume, raporty i checkpoint końcowy |
| `finalize_analysis_checkpoint()` | raportuje istniejący akumulator bez analizy kolejnych sekwencji |

## Ograniczenia i pułapki interpretacyjne

- „Obserwowana” oznacza `activation > activation_threshold`, nie samo wejście
  cechy do top-k. Zmiana progu zmienia statystyki i wymaga nowego checkpointu.
- Pętla przechodzi po wszystkich `F` cechach dla każdego batcha, mimo rzadkiego
  `z`; jest poprawna, ale może być kosztowna dla szerokich SAE.
- Liczniki tokenów wyzwalających nie są ograniczone rozmiarem słownika, w
  przeciwieństwie do implementacji Gemma Scope.
- Konfiguracja checkpointu nie identyfikuje hashem wag SAE ani modelu bazowego.
  Należy ręcznie pilnować, aby resume łączyło ten sam checkpoint, warstwę i model.
- `training_usage_counts` również nie jest częścią porównywanej konfiguracji;
  podmiana go przed finalizacją może zmienić etykiety martwych cech.

