# `sae_pipeline/autoencoder_training.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/sae_pipeline/autoencoder_training.py) · [źródło aktywacji](activations_collecting.md) · [analiza cech](features_analysis.md)

## Rola modułu

Moduł definiuje własny Top-K Sparse Autoencoder i jego główny, resumowalny
trener. Aktywacje modelu są pobierane chunkami, zużywane przez SAE i zwalniane;
pełny zbiór aktywacji nie jest materializowany na dysku ani w RAM.

## Model `TopKSAE`

Dla wejścia `x ∈ R^D` i `F = D × expansion_factor`:

```text
x_centered = x - b_dec
a = ReLU(W_enc x_centered + b_enc)
z_i = a_i, jeżeli i należy do TopK(a); w przeciwnym razie 0
x_hat = z W_dec + b_dec
loss = MSE(x_hat, x)
```

Kształty parametrów PyTorch:

| Parametr | Kształt |
| --- | --- |
| `encoder.weight` | `[F,D]` |
| `encoder.bias` | `[F]` |
| `W_dec` | `[F,D]`; jeden wiersz jest kierunkiem jednej cechy |
| `b_dec` | `[D]` |

`ReLU` usuwa wartości ujemne, a `torch.topk` wybiera `k` największych wartości
wzdłuż cech. Jeśli dodatnich wartości jest mniej niż `k`, do top-k mogą wejść
zera, więc liczba faktycznie niezerowych cech może być mniejsza niż `k`.

Po inicjalizacji i po każdym kroku optymalizatora każdy wiersz `W_dec` jest
normalizowany do normy L2 równej 1 (z dolnym ograniczeniem mianownika `1e-8`).
Ogranicza to swobodę skalowania `z` i dekodera. Wagi enkodera i dekodera nie są
wiązane ze sobą; „tied-dimension” w docstringu oznacza zgodny wymiar wejścia i
rekonstrukcji, nie `W_enc = W_decᵀ`.

## Trening strumieniowy — przebieg krok po kroku

1. Walidowane są dodatnie rozmiary i tryb logowania, po czym `set_seed()`
   ustawia RNG.
2. Model bazowy trafia na wybrane urządzenie i do `eval()`. Jego
   `config.hidden_size` musi być równy `d_model`; przy `d_model=None` wymiar
   jest wykrywany automatycznie.
3. `open_token_store()` sprawdza tablice tokenów. `effective_end` ogranicza
   zbiór przez `max_sequences`.
4. Z maski liczona jest dokładna liczba ważnych tokenów oraz liczba kroków SAE.
   Ponieważ każdy chunk jest batchowany oddzielnie, liczba kroków jest sumą
   `ceil(valid_tokens_in_chunk / batch_size_sae)`, a nie jednym `ceil` po całości.
5. Tworzony jest SAE, `AdamW(weight_decay=0)` i `CosineAnnealingLR` od
   `learning_rate` do `min_learning_rate` przez oczekiwaną liczbę kroków
   wszystkich epok.
6. Jeżeli włączono `resume` i istnieje główny checkpoint, każdy klucz bieżącej
   konfiguracji musi być identyczny z zapisanym. Odtwarzane są wagi, optimizer,
   scheduler, historia, kursor, najlepsza strata, liczniki użycia i RNG.
7. Dla każdej epoki generator aktywacji zaczyna od `next_sequence` (tylko dla
   pierwszej wznawianej epoki), a w kolejnych od zera.
8. Wewnątrz każdego chunka aktywacje `[N,D]` są konwertowane do float32 i
   deterministycznie tasowane lokalnym generatorem o ziarnie
   `seed + epoch×1_000_003 + chunk_start`.
9. Każdy minibatch trafia na urządzenie. Forward SAE daje rekonstrukcję i
   `sparse_acts`; optymalizowana jest wyłącznie średnia strata MSE.
10. Gradient jest obcinany do globalnej normy `1.0`, wykonywany jest krok
    AdamW, normalizacja dekodera i krok schedulera.
11. Dla każdej cechy `usage_counts` zwiększa się o liczbę tokenów, dla których
    jej aktywacja jest niezerowa. Historia zapisuje loss, LR i łączną liczbę
    niezerowych aktywacji batcha.
12. Po zużyciu całego chunka kursor `next_sequence` jest przesuwany. Zgodnie z
    `save_every_n_chunks` powstają checkpoint kroku i checkpoint główny.
13. Po pełnej epoce liczona jest średnia nieważona po stratach batchy. Jeśli
    jest najlepsza, zapisywany jest `*_best.pt`; checkpoint główny zawsze jest
    aktualizowany.
14. Na końcu status ma wartość `complete` albo `paused`, a raport zawiera
    odsetek cech z zerowym licznikiem użycia.

## Semantyka `max_steps`

Limit jest celowo kontrolowany na granicach chunków, aby checkpoint nie musiał
przechowywać lokalnej permutacji i indeksu minibatcha. Po osiągnięciu limitu w
środku chunka kod ustawia `stop_training`, ale kończy pozostałe minibatch'e
tego chunka. Dlatego `max_steps` jest limitem miękkim i może zostać przekroczony
o liczbę pozostałych minibatchy w bieżącym chunku. Kursor checkpointu pozostaje
za to dokładny i wskazuje następny, w całości nieprzetworzony chunk.

## Format checkpointu `topk_sae_streaming_v2`

| Pole | Znaczenie |
| --- | --- |
| `status` | `running`, `paused`, `best` albo `complete` |
| `config` | parametry danych, modelu, SAE i treningu wymagane przy resume |
| `model_state_dict` | parametry TopKSAE |
| `optimizer_state_dict` | momenty i stan AdamW |
| `scheduler_state_dict` | stan cosine annealing |
| `history` | ograniczone listy loss/LR/epoch loss/aktywności |
| `epoch`, `next_sequence` | dokładny kursor wznowienia na granicy chunka |
| `global_step` | liczba wykonanych aktualizacji |
| `best_loss` | najlepsza średnia epoki |
| `usage_counts` | użycia każdej cechy po wszystkich dotychczasowych tokenach |
| `rng_state` | stany PyTorch, NumPy i `random` |

Zapis jest atomowy w obrębie filesystemu: najpierw powstaje plik `.tmp`, potem
`os.replace()` podstawia go pod docelową nazwę. Checkpointy `*_step_<n>.pt` są
rotowane według czasu modyfikacji; `keep_last_checkpoints=0` usuwa wszystkie
checkpointy krokowe, ale zachowuje główny i najlepszy.

## Katalog klas i funkcji

| Element | Odpowiedzialność |
| --- | --- |
| `TopKSAE.__init__()` | waliduje `D`, expansion i `k`; tworzy parametry i inicjalizuje dekoder |
| `TopKSAE.forward(x)` | centrowanie, ReLU, top-k, rzadki tensor i rekonstrukcja |
| `TopKSAE.normalize_decoder()` | normalizuje każdy kierunek dekodera bez gradientu |
| `TopKSAE.config()` | zwraca minimalną konfigurację architektury |
| `_atomic_torch_save(payload, path)` | atomowy zapis checkpointu przez plik tymczasowy |
| `_trim_history(history, max_history)` | zachowuje tylko ostatnie wpisy każdej serii; dla wartości `<1` czyści serie |
| `_cleanup_step_checkpoints(...)` | rotuje checkpointy krokowe |
| `_load_checkpoint(path)` | ładuje pełny obiekt z kompatybilnością starszego `torch.load` |
| `_checkpoint_payload(...)` | składa kompletny, resumowalny stan runu |
| `_restore_rng(payload)` | odtwarza dostępne stany trzech RNG |
| `_count_valid_tokens(...)` | sumuje maskę bounded chunkami |
| `_count_sae_steps(...)` | liczy dokładną sumę kroków przy niezależnym batchowaniu chunków |
| `train_autoencoder_streaming(...)` | kanoniczny trener bounded/resumable |
| `train_autoencoder(...)` | historyczny trener pełnego pliku `[N,S,D]` |

## Ścieżka historyczna `train_autoencoder()`

Legacy trainer otwiera `activations/activations_layer_<L>.npy` jako memmapę,
ale wymaga kształtu `[N,S,D]`. Tworzy globalną permutację `N×S` indeksów dla
każdej epoki, odczytuje pozycje, uczy ten sam model i zapisuje prostszy format
`topk_sae_legacy_v2`. Nie obsługuje resume, jawnej maski paddingu,
`usage_counts`, checkpointów krokowych ani RNG. Nowe runy powinny korzystać z
trenera strumieniowego.

## Ograniczenia i konsekwencje metodologiczne

- Strata to czyste MSE; nie ma osobnej kary L1, ghost gradients, resamplingu
  martwych cech ani auxiliary loss. Rzadkość wymusza operator top-k.
- Średnia epoki jest średnią po batchach, nie średnią ważoną liczbą tokenów;
  ostatni, mniejszy batch chunka ma taką samą wagę jak pełny batch.
- `chunk_loss_sum` i `chunk_count` są obliczane, lecz nie wpływają obecnie na
  checkpoint ani wybór modelu.
- Lokalne tasowanie nie miesza tokenów między chunkami. Oszczędza pamięć i
  zapewnia deterministyczne resume, ale jest słabsze niż globalna permutacja.
- `usage_counts` opisuje niezerowe aktywacje top-k, a nie przekroczenie progu
  używanego później w analizie cech.

