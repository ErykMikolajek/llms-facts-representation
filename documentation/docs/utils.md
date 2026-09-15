# `common/utils.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/common/utils.py)

## Rola modułu

Mały moduł infrastrukturalny ujednolica dobór zasobów obliczeniowych i
reprodukowalność losowania. Jest importowany przez prawie wszystkie etapy
lokalne; jego zachowanie wpływa więc zarówno na wydajność, jak i możliwość
powtórzenia runu.

## Katalog funkcji

### `find_n_proc()`

Zwraca `max(cpu_count() - 1, 1)`. Pozostawia jeden logiczny rdzeń dla systemu,
ale nigdy nie zwraca mniej niż jednego procesu. Wykorzystuje go równoległe
liczenie tokenów w ścieżce CSV.

### `tokenizer_vocab_fingerprint(tokenizer)`

Buduje stabilny SHA-256 semantyki słownika. Pobiera `tokenizer.get_vocab()`,
sortuje pary najpierw po numerycznym `token_id`, potem po tekście tokenu, i do
hasha dopisuje naprzemiennie ID oraz UTF-8 tokenu, rozdzielone bajtem NUL.

Sama zgodność `vocab_size` nie gwarantuje, że dwa tokenizery przypisują te same
znaczenia tym samym ID. Fingerprint jest zapisywany w analizie Gemma Scope i
sprawdzany przez semantic triage przed użyciem zapisanych `token_id`. Nie jest
hashem całej konfiguracji tokenizera — chroni konkretnie mapowanie słownika.

### `_cuda_kernel_supported()`

1. Odczytuje compute capability aktywnej karty, np. `(7,5)` → `sm_75`.
2. Pobiera listę architektur wkompilowanych w bieżący build PyTorch.
3. Zwraca `True`, gdy lista jest pusta/nieznana albo zawiera architekturę GPU.
4. Przy błędzie introspekcji zachowuje optymistyczny fallback `True`, pozwalając
   normalnej ścieżce CUDA obsłużyć starszy lub nietypowy build.

Funkcja chroni przede wszystkim przed uruchomieniem nowego koła PyTorch na
Kaggle P100 (`sm_60`), gdy build zawiera tylko kernela `sm_70+`.

### `find_device()`

Priorytet urządzeń:

1. `SAE_DEVICE=cpu` wymusza CPU;
2. kompatybilna CUDA;
3. Apple Metal Performance Shaders (`mps`);
4. CPU.

Jeśli CUDA jest widoczna, lecz niekompatybilna z buildem, funkcja wypisuje
capability i listę skompilowanych architektur, a następnie bezpiecznie wraca do
CPU. Zmienna `SAE_DEVICE` rozpoznaje jawnie tylko wartość `cpu`; nie pozwala
wybrać numeru konkretnej karty ani wymusić MPS/CUDA.

### `set_seed(seed)`

Ustawia ziarna:

- standardowego `random`,
- NumPy,
- CPU RNG PyTorch,
- wszystkich RNG CUDA, ale tylko gdy CUDA jest dostępna i kompatybilna.

Nie włącza `torch.use_deterministic_algorithms()` i nie ustawia flag cuDNN.
Ziarno zapewnia zatem kontrolę źródeł losowości używanych bezpośrednio przez
projekt, ale nie gwarantuje bitowej identyczności wszystkich kerneli GPU.

## Znaczenie dla reprodukowalności

Trener SAE dodatkowo zapisuje stany RNG w checkpointcie i odtwarza je przy
wznowieniu. Część operacji tworzy także lokalne generatory z ziarnem zależnym
od epoki/chunka. `set_seed()` jest bazą tego mechanizmu, lecz kompletna
reprodukowalność zależy również od:

- identycznych wersji PyTorch i sterowników,
- niezmienionej konfiguracji checkpointu,
- tej samej kolejności danych,
- deterministyczności użytych kerneli.
