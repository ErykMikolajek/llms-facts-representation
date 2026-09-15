# `sae_pipeline/activations_collecting.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/sae_pipeline/activations_collecting.py) · [dane wejściowe](dataset_sequencing.md) · [trening SAE](autoencoder_training.md)

## Rola modułu

Moduł pozyskuje `resid_post` jednej warstwy transformera w sposób
storage-bounded. Zamiast żądać `output_hidden_states=True` dla wszystkich
warstw i zapisywać pełny tensor aktywacji, rejestruje hook tylko na wybranym
bloku i zwraca kolejne chunki ważnych tokenów.

Obsługiwane układy backbone'u:

- GPT-NeoX: `model.gpt_neox.layers`;
- GPT-Neo: `model.transformer.h`;
- Gemma 3/Hugging Face: `model.model.layers`.

## Kontrakt `iter_activation_chunks()`

Generator zwraca trójki:

```text
(sequence_start, sequence_end, activations)
```

`activations` ma kształt `[N_valid_in_chunk, D]` i domyślnie dtype `float16`.
Kolejność wierszy odpowiada spłaszczeniu `[B,S]` w porządku wierszowym po
usunięciu miejsc z maską `False`.

Granice `sequence_start:end` są kursorem checkpointu. Trener może zapisać
`next_sequence=end` dopiero po pełnym zużyciu chunka i dokładnie wznowić run.

## Przebieg krok po kroku

1. `open_token_store()` rozwiązuje domyślne albo jawne ścieżki tokenów i maski,
   a `TokenDataset` sprawdza kształty.
2. Zakres końcowy jest ograniczany przez liczbę sekwencji oraz `max_sequences`
   liczone od `start_sequence`.
3. Model jest przenoszony na urządzenie, przełączany w `eval()`, a z modelu
   wybierany jest backbone i docelowa warstwa.
4. Forward hook normalizuje wynik warstwy do tensora `[B,S,D]`. Obsługuje
   bezpośredni tensor, pierwszy element tuple/list oraz obiekt z
   `last_hidden_state`.
5. Dla każdego chunka sekwencji powstają mniejsze batche modelu. Dane są
   kopiowane z memmapy do zwykłej tablicy NumPy, następnie do tensora na urządzeniu.
6. Jeśli istnieje jawna maska, jest użyta bez wnioskowania z tokenów. Fallback
   `token != 0` służy wyłącznie zgodności ze starym formatem.
7. Uruchamiany jest sam backbone, z `use_cache=False`, więc nie są konstruowane
   logity całego słownika ani stany wszystkich warstw.
8. Hook dostarcza hidden state; `hidden[attention_mask]` usuwa padding, wynik
   jest przenoszony na CPU i konwertowany do `output_dtype`.
9. Części batchy są łączone w jeden ograniczony chunk i przekazywane callerowi.
10. W `finally` hook jest zawsze usuwany, również po wyjątku lub zamknięciu generatora.

!!! important "Zakres `inference_mode`"
    `torch.inference_mode()` obejmuje tylko forward modelu i przygotowanie
    chunka, ale kończy się przed `yield`. Gdyby generator pozostawał w tym
    kontekście podczas pracy callera, PyTorch mógłby wyłączyć gradienty także
    dla straty SAE.

## Katalog klas i funkcji

| Element | Odpowiedzialność |
| --- | --- |
| `get_backbone(model)` | usuwa głowę LM i zwraca właściwy korpus transformera |
| `get_transformer_layers(model)` | normalizuje listę bloków jako `.layers` albo `.h` |
| `get_transformer_layer(model, layer_num)` | waliduje indeks i zwraca pojedynczy blok |
| `_hidden_from_layer_output(output)` | sprowadza różne API hooka do tensora `[B,S,D]` i waliduje typ/kształt |
| `TokenStore` | niemutowalny opis ścieżek, liczby sekwencji i długości |
| `TokenDataset.__init__()` | otwiera memmapy i sprawdza kształt tokenów oraz opcjonalnej maski |
| `TokenDataset.__len__()` | zwraca liczbę wierszy/sekwencji bez materializacji danych |
| `TokenDataset.__getitem__()` | kopiuje jeden wiersz do tensorów i używa jawnej maski lub legacy `token != 0` |
| `open_token_store(...)` | otwiera i waliduje magazyn tokenów, zwraca metadane |
| `iter_activation_chunks(...)` | podstawowy generator bounded activations |
| `collect_activations(...)` | adapter zgodności ze starym API; domyślnie tylko konsumuje i podsumowuje stream |

## `TokenDataset` a bezpośrednia memmapa

`TokenDataset` jest używany przy otwieraniu do walidacji i określenia liczby
wierszy. W głównej pętli generator czyta memmapy bez DataLoadera. Każdy wycinek
jest jawnie kopiowany, co eliminuje ostrzeżenia o niemodyfikowalnym buforze
NumPy i oddziela czas życia tensora od mapowania pliku.

## Ścieżka kompatybilności `collect_activations()`

Domyślnie `keep_full_file=False`; funkcja liczy sekwencje i wektory, lecz nie
zapisuje pełnego pliku. Zwraca słownik z `sequence_count`,
`activation_vectors`, `d_model`, trybem storage i ścieżką tokenów.

`keep_full_file=True` gromadzi wszystkie chunki w liście RAM, konkatenatuje je
i zapisuje `activations_layer_<L>.npy`. Jest to ścieżka historyczna, nieodpowiednia
dla runów Pythia-scale. Co więcej, jej wynik jest dwuwymiarowy `[N_valid,D]`,
podczas gdy legacy trainer w `sae_pipeline/autoencoder_training.py` oczekuje starego
trójwymiarowego `[N,S,D]`; te dwie ścieżki kompatybilności nie tworzą obecnie
spójnego nowego workflowu bez dodatkowej konwersji.

## Ograniczenia pamięci i obliczeń

- Na urządzeniu: co najwyżej `batch_size × S × D` dla modelu.
- Na CPU przed `yield`: w przybliżeniu `chunk_sequences × S × D` plus lista
  części batchy.
- Na dysku: wyłącznie przygotowane tokeny i maska; brak pełnego zbioru aktywacji.
- Model jest wykonywany od początku do końca backbone'u, mimo że hook dotyczy
  jednej warstwy; głowa słownikowa jest pominięta, ale warstwy po hooku nadal
  są liczone. To oszczędza pamięć logitów, nie wszystkie FLOP-y późniejszych bloków.

## Warunki poprawności

- `layer_num` musi należeć do zakresu bloków.
- Kształt tokenów musi być `[N, seq_length]`; kształt maski musi być identyczny.
- Model i tokeny muszą używać zgodnego tokenizera.
- Warstwa musi wywołać hook i zwrócić reprezentację możliwą do sprowadzenia do
  `[B,S,D]`.
