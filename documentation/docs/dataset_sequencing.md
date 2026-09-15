# `sae_pipeline/dataset_sequencing.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/sae_pipeline/dataset_sequencing.py) · [wejście CLI](main.md) · [konsument aktywacji](activations_collecting.md)

## Rola modułu

Moduł zamienia dokumenty tekstowe na stałokształtne, memory-mapped tablice
tokenów i masek. Obsługuje dwa wejścia:

- CSV z kolumną `text` przez bibliotekę Hugging Face `datasets`;
- Pile-style JSONL, również skompresowany jako `.zst` lub `.gz`, czytany
  rekord po rekordzie.

Głównym celem implementacji jest ograniczenie pamięci: dokumenty, zdania i
tokeny przepływają przez generatory, a finalne tablice są zapisywane memmapą.

## Wejścia i wyjścia

### Wejście JSONL

Akceptowane są rekordy `{"text": "..."}` oraz
`{"root": {"text": "..."}}`. Katalog jest przeszukiwany rekurencyjnie według
`file_pattern`, a kolejność plików jest deterministyczna (`sorted`). Limity
`max_files`, `max_documents` i `max_tokens` działają na kolejnych poziomach.

### Wejście CSV

CSV musi zawierać kolumnę `text`. Puste i nietekstowe wiersze są odfiltrowane.
`ds_fraction < 1` wybiera losową, ale reprodukowalną próbkę wierszy według
`seed`; kolejność wybranych indeksów odpowiada kolejności zwróconej przez
`random.sample`.

### Artefakty

| Plik | Format | Znaczenie |
| --- | --- | --- |
| `sequenced/TEMP_tokenized_not_padded.jsonl` | JSONL | przejściowe sekwencje zmiennej długości; usuwane po sukcesie |
| `sequenced/tokens_seqs_padded.npy` | `int32[N_seq,S]` | tokeny dopełnione `pad_token_id` |
| `sequenced/attention_mask.npy` | `uint8[N_seq,S]` | `1` dla ważnego tokenu, `0` dla paddingu |
| `dataset_info.json` | JSON | manifest źródeł i limitów (JSONL) albo cache liczby tokenów (CSV) |

## Przebieg krok po kroku

1. Tekst jest dzielony na zdania przez `syntok`. Gdy biblioteka nie jest
   dostępna, używany jest prosty regex rozdzielający po `.`, `!` lub `?` i
   białych znakach.
2. Zdania są grupowane po `batch_sentences` i tokenizowane bez tokenów
   specjalnych, paddingu i truncation.
3. Zdanie dłuższe niż `seq_length` jest dzielone na kolejne fragmenty tokenów
   długości co najwyżej `S`. Krótsze zdania pozostają osobnymi porcjami.
4. `_pack_and_write_from_sentence_ids()` pakuje kolejne porcje do bufora.
   Jeżeli następna porcja nie mieści się, bieżący bufor jest zapisany, o ile ma
   co najmniej `min_seq_len`; zbyt krótki bufor jest odrzucony.
5. Pełna porcja o długości `S`, napotkana przy pustym buforze, jest zapisywana
   bez kopiowania do bufora.
6. Powstaje tymczasowy JSONL z listami `input_ids` o długościach `≤ S`.
7. `_pad_sequences()` najpierw liczy wiersze, tworzy dwie memmapy o docelowym
   kształcie, wypełnia tokeny paddingiem i maskę zerami, a potem uzupełnia
   ważne prefiksy każdego wiersza.
8. Memmapy są `flush()`-owane, tymczasowy JSONL usuwany, a statystyki zapisywane.

## Subtelność pakowania

Algorytm zachowuje granice porcji zdaniowych tak długo, jak następna porcja
mieści się w buforze. Kiedy się nie mieści, zapisuje bieżący bufor i rozpoczyna
nowy od całej porcji. Nie dzieli krótkiej porcji tylko po to, aby idealnie
wypełnić poprzednią sekwencję. W efekcie sekwencje mogą być krótsze od `S`, a
`avg_len` opisuje rzeczywistą średnią przed paddingiem.

`max_tokens` ogranicza tokeny wyemitowane przez generator tokenizacji, zanim
zostanie zastosowane odrzucanie buforów krótszych niż `min_seq_len`. Finalne
`total_tokens` może więc być nieco mniejsze od limitu.

## Ścieżka JSONL

`prepare_sequences_from_jsonl()`:

1. waliduje dodatniość limitów;
2. rozwiązuje pliki `_jsonl_paths()`;
3. tworzy generator dokumentów `_iter_jsonl_texts()` i zlicza dokumenty per plik;
4. tokenizuje i pakuje strumień;
5. materializuje memmapy;
6. usuwa plik tymczasowy;
7. zapisuje manifest `pile_jsonl_sequence_manifest_v1` z bezwzględną ścieżką
   źródła, listą plików, tokenizerem, długością i statystykami.

Źródła są tylko odczytywane. Jest to istotne na Kaggle, gdzie `/kaggle/input`
jest montowany read-only.

## Ścieżka CSV

`prepare_sequences()` wybiera plik jawnie albo interaktywnie przez `pick`.
Gdy istnieje jeden lub tryb interaktywny jest wyłączony, używa pierwszego
elementu listy z `os.listdir()`; ta kolejność nie jest jawnie sortowana.

Przed właściwą tokenizacją liczba tokenów jest obliczana przez równoległe
`Dataset.map(_count_tokens, num_proc=cpu_count-1)`. Wynik może zostać ponownie
użyty z `dataset_info.json`, jeśli zgodne są: frakcja, seed, bezwzględna ścieżka
źródła i `seq_length`.

!!! warning "Cache CSV"
    Cache liczby tokenów nie sprawdza zawartości ani czasu modyfikacji CSV oraz
    nie zapisuje identyfikatora tokenizera. Po zmianie pliku pod tą samą ścieżką
    albo zmianie tokenizera `dataset_info.json` należy usunąć, aby uniknąć
    mylącego licznika postępu. Sam wynik tokenizacji jest liczony od nowa.

## Katalog funkcji i klas

| Element | Odpowiedzialność |
| --- | --- |
| `_count_tokens(batch, tokenizer)` | tokenizuje batch `text` bez zmian długości i zwraca liczbę tokenów per rekord |
| `_split_into_sentences(text)` | segmentuje tekst przez `syntok` lub awaryjny regex |
| `_encode_sentences_batch(sentences, tokenizer, seq_length)` | batchowo tokenizuje zdania i dzieli zbyt długie listy identyfikatorów |
| `_pack_and_write_from_sentence_ids(...)` | pakuje porcje do sekwencji, odrzuca zbyt krótkie i zapisuje przejściowy JSONL |
| `_null_context` | minimalny context manager zastępujący nieobecny pasek `tqdm` w trybie `low` |
| `_null_context.__enter__()` / `__exit__()` | zwraca `None`, nie tłumi wyjątków i pozwala użyć wspólnego bloku `with` |
| `_sentences_generator_from_texts(texts)` | leniwie rozwija dokumenty do zdań |
| `_token_ids_generator_from_texts(...)` | leniwie batchuje zdania, tokenizuje je i egzekwuje globalny `max_tokens` |
| `_jsonl_paths(input_path, file_pattern, max_files)` | znajduje i sortuje obsługiwane shardy |
| `_iter_jsonl_records(path)` | strumieniuje plain/zstd/gzip i raportuje numer błędnej linii JSON |
| `_extract_jsonl_text(record)` | odczytuje `text` albo `root.text`, w innych przypadkach zwraca pusty tekst |
| `_iter_jsonl_texts(paths, max_documents, source_counts)` | filtruje puste dokumenty i prowadzi liczniki źródeł |
| `prepare_sequences_from_jsonl(...)` | kompletna publiczna ścieżka shardów Pile |
| `_pad_sequences(...)` | tworzy dwie finalne memmapy bez listy rozmiaru całego datasetu |
| `prepare_sequences(...)` | kompletna publiczna ścieżka CSV z samplingiem i cache liczby tokenów |

## Zużycie zasobów

- RAM tokenizacji jest ograniczony batchami zdań i bieżącym buforem sekwencji.
- Finalne tablice są mapowane do pliku; nie powstaje lista `N_seq × S` w RAM.
- Tymczasowy JSONL oznacza jednak dodatkowy zapis i odczyt całego zbioru
  sekwencji przed utworzeniem `.npy`.
- `source_counts` jest aktualizowane po całkowitym przejściu danego pliku. Gdy
  `max_documents` zatrzyma generator wewnątrz pliku, licznik tego ostatniego
  pliku może nie zostać zapisany, ponieważ generator kończy się przed kodem po
  pętli.

## Warunki poprawności

- `attention_mask.npy` jest źródłem prawdy o ważności tokenu; nie należy
  wnioskować paddingu z `token_id == 0` dla nowych danych.
- Tokenizer użyty do sekwencjonowania musi odpowiadać tokenizerowi modelu w
  kolejnych etapach.
- `seq_length` przekazany konsumentom musi odpowiadać drugiemu wymiarowi tablicy.
- Awaryjny regex nie ma jakości pełnego segmentera; brak `syntok` może zmienić
  granice sekwencji, a więc i dane treningowe.
