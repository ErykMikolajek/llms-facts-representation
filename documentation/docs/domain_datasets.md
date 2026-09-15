# Przygotowanie eksperymentu i danych domenowych

## `domain_triage/prepare_gemma_domain_experiment.py`

Skrypt materializuje ręcznie zaakceptowany eksperyment z wyniku stabilności.
Weryfikuje wejściowy hash, wybiera wskazane klastry, przepisuje pełne
`feature_ids`, tokeny, kohezję i stabilność oraz zapisuje `domains.json` z
`downstream_approved=true`. W bieżącym eksperymencie przypisano klastry
20/17/16/22/15/5 do domen prawo, biomedycyna, sport, polityka,
matematyka/LaTeX i Python.

Automatyzacja nie zastępuje decyzji badacza. Lista sześciu nazw i klastrów
jest jawna w kodzie; zmiana shortlisty wymaga świadomej aktualizacji skryptu,
a nie automatycznego przyjęcia kolejnego rankingu.

## `domain_mapping/prepare_domain_datasets.py`

Moduł pobiera źródła z Hugging Face pod dokładnymi rewizjami oraz ekstrahuje
Python z lokalnej biblioteki standardowej. Zapisuje rekordy z `domain_id`,
tekstem, źródłem, splitem, `source_group`, identyfikatorem i SHA-256
znormalizowanej treści.

| Domena | Development | Holdout |
| --- | --- | --- |
| prawo | LexGLUE SCOTUS `train` | SCOTUS `test` |
| biomedycyna | PubMedQA `pqa_artificial` | `pqa_labeled` |
| sport | DBpedia14 Athlete `train` | `test` |
| polityka | DBpedia14 OfficeHolder `train` | `test` |
| matematyka/LaTeX | MATH, 7 konfiguracji, `train` | `test` |
| Python | funkcje CPython 3.11 z grupy plików dev | rozłączna grupa plików holdout |
| ogólne | WikiText-2 raw `validation` | `test` |

Przypięte rewizje to:

- `coastalcph/lex_glue`: `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c`;
- `qiaojin/PubMedQA`: `9001f2853fb87cab8d220904e0de81ac6973b318`;
- `fancyzhx/dbpedia_14`: `9abd46cf7fc8b4c64290f26993c540b92aa145ac`;
- `EleutherAI/hendrycks_math`: `21a5633873b6a120296cce3e2df9d5550074f4a3`;
- `Salesforce/wikitext`: `b08601e04326c79dfdd32d625aee71d232d685c3`.

Dla Pythona podział następuje **przed** ekstrakcją funkcji: względna ścieżka
pliku jest hashowana i przypisywana regułą modulo 5. Dzięki temu funkcje z
jednego pliku nie trafiają do obu części. Długie teksty są przycinane przez
deterministyczne okno zależne od klucza próbki, nie zawsze przez prefiks.

`validate_disjoint_splits()` blokuje przecięcie hashy treści i grup źródłowych
oraz duplikaty. Finalne liczności to 180/90 rekordów na domenę i 200/100
ogólnych, odpowiednio 1280 i 640 tekstów.

## Artefakty

| Plik | Znaczenie |
| --- | --- |
| `domains.json` | zatwierdzone domeny, cechy i provenance analizy |
| `development_domains.csv` | dane B do mappingu i routera |
| `development_general.csv` | kalibracja fallbacku i regresji |
| `holdout_domains.csv` | zamrożona ocena PPL per domena |
| `holdout_general.csv` | zamrożona ocena ogólna |
| `domain_validation/domain_<id>.jsonl` | kopia development w formacie modułów domenowych |
| `dataset_manifest.json` | rewizje, liczniki, hashe i testy rozłączności |

## Katalog funkcji `domain_mapping/prepare_domain_datasets.py`

| Funkcja | Odpowiedzialność |
| --- | --- |
| `normalized_text()` / `content_hash()` | kanoniczna postać i hash treści |
| `stable_text_window()` | deterministyczne ograniczenie długości |
| `make_sample()` / `take_from_stream()` | wspólny rekord i bounded sampling |
| `resolve_revisions()` / `load_stream()` | przypięcie commita i streaming datasetu |
| `prepare_legal()` / `prepare_biomedical()` | SCOTUS i PubMedQA |
| `prepare_dbpedia_domain()` | Athlete lub OfficeHolder |
| `prepare_math()` | MATH ze wszystkich konfiguracji |
| `python_function_samples()` | ekstrakcja AST funkcji bez wykonywania kodu |
| `prepare_general()` | WikiText-2 raw |
| `validate_disjoint_splits()` | brak duplikatów/przecieków dev–holdout |

`domain_triage/prepare_gemma_domain_experiment.py` składa się z `sha256_file()`,
`write_report()`, `parse_args()` i `main()`. `main()` waliduje shortlistę,
tworzy sześć rekordów domen, kopiuje pełne listy cech i zapisuje raport decyzji.

## Ważne ograniczenie

Rozłączność hashy usuwa bezpośredni duplicate leakage, ale domena jest nadal
silnie związana ze źródłem i formatem dokumentu. Dlatego ten holdout służył do
kontroli zachowania PPL, a niezależny benchmark używa innych repozytoriów i
zadań.
