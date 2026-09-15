# `kaggle/cli/kaggle_pipeline.py`

[Kod źródłowy](https://github.com/ErykMikolajek/llms-facts-representation/blob/gemma-experiments/kaggle/cli/kaggle_pipeline.py) · [analiza Pythii](features_analysis.md) · [analiza Gemmy](gemma_scope_analysis.md) · [benchmark MoE](kaggle_moe_benchmark.md)

## Rola modułu

Skrypt automatyzuje wielosesyjną analizę SAE i jednorazowy benchmark MoE na
Kaggle. Obsługuje trzy
workflowy z niezależnymi notebookami, checkpointami i plikami stanu:

- `pythia` — `analysis_checkpoint_pythia.pt`;
- `gemma` — `analysis_checkpoint_gemma_scope2_270m.pt` oraz gated Kaggle Model.
- `moe_benchmark` — zamrożone assety v3 oraz ten sam gated Kaggle Model;
  używa `launch`, `status` i `publish-assets`, nie pętli `cycle`.

Pełny cykl wykonuje:

```text
status poprzedniego kernela
→ pobranie wyłącznie checkpointu z outputu
→ checksum SHA-256
→ synchronizacja pełnego stagingu datasetu
→ podmiana checkpointu i publikacja nowej wersji
→ oczekiwanie na status ready
→ staging notebooka i kernel-metadata.json
→ push następnego kernela na wybranym koncie
→ atomowy zapis stanu
```

## Dlaczego pełny staging datasetu

`kaggle datasets version` publikuje bieżącą zawartość katalogu i nie dziedziczy
automatycznie plików poprzedniej wersji. Wysłanie samego checkpointu usunęłoby
wagi SAE i przygotowane sekwencje. Pipeline pobiera więc pełny dataset, a przed
publikacją porównuje nazwy plików lokalnych i zdalnych.

Ograniczenia ochronne:

- zdalny dataset musi mieć mniej niż 200 plików (brak implementacji paginacji);
- pliki nie mogą mieć zagnieżdżonych ścieżek;
- lokalny staging nie może zawierać katalogów;
- brakujące lub niespodziewane pliki wymagają `--refresh-dataset`;
- `_safe_clear_directory()` odmawia usunięcia samego `work_dir` lub czegokolwiek
  poza nim.

## Model konfiguracji

### Dataclasses

- `Account`: nazwa profilu, Kaggle username, katalog credentials;
- `Workflow`: datasety, modele, nazwa checkpointu, notebook, state, kernel,
  akcelerator i timeouty;
- `Config`: wspólna ścieżka, executable, work dir, domyślny workflow, mapy
  workflowów i kont.

`load_config()` czyta TOML przez bibliotekę standardową, rozwiązuje ścieżki
względem katalogu configu i dziedziczy wartości `[pipeline]` do poszczególnych
workflowów. Gwarantuje, że dataset kodu i dataset checkpointów znajdują się na
liście `dataset_sources`.

Domyślny `config.toml` definiuje konta `account1..3`, publishera `account1`,
T4, prywatne kernele, poll 60 s, timeout datasetu 30 min i kernela 13 h.

## Credentials i izolacja kont

Każde konto używa osobnego `KAGGLE_CONFIG_DIR` z `kaggle.json`. Walidator
sprawdza:

- brak placeholdera username;
- istnienie katalogu i pliku;
- pola `username` i `key`;
- zgodność username z TOML;
- uprawnienia bez bitów group/other, praktycznie `chmod 600`.

Przed subprocess'em usuwane są odziedziczone `KAGGLE_USERNAME`, `KAGGLE_KEY` i
`KAGGLE_API_TOKEN`, aby profil nie został przypadkowo przesłonięty. Komenda jest
wypisywana przez `shlex.join`, ale sekret nie jest częścią argumentów.

## Wrapper `KaggleCli`

Konstruktor najpierw szuka programu obok aktualnego interpretera Python — to
obsługuje nieaktywowane virtualenv — a potem w `PATH`. `check_version()` wymaga
co najmniej 2.2.4. `run()` uruchamia subprocess, opcjonalnie scala stderr do
stdout i zamienia niezerowy kod w kontrolowany `PipelineError`.

## Stan i blokada

Każdy workflow ma osobny JSON state z ostatnim kernelem, kontem, czasem startu,
źródłem i SHA-256 checkpointu. Zapis używa `.tmp` i `os.replace()`.

`PipelineLock` zakłada nieblokującą, ekskluzywną blokadę `fcntl.flock` na
`.work/pipeline.lock` i wpisuje PID. Druga instancja kończy się czytelnym
błędem, co chroni staging i state przed wyścigiem.

!!! note "Przenośność"
    `fcntl` i `flock` są mechanizmami uniksowymi. Skrypt jest przeznaczony dla
    macOS/Linux; bez modyfikacji nie uruchomi się natywnie na Windows.

## Statusy i oczekiwanie

`_status_kind()` parsuje tekst Kaggle CLI do `complete`, `ready`, `pending`,
`failed` lub `unknown`. `wait_for_kernel()` może działać jednokrotnie albo
pollować do timeoutu. `--allow-failed-source` pozwala pobrać zachowany output
po statusie error/timeout, ale nie omija późniejszej walidacji pliku.

`wait_for_dataset()` zawsze polluje po publikacji do `ready`, ponieważ nowy
kernel nie powinien wystartować, zanim dataset będzie dostępny.

## Pobranie checkpointu

`download_checkpoint()` czyści bezpieczny podkatalog `kernel-output`, wywołuje
`kaggle kernels output` z regexem kończącym się nazwą checkpointu, a następnie:

1. preferuje dokładną `checkpoint_output_relative_path`;
2. w fallbacku wymaga dokładnie jednego pliku o tej nazwie;
3. odrzuca plik mniejszy niż 1024 B;
4. liczy streamingowo SHA-256 blokami 1 MiB.

Checksum trafia do komunikatu wersji datasetu i state, ale nie jest porównywany
z sumą zapisaną wewnątrz checkpointu.

## Publikacja datasetu

`sync_dataset()` pobiera i rozpakowuje dataset oraz jego metadata przy
`refresh=True` lub braku lokalnego metadata. Sprawdza ID datasetu. Następnie
`publish_checkpoint()`:

- pobiera aktualną listę nazw plików zdalnych w CSV verbose output;
- kopiuje checkpoint do stagingu;
- sprawdza kompletność stagingu;
- publikuje wersję z czytelnym komunikatem i `-t`;
- czeka na `ready`.

Jeśli zdalny dataset został zmieniony poza pipeline'em, porównanie wykryje
rozjazd, zamiast cicho nadpisać nową zawartość starą kopią.

## Staging i start kernela

`prepare_kernel_stage()` kopiuje tylko wybrany notebook i generuje
`kernel-metadata.json`: ID zależy od konta runnera, źródła datasetów pozostają
kanoniczne, a dla Gemmy dołączane jest `model_sources`.

`validate_model_sources()` wykonuje mały odczyt listy plików każdego modelu
przed push. Brak zaakceptowanej licencji Gemmy kończy się przed publikacją runu
GPU. `launch_kernel()` wykonuje push ze wskazanym typem akceleratora.

## Komendy CLI

| Komenda | Działanie |
| --- | --- |
| `accounts` | pokazuje profile, credentials, kernel refs i publisherów; nie wymaga poprawnych wszystkich kont |
| `status --account ...` | pyta o status kernela workflowu na koncie |
| `sync-dataset` | wymusza świeży, kompletny staging i go waliduje |
| `launch --account ...` | startuje notebook, zakładając aktualny checkpoint już w datasecie |
| `cycle --account ...` | pełny transfer checkpointu, publikacja datasetu i następny run |
| `publish-assets --directory ...` | tworzy lub wersjonuje samowystarczalny Dataset workflowu, po kontroli jego ID |

`cycle` rozwiązuje źródło w kolejności: jawny `--source-kernel`, ostatni kernel
ze state, kernel bieżącego runnera. Konto credentials źródła jest jawne,
zapamiętane albo dopasowane po ownerze kernel ref.

`--wait-source` polluje poprzedni run; `--wait` także nowy. `--refresh-dataset`
odświeża pełny staging, a `--allow-failed-source` dopuszcza output runu ze
statusem błędu.

## Katalog klas i funkcji

| Element | Odpowiedzialność |
| --- | --- |
| `PipelineError` | oczekiwany błąd workflowu mapowany przez `main()` na exit code 2 |
| `Account`, `Workflow`, `Config` | niemutowalne kontrakty konfiguracji |
| `Workflow.kernel_ref()` | składa `username/kernel_slug` |
| `KaggleCli.__init__()` | rozwiązuje executable z preferencją virtualenv |
| `KaggleCli.check_version()` | parsuje i waliduje minimalną wersję |
| `KaggleCli.run()` | izolowane credentials, subprocess i obsługa błędu |
| `_required()` | wymagany klucz TOML z kontekstem sekcji |
| `_relative_to_config()` | rozwija `~` i ścieżki względne do configu |
| `load_config()` | dziedziczenie/normalizacja pełnej konfiguracji |
| `account_configuration_error()` / `validate_account()` | audyt `kaggle.json` i uprawnień |
| `get_account()` / `get_workflow()` | lookup z listą dostępnych wartości |
| `read_state()` / `write_state()` | odczyt i atomowy zapis stanu workflowu |
| `PipelineLock.__init__()` | przechowuje ścieżkę i pusty uchwyt blokady |
| `PipelineLock.__enter__()` | tworzy plik, zakłada nieblokujący `flock`, zapisuje PID |
| `PipelineLock.__exit__()` | zwalnia blokadę i zamyka uchwyt także przy wyjątku |
| `sha256_file()` | streamingowy digest pliku |
| `_status_kind()` | heurystyczna normalizacja statusu CLI |
| `wait_for_kernel()` / `wait_for_dataset()` | polling, timeout i terminal failures |
| `account_for_kernel()` / `resolve_source()` | dopasowanie refu i credentials źródła |
| `_safe_clear_directory()` | ograniczone czyszczenie podkatalogu work dir |
| `download_checkpoint()` | selektywne pobranie, jednoznaczność, rozmiar i checksum |
| `_parse_dataset_file_names()` / `remote_dataset_files()` | parsuje CSV listy plików i egzekwuje ograniczenia |
| `sync_dataset()` | pełny staging plików i metadata |
| `validate_complete_dataset_stage()` | różnica remote/local bez utraty danych |
| `publish_checkpoint()` | podmiana, wersjonowanie i wait-ready |
| `prepare_kernel_stage()` | notebook plus metadata per konto/workflow |
| `validate_model_sources()` | preflight gated Kaggle Models |
| `launch_kernel()` | push kernela |
| `record_launch()` | aktualizuje osobny state workflowu |
| `command_cycle()` | pełny resumowalny cykl |
| `command_launch()` | pierwszy start bez transferu |
| `command_sync_dataset()` | ręczne odświeżenie stagingu |
| `command_accounts()` | diagnostyka profili bez wymuszania ich poprawności |
| `command_status()` | pojedynczy status |
| `command_publish_assets()` | walidacja metadata i publikacja prywatnych assetów |
| `build_parser()` | subcommands i opcje |
| `main()` | config, lock, version check, dispatch i exit codes 0/2/130 |

## Ograniczenia i ryzyka operacyjne

- Parser statusu zależy od tekstowego outputu Kaggle CLI; zmiana jego formatu
  może dać `unknown` i polling aż do timeoutu.
- Lista datasetu nie obsługuje paginacji ≥200 plików ani zagnieżdżonych nazw.
- `dataset-metadata.json` i lokalna lista plików chronią kompletność nazw, ale
  nie porównują checksum każdego niezmienianego pliku.
- `cycle` publikuje nową wersję zdalnego datasetu i uruchamia kernel — są to
  realne operacje zewnętrzne, nie dry-run.
- Pipeline wersjonuje dataset checkpointów, nie dataset kodu. Przed runem
  aktualny kod musi być opublikowany osobnym procesem opisanym w README.
- Timeout kernela nie zatrzymuje samego kernela na Kaggle; kończy jedynie
  lokalne oczekiwanie.
