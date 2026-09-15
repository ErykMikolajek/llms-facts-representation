# Automatyzacja analiz SAE i MoE na Kaggle

Skrypt `kaggle_pipeline.py` obsługuje cztery niezależne workflowy: `pythia` dla
`kaggle/notebooks/kaggle_pythia160m_sae.ipynb` i `gemma` dla
`kaggle/notebooks/kaggle_gemma_scope2_270m.ipynb` oraz `moe_benchmark` dla
`kaggle/notebooks/kaggle_gemma_moe_benchmark.ipynb`. Workflow
`domain_expert_benchmark` uruchamia osobny benchmark modeli single-expert.
`cycle` dla długich
analiz SAE:

1. sprawdza, czy poprzedni Kaggle Kernel zakończył się powodzeniem;
2. pobiera właściwy checkpoint z jego outputu;
3. publikuje pełną nową wersję datasetu `Trained SAE models` z podmienionym
   checkpointem;
4. czeka, aż wersja datasetu będzie gotowa;
5. publikuje i uruchamia notebook wybranego workflowu na wybranym koncie.

Checkpointy nie kolidują ze sobą:

- `pythia`: `analysis_checkpoint_pythia.pt`;
- `gemma`: `analysis_checkpoint_gemma_scope2_270m.pt`.

Workflowy `moe_benchmark` i `domain_expert_benchmark` są jednorazowymi
ewaluacjami, więc używają `launch`, `status` i `publish-assets`, a nie pętli
checkpointów `cycle`.

## Dlaczego skrypt przechowuje pełny staging datasetu

`kaggle datasets version` nie zachowuje automatycznie plików z poprzedniej
wersji. Wysłanie samego checkpointu usunęłoby z bieżącej wersji m.in. wagi SAE
oraz przygotowane sekwencje. Pierwszy cykl pobiera więc cały dataset do
`kaggle/cli/.work/dataset/`, a przed każdą publikacją porównuje lokalny zestaw
plików z bieżącym zestawem plików na Kaggle. Jeśli dataset został zmieniony z
innego miejsca, skrypt przerywa pracę i prosi o `--refresh-dataset`.

## Wymagania

- Python 3.11 lub nowszy;
- Kaggle CLI 2.2.4
  (`pipx install 'kaggle==2.2.4'` albo instalacja z głównego `requirements.txt`);
- dostęp wszystkich trzech kont do prywatnych datasetów podanych w
  `dataset_sources`;
- zaakceptowane warunki Gemmy na każdym koncie używanym do workflowu `gemma`
  lub `moe_benchmark`;
- konto `publisher_account` musi być właścicielem lub edytorem datasetu
  `erykmikoajek/trained-sae-models`.

Oba własne datasety zawsze zachowują kanonicznego autora:
`erykmikoajek/trained-sae-models` oraz
`erykmikoajek/sae-training-and-moeffication`. Login wybranego konta służy
wyłącznie do identyfikatora uruchamianego kernela. Nie jest podstawiany do
identyfikatorów datasetów. Konta 2 i 3 mogą ich używać jako współautorzy;
publikowanie nowej wersji checkpointu domyślnie wykonuje `account1`.

Nie zapisuj kluczy API w repozytorium. Dla każdego konta utwórz osobny katalog
i umieść w nim pobrany z Kaggle plik `kaggle.json`, na przykład:

```text
~/.kaggle/kaggle.json
~/.config/kaggle/accounts/account2/kaggle.json
~/.config/kaggle/accounts/account3/kaggle.json
```

Pliki powinny mieć uprawnienia tylko dla właściciela (`chmod 600 kaggle.json`).
W tym wielokontowym workflowie używane są osobne legacy API keys w formacie
`kaggle.json`, ponieważ bieżące dane logowania OAuth Kaggle CLI są domyślnie
zapisywane globalnie w `~/.kaggle/credentials.json`. Ścieżka profilu legacy
jest przekazywana przez `KAGGLE_CONFIG_DIR`, a skrypt usuwa z procesu
odziedziczone `KAGGLE_USERNAME`, `KAGGLE_KEY` i `KAGGLE_API_TOKEN`, aby nie
uruchomić komendy przypadkiem na innym koncie.

Następnie uzupełnij loginy kont 2 i 3 w `config.toml`. Slugi notebooków są
ustawiane osobno w sekcjach `workflows.*`.

Dla Gemmy pipeline dodaje do `kernel-metadata.json` model
`google/gemma-3/transformers/gemma-3-270m/2`. Notebook ładuje model i tokenizer
z zamontowanego katalogu `/kaggle/input`, więc nie korzysta z `HF_TOKEN`.
Każde konto Kaggle, które ma uruchamiać Gemmę, musi osobno otworzyć stronę tego
wariantu i zaakceptować warunki licencji. `launch` i `cycle` wykonują przed
wysłaniem notebooka mały odczyt listy plików modelu; brak dostępu przerywa
polecenie przed zużyciem runu GPU lub publikacją checkpointu.

Sprawdź, który plik wykonywalny zostanie użyty (`which kaggle`) oraz wynik
`kaggle --version`. Jeśli nowsza instalacja z `pipx` nie jest pierwsza w
`PATH`, wpisz jej pełną ścieżkę w `kaggle_executable` w `config.toml`. Skrypt
odrzuca wersje starsze niż 2.2.4 przed wykonaniem operacji sieciowych.

## Pierwsze uruchomienie

Sprawdź profile:

```bash
python3 kaggle/cli/kaggle_pipeline.py accounts
```

`pythia` jest workflowem domyślnym, dlatego `--workflow pythia` można pominąć.
Jeżeli aktualny checkpoint znajduje się już w datasecie, można uruchomić
pierwszy kernel bez pobierania outputu poprzedniego kernela:

```bash
python3 kaggle/cli/kaggle_pipeline.py launch --account account1
```

Pierwszy run Gemmy można rozpocząć bez istniejącego checkpointu:

```bash
python3 kaggle/cli/kaggle_pipeline.py launch --workflow gemma --account account1
```

## Uruchomienie benchmarku MoE

Najpierw zbuduj notebook i samowystarczalną paczkę:

```bash
.venv/bin/python kaggle/build_kaggle_moe_benchmark_notebook.py
.venv/bin/python kaggle/prepare_kaggle_moe_benchmark_assets.py \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --model-dir models/gemma-3-270m --workspace-root .
```

Pierwsza publikacja prywatnego Datasetu:

```bash
.venv/bin/python kaggle/cli/kaggle_pipeline.py publish-assets \
  --workflow moe_benchmark --account account1 \
  --directory kaggle/assets/gemma_moe_benchmark_v3 --create
```

Przy kolejnej wersji pomiń `--create`. Następnie uruchom kernel i opcjonalnie
czekaj na wynik:

```bash
.venv/bin/python kaggle/cli/kaggle_pipeline.py launch \
  --workflow moe_benchmark --account account1 --wait
```

Status bez uruchamiania nowego kernela:

```bash
.venv/bin/python kaggle/cli/kaggle_pipeline.py status \
  --workflow moe_benchmark --account account1
```

Jeżeli checkpoint jest outputem istniejącego kernela, wykonaj od razu pełny
cykl i jawnie podaj go przy pierwszym wywołaniu:

```bash
python3 kaggle/cli/kaggle_pipeline.py cycle \
  --workflow pythia \
  --account account1 \
  --source-kernel erykmikoajek/SLUG_POPRZEDNIEGO_NOTEBOOKA \
  --source-account account1 \
  --refresh-dataset
```

`--refresh-dataset` jest wymagane tylko do utworzenia/odświeżenia lokalnego
stagingu. Pobieranie całego datasetu może potrwać, ale chroni przed usunięciem
pozostałych plików podczas wersjonowania przez CLI.

Jeżeli Kaggle oznaczył sesję przerwaną limitem czasu jako `error`, ale zapisał
jej output, dodaj `--allow-failed-source`. Skrypt nadal przerwie pracę, gdy w
output nie znajdzie właściwego checkpointu.

## Następne cykle i zmiana konta

Skrypt zapamiętuje ostatnio uruchomiony kernel oddzielnie dla workflowów w
ignorowanym przez Git katalogu `.state/`. Po zakończeniu runu wystarczy:

```bash
python3 kaggle/cli/kaggle_pipeline.py cycle --account account1
python3 kaggle/cli/kaggle_pipeline.py cycle --workflow gemma --account account1
```

Po wyczerpaniu limitu GPU zmień tylko profil docelowy:

```bash
python3 kaggle/cli/kaggle_pipeline.py cycle --account account2
python3 kaggle/cli/kaggle_pipeline.py cycle --account account3
python3 kaggle/cli/kaggle_pipeline.py cycle --workflow gemma --account account2
python3 kaggle/cli/kaggle_pipeline.py cycle --workflow gemma --account account3
```

Checkpoint nadal zostanie pobrany przy użyciu profilu właściciela poprzedniego
kernela, dataset opublikuje `publisher_account`, a nowy run uruchomi wskazane
konto. Opcja `--wait-source` pozwala uruchomić komendę przed zakończeniem
poprzedniego runu. Opcja `--wait` dodatkowo pozostawia proces do monitorowania
nowego runu.

Przydatne komendy:

```bash
python3 kaggle/cli/kaggle_pipeline.py status --account account2
python3 kaggle/cli/kaggle_pipeline.py status --workflow gemma --account account2
python3 kaggle/cli/kaggle_pipeline.py sync-dataset --workflow pythia
```

## Jednorazowa publikacja kodu

Notebooki czytają kod z datasetu
`erykmikoajek/sae-training-and-moeffication`. Przed pierwszym automatycznym
runem opublikuj w nim aktualne pliki z repozytorium. Dla Pythii istotne są
`sae_pipeline/main.py` i `sae_pipeline/features_analysis.py`, a dla Gemmy także
`sae_pipeline/gemma_scope_analysis.py`. Pipeline celowo nie wersjonuje datasetu z kodem przy
każdym cyklu — zmienia wyłącznie checkpoint analizy w
`erykmikoajek/trained-sae-models`.
