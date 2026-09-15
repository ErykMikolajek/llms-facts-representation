# Układ repozytorium

Kod jest podzielony według etapów eksperymentu. Katalog główny zawiera tylko
pliki konfiguracyjne, zależności, instrukcje oraz katalogi wysokiego poziomu.

| Katalog | Odpowiedzialność | Przykładowy punkt wejścia |
| --- | --- | --- |
| `sae_pipeline/` | dane, aktywacje, trening i analiza SAE | `python -m sae_pipeline.main` |
| `domain_triage/` | discovery, stabilność i wybór domen | `python -m domain_triage.semantic_domain_triage` |
| `domain_mapping/` | dane domenowe, mapowanie neuronów, selektywność, pruning | `python -m domain_mapping.domain_mlp_activation_mapping` |
| `moe/` | router, składanie MoE, freeze i bundle | `python -m moe.moe_assembly` |
| `evaluation/` | holdout PPL i benchmark kompetencyjny | `python -m evaluation.moe_validation` |
| `kaggle/` | CLI, notebooki, buildery i uploadowane assety | `python kaggle/cli/kaggle_pipeline.py` |
| `tests/` | testy jednostkowe i regresyjne | `python -m unittest discover -s tests` |
| `reports/` | raporty metodologiczne i interpretacja | — |
| `results/` | surowe wyniki i raporty konkretnych runów | — |
| `documentation/` | MkDocs, architektura i podsumowanie projektu | `mkdocs build --strict` |
| `common/` | współdzielone helpery techniczne | — |

Pakiety etapowe mają `__init__.py` i używają importów absolutnych, np.
`from domain_mapping...`. Należy uruchamiać je z katalogu głównego przez
`python -m pakiet.moduł`; bezpośredni start pliku wewnątrz pakietu nie jest
kontraktem CLI.

Katalog `kaggle/` celowo **nie** jest pakietem Pythona. Lokalny pakiet o nazwie
`kaggle` przesłaniałby oficjalną bibliotekę używaną przez Kaggle CLI. Skrypty
tego katalogu uruchamia się pełną ścieżką.

Duże dane pozostają w `data/` i `models/`. Wygenerowany `documentation/site/`,
cache Pythona oraz pliki `.DS_Store` nie należą do źródła projektu.
