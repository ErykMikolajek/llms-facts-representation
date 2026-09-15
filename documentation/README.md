# Dokumentacja kodu eksperymentalnego

Ten katalog zawiera szczegółową, polskojęzyczną dokumentację modułów Python
realizujących eksperyment badawczy. Wersję Markdown można czytać bez żadnych
dodatkowych narzędzi, zaczynając od [`docs/index.md`](docs/index.md).

Dokumentacja jest też skonfigurowana jako serwis MkDocs Material. Aby uruchomić
lokalny podgląd:

```bash
cd documentation
python -m pip install -r requirements.txt
python -m mkdocs serve
```

Następnie należy otworzyć adres wypisany przez MkDocs (zwykle
`http://127.0.0.1:8000`). Komenda budująca statyczny serwis to:

```bash
python -m mkdocs build --strict
```

Pliki w `docs/` są źródłem dokumentacji. Katalog `site/`, tworzony przez
MkDocs, nie powinien być wersjonowany.

