# Zamrożenie protokołu i paczka runtime

## `moe/freeze_gemma_moe_protocol.py`

Skrypt został uruchomiony **przed pierwszym odczytem wyników holdoutu**.
Zbiera konfigurację eksperymentu oraz SHA-256 i rozmiary:

- danych development i holdout;
- `domains.json`, manifestu danych i selektywności;
- masek, pruning summary i routera;
- kodu mappingu, pruningu, routera, assembly i walidacji;
- kluczowych plików lokalnego modelu Gemma.

Zapisuje także seed, `batch_size`, `max_length`, próg regresji i deklarację,
że konfiguracja jest zamrożona. `holdout_protocol_frozen.json` jest
provenance, nie mechanizmem kryptograficznego ukrywania odpowiedzi.

## `moe/prepare_gemma_moe_bundle.py`

Po walidacji skrypt tworzy `moe_bundle.json`, samowystarczalny manifest
uruchomieniowy. Wskazuje model bazowy, layer, router, confidence threshold,
sześć masek oraz źródło ekspertów `base_masks`. Każdy plik ma rozmiar i hash;
loader w `moe/moe_assembly.py` może je sprawdzić przed złożeniem modelu.

Maska ma 2048 wartości bool i zajmuje 2176 bajtów w formacie `.npy`; wszystkie
sześć masek to 13 056 bajtów. Historyczne pełne stany ekspertów zajmują około
90 MiB i pozostają do audytu przebiegu, ale nie są potrzebne w paczce runtime.

```bash
.venv/bin/python -m moe.moe_assembly \
  --bundle data/gemma_scope2_270m_pilecc/analysis/\
domain_triage_selected_6_20260914/moe_bundle.json
```

## Zasady reprodukcji

1. Najpierw weryfikuje się hashe modelu, kodu, routera i masek.
2. Eksperci są rekonstruowani z **tych samych** bazowych wag MLP.
3. Kolejność klas routera musi odpowiadać `class_idx` w bundle.
4. Próg 0,8246666193 pochodzi z kalibracji development i nie może być
   dostrajany na zużytym holdoucie ani zapieczętowanym benchmarku.
5. Zmiana któregokolwiek artefaktu tworzy nowy eksperyment i wymaga nowego
   manifestu oraz nowej oceny.

## Katalog funkcji

| Moduł | Funkcja | Odpowiedzialność |
| --- | --- | --- |
| freeze | `sha256_file()` / `file_record()` / `records()` | rekordy integralności |
| freeze | `main()` | walidacja kompletności i zapis protokołu |
| bundle | `relative_record()` | bezpieczna ścieżka względem workspace i hash |
| bundle | `main()` | powiązanie runtime z provenance i wynikiem holdoutu |

