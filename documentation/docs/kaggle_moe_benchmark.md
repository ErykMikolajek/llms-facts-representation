# Notebook i paczka benchmarku Kaggle

## Artefakty v3

| Plik | Rola |
| --- | --- |
| `kaggle/notebooks/kaggle_gemma_moe_benchmark.ipynb` | porównanie base–MoE na GPU |
| `kaggle/build_kaggle_moe_benchmark_notebook.py` | deterministyczny builder notebooka |
| `kaggle/prepare_kaggle_moe_benchmark_assets.py` | minimalna, weryfikowalna paczka runtime |
| `kaggle/assets/gemma_moe_benchmark_v3.zip` | gotowy prywatny Kaggle Dataset |

Poprawki należy wprowadzać w builderze, a następnie ponownie generować
`.ipynb`. Packager domyślnie używa `independent_benchmark_v3_sealed` i odmawia
nadpisania istniejącego katalogu wynikowego.

Paczka zawiera benchmark z manifestem, router, sześć masek, podsumowanie
pruningu i minimalny kod runtime. `asset_manifest.json` przechowuje rozmiary i
SHA-256 wszystkich plików oraz oczekiwanych plików bazowej Gemmy. Eksperci są
odtwarzani z bazowych wag i masek, więc paczka nie dubluje ich checkpointów.
`ATTRIBUTION.md` jest budowany z manifestu źródeł zamiast stałej listy.

## Przebieg notebooka

1. Weryfikuje integralność assetów i dokładny checkpoint Gemmy.
2. Ładuje 600 rekordów v3 i wybiera identyczny zestaw promptów wydajnościowych.
3. Ocenia bazę: NLL, MC w 2–3 kolejnościach, generację MBPP i prefill.
4. Składa MoE z zamrożonego routera, progu i masek i powtarza te same pomiary.
5. Liczy sparowane bootstrap CI, exact paired p-value, routing oraz stabilność
   odpowiedzi względem pozycji.
6. Zapisuje pełne JSON-y, `domain_summary.csv`,
   `prefill_runtime_comparison.csv`, wykres i ZIP.

Pomiar prefill używa batch size 1/4/8, trzech iteracji warm-up, dwudziestu
powtórzeń i synchronizacji CUDA. Kolumna `speedup_base_over_moe > 1` oznacza,
że MoE było szybsze. Wynik nie obejmuje autoregresyjnego decode ani energii.

!!! warning "Bezpieczeństwo MBPP"
    Notebook sprawdza tylko niepusty AST przez `ast.parse`. Nie wykonuje kodu
    modelu. Pass@1 należy liczyć w izolowanym środowisku, bez sekretów i sieci.

## Uruchomienie

```bash
.venv/bin/python kaggle/build_kaggle_moe_benchmark_notebook.py
.venv/bin/python kaggle/prepare_kaggle_moe_benchmark_assets.py \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --model-dir models/gemma-3-270m --workspace-root .
.venv/bin/python kaggle/cli/kaggle_pipeline.py publish-assets \
  --workflow moe_benchmark --account account1 \
  --directory kaggle/assets/gemma_moe_benchmark_v3 --create
.venv/bin/python kaggle/cli/kaggle_pipeline.py launch \
  --workflow moe_benchmark --account account1 --wait
```

Przy aktualizacji istniejącego Datasetu należy pominąć `--create`. Workflow
automatycznie podpina prywatny Dataset v3 i model
`google/gemma-3/transformers/gemma-3-270m/2`. Po runie należy pobrać
`gemma_moe_independent_benchmark_results.zip`.

Builder został sprawdzony przez parsowanie każdej komórki kodowej przez
`ast.parse`. Pakiet v3 ma własne hashe; nie należy mieszać go z wynikami v2.
