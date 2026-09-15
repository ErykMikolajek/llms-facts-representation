# Kaggle

Wszystkie elementy uruchamiania eksperymentów na Kaggle znajdują się tutaj:

- `cli/` — publikowanie assetów, start, status i kontynuacja runów;
- `notebooks/` — notebooki Pythia, Gemma Scope i benchmark MoE;
- `build_kaggle_moe_benchmark_notebook.py` — deterministyczny builder notebooka;
- `prepare_kaggle_moe_benchmark_assets.py` — budowa samowystarczalnego Datasetu;
- `build_kaggle_domain_expert_benchmark_notebook.py` — osobna walidacja sześciu ekspertów;
- `prepare_kaggle_domain_expert_benchmark_assets.py` — paczka bez routera i fallbacku;
- `assets/` — wersjonowane paczki gotowe do uploadu.

Benchmark MoE od budowy do uruchomienia:

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

Szczegóły credentials i wielokontowych runów: [`cli/README.md`](cli/README.md).

Osobny benchmark ekspertów:

```bash
.venv/bin/python kaggle/build_kaggle_domain_expert_benchmark_notebook.py
.venv/bin/python kaggle/prepare_kaggle_domain_expert_benchmark_assets.py \
  --experiment-dir data/gemma_scope2_270m_pilecc/analysis/domain_triage_selected_6_20260914 \
  --model-dir models/gemma-3-270m --workspace-root .
.venv/bin/python kaggle/cli/kaggle_pipeline.py publish-assets \
  --workflow domain_expert_benchmark --account account1 \
  --directory kaggle/assets/gemma_domain_expert_benchmark_v1 --create
.venv/bin/python kaggle/cli/kaggle_pipeline.py launch \
  --workflow domain_expert_benchmark --account account1 --wait
```

Do Kaggle należy przesłać prywatny katalog
`kaggle/assets/gemma_domain_expert_benchmark_v1` i uruchomić notebook
`kaggle_gemma_domain_expert_benchmark.ipynb`.
