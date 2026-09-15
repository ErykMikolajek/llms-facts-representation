# Osobny benchmark ekspertów dziedzinowych

## Cel

Istniejąca funkcja `evaluate_single_experts()` w `moe_validation.py` mierzy
PPL ekspertów na holdoucie. Nie obejmuje jednak zadań kompetencyjnych ani
rzeczywistego kosztu każdego samodzielnego modelu. Ten zakres realizują:

- `evaluation/domain_expert_benchmark.py`;
- `build_single_mask_expert_model()` w `moe/moe_assembly.py`;
- `kaggle/notebooks/kaggle_gemma_domain_expert_benchmark.ipynb`;
- `kaggle/prepare_kaggle_domain_expert_benchmark_assets.py`.

## Konstrukcja modelu

Samodzielny ekspert powstaje z bazowej Gemmy i jednej boolowskiej maski.
Kompaktowy MLP o szerokości 1536 zastępuje gęsty MLP warstwy 9 o szerokości
2048. Model nie zawiera routera, banku sześciu ekspertów ani gęstego fallbacku.
Wybrany ekspert przetwarza każdy token.

Lokalny smoke test potwierdził poprawny forward oraz redukcję liczby
parametrów z 268 098 176 do 267 115 136, czyli dokładnie o 983 040 parametrów.

## Protokół jakości

Notebook najpierw liczy jeden wspólny baseline na 600 rekordach benchmarku v3.
Następnie buduje kolejno sześć modeli i domyślnie ocenia każdy na 100 rekordach
jego domeny. Używa tych samych metryk co benchmark MoE:

- sparowane reference-completion NLL i bootstrapowe 95% CI;
- accuracy oraz exact paired test dla oryginalnego porządku MC;
- accuracy i zgodność po 2–3 wariantach kolejności;
- syntax rate dla eksperta Python bez wykonywania kodu.

`EVALUATE_CROSS_DOMAIN=True` ocenia każdy model na wszystkich 600 rekordach i
zapisuje macierz transferu. Opcja jest domyślnie wyłączona, ponieważ zwiększa
koszt części teacher-forced około sześciokrotnie.

Benchmark v3 był wcześniej użyty do oceny MoE. Analiza single-expert jest więc
post-hoc i eksploracyjna. Można porównać zamrożone modele, ale nie wolno wybrać
na tej podstawie nowej maski i ponownie nazwać wyniku niezależnym.

## Compute i pamięć

Dla gated MLP koszt trzech projekcji wynosi `3 × D × H`. Ekspert 1536/2048
zmniejsza MAC zmodyfikowanej warstwy o 25% i nie ponosi kosztu routera. Ponieważ
zmieniono tylko jedną z 18 warstw, odpowiada to 1,389% kosztu wszystkich MLP,
przed uwzględnieniem attention, norm i LM head.

Notebook mierzy osobno dla każdej domeny i batch size 1/4/8:

- medianę oraz kwartyle latency prefill po warm-upie;
- tokens/s;
- bazową i szczytową alokację CUDA;
- dokładną liczbę parametrów modelu;
- `speedup_base_over_expert`, gdzie wartość powyżej 1 oznacza szybszego eksperta.

Teoretyczna redukcja MAC nie jest traktowana jako zmierzony speed-up. Notebook
nie mierzy autoregresyjnego decode ani energii i nie przechowuje surowych
czasów poszczególnych iteracji, więc nie wyznacza CI dla latency.

## Uruchomienie na Kaggle

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

Należy przesłać jako prywatny Dataset katalog lub ZIP
`kaggle/assets/gemma_domain_expert_benchmark_v1`, podpiąć checkpoint
`google/gemma-3/transformers/gemma-3-270m/2`, a następnie uruchomić notebook
`kaggle_gemma_domain_expert_benchmark.ipynb`.

Wyniki trafiają do
`/kaggle/working/gemma_domain_expert_benchmark_results.zip`. Najważniejsze
pliki to `own_domain_quality_summary.csv`,
`expert_prefill_runtime_summary.csv`, `standalone_expert_summary.json` oraz
osobne JSON-y każdego eksperta.
