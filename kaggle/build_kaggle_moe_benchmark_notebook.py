"""Generate the reproducible Kaggle notebook for the independent MoE benchmark."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown(
            """# Niezależny benchmark Gemma 3 270M: model bazowy kontra domenowy MoE

Notebook porównuje dokładnie ten sam model bazowy i MoE z sześcioma ekspertami warstwy 9 na 600 nowych, zamrożonych przykładach benchmarku v3. Źródła — MMLU, BIG-bench `sports_understanding`, MBPP oraz generowana arytmetyka elementarna — nie były używane podczas triażu, mapowania, pruningu, treningu routera ani kalibracji progu. Przykłady użyte w poprzednim benchmarku v2 są wykluczone.

**Zasada metodologiczna:** wyników tego notebooka nie wolno używać do dostrajania masek, routera, progu confidence ani promptów. Publiczne zbiory mogły wystąpić w pretrainingu Gemmy; „niezależność” oznacza niezależność od naszego pipeline, a nie gwarancję braku kontaminacji pretrainingu.

Przed uruchomieniem:

1. ustaw akcelerator GPU T4/L4/A100;
2. podepnij Kaggle Model `google/gemma-3/transformers/gemma-3-270m/2`;
3. podepnij jako prywatny Kaggle Dataset katalog lub ZIP `kaggle_gemma_moe_benchmark_assets_v3`;
4. włącz Internet tylko dla pierwszej komórki instalującej zgodną wersję Transformers.
"""
        ),
        code(
            """import subprocess
import sys

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
    'transformers==4.57.6',
], check=True)
print('Dependencies installed')
"""
        ),
        markdown(
            """## Konfiguracja eksperymentu

Teacher-forced NLL jest wspólną metryką dla wszystkich domen. Dla pytań wyboru liczymy trafność przez porównanie warunkowych log-prawdopodobieństw odpowiedzi oraz trzy deterministyczne warianty kolejności opcji (dla binarnego sportu istnieją tylko dwa unikalne warianty). Nie powtarzamy identycznego deterministycznego pomiaru, ponieważ dałby dokładnie ten sam wynik. Dla matematyki używamy prostych, zbalansowanych działań z zapisem LaTeX, a dla MBPP mierzymy NLL kodu referencyjnego i poprawność składni wygenerowanego kodu.

Notebook **nie wykonuje wygenerowanego kodu Python**. Kaggle nie stanowi bezpiecznego sandboxa dla arbitralnego kodu modelu; pass@1 należy później policzyć w izolowanym środowisku EvalPlus/Docker.
"""
        ),
        code(
            """import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SEED = 20260916
BATCH_SIZE = 8
MAX_SCORE_LENGTH = 1024
MC_ORDER_RUNS = 3
MC_ORDER_SEED = 20260916
RUN_GENERATION = True
GENERATION_BATCH_SIZE = 8
MAX_NEW_TOKENS = 160
N_BOOTSTRAP = 2000
RUN_PERFORMANCE_BENCHMARK = True
PERF_BATCH_SIZES = (1, 4, 8)
PERF_WARMUP = 3
PERF_REPEATS = 20
PERF_MAX_LENGTH = 512
OUTPUT_DIR = Path('/kaggle/working/gemma_moe_benchmark_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
"""
        ),
        markdown("""## Odszukanie i kryptograficzna kontrola artefaktów"""),
        code(
            """def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifest_candidates = []
for path in Path('/kaggle/input').rglob('asset_manifest.json'):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('format') == 'gemma_moe_kaggle_assets_v1':
        manifest_candidates.append((path, payload))

if len(manifest_candidates) != 1:
    raise RuntimeError(
        f'Expected exactly one gemma_moe_kaggle_assets_v1 manifest, found {len(manifest_candidates)}'
    )
ASSET_MANIFEST_PATH, ASSET_MANIFEST = manifest_candidates[0]
ASSET_ROOT = ASSET_MANIFEST_PATH.parent

for record in ASSET_MANIFEST['runtime_files']:
    path = ASSET_ROOT / record['path']
    if not path.is_file() or path.stat().st_size != int(record['bytes']):
        raise RuntimeError(f'Missing or size-mismatched asset: {path}')
    if sha256_file(path) != record['sha256']:
        raise RuntimeError(f'SHA-256 mismatch: {path}')

print('Assets verified:', ASSET_ROOT)
print('Benchmark SHA-256:', ASSET_MANIFEST['benchmark_sha256'])
"""
        ),
        code(
            """def model_dir_matches(path):
    path = Path(path)
    for record in ASSET_MANIFEST['base_model_files']:
        candidate = path / record['filename']
        if not candidate.is_file() or candidate.stat().st_size != int(record['bytes']):
            return False
        if sha256_file(candidate) != record['sha256']:
            return False
    return True

known_candidates = [
    Path('/kaggle/input/models/google/gemma-3/transformers/gemma-3-270m/2'),
    Path('/kaggle/input/gemma-3/transformers/gemma-3-270m/2'),
]
GEMMA_MODEL_DIR = next((path for path in known_candidates if model_dir_matches(path)), None)
if GEMMA_MODEL_DIR is None:
    for config_path in Path('/kaggle/input').rglob('config.json'):
        if model_dir_matches(config_path.parent):
            GEMMA_MODEL_DIR = config_path.parent
            break
if GEMMA_MODEL_DIR is None:
    raise FileNotFoundError(
        'Could not find the exact Gemma checkpoint declared by asset_manifest.json. '
        'Attach google/gemma-3/transformers/gemma-3-270m/2.'
    )
print('Exact base checkpoint verified:', GEMMA_MODEL_DIR)
"""
        ),
        markdown("""## Wczytanie zamrożonego benchmarku"""),
        code(
            """CODE_DIR = ASSET_ROOT / 'code'
sys.path.insert(0, str(CODE_DIR))

from moe.moe_assembly import build_hard_routed_moe
from evaluation.moe_benchmark import (
    benchmark_prefill_runtime,
    compare_evaluations,
    evaluate_model,
    generate_task_outputs,
    load_benchmark,
    write_json,
)

BENCHMARK_PATH = ASSET_ROOT / 'benchmark' / 'benchmark.jsonl'
BENCHMARK_MANIFEST_PATH = ASSET_ROOT / 'benchmark' / 'benchmark_manifest.json'
BENCHMARK_MANIFEST = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding='utf-8'))
if sha256_file(BENCHMARK_PATH) != ASSET_MANIFEST['benchmark_sha256']:
    raise RuntimeError('Benchmark hash differs from the sealed asset manifest')
benchmark_rows = load_benchmark(BENCHMARK_PATH)

# Ten sam mały, zbalansowany zestaw promptów służy wyłącznie do pomiaru
# implementacyjnego czasu prefill; nie wpływa na metryki jakości ani routing.
performance_prompts = []
for domain_id in sorted({int(row['domain_id']) for row in benchmark_rows}):
    performance_prompts.extend([
        row['prompt'] for row in benchmark_rows if int(row['domain_id']) == domain_id
    ][:4])

counts = pd.Series([row['domain_name'] for row in benchmark_rows]).value_counts().sort_index()
display(counts.rename('examples').to_frame())
print(BENCHMARK_MANIFEST['independence_definition'])
"""
        ),
        markdown("""## Model bazowy"""),
        code(
            """from transformers import AutoModelForCausalLM, AutoTokenizer

if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported():
    # Ampere/Ada/Hopper: BF16 has the exponent range of FP32 and is the
    # natural dtype of the released Gemma checkpoint.
    DTYPE = torch.bfloat16
else:
    # T4/P100: forcing Gemma to FP16 can produce non-finite logits on some
    # benchmark sequences. Gemma 270M fits comfortably in FP32.
    DTYPE = torch.float32
print('Model dtype selected for numerical stability:', DTYPE)

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA_MODEL_DIR), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(GEMMA_MODEL_DIR),
        local_files_only=True,
        torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    return model, tokenizer

base_model, tokenizer = load_base_model()
started = time.time()
base_evaluation = evaluate_model(
    base_model,
    tokenizer,
    benchmark_rows,
    model_variant='base',
    batch_size=BATCH_SIZE,
    max_length=MAX_SCORE_LENGTH,
    mc_order_runs=MC_ORDER_RUNS,
    mc_order_seed=MC_ORDER_SEED,
)
write_json(OUTPUT_DIR / 'base_teacher_forced.json', base_evaluation)
print(f'Base scoring finished in {(time.time() - started) / 60:.1f} min')

base_generation = None
if RUN_GENERATION:
    started = time.time()
    base_generation = generate_task_outputs(
        base_model,
        tokenizer,
        benchmark_rows,
        model_variant='base',
        batch_size=GENERATION_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    write_json(OUTPUT_DIR / 'base_generation.json', base_generation)
    print(f'Base generation finished in {(time.time() - started) / 60:.1f} min')

base_runtime = None
if RUN_PERFORMANCE_BENCHMARK:
    base_runtime = benchmark_prefill_runtime(
        base_model,
        tokenizer,
        performance_prompts,
        batch_sizes=PERF_BATCH_SIZES,
        max_length=PERF_MAX_LENGTH,
        warmup=PERF_WARMUP,
        repeats=PERF_REPEATS,
    )
    write_json(OUTPUT_DIR / 'base_prefill_runtime.json', base_runtime)

del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
        ),
        markdown("""## MoE z zamrożonym routerem i sześcioma maskami"""),
        code(
            """moe_model, tokenizer = load_base_model()
moe_model = build_hard_routed_moe(
    model=moe_model,
    experts_dir=ASSET_ROOT / 'experts',
    router_path=ASSET_ROOT / 'router' / 'router.pt',
    layer_num=int(ASSET_MANIFEST['layer_num']),
    model_name=ASSET_MANIFEST['model_identity'],
    confidence_threshold=float(ASSET_MANIFEST['confidence_threshold']),
    expert_source=ASSET_MANIFEST['expert_source'],
)

started = time.time()
moe_evaluation = evaluate_model(
    moe_model,
    tokenizer,
    benchmark_rows,
    model_variant='moe',
    batch_size=BATCH_SIZE,
    max_length=MAX_SCORE_LENGTH,
    mc_order_runs=MC_ORDER_RUNS,
    mc_order_seed=MC_ORDER_SEED,
)
write_json(OUTPUT_DIR / 'moe_teacher_forced.json', moe_evaluation)
print(f'MoE scoring finished in {(time.time() - started) / 60:.1f} min')

moe_generation = None
if RUN_GENERATION:
    started = time.time()
    moe_generation = generate_task_outputs(
        moe_model,
        tokenizer,
        benchmark_rows,
        model_variant='moe',
        batch_size=GENERATION_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    write_json(OUTPUT_DIR / 'moe_generation.json', moe_generation)
    print(f'MoE generation finished in {(time.time() - started) / 60:.1f} min')

moe_runtime = None
if RUN_PERFORMANCE_BENCHMARK:
    moe_runtime = benchmark_prefill_runtime(
        moe_model,
        tokenizer,
        performance_prompts,
        batch_sizes=PERF_BATCH_SIZES,
        max_length=PERF_MAX_LENGTH,
        warmup=PERF_WARMUP,
        repeats=PERF_REPEATS,
    )
    write_json(OUTPUT_DIR / 'moe_prefill_runtime.json', moe_runtime)
"""
        ),
        markdown("""## Porównanie sparowane i 95% bootstrap CI"""),
        code(
            """comparison = compare_evaluations(
    base_evaluation,
    moe_evaluation,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
)
write_json(OUTPUT_DIR / 'paired_comparison.json', comparison)

table_rows = []
for domain_id, row in comparison['domains'].items():
    base_domain = base_evaluation['domains'][domain_id]
    moe_domain = moe_evaluation['domains'][domain_id]
    routing = moe_domain['routing']
    routed_fraction = 1.0 - routing['fallback_fraction']
    correct_route = next(
        expert['fraction'] for expert in routing['experts']
        if int(expert['domain_id']) == int(domain_id)
    )
    table_rows.append({
        'domain': row['domain_name'],
        'base_ref_nll': base_domain['reference_token_weighted_nll'],
        'moe_ref_nll': moe_domain['reference_token_weighted_nll'],
        'mean_paired_delta_nll': row['mean_delta_gold_nll_moe_minus_base'],
        'delta_nll_ci_low': row['delta_gold_nll_ci95'][0],
        'delta_nll_ci_high': row['delta_gold_nll_ci95'][1],
        'base_mc_accuracy': row['base_multiple_choice_accuracy'],
        'moe_mc_accuracy': row['moe_multiple_choice_accuracy'],
        'mc_accuracy_delta': row['accuracy_delta_moe_minus_base'],
        'mc_accuracy_delta_ci_low': row['accuracy_delta_ci95'][0] if row['accuracy_delta_ci95'] else None,
        'mc_accuracy_delta_ci_high': row['accuracy_delta_ci95'][1] if row['accuracy_delta_ci95'] else None,
        'mc_exact_paired_pvalue': row['exact_paired_accuracy_pvalue'],
        'base_mc_variant_accuracy': row['base_multiple_choice_variant_accuracy'],
        'moe_mc_variant_accuracy': row['moe_multiple_choice_variant_accuracy'],
        'mc_variant_accuracy_delta': row['variant_accuracy_delta_moe_minus_base'],
        'mc_variant_delta_ci_low': row['variant_accuracy_delta_ci95'][0] if row['variant_accuracy_delta_ci95'] else None,
        'mc_variant_delta_ci_high': row['variant_accuracy_delta_ci95'][1] if row['variant_accuracy_delta_ci95'] else None,
        'routed_fraction': routed_fraction,
        'correct_domain_route_fraction': correct_route,
        'fallback_fraction': routing['fallback_fraction'],
        'effective_layer9_width_fraction': routing['fallback_fraction'] + 0.75 * routed_fraction,
        'prompt_truncation_fraction': moe_domain['prompt_truncation_fraction'],
    })

results_table = pd.DataFrame(table_rows)
results_table.to_csv(OUTPUT_DIR / 'domain_summary.csv', index=False)
display(results_table.style.format(precision=4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(results_table))
axes[0].bar(x - 0.18, results_table['base_ref_nll'], width=0.36, label='Base')
axes[0].bar(x + 0.18, results_table['moe_ref_nll'], width=0.36, label='MoE')
axes[0].set_xticks(x, results_table['domain'], rotation=35, ha='right')
axes[0].set_ylabel('Reference-completion NLL (lower is better)')
axes[0].legend()

errors = np.vstack([
    np.maximum(0.0, results_table['mean_paired_delta_nll'] - results_table['delta_nll_ci_low']),
    np.maximum(0.0, results_table['delta_nll_ci_high'] - results_table['mean_paired_delta_nll']),
])
axes[1].errorbar(x, results_table['mean_paired_delta_nll'], yerr=errors, fmt='o', capsize=4)
axes[1].axhline(0.0, color='black', linewidth=1)
axes[1].set_xticks(x, results_table['domain'], rotation=35, ha='right')
axes[1].set_ylabel('MoE − base paired NLL (95% bootstrap CI)')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'benchmark_comparison.png', dpi=180, bbox_inches='tight')
plt.show()
"""
        ),
        markdown("""## Wyniki generacyjne i interpretacja"""),
        code(
            """if RUN_GENERATION:
    generation_table = pd.DataFrame([
        {
            'model': 'base',
            'mbpp_python_syntax_valid': base_generation['summary']['python_generation']['python_syntax_valid'],
        },
        {
            'model': 'moe',
            'mbpp_python_syntax_valid': moe_generation['summary']['python_generation']['python_syntax_valid'],
        },
    ])
    generation_table.to_csv(OUTPUT_DIR / 'generation_summary.csv', index=False)
    display(generation_table.style.format(precision=4))

print('Interpretation rules:')
print('- delta NLL < 0 favors MoE; CI crossing zero means the direction is uncertain;')
print('- MC accuracy covers law, biomedical, sports, politics, and elementary math;')
print('- variant accuracy averages unique answer-order runs and exposes position sensitivity;')
print('- the exact paired p-value uses original-order item outcomes; inspect CI as effect uncertainty;')
print('- MBPP syntax validity is not pass@1 and does not establish functional correctness;')
print('- any later tuning requires a fresh development set, never this benchmark.')
"""
        ),
        markdown("""## Koszt implementacyjny prefill

Pomiar obejmuje wyłącznie przejście prefill na tym samym GPU, dtype, zestawie promptów i długości wejścia. Warm-up i synchronizacja CUDA ograniczają typowe błędy pomiaru. To nie jest pomiar FLOP, energii ani szybkości autoregresyjnego decode. Dodatni `speedup_base_over_moe` oznacza szybsze MoE; wartość poniżej 1 oznacza narzut implementacji routingu/masek.
"""),
        code(
            """if RUN_PERFORMANCE_BENCHMARK:
    base_perf = pd.DataFrame(base_runtime['results']).add_prefix('base_')
    moe_perf = pd.DataFrame(moe_runtime['results']).add_prefix('moe_')
    performance_table = base_perf.merge(
        moe_perf,
        left_on='base_batch_size',
        right_on='moe_batch_size',
        validate='one_to_one',
    )
    performance_table['speedup_base_over_moe'] = (
        performance_table['base_median_seconds'] / performance_table['moe_median_seconds']
    )
    performance_table['throughput_ratio_moe_over_base'] = (
        performance_table['moe_tokens_per_second'] / performance_table['base_tokens_per_second']
    )
    performance_table['incremental_peak_vram_delta_bytes'] = (
        performance_table['moe_cuda_incremental_peak_bytes']
        - performance_table['base_cuda_incremental_peak_bytes']
    )
    performance_table.to_csv(OUTPUT_DIR / 'prefill_runtime_comparison.csv', index=False)
    display(performance_table.style.format(precision=4))
"""
        ),
        markdown("""## Pakiet wyników do pobrania"""),
        code(
            """import shutil

archive = shutil.make_archive(
    '/kaggle/working/gemma_moe_independent_benchmark_results',
    'zip',
    root_dir=OUTPUT_DIR,
)
print('Results archive:', archive)
print('Files:')
for path in sorted(OUTPUT_DIR.iterdir()):
    print('-', path.name, path.stat().st_size, 'bytes')
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "dockerImageVersionId": None,
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path(__file__).resolve().parent / "notebooks" / "kaggle_gemma_moe_benchmark.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written to: {output}")


if __name__ == "__main__":
    main()
