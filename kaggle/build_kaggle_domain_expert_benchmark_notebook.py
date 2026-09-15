"""Generate a Kaggle notebook for standalone Gemma domain-expert evaluation."""

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
            """# Gemma 3 270M: osobna walidacja ekspertów dziedzinowych

Notebook ocenia sześć modeli, w których MLP warstwy 9 zastąpiono jednym kompaktowym ekspertem domenowym. Ekspert przetwarza **każdy token**: nie ma routera ani gęstego fallbacku. To pozwala oddzielić jakość samej maski eksperta od błędów routingu MoE oraz zmierzyć jej rzeczywisty koszt.

Domyślnie każdy ekspert jest oceniany tylko na własnej domenie benchmarku v3. `EVALUATE_CROSS_DOMAIN=True` uruchamia dodatkowo pełną macierz 6 ekspertów × 6 domen, ale jest około sześć razy droższe. Ponieważ benchmark v3 został już użyty do oceny MoE, analiza ekspertów jest **post-hoc i eksploracyjna**. Nie wolno wybierać na jej podstawie nowej maski i raportować wyniku na tym samym zbiorze jako niezależnego.
"""
        ),
        code(
            """import subprocess
import sys

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',
    'transformers==4.57.6',
], check=True)
"""
        ),
        code(
            """import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SEED = 20260915
BATCH_SIZE = 8
MAX_SCORE_LENGTH = 1024
MC_ORDER_RUNS = 3
MC_ORDER_SEED = 20260916
N_BOOTSTRAP = 2000
RUN_PYTHON_GENERATION = True
MAX_NEW_TOKENS = 160
EVALUATE_CROSS_DOMAIN = False
RUN_PERFORMANCE_BENCHMARK = True
PERF_BATCH_SIZES = (1, 4, 8)
PERF_WARMUP = 3
PERF_REPEATS = 20
PERF_MAX_LENGTH = 512
OUTPUT_DIR = Path('/kaggle/working/gemma_domain_expert_benchmark_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)
"""
        ),
        markdown("""## Weryfikacja assetów i checkpointu"""),
        code(
            """def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

matches = []
for path in Path('/kaggle/input').rglob('asset_manifest.json'):
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('format') == 'gemma_domain_expert_kaggle_assets_v1':
        matches.append((path, payload))
if len(matches) != 1:
    raise RuntimeError(f'Expected one standalone-expert asset manifest, found {len(matches)}')
ASSET_MANIFEST_PATH, ASSET_MANIFEST = matches[0]
ASSET_ROOT = ASSET_MANIFEST_PATH.parent
for record in ASSET_MANIFEST['runtime_files']:
    path = ASSET_ROOT / record['path']
    if not path.is_file() or path.stat().st_size != int(record['bytes']):
        raise RuntimeError(f'Missing or size-mismatched asset: {path}')
    if sha256_file(path) != record['sha256']:
        raise RuntimeError(f'SHA-256 mismatch: {path}')

def model_dir_matches(path):
    path = Path(path)
    return all(
        (path / r['filename']).is_file()
        and (path / r['filename']).stat().st_size == int(r['bytes'])
        and sha256_file(path / r['filename']) == r['sha256']
        for r in ASSET_MANIFEST['base_model_files']
    )

known = [
    Path('/kaggle/input/models/google/gemma-3/transformers/gemma-3-270m/2'),
    Path('/kaggle/input/gemma-3/transformers/gemma-3-270m/2'),
]
GEMMA_MODEL_DIR = next((path for path in known if model_dir_matches(path)), None)
if GEMMA_MODEL_DIR is None:
    GEMMA_MODEL_DIR = next(
        (p.parent for p in Path('/kaggle/input').rglob('config.json') if model_dir_matches(p.parent)),
        None,
    )
if GEMMA_MODEL_DIR is None:
    raise FileNotFoundError('Attach google/gemma-3/transformers/gemma-3-270m/2')
print('Assets and base checkpoint verified')
"""
        ),
        markdown("""## Dane i funkcje ewaluacji"""),
        code(
            """CODE_DIR = ASSET_ROOT / 'code'
sys.path.insert(0, str(CODE_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from moe.moe_assembly import build_single_mask_expert_model
from evaluation.moe_benchmark import (
    benchmark_prefill_runtime,
    evaluate_model,
    generate_task_outputs,
    load_benchmark,
    write_json,
)
from evaluation.domain_expert_benchmark import (
    benchmark_domain_ids,
    compare_python_syntax,
    compare_runtime_results,
    compare_standalone_expert,
    select_domain_rows,
    subset_evaluation,
    summarize_own_domain_comparisons,
    theoretical_single_expert_compute,
)

BENCHMARK_PATH = ASSET_ROOT / 'benchmark' / 'benchmark.jsonl'
if sha256_file(BENCHMARK_PATH) != ASSET_MANIFEST['benchmark_sha256']:
    raise RuntimeError('Benchmark hash differs from the asset manifest')
benchmark_rows = load_benchmark(BENCHMARK_PATH)
domain_ids = benchmark_domain_ids(benchmark_rows)
domain_names = {
    domain_id: next(r['domain_name'] for r in benchmark_rows if int(r['domain_id']) == domain_id)
    for domain_id in domain_ids
}
domain_rows = {domain_id: select_domain_rows(benchmark_rows, [domain_id]) for domain_id in domain_ids}
performance_prompts = {
    domain_id: [row['prompt'] for row in domain_rows[domain_id]][:8]
    for domain_id in domain_ids
}
print({domain_names[k]: len(v) for k, v in domain_rows.items()})
"""
        ),
        markdown("""## Model bazowy — jeden wspólny punkt odniesienia"""),
        code(
            """if DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA_MODEL_DIR), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(GEMMA_MODEL_DIR), local_files_only=True, torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    return model, tokenizer

base_model, tokenizer = load_base_model()
base_parameter_count = sum(p.numel() for p in base_model.parameters())
model_dimensions = {
    'd_model': int(base_model.config.hidden_size),
    'dense_width': int(base_model.config.intermediate_size),
    'n_mlp_layers': int(base_model.config.num_hidden_layers),
}
base_evaluation = evaluate_model(
    base_model, tokenizer, benchmark_rows, model_variant='base',
    batch_size=BATCH_SIZE, max_length=MAX_SCORE_LENGTH,
    mc_order_runs=MC_ORDER_RUNS, mc_order_seed=MC_ORDER_SEED,
)
write_json(OUTPUT_DIR / 'base_teacher_forced.json', base_evaluation)

base_python_generation = None
if RUN_PYTHON_GENERATION:
    base_python_generation = generate_task_outputs(
        base_model, tokenizer, domain_rows[5], model_variant='base',
        batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
    )
    write_json(OUTPUT_DIR / 'base_python_generation.json', base_python_generation)

base_runtime = {}
if RUN_PERFORMANCE_BENCHMARK:
    for domain_id in domain_ids:
        base_runtime[str(domain_id)] = benchmark_prefill_runtime(
            base_model, tokenizer, performance_prompts[domain_id],
            batch_sizes=PERF_BATCH_SIZES, max_length=PERF_MAX_LENGTH,
            warmup=PERF_WARMUP, repeats=PERF_REPEATS,
        )
    write_json(OUTPUT_DIR / 'base_prefill_runtime_by_domain.json', base_runtime)

del base_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
        ),
        markdown("""## Sześciu ekspertów bez routera"""),
        code(
            """expert_evaluations = {}
expert_comparisons = []
expert_runtimes = {}
expert_parameter_counts = {}
python_expert_generation = None

for domain_id in domain_ids:
    print(f'Expert {domain_id}: {domain_names[domain_id]}')
    expert_model, tokenizer = load_base_model()
    build_single_mask_expert_model(
        expert_model,
        experts_dir=ASSET_ROOT / 'experts',
        domain_id=domain_id,
        layer_num=int(ASSET_MANIFEST['layer_num']),
        model_name=ASSET_MANIFEST['model_identity'],
    )
    expert_parameter_counts[str(domain_id)] = sum(p.numel() for p in expert_model.parameters())
    rows_to_score = benchmark_rows if EVALUATE_CROSS_DOMAIN else domain_rows[domain_id]
    evaluation = evaluate_model(
        expert_model, tokenizer, rows_to_score,
        model_variant=f'expert_{domain_id}', batch_size=BATCH_SIZE,
        max_length=MAX_SCORE_LENGTH, mc_order_runs=MC_ORDER_RUNS,
        mc_order_seed=MC_ORDER_SEED,
    )
    expert_evaluations[str(domain_id)] = evaluation
    write_json(OUTPUT_DIR / f'expert_{domain_id}_teacher_forced.json', evaluation)

    matching_base = subset_evaluation(
        base_evaluation,
        domain_ids if EVALUATE_CROSS_DOMAIN else [domain_id],
    )
    comparison = compare_standalone_expert(
        matching_base, evaluation, expert_domain_id=domain_id,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    expert_comparisons.append(comparison)
    write_json(OUTPUT_DIR / f'expert_{domain_id}_paired_comparison.json', comparison)

    if RUN_PERFORMANCE_BENCHMARK:
        runtime = benchmark_prefill_runtime(
            expert_model, tokenizer, performance_prompts[domain_id],
            batch_sizes=PERF_BATCH_SIZES, max_length=PERF_MAX_LENGTH,
            warmup=PERF_WARMUP, repeats=PERF_REPEATS,
        )
        expert_runtimes[str(domain_id)] = runtime
        write_json(OUTPUT_DIR / f'expert_{domain_id}_prefill_runtime.json', runtime)

    if RUN_PYTHON_GENERATION and domain_id == 5:
        python_expert_generation = generate_task_outputs(
            expert_model, tokenizer, domain_rows[5], model_variant='expert_5',
            batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
        )
        write_json(OUTPUT_DIR / 'expert_5_python_generation.json', python_expert_generation)

    del expert_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""
        ),
        markdown("""## Podsumowanie jakości i kosztu"""),
        code(
            """summary_rows = summarize_own_domain_comparisons(expert_comparisons)
pruning = json.loads((ASSET_ROOT / 'experts' / 'pruning_summary.json').read_text())
pruning_by_domain = {int(row['domain_id']): row for row in pruning['domains']}

for row in summary_rows:
    domain_id = int(row['domain_id'])
    base_domain = base_evaluation['domains'][str(domain_id)]
    expert_domain = expert_evaluations[str(domain_id)]['domains'][str(domain_id)]
    row['base_reference_token_weighted_nll'] = base_domain['reference_token_weighted_nll']
    row['expert_reference_token_weighted_nll'] = expert_domain['reference_token_weighted_nll']
    row['base_parameter_count'] = base_parameter_count
    row['expert_parameter_count'] = expert_parameter_counts[str(domain_id)]
    row['parameter_reduction'] = base_parameter_count - expert_parameter_counts[str(domain_id)]
    compute = theoretical_single_expert_compute(
        d_model=model_dimensions['d_model'],
        dense_width=model_dimensions['dense_width'],
        expert_width=int(pruning_by_domain[domain_id]['n_kept']),
        n_mlp_layers=model_dimensions['n_mlp_layers'],
    )
    row.update({
        'modified_layer_mac_reduction_fraction': compute['modified_layer_mac_reduction_fraction'],
        'all_mlp_mac_reduction_fraction': compute['all_mlp_mac_reduction_fraction'],
    })

quality_table = pd.DataFrame(summary_rows)
quality_table.to_csv(OUTPUT_DIR / 'own_domain_quality_summary.csv', index=False)
display(quality_table.style.format(precision=4))

runtime_rows = []
if RUN_PERFORMANCE_BENCHMARK:
    for domain_id in domain_ids:
        for row in compare_runtime_results(
            base_runtime[str(domain_id)], expert_runtimes[str(domain_id)]
        ):
            row['domain_id'] = domain_id
            row['domain_name'] = domain_names[domain_id]
            runtime_rows.append(row)
    runtime_table = pd.DataFrame(runtime_rows)
    runtime_table.to_csv(OUTPUT_DIR / 'expert_prefill_runtime_summary.csv', index=False)
    display(runtime_table.style.format(precision=4))

generation_summary = None
if RUN_PYTHON_GENERATION:
    generation_summary = compare_python_syntax(
        base_python_generation, python_expert_generation
    )
    write_json(OUTPUT_DIR / 'python_generation_summary.json', generation_summary)
    display(pd.DataFrame([generation_summary]))

experiment_summary = {
    'format': 'gemma_standalone_domain_expert_benchmark_v1',
    'post_hoc_status': 'exploratory; benchmark v3 was previously consumed by the MoE evaluation',
    'evaluate_cross_domain': EVALUATE_CROSS_DOMAIN,
    'base_parameter_count': base_parameter_count,
    'model_dimensions': model_dimensions,
    'expert_parameter_counts': expert_parameter_counts,
    'quality': summary_rows,
    'runtime': runtime_rows,
    'python_generation': generation_summary,
}
write_json(OUTPUT_DIR / 'standalone_expert_summary.json', experiment_summary)
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(quality_table))
axes[0].bar(x, quality_table['mean_delta_gold_nll_expert_minus_base'])
axes[0].axhline(0, color='black', linewidth=1)
axes[0].set_xticks(x, quality_table['domain_name'], rotation=35, ha='right')
axes[0].set_ylabel('Expert − base mean paired NLL')
axes[0].set_title('Jakość eksperta na własnej domenie')

if RUN_PERFORMANCE_BENCHMARK:
    for batch_size, group in runtime_table.groupby('batch_size'):
        axes[1].plot(group['domain_name'], group['speedup_base_over_expert'], marker='o', label=f'batch={batch_size}')
    axes[1].axhline(1, color='black', linewidth=1)
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].set_ylabel('Base time / expert time (>1 = ekspert szybszy)')
    axes[1].set_title('Prefill na promptach własnej domeny')
    axes[1].legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'standalone_expert_comparison.png', dpi=180, bbox_inches='tight')
plt.show()

if EVALUATE_CROSS_DOMAIN:
    cross_rows = []
    for comparison in expert_comparisons:
        expert_id = int(comparison['expert_domain_id'])
        for tested_id, domain in comparison['domains'].items():
            cross_rows.append({
                'expert_domain_id': expert_id,
                'expert_domain_name': domain_names[expert_id],
                'tested_domain_id': int(tested_id),
                'tested_domain_name': domain['domain_name'],
                'mean_delta_gold_nll_expert_minus_base': domain['mean_delta_gold_nll_moe_minus_base'],
            })
    pd.DataFrame(cross_rows).to_csv(OUTPUT_DIR / 'cross_domain_nll_matrix.csv', index=False)
"""
        ),
        markdown(
            """## Interpretacja

- Ujemne `expert − base NLL` przemawia za ekspertem; CI obejmujące zero nie rozstrzyga kierunku.
- Accuracy należy interpretować razem z wariantami kolejności, ponieważ Gemma 270M ma silny bias pozycyjny.
- Ekspert ma zawsze aktywne 75% MLP-9, więc teoretycznie oszczędza 25% MAC tej warstwy i nie ponosi kosztu routera.
- O rzeczywistym speed-upie decyduje `speedup_base_over_expert > 1`, nie liczba zachowanych neuronów.
- Syntax rate Pythona nie jest pass@1.
- Wyniki są eksploracyjne i nie upoważniają do strojenia na benchmarku v3.
"""
        ),
        code(
            """import shutil

archive = shutil.make_archive(
    '/kaggle/working/gemma_domain_expert_benchmark_results',
    'zip', root_dir=OUTPUT_DIR,
)
print('Results archive:', archive)
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
    output = Path("kaggle/notebooks/kaggle_gemma_domain_expert_benchmark.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook written to: {output}")


if __name__ == "__main__":
    main()
