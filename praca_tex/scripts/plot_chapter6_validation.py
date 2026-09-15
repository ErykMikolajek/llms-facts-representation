"""Reproduce the chapter 6 figure from the current MoE benchmark results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/gemma_moe_validation/gemma_moe_benchmark_results_15_09/paired_comparison.json"
OUTPUT = ROOT / "praca_tex/images/moe_validation_nll.png"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    labels = ["Prawo", "Biomedycyna", "Sport", "Polityka", "Matematyka", "Python", "Łącznie"]
    rows = [data["domains"][str(i)] for i in range(6)] + [data["overall"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    for i, row in enumerate(rows):
        mean = row["mean_delta_gold_nll_moe_minus_base"]
        low, high = row["delta_gold_nll_ci95"]
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o",
                    color="#202020" if i == 6 else "#315e86", capsize=4, markersize=6)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(5.5, color="#cccccc", linewidth=0.8)
    ax.set_yticks(range(7), labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.055, 1.23)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".replace(".", ",")))
    ax.set_xlabel("Średnia sparowana różnica NLL (MoE − model bazowy)")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
