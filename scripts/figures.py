"""
Build the two report figures from ablation_results.json.

    python3 figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.parent / "results"
CONDITIONS = ["none", "light", "medium", "heavy"]
CONDITION_LABELS = ["unablated", "light", "medium", "heavy"]
BIN_LABELS = ["D1 short", "D2", "D3 long"]

DIRECT = "#2a78d6"
COT = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"

TITLES = {
    "gsm8k": "GSM8K (n = 90)",
    "math500": "MATH-500 (n = 90)",
    "aime2024": "AIME 2024 (n = 9)",
}


def dress(axis):
    """Recessive axes, fixed 0 to 1 accuracy range."""
    axis.set_ylim(0, 1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(axis="y", alpha=0.15)
    axis.set_axisbelow(True)


def dose_response(results, path):
    """Accuracy against ablation strength, one panel per task."""
    tasks = ["gsm8k", "math500", "aime2024"]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)

    for axis, task in zip(axes, tasks):
        scores = results["accuracy"][task]
        for style, color, marker in [("direct", DIRECT, "o"), ("cot", COT, "s")]:
            values = [scores[f"{style}_{condition}"] for condition in CONDITIONS]
            label = "direct" if style == "direct" else "chain of thought"
            axis.plot(CONDITION_LABELS, values, marker=marker, lw=2, ms=7,
                      color=color, label=label)
        axis.set_title(TITLES[task], fontsize=10, color=INK)
        axis.tick_params(axis="x", rotation=20)
        dress(axis)

    axes[0].set_ylabel("exact match accuracy", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[2].text(0.5, 0.9, "too small to interpret", fontsize=8, color=MUTED,
                 ha="center", transform=axes[2].transAxes)

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def difficulty(results, path):
    """Accuracy by difficulty bin, unablated against heavy ablation."""
    tasks = ["gsm8k", "math500"]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)

    for axis, task in zip(axes, tasks):
        curves = results["accuracy_by_difficulty"][task]
        for style, color in [("direct", DIRECT), ("cot", COT)]:
            name = "direct" if style == "direct" else "chain of thought"
            axis.plot(BIN_LABELS, curves[f"{style}_none"], marker="o", lw=2, ms=7,
                      color=color, label=f"{name}, unablated")
            axis.plot(BIN_LABELS, curves[f"{style}_heavy"], marker="o", lw=2, ms=7,
                      color=color, linestyle="--", label=f"{name}, heavy")
        axis.set_title(TITLES[task], fontsize=10, color=INK)
        axis.set_xlabel("difficulty bin (gold solution length)", fontsize=9, color=MUTED)
        dress(axis)

    axes[0].set_ylabel("exact match accuracy", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=9, ncol=4,
                  loc="lower center", bbox_to_anchor=(0.5, -0.02))

    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=200)
    plt.close(figure)


results = json.loads((HERE / "ablation_results.json").read_text())
dose_response(results, HERE / "fig1_dose_response.png")
difficulty(results, HERE / "fig2_difficulty.png")
print("wrote fig1_dose_response.png and fig2_difficulty.png")
