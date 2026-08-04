"""
Plots for the ablation: per-task accuracy bars, per-task accuracy by difficulty bin,
and the mean across tasks.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks import CONDITIONS, N_BINS, STYLES, accuracy, accuracy_by_difficulty

STYLE_COLORS = {"direct": "#2a78d6", "cot": "#eb6834"}
STYLE_LABELS = {"direct": "direct", "cot": "chain of thought"}
CONDITION_COLORS = {
    "none": "#4c4c4c",
    "light": "#7fb069",
    "medium": "#e8a33d",
    "heavy": "#d1495b",
}
CONDITION_LABELS = {
    "none": "unablated",
    "light": "light",
    "medium": "medium",
    "heavy": "heavy",
}
BIN_LABELS = [f"D{level}" for level in range(1, N_BINS + 1)]
BAR_WIDTH = 0.35


def dress(axis):
    """Shared axis styling."""
    axis.set_ylim(0, 1)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def label_bars(axis, drawn, values):
    """Print each bar's value just above it."""
    for bar, value in zip(drawn, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )


def bars(axis, scores, title):
    """Grouped accuracy bars, one group per condition and one bar per style."""
    positions = list(range(len(CONDITIONS)))
    for index, style in enumerate(STYLES):
        values = [scores[f"{style}_{condition}"] for condition in CONDITIONS]
        offsets = [position + (index - 0.5) * BAR_WIDTH for position in positions]
        drawn = axis.bar(
            offsets, values, BAR_WIDTH, color=STYLE_COLORS[style],
            label=STYLE_LABELS[style],
        )
        label_bars(axis, drawn, values)

    axis.set_xticks(positions)
    axis.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITIONS])
    axis.set_ylabel("exact match accuracy")
    axis.set_title(title)
    axis.legend(frameon=False)
    dress(axis)


def plot_accuracy(results, task, n, path):
    """Overall accuracy per condition for one task."""
    scores = {name: accuracy(records) for name, records in results.items()}
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    bars(axis, scores, f"J-space ablation on Qwen3-4B: {task} (n={n})")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_difficulty(results, task, path):
    """Accuracy by difficulty bin, one panel per prompt style."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, style in zip(axes, STYLES):
        present = [c for c in CONDITIONS if f"{style}_{c}" in results]
        for condition in present:
            curve = accuracy_by_difficulty(results[f"{style}_{condition}"])
            axis.plot(
                BIN_LABELS, curve, "o-", lw=2, ms=6,
                color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition],
            )
        axis.set_xlabel("difficulty bin (gold solution length)")
        axis.set_title(f"{task}: {STYLE_LABELS[style]}")
        axis.legend(frameon=False, fontsize=9)
        dress(axis)

    axes[0].set_ylabel("exact match accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_baseline(results_by_task, path):
    """Unablated accuracy, one bar group per task and one bar per prompt style."""
    tasks = list(results_by_task)
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for index, style in enumerate(STYLES):
        values = [
            accuracy(results_by_task[task][f"{style}_none"]) for task in tasks
        ]
        offsets = [
            position + (index - 0.5) * BAR_WIDTH for position in range(len(tasks))
        ]
        drawn = axis.bar(
            offsets, values, BAR_WIDTH, color=STYLE_COLORS[style],
            label=STYLE_LABELS[style],
        )
        label_bars(axis, drawn, values)

    axis.set_xticks(range(len(tasks)))
    axis.set_xticklabels(tasks)
    axis.set_ylabel("exact match accuracy")
    axis.set_title("Qwen3-4B baseline, no ablation")
    axis.legend(frameon=False)
    dress(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_average(results_by_task, path):
    """Mean accuracy across the tasks."""
    tasks = list(results_by_task)
    scores = {}
    for style in STYLES:
        for condition in CONDITIONS:
            name = f"{style}_{condition}"
            total = sum(accuracy(results_by_task[task][name]) for task in tasks)
            scores[name] = total / len(tasks)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    bars(axis, scores, "Mean over " + ", ".join(tasks))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return scores, path
