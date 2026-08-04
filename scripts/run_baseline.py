"""
Unablated Qwen3-4B accuracy on GSM8K, MATH-500 and AIME 2024.

    export VIRTUAL_ENV=$PWD/../J-Env && uv run --active python run_baseline.py

Shares the per-cell cache with run_ablation.py, so if the ablation has already run its
unablated cells this reads them back rather than regenerating them.
"""

import argparse
import json
from pathlib import Path

import torch

from lens import load_model
from plot import plot_baseline, plot_difficulty
from tasks import load_task, run_matrix, save_csv, summarize

TASK_ORDER = ["aime2024", "math500", "gsm8k"]
OUT = Path(__file__).parent.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,help="first N problems per task")
    args = parser.parse_args()

    cell_dir = OUT / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model, tokenizer = load_model()

    results, counts = {}, {}
    for task in TASK_ORDER:
        items = load_task(task, limit=args.limit)
        counts[task] = len(items)
        print(f"{task}: {len(items)} problems")

        results[task] = run_matrix(
            model, tokenizer, items, task, band=None, k=None, parts=None,cache_dir=cell_dir, conditions=["none"],
        )
        save_csv(OUT / f"{task}_baseline.csv", results[task])

    plot_baseline(results, OUT / "baseline_accuracy.png")
    for task in results:
        plot_difficulty(results[task], task, OUT / f"{task}_baseline_difficulty.png")

    summary = {"n_problems": counts, **summarize(results)}
    (OUT / "baseline_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
