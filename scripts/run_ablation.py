"""
J-space ablation on Qwen3-4B: delete the top-k Jacobian-lens directions from the
residual stream across a measured mid-network band, then re-measure accuracy on
GSM8K, MATH-500 and AIME 2024 with and without chain of thought.

    export VIRTUAL_ENV=$PWD/../J-Env && uv run --active python run_ablation.py

Per-cell records are cached under output-ablation/cells/, so an interrupted run resumes
where it stopped. The band is cached too, since it depends only on the model.
"""

import argparse
import json
from pathlib import Path

import torch

from ablate import sub_bands
from band import find_band, signatures, wikitext_prompts
from lens import lens_parts, load_lens, load_model
from plot import plot_accuracy, plot_average, plot_difficulty
from tasks import load_task, run_matrix, save_csv, summarize

TASK_ORDER = ["aime2024", "math500", "gsm8k"]
BAND_PROMPTS = 20

# k=5 has the widest specificity margin in the sweep
K = 5

OUT = Path(__file__).parent.parent / "results"


def measure_band(model, tokenizer, lens, path):
    """Measure the workspace band once and cache it."""
    if path.exists():
        saved = json.loads(path.read_text())
        return saved["band"], saved["onset"], saved["offset"]

    prompts = wikitext_prompts(BAND_PROMPTS)
    kurtosis, agreement = signatures(model, tokenizer, lens, prompts)
    band, onset, offset = find_band(kurtosis, agreement)
    path.write_text(json.dumps({
        "band": band,
        "onset": onset,
        "offset": offset,
        "kurtosis": kurtosis,
        "agreement": agreement,
        "n_prompts": BAND_PROMPTS,
    }, indent=2))
    return band, onset, offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="first N problems per task")
    args = parser.parse_args()

    cell_dir = OUT / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model, tokenizer = load_model()
    lens = load_lens()

    band, onset, offset = measure_band(model, tokenizer, lens, OUT / "band.json")
    bands = sub_bands(band)
    parts = lens_parts(model, lens, bands["heavy"])
    print(f"band L{band[0]}..L{band[1]} (onset {onset}, offset {offset}), k={K}")
    for name, layers in bands.items():
        print(f"  {name} L{layers[0]}..L{layers[-1]} ({len(layers)} layers)")

    results, counts = {}, {}
    for task in TASK_ORDER:
        items = load_task(task, limit=args.limit)
        counts[task] = len(items)
        print(f"{task}: {len(items)} problems")

        results[task] = run_matrix(model, tokenizer, items, task, band, K, parts, cache_dir=cell_dir)
        save_csv(OUT / f"{task}_records.csv", results[task])
        plot_accuracy(results[task], task, counts[task], OUT / f"{task}_accuracy.png")
        plot_difficulty(results[task], task, OUT / f"{task}_difficulty.png")

    average, _ = plot_average(results, OUT / "average_accuracy.png")

    summary = {
        "band": band,
        "onset": onset,
        "offset": offset,
        "k": K,
        "sub_bands": bands,
        "n_problems": counts,
        **summarize(results),
        "average": average,
    }
    (OUT / "ablation_results.json").write_text(json.dumps(summary, indent=2))

    for task, cells in summary["accuracy"].items():
        for style in ("direct", "cot"):
            clean = cells[f"{style}_none"]
            ratio = cells[f"{style}_heavy"] / clean if clean else 0.0
            print(f"{task} {style} heavy/unablated = {ratio:.3f}x")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
