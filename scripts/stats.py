"""
Recompute the report's result tables from the per-item records in results/.

    python3 stats.py

Table 2 is accuracy in every condition with the heavy to unablated ratio. Table 3 is the
paired comparison of heavy against unablated: Wilson intervals, the items whose
correctness changed, a two-sided exact binomial test over those items, and a rescore that
drops items truncated at the token limit on either side of the pair.
"""

import csv
import math
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
TASKS = ["gsm8k", "math500", "aime2024"]
CONDITIONS = ["none", "light", "medium", "heavy"]
STYLES = ["direct", "cot"]


def load_cells(task):
    """Records for one task, keyed by (style, condition) then by question."""
    rows = list(csv.DictReader(open(RESULTS / f"{task}_records.csv")))
    cells = {}
    for row in rows:
        cells.setdefault((row["style"], row["condition"]), {})[row["question"]] = row
    return cells


def wilson(correct, total):
    """Wilson 95 percent interval for a binomial proportion."""
    z = 1.96
    p = correct / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - spread) / denominator, (centre + spread) / denominator


def exact_binomial(lost, gained):
    """Two-sided exact binomial test over the items whose correctness changed."""
    total = lost + gained
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(lost, gained) + 1)) / 2**total
    return min(1.0, 2 * tail)


def accuracy(cell):
    """Exact match accuracy over one condition."""
    return sum(int(record["correct"]) for record in cell.values()) / len(cell)


def flips(base, other):
    """Items correct only in the baseline, and correct only in the other condition."""
    lost = sum(1 for q in base if int(base[q]["correct"]) and not int(other[q]["correct"]))
    gained = sum(1 for q in base if not int(base[q]["correct"]) and int(other[q]["correct"]))
    return lost, gained


def untruncated_ratio(base, other):
    """Heavy to unablated ratio over items that terminated in both conditions."""
    keep = [q for q in base if not int(base[q]["hit_cap"]) and not int(other[q]["hit_cap"])]
    clean = sum(int(base[q]["correct"]) for q in keep) / len(keep)
    ablated = sum(int(other[q]["correct"]) for q in keep) / len(keep)
    return (ablated / clean if clean else 0.0), len(keep)


print("Table 2. Accuracy in every condition, and the heavy to unablated ratio.\n")
print(f"{'dataset':10} {'style':6} " + " ".join(f"{c:>7}" for c in CONDITIONS) + "    ratio")
for task in TASKS:
    cells = load_cells(task)
    for style in STYLES:
        scores = [accuracy(cells[(style, condition)]) for condition in CONDITIONS]
        ratio = scores[-1] / scores[0] if scores[0] else 0.0
        print(f"{task:10} {style:6} " + " ".join(f"{s:7.3f}" for s in scores) + f"  {ratio:7.3f}")

print("\nTable 3. Heavy against unablated, paired on the same items.\n")
header = f"{'dataset':10} {'style':6} {'unablated 95% CI':22} {'heavy 95% CI':22}"
print(header + f"{'lost':>5}{'gained':>8}{'p':>12}   truncation excluded")
for task in TASKS:
    cells = load_cells(task)
    for style in STYLES:
        base = cells[(style, "none")]
        heavy = cells[(style, "heavy")]
        clean = accuracy(base)
        ablated = accuracy(heavy)
        low0, high0 = wilson(round(clean * len(base)), len(base))
        low1, high1 = wilson(round(ablated * len(heavy)), len(heavy))
        lost, gained = flips(base, heavy)
        ratio, kept = untruncated_ratio(base, heavy)
        print(
            f"{task:10} {style:6} "
            f"{clean:.3f} [{low0:.3f}, {high0:.3f}]  "
            f"{ablated:.3f} [{low1:.3f}, {high1:.3f}]  "
            f"{lost:5}{gained:8}{exact_binomial(lost, gained):12.5g}"
            f"   {ratio:.3f} (n = {kept})"
        )

print("\nCompletion length and truncation, unablated against heavy.\n")
for task in TASKS:
    cells = load_cells(task)
    for style in STYLES:
        lengths = []
        capped = []
        for condition in ["none", "heavy"]:
            cell = cells[(style, condition)]
            lengths.append(sum(len(r["model_answer"].split()) for r in cell.values()) / len(cell))
            capped.append(sum(int(r["hit_cap"]) for r in cell.values()))
        print(
            f"{task:10} {style:6} length {lengths[0]:6.0f} to {lengths[1]:6.0f}"
            f"   hit cap {capped[0]} to {capped[1]}"
        )
