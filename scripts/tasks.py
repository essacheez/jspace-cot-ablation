"""
Benchmark tasks, prompting and scoring for GSM8K, MATH-500 and AIME 2024.

Difficulty is the length of the gold solution, split into N_BINS equal bins per task.
"""

import csv
import json
import random

from datasets import load_dataset
from tqdm.auto import tqdm

from ablate import generate, sub_bands
from extractor import extract_aime2024, extract_gsm8k, extract_math500

BATCH_SIZE = 16
N_BINS = 3
SEED = 0

# a batch runs until its slowest sequence stops, so caps follow measured answer length
CAPS = {
    "gsm8k": {"direct": 513, "cot": 1023},
    "math500": {"direct": 4096, "cot": 4096},
    "aime2024": {"direct": 8192, "cot": 8192},
}

# sampled at random under SEED rather than taken from the front of the split
N_PROBLEMS = {"gsm8k": 90, "math500": 90, "aime2024": 9}

DIRECT_PROMPT = """Solve the math problem below. Output only the final answer with no explanation and no reasoning.

Question: {question}
Answer:"""

COT_PROMPT = """Solve the math problem below. Show your thinking and reasoning step by step, then
state the final answer on its own line in the form "#### <final answer>".

Question: {question}
Answer:"""

STYLES = {"direct": DIRECT_PROMPT, "cot": COT_PROMPT}
CONDITIONS = ["none", "light", "medium", "heavy"]

TASKS = {
    "gsm8k": {
        "load": ("openai/gsm8k", "main", "test"),
        "question": "question",
        "answer": "answer",
        "solution": "answer",
        "extract": extract_gsm8k,
    },
    "math500": {
        "load": ("HuggingFaceH4/MATH-500", None, "test"),
        "question": "problem",
        "answer": "answer",
        "solution": "solution",
        "extract": extract_math500,
    },
    "aime2024": {
        "load": ("HuggingFaceH4/aime_2024", None, "train"),
        "question": "problem",
        "answer": "answer",
        "solution": "solution",
        "extract": extract_aime2024,
    },
}


def load_task(name, limit=None):
    """Sample one task's problems, each carrying a gold answer and a difficulty bin."""
    spec = TASKS[name]
    path, config, split = spec["load"]
    if config:
        data = load_dataset(path, config, split=split)
    else:
        data = load_dataset(path, split=split)

    rows = list(data)
    extract = spec["extract"]

    missed = [
        row
        for row in rows
        if extract(str(row[spec["solution"]])) != extract(str(row[spec["answer"]]))
    ]
    if missed:
        raise AssertionError(f"{name}: {len(missed)}/{len(rows)} golds do not round-trip")

    count = limit or N_PROBLEMS.get(name, len(rows))
    picked = random.Random(SEED).sample(range(len(rows)), min(count, len(rows)))
    rows = [rows[index] for index in sorted(picked)]

    lengths = [len(str(row[spec["solution"]])) for row in rows]
    order = sorted(range(len(rows)), key=lambda index: lengths[index])
    rank_of = {index: rank for rank, index in enumerate(order)}

    return [
        {
            "question": row[spec["question"]],
            "gold": extract(str(row[spec["answer"]])),
            "raw_gold": str(row[spec["answer"]]),
            "difficulty": rank_of[index] * N_BINS // len(rows) + 1,
        }
        for index, row in enumerate(rows)
    ]


def build_prompt(tokenizer, question, style):
    """Wrap a question in the chat template, thinking mode off."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": STYLES[style].format(question=question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_condition(model, tokenizer, items, task, style, condition, band, k, parts):
    """Generate and score one prompt style under one ablation condition."""
    mode, layers = "none", None
    if condition != "none":
        mode, layers = "lens", sub_bands(band)[condition]

    extract = TASKS[task]["extract"]
    cap = CAPS[task][style]

    records = []
    starts = range(0, len(items), BATCH_SIZE)
    for start in tqdm(starts, desc=f"{task} {style}_{condition}", leave=False):
        batch = items[start : start + BATCH_SIZE]
        prompts = [build_prompt(tokenizer, item["question"], style) for item in batch]
        completions, stopped = generate(
            model, tokenizer, prompts, cap, mode=mode, layers=layers, k=k, parts=parts
        )

        for item, completion, done in zip(batch, completions, stopped):
            extracted = extract(completion)
            records.append({
                "question": item["question"],
                "model_answer": completion,
                "gold": item["gold"],
                "raw_gold": item["raw_gold"],
                "extracted_answer": extracted,
                "difficulty": item["difficulty"],
                "correct": int(extracted is not None and extracted == item["gold"]),
                "hit_cap": int(not done),
            })
    return records


def run_matrix(model, tokenizer, items, task, band, k, parts, cache_dir=None,
               conditions=CONDITIONS):
    """Run every prompt style against each ablation condition, resuming if possible."""
    results = {}
    for style in STYLES:
        for condition in conditions:
            name = f"{style}_{condition}"
            cached = cache_dir / f"{task}_{name}.json" if cache_dir else None
            if cached and cached.exists():
                results[name] = json.loads(cached.read_text())
                print(f"  {task} {name} acc={accuracy(results[name]):.4f} (cached)")
                continue

            records = run_condition(
                model, tokenizer, items, task, style, condition, band, k, parts
            )
            results[name] = records
            if cached:
                cached.write_text(json.dumps(records))

            capped = sum(record["hit_cap"] for record in records)
            print(f"  {task} {name} acc={accuracy(records):.4f} "
                  f"hit_cap={capped}/{len(records)}")
    return results


def accuracy(records):
    """Exact-match accuracy over a list of records."""
    return sum(record["correct"] for record in records) / len(records)


def accuracy_by_difficulty(records):
    """Accuracy in each difficulty bin, easiest first."""
    scores = []
    for level in range(1, N_BINS + 1):
        binned = [record for record in records if record["difficulty"] == level]
        if binned:
            scores.append(sum(record["correct"] for record in binned) / len(binned))
        else:
            scores.append(0.0)
    return scores


def summarize(results):
    """Accuracy, per-bin accuracy and cap hits for every cell of every task."""
    summary = {"accuracy": {}, "accuracy_by_difficulty": {}, "hit_cap": {}}
    for task, cells in results.items():
        summary["accuracy"][task] = {
            name: accuracy(records) for name, records in cells.items()
        }
        summary["accuracy_by_difficulty"][task] = {
            name: accuracy_by_difficulty(records) for name, records in cells.items()
        }
        summary["hit_cap"][task] = {
            name: sum(record["hit_cap"] for record in records)
            for name, records in cells.items()
        }
    return summary


def save_csv(path, results):
    """Write every record of one task, tagged with its style and condition."""
    columns = ["style", "condition", "question", "model_answer", "gold", "raw_gold",
               "extracted_answer", "difficulty", "correct", "hit_cap"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, records in results.items():
            style, condition = name.split("_")
            for record in records:
                writer.writerow({"style": style, "condition": condition, **record})
    return path
