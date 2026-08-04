# Jspace-COT-ablation

Does chain of thought keep protecting a model against J-space ablation as problems get
harder?

This repository ablates the J-space of Qwen3-4B, the set of representations a Jacobian lens
shows the model is poised to verbalize, and measures what that does to accuracy on GSM8K,
MATH-500 and AIME 2024, with and without chain of thought. It extends the ablation result
reported in [Verbalizable Representations Form a Global Workspace in Language
Models](https://transformer-circuits.pub/2026/workspace/index.html) to an open 4B model and
across a wider difficulty range.

## Repository structure

```
scripts/               all code, nine files
  lens.py              model and lens loading, lens direction math, span removal
  band.py              measures the workspace layer band from the model itself
  ablate.py            the ablation hooks, output protection, generation loop
  tasks.py             datasets, prompts, the condition matrix, scoring
  extractor.py         one answer extractor per dataset
  plot.py              the per task figures written by the runs
  run_ablation.py      entry point, runs the full matrix
  run_baseline.py      entry point, runs the unablated condition only
  figures.py           regenerates the two summary figures above
  stats.py             recomputes the accuracy and significance tables
results/               every result, all committed
  ablation_results.json    accuracy, per bucket accuracy and cap hits in every condition
  baseline_results.json    the same for the unablated condition
  band.json                the measured band, and the per layer curves behind it
  ablation_k.json          the sweep that selected k = 5
  gsm8k_records.csv        720 rows, every item in all 8 conditions
  math500_records.csv      720 rows
  aime2024_records.csv     72 rows
  fig1_dose_response.png   accuracy against ablation strength, one panel per dataset
  fig2_difficulty.png      accuracy by difficulty bucket, unablated against ablated
  *_accuracy.png           per dataset accuracy bars
  *_difficulty.png         per dataset accuracy by difficulty bucket
  average_accuracy.png     mean across the three datasets
  baseline_accuracy.png    unablated accuracy across the three datasets
```

The CSV columns are `style`, `condition`, `question`, `model_answer`, `gold`, `raw_gold`,
`extracted_answer`, `difficulty`, `correct` and `hit_cap`. `condition` is one of `none`,
`light`, `medium` or `heavy`, `difficulty` is 1 for the least difficult third of gold
solutions and 3 for the most difficult, and `hit_cap` marks a completion that reached its
token limit instead of stopping on its own. Every number can be recomputed from these
three files, `stats.py` does that.

## Setup

Requires Python 3.10 or newer and one GPU with at least 24 GB of memory. The runs here used
a single H100. 

```bash
git clone https://github.com/essacheez/Jspace-COT-ablation.git
cd Jspace-COT-ablation
uv sync
```

## Reproducing the results

Regenerating the tables and figures from the committed records needs no GPU and takes
seconds:

```bash
cd scripts
python3 stats.py      # prints the accuracy table and the significance table
python3 figures.py    # rewrites results/fig1_dose_response.png and fig2_difficulty.png
```

`stats.py` reads the three CSVs and prints accuracy in every condition with the heavy to
unablated ratio, then the paired comparison of heavy against unablated with Wilson 95 percent
intervals, the items whose correctness changed, a two sided exact binomial test over those
items, and a rescore that drops items truncated at the token limit on either side. It also
prints mean completion length and truncation counts per condition. `figures.py` reads
`results/ablation_results.json` and rewrites the two figures shown above.

Rerunning the experiment itself needs the GPU:

```bash
cd scripts
python3 run_ablation.py            # the full matrix, about 5 hours
python3 run_baseline.py            # the unablated condition only
python3 run_ablation.py --limit 8  # a short smoke test first
```

`run_ablation.py` measures the band once and caches it in `results/band.json`, then runs
every prompt style against every ablation strength for each dataset. Each condition is
cached under `results/cells/` as it finishes, so an interrupted run resumes where it
stopped. Delete a cell file to force that condition to run again. The script writes the
per item CSVs, the summary JSON and the per dataset figures, then prints the heavy to
unablated ratios.

Results are deterministic given the same weights, since decoding is greedy and the sample is
drawn under seed 0. The one thing that will not reproduce is timing.

## Experimental setup in brief

| | |
|---|---|
| model | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), revision `1cfa9a7208912126459214e8b04321603b3df60c`, bfloat16 |
| lens | [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens), `qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt` |
| band | layers 20 to 31, measured from this model on 20 wikitext-103 passages |
| strengths | none, light L20 to L22, medium L23 to L28, heavy L20 to L31 |
| k | 5 lens directions removed per token position |
| decoding | greedy, seed 0, the model's own reasoning mode disabled |

Datasets, all sampled at random under seed 0:

| dataset | source | split | sampled |
|---|---|---|---|
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), config `main` | test | 90 of 1319 |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | test | 90 of 500 |
| AIME 2024 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | train, the 30 problems of AIME 2024 I and II | 9 of 30 |


## References

- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html), the paper this extends
- [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), the companion code that fits and applies the lens
- Kordi, Nayak, Zuo, Nguyen and Bach (2026), [Revisiting Generalization Across Difficulty Levels: It's Not So Easy](https://aclanthology.org/2026.eacl-long.330), on why difficulty proxies matter
