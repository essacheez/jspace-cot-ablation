"""
Measure the workspace layer band from the model's own per-layer lens statistics.

Offset is where next-token agreement jumps steepest, which is where the output layers
take over. Onset is where excess kurtosis stops falling and starts rising, which is
where content begins to concentrate. The band is [onset, offset - 1].
"""

import torch
from datasets import load_dataset
from jlens import ActivationRecorder

from lens import LENS_LAYERS

MAX_LEN = 128
SKIP_FIRST = 16


def wikitext_prompts(count):
    """Stream the given number of wikitext-103 passages to measure on."""
    data = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts = []
    for record in data:
        if len(record["text"].strip()) >= 600:
            prompts.append(record["text"])
            if len(prompts) == count:
                return prompts
    return prompts


def signatures(model, tokenizer, lens, prompts):
    """Per-layer excess kurtosis and top-10 next-token agreement."""
    kurtosis = {layer: 0.0 for layer in LENS_LAYERS}
    agreement = {layer: 0 for layer in LENS_LAYERS}
    positions = 0

    for prompt in prompts:
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN
        )
        ids = encoded.input_ids.cuda()
        if ids.shape[1] <= SKIP_FIRST + 1:
            continue
        keep = slice(SKIP_FIRST, ids.shape[1] - 1)

        with torch.no_grad():
            with ActivationRecorder(model.model.layers, at=LENS_LAYERS) as recorder:
                model.model(input_ids=ids, use_cache=False)
            residuals = {
                layer: recorder.activations[layer][0].detach() for layer in LENS_LAYERS
            }
            top1 = model(ids).logits[0, keep].argmax(dim=-1)
        positions += top1.shape[0]

        for layer in LENS_LAYERS:
            with torch.no_grad():
                transported = lens.transport(residuals[layer][keep].float(), layer)
                normed = model.model.norm(transported.to(model.dtype))
                logits = model.lm_head(normed).float()

            centered = logits - logits.mean(dim=-1, keepdim=True)
            variance = centered.pow(2).mean(dim=-1)
            excess = centered.pow(4).mean(dim=-1) / variance.pow(2) - 3.0
            kurtosis[layer] += excess.sum().item()

            hits = (logits.topk(10, dim=-1).indices == top1.unsqueeze(1)).any(dim=1)
            agreement[layer] += int(hits.sum().item())

    kurtosis = {layer: kurtosis[layer] / positions for layer in LENS_LAYERS}
    agreement = {layer: agreement[layer] / positions for layer in LENS_LAYERS}
    return kurtosis, agreement


def find_band(kurtosis, agreement):
    """Return ([onset, offset - 1], onset, offset) from the two signatures."""
    upper = LENS_LAYERS[len(LENS_LAYERS) // 2 :]
    offset = max(upper, key=lambda layer: agreement[layer] - agreement[layer - 1])

    below = [layer for layer in LENS_LAYERS if layer < offset]
    onset = min(below, key=lambda layer: kurtosis[layer])
    return [onset, offset - 1], onset, offset
