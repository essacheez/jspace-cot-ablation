"""
Ablate J-space: delete each position's top-k lens directions across a layer band.

Tokens already in the clean model's top-10 are exempt, so the ablation removes internal
content rather than the output the model was about to emit. That needs the unablated
prediction at every step, so ablated generation shadows a clean pass with its own cache.
"""

import torch

from lens import direction_rows, inner_products, remove_span

N_PROTECTED = 10
CHUNK = 512


def sub_bands(band):
    """Light, medium and heavy layer ranges inside a measured band."""
    first, last = band
    width = last - first + 1
    return {
        "light": list(range(first, first + max(2, width // 4))),
        "medium": list(range(first + width // 4, last - width // 4 + 1)),
        "heavy": list(range(first, last + 1)),
    }


def ablate_lens(residual, jacobian, unembed, gain, protected, k):
    """Remove each position's top-k unprotected lens directions."""
    pieces = []
    for start in range(0, residual.shape[0], CHUNK):
        chunk = residual[start : start + CHUNK]
        scores = inner_products(chunk, jacobian, unembed, gain)
        if protected is not None:
            scores.scatter_(1, protected[start : start + CHUNK], float("-inf"))

        picked = scores.topk(k, dim=1).indices
        directions = direction_rows(picked.reshape(-1), jacobian, unembed, gain)
        pieces.append(remove_span(chunk, directions.reshape(chunk.shape[0], k, -1)))
    return torch.cat(pieces)


def make_hook(layer, parts, k, state):
    """Forward hook that ablates one layer's residual stream."""
    jacobians, unembed, gain = parts

    def hook(module, inputs, output):
        if not state["on"]:
            return None

        is_tuple = not torch.is_tensor(output)
        hidden = output[0] if is_tuple else output
        flat = hidden.reshape(-1, hidden.shape[-1]).float()

        protected = state["protected"]
        if protected is not None:
            protected = protected.reshape(flat.shape[0], -1)

        ablated = ablate_lens(flat, jacobians[layer], unembed, gain, protected, k)
        new_hidden = ablated.reshape(hidden.shape).to(hidden.dtype)
        if is_tuple:
            return (new_hidden,) + tuple(output[1:])
        return new_hidden

    return hook


def install(model, layers, k, state, parts):
    """Attach ablation hooks to the given layers and return their handles."""
    handles = []
    for layer in layers:
        hook = make_hook(layer, parts, k, state)
        handles.append(model.model.layers[layer].register_forward_hook(hook))
    return handles


def refresh_protected(model, step, mask, cache, state):
    """Run the clean pass that decides which tokens the ablation must leave alone."""
    state["on"] = False
    with torch.no_grad():
        clean = model(step, attention_mask=mask, past_key_values=cache, use_cache=True)
    state["on"] = True
    state["protected"] = clean.logits.topk(N_PROTECTED, dim=-1).indices
    return clean.past_key_values


def generate(model, tokenizer, prompts, max_new_tokens, mode="none", layers=None, k=None, parts=None):
    """Greedy-decode the prompts, clean or ablated, returning completions and flags."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    ids = encoded.input_ids.cuda()
    mask = encoded.attention_mask.cuda()

    state = {"on": True, "protected": None}
    ablated = mode == "lens"
    handles = install(model, layers, k, state, parts) if ablated else []

    stop = model.generation_config.eos_token_id
    stop = torch.tensor(stop if isinstance(stop, list) else [stop], device=ids.device)
    done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    tokens = [[] for _ in prompts]

    try:
        clean_cache = None

        if ablated:
            clean_cache = refresh_protected(model, ids, mask, None, state)

        with torch.no_grad():
            out = model(ids, attention_mask=mask, use_cache=True)

        cache = out.past_key_values
        next_tokens = out.logits[:, -1].argmax(dim=-1)

        for _ in range(max_new_tokens):
            for row in range(len(prompts)):
                if not done[row]:
                    tokens[row].append(int(next_tokens[row]))

            done = done | torch.isin(next_tokens, stop)

            if bool(done.all()):
                break

            step = next_tokens.unsqueeze(1)
            mask = torch.cat([mask, torch.ones_like(step)], dim=1)

            if ablated:
                clean_cache = refresh_protected(model, step, mask, clean_cache, state)

            with torch.no_grad():
                out = model(step, attention_mask=mask, past_key_values=cache, use_cache=True)

            cache = out.past_key_values
            next_tokens = out.logits[:, -1].argmax(dim=-1)

    finally:
        for handle in handles:
            handle.remove()

    completions = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in tokens]
    
    return completions, done.tolist()
