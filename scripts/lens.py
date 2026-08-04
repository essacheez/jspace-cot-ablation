"""
Load Qwen3-4B with its published Jacobian lens and derive J-space directions.

A token's lens direction at layer L is (W_U[t] * gain) @ J_L, which lives in layer-L
residual space, so the lens readout for that token is an inner product against it.
"""

import torch
import transformers
from jlens import JacobianLens

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"

# layer L means the output of block L; the lens covers layers 0 to 34 of 36
LENS_LAYERS = list(range(35))


def load_model():
    """Load Qwen3-4B in bf16 on the GPU, with its tokenizer."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16
    ).cuda()
    return model.eval(), tokenizer


def load_lens():
    """Load the published Jacobian lens for this exact checkpoint."""
    return JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)


def lens_parts(model, lens, layers):
    """Put the per-layer Jacobians, unembedding and final-norm gain on the GPU."""
    jacobians = {layer: lens.jacobians[layer].cuda() for layer in layers}
    unembed = model.lm_head.weight.detach().float()
    gain = model.model.norm.weight.detach().float()
    return jacobians, unembed, gain


def inner_products(residual, jacobian, unembed, gain):
    """Lens inner products of a residual against every token's direction."""
    return ((residual.float() @ jacobian.T) * gain) @ unembed.T


def direction_rows(token_ids, jacobian, unembed, gain):
    """Lens directions for the given token ids."""
    return (unembed[token_ids] * gain) @ jacobian


def remove_span(residuals, directions):
    """Project the span of the given directions out of each residual at once."""
    basis, _ = torch.linalg.qr(directions.transpose(1, 2))
    weights = torch.einsum("ndk,nd->nk", basis, residuals)
    return residuals - torch.einsum("nk,ndk->nd", weights, basis)
