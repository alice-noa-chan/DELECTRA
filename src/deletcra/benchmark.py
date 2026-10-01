"""Common shifted token scoring for Hugging Face and causal ELECTRA models."""

import math

import torch
from torch import Tensor, nn

from deletcra.model import CausalElectra, validate_batch
from deletcra.objectives import causal_lm_loss


@torch.no_grad()
def score_language_model(
    model: nn.Module, tokens: Tensor, *, pad_token_id: int, batch_size: int = 4
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validate_batch(tokens, tokens.ne(pad_token_id))
    model.eval()
    device = next(model.parameters()).device
    nll_sum = count = 0
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size].to(device)
        mask = batch.ne(pad_token_id)
        if isinstance(model, CausalElectra):
            logits = model(batch, mask).lm_logits
        else:
            logits = model(input_ids=batch, attention_mask=mask, use_cache=False).logits
        loss = causal_lm_loss(logits, batch, mask)
        targets = int((mask[:, 1:] & mask[:, :-1]).sum())
        nll_sum += loss.item() * targets
        count += targets
    nll = nll_sum / count
    return {
        "next_token_targets": count,
        "nll": nll,
        "perplexity": math.exp(nll) if nll < 709 else None,
    }
