"""Common shifted token scoring for Hugging Face and causal ELECTRA models."""

import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from deletcra.model import CausalElectra, validate_batch
from deletcra.objectives import causal_lm_loss


def load_story_candidate(directory: Path, metadata: dict) -> CausalElectra:
    """Load a trained main LM only when its data/tokenizer protocol matches.

    The adjacent training report distinguishes CLM/joint checkpoints from an
    RTD-only model whose main vocabulary projection was never trained. Matching
    vocabulary size alone is not enough: tokenizer revisions and token IDs must
    agree with the validation data before perplexity can be compared.
    """
    report = json.loads((directory.parent / "report.json").read_text(encoding="utf-8"))
    if directory.name not in {"model", "best_model"}:
        raise ValueError("benchmark requires a trained main model checkpoint")
    if report["objective_config"]["mode"] not in {"clm", "joint"}:
        raise ValueError("RTD-only has no trained main LM head")
    for key in ("dataset", "dataset_revision", "tokenizer", "tokenizer_revision"):
        if report["dataset"].get(key) != metadata.get(key):
            raise ValueError(f"candidate data protocol mismatch: {key}")
    # This benchmark evaluates CPU FP32 weights, including flash-trained models.
    model = CausalElectra.load(directory, attention_backend="sdpa").eval()
    if (
        model.config.vocab_size != metadata["vocab_size"]
        or model.config.pad_token_id != metadata["pad_token_id"]
        or model.config.bos_token_id != metadata["bos_token_id"]
        or model.config.max_position_embeddings < metadata["sequence_length"]
    ):
        raise ValueError("candidate configuration does not match validation tokens")
    return model


@torch.no_grad()
def score_language_model(
    model: nn.Module, tokens: Tensor, *, pad_token_id: int, batch_size: int = 4
) -> dict:
    """Average next-token loss over actual targets, then exponentiate for PPL.

    Weight each batch by its nonpadding target count so partial batches and
    padded blocks contribute correctly. Both model types use the same shifted
    targets; Hugging Face's built-in label loss is intentionally not used here.
    """
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
