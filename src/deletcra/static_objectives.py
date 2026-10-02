"""Fixed-shape ELECTRA objectives for XLA, with the same token alignment.

Categorical proposals use inverse CDF and host-generated uniforms. This avoids
variable-sized selected-token tensors and keeps the proposal RNG checkpointable
without exposing a CUDA generator to XLA. It changes random draws, not the
categorical distribution or the detached generator-to-discriminator boundary.
"""

import torch
from torch.nn import functional as F

from deletcra.objectives import Corruption, PretrainingOutput


def masked_lm_loss(logits, targets, valid):
    """Fixed [B,L,V] CE; invalid targets contribute zero, including padded rows."""
    labels = targets.masked_fill(~valid, -100)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    return losses.sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def static_corruption(tokens, mask, logits, settings, specials, rng):
    eligible = mask.bool().clone()
    eligible[:, 0] = False
    for token in specials:
        eligible = eligible & tokens.ne(token)
    uniforms = torch.rand(tokens.shape, generator=rng).to(tokens.device)
    selected = (uniforms < settings.replacement_probability) & eligible
    probabilities = logits[:, :-1].float() / settings.temperature
    probabilities = probabilities.clone()
    probabilities[:, :, list(specials)] = -torch.inf
    cdf = probabilities.softmax(-1).cumsum(-1)
    cdf = cdf / cdf[:, :, -1:]
    draws = torch.rand(tokens.shape[0], tokens.shape[1] - 1, generator=rng)
    draws = draws.to(tokens.device).unsqueeze(-1)
    # Counting CDF entries below u is a categorical sample. Clamp protects the
    # final bin from small floating-point cumulative-sum errors near one.
    proposals = (cdf <= draws).sum(-1).clamp_max(logits.shape[-1] - 1)
    proposals = torch.cat((tokens[:, :1], proposals), dim=1)
    corrupted = torch.where(selected, proposals, tokens)
    labels = corrupted.ne(tokens) & eligible
    return Corruption(corrupted, labels, selected, eligible)


def static_pretraining_step(
    model,
    generator,
    tokens,
    mask,
    settings,
    *,
    rng,
    special_token_ids,
    backward_clean=False,
    clean_backward=None,
):
    """Keep generator CLM, main RTD and optional main clean CLM gradients."""
    if settings.lm_loss_backend != "torch":
        raise ValueError("static objectives require the torch loss backend")
    valid = mask[:, :-1].bool() & mask[:, 1:].bool()
    zero = torch.zeros((), device=tokens.device)
    generator_loss = lm_loss = discriminator_loss = zero
    corruption = rtd_logits = None
    if settings.generator_mode == "self":
        if generator is not None:
            raise ValueError("self replacement uses only the main model")
        clean = model(tokens, mask, compute_rtd=False)
        lm_loss = masked_lm_loss(clean.lm_logits[:, :-1], tokens[:, 1:], valid)
        corruption = static_corruption(
            tokens,
            mask,
            clean.lm_logits,
            settings,
            sorted(
                {
                    model.config.pad_token_id,
                    model.config.bos_token_id,
                    *special_token_ids,
                }
            ),
            rng,
        )
        if backward_clean:
            clean_backward(settings.lm_weight * lm_loss)
            lm_loss = lm_loss.detach()
        del clean
    elif settings.mode != "clm":
        if generator is None:
            raise ValueError("RTD requires a learned generator")
        generated = generator(tokens, mask, compute_rtd=False)
        generator_loss = masked_lm_loss(
            generated.lm_logits[:, :-1], tokens[:, 1:], valid
        )
        corruption = static_corruption(
            tokens,
            mask,
            generated.lm_logits,
            settings,
            sorted(
                {
                    model.config.pad_token_id,
                    model.config.bos_token_id,
                    *special_token_ids,
                }
            ),
            rng,
        )
        del generated
    if settings.mode != "clm":
        rtd_logits = model(corruption.input_ids, mask, compute_lm=False).rtd_logits
        losses = F.binary_cross_entropy_with_logits(
            rtd_logits.float(), corruption.labels.float(), reduction="none"
        )
        discriminator_loss = (
            losses * corruption.eligible
        ).sum() / corruption.eligible.sum().clamp_min(1)
    if settings.mode != "rtd" and settings.generator_mode != "self":
        clean = model(tokens, mask, compute_rtd=False)
        lm_loss = masked_lm_loss(clean.lm_logits[:, :-1], tokens[:, 1:], valid)
    loss = (
        settings.generator_weight * generator_loss
        + settings.lm_weight * lm_loss
        + settings.rtd_weight * discriminator_loss
    )
    return PretrainingOutput(
        loss, generator_loss, discriminator_loss, lm_loss, rtd_logits, corruption
    )
