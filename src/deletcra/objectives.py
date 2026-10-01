"""Shifted causal generation, replacement labels, and three learning objectives."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from deletcra.model import CausalElectra, validate_batch


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "joint"
    replacement_probability: float = 0.15
    temperature: float = 1.0
    generator_weight: float = 1.0
    rtd_weight: float = 5.0
    lm_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"rtd", "clm", "joint"}:
            raise ValueError("mode must be rtd, clm, or joint")
        if not math.isfinite(self.replacement_probability) or not (
            0 <= self.replacement_probability <= 1
        ):
            raise ValueError("replacement_probability must be finite and in [0, 1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for name in ("generator_weight", "rtd_weight", "lm_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mode != "clm" and (self.generator_weight == 0 or self.rtd_weight == 0):
            raise ValueError("RTD needs positive generator and RTD weights")
        if self.mode != "rtd" and self.lm_weight == 0:
            raise ValueError("CLM needs a positive LM weight")


def causal_lm_loss(logits: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Only t's logits can predict t+1; padded pairs contribute nothing."""
    validate_batch(input_ids, attention_mask)
    if logits.shape[:2] != input_ids.shape or logits.ndim != 3:
        raise ValueError("LM logits must have [batch, sequence, vocabulary] shape")
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    if not valid.any():
        raise ValueError("CLM requires at least one valid next-token pair")
    return F.cross_entropy(logits[:, :-1][valid], input_ids[:, 1:][valid])


@dataclass
class Corruption:
    input_ids: Tensor
    labels: Tensor
    selected: Tensor
    eligible: Tensor


@torch.no_grad()
def corrupt_tokens(
    input_ids: Tensor,
    attention_mask: Tensor,
    generator_logits: Tensor,
    *,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    special_token_ids: tuple[int, ...] = (),
    probability: float = 0.15,
    temperature: float = 1.0,
    rng: torch.Generator | None = None,
    selected: Tensor | None = None,
) -> Corruption:
    """Sample x'_t from the generator's x_<t distribution, never from logits[t].

    This is parallel, teacher-forced *clean-prefix* corruption: generator
    prefixes are original tokens, discriminator prefixes are corrupted tokens.
    It is not autoregressive sampling conditioned on previously replaced tokens.
    A sampled token identical to the original always has a negative RTD label.
    """
    validate_batch(input_ids, attention_mask)
    if generator_logits.ndim != 3 or generator_logits.shape[:2] != input_ids.shape:
        raise ValueError("generator logits must match the input batch and sequence")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    vocab_size = generator_logits.shape[-1]
    specials = sorted({pad_token_id, bos_token_id, *special_token_ids})
    if any(token < 0 or token >= vocab_size for token in specials):
        raise ValueError("special token IDs must be inside the vocabulary")
    if len(specials) >= vocab_size:
        raise ValueError("need at least one nonspecial token to sample")
    eligible = attention_mask.bool().clone()
    eligible[:, 0] = False
    for token in specials:
        eligible &= input_ids.ne(token)
    if selected is None:
        selected = torch.rand(input_ids.shape, device=input_ids.device, generator=rng)
        selected = (selected < probability) & eligible
    elif selected.shape != input_ids.shape or selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean tensor matching input_ids")
    elif (selected & ~eligible).any():
        raise ValueError("cannot replace a special or padded position")

    corrupted = input_ids.clone()
    if selected.any():
        shifted = torch.zeros_like(generator_logits)
        shifted[:, 1:] = generator_logits[:, :-1]
        sample_logits = shifted[selected].float().clone()
        sample_logits[:, specials] = -torch.inf
        probs = F.softmax(sample_logits / temperature, dim=-1)
        samples = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        corrupted[selected] = samples
    labels = corrupted.ne(input_ids) & eligible
    return Corruption(corrupted, labels, selected, eligible)


def rtd_loss(logits: Tensor, corruption: Corruption) -> Tensor:
    if logits.shape != corruption.labels.shape:
        raise ValueError("RTD logits must match replacement labels")
    valid = corruption.eligible
    if not valid.any():
        raise ValueError("RTD requires at least one nonspecial content token")
    return F.binary_cross_entropy_with_logits(
        logits[valid], corruption.labels[valid].to(logits.dtype)
    )


@dataclass
class PretrainingOutput:
    loss: Tensor
    generator_loss: Tensor
    rtd_loss: Tensor
    lm_loss: Tensor
    rtd_logits: Tensor | None
    corruption: Corruption | None


def pretraining_step(
    model: CausalElectra,
    generator: CausalElectra | None,
    input_ids: Tensor,
    attention_mask: Tensor,
    settings: ObjectiveConfig,
    *,
    rng: torch.Generator | None = None,
    special_token_ids: tuple[int, ...] = (),
) -> PretrainingOutput:
    validate_batch(input_ids, attention_mask)
    if not input_ids[:, 0].eq(model.config.bos_token_id).all():
        raise ValueError("training sequences must begin with BOS")
    zero = torch.zeros((), device=input_ids.device)
    generator_loss = discriminator_loss = lm_loss = zero
    corruption = None
    rtd_logits = None
    if settings.mode != "clm":
        if generator is None:
            raise ValueError("RTD objectives require a causal generator")
        for key in ("vocab_size", "pad_token_id", "bos_token_id"):
            if getattr(generator.config, key) != getattr(model.config, key):
                raise ValueError("generator and discriminator must share token IDs")
        generator_logits = generator(input_ids, attention_mask).lm_logits
        generator_loss = causal_lm_loss(generator_logits, input_ids, attention_mask)
        corruption = corrupt_tokens(
            input_ids,
            attention_mask,
            generator_logits,
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            special_token_ids=special_token_ids,
            probability=settings.replacement_probability,
            temperature=settings.temperature,
            rng=rng,
        )
        output = model(corruption.input_ids, attention_mask, compute_lm=False)
        rtd_logits = output.rtd_logits
        discriminator_loss = rtd_loss(rtd_logits, corruption)
    if settings.mode != "rtd":
        # CLM uses a separate CLEAN pass: corrupted prefixes would change its task.
        logits = model(input_ids, attention_mask).lm_logits
        lm_loss = causal_lm_loss(logits, input_ids, attention_mask)
    loss = (
        settings.generator_weight * generator_loss
        + settings.rtd_weight * discriminator_loss
        + settings.lm_weight * lm_loss
    )
    return PretrainingOutput(
        loss, generator_loss, discriminator_loss, lm_loss, rtd_logits, corruption
    )
