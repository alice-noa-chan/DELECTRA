"""Shifted causal generation, replacement labels, and three learning objectives."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from deletcra.losses import fused_causal_lm_loss
from deletcra.model import CausalElectra, validate_batch


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "joint"
    generator_mode: str = "separate"
    lm_loss_backend: str = "torch"
    replacement_probability: float = 0.15
    temperature: float = 1.0
    generator_weight: float = 1.0
    rtd_weight: float = 5.0
    lm_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"rtd", "clm", "joint"}:
            raise ValueError("mode must be rtd, clm, or joint")
        if self.generator_mode not in {"separate", "self"}:
            raise ValueError("generator_mode must be separate or self")
        if self.lm_loss_backend not in {"torch", "liger"}:
            raise ValueError("lm_loss_backend must be torch or liger")
        if self.generator_mode == "self" and self.mode != "joint":
            raise ValueError("self replacement requires joint RTD and CLM")
        if self.generator_mode == "self" and self.generator_weight != 1.0:
            raise ValueError("self replacement has no separate generator loss weight")
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
    """Corrupted batch plus masks needed to train and evaluate the discriminator.

    ``selected`` marks requested replacements; ``labels`` marks actual changes.
    ``eligible`` includes every valid nonspecial position, even unselected ones.
    All three masks have the same [batch, sequence] shape as ``input_ids``.
    """

    input_ids: Tensor
    labels: Tensor
    selected: Tensor
    eligible: Tensor


@torch.no_grad()
def corrupt_tokens(
    input_ids: Tensor,
    attention_mask: Tensor,
    generator_logits: Tensor | None,
    *,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    special_token_ids: tuple[int, ...] = (),
    probability: float = 0.15,
    temperature: float = 1.0,
    rng: torch.Generator | None = None,
    selected: Tensor | None = None,
    generator_features: Tensor | None = None,
    generator_head: nn.Linear | None = None,
) -> Corruption:
    """Sample x'_t from the generator's x_<t distribution, never from logits[t].

    This is parallel, teacher-forced *clean-prefix* corruption: generator
    prefixes are original tokens, discriminator prefixes are corrupted tokens.
    It is not autoregressive sampling conditioned on previously replaced tokens.
    A sampled token identical to the original always has a negative RTD label.
    """
    validate_batch(input_ids, attention_mask)
    if generator_logits is not None:
        if generator_features is not None or generator_head is not None:
            raise ValueError("provide logits or features/head, not both")
        if generator_logits.ndim != 3 or generator_logits.shape[:2] != input_ids.shape:
            raise ValueError("generator logits must match the input batch and sequence")
        vocab_size = generator_logits.shape[-1]
    else:
        if generator_features is None or generator_head is None:
            raise ValueError("replacement sampling requires logits or features/head")
        if (
            generator_features.ndim != 3
            or generator_features.shape[:2] != input_ids.shape
        ):
            raise ValueError(
                "generator features must match the input batch and sequence"
            )
        vocab_size = generator_head.out_features
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
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
        # Logits at t have already seen x_t, so using them to replace x_t would
        # leak the answer. Logits at t-1 only saw the original prefix x_<t.
        # Target t selects source t-1 directly. Avoid a full [B,L,V] shifted copy.
        # Fused-loss callers materialize logits only at selected proposal sites.
        sites = selected[:, 1:]
        if generator_logits is not None:
            sample_logits = generator_logits[:, :-1][sites].float()
        else:
            sample_logits = generator_head(generator_features[:, :-1][sites]).float()
        sample_logits[:, specials] = -torch.inf
        probs = F.softmax(sample_logits / temperature, dim=-1)
        samples = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
        corrupted[selected] = samples
    # Selection alone does not make a token fake. The generator can sample the
    # original token again; those positions must still be labeled original.
    labels = corrupted.ne(input_ids) & eligible
    return Corruption(corrupted, labels, selected, eligible)


def rtd_loss(logits: Tensor, corruption: Corruption) -> Tensor:
    """Detect actual changes at every eligible position, not just selected ones.

    A label of 1 means replaced and 0 means original. Padding and special tokens
    do not contribute. Unchanged content is needed to learn the original class.
    """
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
    backward_clean: bool = False,
    clean_backward: Callable[[Tensor], None] | None = None,
) -> PretrainingOutput:
    """Compute losses without updating parameters; optionally backpropagate clean CLM.

    RTD/joint first train the generator on clean next-token targets, sample a
    corrupted batch without gradients, and classify that batch with the main
    model. CLM/joint also run the main model on a separate clean batch.
    By default the training loop owns backward. In self mode, backward_clean=True
    frees the clean graph before RTD; the caller then backpropagates the returned
    loss (whose clean term is detached), clips accumulated gradients and updates
    once. This execution option changes graph lifetime, not objective weights.
    """
    validate_batch(input_ids, attention_mask)
    if backward_clean and (
        settings.generator_mode != "self" or not torch.is_grad_enabled()
    ):
        raise ValueError(
            "sequential clean backward requires self training with gradients"
        )
    if not input_ids[:, 0].eq(model.config.bos_token_id).all():
        raise ValueError("training sequences must begin with BOS")
    zero = torch.zeros((), device=input_ids.device)
    generator_loss = discriminator_loss = lm_loss = zero
    corruption = None
    rtd_logits = None
    if settings.generator_mode == "self":
        if generator is not None:
            raise ValueError("self replacement uses only the main model")
        # One clean pass learns next tokens AND supplies replacement proposals.
        # Sampling at t uses clean logits[t-1], detached inside corrupt_tokens.
        # A second pass is essential: RTD must see the actually corrupted prefix.
        clean = model(
            input_ids,
            attention_mask,
            compute_lm=settings.lm_loss_backend == "torch",
            compute_rtd=False,
        )
        features = None
        if settings.lm_loss_backend == "liger":
            features = model.lm_projection(clean.hidden_states)
            lm_loss = fused_causal_lm_loss(
                model.lm_head, features, input_ids, attention_mask
            )
        else:
            lm_loss = causal_lm_loss(clean.lm_logits, input_ids, attention_mask)
        corruption = corrupt_tokens(
            input_ids,
            attention_mask,
            clean.lm_logits,
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            special_token_ids=special_token_ids,
            probability=settings.replacement_probability,
            temperature=settings.temperature,
            rng=rng,
            generator_features=features,
            generator_head=model.lm_head if features is not None else None,
        )
        if backward_clean:
            if not torch.isfinite(lm_loss):
                raise RuntimeError("nonfinite clean CLM loss")
            weighted = settings.lm_weight * lm_loss
            if clean_backward is None:
                weighted.backward()
            else:
                clean_backward(weighted)
            # The caller backpropagates returned loss to accumulate RTD gradients,
            # then clips/updates ONCE. Detaching prevents a second CLM backward.
            lm_loss = lm_loss.detach()
        del clean, features
        rtd_logits = model(
            corruption.input_ids, attention_mask, compute_lm=False
        ).rtd_logits
        discriminator_loss = rtd_loss(rtd_logits, corruption)
        # Do not count the same CLM supervision a second time as generator loss.
        loss = settings.lm_weight * lm_loss + settings.rtd_weight * discriminator_loss
        return PretrainingOutput(
            loss, zero, discriminator_loss, lm_loss, rtd_logits, corruption
        )
    if settings.mode != "clm":
        if generator is None:
            raise ValueError("RTD objectives require a causal generator")
        for key in ("vocab_size", "pad_token_id", "bos_token_id"):
            if getattr(generator.config, key) != getattr(model.config, key):
                raise ValueError("generator and discriminator must share token IDs")
        generated = generator(
            input_ids,
            attention_mask,
            compute_lm=settings.lm_loss_backend == "torch",
            compute_rtd=False,
        )
        generator_logits = generated.lm_logits
        features = None
        # The generator learns through this CLM loss. RTD loss cannot train it
        # through discrete sampling; shared embeddings are the separate path
        # by which both objectives can update the same embedding parameters.
        if settings.lm_loss_backend == "liger":
            features = generator.lm_projection(generated.hidden_states)
            generator_loss = fused_causal_lm_loss(
                generator.lm_head, features, input_ids, attention_mask
            )
        else:
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
            generator_features=features,
            generator_head=generator.lm_head if features is not None else None,
        )
        del generated, generator_logits, features
        output = model(corruption.input_ids, attention_mask, compute_lm=False)
        rtd_logits = output.rtd_logits
        discriminator_loss = rtd_loss(rtd_logits, corruption)
    if settings.mode != "rtd":
        # CLM uses a separate CLEAN pass: corrupted prefixes would change its task.
        clean = model(
            input_ids,
            attention_mask,
            compute_lm=settings.lm_loss_backend == "torch",
            compute_rtd=False,
        )
        if settings.lm_loss_backend == "liger":
            features = model.lm_projection(clean.hidden_states)
            lm_loss = fused_causal_lm_loss(
                model.lm_head, features, input_ids, attention_mask
            )
        else:
            lm_loss = causal_lm_loss(clean.lm_logits, input_ids, attention_mask)
    loss = (
        settings.generator_weight * generator_loss
        + settings.rtd_weight * discriminator_loss
        + settings.lm_weight * lm_loss
    )
    return PretrainingOutput(
        loss, generator_loss, discriminator_loss, lm_loss, rtd_logits, corruption
    )
