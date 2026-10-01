"""Validated, serializable settings for a small ELECTRA backbone."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings independent of the dataset and training loop.

    ``embedding_size`` is the width of token and position embeddings.
    ``hidden_size`` is the width used inside attention and feed-forward blocks.
    ELECTRA projects between them when they differ, allowing a smaller token
    table without shrinking the entire backbone. ``max_positions`` includes
    the initial BOS token and must fit each prepared block.
    """

    vocab_size: int = 32
    embedding_size: int = 16
    hidden_size: int = 32
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 64
    max_positions: int = 64
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "embedding_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "intermediate_size",
            "max_positions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        for name in ("pad_token_id", "bos_token_id"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} must be inside the vocabulary")
        if self.pad_token_id == self.bos_token_id:
            raise ValueError("PAD and BOS must have different token IDs")
        if self.vocab_size < 3 or self.max_positions < 2:
            raise ValueError("need at least one content token and two positions")


def generator_model_config(discriminator: ModelConfig) -> ModelConfig:
    """Build a smaller causal generator with compatible embeddings and IDs.

    Preserve embedding width, vocabulary, positions, and special-token IDs so
    generator and discriminator can share their token/position tables. Reduce
    attention width and feed-forward width to about one quarter and depth to
    about one third. These are this project's experiment defaults, not a claim
    to reproduce the original ELECTRA hyperparameters.
    """
    # Each attention head needs an equal, nonzero slice of the hidden vector.
    # Round down to a multiple of the head count, with at least one value/head.
    heads = discriminator.num_heads
    hidden_size = max(heads, (discriminator.hidden_size // 4 // heads) * heads)
    return ModelConfig(
        vocab_size=discriminator.vocab_size,
        embedding_size=discriminator.embedding_size,
        hidden_size=hidden_size,
        num_layers=max(1, discriminator.num_layers // 3),
        num_heads=heads,
        intermediate_size=max(heads, discriminator.intermediate_size // 4),
        max_positions=discriminator.max_positions,
        dropout=discriminator.dropout,
        pad_token_id=discriminator.pad_token_id,
        bos_token_id=discriminator.bos_token_id,
    )
