"""Validated, serializable settings for a small ELECTRA backbone."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
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
