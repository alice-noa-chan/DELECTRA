"""Pinned English TinyStories target and a parameter-matched causal ELECTRA."""

from deletcra.config import ModelConfig

REFERENCE_MODEL = "nickypro/tinyllama-15M"
REFERENCE_REVISION = "a97e01fa7d54c088df9dc1b53d581a863ce08cd6"
STORIES_DATASET = "roneneldan/TinyStories"
STORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def story_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        embedding_size=288,
        hidden_size=288,
        num_layers=6,
        num_heads=6,
        intermediate_size=1024,
        max_positions=256,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
    )
