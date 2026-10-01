import pytest

from deletcra import ModelConfig


@pytest.mark.parametrize(
    "settings",
    [
        {"hidden_size": 31},
        {"num_heads": 0},
        {"num_layers": -1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"pad_token_id": 32},
        {"bos_token_id": 0},
        {"vocab_size": 2},
        {"max_positions": 1},
    ],
)
def test_rejects_invalid_model_settings(settings):
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_embedding_projection_can_differ_from_hidden_size():
    config = ModelConfig(embedding_size=8, hidden_size=32)
    assert config.embedding_size < config.hidden_size
