import pytest
import torch

from deletcra import ModelConfig
from deletcra.config import generator_model_config
from deletcra.model import CausalElectra


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
        {"attention_backend": "unknown"},
    ],
)
def test_rejects_invalid_model_settings(settings):
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_embedding_projection_can_differ_from_hidden_size():
    config = ModelConfig(embedding_size=8, hidden_size=32)
    assert config.embedding_size < config.hidden_size


def test_small_generator_can_share_embeddings_and_predict_with_narrow_attention():
    # A valid backbone can be too narrow to divide its hidden width by four.
    # The smaller generator must still support attention, projection and sharing.
    config = ModelConfig(hidden_size=4, num_heads=4, num_layers=1)
    main = CausalElectra(config)
    generator = CausalElectra(generator_model_config(config))
    main.share_generator_embeddings(generator)
    output = generator(torch.tensor([[1, 3, 4]]))
    assert output.lm_logits.shape == (1, 3, config.vocab_size)
    assert torch.isfinite(output.lm_logits).all()
    assert generator.lm_head.weight is main.electra.embeddings.word_embeddings.weight
