import pytest
import torch

from deletcra.config import ModelConfig
from deletcra.data import synthetic_sequences
from deletcra.experiment import TrainConfig, run_experiment
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, causal_lm_loss


def test_generator_lm_updates_shared_discriminator_embeddings_without_rtd_gradients():
    model = CausalElectra(ModelConfig(vocab_size=8))
    generator = CausalElectra(ModelConfig(vocab_size=8, hidden_size=16, num_layers=1))
    model.share_generator_embeddings(generator)
    assert generator.lm_head.weight is model.electra.embeddings.word_embeddings.weight
    assert (
        generator.electra.embeddings.position_embeddings
        is model.electra.embeddings.position_embeddings
    )
    tokens = synthetic_sequences(2, 6, 8, seed=1)
    causal_lm_loss(generator(tokens).lm_logits, tokens, tokens.ne(0)).backward()
    assert model.electra.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert model.electra.embeddings.position_embeddings.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for parameter in model.electra.encoder.parameters()
    )


def test_shared_training_deduplicates_optimizer_and_saves_matching_embeddings(tmp_path):
    data = synthetic_sequences(8, 6, 8, seed=1)
    report = run_experiment(
        data,
        data,
        ModelConfig(vocab_size=8),
        ObjectiveConfig(mode="rtd"),
        TrainConfig(steps=2, batch_size=4, probe_steps=0, share_embeddings=True),
        tmp_path / "shared",
    )
    model = CausalElectra.load(tmp_path / "shared" / "model")
    generator = CausalElectra.load(tmp_path / "shared" / "generator")
    shared_count = (
        model.electra.embeddings.word_embeddings.weight.numel()
        + model.electra.embeddings.position_embeddings.weight.numel()
    )
    assert (
        report["unique_optimized_parameters"]
        == report["model_parameters"] + report["generator_parameters"] - shared_count
    )
    for name in ("word_embeddings", "position_embeddings"):
        torch.testing.assert_close(
            getattr(model.electra.embeddings, name).weight,
            getattr(generator.electra.embeddings, name).weight,
            rtol=0,
            atol=0,
        )


def test_incompatible_shared_embeddings_fail_before_changing_aliases():
    model = CausalElectra(ModelConfig(vocab_size=8))
    generator = CausalElectra(ModelConfig(vocab_size=9))
    original = generator.lm_head.weight
    with pytest.raises(ValueError, match="vocab_size"):
        model.share_generator_embeddings(generator)
    assert generator.lm_head.weight is original


def test_sharing_without_a_generator_fails_before_creating_output(tmp_path):
    data = synthetic_sequences(8, 6, 8, seed=1)
    with pytest.raises(ValueError, match="generator"):
        run_experiment(
            data,
            data,
            ModelConfig(vocab_size=8),
            ObjectiveConfig(mode="clm"),
            TrainConfig(steps=1, share_embeddings=True),
            tmp_path / "invalid",
        )
    assert not (tmp_path / "invalid").exists()
