import pytest
import torch
from torch.nn import functional as F

from deletcra import ModelConfig
from deletcra.model import CausalElectra
from deletcra.objectives import (
    ObjectiveConfig,
    causal_lm_loss,
    corrupt_tokens,
    pretraining_step,
    rtd_loss,
)


def test_lm_loss_is_shifted_and_padding_is_ignored():
    ids = torch.tensor([[1, 3, 4, 0]])
    mask = ids.ne(0)
    logits = torch.zeros(1, 4, 8, requires_grad=True)
    expected = F.cross_entropy(logits[0, :2], torch.tensor([3, 4]))
    loss = causal_lm_loss(logits, ids, mask)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, :2].abs().sum() > 0
    assert logits.grad[0, 2:].abs().sum() == 0


def test_replacements_use_previous_logits_and_matches_are_not_positive_labels():
    ids = torch.tensor([[1, 3, 4, 0]])
    logits = torch.full((1, 4, 8), -torch.inf)
    logits[0, 0, 3] = 0  # position 1: unchanged, despite being selected
    logits[0, 1, 5] = 0  # position 2: actually replaced
    logits[0, 2:, 6] = 0
    chosen = torch.tensor([[False, True, True, False]])
    corruption = corrupt_tokens(ids, ids.ne(0), logits, selected=chosen)
    assert corruption.input_ids.tolist() == [[1, 3, 5, 0]]
    assert corruption.labels.tolist() == [[False, False, True, False]]
    assert corruption.eligible.tolist() == [[False, True, True, False]]


def test_corruption_does_not_use_current_or_future_generator_logits():
    ids = torch.tensor([[1, 3, 4, 5, 6]])
    logits = torch.randn(1, 5, 8)
    altered = logits.clone()
    altered[:, 2:] = torch.randn_like(altered[:, 2:]) * 100
    chosen = torch.tensor([[False, True, True, False, False]])
    first = corrupt_tokens(
        ids, ids.ne(0), logits, selected=chosen, rng=torch.Generator().manual_seed(4)
    )
    second = corrupt_tokens(
        ids, ids.ne(0), altered, selected=chosen, rng=torch.Generator().manual_seed(4)
    )
    assert torch.equal(first.input_ids, second.input_ids)


def test_special_tokens_cannot_be_replaced_or_sampled():
    ids = torch.tensor([[1, 3, 2, 4, 0]])
    logits = torch.zeros(1, 5, 8)
    logits[:, :, :3] = 100
    corruption = corrupt_tokens(
        ids, ids.ne(0), logits, special_token_ids=(2,), probability=1
    )
    assert corruption.selected.tolist() == [[False, True, False, True, False]]
    assert (corruption.input_ids[corruption.selected] >= 3).all()


def test_rtd_loss_excludes_padding_and_bos_but_includes_unchanged_content():
    ids = torch.tensor([[1, 3, 4, 0]])
    corruption = corrupt_tokens(ids, ids.ne(0), torch.zeros(1, 4, 8), probability=0)
    logits = torch.zeros(1, 4, requires_grad=True)
    loss = rtd_loss(logits, corruption)
    loss.backward()
    assert logits.grad.tolist() == [[0.0, 0.25, 0.25, 0.0]]


@pytest.mark.parametrize("mode", ["rtd", "clm", "joint"])
def test_objective_gradients_reach_only_the_intended_models_and_heads(mode):
    torch.manual_seed(3)
    model = CausalElectra(ModelConfig())
    generator = CausalElectra(ModelConfig(hidden_size=16, num_layers=1))
    ids = torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]])
    result = pretraining_step(
        model, generator, ids, ids.ne(0), ObjectiveConfig(mode=mode)
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.electra.encoder.layer[0].attention.self.query.weight.grad is not None
    assert (model.rtd_head.dense.weight.grad is not None) == (mode != "clm")
    assert (model.lm_projection[0].weight.grad is not None) == (mode != "rtd")
    assert (generator.lm_projection[0].weight.grad is not None) == (mode != "clm")


def test_rtd_does_not_backpropagate_through_sampling_into_generator():
    model = CausalElectra(ModelConfig())
    generator = CausalElectra(ModelConfig())
    ids = torch.tensor([[1, 3, 4, 5]])
    result = pretraining_step(
        model, generator, ids, ids.ne(0), ObjectiveConfig(mode="rtd")
    )
    result.rtd_loss.backward()
    assert all(parameter.grad is None for parameter in generator.parameters())


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "unknown"},
        {"temperature": 0},
        {"temperature": float("inf")},
        {"replacement_probability": -0.1},
        {"replacement_probability": float("nan")},
        {"rtd_weight": -1},
        {"generator_weight": 0},
        {"lm_weight": 0},
    ],
)
def test_invalid_objective_settings(settings):
    with pytest.raises(ValueError):
        ObjectiveConfig(**settings)


def test_missing_generator_and_empty_lm_targets_are_rejected():
    model = CausalElectra(ModelConfig())
    ids = torch.tensor([[1, 3]])
    with pytest.raises(ValueError, match="generator"):
        pretraining_step(model, None, ids, ids.ne(0), ObjectiveConfig(mode="rtd"))
    with pytest.raises(ValueError, match="next-token"):
        causal_lm_loss(torch.zeros(1, 1, 8), ids[:, :1], ids[:, :1].ne(0))
