from copy import deepcopy

import pytest
import torch

from deletcra.config import ModelConfig
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, causal_lm_loss, rtd_loss
from deletcra.static_objectives import (
    masked_lm_loss,
    static_corruption,
    static_pretraining_step,
)


def test_static_ce_matches_sparse_loss_and_gradients_with_padding():
    tokens = torch.tensor([[1, 3, 4, 0], [1, 4, 5, 6]])
    dense = torch.randn(2, 4, 8, requires_grad=True)
    sparse = dense.detach().clone().requires_grad_()
    valid = tokens[:, :-1].ne(0) & tokens[:, 1:].ne(0)
    a = masked_lm_loss(dense[:, :-1], tokens[:, 1:], valid)
    b = causal_lm_loss(sparse, tokens, tokens.ne(0))
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(dense.grad, sparse.grad)


def test_static_sampling_respects_shift_specials_and_actual_change_labels():
    tokens = torch.tensor([[1, 3, 4, 5, 0]])
    logits = torch.full((1, 5, 8), -100.0)
    for position, target in enumerate((3, 4, 6, 7, 5)):
        logits[:, position, target] = 100
    corruption = static_corruption(
        tokens,
        tokens.ne(0),
        logits,
        ObjectiveConfig(replacement_probability=1),
        [0, 1, 2],
        torch.Generator().manual_seed(9),
    )
    assert corruption.input_ids.tolist() == [[1, 3, 4, 6, 0]]
    assert corruption.labels.tolist() == [[False, False, False, True, False]]
    scores = torch.randn_like(tokens.float())
    reference = rtd_loss(scores, corruption)
    dense = torch.nn.functional.binary_cross_entropy_with_logits(
        scores, corruption.labels.float(), reduction="none"
    )
    torch.testing.assert_close(
        reference, (dense * corruption.eligible).sum() / corruption.eligible.sum()
    )


@pytest.mark.parametrize("mode", ["rtd", "clm", "joint"])
def test_static_supports_separate_generator_and_gradient_paths(mode):
    model = CausalElectra(ModelConfig(vocab_size=8))
    generator = CausalElectra(ModelConfig(vocab_size=8)) if mode != "clm" else None
    tokens = torch.tensor([[1, 3, 4, 5]])
    out = static_pretraining_step(
        model,
        generator,
        tokens,
        tokens.ne(0),
        ObjectiveConfig(mode=mode),
        rng=torch.Generator().manual_seed(9),
        special_token_ids=(2,),
    )
    out.loss.backward()
    if mode != "clm":
        assert model.rtd_head.dense.weight.grad.abs().sum() > 0
        assert generator.lm_head.weight.grad.abs().sum() > 0
    if mode != "rtd":
        assert model.lm_head.weight.grad.abs().sum() > 0


def test_static_sequential_matches_combined_and_excludes_bos_only_padding_rows():
    model = CausalElectra(ModelConfig(vocab_size=8))
    other = deepcopy(model)
    tokens = torch.tensor([[1, 3, 4, 5], [1, 0, 0, 0]])
    settings = ObjectiveConfig(generator_mode="self")
    for candidate, sequential in ((model, False), (other, True)):
        output = static_pretraining_step(
            candidate,
            None,
            tokens,
            tokens.ne(0),
            settings,
            rng=torch.Generator().manual_seed(9),
            special_token_ids=(2,),
            backward_clean=sequential,
            clean_backward=lambda loss: loss.backward(),
        )
        output.loss.backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=1e-7)
    solo = static_pretraining_step(
        model,
        None,
        tokens[:1],
        tokens[:1].ne(0),
        settings,
        rng=torch.Generator().manual_seed(9),
        special_token_ids=(2,),
    )
    # Proposal RNG consumes a different-sized batch, so compare deterministic
    # clean CLM only; RTD is validated with identical draws above.
    torch.testing.assert_close(solo.lm_loss, output.lm_loss)
