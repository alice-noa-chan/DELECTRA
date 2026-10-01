from copy import deepcopy

import pytest
import torch

from deletcra.config import ModelConfig
from deletcra.data import synthetic_sequences
from deletcra.experiment import TrainConfig, run_experiment
from deletcra.losses import fused_causal_lm_loss
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, corrupt_tokens, pretraining_step


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_sequential_backward_matches_combined_gradients_and_one_optimizer_update(
    dropout,
):
    torch.manual_seed(3)
    a = CausalElectra(ModelConfig(vocab_size=8, dropout=dropout))
    b = deepcopy(a)
    ids = torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]])
    settings = ObjectiveConfig(generator_mode="self", lm_weight=0.7, rtd_weight=3)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001) for m in (a, b)]
    results = []
    for model, sequential in zip((a, b), (False, True), strict=True):
        torch.manual_seed(11)
        out = pretraining_step(
            model,
            None,
            ids,
            ids.ne(0),
            settings,
            backward_clean=sequential,
            rng=torch.Generator().manual_seed(9),
        )
        out.loss.backward()
        results.append(out)
    torch.testing.assert_close(results[0].loss, results[1].loss)
    assert torch.equal(results[0].corruption.input_ids, results[1].corruption.input_ids)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-5, atol=1e-7)
    for model, optimizer in zip((a, b), optimizers, strict=True):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=2e-7)


def test_sparse_feature_sampling_matches_dense_logits():
    torch.manual_seed(8)
    features = torch.randn(2, 5, 16, requires_grad=True)
    head = torch.nn.Linear(16, 8)
    ids = torch.tensor([[1, 3, 4, 5, 0], [1, 4, 2, 6, 7]])
    selected = torch.tensor(
        [[False, True, False, True, False], [False, False, False, True, False]]
    )
    args = dict(selected=selected, special_token_ids=(2,))
    a = corrupt_tokens(
        ids, ids.ne(0), head(features), rng=torch.Generator().manual_seed(9), **args
    )
    calls = []
    hook = head.register_forward_pre_hook(
        lambda _, inputs: calls.append(inputs[0].shape)
    )
    b = corrupt_tokens(
        ids,
        ids.ne(0),
        None,
        generator_features=features,
        generator_head=head,
        rng=torch.Generator().manual_seed(9),
        **args,
    )
    hook.remove()
    assert calls == [torch.Size([3, 16])]
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)
    assert not b.input_ids.requires_grad and features.grad is None


def test_empty_selected_feature_sampling_skips_projection():
    ids = torch.tensor([[1, 3, 4]])
    head = torch.nn.Linear(16, 8)

    def forbid(*args):
        raise AssertionError("no proposals need a projection")

    hook = head.register_forward_pre_hook(forbid)
    out = corrupt_tokens(
        ids,
        ids.ne(0),
        None,
        generator_features=torch.zeros(1, 3, 16),
        generator_head=head,
        probability=0,
    )
    hook.remove()
    assert torch.equal(out.input_ids, ids)


def test_liger_cpu_and_incompatible_execution_are_rejected_before_outputs(tmp_path):
    ids = synthetic_sequences(4, 6, 8, seed=3)
    with pytest.raises(ValueError, match="CUDA"):
        fused_causal_lm_loss(
            torch.nn.Linear(16, 8), torch.zeros(4, 6, 16), ids, ids.ne(0)
        )
    for objective, training in [
        (ObjectiveConfig(lm_loss_backend="liger"), TrainConfig(steps=1)),
        (ObjectiveConfig(), TrainConfig(steps=1, sequential_backward=True)),
    ]:
        with pytest.raises(ValueError):
            run_experiment(
                ids,
                ids,
                ModelConfig(vocab_size=8),
                objective,
                training,
                tmp_path / "invalid",
            )
    assert not (tmp_path / "invalid").exists()


def test_sequential_experiment_preserves_history_and_final_weights(tmp_path):
    ids = synthetic_sequences(8, 6, 8, seed=3)
    reports = []
    for sequential in (False, True):
        reports.append(
            run_experiment(
                ids,
                ids,
                ModelConfig(vocab_size=8),
                ObjectiveConfig(generator_mode="self"),
                TrainConfig(
                    steps=4, batch_size=4, probe_steps=0, sequential_backward=sequential
                ),
                tmp_path / str(sequential),
            )
        )
    for a, b in zip(reports[0]["history"], reports[1]["history"], strict=True):
        assert abs(a["loss"] - b["loss"]) < 1e-5
    a = CausalElectra.load(tmp_path / "False" / "model")
    b = CausalElectra.load(tmp_path / "True" / "model")
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=2e-7)
