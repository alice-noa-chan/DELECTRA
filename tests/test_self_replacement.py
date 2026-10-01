from copy import deepcopy

import pytest
import torch

from deletcra.cli import main
from deletcra.config import ModelConfig
from deletcra.data import synthetic_sequences
from deletcra.experiment import TrainConfig, run_experiment
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, pretraining_step


def test_self_replacement_uses_two_passes_and_does_not_duplicate_clm_loss():
    torch.manual_seed(4)
    model = CausalElectra(ModelConfig(vocab_size=8))
    ids = torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]])
    passes = []
    handle = model.register_forward_pre_hook(lambda _, args: passes.append(args[0]))
    result = pretraining_step(
        model,
        None,
        ids,
        ids.ne(0),
        ObjectiveConfig(generator_mode="self", replacement_probability=1),
        rng=torch.Generator().manual_seed(6),
    )
    handle.remove()
    assert len(passes) == 2
    assert torch.equal(passes[0], ids)
    assert torch.equal(passes[1], result.corruption.input_ids)
    assert result.generator_loss.item() == 0
    torch.testing.assert_close(result.loss, result.lm_loss + 5 * result.rtd_loss)
    result.loss.backward()
    assert model.lm_projection[0].weight.grad.abs().sum() > 0
    assert model.rtd_head.dense.weight.grad.abs().sum() > 0


def test_self_rtd_has_no_gradient_through_the_lm_sampling_head():
    model = CausalElectra(ModelConfig(vocab_size=8))
    ids = torch.tensor([[1, 3, 4, 5]])
    result = pretraining_step(
        model, None, ids, ids.ne(0), ObjectiveConfig(generator_mode="self")
    )
    result.rtd_loss.backward()
    assert model.lm_projection[0].weight.grad is None
    assert model.electra.embeddings.word_embeddings.weight.grad is not None


def test_self_replacement_keeps_future_tokens_out_of_proposals():
    model = CausalElectra(ModelConfig(vocab_size=8)).eval()
    other = deepcopy(model)
    ids = torch.tensor([[1, 3, 4, 5, 6]])
    changed = ids.clone()
    changed[:, -1] = 7
    settings = ObjectiveConfig(generator_mode="self", replacement_probability=1)
    a = pretraining_step(
        model, None, ids, ids.ne(0), settings, rng=torch.Generator().manual_seed(9)
    )
    b = pretraining_step(
        other,
        None,
        changed,
        changed.ne(0),
        settings,
        rng=torch.Generator().manual_seed(9),
    )
    assert torch.equal(a.corruption.input_ids[:, :-1], b.corruption.input_ids[:, :-1])
    torch.testing.assert_close(
        a.rtd_logits[:, :-1], b.rtd_logits[:, :-1], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kwargs", [{"mode": "clm"}, {"mode": "rtd"}, {"generator_weight": 2}]
)
def test_self_rejects_ambiguous_loss_settings(kwargs):
    with pytest.raises(ValueError):
        ObjectiveConfig(generator_mode="self", **kwargs)


def test_self_experiment_saves_one_reloadable_model_and_reports_actual_parameters(
    tmp_path,
):
    data = synthetic_sequences(8, 6, 8, seed=1)
    path = tmp_path / "self"
    report = run_experiment(
        data,
        data,
        ModelConfig(vocab_size=8),
        ObjectiveConfig(generator_mode="self"),
        TrainConfig(steps=3, batch_size=4, probe_steps=0),
        path,
    )
    assert report["generator_parameters"] == 0
    assert report["unique_optimized_parameters"] == report["model_parameters"]
    assert "generator_loss" not in report["final_validation"]
    assert "rtd_loss" in report["final_validation"]
    assert not (path / "generator").exists()
    assert torch.isfinite(CausalElectra.load(path / "model")(data).lm_logits).all()


def test_self_cli_requires_explicit_joint_mode_before_creating_outputs(tmp_path):
    path = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(["run", "--generator-mode", "self", "--output-dir", str(path)])
    assert not path.exists()
