from copy import deepcopy

import pytest
import torch

from deletcra.config import ModelConfig
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, pretraining_step
from deletcra.runtime import TrainingRuntime, cpu_tree
from deletcra.training import ProductionPlan


def test_scaled_sequential_backward_matches_scaled_combined_gradients():
    model = CausalElectra(ModelConfig(vocab_size=8, dropout=0.1))
    other = deepcopy(model)
    tokens = torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0]])
    config = ObjectiveConfig(generator_mode="self", lm_weight=0.7, rtd_weight=3)
    for candidate, sequential in ((model, False), (other, True)):
        torch.manual_seed(11)
        result = pretraining_step(
            candidate,
            None,
            tokens,
            tokens.ne(0),
            config,
            rng=torch.Generator().manual_seed(9),
            backward_clean=sequential,
            clean_backward=lambda loss: (loss * 128).backward(),
        )
        (result.loss * 128).backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-5)


def test_cpu_runtime_clips_then_updates_and_serializes_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    runtime = TrainingRuntime("cpu", "fp32", 7)
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    runtime.backward(parameter.square().sum())
    assert runtime.step(optimizer, [parameter], 1.0)
    assert parameter.item() < 2
    checkpoint = cpu_tree({"optimizer": optimizer.state_dict()})
    assert (
        next(iter(checkpoint["optimizer"]["state"].values()))["exp_avg"].device.type
        == "cpu"
    )


def test_fp16_requires_cuda():
    with pytest.raises(ValueError, match="FP16 requires CUDA"):
        ProductionPlan(max_input_positions=16, warmup_positions=0, precision="fp16")
