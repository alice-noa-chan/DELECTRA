"""Bounded Colab TPU identity/update/resume check; allocate and export separately."""

import json
import shutil
from pathlib import Path

import torch
import torch_xla.debug.metrics as metrics

from deletcra.corpus import write_json
from deletcra.objectives import ObjectiveConfig
from deletcra.target import story_model_config
from deletcra.training import ProductionPlan, ProductionTrainer
from deletcra.verify_training import VerificationCorpus

root = Path("/content/tied-xla")
plan = ProductionPlan(
    max_input_positions=512,
    batch_size=1,
    warmup_positions=256,
    device="xla",
    precision="bf16",
    checkpoint_every=1,
    evaluate_every=100,
    validation_blocks=1,
    sequential_backward=True,
)
train, validation = VerificationCorpus(9), VerificationCorpus(10)


def construct(name, resume=False):
    trainer = ProductionTrainer(
        train,
        validation,
        story_model_config(),
        ObjectiveConfig(generator_mode="self"),
        plan,
        root / name,
        resume=resume,
    )
    assert (
        trainer.model.lm_head.weight
        is trainer.model.electra.embeddings.word_embeddings.weight
    )
    assert sum(p.numel() for p in trainer.model.parameters()) == 15041505
    assert len(trainer.optimizer.param_groups[0]["params"]) == 110
    return trainer


metrics.clear_all()
try:
    whole = construct("whole")
    whole.fit()
    print("Tied 15M whole run: two BF16 joint updates applied", flush=True)
    partial = construct("partial")
    partial.fit(pause_after_steps=1)
    shutil.copytree(root / "partial", root / "handoff")
    saved = torch.load(
        root / "partial/latest.pt", map_location="cpu", weights_only=True
    )
    resumed = construct("partial", resume=True)
    assert resumed.runtime.rng_state() == saved["xla_rng"]
    resumed.fit()
    maximum_error = 0.0
    for name, value in whole.model.state_dict().items():
        left, right = value.cpu(), resumed.model.state_dict()[name].cpu()
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        maximum_error = max(maximum_error, float((left - right).abs().max()))
    for left, right in zip(
        whole.optimizer.state.values(), resumed.optimizer.state.values(), strict=True
    ):
        for name in left:
            torch.testing.assert_close(
                left[name].cpu(), right[name].cpu(), rtol=1e-5, atol=1e-6
            )
    assert whole.sampler.offset == resumed.sampler.offset
    assert resumed.state["optimizer_updates"] == 2
    gradients = [
        p.grad.cpu() for p in resumed.model.rtd_head.parameters() if p.grad is not None
    ]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    report = {
        "status": "passed",
        "state": resumed.state,
        "model_maximum_absolute_error": maximum_error,
        "model_and_optimizer_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "device_rng_restored": True,
        "vocabulary_weights_tied": True,
        "model_parameters": 15041505,
        "optimizer_parameter_entries": 110,
        "rtd_gradients": "finite_nonzero",
    }
except Exception as error:
    report = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2500],
    }
report.update(
    scope="tied_15m_xla_bf16_joint_and_same_device_resume",
    source_commit="9f12ccd",
    quality_evaluated=False,
    production_throughput_measured=False,
    xla_metrics=metrics.metrics_report(),
)
write_json(Path("/content/tied-xla-verification.json"), report)
print(
    json.dumps(
        {key: value for key, value in report.items() if key != "xla_metrics"}, indent=2
    ),
    flush=True,
)
