"""Check the real 15M trainer and interrupted resume on a chosen accelerator.

This small synthetic run checks execution, not language quality or throughput.
The caller owns accelerator allocation, durable export and runtime shutdown.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from deletcra.corpus import write_json
from deletcra.data import synthetic_sequences
from deletcra.objectives import ObjectiveConfig
from deletcra.target import story_model_config
from deletcra.training import ProductionPlan, ProductionTrainer


class VerificationCorpus:
    """Four fixed blocks use the production model's complete vocabulary/context."""

    def __init__(self, seed: int):
        self.tokens = synthetic_sequences(4, 256, 32000, seed=seed)
        self.metadata = {
            "status": "complete",
            "sequence_length": 256,
            "vocab_size": 32000,
            "bos_token_id": 1,
            "pad_token_id": 0,
            "kind": "synthetic-accelerator-verification",
        }

    def __len__(self):
        return len(self.tokens)

    def batch(self, indices):
        return self.tokens[list(indices)]


def verify_training(device: str, directory: Path) -> dict:
    """Keep optimizer and sampler state when resuming on the same device type.

    GPU kernels may differ numerically, so compare FP32 weights and Adam moments
    with explicit tolerances rather than claiming cross-device bitwise equality.
    This does not enable TPU or a change of device/configuration during resume.
    """
    train, validation = VerificationCorpus(9), VerificationCorpus(10)
    config = replace(story_model_config(), attention_backend="sdpa")
    objective = ObjectiveConfig(mode="joint", generator_mode="self")
    plan = ProductionPlan(
        max_input_positions=1024,
        batch_size=1,
        warmup_positions=256,
        device=device,
        precision="fp32",
        checkpoint_every=2,
        evaluate_every=2,
        validation_blocks=2,
        sequential_backward=True,
    )
    whole = ProductionTrainer(
        train, validation, config, objective, plan, directory / "whole"
    )
    whole.fit()
    partial = ProductionTrainer(
        train, validation, config, objective, plan, directory / "resumed"
    )
    partial.fit(pause_after_steps=2)
    resumed = ProductionTrainer(
        train, validation, config, objective, plan, directory / "resumed", resume=True
    )
    resumed.fit()
    for field in ("step", "input_positions", "prediction_targets"):
        if whole.state[field] != resumed.state[field]:
            raise AssertionError(f"resume changed {field}")
    if (whole.sampler.epoch, whole.sampler.offset) != (
        resumed.sampler.epoch,
        resumed.sampler.offset,
    ):
        raise AssertionError("resume changed the data cursor")
    maximum_error = 0.0
    for name, tensor in whole.model.state_dict().items():
        other = resumed.model.state_dict()[name]
        torch.testing.assert_close(tensor, other, rtol=1e-5, atol=1e-6)
        maximum_error = max(maximum_error, float((tensor - other).abs().max()))
    for left, right in zip(
        whole.optimizer.state.values(), resumed.optimizer.state.values(), strict=True
    ):
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=1e-5, atol=1e-6)
    report = {
        "status": "passed",
        "scope": "15m_synthetic_joint_fp32_and_same_device_resume",
        "device": device,
        "model_parameters": sum(value.numel() for value in whole.model.parameters()),
        "state": resumed.state,
        "model_maximum_absolute_error": maximum_error,
        "model_and_optimizer_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "production_throughput_measured": False,
        "quality_evaluated": False,
        "cross_device_resume_verified": False,
    }
    write_json(directory / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(verify_training(args.device, args.output), indent=2))


if __name__ == "__main__":
    main()
