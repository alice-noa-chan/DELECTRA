"""Check the exported TPU run on a Colab GPU; no allocation or publication."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from deletcra.corpus import write_json
from deletcra.objectives import ObjectiveConfig
from deletcra.target import story_model_config
from deletcra.training import ProductionPlan, ProductionTrainer
from deletcra.verify_training import VerificationCorpus

directory = Path("/content/tied-xla/handoff")
source = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
plan = ProductionPlan(**source["spec"]["plan"])
plan = replace(plan, device="cuda", precision="fp16", fused_optimizer=True)
try:
    trainer = ProductionTrainer(
        VerificationCorpus(9),
        VerificationCorpus(10),
        replace(story_model_config(), attention_backend="sdpa"),
        ObjectiveConfig(generator_mode="self"),
        plan,
        directory,
        resume=True,
        migrate=True,
    )
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), source["model"][name], rtol=0, atol=0)
    for old, new in zip(
        source["optimizer"]["state"].values(),
        trainer.optimizer.state.values(),
        strict=True,
    ):
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(old[name], new[name].cpu(), rtol=0, atol=0)
    assert trainer.sampler.offset == source["sampler"]["offset"]
    assert trainer.state["input_positions"] == source["state"]["input_positions"]
    print(
        "Weights, Adam moments and data cursor survived the handoff exactly", flush=True
    )
    result = trainer.fit()
    assert trainer.state["optimizer_updates"] > source["state"]["optimizer_updates"]
    assert trainer.state["input_positions"] == 512
    assert (
        trainer.model.lm_head.weight
        is trainer.model.electra.embeddings.word_embeddings.weight
    )
    report = {
        "status": "passed",
        "state": result["state"],
        "before_update_state_preservation": "exact",
        "migrations": trainer.migrations,
        "grad_scaler": trainer.runtime.scaler.state_dict(),
    }
except Exception as error:
    report = {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error)[:2500],
    }
report.update(
    scope="15m_xla_bf16_to_t4_fp16_joint_checkpoint_handoff", quality_evaluated=False
)
write_json(Path("/content/gpu-handoff-verification.json"), report)
print(json.dumps(report, indent=2), flush=True)
