"""Train a full-corpus window; allocation, export and shutdown are caller-owned."""

import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from deletcra.corpus import write_json
from deletcra.corpus_cache import IndexedSplit
from deletcra.objectives import ObjectiveConfig
from deletcra.target import story_model_config
from deletcra.training import ProductionPlan, ProductionTrainer

root = Path("/content/stories-window")
directory = Path("/content/stories-training")
device = os.environ["DELECTRA_TRAIN_DEVICE"]
stop_step = int(os.environ["DELECTRA_TRAIN_STOP_STEP"])
precision = "bf16" if device == "xla" else "fp16"
plan = ProductionPlan(
    **json.loads((root / "plan.json").read_text()),
    device=device,
    precision=precision,
    fused_optimizer=device == "cuda",
)
if "window_trainer" not in globals():
    train, validation = IndexedSplit(root, "train"), IndexedSplit(root, "validation")
    checkpoint = directory / "latest.pt"
    before = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.exists()
        else None
    )
    window_trainer = ProductionTrainer(
        train,
        validation,
        replace(
            story_model_config(),
            attention_backend="eager" if device == "xla" else "sdpa",
        ),
        ObjectiveConfig(generator_mode="self"),
        plan,
        directory,
        resume=before is not None,
        migrate=before is not None and before["spec"]["plan"]["device"] != device,
    )
    assert sum(p.numel() for p in window_trainer.model.parameters()) == 15041505
    assert (
        window_trainer.model.lm_head.weight
        is window_trainer.model.electra.embeddings.word_embeddings.weight
    )
    if before is not None:
        for name, value in window_trainer.model.state_dict().items():
            torch.testing.assert_close(
                value.cpu(), before["model"][name], rtol=0, atol=0
            )
        for old, new in zip(
            before["optimizer"]["state"].values(),
            window_trainer.optimizer.state.values(),
            strict=True,
        ):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(old[name], new[name].cpu(), rtol=0, atol=0)
        assert window_trainer.sampler.offset == before["sampler"]["offset"]
        print("Checkpoint model/Adam/cursor preservation verified", flush=True)
    else:
        baseline = window_trainer.evaluate(validation, 128)
        write_json(directory / "initial-validation.json", baseline)
        print("Initial validation", json.dumps(baseline), flush=True)
if window_trainer.plan != plan:
    raise ValueError("existing kernel trainer has a different plan")
report = window_trainer.fit(pause_after_steps=stop_step)
report.update(
    phase_device=device,
    phase_precision=precision,
    requested_stop_step=stop_step,
    requested_stop_reached=window_trainer.state["step"] == stop_step,
    full_corpus_trained=False,
    release_quality_evaluated=False,
    migrations=window_trainer.migrations,
    source_corpus_manifest_sha256=window_trainer.spec["corpus_manifest_sha256"],
)
write_json(directory / f"phase-{device}-{window_trainer.state['step']}.json", report)
print(json.dumps(report, indent=2), flush=True)
