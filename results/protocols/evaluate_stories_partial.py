"""Frozen CPU validation of the partial checkpoint, including RTD confusion counts."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from deletcra.config import ModelConfig
from deletcra.corpus import file_sha256, write_json
from deletcra.corpus_cache import IndexedSplit
from deletcra.evaluation import evaluate_frozen
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig

torch.set_num_threads(2)
destination = Path("results/colab-stories-frozen-validation-20261002.json")
if destination.exists():
    raise FileExistsError("refusing to overwrite frozen validation evidence")
checkpoint = Path("runs/colab-stories-cuda-000512-20261002/stories-training/latest.pt")
saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
model = CausalElectra(
    replace(ModelConfig(**saved["spec"]["model"]), attention_backend="sdpa")
)
model.load_state_dict(saved["model"])
result = evaluate_frozen(
    model,
    IndexedSplit(Path("runs/colab-stories-window-20261002"), "validation"),
    batch_size=8,
    max_blocks=128,
    objective=ObjectiveConfig(**saved["spec"]["objective"]),
)
result.update(
    scope="frozen_partial_checkpoint_on_training_validation_window",
    checkpoint_sha256=file_sha256(checkpoint),
    checkpoint_step=512,
    device="cpu",
    precision="fp32",
    proposal_rng_seed=1007,
    independent_test=False,
    quality_gate_passed=False,
)
write_json(destination, result)
print(json.dumps(result, indent=2))
