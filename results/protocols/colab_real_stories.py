"""Check four joint updates on real TinyStories blocks; this is a private pilot."""

import json
from dataclasses import replace
from pathlib import Path

import torch

from deletcra.corpus import MmapSplit, write_json
from deletcra.objectives import ObjectiveConfig
from deletcra.target import story_model_config
from deletcra.training import ProductionPlan, ProductionTrainer

root = Path("/content/colab-story-subset")
train, validation = MmapSplit(root, "train"), MmapSplit(root, "validation")
trainer = ProductionTrainer(
    train,
    validation,
    replace(story_model_config(), attention_backend="sdpa"),
    ObjectiveConfig(generator_mode="self"),
    ProductionPlan(
        max_input_positions=16384,
        batch_size=16,
        warmup_positions=4096,
        device="cuda",
        precision="fp16",
        checkpoint_every=4,
        evaluate_every=4,
        validation_blocks=16,
        sequential_backward=True,
        fused_optimizer=True,
    ),
    Path("/content/real-stories-check"),
)
torch.cuda.reset_peak_memory_stats()
result = trainer.fit()
assert result["state"]["optimizer_updates"] == 4
assert result["state"]["skipped_updates"] == 0
report = {
    "status": "passed",
    "scope": "bounded_tinystories_subset_fp16_joint_execution",
    "state": result["state"],
    "plan": trainer.spec["plan"],
    "corpus": train.metadata,
    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    "quality_evaluated": False,
    "full_corpus_trained": False,
    "production_throughput_measured": False,
}
write_json(Path("/content/real-stories-verification.json"), report)
print(
    json.dumps(
        {key: value for key, value in report.items() if key != "corpus"}, indent=2
    ),
    flush=True,
)
