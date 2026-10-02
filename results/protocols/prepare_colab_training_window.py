"""Copy the first full-corpus shuffle window for a bounded Colab training phase."""

import json
import tarfile
from pathlib import Path

from deletcra.corpus import MmapSplit, file_sha256, write_json
from deletcra.corpus_cache import write_indexed_cache
from deletcra.training import EpochSampler

source = Path("data/tinystories-full-256")
directory = Path("runs/colab-stories-window-20261002")
train, validation = MmapSplit(source, "train"), MmapSplit(source, "validation")
sampler = EpochSampler(len(train), 9)
cache = write_indexed_cache(
    directory,
    {"train": train, "validation": validation},
    {
        "train": sampler.next(8192),
        "validation": range(128),
    },
)
plan = {
    "max_input_positions": len(train) * 256,
    "batch_size": 16,
    "warmup_positions": 1048576,
    "learning_rate": 0.0003,
    "checkpoint_every": 64,
    "evaluate_every": 64,
    "validation_blocks": 128,
    "max_wall_seconds": 3600,
    "sequential_backward": True,
    "seed": 7,
}
write_json(directory / "plan.json", plan)
archive = Path("runs/colab-stories-window-20261002.tar")
with tarfile.open(archive, "w") as handle:
    for path in directory.iterdir():
        handle.add(path, arcname="stories-window/" + path.name)
record = {
    "archive_sha256": file_sha256(archive),
    "archive_bytes": archive.stat().st_size,
    "source_manifest_file_sha256": file_sha256(source / "metadata.json"),
    "cache": cache,
    "plan": plan,
    "requested_pause_steps": {"xla": 256, "cuda": 512},
    "phase_input_positions": 1048576,
    "current_training_input_limit": 2097152,
    "scope": "Partial first pass of the complete shuffled TinyStories corpus",
    "test_split_used": False,
    "publication": False,
}
write_json(Path("results/colab-stories-window-plan-20261002.json"), record)
print(json.dumps(record, indent=2))
