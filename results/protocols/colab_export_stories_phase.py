"""Export a paused training phase with hashes, environment and raw XLA metrics."""

import json
import os
import platform
import tarfile
from pathlib import Path

import torch
import transformers

from deletcra.corpus import file_sha256, write_json

root = Path("/content/stories-training")
device = os.environ["DELECTRA_TRAIN_DEVICE"]
saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=True)
step = saved["state"]["step"]
label = f"stories-{device}-{step:06d}"
if device == "xla":
    import torch_xla.debug.metrics as metrics

    write_json(
        root / "xla-metrics.json",
        {"scope": "training_and_validation_graphs", "raw": metrics.metrics_report()},
    )
environment = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "device": device,
}
if device == "cuda":
    environment.update(
        name=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
archive = Path("/content") / f"{label}.tar"
with tarfile.open(archive, "w") as handle:
    for path in sorted(root.iterdir()):
        if (
            path.suffix == ".json"
            or path.name in {"latest.pt", "metrics.jsonl"}
            or (step in {256, 512} and path.name == "best.pt")
        ):
            handle.add(path, arcname="stories-training/" + path.name)
record = {
    "archive": archive.name,
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": file_sha256(archive),
    "checkpoint_sha256": file_sha256(root / "latest.pt"),
    "state": saved["state"],
    "sampler": saved["sampler"],
    "spec": saved["spec"],
    "environment": environment,
    "code_commit": "c51a60a",
    "wheel_sha256": "e82834f3840a48debf7beba1fe47f6f92de6d437ba02026eac080fa12e7c8ac5",
    "test_split_used": False,
    "full_corpus_trained": False,
    "published": False,
}
write_json(Path("/content") / f"{label}-export.json", record)
print(json.dumps(record, indent=2), flush=True)
