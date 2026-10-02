"""Confirm that the full epoch shuffle is identical across host Torch versions."""

import hashlib
import json
from pathlib import Path

from deletcra.corpus_cache import IndexedSplit
from deletcra.training import EpochSampler

reader = IndexedSplit(Path("/content/stories-window"), "train")
sampler = EpochSampler(len(reader), 9)
digest = hashlib.sha256(sampler.order.astype("<i8").tobytes()).hexdigest()
report = {
    "status": "passed"
    if digest == "1bfb2f5dc82710dab33aaa3775a9227c3880ed3c637d8e8492efc5bafc6046df"
    else "failed",
    "epoch": 0,
    "source_blocks": len(reader),
    "sampler_seed": 9,
    "global_order_sha256": digest,
}
Path("/content/stories-training-shuffle-audit.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
