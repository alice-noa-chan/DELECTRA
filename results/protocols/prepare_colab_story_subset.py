"""Export a provenance-tracked TinyStories sample without accessing the test split."""

import json
from pathlib import Path

import numpy as np

from deletcra.corpus import MmapSplit, file_sha256, write_json

source = Path("data/tinystories-full-256")
destination = Path("runs/colab-story-subset")
destination.mkdir(exist_ok=True)
metadata = json.loads((source / "metadata.json").read_text())
metadata["kind"] = "bounded-real-corpus-verification-subset"
metadata["source_manifest_sha256"] = file_sha256(source / "metadata.json")
metadata["splits"] = {}
for split, count in (("train", 256), ("validation", 16)):
    reader = MmapSplit(source, split)
    indices = np.linspace(0, len(reader) - 1, count, dtype=np.int64)
    tokens = reader.batch(indices)[:, 1:].numpy().astype("<i4")
    path = destination / f"{split}.bin"
    path.write_bytes(tokens.tobytes())
    metadata["splits"][split] = [
        {
            "path": path.name,
            "blocks": count,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    ]
    metadata[f"source_{split}_block_indices"] = indices.tolist()
write_json(destination / "metadata.json", metadata)
print("Prepared 256 training / 16 validation blocks; no test data used")
