"""Indexed corpus windows for short sessions without changing sampler identity."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deletcra.corpus import file_sha256, write_json


def manifest_identity(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def write_indexed_cache(directory: Path, splits: dict, indices: dict) -> dict:
    """Copy named global blocks from verified readers, retaining the full manifest.

    Readers must share a manifest. Cached rows are sorted by global block index;
    the sampler still sees the original split length and ordering. A cache is a
    transfer window, not a smaller dataset or evidence of complete coverage.
    """
    metadata = next(iter(splits.values())).metadata
    if any(reader.metadata != metadata for reader in splits.values()):
        raise ValueError("cache readers must share a corpus manifest")
    selected = {}
    for split, reader in splits.items():
        values = np.sort(np.asarray(indices[split], dtype=np.int64))
        if (
            values.ndim != 1
            or not len(values)
            or values[0] < 0
            or values[-1] >= len(reader)
            or np.any(values[1:] == values[:-1])
        ):
            raise ValueError("cache indices must be unique valid global blocks")
        selected[split] = values
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "metadata.json", metadata)
    cache = {
        "status": "complete",
        "kind": "indexed-transfer-window",
        "corpus_manifest_sha256": manifest_identity(metadata),
        "splits": {},
    }
    for split, reader in splits.items():
        values = selected[split]
        index_path, token_path = directory / f"{split}.npy", directory / f"{split}.bin"
        np.save(index_path, values, allow_pickle=False)
        with token_path.open("wb") as handle:
            for start in range(0, len(values), 128):
                tokens = reader.batch(values[start : start + 128])
                if not tokens[:, 0].eq(metadata["bos_token_id"]).all():
                    raise ValueError("cache source lacks a BOS prefix")
                handle.write(tokens[:, 1:].numpy().astype("<i4").tobytes())
        cache["splits"][split] = {
            "source_blocks": len(reader),
            "cached_blocks": len(values),
            "indices": index_path.name,
            "indices_sha256": file_sha256(index_path),
            "tokens": token_path.name,
            "tokens_sha256": file_sha256(token_path),
            "token_bytes": token_path.stat().st_size,
        }
    write_json(directory / "cache.json", cache)
    return cache


class IndexedSplit:
    """Read cached global blocks while exposing the original corpus identity.

    Missing blocks raise before an update; they are never substituted or skipped.
    The unchanged manifest and split length let checkpoints resume on a later
    cache window or the complete MmapSplit without restarting shuffle or warmup.
    """

    def __init__(self, directory: Path, split: str):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.cache = json.loads((directory / "cache.json").read_text())
        if self.cache.get("status") != "complete" or self.cache.get(
            "corpus_manifest_sha256"
        ) != manifest_identity(self.metadata):
            raise ValueError("incomplete cache or corpus identity mismatch")
        entry = self.cache["splits"][split]
        self.source_blocks = sum(x["blocks"] for x in self.metadata["splits"][split])
        if self.source_blocks != entry["source_blocks"]:
            raise ValueError("cache source length mismatch")
        paths = {}
        for kind in ("indices", "tokens"):
            path = (directory / entry[kind]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("cache path escapes corpus")
            if file_sha256(path) != entry[f"{kind}_sha256"]:
                raise ValueError("cache hash mismatch")
            paths[kind] = path
        self.indices = np.load(paths["indices"], allow_pickle=False)
        if (
            self.indices.dtype != np.dtype("int64")
            or self.indices.shape != (entry["cached_blocks"],)
            or not len(self.indices)
            or self.indices[0] < 0
            or self.indices[-1] >= self.source_blocks
            or np.any(self.indices[1:] <= self.indices[:-1])
        ):
            raise ValueError("invalid cached global indices")
        self.width = self.metadata["sequence_length"] - 1
        expected_bytes = len(self.indices) * self.width * 4
        if (
            paths["tokens"].stat().st_size != expected_bytes
            or entry["token_bytes"] != expected_bytes
        ):
            raise ValueError("cache token size mismatch")
        self.tokens = np.memmap(paths["tokens"], dtype="<i4", mode="r").reshape(
            len(self.indices), self.width
        )

    def __len__(self):
        return self.source_blocks

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(self.indices, indices)
        if not len(indices) or np.any(positions >= len(self.indices)):
            raise ValueError("requested block is not cached")
        if np.any(self.indices[positions] != indices):
            raise ValueError("requested block is not cached")
        result = np.full(
            (len(indices), self.width + 1),
            self.metadata["bos_token_id"],
            dtype=np.int64,
        )
        result[:, 1:] = self.tokens[positions]
        if np.any(result < 0) or np.any(result >= self.metadata["vocab_size"]):
            raise ValueError("corrupt cached token IDs")
        return torch.from_numpy(result)
