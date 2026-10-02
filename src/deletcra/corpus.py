"""Compact disk-backed token streams with explicit split and tail accounting."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Replace a manifest only after its complete contents reach disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


class TokenWriter:
    """Append little-endian int32 IDs without retaining the corpus in RAM.

    A shard stores the record-separated stream. Loading inserts BOS before each
    255-token block, matching the pilot's context-256 protocol. Incomplete tails
    are counted and removed once per shard; they never enter another split.
    """

    def __init__(self, path: Path, sequence_length: int, vocab_size: int):
        if sequence_length < 3 or vocab_size < 4:
            raise ValueError("invalid sequence length or vocabulary")
        self.path = path
        self.width = sequence_length - 1
        self.vocab_size = vocab_size
        self.count = 0
        self.content = 0
        self.documents = 0
        self.source = path.open("wb")

    def append(self, ids: list[int], separator: int) -> None:
        values = np.asarray([*ids, separator], dtype="<i4")
        if np.any(values < 0) or np.any(values >= self.vocab_size):
            raise ValueError("token outside vocabulary")
        self.source.write(values.tobytes())
        self.count += len(values)
        self.content += len(ids)
        self.documents += 1

    def finish(self) -> dict:
        tail = self.count % self.width
        self.source.truncate((self.count - tail) * 4)
        self.source.flush()
        os.fsync(self.source.fileno())
        self.source.close()
        return {
            "path": self.path.name,
            "documents": self.documents,
            "content_tokens_before_tail": self.content,
            "record_separators": self.documents,
            "stream_tokens_before_tail": self.count,
            "dropped_tail_tokens": tail,
            "prediction_targets": self.count - tail,
            "blocks": (self.count - tail) // self.width,
            "bytes": (self.count - tail) * 4,
            "sha256": file_sha256(self.path),
        }


class MmapSplit:
    """Read [batch, sequence] int64 tensors from small int32 disk shards.

    The corpus stays read-only. Advanced indexing copies just the requested
    blocks, preventing PyTorch from exposing writable aliases to the mmap.
    """

    def __init__(self, directory: str | Path, split: str, *, verify: bool = True):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.split = split
        self.width = self.metadata["sequence_length"] - 1
        self.shards = []
        self.ends = []
        total = 0
        for entry in self.metadata["splits"][split]:
            path = (directory / entry["path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("shard path escapes corpus")
            if path.stat().st_size != entry["bytes"]:
                raise ValueError("shard size mismatch")
            if verify and file_sha256(path) != entry["sha256"]:
                raise ValueError("shard hash mismatch")
            if entry["blocks"]:
                self.shards.append(
                    np.memmap(path, dtype="<i4", mode="r").reshape(
                        entry["blocks"], self.width
                    )
                )
                total += entry["blocks"]
                self.ends.append(total)
        self.ends = np.asarray(self.ends)
        if total == 0:
            raise ValueError("split has no complete blocks")

    def __len__(self):
        return int(self.ends[-1])

    def batch(self, indices) -> torch.Tensor:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not len(indices) or np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError("batch indices outside split")
        result = np.full(
            (len(indices), self.width + 1),
            self.metadata["bos_token_id"],
            dtype=np.int64,
        )
        owners = np.searchsorted(self.ends, indices, side="right")
        for owner in np.unique(owners):
            selected = owners == owner
            offset = self.ends[owner - 1] if owner else 0
            result[selected, 1:] = self.shards[owner][indices[selected] - offset]
        if np.any(result < 0) or np.any(result >= self.metadata["vocab_size"]):
            raise ValueError("corrupt token IDs")
        return torch.from_numpy(result)
