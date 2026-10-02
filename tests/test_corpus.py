import json

import pytest
import torch

from deletcra.corpus import MmapSplit, TokenWriter, write_json
from deletcra.data import pack_texts


class Tokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(x) for x in text.split()]}


def test_disk_stream_matches_pilot_and_accounts_for_tail(tmp_path):
    texts = ["2 3 4", "5 6", "7 8 9"]
    writer = TokenWriter(tmp_path / "train.bin", 5, 20)
    for text in texts:
        writer.append(Tokenizer()(text)["input_ids"], 1)
    entry = writer.finish()
    write_json(
        tmp_path / "metadata.json",
        {
            "sequence_length": 5,
            "bos_token_id": 1,
            "vocab_size": 20,
            "splits": {"train": [entry]},
        },
    )
    corpus = MmapSplit(tmp_path, "train")
    expected = pack_texts(texts, Tokenizer(), sequence_length=5, max_tokens=100)
    assert torch.equal(corpus.batch(range(len(corpus))), expected)
    assert torch.equal(corpus.batch([1, 0, 1]), expected[[1, 0, 1]])
    assert entry["documents"] == 3
    assert entry["content_tokens_before_tail"] == 8
    assert entry["dropped_tail_tokens"] == 3
    assert entry["bytes"] == 32
    with pytest.raises(ValueError, match="indices"):
        corpus.batch([-1])
    with (tmp_path / "train.bin").open("r+b") as source:
        source.write(b"\x0f")
    with pytest.raises(ValueError, match="hash"):
        MmapSplit(tmp_path, "train")


def test_mmap_reads_across_shards_and_rejects_path_escape(tmp_path):
    entries = []
    for i in range(2):
        writer = TokenWriter(tmp_path / f"part{i}.bin", 4, 20)
        writer.append([i + 2, i + 3], 1)
        entries.append(writer.finish())
    metadata = {
        "sequence_length": 4,
        "bos_token_id": 1,
        "vocab_size": 20,
        "splits": {"train": entries},
    }
    write_json(tmp_path / "metadata.json", metadata)
    corpus = MmapSplit(tmp_path, "train")
    assert corpus.batch([1, 0]).tolist() == [[1, 3, 4, 1], [1, 2, 3, 1]]
    metadata["splits"]["train"][0]["path"] = "../escape.bin"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="escapes"):
        MmapSplit(tmp_path, "train")


def test_atomic_replace_retries_sharing_lock_and_keeps_failure_bounded(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from deletcra.corpus import atomic_replace

    source, target = tmp_path / "temporary", tmp_path / "latest"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    replace = Path.replace
    calls, delays = [], []

    def locked(path, destination):
        calls.append(destination)
        if len(calls) < 3:
            raise PermissionError("temporary sharing lock")
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr("deletcra.corpus.time.sleep", delays.append)
    atomic_replace(source, target)
    assert target.read_bytes() == b"new"
    assert delays == [0.05, 0.1]

    def denied(*args):
        calls.append(target)
        raise PermissionError("persistent")

    calls.clear()
    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError, match="persistent"):
        atomic_replace(source, target)
    assert len(calls) == 6
    assert target.read_bytes() == b"new"
