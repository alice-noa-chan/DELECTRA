import json

import numpy as np
import pytest
import torch

from deletcra.config import ModelConfig
from deletcra.corpus import MmapSplit, file_sha256, write_json
from deletcra.corpus_cache import IndexedSplit, write_indexed_cache
from deletcra.objectives import ObjectiveConfig
from deletcra.training import EpochSampler, ProductionPlan, ProductionTrainer


def readers(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    splits = {}
    for name, count in (("train", 12), ("validation", 4)):
        path = source / f"{name}.bin"
        tokens = np.arange(count * 7, dtype=np.int32).reshape(count, 7) % 29 + 3
        path.write_bytes(tokens.astype("<i4").tobytes())
        splits[name] = [
            {
                "path": path.name,
                "blocks": count,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        ]
    write_json(
        source / "metadata.json",
        {
            "status": "complete",
            "sequence_length": 8,
            "vocab_size": 32,
            "bos_token_id": 1,
            "pad_token_id": 0,
            "splits": splits,
        },
    )
    return {name: MmapSplit(source, name) for name in splits}


def test_global_blocks_identity_and_missing_blocks(tmp_path):
    full = readers(tmp_path)
    cache = tmp_path / "cache"
    write_indexed_cache(cache, full, {"train": [8, 1, 4], "validation": [0, 1]})
    window = IndexedSplit(cache, "train")
    assert len(window) == len(full["train"]) == 12
    assert window.metadata == full["train"].metadata
    torch.testing.assert_close(
        window.batch([8, 1, 8]), full["train"].batch([8, 1, 8]), rtol=0, atol=0
    )
    for missing in ([0], [12], [-1], []):
        with pytest.raises(ValueError, match="not cached"):
            window.batch(missing)
    with (cache / "train.bin").open("r+b") as handle:
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="hash"):
        IndexedSplit(cache, "train")


@pytest.mark.parametrize("indices", [[1, 1], [-1], [12], []])
def test_invalid_window_fails_without_creating_output(tmp_path, indices):
    full = readers(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="unique valid"):
        write_indexed_cache(cache, full, {"train": indices, "validation": [0]})
    assert not cache.exists()


def test_manifest_tampering_is_rejected(tmp_path):
    full = readers(tmp_path)
    cache = tmp_path / "cache"
    write_indexed_cache(cache, full, {"train": [1], "validation": [0]})
    metadata = json.loads((cache / "metadata.json").read_text())
    metadata["vocab_size"] = 33
    write_json(cache / "metadata.json", metadata)
    with pytest.raises(ValueError, match="identity"):
        IndexedSplit(cache, "train")


def test_window_to_full_corpus_resume_keeps_shuffle_optimizer_and_schedule(tmp_path):
    full = readers(tmp_path)
    plan = ProductionPlan(
        max_input_positions=96,
        batch_size=2,
        warmup_positions=16,
        checkpoint_every=20,
        evaluate_every=20,
    )
    sampler = EpochSampler(len(full["train"]), plan.seed + 2)
    cache = tmp_path / "cache"
    write_indexed_cache(cache, full, {"train": sampler.next(4), "validation": range(4)})
    config, objective = (
        ModelConfig(max_positions=8),
        ObjectiveConfig(generator_mode="self"),
    )
    whole = ProductionTrainer(
        full["train"], full["validation"], config, objective, plan, tmp_path / "whole"
    )
    whole.fit()
    partial = ProductionTrainer(
        IndexedSplit(cache, "train"),
        IndexedSplit(cache, "validation"),
        config,
        objective,
        plan,
        tmp_path / "partial",
    )
    partial.fit(pause_after_steps=2)
    resumed = ProductionTrainer(
        full["train"],
        full["validation"],
        config,
        objective,
        plan,
        tmp_path / "partial",
        resume=True,
    )
    assert resumed.sampler.offset == 4
    resumed.fit()
    for name, tensor in whole.model.state_dict().items():
        torch.testing.assert_close(
            tensor, resumed.model.state_dict()[name], rtol=0, atol=0
        )
    for left, right in zip(
        whole.optimizer.state.values(), resumed.optimizer.state.values(), strict=True
    ):
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    assert whole.state["input_positions"] == resumed.state["input_positions"] == 96
