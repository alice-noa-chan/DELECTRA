from dataclasses import replace

import pytest
import torch

from deletcra.config import ModelConfig
from deletcra.data import synthetic_sequences
from deletcra.objectives import ObjectiveConfig
from deletcra.training import (
    EpochSampler,
    ProductionPlan,
    ProductionTrainer,
    scheduled_lr,
)


class Corpus:
    def __init__(self, tokens, metadata=None):
        self.tokens = tokens
        self.metadata = metadata or {
            "status": "complete",
            "sequence_length": 8,
            "vocab_size": 32,
            "bos_token_id": 1,
            "pad_token_id": 0,
        }

    def __len__(self):
        return len(self.tokens)

    def batch(self, indices):
        return self.tokens[list(indices)]


def test_epoch_coverage_and_sampler_resume():
    sampler = EpochSampler(7, 5)
    first = [*sampler.next(3), *sampler.next(3), *sampler.next(3)]
    assert sorted(first) == list(range(7))
    assert len(set(first)) == 7
    next_batch = sampler.next(3)
    restored = EpochSampler(7, 5, epoch=sampler.epoch, offset=sampler.offset)
    assert len(next_batch) == 3
    assert (sampler.next(3) == restored.next(3)).all()


@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveConfig(mode="clm"),
        ObjectiveConfig(mode="joint", generator_mode="self"),
        ObjectiveConfig(mode="joint", generator_mode="separate"),
    ],
)
def test_interrupted_training_exactly_matches_uninterrupted(tmp_path, objective):
    config = ModelConfig(max_positions=8, dropout=0.1)
    plan = ProductionPlan(
        max_input_positions=128,
        batch_size=2,
        warmup_positions=32,
        cpu_threads=1,
        checkpoint_every=2,
        evaluate_every=2,
        validation_blocks=4,
        sequential_backward=objective.generator_mode == "self",
    )
    train = Corpus(synthetic_sequences(7, 8, 32, seed=9))
    validation = Corpus(synthetic_sequences(4, 8, 32, seed=10))
    whole = ProductionTrainer(
        train, validation, config, objective, plan, tmp_path / "whole"
    )
    whole.fit()
    partial = ProductionTrainer(
        train, validation, config, objective, plan, tmp_path / "resume"
    )
    partial.fit(pause_after_steps=3)
    resumed = ProductionTrainer(
        train, validation, config, objective, plan, tmp_path / "resume", resume=True
    )
    resumed.fit()
    assert resumed.state["input_positions"] == whole.state["input_positions"] == 128
    assert resumed.state["step"] == whole.state["step"]
    for key, value in whole.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )
    if whole.generator is not None:
        for key, value in whole.generator.state_dict().items():
            torch.testing.assert_close(
                value, resumed.generator.state_dict()[key], rtol=0, atol=0
            )
    for left, right in zip(
        whole.optimizer.state.values(), resumed.optimizer.state.values(), strict=True
    ):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    changed = replace(plan, learning_rate=1e-4)
    with pytest.raises(ValueError, match="resume configuration"):
        ProductionTrainer(
            train,
            validation,
            config,
            objective,
            changed,
            tmp_path / "resume",
            resume=True,
        )


def test_schedule_and_incomplete_corpus_guard(tmp_path):
    plan = ProductionPlan(max_input_positions=100, batch_size=1, warmup_positions=20)
    assert scheduled_lr(plan, 0, 10) == pytest.approx(plan.learning_rate / 2)
    assert scheduled_lr(plan, 10, 10) == plan.learning_rate
    assert scheduled_lr(plan, 90, 10) == pytest.approx(plan.learning_rate * 0.1)
    data = Corpus(synthetic_sequences(4, 8, 32, seed=7))
    data.metadata["status"] = "preparing"
    with pytest.raises(ValueError, match="not complete"):
        ProductionTrainer(
            data, data, ModelConfig(), ObjectiveConfig(), plan, tmp_path / "run"
        )


def test_checkpoint_replacement_failure_preserves_latest_and_previous(
    tmp_path, monkeypatch
):
    from deletcra.corpus import atomic_replace

    data = Corpus(synthetic_sequences(4, 8, 32, seed=7))
    trainer = ProductionTrainer(
        data,
        data,
        ModelConfig(max_positions=8),
        ObjectiveConfig(mode="clm"),
        ProductionPlan(max_input_positions=64, warmup_positions=0),
        tmp_path / "run",
    )
    latest = trainer.directory / "latest.pt"
    old_bytes = latest.read_bytes()

    def failed_replacement(source, target):
        if target.name == "latest.pt":
            raise PermissionError("persistent lock")
        atomic_replace(source, target)

    monkeypatch.setattr("deletcra.training.atomic_replace", failed_replacement)
    trainer.state["step"] = 1
    with pytest.raises(PermissionError, match="persistent"):
        trainer.save()
    assert latest.read_bytes() == old_bytes
    assert latest.with_suffix(".previous.pt").read_bytes() == old_bytes
    saved = torch.load(latest, weights_only=True)
    assert saved["state"]["step"] == 0
