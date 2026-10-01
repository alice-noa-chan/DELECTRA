from copy import deepcopy

import pytest

from deletcra.optimization_benchmark import (
    CASES,
    FOLLOW_UP_CASES,
    FOLLOW_UP_MEASURED,
    MEASURED,
    REPETITIONS,
    WARMUP,
    summarize_cases,
)


def complete_trials(*, follow_up=False):
    measured = FOLLOW_UP_MEASURED if follow_up else MEASURED
    return [
        {
            "case": name,
            "status": "completed",
            "batch_size": batch,
            "generator_mode": generator,
            "lm_loss_backend": loss,
            "sequential_backward": sequential,
            "fused_optimizer": fused,
            "backend": "flash",
            "mode": "joint",
            "context": 256,
            "measured_steps": measured,
            "warmup_steps": WARMUP,
            "measured_input_tokens": measured * batch * 255,
            "training_seconds": measured * batch * 255 / rate,
            "input_tokens_per_second": rate,
            "peak_cuda_allocated_bytes": 123,
        }
        for name, generator, loss, sequential, fused, batch in (
            FOLLOW_UP_CASES if follow_up else CASES
        )
        for rate in (100000, 200000, 300000)
    ]


def test_summary_uses_total_time_and_rejects_incomplete_or_changed_protocol():
    trials = complete_trials()
    result = summarize_cases(trials)
    assert result[0]["input_tokens_per_second"] == pytest.approx(163636.363636)
    for key, value in (
        ("batch_size", 128),
        ("generator_mode", "self"),
        ("lm_loss_backend", "liger"),
        ("sequential_backward", True),
        ("measured_steps", 20),
    ):
        changed = deepcopy(trials)
        changed[0][key] = value
        with pytest.raises(ValueError, match="all full trials"):
            summarize_cases(changed)
    with pytest.raises(ValueError, match="all full trials"):
        summarize_cases(trials[:-1])
    assert REPETITIONS == 3


def test_follow_up_requires_its_own_complete_protocol_and_control():
    follow_up = complete_trials(follow_up=True)
    assert len(summarize_cases(follow_up, follow_up=True)) == 3
    for rows, flag in ((follow_up, False), (complete_trials(), True)):
        with pytest.raises(ValueError, match="all full trials"):
            summarize_cases(rows, follow_up=flag)
