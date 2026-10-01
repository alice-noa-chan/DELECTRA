from copy import deepcopy

import pytest

from deletcra.optimization_benchmark import (
    CASES,
    MEASURED,
    REPETITIONS,
    WARMUP,
    summarize_cases,
)


def complete_trials():
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
            "measured_steps": MEASURED,
            "warmup_steps": WARMUP,
            "measured_input_tokens": MEASURED * batch * 255,
            "training_seconds": MEASURED * batch * 255 / rate,
            "input_tokens_per_second": rate,
            "peak_cuda_allocated_bytes": 123,
        }
        for name, generator, loss, sequential, fused, batch in CASES
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
