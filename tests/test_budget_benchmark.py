from copy import deepcopy

import pytest

from deletcra.budget_benchmark import BATCHES, summarize, validate_device


def trials():
    return [
        {
            "batch_size": batch,
            "repetition": repetition,
            "status": "completed",
            "backend": "flash",
            "mode": "joint",
            "generator_mode": "self",
            "lm_loss_backend": "liger",
            "sequential_backward": True,
            "fused_optimizer": True,
            "context": 256,
            "dropout": 0.1,
            "warmup_steps": 20,
            "measured_steps": 100,
            "measured_input_tokens": batch * 255 * 100,
            "training_seconds": batch * 255 * 100 / rate,
            "model_parameters": 15_041_505,
            "unique_optimized_parameters": 15_041_505,
            "finite_gradients": "passed",
            "peak_cuda_allocated_bytes": 100,
            "attention_dispatcher_ops": {
                "aten::_scaled_dot_product_flash_attention": 12,
                "aten::_scaled_dot_product_flash_attention_backward": 12,
            },
        }
        for batch in BATCHES
        for repetition, rate in enumerate((100000, 200000, 300000))
    ]


def test_summary_weights_time_and_cost_and_rejects_partial_evidence():
    rows = trials()
    result = summarize(rows, 0.16)
    assert result[0]["prediction_targets_per_second"] == pytest.approx(163636.363636)
    assert result[0]["base_gpu_only_estimate_usd"] == pytest.approx(
        16_400_000_000 / 163636.363636 / 3600 * 0.16
    )
    for key, value in (
        ("repetition", 1),
        ("model_parameters", 15000000),
        ("generator_mode", "separate"),
        ("measured_input_tokens", 1),
        ("training_seconds", float("nan")),
        ("training_seconds", 0),
        ("attention_dispatcher_ops", {}),
    ):
        changed = deepcopy(rows)
        changed[0][key] = value
        with pytest.raises(ValueError):
            summarize(changed, 0.16)
    with pytest.raises(ValueError):
        summarize(rows[:-1], 0.16)


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf")])
def test_invalid_prices_are_rejected(price):
    with pytest.raises(ValueError):
        summarize(trials(), price)


def test_device_guard_does_not_accept_a_different_or_small_gpu():
    validate_device("NVIDIA RTX A5000", 24 * 2**30, (8, 6))
    validate_device("NVIDIA RTX 3090", 24 * 2**30, (8, 6), "RTX 3090")
    validate_device("NVIDIA GeForce RTX 3090", 24 * 2**30, (8, 6), "RTX 3090")
    with pytest.raises(ValueError):
        validate_device("NVIDIA RTX A5000", 24 * 2**30, (8, 6), "RTX 3090")
    for name, memory, capability in (
        ("NVIDIA RTX A5000 Laptop GPU", 24 * 2**30, (8, 6)),
        ("NVIDIA RTX A6000", 48 * 2**30, (8, 6)),
        ("NVIDIA RTX 3090", 24 * 2**30, (8, 6)),
        ("NVIDIA RTX A5000", 16 * 2**30, (8, 6)),
        ("NVIDIA RTX A5000", 24 * 2**30, (7, 5)),
    ):
        with pytest.raises(ValueError):
            validate_device(name, memory, capability)
