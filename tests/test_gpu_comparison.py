import pytest

from deletcra.gpu_comparison import (
    GPU_RATES,
    MEASURED,
    REPETITIONS,
    WARMUP,
    summarize_comparison,
    validate_device,
)


def report(gpu, throughput):
    return {
        "requested_gpu": gpu,
        "status": "completed",
        "environment": {"gpu": gpu},
        "cases": [
            {
                "mode": mode,
                "status": "completed",
                "batch_size": 64,
                "context": 256,
                "measured_steps": MEASURED,
                "warmup_steps": WARMUP,
                "backend": "flash",
                "measured_input_tokens": MEASURED * 64 * 255,
                "training_seconds": MEASURED * 64 * 255 / throughput,
                "input_tokens_per_second": throughput,
                "peak_cuda_allocated_bytes": 123,
            }
            for mode in ("clm", "joint")
            for _ in range(REPETITIONS)
        ],
    }


def test_token_cost_ranking_respects_price_instead_of_raw_speed():
    reports = [
        report("L40S", 100000),
        report("H100!", 200000),
        report("RTX-PRO-6000", 200000),
        {"requested_gpu": "A100-80GB", "status": "failed", "error": "no kernel"},
    ]
    summary = summarize_comparison(reports)
    assert summary["base_gpu_cost_ranking"] == ["RTX-PRO-6000", "L40S", "H100!"]
    assert summary["failed_requests"] == [{"gpu": "A100-80GB", "error": "no kernel"}]
    row = summary["measurements"][0]
    assert row["gpu_usd_per_million_tokens"] == pytest.approx(
        1e6 / 100000 * GPU_RATES["L40S"]
    )
    assert row["base_16_4b_training_hours"] == pytest.approx(16.4e9 / 100000 / 3600)


def test_cost_comparison_rejects_partial_or_changed_batch_protocol():
    partial = report("L40S", 100000)
    partial["cases"].pop()
    with pytest.raises(ValueError, match="full fixed protocol"):
        summarize_comparison([partial])
    changed = report("L40S", 100000)
    changed["cases"][0]["batch_size"] = 128
    with pytest.raises(ValueError, match="full fixed protocol"):
        summarize_comparison([changed])


def test_exact_gpu_validation_rejects_free_upgrades_and_wrong_a100_memory():
    validate_device("H100!", "NVIDIA H100 80GB HBM3", 80 * 1024**3)
    with pytest.raises(ValueError, match="differs"):
        validate_device("H100!", "NVIDIA H200", 140 * 1024**3)
    with pytest.raises(ValueError, match="differs"):
        validate_device("A100-80GB", "NVIDIA A100-SXM4-40GB", 40 * 1024**3)
