"""Compare real training throughput and token cost at one fixed physical batch."""

import gc
import statistics
import time

import torch
from torch import Tensor

from deletcra.attention_benchmark import check_flash_equivalence, measure_case

# Official Modal rates checked 2026-10-01. Exact requests avoid GPU fallbacks;
# H100! disables the otherwise possible free H200 upgrade for fair benchmarking.
GPU_RATES = {
    "L40S": 0.000542,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100!": 0.001097,
}
REQUESTED_HOST_RATE = 2 * 0.0000131 + 8 * 0.00000222
BATCH_SIZE = 64
REPETITIONS = 3
WARMUP = 20
MEASURED = 200
BASE_INPUT_TOKENS = 16_400_000_000


def validate_device(requested: str, name: str, memory_bytes: int) -> None:
    """Do not label an upgraded or unrelated GPU as the requested hardware."""
    if requested not in GPU_RATES:
        raise ValueError("unsupported GPU comparison request")
    fragment = {
        "L40S": "L40S",
        "A100-80GB": "A100",
        "RTX-PRO-6000": "RTX PRO 6000",
        "H100!": "H100",
    }[requested]
    if fragment not in name or (
        requested == "A100-80GB" and memory_bytes < 70 * 1024**3
    ):
        raise ValueError(f"GPU differs from exact request {requested}: {name}")


def summarize_comparison(
    reports: list[dict], *, joint_confirmation: bool = False
) -> dict:
    """Rank complete repeated measurements by cost, keeping failures visible.

    Use total measured tokens divided by total measured optimizer time. Host
    charges below are estimates for requested resources, not metered invoices.
    """
    rows = []
    for report in reports:
        if report["status"] != "completed":
            continue
        requested = report["requested_gpu"]
        for mode in ("joint",) if joint_confirmation else ("clm", "joint"):
            cases = [x for x in report["cases"] if x["mode"] == mode]
            if len(cases) != REPETITIONS or any(
                x["status"] != "completed"
                or x["batch_size"] != BATCH_SIZE
                or x["context"] != 256
                or x["measured_steps"] != (1000 if joint_confirmation else MEASURED)
                or x["warmup_steps"] != WARMUP
                or x["backend"] != "flash"
                for x in cases
            ):
                raise ValueError("cost comparison requires the full fixed protocol")
            tokens = sum(x["measured_input_tokens"] for x in cases)
            seconds = sum(x["training_seconds"] for x in cases)
            throughput = tokens / seconds
            rates = [x["input_tokens_per_second"] for x in cases]
            gpu_rate = GPU_RATES[requested]
            rows.append(
                {
                    "requested_gpu": requested,
                    "actual_gpu": report["environment"]["gpu"],
                    "mode": mode,
                    "input_tokens_per_second": throughput,
                    "repetition_throughputs": rates,
                    "throughput_sample_std_percent": statistics.stdev(rates)
                    / statistics.mean(rates)
                    * 100,
                    "peak_cuda_allocated_bytes": max(
                        x["peak_cuda_allocated_bytes"] for x in cases
                    ),
                    "gpu_usd_per_million_tokens": 1e6 / throughput * gpu_rate,
                    "requested_compute_usd_per_million_tokens": 1e6
                    / throughput
                    * (gpu_rate + REQUESTED_HOST_RATE),
                    "base_16_4b_training_hours": BASE_INPUT_TOKENS / throughput / 3600,
                    "base_16_4b_gpu_usd": BASE_INPUT_TOKENS / throughput * gpu_rate,
                    "gpu_usd_per_second": gpu_rate,
                }
            )
    joint = sorted(
        [x for x in rows if x["mode"] == "joint"],
        key=lambda x: x["gpu_usd_per_million_tokens"],
    )
    return {
        "measurements": rows,
        "base_gpu_cost_ranking": [x["requested_gpu"] for x in joint],
        "failed_requests": [
            {"gpu": x["requested_gpu"], "error": x["error"]}
            for x in reports
            if x["status"] != "completed"
        ],
        "pricing_source": "https://modal.com/pricing",
        "note": (
            "Identical physical batch/context/objective; ranking is throughput "
            "economics, not quality. Training projections exclude preparation, "
            "validation, saving, startup/scaledown, storage and egress. Requested "
            "CPU/host-memory cost is not actual metered resource use. No full "
            "training or publication occurs in this benchmark."
        ),
    }


def benchmark_gpu(
    train: Tensor,
    special_ids: tuple[int, ...],
    requested: str,
    *,
    joint_confirmation: bool = False,
) -> dict:
    """Run six fresh-model cases, alternating mode order across repetitions."""
    started = time.perf_counter()
    report = {
        "requested_gpu": requested,
        "status": "failed",
        "cases": [],
        "protocol": {
            "joint_confirmation": joint_confirmation,
            "measured_steps": 1000 if joint_confirmation else MEASURED,
            "warmup_steps": WARMUP,
            "repetitions": REPETITIONS,
            "batch_size": BATCH_SIZE,
        },
    }
    try:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is required for GPU comparison")
        properties = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "gpu": properties.name,
            "memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        validate_device(requested, properties.name, properties.total_memory)
        if train.ndim != 2 or train.shape[1] != 256 or train.eq(0).any():
            raise ValueError("comparison needs unpadded 256-token cached blocks")
        torch.set_num_threads(2)
        report["checks"] = check_flash_equivalence()
        gc.collect()
        torch.cuda.empty_cache()
        for repetition in range(REPETITIONS):
            modes = ("clm", "joint") if repetition % 2 == 0 else ("joint", "clm")
            if joint_confirmation:
                modes = ("joint",)
            for mode in modes:
                if time.perf_counter() - started > 450:
                    raise TimeoutError("GPU comparison exceeded its execution budget")
                print(f"{requested}: repetition {repetition + 1}/3, {mode}", flush=True)
                case = measure_case(
                    train,
                    special_ids,
                    "flash",
                    mode,
                    BATCH_SIZE,
                    warmup_steps=WARMUP,
                    measured_steps=1000 if joint_confirmation else MEASURED,
                )
                case["repetition"] = repetition + 1
                report["cases"].append(case)
                gc.collect()
                torch.cuda.empty_cache()
        report["status"] = "completed"
    except Exception as error:
        # Preserve failures (including unsupported Flash kernels) rather than
        # silently switching backend and misreporting comparable performance.
        report["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
    report["benchmark_seconds"] = time.perf_counter() - started
    report["gpu_only_benchmark_estimate_usd"] = (
        report["benchmark_seconds"] * GPU_RATES[requested]
    )
    return report
