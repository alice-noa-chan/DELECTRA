"""Portable budget-GPU measurement of the integrated 15M training step."""

import argparse
import gc
import hashlib
import json
import math
import signal
import statistics
import subprocess
import time
from pathlib import Path

import torch

from deletcra.attention_benchmark import check_flash_equivalence, measure_case
from deletcra.data import load_prepared
from deletcra.optimization_benchmark import (
    check_liger_equivalence,
    check_sequential_equivalence,
)
from deletcra.target import REFERENCE_REVISION, STORIES_REVISION

BATCHES = (64, 128, 256)
REPETITIONS = 3
WARMUP = 20
MEASURED = 100
BASE_TARGETS = 16_400_000_000


def validate_device(
    name: str,
    memory: int,
    capability: tuple[int, int],
    expected: str = "RTX A5000",
) -> None:
    """Reject another GPU instead of attaching the requested device's price."""
    if expected not in ("RTX A5000", "RTX 3090"):
        raise ValueError("unsupported budget GPU")
    canonical = name.removeprefix("NVIDIA ").removeprefix("GeForce ")
    if canonical != expected:
        raise ValueError(f"expected {expected}, received {name}")
    if memory < 22 * 2**30 or capability != (8, 6):
        raise ValueError("expected a 24GB Ampere GPU")


def summarize(rows: list[dict], hourly_price: float) -> list[dict]:
    """Use total tokens / total time; incomplete runs cannot support a claim."""
    if not math.isfinite(hourly_price) or hourly_price <= 0:
        raise ValueError("hourly price must be finite and positive")
    if len(rows) != len(BATCHES) * REPETITIONS:
        raise ValueError("summary requires all nine full trials")
    results = []
    for batch in BATCHES:
        trials = [row for row in rows if row["batch_size"] == batch]
        if len(trials) != REPETITIONS or {row["repetition"] for row in trials} != set(
            range(REPETITIONS)
        ):
            raise ValueError("summary requires distinct complete repetitions")
        for row in trials:
            expected = {
                "status": "completed",
                "backend": "flash",
                "mode": "joint",
                "generator_mode": "self",
                "lm_loss_backend": "liger",
                "sequential_backward": True,
                "fused_optimizer": True,
                "context": 256,
                "dropout": 0.1,
                "warmup_steps": WARMUP,
                "measured_steps": MEASURED,
                "measured_input_tokens": batch * 255 * MEASURED,
                "model_parameters": 15_041_505,
                "unique_optimized_parameters": 15_041_505,
                "finite_gradients": "passed",
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("trial differs from the fixed budget GPU protocol")
            seconds = row["training_seconds"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("training time must be finite and positive")
            ops = row["attention_dispatcher_ops"]
            for key in (
                "aten::_scaled_dot_product_flash_attention",
                "aten::_scaled_dot_product_flash_attention_backward",
            ):
                if ops.get(key) != 12:
                    raise ValueError("both joint passes must use native flash")
        rates = [
            row["measured_input_tokens"] / row["training_seconds"] for row in trials
        ]
        rate = sum(row["measured_input_tokens"] for row in trials) / sum(
            row["training_seconds"] for row in trials
        )
        hours = BASE_TARGETS / rate / 3600
        results.append(
            {
                "batch_size": batch,
                "prediction_targets_per_second": rate,
                "repetition_throughputs": rates,
                "throughput_sample_std_percent": statistics.stdev(rates)
                / statistics.mean(rates)
                * 100,
                "peak_cuda_allocated_bytes": max(
                    row["peak_cuda_allocated_bytes"] for row in trials
                ),
                "base_16_4b_targets_hours": hours,
                "base_gpu_only_estimate_usd": hours * hourly_price,
            }
        )
    return results


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hourly-price", type=float, required=True)
    parser.add_argument("--gpu", choices=("RTX A5000", "RTX 3090"), default="RTX A5000")
    args = parser.parse_args()
    if not math.isfinite(args.hourly_price) or args.hourly_price <= 0:
        parser.error("hourly price must be finite and positive")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("commit benchmark code before measurement")
    # Reserve an exclusive destination before GPU work. Completed historical
    # evidence is never overwritten; partial evidence stays readable on failure.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("x", encoding="utf-8")
    result = {
        "status": "started",
        "cases": [],
        "hourly_gpu_price_usd": args.hourly_price,
    }
    started = time.perf_counter()

    def timeout(_signal, _frame):
        raise TimeoutError("GPU benchmark exceeded its 900-second process budget")

    try:
        if not torch.cuda.is_available():
            raise ValueError("budget GPU benchmark requires CUDA")
        device = torch.cuda.get_device_properties(0)
        validate_device(
            device.name,
            device.total_memory,
            torch.cuda.get_device_capability(),
            args.gpu,
        )
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("native BF16 is required")
        if not hasattr(signal, "SIGALRM"):
            raise ValueError("run this bounded GPU benchmark on Linux")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(900)
        train, _, metadata = load_prepared(args.data_dir)
        if (
            metadata["dataset_revision"] != STORIES_REVISION
            or metadata["tokenizer_revision"] != REFERENCE_REVISION
            or metadata["train_content_tokens"] != 9_999_825
            or metadata["vocab_size"] != 32000
            or train.shape != (39_215, 256)
            or train.eq(0).any()
        ):
            raise ValueError("data differs from the pinned 10M TinyStories cache")
        result.update(
            source_commit=source_commit,
            dataset=metadata,
            dataset_sha256={
                name: file_sha256(args.data_dir / name)
                for name in ("metadata.json", "train.pt", "validation.pt")
            },
            environment={
                "gpu": device.name,
                "memory_bytes": device.total_memory,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "capability": torch.cuda.get_device_capability(),
            },
        )
        torch.set_num_threads(2)
        result["checks"] = {
            "flash": check_flash_equivalence(),
            "liger": check_liger_equivalence(),
            "sequential": check_sequential_equivalence(),
        }
        for repetition in range(REPETITIONS):
            # Rotate batch order to reduce a consistent warmup/thermal bias.
            order = BATCHES[repetition:] + BATCHES[:repetition]
            for batch in order:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"{args.gpu} repetition={repetition} batch={batch}", flush=True)
                row = measure_case(
                    train,
                    tuple(metadata["special_token_ids"]),
                    "flash",
                    "joint",
                    batch,
                    warmup_steps=WARMUP,
                    measured_steps=MEASURED,
                    generator_mode="self",
                    lm_loss_backend="liger",
                    sequential_backward=True,
                    fused_optimizer=True,
                )
                row["repetition"] = repetition
                result["cases"].append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
        result["summary"] = summarize(result["cases"], args.hourly_price)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        result["benchmark_process_seconds"] = time.perf_counter() - started
        result["note"] = (
            "Short training-step measurement, not model quality or full training. "
            "Rates count 255 prediction targets per 256-token block, including both "
            "joint passes and optimizer work. Extrapolation uses 16.4B targets "
            "conservatively; it excludes preprocessing, startup, storage, failures, "
            "evaluation, TinyStories completion and IT. No trained model is published."
        )
        with output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
