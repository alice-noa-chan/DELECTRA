"""Bounded profiling of the real TinyStories/15M target before longer training."""

import json
from pathlib import Path

from deletcra.cloud import validate_run_id
from deletcra.target import profile_command
from modal_app import GPU, GPU_RATE_PER_SECOND, ROOT, app, cuda_checks, volume

TARGET_DATA = "/vol/data/tinystories-ref-256-10m-100k"


@app.function(
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=4096,
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def prepare_target() -> dict:
    from uuid import uuid4

    from deletcra.data import load_prepared, prepare_tinystories

    destination = Path(TARGET_DATA)
    if destination.exists():
        return load_prepared(destination)[2]
    staging = destination.with_name(f"{destination.name}-{uuid4().hex}")
    metadata = prepare_tinystories(
        staging,
        max_train_tokens=10_000_000,
        max_validation_tokens=100_000,
        allow_download=True,
    )
    staging.rename(destination)
    volume.commit()
    return metadata


@app.function(
    gpu=GPU,
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=8192,
    timeout=300,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def profile_target(run_id: str) -> dict:
    import subprocess
    import sys
    import time

    validate_run_id(run_id)
    volume.reload()
    directory = Path("/vol/runs") / run_id
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite: {directory}")
    started = time.perf_counter()
    checks = cuda_checks()
    reports, commands = {}, {}
    try:
        for mode in ("clm", "joint"):
            command = [
                sys.executable,
                *profile_command(mode, TARGET_DATA, str(directory / mode)),
            ]
            remaining = 240 - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("target profiling exhausted its execution budget")
            subprocess.run(command, check=True, timeout=remaining)
            reports[mode] = json.loads(
                (directory / mode / mode / "report.json").read_text()
            )
            commands[mode] = command
            volume.commit()
        elapsed = time.perf_counter() - started
        extrapolation = {}
        for mode, report in reports.items():
            extrapolation[mode] = {
                str(tokens): {
                    "training_hours": tokens / report["input_tokens_per_second"] / 3600,
                    "training_gpu_estimate_usd": tokens
                    / report["input_tokens_per_second"]
                    * GPU_RATE_PER_SECOND,
                }
                for tokens in (100_000_000, 1_000_000_000, 3_000_000_000)
            }
        result = {
            "run_id": run_id,
            "reports": reports,
            "commands": commands,
            "cuda_checks": checks,
            "remote_gpu_function_seconds": elapsed,
            "estimated_gpu_charge_usd": elapsed * GPU_RATE_PER_SECOND,
            "extrapolation": extrapolation,
            "note": (
                "300-step throughput/memory pilot, not target-level training. "
                "Extrapolated longer-run time and GPU cost exclude data preparation, "
                "validation, checkpoints, startup/scaledown, CPU, memory, storage "
                "and egress. Short pilots can overestimate sustained throughput. "
                "100M/1B/3B are proposed input budgets, not known reference budgets "
                "or guarantees of beating TinyLlama."
            ),
        }
        (directory / "target-result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def target_main(run_id: str) -> None:
    validate_run_id(run_id)
    destination = ROOT / "results" / f"{run_id}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    metadata = prepare_target.remote()
    print(f"Prepared {metadata['train_content_tokens']} training tokens", flush=True)
    result = profile_target.remote(run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved target profile: {destination}")
