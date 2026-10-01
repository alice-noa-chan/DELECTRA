"""Run one bounded RTX PRO 6000 optimization experiment on cached TinyStories."""

import json
import subprocess
import time
from pathlib import Path

import modal

from deletcra.cloud import validate_run_id
from deletcra.gpu_comparison import GPU_RATES, validate_device
from modal_app import ROOT, volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.6",
        "datasets==4.8.5",
        "liger-kernel==0.8.4",
    )
    .env({"PYTHONPATH": "/workspace/src", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "LICENSE", "/workspace/LICENSE")
    .add_local_file(ROOT / "NOTICE", "/workspace/NOTICE")
    .add_local_dir(ROOT / "LICENSES", "/workspace/LICENSES")
    .add_local_file(ROOT / "modal_app.py", "/root/modal_app.py")
)
app = modal.App("deletcra-integration-optimizations", image=image)


@app.function(
    gpu="RTX-PRO-6000",
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=8192,
    timeout=600,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def profile(run_id: str, source_commit: str) -> dict:
    import torch

    from deletcra.data import load_prepared
    from deletcra.optimization_benchmark import benchmark_optimizations
    from deletcra.target import REFERENCE_REVISION, STORIES_REVISION

    validate_run_id(run_id)
    validate_device(
        "RTX-PRO-6000",
        torch.cuda.get_device_name(),
        torch.cuda.get_device_properties(0).total_memory,
    )
    volume.reload()
    destination = Path("/vol/runs") / run_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    started = time.perf_counter()
    train, _, metadata = load_prepared("/vol/data/tinystories-ref-256-10m-100k")
    if (
        metadata["dataset_revision"] != STORIES_REVISION
        or metadata["tokenizer_revision"] != REFERENCE_REVISION
    ):
        raise ValueError("cached data differs from the pinned protocol")
    result = benchmark_optimizations(train, tuple(metadata["special_token_ids"]))
    seconds = time.perf_counter() - started
    result.update(
        run_id=run_id,
        source_commit=source_commit,
        dataset=metadata,
        gpu_function_seconds=seconds,
        gpu_only_function_estimate_usd=seconds * GPU_RATES["RTX-PRO-6000"],
    )
    destination.mkdir(parents=True)
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(run_id: str) -> None:
    validate_run_id(run_id)
    destination = ROOT / "results" / f"{run_id}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("commit benchmark code before allocating the GPU")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = profile.remote(run_id, source_commit)
    destination.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved optimization evidence: {destination}", flush=True)
