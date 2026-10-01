"""Run a bounded four-GPU cost comparison; never train or publish release models."""

import json
import subprocess
import time
from pathlib import Path

import modal

from deletcra.cloud import validate_run_id
from deletcra.gpu_comparison import GPU_RATES, summarize_comparison
from modal_app import ROOT, image, volume

app = modal.App("deletcra-gpu-comparison", image=image)
RESOURCES = {
    "volumes": {"/vol": volume},
    "cpu": (2, 2),
    "memory": 8192,
    "timeout": 600,
    "startup_timeout": 300,
    "retries": 0,
    "max_containers": 1,
    "scaledown_window": 2,
}


def profile(
    requested: str, run_id: str, source_commit: str, joint_confirmation: bool
) -> dict:
    """Persist each GPU's results separately so completed evidence survives failures."""
    from deletcra.data import load_prepared
    from deletcra.gpu_comparison import benchmark_gpu
    from deletcra.target import REFERENCE_REVISION, STORIES_REVISION

    validate_run_id(run_id)
    volume.reload()
    destination = Path("/vol/runs") / run_id / requested.replace("!", "")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    started = time.perf_counter()
    train, _, metadata = load_prepared("/vol/data/tinystories-ref-256-10m-100k")
    if (
        metadata["dataset_revision"] != STORIES_REVISION
        or metadata["tokenizer_revision"] != REFERENCE_REVISION
    ):
        raise ValueError("cached data differs from the pinned target protocol")
    result = benchmark_gpu(
        train,
        tuple(metadata["special_token_ids"]),
        requested,
        joint_confirmation=joint_confirmation,
    )
    result.update(
        run_id=run_id,
        source_commit=source_commit,
        dataset=metadata,
        gpu_function_seconds=time.perf_counter() - started,
    )
    result["gpu_only_function_estimate_usd"] = (
        result["gpu_function_seconds"] * GPU_RATES[requested]
    )
    destination.mkdir(parents=True)
    (destination / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.function(gpu="L40S", **RESOURCES)
def l40s(run_id: str, source_commit: str, joint_confirmation: bool = False) -> dict:
    return profile("L40S", run_id, source_commit, joint_confirmation)


@app.function(gpu="A100-80GB", **RESOURCES)
def a100(run_id: str, source_commit: str, joint_confirmation: bool = False) -> dict:
    return profile("A100-80GB", run_id, source_commit, joint_confirmation)


@app.function(gpu="RTX-PRO-6000", **RESOURCES)
def rtx_pro(run_id: str, source_commit: str, joint_confirmation: bool = False) -> dict:
    return profile("RTX-PRO-6000", run_id, source_commit, joint_confirmation)


@app.function(gpu="H100!", **RESOURCES)
def h100(run_id: str, source_commit: str, joint_confirmation: bool = False) -> dict:
    return profile("H100!", run_id, source_commit, joint_confirmation)


@app.local_entrypoint()
def compare_main(run_id: str) -> None:
    run_comparison(
        run_id,
        (
            ("L40S", l40s),
            ("A100-80GB", a100),
            ("RTX-PRO-6000", rtx_pro),
            ("H100!", h100),
        ),
        joint_confirmation=False,
    )


@app.local_entrypoint()
def confirm_main(run_id: str) -> None:
    """Resolve close RTX/H100 economics with three 1000-step joint trials."""
    run_comparison(
        run_id, (("RTX-PRO-6000", rtx_pro), ("H100!", h100)), joint_confirmation=True
    )


def run_comparison(run_id: str, functions: tuple, *, joint_confirmation: bool) -> None:
    validate_run_id(run_id)
    destination = ROOT / "results" / f"{run_id}.json"
    if destination.exists() or list(ROOT.glob(f"results/{run_id}-*.json")):
        raise FileExistsError("comparison outputs already exist; use a new run ID")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    # Independent workers read the same cached data and write different folders.
    # Each has one container, zero retries and a fixed timeout; no fallback GPU.
    calls = [
        (requested, function.spawn(run_id, source_commit, joint_confirmation))
        for requested, function in functions
    ]
    reports = []
    for requested, call in calls:
        try:
            result = call.get()
        except Exception as error:
            result = {
                "requested_gpu": requested,
                "status": "failed",
                "error": f"{type(error).__name__}: {str(error)[:2000]}",
                "source_commit": source_commit,
            }
        reports.append(result)
        path = destination.with_name(
            f"{run_id}-{requested.replace('!', '').lower()}.json"
        )
        path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Saved {requested}: {result['status']}", flush=True)
    summary = summarize_comparison(reports, joint_confirmation=joint_confirmation)
    summary.update(run_id=run_id, source_commit=source_commit)
    destination.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved comparison: {destination}", flush=True)
