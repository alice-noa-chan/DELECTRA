"""Bounded seed-7 controls for embedding sharing and CLM overfitting."""

import json
from pathlib import Path

from deletcra.cloud import validate_run_id
from deletcra.research import research_command
from modal_app import (
    DATA_DIRECTORY,
    GPU,
    GPU_RATE_PER_SECOND,
    ROOT,
    VOLUME_NAME,
    app,
    cuda_checks,
    prepare_data,
    volume,
)

MATRIX_ID = "modal-l40s-wikitext2-20261001-research"


def ablation_command(
    kind: str, setting: str, output: str, time_budget: float
) -> list[str]:
    if kind == "embeddings" and setting in {"separate", "shared"}:
        command = research_command(7, "joint", DATA_DIRECTORY, output)
        command[command.index("--mode") + 1] = "rtd"
        if setting == "shared":
            command.append("--share-embeddings")
        return command
    if kind == "regularization" and setting in {"dropout0", "dropout01"}:
        command = research_command(
            7, "clm_time", DATA_DIRECTORY, output, time_budget=time_budget
        )
        return [
            *command,
            "--dropout",
            "0.1" if setting == "dropout01" else "0",
            "--eval-every-steps",
            "500",
        ]
    raise ValueError("only predeclared embedding/dropout controls are supported")


@app.function(
    gpu=GPU,
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=8192,
    timeout=600,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def run_ablation(run_id: str, kind: str) -> dict:
    import subprocess
    import sys
    import time

    import torch

    from deletcra.data import load_prepared
    from deletcra.experiment import TrainConfig, evaluate
    from deletcra.model import CausalElectra
    from deletcra.objectives import ObjectiveConfig

    validate_run_id(run_id)
    if kind not in {"embeddings", "regularization"}:
        raise ValueError("unsupported ablation kind")
    volume.reload()
    directory = Path("/vol/runs") / run_id / kind
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite: {directory}")
    paired = Path("/vol/runs") / MATRIX_ID / "seed-7/joint/joint/report.json"
    time_budget = json.loads(paired.read_text())["training_seconds"]
    # Validate bounds before CUDA allocation checks or creating output directories.
    settings_list = (
        ("separate", "shared") if kind == "embeddings" else ("dropout0", "dropout01")
    )
    commands = {
        setting: [
            sys.executable,
            *ablation_command(kind, setting, str(directory / setting), time_budget),
        ]
        for setting in settings_list
    }
    started = time.perf_counter()
    checks = cuda_checks()
    reports = {}
    try:
        for setting, command in commands.items():
            remaining = 540 - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("ablation exhausted its execution budget")
            print(f"Starting {kind}/{setting}", flush=True)
            subprocess.run(command, check=True, timeout=remaining)
            mode = "rtd" if kind == "embeddings" else "clm"
            reports[setting] = json.loads(
                (directory / setting / mode / "report.json").read_text()
            )
            volume.commit()
            print(f"Completed {kind}/{setting}", flush=True)
        diagnostics = {}
        if kind == "embeddings":
            torch.set_num_threads(2)
            _, validation, metadata = load_prepared(DATA_DIRECTORY)
            reference_path = Path(
                "/vol/runs/modal-l40s-wikitext2-20261001-10m/joint/generator"
            )
            reference = CausalElectra.load(reference_path).cuda().eval()
            settings = TrainConfig(
                batch_size=32,
                eval_batches=32,
                device="cuda",
                precision="bf16",
                seed=7,
                probe_steps=0,
            )
            for setting in settings_list:
                model = (
                    CausalElectra.load(directory / setting / "rtd/model").cuda().eval()
                )
                diagnostics[setting] = evaluate(
                    model,
                    reference,
                    validation,
                    ObjectiveConfig(mode="rtd"),
                    settings,
                    special_token_ids=tuple(metadata["special_token_ids"]),
                )
                del model
        elapsed = time.perf_counter() - started
        result = {
            "run_id": run_id,
            "kind": kind,
            "seed": 7,
            "reports": reports,
            "commands": commands,
            "cuda_checks": checks,
            "fixed_generator_diagnostics": diagnostics,
            "remote_gpu_function_seconds": elapsed,
            "estimated_gpu_charge_usd": elapsed * GPU_RATE_PER_SECOND,
            "modal_volume": VOLUME_NAME,
            "remote_directory": str(directory),
            "note": (
                "Exploratory single-seed follow-up selected after observing the "
                "matrix seed-7 result. Separate/shared RTD uses 10M tokens; dropout "
                "CLM controls match seed-7 joint training time and evaluate every "
                "500 steps. Best validation checkpoints are selected, not test "
                "results. Final probes use final backbones. GPU estimate excludes "
                "startup/scaledown, CPU, memory, storage, and egress."
            ),
        }
        (directory / "ablation-result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def ablation_main(run_id: str) -> None:
    validate_run_id(run_id)
    destination = ROOT / "results" / f"{run_id}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    prepare_data.remote()
    result = {"run_id": run_id, "ablations": []}
    for kind in ("embeddings", "regularization"):
        result["ablations"].append(run_ablation.remote(run_id, kind))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved ablations: {destination}")
