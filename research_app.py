"""Three-seed, token-matched and training-time-matched Modal research matrix."""

import json
from pathlib import Path

from deletcra.cloud import validate_run_id
from deletcra.research import SEEDS, STAGES, research_command
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


@app.function(
    gpu=GPU,
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=8192,
    timeout=900,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def research_seed(run_id: str, seed: int) -> dict:
    import subprocess
    import sys
    import time

    validate_run_id(run_id)
    if seed not in SEEDS:
        raise ValueError("seed must belong to the predeclared matrix")
    volume.reload()
    directory = Path("/vol/runs") / run_id / f"seed-{seed}"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite: {directory}")
    started = time.perf_counter()
    checks = cuda_checks()
    reports, commands = {}, {}
    try:
        for stage in STAGES:
            time_budget = (
                reports["joint"]["training_seconds"] if stage == "clm_time" else None
            )
            command = [
                sys.executable,
                *research_command(
                    seed,
                    stage,
                    DATA_DIRECTORY,
                    str(directory / stage),
                    time_budget=time_budget,
                ),
            ]
            print(
                f"Starting seed={seed}, stage={stage}, time_budget={time_budget}",
                flush=True,
            )
            remaining = 840 - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("research seed exhausted its execution budget")
            subprocess.run(command, check=True, timeout=remaining)
            mode = "joint" if stage == "joint" else "clm"
            reports[stage] = json.loads(
                (directory / stage / mode / "report.json").read_text()
            )
            commands[stage] = command
            volume.commit()
            print(
                f"Finished seed={seed}, stage={stage}: "
                f"PPL={reports[stage]['final_validation']['perplexity']:.3f}, "
                f"seconds={reports[stage]['training_seconds']:.2f}",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        result = {
            "run_id": run_id,
            "seed": seed,
            "cuda_checks": checks,
            "reports": reports,
            "commands": commands,
            "modal_volume": VOLUME_NAME,
            "remote_directory": str(directory),
            "remote_gpu_function_seconds": elapsed,
            "estimated_gpu_charge_usd": elapsed * GPU_RATE_PER_SECOND,
            "cost_note": (
                "GPU runtime estimate, not invoice; excludes startup/scaledown, "
                "CPU, memory, storage and egress."
            ),
            "design_note": (
                "Three predeclared seeds; 10M input-token joint/CLM, CLM time "
                "control capped at 30M tokens and 300 seconds. Full prepared "
                "validation, no test split. Time control is not FLOP matching."
            ),
        }
        (directory / "research-result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result
    finally:
        volume.commit()


@app.function(
    gpu=GPU,
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=8192,
    timeout=180,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def old_checkpoint_diagnostics(run_id: str) -> dict:
    """Evaluate old RTD/joint against one fixed generator and identical corruption."""
    import time

    from deletcra.data import load_prepared
    from deletcra.experiment import TrainConfig, evaluate
    from deletcra.model import CausalElectra
    from deletcra.objectives import ObjectiveConfig

    validate_run_id(run_id)
    volume.reload()
    destination = Path("/vol/runs") / run_id / "diagnostics.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    started = time.perf_counter()
    _, validation, metadata = load_prepared(DATA_DIRECTORY)
    settings = TrainConfig(
        batch_size=32,
        eval_batches=32,
        device="cuda",
        precision="bf16",
        seed=7,
        probe_steps=0,
    )
    import torch

    torch.set_num_threads(2)
    old = Path("/vol/runs/modal-l40s-wikitext2-20261001-10m")
    reference = CausalElectra.load(old / "joint" / "generator").cuda().eval()
    metrics = {}
    for mode in ("rtd", "joint"):
        model = CausalElectra.load(old / mode / "model").cuda().eval()
        metrics[mode] = evaluate(
            model,
            reference,
            validation,
            ObjectiveConfig(mode=mode),
            settings,
            special_token_ids=tuple(metadata["special_token_ids"]),
        )
        del model
    elapsed = time.perf_counter() - started
    result = {
        "metrics": metrics,
        "reference_generator": str(old / "joint" / "generator"),
        "note": (
            "Same frozen joint generator, validation tokens and RNG seed for both "
            "checkpoints. Generator loss refers to this reference. AP baseline "
            "is replacement rate; F1 uses sigmoid threshold 0.5."
        ),
        "remote_gpu_function_seconds": elapsed,
        "estimated_gpu_charge_usd": elapsed * GPU_RATE_PER_SECOND,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def research_main(run_id: str) -> None:
    validate_run_id(run_id)
    destination = ROOT / "results" / f"{run_id}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    prepare_data.remote()
    result = {"run_id": run_id, "seeds": [], "diagnostics": None}
    for seed in SEEDS:
        result["seeds"].append(research_seed.remote(run_id, seed))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["diagnostics"] = old_checkpoint_diagnostics.remote(run_id)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved research matrix: {destination}")
