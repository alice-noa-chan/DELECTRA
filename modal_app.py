"""Run a bounded WikiText-2 GPU pilot on Modal, persisting all checkpoints.

    python -m modal run modal_app.py --run-id YOUR_UNIQUE_RUN_ID

Only project source/license files are mounted; local credentials are not uploaded.
"""

import json
from pathlib import Path

import modal

from deletcra.cloud import PilotConfig, validate_run_id

ROOT = Path(__file__).resolve().parent
VOLUME_NAME = "deletcra-experiments"
DATA_DIRECTORY = "/vol/data/wikitext2-128-1m-100k"
GPU = "L40S"
GPU_TIMEOUT_SECONDS = 900
GPU_RATE_PER_SECOND = 0.000542  # Modal published rate checked 2026-10-01.

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.57.6", "datasets==4.8.5")
    .env(
        {
            "PYTHONPATH": "/workspace/src",
            "HF_HOME": "/vol/hf-cache",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "LICENSE", "/workspace/LICENSE")
    .add_local_file(ROOT / "NOTICE", "/workspace/NOTICE")
    .add_local_dir(ROOT / "LICENSES", "/workspace/LICENSES")
)
app = modal.App("deletcra-pilot", image=image)


@app.function(
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=4096,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def prepare_data() -> dict:
    """Prepare/cache data on CPU before allocating the paid GPU."""
    from uuid import uuid4

    from deletcra.data import load_prepared, prepare_wikitext

    target = Path(DATA_DIRECTORY)
    if (target / "metadata.json").exists():
        _, _, metadata = load_prepared(target)
        return metadata
    if target.exists():
        raise RuntimeError(f"incomplete cache requires inspection: {target}")
    staging = target.with_name(f"{target.name}-{uuid4().hex}")
    metadata = prepare_wikitext(
        staging,
        allow_download=True,
        sequence_length=128,
        max_train_tokens=1_000_000,
        max_validation_tokens=100_000,
    )
    staging.rename(target)
    volume.commit()
    return metadata


def cuda_checks() -> dict:
    """Check actual CUDA/bf16 execution, future isolation, and gradient finiteness."""
    import gc

    import torch

    from deletcra.config import ModelConfig
    from deletcra.model import CausalElectra
    from deletcra.objectives import ObjectiveConfig, pretraining_step

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 support is required")
    torch.manual_seed(7)
    model = CausalElectra(ModelConfig()).cuda().eval()
    generator = CausalElectra(ModelConfig(hidden_size=16, num_layers=1)).cuda()
    tokens = torch.tensor([[1, 4, 5, 6]], device="cuda")
    altered = torch.tensor([[1, 4, 5, 19]], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first, second = model(tokens), model(altered)
        difference = (first.lm_logits[:, :3] - second.lm_logits[:, :3]).abs().max()
        torch.testing.assert_close(
            first.lm_logits[:, :3], second.lm_logits[:, :3], rtol=0, atol=0
        )
        torch.testing.assert_close(
            first.rtd_logits[:, :3], second.rtd_logits[:, :3], rtol=0, atol=0
        )
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = pretraining_step(
            model, generator, tokens, tokens.ne(0), ObjectiveConfig(mode="joint")
        )
    output.loss.backward()
    for module in (model, generator):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all(
            torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise RuntimeError("CUDA/bf16 gradients are missing or nonfinite")
    result = {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "bf16_supported": True,
        "future_prefix_max_difference": difference.item(),
        "joint_loss": output.loss.item(),
        "gradient_check": "passed",
    }
    del model, generator, output, first, second
    gc.collect()
    torch.cuda.empty_cache()
    return result


@app.function(
    gpu=GPU,
    volumes={"/vol": volume},
    cpu=(2, 2),
    memory=8192,
    timeout=GPU_TIMEOUT_SECONDS,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def run_pilot(run_id: str, config: dict) -> dict:
    import subprocess
    import sys
    import time

    validate_run_id(run_id)
    settings = PilotConfig(**config)
    volume.reload()
    directory = Path("/vol/runs") / run_id
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite run: {directory}")
    started = time.perf_counter()
    checks = cuda_checks()
    command = [sys.executable, *settings.command(DATA_DIRECTORY, str(directory))]
    try:
        subprocess.run(command, check=True, timeout=GPU_TIMEOUT_SECONDS - 60)
        summary = json.loads((directory / "summary.json").read_text())
        reports = {
            mode: json.loads((directory / mode / "report.json").read_text())
            for mode in ("clm", "rtd", "joint")
        }
        elapsed = time.perf_counter() - started
        result = {
            "run_id": run_id,
            "modal_volume": VOLUME_NAME,
            "remote_directory": str(directory),
            "requested_gpu": GPU,
            "cuda_checks": checks,
            "remote_gpu_function_seconds": elapsed,
            "gpu_rate_usd_per_second": GPU_RATE_PER_SECOND,
            "estimated_gpu_charge_usd": elapsed * GPU_RATE_PER_SECOND,
            "cost_note": (
                "Runtime-based GPU estimate, not a billing receipt. Excludes startup, "
                "scaledown, CPU, host memory, image building, storage, and egress. "
                "Published rate checked https://modal.com/pricing on 2026-10-01."
            ),
            "command": command,
            "summary": summary,
            "reports": reports,
        }
        (directory / "modal-result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def main(
    run_id: str,
    steps: int = 100,
    batch_size: int = 32,
    probe_steps: int = 20,
    train_tokens: int | None = None,
) -> None:
    from dataclasses import asdict

    validate_run_id(run_id)
    config = PilotConfig(
        steps=steps,
        batch_size=batch_size,
        probe_steps=probe_steps,
        train_tokens=train_tokens,
    )
    destination = ROOT / "results" / f"{run_id}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite local results: {destination}")
    metadata = prepare_data.remote()
    print(f"Prepared {metadata['train_content_tokens']} training content tokens")
    result = run_pilot.remote(run_id, asdict(config))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved GPU pilot results: {destination}")
    print(json.dumps(result["summary"], indent=2))
