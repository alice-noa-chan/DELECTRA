"""Profile ELECTRA flash attention and physical batches on one bounded L40S."""

import json
from pathlib import Path

import target_app
from deletcra.cloud import validate_run_id
from modal_app import GPU, GPU_RATE_PER_SECOND, ROOT, app, image, volume


@app.function(
    image=image.add_local_file(ROOT / "target_app.py", "/root/target_app.py"),
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
def profile_attention(run_id: str) -> dict:
    from deletcra.attention_benchmark import benchmark_attention
    from deletcra.data import load_prepared

    validate_run_id(run_id)
    volume.reload()
    destination = Path("/vol/runs") / run_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    train, _, metadata = load_prepared("/vol/data/tinystories-ref-256-10m-100k")
    result = benchmark_attention(train, tuple(metadata["special_token_ids"]))
    result["run_id"] = run_id
    result["dataset"] = metadata
    result["gpu_only_estimate_usd"] = (
        result["gpu_function_seconds"] * GPU_RATE_PER_SECOND
    )
    destination.mkdir(parents=True)
    (destination / "attention-result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def attention_main(run_id: str) -> None:
    validate_run_id(run_id)
    output = ROOT / "results" / f"{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    target_app.prepare_target.remote()
    result = profile_attention.remote(run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved attention benchmark: {output}")
