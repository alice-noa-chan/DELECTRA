"""Predeclared small research matrix and bounded per-stage commands."""

import math

SEEDS = (7, 19, 37)
STAGES = ("joint", "clm", "clm_time")


def research_command(
    seed: int,
    stage: str,
    data_directory: str,
    output_directory: str,
    *,
    time_budget: float | None = None,
) -> list[str]:
    if seed not in SEEDS or stage not in STAGES:
        raise ValueError("research is restricted to predeclared seeds and stages")
    if stage == "clm_time":
        if (
            time_budget is None
            or not math.isfinite(time_budget)
            or not 0 < time_budget <= 300
        ):
            raise ValueError("time control requires a measured budget in (0, 300]")
    elif time_budget is not None:
        raise ValueError("only the time control accepts a time budget")
    return [
        "-m",
        "deletcra",
        "run",
        "--data-dir",
        data_directory,
        "--output-dir",
        output_directory,
        "--mode",
        "joint" if stage == "joint" else "clm",
        "--preset",
        "small",
        "--train-tokens",
        "30000000" if stage == "clm_time" else "10000000",
        "--batch-size",
        "32",
        "--probe-steps",
        "100",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--cpu-threads",
        "2",
        "--eval-batches",
        "32",
        "--quiet",
        *(["--max-training-seconds", str(time_budget)] if stage == "clm_time" else []),
    ]


def ablation_command(
    kind: str, setting: str, data_directory: str, output: str, time_budget: float
) -> list[str]:
    """Build exploratory controls without importing optional cloud dependencies."""
    if kind == "embeddings" and setting in {"separate", "shared"}:
        command = research_command(7, "joint", data_directory, output)
        command[command.index("--mode") + 1] = "rtd"
        if setting == "shared":
            command.append("--share-embeddings")
        return command
    if kind == "regularization" and setting in {"dropout0", "dropout01"}:
        command = research_command(
            7, "clm_time", data_directory, output, time_budget=time_budget
        )
        return [
            *command,
            "--dropout",
            "0.1" if setting == "dropout01" else "0",
            "--eval-every-steps",
            "500",
        ]
    raise ValueError("only predeclared embedding/dropout controls are supported")
