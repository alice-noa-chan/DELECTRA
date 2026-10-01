"""Bounded, serializable settings for a short Modal GPU pilot."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PilotConfig:
    steps: int = 100
    batch_size: int = 32
    probe_steps: int = 20
    seed: int = 7

    def __post_init__(self) -> None:
        for name, maximum in (("steps", 500), ("batch_size", 128)):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not (1 <= value <= maximum)
            ):
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        if not isinstance(self.probe_steps, int) or not 0 <= self.probe_steps <= 100:
            raise ValueError("probe_steps must be an integer in [0, 100]")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")

    def command(self, data_directory: str, output_directory: str) -> list[str]:
        return [
            "-m",
            "deletcra",
            "run",
            "--data-dir",
            data_directory,
            "--output-dir",
            output_directory,
            "--mode",
            "all",
            "--preset",
            "small",
            "--steps",
            str(self.steps),
            "--batch-size",
            str(self.batch_size),
            "--probe-steps",
            str(self.probe_steps),
            "--seed",
            str(self.seed),
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--cpu-threads",
            "2",
            "--eval-batches",
            "4",
        ]


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError(
            "run_id must be a short alphanumeric, underscore, or hyphen ID"
        )
