"""Standalone accelerator checks; does not rent hardware or train DELECTRA.

Colab CLI can execute this file before the project is installed on the VM.
Only package versions and hardware properties are reported, never credentials.
"""

import importlib.metadata
import json
import os
import platform
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch


def probe_runtime(requested: str) -> dict:
    """Require the requested accelerator and check actual forward/backward work.

    This is a small deterministic matrix check, not a throughput benchmark or
    evidence that DELECTRA's RTD sampling and production trainer support XLA.
    """
    if requested not in {"cpu", "cuda", "xla"}:
        raise ValueError("requested device must be cpu, cuda, or xla")
    accelerator = {}

    def synchronize():
        pass

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        accelerator = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "device_count": torch.cuda.device_count(),
        }
        # Exclude CUDA's optional BF16 emulation: a free T4 needs FP16 instead.
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        synchronize = torch.cuda.synchronize
    elif requested == "xla":
        import torch_xla
        import torch_xla.runtime as xr

        if xr.device_type() != "TPU":
            raise RuntimeError("XLA was requested but its backend is not TPU")
        device = torch_xla.device()
        accelerator = {"backend": "TPU", "device_count": torch_xla.device_count()}
        native_bf16 = True

        def synchronize():
            torch_xla.sync(wait=True)
    else:
        device = torch.device("cpu")
        native_bf16 = False

    checks = {}
    precisions = ["fp32"]
    if requested == "cuda":
        precisions.append("fp16")
    if native_bf16:
        precisions.append("bf16")
    for precision in precisions:
        # [32,32] ones @ identity gives ones, loss=1, and dloss/dweight=1/16.
        # The reference gradient is independent of accelerator RNG behavior.
        inputs = torch.ones(32, 32, device=device)
        weights = torch.eye(32, device=device, requires_grad=True)
        expected_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]
        context = (
            torch.autocast(device.type, dtype=expected_dtype)
            if precision != "fp32"
            else nullcontext()
        )
        with context:
            output = inputs @ weights
            loss = output.float().square().mean()
        loss.backward()
        synchronize()
        if output.dtype != expected_dtype:
            raise RuntimeError(f"{precision} matmul did not use {expected_dtype}")
        torch.testing.assert_close(loss.detach().cpu(), torch.tensor(1.0))
        torch.testing.assert_close(weights.grad.cpu(), torch.full((32, 32), 1 / 16))
        checks[precision] = {"status": "passed", "matmul_dtype": str(output.dtype)}

    versions = {}
    for package in ("torch", "torch-xla", "transformers", "google-colab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "status": "passed",
        "scope": "accelerator_forward_backward_only",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "system": platform.system(),
        "requested_device": requested,
        "actual_device": str(device),
        "accelerator": accelerator,
        "native_bf16": native_bf16,
        "checks": checks,
        "packages": versions,
        "delectra_training_verified": False,
    }


def main():
    """Colab uses environment flags because exec does not forward script argv."""
    requested = os.environ.get("DELECTRA_PROBE_DEVICE", "cpu")
    try:
        report = probe_runtime(requested)
    except Exception as error:
        report = {
            "status": "failed",
            "requested_device": requested,
            "error_type": type(error).__name__,
            "delectra_training_verified": False,
        }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    destination = os.environ.get("DELECTRA_PROBE_OUTPUT")
    if destination:
        Path(destination).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
