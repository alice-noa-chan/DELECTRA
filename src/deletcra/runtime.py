"""Device execution, mixed precision and portable checkpoint tensor storage."""

from contextlib import nullcontext

import torch


def cpu_tree(value):
    """Materialize checkpoint tensors on CPU; keep nested optimizer containers."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


class TrainingRuntime:
    """Single-device operations; imports XLA only when explicitly requested."""

    def __init__(self, device: str, precision: str, seed: int):
        self.kind, self.precision = device, precision
        self.xla = None
        if device == "xla":
            import torch_xla
            import torch_xla.runtime as xr

            if xr.device_type() != "TPU":
                raise ValueError("XLA training requires an actual TPU backend")
            self.xla = torch_xla
            self.device = torch_xla.device()
            torch_xla.manual_seed(seed)
        else:
            self.device = torch.device(device)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "fp16", init_scale=1024
        )

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(self.kind, dtype=dtype)

    def backward(self, loss):
        # Both clean CLM and corrupted RTD must use this same scale before the
        # shared gradients are unscaled and clipped exactly once.
        self.scaler.scale(loss).backward()

    def step(self, optimizer, parameters, max_grad_norm: float) -> bool:
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=self.precision != "fp16"
        )
        old_scale = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        self.synchronize()
        return self.scaler.get_scale() >= old_scale

    def synchronize(self):
        if self.xla is not None:
            self.xla.sync(wait=True)

    def rng_state(self):
        if self.xla is None:
            return None
        import torch_xla.core.xla_model as xm

        # Reading the seed must not introduce a new XLA step: sync advances
        # the device RNG even when there are no pending random operations.
        # The checkpoint writer synchronizes before taking this snapshot.
        return xm.get_rng_state(self.device)

    def restore_rng(self, state):
        if self.xla is not None:
            import torch_xla.core.xla_model as xm

            if state is None:
                raise ValueError("XLA resume requires saved device RNG state")
            # Drain model/optimizer loading before restoring the next-step seed.
            self.synchronize()
            xm.set_rng_state(state, self.device)
