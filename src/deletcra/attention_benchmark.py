"""Bounded real-objective benchmarks for attention kernels and physical batches."""

import gc
import time
from dataclasses import replace

import torch
from torch import Tensor

from deletcra.config import generator_model_config
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, pretraining_step
from deletcra.target import story_model_config

BACKENDS = ("eager", "flash")
BATCH_SIZES = (16, 32, 64, 128, 192, 256)
WARMUP_STEPS = 5
MEASURED_STEPS = 20


def check_flash_equivalence() -> dict:
    """Check actual BF16 flash outputs against eager and prove prefix isolation."""
    torch.manual_seed(7)
    eager = CausalElectra(story_model_config()).cuda().eval()
    flash = (
        CausalElectra(replace(story_model_config(), attention_backend="flash"))
        .cuda()
        .eval()
    )
    flash.load_state_dict(eager.state_dict(), strict=True)
    tokens = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]], device="cuda")
    changed = tokens.clone()
    changed[:, -1] = 19
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline, actual, future = eager(tokens), flash(tokens), flash(changed)
    differences = {}
    for name in ("hidden_states", "lm_logits", "rtd_logits"):
        original, optimized = getattr(baseline, name), getattr(actual, name)
        torch.testing.assert_close(original, optimized, rtol=0.03, atol=0.02)
        torch.testing.assert_close(
            optimized[:, :-1], getattr(future, name)[:, :-1], rtol=0, atol=0
        )
        differences[name] = float((original - optimized).abs().max())
    return {
        "bf16_max_absolute_differences": differences,
        "future_prefix_max_difference": 0.0,
    }


def measure_case(
    train: Tensor,
    special_ids: tuple[int, ...],
    backend: str,
    mode: str,
    batch_size: int,
    *,
    warmup_steps: int = WARMUP_STEPS,
    measured_steps: int = MEASURED_STEPS,
) -> dict:
    """Measure complete optimizer steps, including generator and both joint passes.

    Every case uses the same architecture, seed, dropout, context and optimizer.
    Only kernel and physical batch change. Larger batches imply fewer updates at
    equal input tokens; this benchmark does not claim equivalent trained quality.
    """
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError("benchmark warmup and measured steps must be positive")
    torch.manual_seed(7)
    config = replace(story_model_config(), attention_backend=backend)
    main = CausalElectra(config).cuda().train()
    generator = None
    if mode == "joint":
        generator = CausalElectra(generator_model_config(config)).cuda().train()
        main.share_generator_embeddings(generator)
    parameters = list(main.parameters())
    if generator is not None:
        parameters += list(generator.parameters())
    parameters = list({id(parameter): parameter for parameter in parameters}.values())
    optimizer = torch.optim.AdamW(parameters, lr=0.0003, weight_decay=0.01)
    objective = ObjectiveConfig(mode=mode)
    batch_rng = torch.Generator().manual_seed(9)
    noise_rng = torch.Generator(device="cuda").manual_seed(10)

    def step() -> float:
        indices = torch.randint(len(train), (batch_size,), generator=batch_rng)
        tokens = train[indices].cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pretraining_step(
                main,
                generator,
                tokens,
                tokens.ne(0),
                objective,
                rng=noise_rng,
                special_token_ids=special_ids,
            )
        if not torch.isfinite(output.loss):
            raise RuntimeError("nonfinite benchmark loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        return output.loss.item()

    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    last_loss = None
    for _ in range(measured_steps):
        last_loss = step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    # Record dispatcher operators in a separate step so profiler overhead is not
    # included in throughput. Forced flash cannot use another SDPA backend.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        step()
    attention_ops = {
        event.key: event.count
        for event in profiler.key_averages()
        if "attention" in event.key.lower()
    }
    if backend == "flash":
        expected = config.num_layers * (2 if mode == "joint" else 1)
        if generator is not None:
            expected += generator.config.num_hidden_layers
        forward_count = attention_ops.get(
            "aten::_scaled_dot_product_flash_attention", 0
        )
        backward_count = attention_ops.get(
            "aten::_scaled_dot_product_flash_attention_backward", 0
        )
        if forward_count != expected or backward_count != expected:
            raise RuntimeError(
                f"flash operator counts differ from {expected}: {attention_ops}"
            )
    return {
        "status": "completed",
        "backend": backend,
        "mode": mode,
        "batch_size": batch_size,
        "context": train.shape[1],
        "dropout": config.dropout,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "measured_input_tokens": measured_steps * batch_size * (train.shape[1] - 1),
        "training_seconds": elapsed,
        "input_tokens_per_second": measured_steps
        * batch_size
        * (train.shape[1] - 1)
        / elapsed,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "last_training_loss": last_loss,
        "finite_gradients": "passed",
        "attention_dispatcher_ops": attention_ops,
        "model_parameters": sum(p.numel() for p in main.parameters()),
        "unique_optimized_parameters": sum(p.numel() for p in parameters),
    }


def benchmark_attention(train: Tensor, special_ids: tuple[int, ...]) -> dict:
    if not torch.cuda.is_available():
        raise ValueError("attention batch benchmark requires CUDA")
    if train.ndim != 2 or train.shape[1] != 256 or train.eq(0).any():
        raise ValueError("benchmark requires unpadded 256-token prepared blocks")
    torch.set_num_threads(2)
    started = time.perf_counter()
    checks = check_flash_equivalence()
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    for mode in ("clm", "joint"):
        for backend in BACKENDS:
            for batch_size in BATCH_SIZES:
                if time.perf_counter() - started > 480:
                    raise TimeoutError(
                        "attention benchmark exceeded its execution budget"
                    )
                print(f"Benchmark {mode}/{backend}/batch={batch_size}", flush=True)
                try:
                    row = measure_case(train, special_ids, backend, mode, batch_size)
                except torch.OutOfMemoryError as error:
                    row = {
                        "status": "oom",
                        "backend": backend,
                        "mode": mode,
                        "batch_size": batch_size,
                        "error": str(error)[:500],
                    }
                rows.append(row)
                gc.collect()
                torch.cuda.empty_cache()
                if row["status"] == "oom":
                    break
    return {
        "checks": checks,
        "cases": rows,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_function_seconds": time.perf_counter() - started,
        "note": (
            "Short throughput/memory benchmark, not validation or target-level "
            "training. All cases use real objective backward, clipping and AdamW. "
            "Batch changes affect updates per token; no quality equivalence is "
            "asserted. Peak allocated/reserved exclude other device processes. "
            "Native flash is forced and profiled; no external flash-attn package "
            "or silent fallback."
        ),
    }
