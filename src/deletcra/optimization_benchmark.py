"""Correctness gates and bounded, repeated integrated-training measurements."""

import gc
import statistics
import time
from copy import deepcopy

import torch
from torch import Tensor, nn

from deletcra.attention_benchmark import measure_case
from deletcra.config import ModelConfig
from deletcra.losses import fused_causal_lm_loss
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, causal_lm_loss, pretraining_step

REPETITIONS = 3
WARMUP = 20
MEASURED = 200
# Keep architecture, dropout, LR, context and seeds fixed. Self proposals change
# the learning algorithm; loss/backward/optimizer flags only change execution.
CASES = (
    ("separate", "separate", "torch", False, False, 64),
    ("self", "self", "torch", False, False, 64),
    ("self-liger", "self", "liger", False, False, 64),
    ("self-sequential", "self", "liger", True, False, 64),
    ("self-fused", "self", "liger", True, True, 64),
    ("self-large-batch", "self", "liger", True, True, 256),
)


def check_liger_equivalence() -> dict:
    """Compare masked shifted losses and all three gradients, including LM bias.

    A tied input/output table needs a real weight gradient. Checking loss alone
    would miss errors in accumulation, bias, or the upstream feature gradient.
    FP32 is a tight arithmetic check; BF16 allows rounding differences and also
    checks gradient direction and relative error. Tolerances are fixed in code.
    """
    results = {}
    for precision in ("fp32", "bf16"):
        torch.manual_seed(23)
        head = nn.Linear(32, 257).cuda()
        other = deepcopy(head)
        features = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
        other_features = features.detach().clone().requires_grad_()
        ids = torch.randint(3, 257, (2, 8), device="cuda")
        ids[1, -2:] = 0
        mask = ids.ne(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            reference = causal_lm_loss(head(features), ids, mask)
            actual = fused_causal_lm_loss(other, other_features, ids, mask)
        reference.backward()
        actual.backward()
        torch.testing.assert_close(reference, actual, rtol=0.002, atol=0.02)
        errors = {}
        for name, expected, got in (
            ("weight", head.weight.grad, other.weight.grad),
            ("bias", head.bias.grad, other.bias.grad),
            ("features", features.grad, other_features.grad),
        ):
            a, b = expected.float().flatten(), got.float().flatten()
            if precision == "fp32":
                torch.testing.assert_close(a, b, rtol=0.0005, atol=0.00001)
            relative = float((a - b).norm() / a.norm())
            cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            if relative > 0.03 or cosine < 0.995:
                raise RuntimeError(f"{precision} {name} gradient differs")
            errors[name] = {
                "max_absolute_difference": float((a - b).abs().max()),
                "relative_l2_error": relative,
                "cosine_similarity": cosine,
            }
        results[precision] = {
            "loss_absolute_difference": float((reference - actual).abs()),
            "gradients": errors,
        }
    return results


def check_sequential_equivalence() -> dict:
    """Keep dropout, sampled replacements and tied gradients in a CUDA check."""
    torch.manual_seed(23)
    a = CausalElectra(
        ModelConfig(vocab_size=257, dropout=0.1, attention_backend="flash")
    ).cuda()
    b = deepcopy(a)
    ids = torch.randint(3, 257, (2, 16), device="cuda")
    ids[:, 0] = 1
    outputs = []
    settings = ObjectiveConfig(
        generator_mode="self", lm_loss_backend="liger", lm_weight=0.7, rtd_weight=3
    )
    for model, sequential in ((a, False), (b, True)):
        torch.manual_seed(31)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pretraining_step(
                model,
                None,
                ids,
                ids.ne(0),
                settings,
                backward_clean=sequential,
                rng=torch.Generator(device="cuda").manual_seed(41),
            )
        out.loss.backward()
        outputs.append(out)
    torch.testing.assert_close(outputs[0].loss, outputs[1].loss)
    if not torch.equal(
        outputs[0].corruption.input_ids, outputs[1].corruption.input_ids
    ):
        raise RuntimeError("sequential execution changed sampled replacements")
    differences = []
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        if pa.grad is None or pb.grad is None:
            if pa.grad is not None or pb.grad is not None:
                raise RuntimeError("sequential gradient path differs")
            continue
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0.03, atol=0.005)
        differences.append(float((pa.grad - pb.grad).abs().max()))
    return {
        "same_replacements": True,
        "loss_absolute_difference": float((outputs[0].loss - outputs[1].loss).abs()),
        "gradient_max_absolute_difference": max(differences),
    }


def summarize_cases(rows: list[dict]) -> list[dict]:
    """Require complete repetitions before claiming a throughput or memory result."""
    summary = []
    for name, generator, loss, sequential, fused, batch in CASES:
        cases = [row for row in rows if row["case"] == name]
        if len(cases) != REPETITIONS or any(
            row["status"] != "completed"
            or row["measured_steps"] != MEASURED
            or row["warmup_steps"] != WARMUP
            or row["batch_size"] != batch
            or row["generator_mode"] != generator
            or row["lm_loss_backend"] != loss
            or row["sequential_backward"] != sequential
            or row["fused_optimizer"] != fused
            or row["backend"] != "flash"
            or row["mode"] != "joint"
            or row["context"] != 256
            for row in cases
        ):
            raise ValueError("optimization summary requires all full trials")
        rates = [row["input_tokens_per_second"] for row in cases]
        summary.append(
            {
                "case": name,
                "batch_size": cases[0]["batch_size"],
                "input_tokens_per_second": sum(
                    row["measured_input_tokens"] for row in cases
                )
                / sum(row["training_seconds"] for row in cases),
                "repetition_throughputs": rates,
                "throughput_sample_std_percent": statistics.stdev(rates)
                / statistics.mean(rates)
                * 100,
                "peak_cuda_allocated_bytes": max(
                    row["peak_cuda_allocated_bytes"] for row in cases
                ),
            }
        )
    return summary


def benchmark_optimizations(train: Tensor, special_ids: tuple[int, ...]) -> dict:
    if not torch.cuda.is_available():
        raise ValueError("optimization benchmark requires CUDA")
    if train.ndim != 2 or train.shape[1] != 256 or train.eq(0).any():
        raise ValueError("benchmark requires unpadded 256-token blocks")
    torch.set_num_threads(2)
    started = time.perf_counter()
    checks = {
        "liger_loss_and_gradients": check_liger_equivalence(),
        "sequential_execution": check_sequential_equivalence(),
    }
    rows = []
    # Rotate execution order across repetitions to reduce first/last-case bias.
    for repetition in range(REPETITIONS):
        order = CASES[repetition:] + CASES[:repetition]
        for name, generator, loss, sequential, fused, batch in order:
            if time.perf_counter() - started > 450:
                raise TimeoutError("optimization benchmark exceeded its work budget")
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Optimization {name}: trial {repetition + 1}", flush=True)
            row = measure_case(
                train,
                special_ids,
                "flash",
                "joint",
                batch,
                warmup_steps=WARMUP,
                measured_steps=MEASURED,
                generator_mode=generator,
                lm_loss_backend=loss,
                sequential_backward=sequential,
                fused_optimizer=fused,
            )
            row.update(case=name, repetition=repetition + 1)
            rows.append(row)
    return {
        "status": "completed",
        "checks": checks,
        "cases": rows,
        "summary": summarize_cases(rows),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "note": (
            "Full optimizer throughput, not convergence or final model quality. "
            "Self proposals change the algorithm; larger batches change updates "
            "per token. No trained-model quality equivalence is claimed."
        ),
    }
