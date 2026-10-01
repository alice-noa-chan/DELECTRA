"""Bounded training, held-out evaluation, and common frozen-backbone probes."""

import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers
from torch import Tensor, nn

from deletcra.config import ModelConfig
from deletcra.metrics import binary_ranking_metrics
from deletcra.model import CausalElectra, validate_batch
from deletcra.objectives import ObjectiveConfig, causal_lm_loss, pretraining_step


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 7
    device: str = "cpu"
    precision: str = "fp32"
    cpu_threads: int = 1
    eval_batches: int = 8
    probe_steps: int = 50
    probe_learning_rate: float = 0.01
    max_training_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "cpu_threads", "eval_batches"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.probe_steps, int) or self.probe_steps < 0:
            raise ValueError("probe_steps must be a nonnegative integer")
        for name in ("learning_rate", "max_grad_norm", "probe_learning_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.precision == "bf16" and self.device != "cuda":
            raise ValueError("bf16 training is supported only on CUDA")
        if self.max_training_seconds is not None and (
            not math.isfinite(self.max_training_seconds)
            or self.max_training_seconds <= 0
        ):
            raise ValueError("max_training_seconds must be finite and positive")


def _autocast(settings: TrainConfig):
    return torch.autocast(
        device_type=settings.device,
        dtype=torch.bfloat16,
        enabled=settings.precision == "bf16",
    )


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _perplexity(nll: float) -> float | None:
    return math.exp(nll) if nll < 709 else None


@torch.no_grad()
def evaluate(
    model: CausalElectra,
    generator: CausalElectra | None,
    validation: Tensor,
    objective: ObjectiveConfig,
    settings: TrainConfig,
    *,
    special_token_ids: tuple[int, ...] = (),
) -> dict:
    model.eval()
    if generator is not None:
        generator.eval()
    rng = torch.Generator(device=settings.device).manual_seed(settings.seed + 1000)
    lm_sum = gen_sum = rtd_sum = 0.0
    lm_count = rtd_count = tp = tn = fp = fn = 0
    ranking_scores, ranking_labels = [], []
    limit = settings.batch_size * settings.eval_batches
    for start in range(0, min(len(validation), limit), settings.batch_size):
        tokens = validation[start : start + settings.batch_size].to(settings.device)
        mask = tokens.ne(model.config.pad_token_id)
        with _autocast(settings):
            output = pretraining_step(
                model,
                generator,
                tokens,
                mask,
                objective,
                rng=rng,
                special_token_ids=special_token_ids,
            )
        count = int((mask[:, 1:] & mask[:, :-1]).sum())
        lm_count += count
        lm_sum += output.lm_loss.item() * count
        gen_sum += output.generator_loss.item() * count
        if output.corruption is not None:
            valid = output.corruption.eligible
            count = int(valid.sum())
            rtd_count += count
            rtd_sum += output.rtd_loss.item() * count
            predicted = output.rtd_logits[valid] >= 0
            labels = output.corruption.labels[valid]
            ranking_scores.append(output.rtd_logits[valid].float().cpu())
            ranking_labels.append(labels.cpu())
            tp += int((predicted & labels).sum())
            tn += int((~predicted & ~labels).sum())
            fp += int((predicted & ~labels).sum())
            fn += int((~predicted & labels).sum())
    metrics: dict = {"next_token_targets": lm_count}
    if objective.mode != "rtd":
        metrics["lm_loss"] = lm_sum / lm_count
        metrics["perplexity"] = _perplexity(metrics["lm_loss"])
    if objective.mode != "clm":
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        metrics.update(
            generator_loss=gen_sum / lm_count,
            rtd_loss=rtd_sum / rtd_count,
            rtd_accuracy=(tp + tn) / rtd_count,
            rtd_majority_baseline=max(tp + fn, tn + fp) / rtd_count,
            rtd_replacement_rate=(tp + fn) / rtd_count,
            rtd_precision=tp / (tp + fp) if tp + fp else 0.0,
            rtd_recall=recall,
            rtd_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            rtd_balanced_accuracy=(recall + specificity) / 2
            if recall is not None and specificity is not None
            else None,
            rtd_confusion={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        )
        metrics.update(
            binary_ranking_metrics(torch.cat(ranking_scores), torch.cat(ranking_labels))
        )
    return metrics


def frozen_probe(
    model: CausalElectra,
    train: Tensor,
    validation: Tensor,
    settings: TrainConfig,
) -> tuple[dict, nn.Linear]:
    """Train an identical linear next-token probe while keeping each backbone fixed.

    This provides a shared representation task even for RTD-only pretraining.
    Probe perplexity belongs to the probe, not to the pretrained RTD LM head.
    """
    model.eval()
    torch.manual_seed(settings.seed + 2000)
    probe = nn.Linear(model.config.hidden_size, model.config.vocab_size).to(
        settings.device
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=settings.probe_learning_rate)
    rng = torch.Generator().manual_seed(settings.seed + 2001)
    for _ in range(settings.probe_steps):
        indices = torch.randint(len(train), (settings.batch_size,), generator=rng)
        tokens = train[indices].to(settings.device)
        mask = tokens.ne(model.config.pad_token_id)
        with torch.no_grad(), _autocast(settings):
            hidden = model(tokens, mask, compute_lm=False).hidden_states
        loss = causal_lm_loss(probe(hidden.float()), tokens, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    nll_sum = count = 0
    limit = settings.batch_size * settings.eval_batches
    with torch.no_grad():
        for start in range(0, min(len(validation), limit), settings.batch_size):
            tokens = validation[start : start + settings.batch_size].to(settings.device)
            mask = tokens.ne(model.config.pad_token_id)
            with _autocast(settings):
                hidden = model(tokens, mask, compute_lm=False).hidden_states
            nll = causal_lm_loss(probe(hidden.float()), tokens, mask)
            targets = int((mask[:, 1:] & mask[:, :-1]).sum())
            nll_sum += nll.item() * targets
            count += targets
    nll = nll_sum / count
    return {
        "steps": settings.probe_steps,
        "nll": nll,
        "perplexity": _perplexity(nll),
    }, probe


def run_experiment(
    train: Tensor,
    validation: Tensor,
    model_settings: ModelConfig,
    objective: ObjectiveConfig,
    settings: TrainConfig,
    output_directory: str | Path,
    *,
    special_token_ids: tuple[int, ...] = (),
    progress: Callable[[dict], None] | None = None,
    dataset_metadata: dict | None = None,
) -> dict:
    """Train a fresh model; save reports and checkpoints without overwriting runs."""
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch environment")
    if settings.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bf16")
    for tokens in (train, validation):
        if tokens.device.type != "cpu":
            raise ValueError(
                "dataset tensors must be on CPU; batches move to the device"
            )
        validate_batch(tokens, tokens.ne(model_settings.pad_token_id))
        if tokens.shape[1] < 2 or tokens.shape[1] > model_settings.max_positions:
            raise ValueError("dataset sequence length does not fit model context")
        if tokens.min() < 0 or tokens.max() >= model_settings.vocab_size:
            raise ValueError("dataset has out-of-vocabulary tokens")
        if not tokens[:, 0].eq(model_settings.bos_token_id).all():
            raise ValueError("dataset sequences must begin with BOS")
    directory = Path(output_directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {directory}")
    torch.set_num_threads(settings.cpu_threads)
    torch.manual_seed(settings.seed)
    model = CausalElectra(model_settings).to(settings.device)
    generator = None
    if objective.mode != "clm":
        generator_settings = ModelConfig(
            vocab_size=model_settings.vocab_size,
            embedding_size=model_settings.embedding_size,
            hidden_size=max(
                model_settings.num_heads,
                (model_settings.hidden_size // 4 // model_settings.num_heads)
                * model_settings.num_heads,
            ),
            num_layers=max(1, model_settings.num_layers // 3),
            num_heads=model_settings.num_heads,
            intermediate_size=max(
                model_settings.num_heads, model_settings.intermediate_size // 4
            ),
            max_positions=model_settings.max_positions,
            dropout=model_settings.dropout,
            pad_token_id=model_settings.pad_token_id,
            bos_token_id=model_settings.bos_token_id,
        )
        generator = CausalElectra(generator_settings).to(settings.device)
    parameters = list(model.parameters())
    if generator is not None:
        parameters += list(generator.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    directory.mkdir(parents=True)
    initial = evaluate(
        model,
        generator,
        validation,
        objective,
        settings,
        special_token_ids=special_token_ids,
    )
    model.train()
    if generator is not None:
        generator.train()
    torch.manual_seed(settings.seed + 1)
    batch_rng = torch.Generator().manual_seed(settings.seed + 2)
    noise_rng = torch.Generator(device=settings.device).manual_seed(settings.seed + 3)
    if settings.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = []
    # A time-limited run can stop before planned warmup finishes; measure all steps.
    warmup_steps = 0 if settings.max_training_seconds else min(5, settings.steps // 5)
    training_tokens = measured_tokens = 0
    _synchronize(settings.device)
    started = measured_started = time.perf_counter()
    for step in range(1, settings.steps + 1):
        indices = torch.randint(len(train), (settings.batch_size,), generator=batch_rng)
        batch = train[indices]
        tokens_in_batch = int(batch[:, 1:].ne(model_settings.pad_token_id).sum())
        tokens = batch.to(settings.device)
        mask = tokens.ne(model_settings.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(settings):
            output = pretraining_step(
                model,
                generator,
                tokens,
                mask,
                objective,
                rng=noise_rng,
                special_token_ids=special_token_ids,
            )
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        output.loss.backward()
        nn.utils.clip_grad_norm_(
            parameters, settings.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        training_tokens += tokens_in_batch
        if step > warmup_steps:
            measured_tokens += tokens_in_batch
        if step == warmup_steps:
            _synchronize(settings.device)
            measured_started = time.perf_counter()
        if step == 1 or step % 10 == 0 or step == settings.steps:
            record = {
                "step": step,
                "loss": output.loss.item(),
                "generator_loss": output.generator_loss.item(),
                "rtd_loss": output.rtd_loss.item(),
                "lm_loss": output.lm_loss.item(),
            }
            history.append(record)
            if progress is not None:
                progress(record)
        if settings.max_training_seconds is not None:
            _synchronize(settings.device)
            if time.perf_counter() - started >= settings.max_training_seconds:
                break
    completed_steps = step
    _synchronize(settings.device)
    ended = time.perf_counter()
    throughput = measured_tokens / (ended - measured_started)
    peak_memory = (
        torch.cuda.max_memory_allocated() if settings.device == "cuda" else None
    )
    final = evaluate(
        model,
        generator,
        validation,
        objective,
        settings,
        special_token_ids=special_token_ids,
    )
    probe_metrics = None
    if settings.probe_steps:
        probe_metrics, probe = frozen_probe(model, train, validation, settings)
        torch.save(probe.state_dict(), directory / "probe.pt")
    model.save(directory / "model")
    if generator is not None:
        generator.save(directory / "generator")
    report = {
        "model_config": asdict(model_settings),
        "objective_config": asdict(objective),
        "train_config": asdict(settings),
        "dataset": dataset_metadata or {"kind": "caller-provided token tensors"},
        "special_token_ids": list(special_token_ids),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name()
            if settings.device == "cuda"
            else "cpu",
        },
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "generator_parameters": sum(p.numel() for p in generator.parameters())
        if generator is not None
        else 0,
        "initial_validation": initial,
        "final_validation": final,
        "frozen_probe": probe_metrics,
        "history": history,
        "training_tokens": training_tokens,
        "completed_steps": completed_steps,
        "stop_reason": "time_budget"
        if settings.max_training_seconds is not None
        and ended - started >= settings.max_training_seconds
        else "step_budget",
        "training_seconds": ended - started,
        "warmup_steps": warmup_steps,
        "input_tokens_per_second": throughput,
        "peak_cuda_allocated_bytes": peak_memory,
        "estimated_hours_per_10m_tokens": 10_000_000 / throughput / 3600,
        "measurement_note": (
            "Throughput includes this objective's generator/discriminator passes and "
            "optimizer step, excludes warmup, validation, probe training, and saving. "
            "RTD/joint use a changing generator, so initial/final RTD corruption is "
            "not identical. Equal input tokens do not imply equal compute."
        ),
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
