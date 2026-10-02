"""Production token budgets, shuffled corpus coverage and exact resume state.

This module never allocates or stops a cloud instance. A wall-clock stop bounds
the trainer process only; the caller must separately stop rented infrastructure.
"""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from deletcra.config import ModelConfig, generator_model_config
from deletcra.corpus import file_sha256, write_json
from deletcra.losses import response_lm_loss
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, pretraining_step


@dataclass(frozen=True)
class ProductionPlan:
    max_input_positions: int
    batch_size: int = 256
    learning_rate: float = 3e-4
    warmup_positions: int = 10_000_000
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 7
    device: str = "cpu"
    precision: str = "fp32"
    cpu_threads: int = 2
    checkpoint_every: int = 1000
    evaluate_every: int = 1000
    validation_blocks: int = 4096
    max_wall_seconds: float | None = None
    sequential_backward: bool = False
    fused_optimizer: bool = False
    share_embeddings: bool = True

    def __post_init__(self):
        for key in (
            "max_input_positions",
            "batch_size",
            "cpu_threads",
            "checkpoint_every",
            "evaluate_every",
            "validation_blocks",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.warmup_positions < self.max_input_positions:
            raise ValueError("warmup must be shorter than the input budget")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("minimum LR ratio must be in [0, 1]")
        for key in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        if self.device not in {"cpu", "cuda"} or self.precision not in {"fp32", "bf16"}:
            raise ValueError("unsupported device or precision")
        if self.precision == "bf16" and self.device != "cuda":
            raise ValueError("BF16 requires CUDA")
        if self.fused_optimizer and self.device != "cuda":
            raise ValueError("fused optimizer requires CUDA")
        if self.max_wall_seconds is not None and (
            not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
        ):
            raise ValueError("wall-clock limit must be finite and positive")


def scheduled_lr(plan: ProductionPlan, completed: int, upcoming: int) -> float:
    """Warm up by processed input positions, then cosine-decay to a floor."""
    position = min(completed + upcoming, plan.max_input_positions)
    if plan.warmup_positions and position < plan.warmup_positions:
        ratio = position / plan.warmup_positions
    else:
        progress = (position - plan.warmup_positions) / (
            plan.max_input_positions - plan.warmup_positions
        )
        ratio = (
            plan.min_lr_ratio
            + (1 - plan.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )
    return plan.learning_rate * ratio


class EpochSampler:
    """Uniform shuffled coverage with just epoch/offset in each checkpoint.

    Regenerate the permutation from seed+epoch rather than saving a multi-million
    index tensor every time. All blocks appear once before beginning another
    epoch. The last batch of an epoch can be smaller; none is silently dropped.
    """

    def __init__(self, size: int, seed: int, *, epoch: int = 0, offset: int = 0):
        if size < 1 or not 0 <= offset <= size or epoch < 0:
            raise ValueError("invalid sampler state")
        self.size, self.seed, self.epoch, self.offset = size, seed, epoch, offset
        self._refresh()

    def _refresh(self):
        self.order = torch.randperm(
            self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)
        ).numpy()

    def next(self, count: int):
        if count < 1:
            raise ValueError("sampler batch must be positive")
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0
            self._refresh()
        end = min(self.offset + count, self.size)
        result = self.order[self.offset : end].copy()
        self.offset = end
        return result


class ProductionTrainer:
    def __init__(
        self,
        train,
        validation,
        model_config: ModelConfig,
        objective: ObjectiveConfig,
        plan: ProductionPlan,
        directory: Path,
        *,
        resume: bool = False,
        initialize_from: Path | None = None,
    ):
        if train.metadata.get("status") != "complete":
            raise ValueError("training corpus preparation is not complete")
        if (
            len(train) < 1
            or model_config.max_positions < train.metadata["sequence_length"]
        ):
            raise ValueError("invalid corpus size or context")
        for key in ("vocab_size", "bos_token_id", "pad_token_id"):
            if train.metadata[key] != getattr(model_config, key):
                raise ValueError(f"corpus and model disagree on {key}")
        if train.metadata != validation.metadata:
            raise ValueError("training and validation need the same corpus manifest")
        self.sft = train.metadata.get("kind") == "assistant-only-sft"
        if self.sft and objective.mode != "clm":
            raise ValueError("instruction tuning uses response-only CLM")
        if plan.sequential_backward and objective.generator_mode != "self":
            raise ValueError("sequential backward requires self joint")
        if objective.lm_loss_backend == "liger" and plan.device != "cuda":
            raise ValueError("Liger requires CUDA")
        if plan.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        if plan.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support BF16")
        self.directory = Path(directory)
        checkpoint = self.directory / "latest.pt"
        if self.directory.exists() and not resume:
            raise FileExistsError("refusing to overwrite a production run")
        if resume and not checkpoint.is_file():
            raise FileNotFoundError("resume checkpoint missing")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.train, self.validation, self.objective, self.plan = (
            train,
            validation,
            objective,
            plan,
        )
        self.sequence_length = train.metadata["sequence_length"]
        self.spec = {
            "model": asdict(model_config),
            "objective": asdict(objective),
            "plan": asdict(plan),
            "corpus_manifest_sha256": hashlib.sha256(
                json.dumps(train.metadata, sort_keys=True).encode()
            ).hexdigest(),
            "initialize_from_sha256": file_sha256(initialize_from)
            if initialize_from is not None
            else None,
        }
        torch.set_num_threads(plan.cpu_threads)
        torch.manual_seed(plan.seed)
        random.seed(plan.seed)
        np.random.seed(plan.seed)
        self.model = CausalElectra(model_config).to(plan.device)
        self.generator = None
        if (
            not self.sft
            and objective.mode != "clm"
            and objective.generator_mode == "separate"
        ):
            self.generator = CausalElectra(generator_model_config(model_config)).to(
                plan.device
            )
            if plan.share_embeddings:
                self.model.share_generator_embeddings(self.generator)
        parameters = list(self.model.parameters())
        if self.generator is not None:
            parameters += list(self.generator.parameters())
        self.parameters = list({id(value): value for value in parameters}.values())
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=plan.learning_rate,
            weight_decay=plan.weight_decay,
            fused=True if plan.fused_optimizer else None,
        )
        self.noise = torch.Generator(device=plan.device).manual_seed(plan.seed + 3)
        self.sampler = EpochSampler(len(train), plan.seed + 2)
        self.state = {
            "step": 0,
            "input_positions": 0,
            "prediction_targets": 0,
            "assistant_targets": 0,
            "active_wall_seconds": 0.0,
            "best_validation_nll": None,
            "best_validation_step": None,
        }
        if initialize_from is not None and not resume:
            source = torch.load(initialize_from, map_location="cpu", weights_only=True)
            self.model.load_state_dict(source["model"])
        if resume:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["spec"] != self.spec:
                raise ValueError(
                    "resume configuration, data, or initialization changed"
                )
            self.model.load_state_dict(saved["model"])
            if self.generator is not None:
                self.generator.load_state_dict(saved["generator"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.state = saved["state"]
            self.sampler = EpochSampler(
                len(train),
                plan.seed + 2,
                epoch=saved["sampler"]["epoch"],
                offset=saved["sampler"]["offset"],
            )
            torch.set_rng_state(saved["torch_rng"])
            if plan.device == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            self.noise.set_state(saved["noise_rng"])
            random.setstate(saved["python_rng"])
            array_state = saved["numpy_rng"]
            np.random.set_state(
                (
                    array_state[0],
                    np.asarray(array_state[1], dtype=np.uint32),
                    *array_state[2:],
                )
            )
            # Discard metrics from work after the last durable optimizer state.
            ledger = self.directory / "metrics.jsonl"
            if ledger.exists():
                retained = [
                    line
                    for line in ledger.read_text().splitlines()
                    if json.loads(line)["step"] <= self.state["step"]
                ]
                ledger.write_text("\n".join(retained) + "\n", encoding="utf-8")
        write_json(self.directory / "run.json", self.spec)
        if not resume:
            self.save()

    def _autocast(self):
        return torch.autocast(
            self.plan.device,
            dtype=torch.bfloat16,
            enabled=self.plan.precision == "bf16",
        )

    def _loss(self, corpus, indices, *, evaluation: bool = False):
        tokens = corpus.batch(indices).to(self.plan.device)
        mask = tokens.ne(self.model.config.pad_token_id)
        with self._autocast():
            if self.sft:
                labels = corpus.labels(indices).to(self.plan.device)
                hidden = self.model(
                    tokens, mask, compute_lm=False, compute_rtd=False
                ).hidden_states
                loss = response_lm_loss(
                    self.model.lm_head,
                    self.model.lm_projection(hidden),
                    labels,
                    mask,
                    backend=self.objective.lm_loss_backend,
                )
                count = int((labels[:, 1:] != -100).sum())
                return (
                    loss,
                    {"nll": float(loss.detach()), "assistant_targets": count},
                    tokens,
                    mask,
                )
            output = pretraining_step(
                self.model,
                self.generator,
                tokens,
                mask,
                self.objective,
                rng=self.noise,
                special_token_ids=(self.model.config.bos_token_id, 2),
                backward_clean=self.plan.sequential_backward and not evaluation,
            )
            return (
                output.loss,
                {
                    "nll": float(output.lm_loss.detach()),
                    "rtd_loss": float(output.rtd_loss.detach()),
                    "assistant_targets": 0,
                },
                tokens,
                mask,
            )

    @torch.no_grad()
    def evaluate(self, corpus, limit: int | None):
        old_noise = self.noise.get_state()
        self.noise.manual_seed(self.plan.seed + 1000)
        self.model.eval()
        if self.generator is not None:
            self.generator.eval()
        nll_sum = loss_sum = total = blocks = 0
        try:
            for start in range(
                0, min(len(corpus), limit or len(corpus)), self.plan.batch_size
            ):
                end = min(
                    start + self.plan.batch_size, len(corpus), limit or len(corpus)
                )
                loss, record, tokens, mask = self._loss(
                    corpus, range(start, end), evaluation=True
                )
                count = (
                    record["assistant_targets"]
                    if self.sft
                    else int((mask[:, 1:] & mask[:, :-1]).sum())
                )
                nll_sum += record["nll"] * count
                loss_sum += float(loss) * count
                total += count
                blocks += len(tokens)
        finally:
            self.noise.set_state(old_noise)
            self.model.train()
            if self.generator is not None:
                self.generator.train()
        nll = nll_sum / total
        return {
            "nll": nll if self.objective.mode != "rtd" else None,
            "perplexity": math.exp(nll)
            if self.objective.mode != "rtd" and nll < 709
            else None,
            "objective_loss": loss_sum / total,
            "targets": total,
            "blocks": blocks,
            "full_split": blocks == len(corpus),
        }

    def save(self, *, filename: str = "latest.pt"):
        array_state = np.random.get_state()
        saved = {
            "spec": self.spec,
            "state": dict(self.state),
            "model": self.model.state_dict(),
            "generator": self.generator.state_dict()
            if self.generator is not None
            else None,
            "optimizer": self.optimizer.state_dict(),
            "sampler": {"epoch": self.sampler.epoch, "offset": self.sampler.offset},
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": [array_state[0], array_state[1].tolist(), *array_state[2:]],
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.plan.device == "cuda"
            else [],
            "noise_rng": self.noise.get_state(),
        }
        target = self.directory / filename
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            torch.save(saved, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            target.replace(target.with_suffix(".previous.pt"))
        temporary.replace(target)

    def fit(self, *, pause_after_steps: int | None = None):
        started = time.monotonic()
        previous_wall = self.state["active_wall_seconds"]
        stop_reason = "input_budget"
        self.model.train()
        if self.generator is not None:
            self.generator.train()
        while (
            self.state["input_positions"] + self.sequence_length
            <= self.plan.max_input_positions
        ):
            if (
                pause_after_steps is not None
                and self.state["step"] >= pause_after_steps
            ):
                stop_reason = "requested_checkpoint_pause"
                break
            if self.plan.max_wall_seconds is not None and (
                previous_wall + time.monotonic() - started >= self.plan.max_wall_seconds
            ):
                stop_reason = "process_wall_budget"
                break
            remaining = (
                self.plan.max_input_positions - self.state["input_positions"]
            ) // self.sequence_length
            indices = self.sampler.next(min(self.plan.batch_size, remaining))
            self.optimizer.zero_grad(set_to_none=True)
            lr = scheduled_lr(
                self.plan,
                self.state["input_positions"],
                len(indices) * self.sequence_length,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            loss, record, tokens, mask = self._loss(self.train, indices)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite production loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.parameters, self.plan.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.state["step"] += 1
            self.state["input_positions"] += tokens.numel()
            self.state["prediction_targets"] += int((mask[:, 1:] & mask[:, :-1]).sum())
            self.state["assistant_targets"] += record["assistant_targets"]
            record.update(
                step=self.state["step"],
                loss=float(loss.detach()),
                lr=lr,
                input_positions=self.state["input_positions"],
                epoch=self.sampler.epoch,
                epoch_offset=self.sampler.offset,
            )
            if self.state["step"] % self.plan.evaluate_every == 0:
                metrics = self.evaluate(self.validation, self.plan.validation_blocks)
                record["validation"] = metrics
                if metrics["nll"] is not None and (
                    self.state["best_validation_nll"] is None
                    or metrics["nll"] < self.state["best_validation_nll"]
                ):
                    self.state["best_validation_nll"] = metrics["nll"]
                    self.state["best_validation_step"] = self.state["step"]
                    self.save(filename="best.pt")
            with (self.directory / "metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as ledger:
                ledger.write(json.dumps(record, allow_nan=False) + "\n")
            self.state["active_wall_seconds"] = (
                previous_wall + time.monotonic() - started
            )
            if self.state["step"] % self.plan.checkpoint_every == 0:
                self.save()
        self.state["active_wall_seconds"] = previous_wall + time.monotonic() - started
        self.save()
        report = {
            "state": self.state,
            "stop_reason": stop_reason,
            "completed_epochs": self.sampler.epoch
            + int(self.sampler.offset == self.sampler.size),
            "offset_in_epoch": self.sampler.offset,
            "unused_input_budget": self.plan.max_input_positions
            - self.state["input_positions"],
            "published": False,
            "note": (
                "Input budget completion is not release quality; "
                "final test is separate. "
                "Process exit does not stop a rented GPU."
            ),
        }
        write_json(self.directory / "report.json", report)
        return report
