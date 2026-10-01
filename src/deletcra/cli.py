"""Explicit commands for data preparation and bounded local experiments."""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

from deletcra.config import ModelConfig
from deletcra.data import load_prepared, prepare_wikitext, synthetic_sequences
from deletcra.experiment import TrainConfig, run_experiment
from deletcra.objectives import ObjectiveConfig


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Causal ELECTRA experiments")
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare opt-in WikiText data")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--subset",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        default="wikitext-2-raw-v1",
    )
    prepare.add_argument("--tokenizer", default="google/electra-small-discriminator")
    prepare.add_argument("--sequence-length", type=int, default=128)
    prepare.add_argument("--max-train-tokens", type=int, default=1_000_000)
    prepare.add_argument("--max-validation-tokens", type=int, default=100_000)
    prepare.add_argument("--allow-download", action="store_true")
    run = commands.add_parser("run", help="train and compare objectives")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--mode", choices=["all", "rtd", "clm", "joint"], default="all")
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--preset", choices=["tiny", "small"], default="tiny")
    run.add_argument("--sequence-length", type=int)
    run.add_argument("--vocab-size", type=int, default=16)
    budget = run.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int)
    budget.add_argument("--train-tokens", type=int)
    run.add_argument("--max-training-seconds", type=float)
    run.add_argument("--share-embeddings", action="store_true")
    run.add_argument("--dropout", type=float)
    run.add_argument("--eval-every-steps", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--learning-rate", type=float)
    run.add_argument("--replacement-probability", type=float, default=0.15)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--rtd-weight", type=float, default=5.0)
    run.add_argument("--lm-weight", type=float, default=1.0)
    run.add_argument("--generator-weight", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    run.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    run.add_argument("--cpu-threads", type=int, default=1)
    run.add_argument("--eval-batches", type=int, default=8)
    run.add_argument("--probe-steps", type=int, default=50)
    run.add_argument("--quiet", action="store_true")
    return result


def _run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    special_ids = ()
    if args.data_dir is not None:
        train, validation, metadata = load_prepared(args.data_dir)
        sequence_length = metadata["sequence_length"]
        if args.sequence_length is not None and args.sequence_length != sequence_length:
            raise ValueError("--sequence-length must match the prepared dataset")
        vocab_size = metadata["vocab_size"]
        pad_id, bos_id = metadata["pad_token_id"], metadata["bos_token_id"]
        special_ids = tuple(metadata["special_token_ids"])
    else:
        sequence_length = args.sequence_length or 16
        vocab_size = args.vocab_size
        pad_id, bos_id = 0, 1
        train = synthetic_sequences(
            256, sequence_length, vocab_size, seed=args.seed + 10
        )
        validation = synthetic_sequences(
            64, sequence_length, vocab_size, seed=args.seed + 11
        )
        metadata = {
            "kind": "synthetic cyclic grammar",
            "train_seed": args.seed + 10,
            "validation_seed": args.seed + 11,
            "note": "optimization smoke test, not a real-language benchmark",
        }
    if args.preset == "tiny":
        model_settings = ModelConfig()
    else:
        model_settings = ModelConfig(
            embedding_size=128,
            hidden_size=256,
            num_layers=12,
            num_heads=4,
            intermediate_size=1024,
            max_positions=128,
        )
    model_settings = replace(
        model_settings,
        vocab_size=vocab_size,
        max_positions=max(model_settings.max_positions, sequence_length),
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        dropout=args.dropout if args.dropout is not None else model_settings.dropout,
    )
    if args.train_tokens is not None:
        if args.train_tokens < 1 or args.batch_size < 1:
            raise ValueError("train-tokens and batch-size must be positive")
        steps = math.ceil(args.train_tokens / (args.batch_size * (sequence_length - 1)))
    else:
        steps = args.steps if args.steps is not None else 100
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    settings = TrainConfig(
        steps=steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
        if args.learning_rate is not None
        else (0.001 if args.preset == "tiny" else 0.0003),
        seed=args.seed,
        device=device,
        precision=args.precision,
        cpu_threads=args.cpu_threads,
        eval_batches=args.eval_batches,
        probe_steps=args.probe_steps,
        max_training_seconds=args.max_training_seconds,
        share_embeddings=args.share_embeddings,
        eval_every_steps=args.eval_every_steps,
    )
    modes = ["clm", "rtd", "joint"] if args.mode == "all" else [args.mode]
    rows = []
    for mode in modes:
        objective = ObjectiveConfig(
            mode=mode,
            replacement_probability=args.replacement_probability,
            temperature=args.temperature,
            rtd_weight=args.rtd_weight,
            lm_weight=args.lm_weight,
            generator_weight=args.generator_weight,
        )

        def progress(record: dict, *, current_mode: str = mode) -> None:
            if not args.quiet:
                print(
                    f"{current_mode}: {json.dumps(record)}", file=sys.stderr, flush=True
                )

        report = run_experiment(
            train,
            validation,
            model_settings,
            objective,
            settings,
            args.output_dir / mode,
            special_token_ids=special_ids,
            progress=progress,
            dataset_metadata=metadata,
        )
        rows.append(
            {
                "mode": mode,
                "initial_validation": report["initial_validation"],
                "final_validation": report["final_validation"],
                "frozen_probe": report["frozen_probe"],
                "training_tokens": report["training_tokens"],
                "completed_steps": report["completed_steps"],
                "stop_reason": report["stop_reason"],
                "training_seconds": report["training_seconds"],
                "input_tokens_per_second": report["input_tokens_per_second"],
                "estimated_hours_per_10m_tokens": report[
                    "estimated_hours_per_10m_tokens"
                ],
                "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
            }
        )
    summary = {
        "results": rows,
        "total_training_seconds": sum(row["training_seconds"] for row in rows),
        "estimated_total_hours_per_10m_tokens_each": sum(
            row["estimated_hours_per_10m_tokens"] for row in rows
        ),
        "comparison_note": (
            "Matched initialization, data, and seed. Step/token caps and optional "
            "training wall-time limits are reported per run; neither guarantees "
            "equal FLOPs. Probe uses identical fresh linear heads and a frozen "
            "backbone. RTD-only has no trained main LM head or main LM perplexity."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_wikitext(
                args.output_dir,
                subset=args.subset,
                tokenizer_name=args.tokenizer,
                sequence_length=args.sequence_length,
                max_train_tokens=args.max_train_tokens,
                max_validation_tokens=args.max_validation_tokens,
                allow_download=args.allow_download,
            )
        else:
            result = _run(args)
    except (ValueError, FileExistsError, RuntimeError) as error:
        argument_parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
