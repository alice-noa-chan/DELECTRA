"""Explicit single-device production entry point; never rents a GPU."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from deletcra.corpus import MmapSplit
from deletcra.instructions import InstructionSplit
from deletcra.objectives import ObjectiveConfig
from deletcra.target import story_model_config
from deletcra.training import ProductionPlan, ProductionTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-positions", type=int, required=True)
    parser.add_argument("--warmup-positions", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda", "xla"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument(
        "--attention", choices=["eager", "sdpa", "flash"], default="sdpa"
    )
    parser.add_argument("--mode", choices=["joint", "rtd", "clm"], default="joint")
    parser.add_argument("--generator", choices=["self", "separate"], default="self")
    parser.add_argument("--loss", choices=["torch", "liger"], default="torch")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--evaluate-every", type=int, default=1000)
    parser.add_argument("--validation-blocks", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--pause-after-steps", type=int)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Allow recorded execution changes while preserving the research recipe",
    )
    args = parser.parse_args()
    metadata = json.loads((args.data / "metadata.json").read_text())
    reader = (
        InstructionSplit if metadata.get("kind") == "assistant-only-sft" else MmapSplit
    )
    train, validation = reader(args.data, "train"), reader(args.data, "validation")
    objective = ObjectiveConfig(
        mode=args.mode, generator_mode=args.generator, lm_loss_backend=args.loss
    )
    plan = ProductionPlan(
        max_input_positions=args.max_input_positions,
        warmup_positions=args.warmup_positions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        precision=args.precision,
        checkpoint_every=args.checkpoint_every,
        evaluate_every=args.evaluate_every,
        validation_blocks=args.validation_blocks,
        cpu_threads=args.cpu_threads,
        max_wall_seconds=args.max_wall_seconds,
        sequential_backward=args.generator == "self",
        fused_optimizer=args.device == "cuda",
    )
    trainer = ProductionTrainer(
        train,
        validation,
        replace(story_model_config(), attention_backend=args.attention),
        objective,
        plan,
        args.output,
        resume=args.resume,
        initialize_from=args.initialize_from,
        migrate=args.migrate,
    )
    print(json.dumps(trainer.fit(pause_after_steps=args.pause_after_steps), indent=2))


if __name__ == "__main__":
    main()
