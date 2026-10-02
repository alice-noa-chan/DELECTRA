"""Frozen checkpoint or pinned reference scoring on explicit corpus partitions."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from deletcra.config import ModelConfig, generator_model_config
from deletcra.corpus import MmapSplit, file_sha256, write_json
from deletcra.instructions import InstructionSplit
from deletcra.losses import response_lm_loss
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, causal_lm_loss, pretraining_step
from deletcra.target import REFERENCE_MODEL, REFERENCE_REVISION


@torch.no_grad()
def evaluate_frozen(
    model,
    corpus,
    *,
    batch_size: int,
    max_blocks: int | None = None,
    generator=None,
    objective=None,
    reference: bool = False,
):
    """Aggregate NLL by actual supervised targets, not by equal-sized batches.

    No optimizer, checkpoint selection or weight mutation occurs. A limit makes
    the result a bounded diagnostic; only full partition coverage is a full test.
    Reference loss uses the same BOS/block convention as the candidate.
    """
    if batch_size < 1 or (max_blocks is not None and max_blocks < 1):
        raise ValueError("evaluation limits must be positive")
    device = next(model.parameters()).device
    model.eval()
    if generator is not None:
        generator.eval()
    rng = torch.Generator(device=device).manual_seed(1007)
    nll_sum = targets = blocks = tp = tn = fp = fn = 0
    sft = corpus.metadata.get("kind") == "assistant-only-sft"
    limit = min(len(corpus), max_blocks or len(corpus))
    started = time.monotonic()
    for start in range(0, limit, batch_size):
        indices = range(start, min(start + batch_size, limit))
        tokens = corpus.batch(indices).to(device)
        mask = tokens.ne(corpus.metadata["pad_token_id"])
        count = int((mask[:, 1:] & mask[:, :-1]).sum())
        if sft:
            labels = corpus.labels(indices).to(device)
            hidden = model(
                tokens, mask, compute_lm=False, compute_rtd=False
            ).hidden_states
            loss = response_lm_loss(
                model.lm_head, model.lm_projection(hidden), labels, mask
            )
            count = int((labels[:, 1:] != -100).sum())
        elif reference:
            loss = causal_lm_loss(
                model(tokens, attention_mask=mask).logits, tokens, mask
            )
        else:
            output = pretraining_step(
                model,
                generator,
                tokens,
                mask,
                objective,
                rng=rng,
                special_token_ids=(1, 2),
            )
            loss = output.lm_loss
            if output.corruption is not None:
                valid = output.corruption.eligible
                predicted = output.rtd_logits[valid] >= 0
                labels = output.corruption.labels[valid]
                tp += int((predicted & labels).sum())
                tn += int((~predicted & ~labels).sum())
                fp += int((predicted & ~labels).sum())
                fn += int((~predicted & labels).sum())
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite frozen evaluation loss")
        nll_sum += float(loss) * count
        targets += count
        blocks += len(tokens)
        if start % (batch_size * 100) == 0:
            print(
                json.dumps(
                    {
                        "evaluated_blocks": blocks,
                        "total_blocks": limit,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    nll = nll_sum / targets
    rtd_only = objective is not None and objective.mode == "rtd"
    report = {
        "nll": None if rtd_only else nll,
        "perplexity": math.exp(nll) if not rtd_only and nll < 709 else None,
        "supervised_targets": targets,
        "blocks": blocks,
        "full_split": blocks == len(corpus),
        "seconds": time.monotonic() - started,
    }
    if tp + tn + fp + fn:
        report["rtd"] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "replacement_rate": (tp + fn) / (tp + tn + fp + fn),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--checkpoint", type=Path)
    sources.add_argument("--reference", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen evaluation evidence")
    torch.set_num_threads(args.cpu_threads)
    metadata = json.loads((args.data / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("corpus is not complete")
    reader = (
        InstructionSplit if metadata.get("kind") == "assistant-only-sft" else MmapSplit
    )
    corpus = reader(args.data, args.split)
    objective = generator = None
    if args.reference:
        if reader is InstructionSplit:
            raise ValueError("reference story model is not an instruction comparator")
        model = AutoModelForCausalLM.from_pretrained(
            REFERENCE_MODEL, revision=REFERENCE_REVISION
        )
        provenance = {
            "model": REFERENCE_MODEL,
            "revision": REFERENCE_REVISION,
            "reference_pretraining_overlap": "unknown",
        }
    else:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        if digest != saved["spec"]["corpus_manifest_sha256"]:
            raise ValueError("evaluation corpus differs from checkpoint corpus")
        config = replace(
            ModelConfig(**saved["spec"]["model"]), attention_backend="sdpa"
        )
        objective = replace(
            ObjectiveConfig(**saved["spec"]["objective"]), lm_loss_backend="torch"
        )
        model = CausalElectra(config)
        model.load_state_dict(saved["model"])
        if saved["generator"] is not None:
            generator = CausalElectra(generator_model_config(config))
            generator.load_state_dict(saved["generator"])
        provenance = {
            "checkpoint": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "training_step": saved["state"]["step"],
            "best_validation_step": saved["state"]["best_validation_step"],
        }
    report = evaluate_frozen(
        model,
        corpus,
        batch_size=args.batch_size,
        max_blocks=args.max_blocks,
        objective=objective,
        generator=generator,
        reference=args.reference,
    )
    report.update(
        split=args.split,
        selection=False,
        provenance=provenance,
        purpose="bounded diagnostic"
        if not report["full_split"]
        else "frozen full-partition scoring",
        note="Full partition coverage alone does not establish release quality.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
