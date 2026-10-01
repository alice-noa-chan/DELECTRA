"""Measure the pinned TinyLlama reference on our declared TinyStories protocol."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from deletcra.benchmark import score_language_model
from deletcra.data import load_prepared
from deletcra.target import REFERENCE_MODEL, REFERENCE_REVISION

PROMPTS = (
    "Once upon a time, a little girl found a key",
    "Tom wanted to help his friend, but",
    "The little dog was afraid of the rain. One day,",
    "Lily and her brother went to the park. They saw",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("reference checkpoint download requires --allow-download")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    _, validation, metadata = load_prepared(args.data_dir)
    if (metadata.get("tokenizer"), metadata.get("tokenizer_revision")) != (
        REFERENCE_MODEL,
        REFERENCE_REVISION,
    ):
        raise ValueError("benchmark requires the pinned reference tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.data_dir / "tokenizer", local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        REFERENCE_MODEL,
        revision=REFERENCE_REVISION,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval()
    started = time.perf_counter()
    metrics = score_language_model(
        model, validation, pad_token_id=metadata["pad_token_id"]
    )
    elapsed = time.perf_counter() - started
    samples = []
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=metadata["pad_token_id"],
            use_cache=True,
        )
        samples.append(
            {
                "prompt": prompt,
                "generated_text": tokenizer.decode(output[0], skip_special_tokens=True),
            }
        )
    result = {
        "reference_model": REFERENCE_MODEL,
        "reference_revision": REFERENCE_REVISION,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "dataset": metadata,
        "validation_sha256": hashlib.sha256(
            (args.data_dir / "validation.pt").read_bytes()
        ).hexdigest(),
        "metrics": metrics,
        "evaluation_seconds": elapsed,
        "environment": {
            "device": "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "generation": {"max_new_tokens": 96, "do_sample": False, "samples": samples},
        "note": (
            "Common BOS-prefixed packed-block protocol, context 256; measure candidate "
            "with identical validation tokens. This is not upstream's private shard "
            "loss or independent held-out evidence for the reference (training overlap "
            "unknown). Prior WikiText/ELECTRA-tokenizer perplexities are incomparable. "
            "No candidate has yet been trained on this TinyStories protocol."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved reference benchmark: {args.output}")


if __name__ == "__main__":
    main()
