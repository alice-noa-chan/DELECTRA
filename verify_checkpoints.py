"""Reload real experiment checkpoints, check causality, and hash saved artifacts."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from deletcra.model import CausalElectra


def verify(root: Path) -> dict:
    torch.set_num_threads(2)
    checkpoints = sorted(root.rglob("model.pt"))
    if not checkpoints:
        raise ValueError(f"no model checkpoints found under {root}")
    results = []
    for checkpoint in checkpoints:
        model = CausalElectra.load(checkpoint.parent, attention_backend="sdpa").eval()
        prefix = torch.tensor([[model.config.bos_token_id, 3, 4, 5]])
        changed = prefix.clone()
        changed[:, -1] = 19
        with torch.no_grad():
            first, second = model(prefix), model(changed)
        for name in ("lm_logits", "rtd_logits", "hidden_states"):
            before, after = getattr(first, name), getattr(second, name)
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise ValueError(f"nonfinite checkpoint outputs: {checkpoint}")
            torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
        results.append(
            {
                "checkpoint": str(checkpoint.relative_to(root)),
                "strict_reload": "passed",
                "finite_outputs": "passed",
                "future_prefix_max_difference": 0.0,
            }
        )
        del model, first, second
        gc.collect()
    # Cross-model aliases are intentionally restored only for new training. Saved
    # shared checkpoints must nonetheless contain identical embedding values.
    shared_pairs = []
    for generator_path in root.rglob("shared/rtd/generator/model.pt"):
        main_path = generator_path.parents[1] / "model/model.pt"
        main = torch.load(main_path, map_location="cpu", weights_only=True)
        generator = torch.load(generator_path, map_location="cpu", weights_only=True)
        for name in ("word_embeddings", "position_embeddings"):
            key = f"electra.embeddings.{name}.weight"
            torch.testing.assert_close(main[key], generator[key], rtol=0, atol=0)
        shared_pairs.append(str(generator_path.parent.relative_to(root)))
        del main, generator
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and (path.suffix == ".pt" or path.name == "config.json")
    }
    return {
        "root": str(root.resolve()),
        "torch": torch.__version__,
        "checkpoints": results,
        "shared_embedding_pairs_verified": shared_pairs,
        "sha256": hashes,
        "note": (
            "Local CPU SDPA strict loading, finite outputs and bit-exact future "
            "isolation; not a generation quality benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    result = verify(args.root)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(result['checkpoints'])} checkpoints; saved {args.output}")


if __name__ == "__main__":
    main()
