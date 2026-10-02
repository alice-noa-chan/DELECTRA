"""Verify every prepared token file on CPU with bounded working memory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from deletcra.corpus import file_sha256, write_json


def checked_path(directory: Path, entry: dict) -> Path:
    path = (directory / entry["path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("artifact path escapes corpus")
    return path


def audit_corpus(directory: Path) -> dict:
    """Check hashes, all IDs, alignment and totals; do not modify the corpus.

    Ordinary pretraining shards contain [blocks, sequence_length - 1] IDs.
    SFT stores [examples, sequence_length] tokens and unshifted labels. Scan
    those together so supervised labels must equal their corresponding tokens.
    Each read is at most four MiB per file, independent of corpus size.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("corpus preparation is not complete")
    length, vocab = metadata["sequence_length"], metadata["vocab_size"]
    sft = metadata.get("kind") == "assistant-only-sft"
    totals = {}
    for split, entries in metadata["splits"].items():
        if sft:
            examples = entries["examples"]
            token_path = checked_path(directory, entries["tokens"])
            label_path = checked_path(directory, entries["labels"])
            for path in (token_path, label_path):
                if path.stat().st_size != examples * length * 4:
                    raise ValueError("instruction file size mismatch")
            token_hash, label_hash = hashlib.sha256(), hashlib.sha256()
            visible = supervised = 0
            with token_path.open("rb") as tokens, label_path.open("rb") as labels:
                while raw := tokens.read(length * 4 * 4096):
                    raw_labels = labels.read(len(raw))
                    token_hash.update(raw)
                    label_hash.update(raw_labels)
                    ids = np.frombuffer(raw, dtype="<i4").reshape(-1, length)
                    targets = np.frombuffer(raw_labels, dtype="<i4").reshape(-1, length)
                    if np.any(ids < 0) or np.any(ids >= vocab):
                        raise ValueError("invalid token ID")
                    visible_mask = ids != metadata["pad_token_id"]
                    if np.any(ids[:, 0] != metadata["bos_token_id"]):
                        raise ValueError("instruction sequence must start with BOS")
                    if np.any(visible_mask[:, 1:] & ~visible_mask[:, :-1]):
                        raise ValueError("instruction padding must be on the right")
                    active = targets != -100
                    if np.any(targets[active] != ids[active]):
                        raise ValueError("supervised label differs from token")
                    if np.any(active[:, 0]) or np.any(
                        active & (ids == metadata["pad_token_id"])
                    ):
                        raise ValueError("BOS or padding is supervised")
                    visible += int(np.count_nonzero(visible_mask))
                    supervised += int(np.count_nonzero(active))
            if (
                token_hash.hexdigest() != entries["tokens"]["sha256"]
                or label_hash.hexdigest() != entries["labels"]["sha256"]
            ):
                raise ValueError("instruction file hash mismatch")
            if (
                visible != entries["visible_positions"]
                or supervised != entries["assistant_targets"]
                or examples * length != entries["padded_compute_positions"]
            ):
                raise ValueError("instruction counts mismatch")
            totals[split] = {
                "examples": examples,
                "visible_positions": visible,
                "input_positions": examples * length,
                "assistant_targets": supervised,
            }
            continue

        blocks = content = targets = documents = tail = 0
        for entry in entries:
            path = checked_path(directory, entry)
            expected = entry["blocks"] * (length - 1)
            if (
                path.stat().st_size != expected * 4
                or entry["bytes"] != expected * 4
                or entry["prediction_targets"] != expected
                or entry["content_tokens_before_tail"]
                + entry["record_separators"]
                - entry["dropped_tail_tokens"]
                != expected
                or entry["documents"] != entry["record_separators"]
            ):
                raise ValueError("pretraining shard counts mismatch")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while raw := handle.read(4 * 1024 * 1024):
                    digest.update(raw)
                    ids = np.frombuffer(raw, dtype="<i4")
                    if np.any(ids <= 0) or np.any(ids >= vocab):
                        raise ValueError("invalid token ID or unexpected UNK/PAD")
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("pretraining shard hash mismatch")
            blocks += entry["blocks"]
            content += entry["content_tokens_before_tail"]
            targets += expected
            documents += entry["documents"]
            tail += entry["dropped_tail_tokens"]
        if content != metadata["counts"].get(
            f"{split}_content_tokens", 0
        ) or documents != metadata["counts"].get(f"{split}_documents", 0):
            raise ValueError("manifest totals differ from shard totals")
        totals[split] = {
            "documents": documents,
            "content_tokens": content,
            "prediction_targets": targets,
            "input_positions": blocks * length,
            "blocks": blocks,
            "dropped_tail_tokens": tail,
        }
    return {
        "corpus": str(directory),
        "metadata_sha256": file_sha256(directory / "metadata.json"),
        "status": "verified",
        "all_token_files_checked": True,
        "splits": totals,
        "note": "Data integrity alone is not model training or quality evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, audit_corpus(args.data))


if __name__ == "__main__":
    main()
