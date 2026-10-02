"""Short, response-masked instruction examples using the existing vocabulary."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from deletcra.corpus import file_sha256, write_json
from deletcra.prepare_cpu import document_key, web_split
from deletcra.target import REFERENCE_MODEL, REFERENCE_REVISION

IT_DATASET = "HuggingFaceTB/smoltalk2"
IT_REVISION = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
IT_COMPONENTS = (
    "smoltalk_smollm3_everyday_conversations_no_think",
    "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_smol_summarize_no_think",
)


def encode_conversation(messages, tokenizer, *, context: int = 256):
    """Mask role markers and prompts; supervise assistant content and its EOS.

    Each role marker/content is tokenized separately for deterministic boundaries.
    Labels align with input positions; the trainer shifts once so logits[t]
    predict labels[t+1]. Reject long conversations intact rather than truncating
    a reply. Padding is always invisible and has label -100.
    """
    if not messages or messages[-1]["role"] != "assistant":
        return None, "unfinished"
    ids, labels = [tokenizer.bos_token_id], [-100]
    previous = None
    for i, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None, "schema"
        if role == "system" and i != 0:
            return None, "roles"
        if role == "assistant" and previous != "user":
            return None, "roles"
        if role == "user" and previous not in {None, "system", "assistant"}:
            return None, "roles"
        previous = role
        content = content.replace("<think>\n\n</think>", "").strip()
        if not content or "<think>" in content or "</think>" in content:
            return None, "reasoning_or_empty"
        prefix = tokenizer(role.title() + ":\n", add_special_tokens=False)["input_ids"]
        body = tokenizer(content, add_special_tokens=False)["input_ids"]
        ending = [tokenizer.eos_token_id] if role == "assistant" else []
        segment = [*prefix, *body, *ending]
        ids.extend(segment)
        labels.extend(
            [-100] * len(prefix)
            + (body + ending if role == "assistant" else [-100] * len(body))
        )
    if len(ids) > context:
        return None, "over_context"
    if tokenizer.pad_token_id in ids:
        return None, "unknown_token"
    count = sum(label != -100 for label in labels)
    if count < 2:
        return None, "short_response"
    visible = len(ids)
    ids.extend([tokenizer.pad_token_id] * (context - visible))
    labels.extend([-100] * (context - visible))
    return (ids, labels, visible, count), None


class InstructionSplit:
    def __init__(self, directory: str | Path, split: str):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        entry = self.metadata["splits"][split]
        self.context = self.metadata["sequence_length"]
        for kind in ("tokens", "labels"):
            path = (directory / entry[kind]["path"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("instruction path escapes corpus")
            if file_sha256(path) != entry[kind]["sha256"]:
                raise ValueError("instruction shard hash mismatch")
            if path.stat().st_size != entry["examples"] * self.context * 4:
                raise ValueError("instruction shard size mismatch")
            if not entry["examples"]:
                raise ValueError("empty instruction split")
            setattr(
                self,
                "_" + kind,
                np.memmap(path, dtype="<i4", mode="r").reshape(-1, self.context),
            )

    def __len__(self):
        return len(self._tokens)

    def batch(self, indices):
        return torch.from_numpy(
            np.asarray(self._tokens[indices], dtype=np.int64).copy()
        )

    def labels(self, indices):
        return torch.from_numpy(
            np.asarray(self._labels[indices], dtype=np.int64).copy()
        )


def prepare(directory: Path):
    if directory.exists():
        raise FileExistsError("refusing to overwrite instruction preparation")
    directory.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(
        REFERENCE_MODEL, revision=REFERENCE_REVISION
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.save_pretrained(directory / "tokenizer")
    counts = Counter()
    seen = set()
    splits = {name: Counter() for name in ("train", "validation", "test")}
    handles = {
        (split, kind): (directory / f"{split}-{kind}.bin").open("wb")
        for split in splits
        for kind in ("tokens", "labels")
    }
    sources = []
    try:
        for component in IT_COMPONENTS:
            name = f"SFT/{component}-00000-of-00001.parquet"
            path = Path(
                hf_hub_download(
                    IT_DATASET, name, repo_type="dataset", revision=IT_REVISION
                )
            )
            local = Counter()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    local["seen"] += 1
                    messages = row["messages"]
                    key = document_key(json.dumps(messages, sort_keys=True))
                    if key in seen:
                        local["duplicate"] += 1
                        continue
                    seen.add(key)
                    encoded, reason = encode_conversation(messages, tokenizer)
                    if reason:
                        local["rejected_" + reason] += 1
                        continue
                    split = web_split(key)
                    ids, labels, visible, targets = encoded
                    for kind, values in (("tokens", ids), ("labels", labels)):
                        handles[split, kind].write(
                            np.asarray(values, dtype="<i4").tobytes()
                        )
                    splits[split].update(
                        examples=1,
                        visible_positions=visible,
                        padded_compute_positions=256,
                        assistant_targets=targets,
                    )
                    local["accepted"] += 1
            counts.update(local)
            sources.append(
                {"path": name, "sha256": file_sha256(path), "counts": dict(local)}
            )
            print(json.dumps(sources[-1]), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    entries = {}
    for split, values in splits.items():
        entries[split] = dict(values)
        for kind in ("tokens", "labels"):
            path = directory / f"{split}-{kind}.bin"
            entries[split][kind] = {"path": path.name, "sha256": file_sha256(path)}
    write_json(
        directory / "metadata.json",
        {
            "status": "complete",
            "kind": "assistant-only-sft",
            "dataset": IT_DATASET,
            "revision": IT_REVISION,
            "tokenizer": REFERENCE_MODEL,
            "tokenizer_revision": REFERENCE_REVISION,
            "sequence_length": 256,
            "vocab_size": len(tokenizer),
            "bos_token_id": 1,
            "pad_token_id": 0,
            "eos_token_id": tokenizer.eos_token_id,
            "sources": sources,
            "counts": dict(counts),
            "splits": entries,
            "format": (
                "BOS + separately tokenized Role:\\n/content; "
                "EOS after each assistant reply"
            ),
            "loss": "shift labels once; assistant content and EOS only; -100 elsewhere",
            "partition": (
                "normalized conversation hash; 98% train / 1% validation / 1% test"
            ),
            "license": (
                "SmolTalk2 component/source licenses apply separately; "
                "preserve upstream card"
            ),
            "freshness": (
                "Pinned candidate updated October 2025; not claimed collected in 2026"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
