"""Offline synthetic batches and opt-in WikiText/TinyStories preparation."""

import json
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def synthetic_sequences(
    count: int,
    sequence_length: int,
    vocab_size: int,
    *,
    seed: int,
) -> Tensor:
    """Learnable cyclic sequences: BOS, start, start+1, ... with wraparound.

    Training and validation use independent sampled starts. They share a simple
    grammar, so this checks optimization, not generalization to real language.
    """
    if count < 1 or sequence_length < 3 or vocab_size < 4:
        raise ValueError("need positive count, length >= 3, and vocabulary >= 4")
    rng = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, vocab_size - 2, (count, 1), generator=rng)
    offsets = torch.arange(sequence_length - 1).unsqueeze(0)
    content = (starts + offsets) % (vocab_size - 2) + 2
    return torch.cat((torch.ones(count, 1, dtype=torch.long), content), dim=1)


def pack_texts(
    texts: Iterable[str],
    tokenizer: Any,
    *,
    sequence_length: int,
    max_tokens: int,
) -> Tensor:
    """Pack independent split text into full BOS-prefixed blocks, drop the tail.

    SEP separates records when available, otherwise BOS does. ``max_tokens``
    counts content and record separators before inserting a BOS per block.
    Each block therefore holds ``sequence_length - 1`` stream tokens plus BOS.
    Call this function separately for each split so no block mixes their texts.
    """
    if sequence_length < 3 or max_tokens < sequence_length - 1:
        raise ValueError("length >= 3 and a budget of at least one block are required")
    bos_id = getattr(tokenizer, "cls_token_id", None)
    if bos_id is None:
        bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define CLS/BOS and PAD")
    separator = getattr(tokenizer, "sep_token_id", None)
    if separator is None:
        separator = bos_id
    # Compact storage avoids millions of boxed Python integers for larger corpora.
    content = array("i")
    for text in texts:
        if not text.strip():
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens.append(separator)
        remaining = max_tokens - len(content)
        content.extend(tokens[:remaining])
        if len(content) >= max_tokens:
            break
    block_size = sequence_length - 1
    block_count = len(content) // block_size
    if block_count == 0:
        raise ValueError("text split contains too few tokens for one complete block")
    # Drop the incomplete tail, then widen compact int32 storage to the int64
    # token IDs required by PyTorch embeddings. Prefix each full block with BOS.
    data = torch.frombuffer(content, dtype=torch.int32)[
        : block_count * block_size
    ].long()
    content_tensor = data.reshape(block_count, block_size)
    bos = torch.full((block_count, 1), bos_id, dtype=torch.long)
    return torch.cat((bos, content_tensor), dim=1)


def load_prepared(directory: str | Path) -> tuple[Tensor, Tensor, dict]:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    splits = []
    for split in ("train", "validation"):
        tokens = torch.load(directory / f"{split}.pt", weights_only=True)
        if (
            not isinstance(tokens, Tensor)
            or tokens.dtype != torch.long
            or tokens.ndim != 2
            or min(tokens.shape) < 1
            or tokens.shape[1] != metadata["sequence_length"]
        ):
            raise ValueError(f"invalid token tensor in {split}.pt")
        if tokens.min() < 0 or tokens.max() >= metadata["vocab_size"]:
            raise ValueError(f"out-of-vocabulary token in {split}.pt")
        if not tokens[:, 0].eq(metadata["bos_token_id"]).all():
            raise ValueError(f"sequences in {split}.pt must begin with BOS")
        splits.append(tokens)
    return splits[0], splits[1], metadata


def prepare_wikitext(
    directory: str | Path,
    *,
    subset: str = "wikitext-2-raw-v1",
    tokenizer_name: str = "google/electra-small-discriminator",
    sequence_length: int = 128,
    max_train_tokens: int = 1_000_000,
    max_validation_tokens: int = 100_000,
    allow_download: bool = False,
) -> dict:
    """Opt-in network preparation; train and validation only, test stays untouched."""
    if not allow_download:
        raise ValueError("preparation downloads data; pass --allow-download explicitly")
    if subset not in {"wikitext-2-raw-v1", "wikitext-103-raw-v1"}:
        raise ValueError("unsupported WikiText subset")
    if sequence_length < 3 or min(max_train_tokens, max_validation_tokens) < (
        sequence_length - 1
    ):
        raise ValueError("token budgets must each fit at least one complete block")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite prepared data: {directory}")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'install the data extra: pip install -e ".[data]"'
        ) from error
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    metadata = {
        "dataset": "Salesforce/wikitext",
        "subset": subset,
        "dataset_source": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license_note": (
            "WikiText has separate CC BY-SA/GFDL notices. Its current card lists "
            "CC BY-SA 3.0/GFDL in metadata and CC BY-SA 4.0 in prose; preserve "
            "the upstream card and review that inconsistency before redistribution."
        ),
        "tokenizer": tokenizer_name,
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.cls_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "special_token_ids": tokenizer.all_special_ids,
        "sequence_length": sequence_length,
        "max_train_tokens": max_train_tokens,
        "max_validation_tokens": max_validation_tokens,
    }
    tensors = {}
    for split, budget in (
        ("train", max_train_tokens),
        ("validation", max_validation_tokens),
    ):
        dataset = load_dataset(
            "Salesforce/wikitext", subset, split=split, streaming=True
        )
        tensors[split] = pack_texts(
            (row["text"] for row in dataset),
            tokenizer,
            sequence_length=sequence_length,
            max_tokens=budget,
        )
        metadata[f"{split}_content_tokens"] = tensors[split].shape[0] * (
            sequence_length - 1
        )
    directory.mkdir(parents=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    for split, tensor in tensors.items():
        torch.save(tensor, directory / f"{split}.pt")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def prepare_tinystories(
    directory: str | Path,
    *,
    sequence_length: int = 256,
    max_train_tokens: int = 100_000_000,
    max_validation_tokens: int = 1_000_000,
    allow_download: bool = False,
) -> dict:
    """Pinned TinyStories parquet splits and the reference's exact tokenizer."""
    if not allow_download:
        raise ValueError("preparation downloads data; pass --allow-download explicitly")
    if not 3 <= sequence_length <= 256:
        raise ValueError("TinyStories target context must be in [3, 256]")
    if min(max_train_tokens, max_validation_tokens) < sequence_length - 1:
        raise ValueError("budgets must fit a complete block")
    if max_train_tokens > 100_000_000 or max_validation_tokens > 5_000_000:
        raise ValueError("in-memory preparation capped at 100M train / 5M validation")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite prepared data: {directory}")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from deletcra.target import (
        REFERENCE_MODEL,
        REFERENCE_REVISION,
        STORIES_DATASET,
        STORIES_REVISION,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        REFERENCE_MODEL, revision=REFERENCE_REVISION
    )
    # Llama's reserved UNK=0 is used only for padding; byte fallback must keep
    # real English content free of UNK/PAD. Training validates visible sequences.
    tokenizer.pad_token = tokenizer.unk_token
    tensors = {}
    for split, budget in (
        ("train", max_train_tokens),
        ("validation", max_validation_tokens),
    ):
        dataset = load_dataset(
            STORIES_DATASET, revision=STORIES_REVISION, split=split, streaming=True
        )
        tensors[split] = pack_texts(
            (row["text"] for row in dataset),
            tokenizer,
            sequence_length=sequence_length,
            max_tokens=budget,
        )
        if tensors[split].eq(tokenizer.pad_token_id).any():
            raise ValueError("reference tokenizer produced UNK/PAD in content")
    metadata = {
        "dataset": STORIES_DATASET,
        "dataset_revision": STORIES_REVISION,
        "tokenizer": REFERENCE_MODEL,
        "tokenizer_revision": REFERENCE_REVISION,
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "special_token_ids": tokenizer.all_special_ids,
        "sequence_length": sequence_length,
        "train_content_tokens": tensors["train"].shape[0] * (sequence_length - 1),
        "validation_content_tokens": tensors["validation"].shape[0]
        * (sequence_length - 1),
        "license_note": (
            "TinyStories data is CDLA-Sharing-1.0; reference model card labels MIT. "
            "These are separate from project licenses."
        ),
        "protocol": (
            "BOS-prefixed packed blocks; BOS separates stories, no appended EOS. "
            "Official HF train/validation; reference pretraining overlap is unknown. "
            "Different from upstream's private shard evaluation."
        ),
    }
    directory.mkdir(parents=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    for split, tensor in tensors.items():
        torch.save(tensor, directory / f"{split}.pt")
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
