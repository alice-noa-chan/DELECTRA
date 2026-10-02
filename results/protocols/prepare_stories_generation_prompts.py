"""Encode three predetermined diagnostic prompts with the pinned local tokenizer."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from deletcra.target import REFERENCE_MODEL, REFERENCE_REVISION

tokenizer = AutoTokenizer.from_pretrained(
    REFERENCE_MODEL, revision=REFERENCE_REVISION, local_files_only=True
)
prompts = [
    "Once upon a time, there was a little rabbit.",
    "Lily found a shiny key in the garden.",
    "Tom wanted to help his friend who was sad.",
]
record = {
    "tokenizer": REFERENCE_MODEL,
    "revision": REFERENCE_REVISION,
    "max_new_tokens": 64,
    "prompts": [
        {
            "text": text,
            "input_ids": [1, *tokenizer.encode(text, add_special_tokens=False)],
        }
        for text in prompts
    ],
}
Path("runs/stories-generation-prompts.json").write_text(json.dumps(record, indent=2))
