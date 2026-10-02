"""Greedy generation from the retained Colab trainer; no checkpoint selection."""

import json
from pathlib import Path

import torch

record = json.loads(Path("/content/stories-generation-prompts.json").read_text())
trainer = globals()["window_trainer"]
outputs = []
for prompt in record["prompts"]:
    inputs = torch.tensor([prompt["input_ids"]], device="cuda")
    with torch.no_grad(), trainer.runtime.autocast():
        tokens = trainer.model.generate(inputs, max_new_tokens=64, eos_token_id=2)
    assert torch.equal(tokens[0, : inputs.shape[1]], inputs[0])
    outputs.append(
        {
            "prompt": prompt["text"],
            "input_ids": prompt["input_ids"],
            "output_ids": tokens[0].tolist(),
        }
    )
report = {
    "scope": "greedy_partial_checkpoint_generation_diagnostic",
    "state": trainer.state,
    "tokenizer": record["tokenizer"],
    "tokenizer_revision": record["revision"],
    "samples": outputs,
    "test_split_used": False,
    "release_quality_evaluated": False,
}
Path("/content/stories-generation-diagnostic.json").write_text(
    json.dumps(report, indent=2)
)
print("Three generation calls completed; all prompt prefixes preserved")
