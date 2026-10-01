import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deletcra.benchmark import load_story_candidate, score_language_model
from deletcra.config import ModelConfig
from deletcra.model import CausalElectra


class UniformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert not use_cache
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 8)))


def test_common_scoring_uses_shifted_targets_and_weights_partial_batches_and_padding():
    data = torch.tensor([[1, 3, 4, 5], [1, 6, 0, 0], [1, 3, 4, 0]])
    first = score_language_model(UniformModel(), data, pad_token_id=0, batch_size=2)
    second = score_language_model(UniformModel(), data, pad_token_id=0, batch_size=3)
    assert first["next_token_targets"] == 6
    assert first["nll"] == pytest.approx(math.log(8))
    assert first["perplexity"] == pytest.approx(8)
    assert first["nll"] == pytest.approx(second["nll"])


def test_candidate_loading_requires_trained_lm_and_matching_data(tmp_path):
    metadata = {
        "dataset": "stories",
        "dataset_revision": "fixed-data",
        "tokenizer": "reference",
        "tokenizer_revision": "fixed-tokenizer",
        "vocab_size": 16,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "sequence_length": 4,
    }
    directory = tmp_path / "model"
    CausalElectra(ModelConfig(vocab_size=16)).save(directory)
    report_path = tmp_path / "report.json"
    report = {"objective_config": {"mode": "clm"}, "dataset": metadata}
    report_path.write_text(json.dumps(report))
    model = load_story_candidate(directory, metadata)
    assert not model.training
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_story_candidate(directory, {**metadata, "tokenizer_revision": "changed"})
    report["objective_config"]["mode"] = "rtd"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="no trained main LM"):
        load_story_candidate(directory, metadata)
