import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deletcra.benchmark import score_language_model


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
