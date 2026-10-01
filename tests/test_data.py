import json

import pytest
import torch

from deletcra.data import (
    load_prepared,
    pack_texts,
    prepare_wikitext,
    synthetic_sequences,
)


class TinyTokenizer:
    cls_token_id = 1
    pad_token_id = 0
    sep_token_id = 2

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [int(token) for token in text.split()]}


def test_synthetic_data_is_reproducible_learnable_and_uses_no_padding():
    first = synthetic_sequences(8, 8, 12, seed=4)
    second = synthetic_sequences(8, 8, 12, seed=4)
    other = synthetic_sequences(8, 8, 12, seed=9)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert (first[:, 0] == 1).all()
    assert (first[:, 1:] >= 2).all()
    assert (first[:, 1:] < 12).all()
    assert torch.equal((first[:, 1:-1] - 2 + 1) % 10 + 2, first[:, 2:])


def test_packing_adds_bos_and_separators_respects_budget_and_drops_tail():
    result = pack_texts(
        ["", "3 4", "5 6 7 8 9"], TinyTokenizer(), sequence_length=4, max_tokens=7
    )
    assert result.tolist() == [[1, 3, 4, 2], [1, 5, 6, 7]]


def test_packing_and_download_preparation_reject_invalid_requests(tmp_path):
    with pytest.raises(ValueError, match="too few"):
        pack_texts([""], TinyTokenizer(), sequence_length=4, max_tokens=9)
    with pytest.raises(ValueError, match="allow-download"):
        prepare_wikitext(tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_load_prepared_round_trip_and_rejects_corrupt_tokens(tmp_path):
    metadata = {
        "sequence_length": 4,
        "vocab_size": 8,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "special_token_ids": [0, 1, 2],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    tokens = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 7]])
    for split in ("train", "validation"):
        torch.save(tokens, tmp_path / f"{split}.pt")
    train, validation, restored = load_prepared(tmp_path)
    assert restored == metadata
    assert torch.equal(train, validation)
    assert torch.equal(train, tokens)
    torch.save(torch.tensor([[1, 3, 4, 9]]), tmp_path / "train.pt")
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        load_prepared(tmp_path)
