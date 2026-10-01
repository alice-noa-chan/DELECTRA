import pytest

from deletcra.data import pack_texts, prepare_tinystories
from deletcra.model import CausalElectra
from deletcra.target import story_model_config


class LlamaLikeTokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": [int(word) for word in text.split()]}


def test_story_packing_uses_bos_for_llama_tokenizers_without_cls_or_sep():
    tokens = pack_texts(
        ["3 4", "5 6"], LlamaLikeTokenizer(), sequence_length=4, max_tokens=6
    )
    assert tokens.tolist() == [[1, 3, 4, 1], [1, 5, 6, 1]]


def test_story_target_has_reference_scale_and_context():
    model = CausalElectra(story_model_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == 15041505
    assert model.config.max_position_embeddings == 256
    assert model.config.is_decoder


def test_story_downloads_are_opt_in_and_memory_budgets_bounded(tmp_path):
    with pytest.raises(ValueError, match="allow-download"):
        prepare_tinystories(tmp_path / "data")
    with pytest.raises(ValueError, match="capped"):
        prepare_tinystories(
            tmp_path / "data", max_train_tokens=100000001, allow_download=True
        )
    assert not (tmp_path / "data").exists()
