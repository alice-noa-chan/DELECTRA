import pytest

from deletcra.config import generator_model_config
from deletcra.data import pack_texts, prepare_tinystories
from deletcra.model import CausalElectra
from deletcra.target import profile_command, story_model_config


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


def test_story_generator_preserves_the_executed_pilot_parameter_counts():
    main = CausalElectra(story_model_config())
    generator = CausalElectra(generator_model_config(story_model_config()))
    assert sum(parameter.numel() for parameter in generator.parameters()) == 9487625
    main.share_generator_embeddings(generator)
    parameters = {id(p): p for p in [*main.parameters(), *generator.parameters()]}
    assert sum(parameter.numel() for parameter in parameters.values()) == 15239402


def test_story_downloads_are_opt_in_and_memory_budgets_bounded(tmp_path):
    with pytest.raises(ValueError, match="allow-download"):
        prepare_tinystories(tmp_path / "data")
    with pytest.raises(ValueError, match="capped"):
        prepare_tinystories(
            tmp_path / "data", max_train_tokens=100000001, allow_download=True
        )
    assert not (tmp_path / "data").exists()


def test_target_profile_is_bounded_and_shares_only_the_joint_generator():
    for mode in ("clm", "joint"):
        command = profile_command(mode, "/data", "/runs")
        assert command[command.index("--steps") + 1] == "300"
        assert command[command.index("--preset") + 1] == "story15m"
        assert ("--share-embeddings" in command) == (mode == "joint")
