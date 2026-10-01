import pytest
import torch
from transformers import ElectraForPreTraining

from deletcra import ModelConfig
from deletcra.model import CausalElectra, electra_config


def test_future_tokens_cannot_change_prefix_outputs():
    torch.manual_seed(7)
    model = CausalElectra(ModelConfig()).eval()
    original = torch.tensor([[1, 4, 5, 6, 7]])
    altered = torch.tensor([[1, 4, 5, 21, 22]])
    with torch.no_grad():
        first, second = model(original), model(altered)
    for name in ("hidden_states", "rtd_logits", "lm_logits"):
        torch.testing.assert_close(
            getattr(first, name)[:, :3], getattr(second, name)[:, :3], rtol=0, atol=0
        )
    assert not torch.equal(first.hidden_states[:, 3:], second.hidden_states[:, 3:])
    assert model.config.is_decoder and not model.config.add_cross_attention


def test_padding_content_cannot_change_visible_outputs():
    model = CausalElectra(ModelConfig()).eval()
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    with torch.no_grad():
        padded = model(torch.tensor([[1, 3, 4, 0, 0]]), mask)
        altered = model(torch.tensor([[1, 3, 4, 15, 20]]), mask)
        short = model(torch.tensor([[1, 3, 4]]))
    torch.testing.assert_close(padded.lm_logits[:, :3], altered.lm_logits[:, :3])
    torch.testing.assert_close(padded.lm_logits[:, :3], short.lm_logits)


def test_save_load_preserves_outputs_and_embedding_tie(tmp_path):
    model = CausalElectra(ModelConfig()).eval()
    tokens = torch.tensor([[1, 4, 5]])
    model.save(tmp_path)
    restored = CausalElectra.load(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(tokens).lm_logits, restored(tokens).lm_logits)
    assert restored.lm_head.weight is restored.electra.embeddings.word_embeddings.weight


def test_encoder_conversion_preserves_weights_and_removes_future_access(tmp_path):
    config = electra_config(ModelConfig())
    config.is_decoder = False
    encoder = ElectraForPreTraining(config)
    encoder.save_pretrained(tmp_path)
    converted = CausalElectra.from_encoder_checkpoint(tmp_path, bos_token_id=1).eval()
    for key, value in encoder.electra.state_dict().items():
        torch.testing.assert_close(converted.electra.state_dict()[key], value)
    for key, value in encoder.discriminator_predictions.state_dict().items():
        torch.testing.assert_close(converted.rtd_head.state_dict()[key], value)
    with torch.no_grad():
        first = converted(torch.tensor([[1, 4, 5]]))
        second = converted(torch.tensor([[1, 4, 8]]))
    torch.testing.assert_close(first.rtd_logits[:, :2], second.rtd_logits[:, :2])


def test_generation_preserves_prefix_excludes_special_tokens_and_restores_mode():
    model = CausalElectra(ModelConfig())
    prefix = torch.tensor([[1, 4]])
    generated = model.generate(prefix, max_new_tokens=3)
    assert generated.shape == (1, 5)
    assert torch.equal(generated[:, :2], prefix)
    assert not torch.isin(generated[:, 2:], torch.tensor([0, 1])).any()
    assert model.training


@pytest.mark.parametrize("mask", [[[0, 1, 1]], [[1, 0, 1]], [[1, 2, 1]], [[1, 1]]])
def test_invalid_attention_masks_are_rejected(mask):
    model = CausalElectra(ModelConfig())
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 4, 5]]), torch.tensor(mask))


def test_generation_rejects_context_overflow_and_padding():
    model = CausalElectra(ModelConfig(max_positions=4))
    with pytest.raises(ValueError, match="exceed"):
        model.generate(torch.tensor([[1, 3]]), max_new_tokens=3)
    with pytest.raises(ValueError, match="unpadded"):
        model.generate(torch.tensor([[1, 3, 0]]), max_new_tokens=1)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_last_position_projection_preserves_logits_and_greedy_tokens(backend):
    torch.manual_seed(12)
    model = CausalElectra(ModelConfig(attention_backend=backend)).eval()
    prefix = torch.tensor([[1, 3, 4, 5], [1, 8, 9, 10]])
    with torch.no_grad():
        full = model(prefix)
        last = model(prefix, logits_to_keep=1, compute_rtd=False)
        torch.testing.assert_close(last.lm_logits, full.lm_logits[:, -1:])
        assert last.rtd_logits is None
        torch.testing.assert_close(last.hidden_states, full.hidden_states)
        expected = prefix.clone()
        for _ in range(4):
            logits = model(expected).lm_logits[:, -1].clone()
            logits[:, [0, 1]] = -torch.inf
            expected = torch.cat((expected, logits.argmax(-1, keepdim=True)), dim=1)
    projection_shapes, rtd_calls = [], []
    projection_hook = model.lm_projection.register_forward_pre_hook(
        lambda _, inputs: projection_shapes.append(inputs[0].shape)
    )
    rtd_hook = model.rtd_head.register_forward_pre_hook(
        lambda *_: rtd_calls.append(True)
    )
    actual = model.generate(prefix, max_new_tokens=4)
    projection_hook.remove()
    rtd_hook.remove()
    assert torch.equal(actual, expected)
    assert projection_shapes == [torch.Size([2, 1, 32])] * 4
    assert rtd_calls == []


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_invalid_projection_position_count_is_rejected(invalid):
    with pytest.raises(ValueError, match="nonnegative integer"):
        CausalElectra(ModelConfig())(torch.tensor([[1, 3]]), logits_to_keep=invalid)
