from dataclasses import replace

import pytest
import torch

from deletcra.config import ModelConfig, generator_model_config
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, pretraining_step


@pytest.mark.parametrize("padded", [False, True])
def test_sdpa_preserves_electra_outputs_and_gradients(padded):
    settings = ModelConfig()
    eager = CausalElectra(settings)
    sdpa = CausalElectra(replace(settings, attention_backend="sdpa"))
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    assert sdpa.state_dict().keys() == eager.state_dict().keys()
    tokens = torch.tensor([[1, 3, 4, 5], [1, 4, 5, 0 if padded else 6]])
    first, second = eager(tokens), sdpa(tokens)
    for name in ("hidden_states", "lm_logits", "rtd_logits"):
        torch.testing.assert_close(
            getattr(first, name), getattr(second, name), rtol=1e-5, atol=2e-6
        )
    first.lm_logits.square().mean().backward()
    second.lm_logits.square().mean().backward()
    for (name, original), (_, optimized) in zip(
        eager.named_parameters(), sdpa.named_parameters(), strict=True
    ):
        if original.grad is not None:
            torch.testing.assert_close(
                original.grad, optimized.grad, rtol=1e-4, atol=2e-6, msg=name
            )


def test_sdpa_masks_future_and_padded_tokens():
    model = CausalElectra(ModelConfig(attention_backend="sdpa")).eval()
    tokens = torch.tensor([[1, 3, 4, 5, 0]])
    mask = tokens.ne(0)
    changed = torch.tensor([[1, 3, 4, 9, 20]])
    with torch.no_grad():
        first, second = model(tokens, mask), model(changed, mask)
        short = model(tokens[:, :4])
    torch.testing.assert_close(
        first.lm_logits[:, :3], second.lm_logits[:, :3], rtol=0, atol=0
    )
    torch.testing.assert_close(first.lm_logits[:, :4], short.lm_logits)


@pytest.mark.parametrize("mode", ["clm", "rtd", "joint"])
def test_sdpa_objectives_keep_sampling_and_gradient_routing(mode):
    settings = ModelConfig(attention_backend="sdpa")
    main = CausalElectra(settings)
    generator = CausalElectra(generator_model_config(settings))
    tokens = torch.tensor([[1, 3, 4, 5], [1, 5, 6, 0]])
    output = pretraining_step(
        main, generator, tokens, tokens.ne(0), ObjectiveConfig(mode=mode)
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert (main.rtd_head.dense.weight.grad is not None) == (mode != "clm")
    assert (main.lm_projection[0].weight.grad is not None) == (mode != "rtd")
    assert (generator.lm_projection[0].weight.grad is not None) == (mode != "clm")


def test_sdpa_roundtrip_and_explicit_flash_checkpoint_cpu_override(tmp_path):
    model = CausalElectra(ModelConfig(attention_backend="sdpa")).eval()
    model.save(tmp_path / "sdpa")
    restored = CausalElectra.load(tmp_path / "sdpa").eval()
    tokens = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(model(tokens).lm_logits, restored(tokens).lm_logits)
    assert restored.attention_backend == "sdpa"
    flash = CausalElectra(ModelConfig(attention_backend="flash"))
    flash.save(tmp_path / "flash")
    inspected = CausalElectra.load(tmp_path / "flash", attention_backend="sdpa")
    assert inspected.attention_backend == "sdpa"
    for name, weights in flash.state_dict().items():
        torch.testing.assert_close(
            weights, inspected.state_dict()[name], rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="CUDA"):
        flash(tokens)


def test_sdpa_evaluation_disables_attention_dropout():
    model = CausalElectra(ModelConfig(attention_backend="sdpa", dropout=0.2)).eval()
    tokens = torch.tensor([[1, 3, 4]])
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        first, second = model(tokens), model(tokens)
    torch.testing.assert_close(first.lm_logits, second.lm_logits, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
