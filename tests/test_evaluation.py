import torch
from test_production_training import Corpus

from deletcra.config import ModelConfig
from deletcra.data import synthetic_sequences
from deletcra.evaluation import evaluate_frozen
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig, causal_lm_loss


def test_frozen_score_weights_partial_batch_and_marks_coverage():
    torch.manual_seed(7)
    model = CausalElectra(ModelConfig(max_positions=8))
    corpus = Corpus(synthetic_sequences(5, 8, 32, seed=7))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = evaluate_frozen(
        model, corpus, batch_size=2, objective=ObjectiveConfig(mode="clm")
    )
    model.eval()
    tokens = corpus.tokens
    with torch.no_grad():
        expected = causal_lm_loss(model(tokens).lm_logits, tokens, tokens.ne(0))
    assert abs(report["nll"] - float(expected)) < 1e-6
    assert report["full_split"] and report["supervised_targets"] == 35
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    partial = evaluate_frozen(
        model, corpus, batch_size=2, max_blocks=3, objective=ObjectiveConfig(mode="clm")
    )
    assert not partial["full_split"] and partial["blocks"] == 3
