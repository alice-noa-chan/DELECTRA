import json
from dataclasses import replace

import pytest
import torch

from deletcra import ModelConfig
from deletcra.cli import main
from deletcra.data import synthetic_sequences
from deletcra.experiment import TrainConfig, frozen_probe, run_experiment
from deletcra.model import CausalElectra
from deletcra.objectives import ObjectiveConfig


@pytest.mark.parametrize("mode", ["clm", "rtd", "joint"])
def test_experiment_trains_evaluates_and_writes_reloadable_artifacts(tmp_path, mode):
    train = synthetic_sequences(16, 6, 8, seed=1)
    validation = synthetic_sequences(8, 6, 8, seed=2)
    directory = tmp_path / mode
    report = run_experiment(
        train,
        validation,
        ModelConfig(vocab_size=8),
        ObjectiveConfig(mode=mode),
        TrainConfig(steps=3, batch_size=4, eval_batches=2, probe_steps=2),
        directory,
    )
    assert report == json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert report["training_tokens"] == 3 * 4 * 5
    assert report["input_tokens_per_second"] > 0
    assert report["peak_cuda_allocated_bytes"] is None
    assert ("perplexity" in report["final_validation"]) == (mode != "rtd")
    assert report["frozen_probe"]["perplexity"] > 0
    restored = CausalElectra.load(directory / "model").eval()
    with torch.no_grad():
        assert torch.isfinite(restored(validation).rtd_logits).all()
    if mode != "clm":
        assert (directory / "generator" / "model.pt").exists()
        metrics = report["final_validation"]
        assert sum(metrics["rtd_confusion"].values()) == 8 * 5
        assert 0 <= metrics["rtd_f1"] <= 1


def test_clm_learning_reduces_held_out_loss_and_runs_are_seeded(tmp_path):
    train = synthetic_sequences(32, 8, 8, seed=10)
    validation = synthetic_sequences(16, 8, 8, seed=11)
    settings = TrainConfig(steps=80, batch_size=8, learning_rate=0.005, probe_steps=0)
    first = run_experiment(
        train,
        validation,
        ModelConfig(vocab_size=8),
        ObjectiveConfig(mode="clm"),
        settings,
        tmp_path / "first",
    )
    assert first["final_validation"]["lm_loss"] < (
        first["initial_validation"]["lm_loss"] * 0.8
    )
    second = run_experiment(
        train,
        validation,
        ModelConfig(vocab_size=8),
        ObjectiveConfig(mode="clm"),
        replace(settings, steps=1),
        tmp_path / "second",
    )
    assert first["history"][0] == second["history"][0]
    assert first["initial_validation"] == second["initial_validation"]


def test_frozen_probe_does_not_modify_pretrained_weights():
    model = CausalElectra(ModelConfig(vocab_size=8))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    data = synthetic_sequences(8, 6, 8, seed=1)
    metrics, _ = frozen_probe(
        model, data, data, TrainConfig(batch_size=4, probe_steps=3)
    )
    assert metrics["steps"] == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_rtd_learns_replacement_detection_with_a_controlled_noise_budget(tmp_path):
    # Default low-temperature toy generators quickly eliminate positive labels.
    # This control checks actual detection learning rather than majority accuracy.
    train = synthetic_sequences(256, 16, 16, seed=17)
    validation = synthetic_sequences(64, 16, 16, seed=18)
    report = run_experiment(
        train,
        validation,
        ModelConfig(vocab_size=16),
        ObjectiveConfig(mode="rtd", replacement_probability=0.5, temperature=5),
        TrainConfig(steps=1000, batch_size=16, probe_steps=0),
        tmp_path / "rtd-control",
    )
    metrics = report["final_validation"]
    assert metrics["rtd_f1"] > 0.4
    assert metrics["rtd_balanced_accuracy"] > 0.6
    assert metrics["rtd_accuracy"] > metrics["rtd_majority_baseline"] + 0.05


def test_cli_runs_all_modes_and_refuses_to_overwrite(tmp_path, capsys):
    output = tmp_path / "run"
    argv = [
        "run",
        "--output-dir",
        str(output),
        "--train-tokens",
        "20",
        "--batch-size",
        "2",
        "--sequence-length",
        "6",
        "--vocab-size",
        "8",
        "--probe-steps",
        "0",
        "--eval-batches",
        "1",
        "--quiet",
    ]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [row["mode"] for row in summary["results"]] == ["clm", "rtd", "joint"]
    assert all(row["training_tokens"] == 20 for row in summary["results"])
    assert (output / "summary.json").exists()
    with pytest.raises(SystemExit):
        main(argv)


def test_run_rejects_context_overflow_before_creating_outputs(tmp_path):
    data = synthetic_sequences(8, 6, 8, seed=1)
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="context"):
        run_experiment(
            data,
            data,
            ModelConfig(vocab_size=8, max_positions=4),
            ObjectiveConfig(mode="clm"),
            TrainConfig(steps=1),
            directory,
        )
    assert not directory.exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"steps": 0},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"probe_steps": -1},
        {"device": "xpu"},
        {"precision": "fp16"},
        {"precision": "bf16", "device": "cpu"},
    ],
)
def test_invalid_training_settings(settings):
    with pytest.raises(ValueError):
        TrainConfig(**settings)
