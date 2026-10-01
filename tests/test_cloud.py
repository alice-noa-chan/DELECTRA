import pytest

from deletcra.cloud import PilotConfig, validate_run_id


def test_pilot_command_uses_all_objectives_actual_data_and_cuda_bf16():
    command = PilotConfig(steps=100, batch_size=32, probe_steps=20).command(
        "/vol/data", "/vol/runs/pilot"
    )
    assert command[:3] == ["-m", "deletcra", "run"]
    options = dict(zip(command[3::2], command[4::2], strict=True))
    assert options["--mode"] == "all"
    assert options["--preset"] == "small"
    assert options["--data-dir"] == "/vol/data"
    assert options["--device"] == "cuda"
    assert options["--precision"] == "bf16"
    assert options["--steps"] == "100"


@pytest.mark.parametrize(
    "settings",
    [
        {"steps": 0},
        {"steps": 501},
        {"batch_size": 129},
        {"probe_steps": 101},
        {"seed": -1},
        {"seed": 2**63},
        {"train_tokens": 0},
        {"train_tokens": 10_000_001},
        {"train_tokens": True},
    ],
)
def test_pilot_rejects_unbounded_or_invalid_requests(settings):
    with pytest.raises(ValueError):
        PilotConfig(**settings)


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "", "a" * 81])
def test_cloud_run_ids_cannot_escape_artifact_paths(run_id):
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_cloud_run_ids_allow_descriptive_unique_names():
    validate_run_id("modal-l40s-wikitext2-20261001-pilot")


def test_explicit_token_budget_replaces_step_budget_without_ambiguity():
    command = PilotConfig(train_tokens=10_000_000).command("/data", "/results")
    assert "--steps" not in command
    assert command[command.index("--train-tokens") + 1] == "10000000"
