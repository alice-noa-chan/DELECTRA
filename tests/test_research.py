import pytest

from deletcra.research import SEEDS, research_command


def test_research_matrix_matches_data_evaluation_and_seed_and_caps_time_control():
    for seed in SEEDS:
        commands = {
            stage: research_command(
                seed,
                stage,
                "/data",
                f"/runs/{stage}",
                time_budget=242.5 if stage == "clm_time" else None,
            )
            for stage in ("clm", "joint", "clm_time")
        }
        for command in commands.values():
            assert command[command.index("--eval-batches") + 1] == "32"
            assert command[command.index("--seed") + 1] == str(seed)
        assert commands["clm_time"][-2:] == ["--max-training-seconds", "242.5"]
        assert (
            commands["joint"][commands["joint"].index("--train-tokens") + 1]
            == "10000000"
        )


@pytest.mark.parametrize(
    "seed,stage,budget",
    [
        (1, "joint", None),
        (7, "all", None),
        (7, "clm_time", None),
        (7, "clm_time", 301),
        (7, "clm_time", float("nan")),
        (7, "joint", 1),
    ],
)
def test_research_rejects_unplanned_or_unbounded_stages(seed, stage, budget):
    with pytest.raises(ValueError):
        research_command(seed, stage, "/data", "/runs", time_budget=budget)
