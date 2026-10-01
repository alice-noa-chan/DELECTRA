from copy import deepcopy

import pytest

from analyze_research import summarize
from deletcra.research import SEEDS, STAGES


@pytest.fixture
def complete_matrix():
    runs = []
    for seed in SEEDS:
        reports = {}
        for stage, ppl, seconds, tokens in (
            ("joint", 200, 240, 10000000),
            ("clm", 250, 110, 10000000),
            ("clm_time", 150, 240.01, 22000000),
        ):
            reports[stage] = {
                "train_config": {
                    "seed": seed,
                    "max_training_seconds": 240 if stage == "clm_time" else None,
                },
                "model_config": {"hidden_size": 256},
                "dataset": {"kind": "fixture"},
                "final_validation": {"perplexity": ppl, "next_token_targets": 99949},
                "frozen_probe": {"perplexity": ppl * 2},
                "training_tokens": tokens,
                "training_seconds": seconds,
                "stop_reason": "time_budget" if stage == "clm_time" else "step_budget",
            }
        runs.append(
            {
                "seed": seed,
                "reports": reports,
                "remote_gpu_function_seconds": 600,
                "estimated_gpu_charge_usd": 0.3252,
            }
        )
    return {
        "run_id": "fixture",
        "seeds": runs,
        "diagnostics": {
            "remote_gpu_function_seconds": 10,
            "estimated_gpu_charge_usd": 0.00542,
        },
    }


def test_paired_analysis_distinguishes_token_gain_from_time_control(complete_matrix):
    summary = summarize(complete_matrix)
    row = summary["paired_results"][0]
    assert row["joint_ppl_change_vs_token_clm_percent"] == pytest.approx(-20)
    assert row["joint_ppl_change_vs_time_clm_percent"] == pytest.approx(100 / 3)
    assert summary["validation_targets_per_model"] == 99949
    assert summary["total_remote_gpu_seconds"] == 1810
    assert set(summary["aggregates"]) == set(STAGES)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_seed",
        "duplicate_seed",
        "time_cap",
        "time_mismatch",
        "validation",
        "tokens",
    ],
)
def test_analysis_rejects_incomplete_or_unmatched_controls(complete_matrix, problem):
    result = deepcopy(complete_matrix)
    if problem == "missing_seed":
        result["seeds"].pop()
    elif problem == "duplicate_seed":
        result["seeds"][2] = result["seeds"][0]
    else:
        reports = result["seeds"][0]["reports"]
        if problem == "time_cap":
            reports["clm_time"]["stop_reason"] = "step_budget"
        elif problem == "time_mismatch":
            reports["clm_time"]["training_seconds"] = 250
        elif problem == "validation":
            reports["clm"]["final_validation"]["next_token_targets"] = 16256
        else:
            reports["clm"]["training_tokens"] = 999
    with pytest.raises(ValueError):
        summarize(result)
