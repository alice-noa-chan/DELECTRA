"""Summarize a complete research matrix; optionally render a static research plot."""

import argparse
import json
import math
import statistics
from pathlib import Path

from deletcra.research import SEEDS, STAGES


def summarize(result: dict) -> dict:
    runs = result["seeds"]
    if sorted(run["seed"] for run in runs) != sorted(SEEDS):
        raise ValueError("analysis requires each predeclared seed exactly once")
    if result.get("diagnostics") is None:
        raise ValueError("analysis requires the completed checkpoint diagnostics")
    rows = []
    for run in sorted(runs, key=lambda run: run["seed"]):
        reports = run["reports"]
        if set(reports) != set(STAGES):
            raise ValueError("research stages are incomplete")
        joint, clm, timed = (reports[stage] for stage in STAGES)
        for report in reports.values():
            if report["train_config"]["seed"] != run["seed"]:
                raise ValueError("report seed disagrees with paired run")
            if (
                report["model_config"] != joint["model_config"]
                or report["dataset"] != joint["dataset"]
            ):
                raise ValueError("paired runs must use identical models and data")
            if (
                report["final_validation"]["next_token_targets"]
                != joint["final_validation"]["next_token_targets"]
            ):
                raise ValueError("paired validation coverage differs")
        if joint["training_tokens"] != clm["training_tokens"]:
            raise ValueError("token-matched control has a different token count")
        if timed["stop_reason"] != "time_budget":
            raise ValueError("time control reached the token cap before matching time")
        if timed["train_config"]["max_training_seconds"] != joint["training_seconds"]:
            raise ValueError("time control did not target the paired joint duration")
        if abs(timed["training_seconds"] / joint["training_seconds"] - 1) > 0.01:
            raise ValueError("actual training-time mismatch exceeds one percent")
        values = {
            stage: reports[stage]["final_validation"]["perplexity"] for stage in STAGES
        }
        rows.append(
            {
                "seed": run["seed"],
                "perplexity": values,
                "probe_perplexity": {
                    stage: reports[stage]["frozen_probe"]["perplexity"]
                    for stage in STAGES
                },
                "training_seconds": {
                    stage: reports[stage]["training_seconds"] for stage in STAGES
                },
                "training_tokens": {
                    stage: reports[stage]["training_tokens"] for stage in STAGES
                },
                "joint_ppl_change_vs_token_clm_percent": (
                    values["joint"] / values["clm"] - 1
                )
                * 100,
                "joint_ppl_change_vs_time_clm_percent": (
                    values["joint"] / values["clm_time"] - 1
                )
                * 100,
                "joint_nll_minus_token_clm": math.log(values["joint"] / values["clm"]),
                "joint_nll_minus_time_clm": math.log(
                    values["joint"] / values["clm_time"]
                ),
            }
        )
    aggregates = {}
    for stage in STAGES:
        values = [row["perplexity"][stage] for row in rows]
        aggregates[stage] = {
            "ppl_mean": statistics.mean(values),
            "ppl_sample_std": statistics.stdev(values),
            "ppl_geometric_mean": math.exp(statistics.mean(map(math.log, values))),
            "probe_ppl_mean": statistics.mean(
                row["probe_perplexity"][stage] for row in rows
            ),
            "training_seconds_mean": statistics.mean(
                row["training_seconds"][stage] for row in rows
            ),
        }
    return {
        "run_id": result["run_id"],
        "paired_results": rows,
        "aggregates": aggregates,
        "validation_targets_per_model": runs[0]["reports"]["joint"]["final_validation"][
            "next_token_targets"
        ],
        "total_remote_gpu_seconds": sum(
            run["remote_gpu_function_seconds"] for run in runs
        )
        + result["diagnostics"]["remote_gpu_function_seconds"],
        "estimated_gpu_charge_usd": sum(run["estimated_gpu_charge_usd"] for run in runs)
        + result["diagnostics"]["estimated_gpu_charge_usd"],
        "diagnostics": result["diagnostics"],
        "note": (
            "Arithmetic PPL means and sample SD describe three seeds, not a "
            "significance test. Negative paired PPL/NLL changes favor joint. "
            "Full prepared validation; no test set or FLOP matching."
        ),
    }


def plot(summary: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 4.6), layout="constrained")
    labels = ["Joint / 10M tokens", "CLM / 10M tokens", "CLM / matched time"]
    for row in summary["paired_results"]:
        axis.plot(
            range(3),
            [row["perplexity"][stage] for stage in STAGES],
            marker="o",
            label=f"seed {row['seed']}",
        )
    axis.set_xticks(range(3), labels)
    axis.set_ylabel("Validation perplexity (lower is better)")
    axis.set_title("Causal ELECTRA: paired seed comparison on WikiText-2")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    summary = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if args.plot is not None:
        plot(summary, args.plot)
    print(json.dumps(summary["aggregates"], indent=2))


if __name__ == "__main__":
    main()
