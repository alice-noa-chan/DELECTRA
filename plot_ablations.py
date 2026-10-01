"""Render the observed CLM learning curves and RTD representation controls."""

import argparse
import json
from pathlib import Path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    ablations = {entry["kind"]: entry for entry in result["ablations"]}
    if set(ablations) != {"regularization", "embeddings"}:
        raise ValueError("both completed ablations are required")
    fig, (loss_axis, probe_axis) = plt.subplots(
        1, 2, figsize=(11, 4.6), layout="constrained"
    )
    for setting, report in ablations["regularization"]["reports"].items():
        config = report["train_config"]
        tokens_per_step = config["batch_size"] * (
            report["model_config"]["max_positions"] - 1
        )
        color = "#2d6a9f" if setting == "dropout0" else "#c85a28"
        label = "dropout 0" if setting == "dropout0" else "dropout 0.1"
        losses = report["history"]
        loss_axis.plot(
            [item["step"] * tokens_per_step / 1e6 for item in losses],
            [item["lm_loss"] for item in losses],
            color=color,
            alpha=0.65,
            linestyle="--",
            label=f"{label}: train",
        )
        evaluations = report["validation_history"]
        loss_axis.plot(
            [item["step"] * tokens_per_step / 1e6 for item in evaluations],
            [item["metrics"]["lm_loss"] for item in evaluations],
            color=color,
            marker="o",
            markersize=4,
            label=f"{label}: validation",
        )
    loss_axis.set(
        xlabel="Training input tokens (millions, reused corpus)",
        ylabel="Next-token NLL",
        title="Seed 7: training versus validation",
    )
    loss_axis.grid(alpha=0.2)
    loss_axis.legend(fontsize=8)
    controls = ablations["embeddings"]["reports"]
    values = [
        controls[key]["frozen_probe"]["perplexity"] for key in ("separate", "shared")
    ]
    bars = probe_axis.bar(
        ["Separate embeddings", "Shared embeddings"],
        values,
        color=["#777777", "#2d6a9f"],
    )
    probe_axis.bar_label(bars, fmt="%.0f")
    probe_axis.set(
        ylabel="Frozen-probe perplexity (lower is better)",
        title="RTD only: matched 10M input tokens",
    )
    probe_axis.set_ylim(0, max(values) * 1.15)
    fig.suptitle("Exploratory single-seed controls; validation, not independent test")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
