"""Recompute conditional costs from archived RTX 3090 and RTX 4090 trials."""

import argparse
import hashlib
import json
from pathlib import Path

from deletcra.budget_benchmark import summarize

ROOT = Path(__file__).resolve().parents[2]
STAGES = {"TinyStories": 3_000_000_000, "Base": 16_400_000_000, "IT": 300_000_000}
SETUP_HOURS = 3
CONTAINER_HOURLY = 0.004
VOLUME_HOURLY = 50 * 0.10 / 730


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def scenario(rate: float, gpu_hourly: float, fraction: float) -> dict:
    """Keep token work, fixed overhead and proposed storage charges separate."""
    stages = {name: tokens / rate / 3600 / fraction for name, tokens in STAGES.items()}
    hours = sum(stages.values()) + SETUP_HOURS
    cost = hours * (gpu_hourly + CONTAINER_HOURLY + VOLUME_HOURLY)
    return {
        "sustained_speed_fraction": fraction,
        "stage_training_hours": stages,
        "fixed_overhead_hours": SETUP_HOURS,
        "total_hours": hours,
        "compute_and_proposed_disks_usd": cost,
        "with_10_percent_reserve_usd": cost * 1.1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/runpod-4090-comparison-20261002.json",
    )
    args = parser.parse_args()
    old_path = ROOT / "results/runpod-3090-20261001.json"
    new_path = ROOT / "results/runpod-4090-20261002.json"
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    assert old["status"] == new["status"] == "completed"
    assert old["dataset_sha256"] == new["dataset_sha256"]
    assert new["summary"] == summarize(new["cases"], new["hourly_gpu_price_usd"])
    by_batch = []
    for previous, current in zip(old["summary"], new["summary"], strict=True):
        assert previous["batch_size"] == current["batch_size"]
        by_batch.append(
            {
                "batch_size": current["batch_size"],
                "rtx3090_targets_per_second": previous["prediction_targets_per_second"],
                "rtx4090_targets_per_second": current["prediction_targets_per_second"],
                "throughput_ratio": current["prediction_targets_per_second"]
                / previous["prediction_targets_per_second"],
            }
        )
    best_old = max(old["summary"], key=lambda row: row["prediction_targets_per_second"])
    best_new = max(new["summary"], key=lambda row: row["prediction_targets_per_second"])
    rate = best_new["prediction_targets_per_second"]
    prices = {"measured_secure": 0.74, "unavailable_community_quote": 0.34}
    report = {
        "date": "2026-10-02",
        "type": "archived short throughput measurements and conditional extrapolation",
        "sources": {old_path.name: sha256(old_path), new_path.name: sha256(new_path)},
        "matched_batch_comparison": by_batch,
        "best_observed_batches": {
            "rtx3090": best_old["batch_size"],
            "rtx4090": best_new["batch_size"],
        },
        "best_to_best_throughput_ratio": rate
        / best_old["prediction_targets_per_second"],
        "rtx4090_equal_gpu_cost_hourly_threshold_vs_3090_022": 0.22
        * rate
        / best_old["prediction_targets_per_second"],
        "base_compute_only": {
            name: {
                "gpu_hourly_usd": price,
                "hours": STAGES["Base"] / rate / 3600,
                "gpu_only_usd": STAGES["Base"] / rate / 3600 * price,
                "gpu_and_container_usd": STAGES["Base"]
                / rate
                / 3600
                * (price + CONTAINER_HOURLY),
            }
            for name, price in prices.items()
        },
        "provisional_stage_targets_or_positions": STAGES,
        "proposed_storage": {
            "container_hourly_usd": CONTAINER_HOURLY,
            "persistent_volume_gb": 50,
            "persistent_volume_hourly_usd": VOLUME_HOURLY,
            "allocated_in_measurement": False,
        },
        "whole_lineup_scenarios": {
            name: [scenario(rate, price, fraction) for fraction in (1, 0.8, 0.7)]
            for name, price in prices.items()
        },
        "limitations": [
            "Same pinned corpus, model dimensions, objectives, precision, kernels "
            "and trial schedule; source revisions, GPU, host and driver differ.",
            "Only short cached-data trials; no production-loader, checkpoint "
            "or complete-validation timing.",
            "Best-to-best comparison changes physical batch, so optimizer updates "
            "per token differ; no convergence equivalence is established.",
            "TinyStories 3B and IT 0.3B are provisional planning allowances, "
            "not accepted or sufficient-quality budgets.",
            "IT throughput is unmeasured and reuses joint pretraining speed "
            "only for planning.",
            "Existing TinyStories checkpoint has batch 16 and torch loss; these "
            "optimized batch-128 times are not measurements of that resume recipe.",
            "Three overhead hours and slowdown fractions are assumptions, "
            "not measured whole-run costs.",
            "No full training or release-quality evaluation was performed; "
            "unavailable prices cannot be rented by this report.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "base": report["base_compute_only"],
                "lineup": report["whole_lineup_scenarios"],
                "ratio": report["best_to_best_throughput_ratio"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
