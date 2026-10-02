"""Strict research identity and explicitly recorded execution migrations."""

from copy import deepcopy


def compatible_migration(source, target):
    """Allow execution changes only; batch size, objectives and budgets stay fixed."""
    old, new = deepcopy(source), deepcopy(target)
    for spec in (old, new):
        spec["model"].pop("attention_backend", None)
        spec["objective"].pop("lm_loss_backend", None)
        for field in (
            "device",
            "precision",
            "fused_optimizer",
            "cpu_threads",
            "max_wall_seconds",
        ):
            spec["plan"].pop(field, None)
    return old == new


def migration_record(source, target, step, positions):
    return {
        "at_step": step,
        "input_positions": positions,
        "source_spec": source,
        "target_spec": target,
        "rng_policy": (
            "Preserve host and sampler state; reseed device and proposal RNG "
            "at this documented boundary."
        ),
        "bitwise_continuation": False,
    }
