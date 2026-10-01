"""Freeze the existing TinyStories pilot without publishing or overwriting it."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "runs/modal-checkpoints/modal-l40s-tinystories15m-20261001-profile"


def sha256(path: Path) -> str:
    """Hash incrementally so model files do not need a second in-memory copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_pilot(destination: Path) -> dict:
    """Copy only explicit models, tokenizer, tracked code, licenses, and evidence.

    Copy to a new staging folder first, compare every source/copy hash, and rename
    only after the manifest is complete. Existing archives are never overwritten.
    A failed copy leaves staging files for inspection; originals remain in place.
    """
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    files = {}
    for source_dir, target_dir in (
        (PILOT / "joint/joint/model", "model"),
        (PILOT / "joint/joint/generator", "generator"),
        (PILOT / "clm/clm/model", "controls/clm"),
        (ROOT / "data/tinystories-reference-256-100k/tokenizer", "tokenizer"),
    ):
        for path in source_dir.iterdir():
            if path.is_file():
                files[f"{target_dir}/{path.name}"] = path
    for mode in ("clm", "joint"):
        files[f"reports/{mode}.json"] = PILOT / mode / mode / "report.json"
    files["reports/target-result.json"] = PILOT / "target-result.json"
    files["validation/validation.pt"] = (
        ROOT / "data/tinystories-reference-256-100k/validation.pt"
    )
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "LICENSES", "results"], cwd=ROOT, text=True
    ).splitlines()
    tracked += ["pyproject.toml", "README.md", "LICENSE", "NOTICE"]
    for relative in tracked:
        files[f"source/{relative}"] = ROOT / relative
    # Validate all inputs before creating any destination files.
    for source in files.values():
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"archive input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}-staging-{uuid4().hex}")
    staging.mkdir()
    entries = {}
    for relative, source in sorted(files.items()):
        copied = staging / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copy2(source, copied)
        if sha256(copied) != before or sha256(source) != before:
            raise RuntimeError(f"archive copy changed; inspect {staging}: {relative}")
        entries[relative] = {"sha256": before, "bytes": copied.stat().st_size}
    manifest = {
        "status": "archived_partial_training_pilot",
        "published": False,
        "training_complete": False,
        "purpose": "Preserve existing evidence; not a release candidate.",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "original_checkpoints": PILOT.relative_to(ROOT).as_posix(),
        "modal_volume": "deletcra-experiments",
        "modal_run": "modal-l40s-tinystories15m-20261001-profile",
        "main_objective": "joint",
        "control_objective": "clm",
        "steps_per_condition": 300,
        "input_tokens_per_condition": 1224000,
        "prepared_train_tokens": 9999825,
        "files": entries,
        "note": (
            "Main and generator are evaluation weights, not optimizer/RNG resume "
            "states. Full training data remains in the Modal cache. This archive "
            "includes matching validation tokens but no raw training corpus. "
            "Use the bundled deletcra loader; these are not AutoModel weights."
        ),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(destination)
    return {
        "archive": destination.relative_to(ROOT).as_posix()
        if destination.is_relative_to(ROOT)
        else str(destination),
        "manifest_sha256": sha256(destination / "manifest.json"),
        "copied_files": len(entries),
        "copied_bytes": sum(item["bytes"] for item in entries.values()),
        **{key: value for key, value in manifest.items() if key != "files"},
        "files": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite: {args.report}")
    report = archive_pilot(args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Archived {report['copied_files']} files at {report['archive']}")


if __name__ == "__main__":
    main()
