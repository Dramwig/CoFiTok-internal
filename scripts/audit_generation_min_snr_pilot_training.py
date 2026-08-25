from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from cofitok.configs import load_config
from cofitok.generation.min_snr_pilot import (
    DATASET_IDENTITY_SHA256,
    EFFECTIVE_BATCH_SIZE,
    GAMMA,
    IMAGES_PER_METHOD,
    LEGACY_RUNTIME_ENVIRONMENT_SHA256,
    METHODS,
    PILOT_STEP,
    SCHEDULER_HORIZON,
)
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically audit a completed 50K matched Min-SNR pilot arm."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"metrics line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("pilot metrics are empty")
    return rows


def main() -> None:
    args = parse_args()
    revision = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    branch = _git("branch", "--show-current")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if (
        revision != args.expected_revision
        or tree != args.expected_tree
        or branch != args.expected_branch
        or dirty
    ):
        raise ValueError("pilot training checkout identity differs")
    config_path = Path(args.config).resolve()
    configured = load_config(config_path)
    resolved = replace(
        configured,
        data=replace(configured.data, batch_size=EFFECTIVE_BATCH_SIZE),
        optimization=replace(
            configured.optimization, gradient_accumulation_steps=1
        ),
    )
    if (
        resolved.loss.min_snr_gamma != GAMMA
        or resolved.runtime.steps != SCHEDULER_HORIZON
        or resolved.runtime.seed != 2027
        or resolved.diffusion.prediction_target != "epsilon"
        or resolved.data.dataset != "imagenet_256"
    ):
        raise ValueError("pilot resolved config differs")
    run_dir = Path(args.run_dir).resolve()
    report_path = run_dir / "training_report.json"
    latest_path = run_dir / "latest.json"
    metrics_path = run_dir / "train_metrics.jsonl"
    report = _read(report_path)
    latest = _read(latest_path)
    git = report.get("git", {})
    provenance = report.get("dataset_provenance", {})
    checkpoint_path = run_dir / f"checkpoint_step_{PILOT_STEP:08d}.pt"
    integrity_path = checkpoint_integrity_path(checkpoint_path)
    integrity = verify_training_checkpoint(checkpoint_path)
    if (
        report.get("config") != asdict(resolved)
        or int(report.get("completed_steps", -1)) != PILOT_STEP
        or int(report.get("target_steps", -1)) != SCHEDULER_HORIZON
        or report.get("training_complete") is not False
        or report.get("stop_requested") is not False
        or report.get("stop_signal") is not None
        or git.get("revision") != args.expected_revision
        or git.get("branch") != args.expected_branch
        or git.get("dirty") is not False
        or report.get("runtime_environment_sha256")
        != LEGACY_RUNTIME_ENVIRONMENT_SHA256
        or provenance.get("identity_sha256") != DATASET_IDENTITY_SHA256
        or int(latest.get("step", -1)) != PILOT_STEP
        or latest.get("checkpoint") != checkpoint_path.name
        or latest.get("checkpoint_sha256") != integrity.get("checkpoint_sha256")
        or latest.get("integrity_manifest") != integrity_path.name
        or latest.get("git_revision") != args.expected_revision
        or latest.get("git_branch") != args.expected_branch
        or latest.get("git_dirty") is not False
        or int(integrity.get("step", -1)) != PILOT_STEP
        or integrity.get("git_revision") != args.expected_revision
        or integrity.get("git_branch") != args.expected_branch
        or integrity.get("git_dirty") is not False
        or integrity.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256
        or integrity.get("runtime_environment_sha256")
        != LEGACY_RUNTIME_ENVIRONMENT_SHA256
    ):
        raise ValueError("pilot training report, latest, or checkpoint differs")
    rows = _read_metrics(metrics_path)
    steps = [int(row.get("step", -1)) for row in rows]
    if (
        steps[0] != 1
        or steps[-1] != PILOT_STEP
        or any(current <= previous for previous, current in zip(steps, steps[1:]))
    ):
        raise ValueError("pilot metrics steps are not strictly bound")
    min_snr_weight_below_one = False
    for index, row in enumerate(rows):
        step = int(row.get("step", -1))
        if int(row.get("samples_seen", -1)) != step * EFFECTIVE_BATCH_SIZE:
            raise ValueError(f"pilot metrics row {index} exposure differs")
        for field in (
            "total",
            "epsilon",
            "epsilon_unweighted",
            "min_snr_weight_mean",
            "learning_rate",
            "grad_norm",
        ):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"pilot metrics row {index} lacks finite {field}")
        weight = float(row["min_snr_weight_mean"])
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"pilot metrics row {index} has invalid Min-SNR weight")
        min_snr_weight_below_one = min_snr_weight_below_one or weight < 0.999999
    if not min_snr_weight_below_one:
        raise ValueError("pilot metrics never applied Min-SNR downweighting")
    output = {
        "schema_version": 1,
        "role": "generation_matched_min_snr_pilot_training_physical_audit",
        "status": "pass",
        "method": args.method,
        "git": {
            "revision": revision,
            "tree": tree,
            "branch": branch,
            "tracked_dirty": False,
        },
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": file_sha256(config_path),
            "min_snr_gamma": GAMMA,
            "scheduler_horizon_steps": SCHEDULER_HORIZON,
        },
        "training": {
            "completed_steps": PILOT_STEP,
            "target_steps": SCHEDULER_HORIZON,
            "training_complete": False,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "samples_seen": IMAGES_PER_METHOD,
            "metric_row_count": len(rows),
            "strictly_increasing_metrics": True,
            "samples_seen_binding_verified": True,
            "min_snr_metrics_verified": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": integrity["checkpoint_sha256"],
            "integrity_manifest": {
                "path": str(integrity_path),
                "bytes": integrity_path.stat().st_size,
                "sha256": file_sha256(integrity_path),
            },
            "physical_sha256_verified": True,
            "latest_exact_binding": True,
        },
        "sources": {
            "training_report": {
                "path": str(report_path),
                "bytes": report_path.stat().st_size,
                "sha256": file_sha256(report_path),
            },
            "latest": {
                "path": str(latest_path),
                "bytes": latest_path.stat().st_size,
                "sha256": file_sha256(latest_path),
            },
            "metrics": {
                "path": str(metrics_path),
                "bytes": metrics_path.stat().st_size,
                "sha256": file_sha256(metrics_path),
            },
        },
        "authorization_boundary": {
            "continuation_beyond_50000_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
            "process_signals_allowed": False,
        },
    }
    write_json_report(args.output, output)
    print(file_sha256(args.output))


if __name__ == "__main__":
    main()
