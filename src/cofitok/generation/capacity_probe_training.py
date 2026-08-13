from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import runtime_environment_sha256
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


PARTIAL_TRAINING_SCHEMA_VERSION = 1
PARTIAL_TRAINING_ROLE = "generation_capacity_probe_partial_training_validation"


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def validate_capacity_probe_partial_training(
    *,
    report_path: str | Path,
    config_path: str | Path,
    expected_stop_step: int,
    expected_configured_steps: int,
    expected_revision: str,
    expected_branch: str,
    expected_parameter_count: int,
    expected_micro_batch_size: int,
    expected_gradient_accumulation_steps: int,
    expected_dataset: str = "imagenet_256",
) -> dict[str, Any]:
    if (
        expected_stop_step < 1
        or expected_configured_steps <= expected_stop_step
        or len(expected_revision) != 40
        or not expected_branch
        or expected_parameter_count < 1
        or expected_micro_batch_size < 1
        or expected_gradient_accumulation_steps < 1
    ):
        raise ValueError("capacity probe partial-training expectation is invalid")
    effective_batch = (
        expected_micro_batch_size * expected_gradient_accumulation_steps
    )
    if effective_batch != 64:
        raise ValueError("capacity probe partial training must use effective batch 64")

    report_file = Path(report_path).resolve()
    config_file = Path(config_path).resolve()
    run_dir = report_file.parent
    report = _read(report_file)
    if (
        report.get("training_complete") is not False
        or int(report.get("completed_steps", -1)) != expected_stop_step
        or int(report.get("target_steps", -1)) != expected_configured_steps
        or report.get("stop_requested") is not False
        or report.get("stop_signal") is not None
    ):
        raise ValueError("capacity probe training report is not the exact clean stop")

    config = load_config(config_file)
    config = replace(
        config,
        data=replace(config.data, batch_size=expected_micro_batch_size),
        optimization=replace(
            config.optimization,
            gradient_accumulation_steps=expected_gradient_accumulation_steps,
        ),
    )
    expected_config = json.loads(json.dumps(config_to_dict(config)))
    if report.get("config") != expected_config:
        raise ValueError("capacity probe resolved training config differs")
    if (
        config.runtime.steps != expected_configured_steps
        or config.data.dataset != expected_dataset
        or config.runtime.protected_checkpoint_steps != [expected_stop_step]
    ):
        raise ValueError("capacity probe configured horizon/dataset/protection differs")

    git = report.get("git")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "dirty": False,
    }:
        raise ValueError("capacity probe training Git identity differs")
    if report.get("training_authorization") is not None:
        raise ValueError("bounded capacity probe must not consume a full-training gate")
    if (
        int(report.get("parameter_count", -1)) != expected_parameter_count
        or int(report.get("trainable_parameter_count", -1))
        != expected_parameter_count
    ):
        raise ValueError("capacity probe training parameter count differs")
    if Path(str(report.get("config_path", ""))).resolve() != config_file:
        raise ValueError("capacity probe report config path differs")
    if Path(str(report.get("output_dir", ""))).resolve() != run_dir:
        raise ValueError("capacity probe report output directory differs")

    environment = report.get("runtime_environment")
    if not isinstance(environment, dict):
        raise ValueError("capacity probe runtime environment is missing")
    environment_sha = runtime_environment_sha256(environment)
    if report.get("runtime_environment_sha256") != environment_sha:
        raise ValueError("capacity probe runtime environment SHA256 differs")
    provenance = report.get("dataset_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("capacity probe dataset provenance is missing")
    validated_dataset = validate_dataset_provenance(
        provenance,
        expected_dataset=expected_dataset,
    )
    dataset_sha = validated_dataset["identity_sha256"]

    final = report.get("final_metrics")
    if not isinstance(final, dict):
        raise ValueError("capacity probe final metrics are missing")
    if (
        int(final.get("step", -1)) != expected_stop_step
        or int(final.get("samples_seen", -1))
        != expected_stop_step * effective_batch
    ):
        raise ValueError("capacity probe sample/step accounting differs")
    for field in (
        "total_loss",
        "epsilon_loss",
        "grad_norm",
        "learning_rate",
        "elapsed_seconds",
        "cumulative_elapsed_seconds",
    ):
        _finite_positive(final.get(field), label=f"final_metrics.{field}")
    _finite_positive(report.get("elapsed_seconds"), label="elapsed_seconds")
    if int(report.get("peak_vram_bytes", 0)) < 1:
        raise ValueError("capacity probe peak VRAM accounting is missing")

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read(manifest_path)
    for key, value in manifest.items():
        if report.get(key) != value:
            raise ValueError(f"capacity probe run manifest differs: {key}")

    metrics_path = run_dir / "train_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(
                    f"capacity probe metric row {line_number} is not an object"
                )
            rows.append(row)
    steps = [int(row.get("step", -1)) for row in rows]
    if (
        not steps
        or steps[0] != 1
        or steps[-1] != expected_stop_step
        or any(left >= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError("capacity probe metric steps are not canonical and increasing")
    if rows[-1] != final:
        raise ValueError("capacity probe final metric row differs from training report")
    validation_rows = [
        row for row in rows if "validation_epsilon_mse" in row
    ]
    expected_validation_events = (
        expected_stop_step // config.runtime.evaluation_interval
    )
    if (
        len(validation_rows) != expected_validation_events
        or [int(row.get("validation_event_index", -1)) for row in validation_rows]
        != list(range(expected_validation_events))
    ):
        raise ValueError("capacity probe scheduled validation events differ")

    latest_path = run_dir / "latest.json"
    latest = _read(latest_path)
    if report.get("latest_checkpoint") != latest:
        raise ValueError("capacity probe report and latest.json differ")
    checkpoint_name = f"checkpoint_step_{expected_stop_step:08d}.pt"
    checkpoint = run_dir / checkpoint_name
    if (
        latest.get("checkpoint") != checkpoint_name
        or int(latest.get("step", -1)) != expected_stop_step
        or not checkpoint.is_file()
    ):
        raise ValueError("capacity probe latest checkpoint differs")
    integrity = verify_training_checkpoint(checkpoint)
    if (
        int(integrity.get("step", -1)) != expected_stop_step
        or integrity.get("git_revision") != expected_revision
        or integrity.get("git_branch") != expected_branch
        or integrity.get("git_dirty") is not False
        or integrity.get("runtime_environment_sha256") != environment_sha
        or integrity.get("dataset_identity_sha256") != dataset_sha
        or latest.get("checkpoint_sha256") != integrity.get("checkpoint_sha256")
        or latest.get("integrity_manifest")
        != checkpoint_integrity_path(checkpoint).name
    ):
        raise ValueError("capacity probe checkpoint integrity identity differs")
    later = sorted(
        path.name
        for path in run_dir.glob("checkpoint_step_*.pt")
        if int(path.stem.rsplit("_", 1)[-1]) > expected_stop_step
    )
    if later:
        raise ValueError("capacity probe run contains checkpoints after the stop step")

    return {
        "schema_version": PARTIAL_TRAINING_SCHEMA_VERSION,
        "status": "pass",
        "role": PARTIAL_TRAINING_ROLE,
        "run_dir": run_dir.as_posix(),
        "training_report": report_file.as_posix(),
        "config": config_file.as_posix(),
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        },
        "configured_steps": expected_configured_steps,
        "completed_steps": expected_stop_step,
        "training_complete": False,
        "intentional_partial_stop": True,
        "micro_batch_size": expected_micro_batch_size,
        "gradient_accumulation_steps": expected_gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "images_seen": expected_stop_step * effective_batch,
        "parameter_count": expected_parameter_count,
        "dataset_identity_sha256": dataset_sha,
        "runtime_environment_sha256": environment_sha,
        "metric_row_count": len(rows),
        "validation_event_count": len(validation_rows),
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": integrity["checkpoint_sha256"],
            "integrity_manifest": checkpoint_integrity_path(checkpoint).as_posix(),
        },
        "authorization_boundary": {
            "capacity_probe_training_complete": True,
            "full_training_complete": False,
            "full_300k_launch_allowed": False,
            "formal_generation_claim_allowed": False,
            "release_allowed": False,
        },
    }
