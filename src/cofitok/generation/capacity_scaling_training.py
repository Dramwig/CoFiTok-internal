from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_STOP_STEP,
)
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_TARGET_STEP,
)
from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


CAPACITY_SCALING_TRAINING_SCHEMA_VERSION = 1
CAPACITY_SCALING_TRAINING_ROLE = (
    "generation_capacity_scaling_50k_partial_training_validation"
)
CAPACITY_SCALING_TRAINING_BOUNDARY = {
    "capacity_scaling_segment_complete": True,
    "configured_100k_training_complete": False,
    "additional_training_allowed": False,
    "full_300k_launch_allowed": False,
    "formal_generation_claim_allowed": False,
    "release_allowed": False,
    "new_source_compatible_decision_required": True,
}


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


def _checkpoint(
    run_dir: Path,
    *,
    step: int,
    expected_revision: str,
    expected_branch: str,
    expected_environment_sha: str,
    expected_dataset_sha: str,
) -> dict[str, Any]:
    path = run_dir / f"checkpoint_step_{step:08d}.pt"
    if not path.is_file():
        raise ValueError(f"capacity scaling checkpoint is missing at step {step}")
    integrity = verify_training_checkpoint(path)
    if (
        int(integrity.get("step", -1)) != step
        or integrity.get("git_revision") != expected_revision
        or integrity.get("git_branch") != expected_branch
        or integrity.get("git_dirty") is not False
        or integrity.get("runtime_environment_sha256") != expected_environment_sha
        or integrity.get("dataset_identity_sha256") != expected_dataset_sha
    ):
        raise ValueError(f"capacity scaling checkpoint identity differs at step {step}")
    integrity_path = checkpoint_integrity_path(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": integrity["checkpoint_sha256"],
        "step": step,
        "integrity_manifest": {
            "path": integrity_path.as_posix(),
            "bytes": integrity_path.stat().st_size,
            "sha256": file_sha256(integrity_path),
        },
    }


def validate_capacity_scaling_partial_training(
    *,
    report_path: str | Path,
    config_path: str | Path,
    expected_revision: str,
    expected_branch: str,
    expected_parameter_count: int,
    expected_micro_batch_size: int,
    expected_gradient_accumulation_steps: int,
    expected_dataset: str = "imagenet_256",
    source_step: int = CAPACITY_PROBE_STOP_STEP,
    stop_step: int = CAPACITY_SCALING_TARGET_STEP,
    configured_steps: int = CAPACITY_PROBE_CONFIGURED_STEPS,
) -> dict[str, Any]:
    effective_batch = (
        expected_micro_batch_size * expected_gradient_accumulation_steps
    )
    if (
        source_step != CAPACITY_PROBE_STOP_STEP
        or stop_step != CAPACITY_SCALING_TARGET_STEP
        or configured_steps != CAPACITY_PROBE_CONFIGURED_STEPS
        or effective_batch != CAPACITY_PROBE_EFFECTIVE_BATCH
        or expected_parameter_count < 1
        or len(expected_revision) != 40
        or not expected_branch
    ):
        raise ValueError("capacity scaling training expectation differs")
    report_file = Path(report_path).resolve()
    config_file = Path(config_path).resolve()
    run_dir = report_file.parent
    report = _read(report_file)
    if (
        report.get("training_complete") is not False
        or int(report.get("completed_steps", -1)) != stop_step
        or int(report.get("target_steps", -1)) != configured_steps
        or report.get("stop_requested") is not False
        or report.get("stop_signal") is not None
    ):
        raise ValueError("capacity scaling training report is not the exact clean stop")

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
        raise ValueError("capacity scaling resolved training config differs")
    if (
        config.runtime.steps != configured_steps
        or config.data.dataset != expected_dataset
        or config.runtime.protected_checkpoint_steps != [source_step]
    ):
        raise ValueError("capacity scaling configured horizon or protection differs")
    git = report.get("git")
    if git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "dirty": False,
    }:
        raise ValueError("capacity scaling training Git identity differs")
    if report.get("training_authorization") is not None:
        raise ValueError("capacity scaling segment must not consume a formal gate")
    if (
        int(report.get("parameter_count", -1)) != expected_parameter_count
        or int(report.get("trainable_parameter_count", -1))
        != expected_parameter_count
        or Path(str(report.get("config_path", ""))).resolve() != config_file
        or Path(str(report.get("output_dir", ""))).resolve() != run_dir
    ):
        raise ValueError("capacity scaling training identity differs")
    resume_value = report.get("resume")
    if not isinstance(resume_value, str) or not resume_value:
        raise ValueError("capacity scaling exact-resume evidence differs")
    resume_path = Path(resume_value).resolve()
    resume_match = re.fullmatch(r"checkpoint_step_(\d{8})\.pt", resume_path.name)
    resume_step = int(resume_match.group(1)) if resume_match is not None else -1
    reconciliation = report.get("metrics_resume_reconciliation")
    if (
        resume_path.parent != run_dir
        or not source_step <= resume_step < stop_step
        or report.get("resume_revision_transition") is not None
        or not isinstance(reconciliation, Mapping)
        or int(reconciliation.get("schema_version", -1)) != 1
        or reconciliation.get("status") not in {"unchanged", "reconciled"}
        or int(reconciliation.get("resume_step", -1)) != resume_step
        or Path(str(reconciliation.get("metrics", ""))).resolve()
        != run_dir / "train_metrics.jsonl"
        or int(reconciliation.get("retained_rows", -1)) < 1
        or int(reconciliation.get("orphaned_rows", -1)) < 0
    ):
        raise ValueError("capacity scaling exact-resume evidence differs")
    if reconciliation.get("status") == "unchanged" and (
        reconciliation.get("orphan_archive") is not None
        or reconciliation.get("orphan_sha256") is not None
    ):
        raise ValueError("capacity scaling unchanged resume evidence differs")
    if reconciliation.get("status") == "reconciled":
        archive = Path(str(reconciliation.get("orphan_archive", ""))).resolve()
        digest = reconciliation.get("orphan_sha256")
        reconciliation_report = Path(str(reconciliation.get("report", ""))).resolve()
        expected_reconciliation = {
            key: value for key, value in reconciliation.items() if key != "report"
        }
        if (
            not archive.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
            or file_sha256(archive) != digest
            or not reconciliation_report.is_file()
            or _read(reconciliation_report) != expected_reconciliation
        ):
            raise ValueError("capacity scaling reconciled resume evidence differs")

    environment = report.get("runtime_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("capacity scaling runtime environment is missing")
    environment_sha = runtime_environment_sha256(dict(environment))
    if report.get("runtime_environment_sha256") != environment_sha:
        raise ValueError("capacity scaling runtime environment identity differs")
    provenance = report.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("capacity scaling dataset provenance is missing")
    dataset_sha = validate_dataset_provenance(
        dict(provenance),
        expected_dataset=expected_dataset,
    )["identity_sha256"]

    final = report.get("final_metrics")
    if not isinstance(final, Mapping):
        raise ValueError("capacity scaling final metrics are missing")
    if (
        int(final.get("step", -1)) != stop_step
        or int(final.get("samples_seen", -1)) != stop_step * effective_batch
    ):
        raise ValueError("capacity scaling step or sample accounting differs")
    for field in (
        "total_loss",
        "epsilon_loss",
        "grad_norm",
        "learning_rate",
        "elapsed_seconds",
        "cumulative_elapsed_seconds",
    ):
        _finite_positive(final.get(field), label=f"final_metrics.{field}")
    segment_elapsed = _finite_positive(
        report.get("segment_elapsed_seconds"),
        label="segment_elapsed_seconds",
    )
    elapsed = _finite_positive(report.get("elapsed_seconds"), label="elapsed_seconds")
    if elapsed <= segment_elapsed:
        raise ValueError("capacity scaling cumulative elapsed accounting differs")
    if int(report.get("peak_vram_bytes", 0)) < 1:
        raise ValueError("capacity scaling peak VRAM accounting is missing")

    manifest = _read(run_dir / "run_manifest.json")
    for key, value in manifest.items():
        if report.get(key) != value:
            raise ValueError(f"capacity scaling run manifest differs: {key}")
    metrics_path = run_dir / "train_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(
                    f"capacity scaling metric row {line_number} is not an object"
                )
            rows.append(row)
    steps = [int(row.get("step", -1)) for row in rows]
    if (
        not steps
        or steps[0] != 1
        or steps[-1] != stop_step
        or any(left >= right for left, right in zip(steps, steps[1:]))
        or any(step > stop_step for step in steps)
        or rows[-1] != final
    ):
        raise ValueError("capacity scaling metric history is not canonical")
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    expected_events = stop_step // config.runtime.evaluation_interval
    if (
        len(validation_rows) != expected_events
        or [int(row.get("validation_event_index", -1)) for row in validation_rows]
        != list(range(expected_events))
    ):
        raise ValueError("capacity scaling scheduled validation events differ")

    source_checkpoint = _checkpoint(
        run_dir,
        step=source_step,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_environment_sha=environment_sha,
        expected_dataset_sha=dataset_sha,
    )
    checkpoint = _checkpoint(
        run_dir,
        step=stop_step,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_environment_sha=environment_sha,
        expected_dataset_sha=dataset_sha,
    )
    latest = _read(run_dir / "latest.json")
    if (
        report.get("latest_checkpoint") != latest
        or latest.get("checkpoint") != Path(checkpoint["path"]).name
        or int(latest.get("step", -1)) != stop_step
        or latest.get("checkpoint_sha256") != checkpoint["sha256"]
        or latest.get("integrity_manifest")
        != Path(checkpoint["integrity_manifest"]["path"]).name
    ):
        raise ValueError("capacity scaling latest checkpoint binding differs")
    later = sorted(
        path.name
        for path in run_dir.glob("checkpoint_step_*.pt")
        if int(path.stem.rsplit("_", 1)[-1]) > stop_step
    )
    if later:
        raise ValueError("capacity scaling run contains checkpoints after the stop step")
    return {
        "schema_version": CAPACITY_SCALING_TRAINING_SCHEMA_VERSION,
        "status": "pass",
        "role": CAPACITY_SCALING_TRAINING_ROLE,
        "run_dir": run_dir.as_posix(),
        "training_report": report_file.as_posix(),
        "config": config_file.as_posix(),
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        },
        "configured_steps": configured_steps,
        "source_step": source_step,
        "completed_steps": stop_step,
        "training_complete": False,
        "exact_resume": True,
        "initial_source_step": source_step,
        "last_resume_step": resume_step,
        "recovery_resume_used": resume_step > source_step,
        "micro_batch_size": expected_micro_batch_size,
        "gradient_accumulation_steps": expected_gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "images_seen": stop_step * effective_batch,
        "parameter_count": expected_parameter_count,
        "dataset_identity_sha256": dataset_sha,
        "runtime_environment_sha256": environment_sha,
        "metric_row_count": len(rows),
        "validation_event_count": len(validation_rows),
        "source_checkpoint": source_checkpoint,
        "checkpoint": checkpoint,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_TRAINING_BOUNDARY
        ),
    }
