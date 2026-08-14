from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from cofitok.configs import load_config
from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_completion_decision import (
    CAPACITY_COMPLETION_TARGET_STEP,
)
from cofitok.generation.capacity_probe import CAPACITY_PROBE_EFFECTIVE_BATCH
from cofitok.generation.capacity_scaling_decision import CAPACITY_SCALING_TARGET_STEP
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)
from cofitok.training.completion import validate_completed_generation_training


CAPACITY_COMPLETION_TRAINING_SCHEMA_VERSION = 1
CAPACITY_COMPLETION_TRAINING_ROLE = (
    "generation_capacity_completion_100k_training_validation"
)
CAPACITY_COMPLETION_TRAINING_BOUNDARY = {
    "capacity_completion_training_complete": True,
    "configured_100k_training_complete": True,
    "additional_training_allowed": False,
    "terminal_evaluation_complete": False,
    "full_300k_launch_allowed": False,
    "formal_generation_claim_allowed": False,
    "release_allowed": False,
    "new_source_compatible_result_required": True,
}


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _checkpoint(
    path: Path,
    *,
    step: int,
    expected_revision: str,
    expected_branch: str,
    expected_environment_sha: str,
    expected_dataset_sha: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"capacity completion checkpoint is missing at step {step}")
    integrity = verify_training_checkpoint(path)
    if (
        int(integrity.get("step", -1)) != step
        or integrity.get("git_revision") != expected_revision
        or integrity.get("git_branch") != expected_branch
        or integrity.get("git_dirty") is not False
        or integrity.get("runtime_environment_sha256")
        != expected_environment_sha
        or integrity.get("dataset_identity_sha256") != expected_dataset_sha
    ):
        raise ValueError(f"capacity completion checkpoint step differs at {step}")
    sidecar = checkpoint_integrity_path(path)
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": integrity["checkpoint_sha256"],
        "step": step,
        "integrity_manifest": {
            "path": sidecar.resolve().as_posix(),
            "bytes": sidecar.stat().st_size,
            "sha256": file_sha256(sidecar),
        },
    }


def validate_capacity_completion_training(
    *,
    report_path: str | Path,
    config_path: str | Path,
    source_archive_path: str | Path,
    expected_source_archive_sha256: str,
    method: str,
    expected_revision: str,
    expected_branch: str,
    expected_parameter_count: int,
    expected_micro_batch_size: int,
    expected_gradient_accumulation_steps: int,
) -> dict[str, Any]:
    if method not in {"cofitok", "dense_identity"}:
        raise ValueError("capacity completion method is unsupported")
    if (
        expected_micro_batch_size * expected_gradient_accumulation_steps
        != CAPACITY_PROBE_EFFECTIVE_BATCH
    ):
        raise ValueError("capacity completion effective batch differs")
    report_file = Path(report_path).resolve()
    config_file = Path(config_path).resolve()
    run_dir = report_file.parent
    report = validate_completed_generation_training(
        report_path=report_file,
        config_path=config_file,
        expected_steps=CAPACITY_COMPLETION_TARGET_STEP,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_micro_batch_size=expected_micro_batch_size,
        expected_gradient_accumulation_steps=(
            expected_gradient_accumulation_steps
        ),
    )
    config = load_config(config_file)
    if (
        config.runtime.steps != CAPACITY_COMPLETION_TARGET_STEP
        or config.data.dataset != "imagenet_256"
        or config.runtime.protected_checkpoint_steps
        != [10_000]
        or int(report.get("parameter_count", -1)) != expected_parameter_count
        or int(report.get("trainable_parameter_count", -1))
        != expected_parameter_count
        or report.get("training_authorization") is not None
        or report.get("stop_requested") is not False
        or report.get("stop_signal") is not None
        or Path(str(report.get("config_path", ""))).resolve() != config_file
        or Path(str(report.get("output_dir", ""))).resolve() != run_dir
    ):
        raise ValueError("capacity completion training identity differs")
    resume_value = report.get("resume")
    if not isinstance(resume_value, str) or not resume_value:
        raise ValueError("capacity completion resume evidence is missing")
    resume = Path(resume_value).resolve()
    match = re.fullmatch(r"checkpoint_step_(\d{8})\.pt", resume.name)
    resume_step = int(match.group(1)) if match is not None else -1
    reconciliation = report.get("metrics_resume_reconciliation")
    if (
        resume.parent != run_dir
        or not CAPACITY_SCALING_TARGET_STEP <= resume_step < CAPACITY_COMPLETION_TARGET_STEP
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
        raise ValueError("capacity completion exact-resume evidence differs")
    if reconciliation.get("status") == "unchanged" and (
        reconciliation.get("orphan_archive") is not None
        or reconciliation.get("orphan_sha256") is not None
    ):
        raise ValueError("capacity completion unchanged resume evidence differs")
    if reconciliation.get("status") == "reconciled":
        orphan = Path(str(reconciliation.get("orphan_archive", ""))).resolve()
        orphan_sha = reconciliation.get("orphan_sha256")
        reconciliation_report = Path(
            str(reconciliation.get("report", ""))
        ).resolve()
        expected_reconciliation = {
            key: value for key, value in reconciliation.items() if key != "report"
        }
        if (
            not orphan.is_file()
            or not isinstance(orphan_sha, str)
            or len(orphan_sha) != 64
            or file_sha256(orphan) != orphan_sha
            or not reconciliation_report.is_file()
            or _read(reconciliation_report) != expected_reconciliation
        ):
            raise ValueError("capacity completion reconciled resume evidence differs")
    environment = report.get("runtime_environment")
    provenance = report.get("dataset_provenance")
    if not isinstance(environment, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("capacity completion runtime or dataset evidence is missing")
    environment_sha = runtime_environment_sha256(dict(environment))
    dataset_sha = validate_dataset_provenance(
        dict(provenance),
        expected_dataset="imagenet_256",
    )["identity_sha256"]
    if (
        report.get("runtime_environment_sha256") != environment_sha
        or report.get("dataset_provenance", {}).get("identity_sha256")
        != dataset_sha
    ):
        raise ValueError("capacity completion runtime or dataset identity differs")
    final = report.get("final_metrics")
    if (
        not isinstance(final, Mapping)
        or int(final.get("step", -1)) != CAPACITY_COMPLETION_TARGET_STEP
        or int(final.get("samples_seen", -1))
        != CAPACITY_COMPLETION_TARGET_STEP * CAPACITY_PROBE_EFFECTIVE_BATCH
    ):
        raise ValueError("capacity completion final metric evidence differs")
    for field in (
        "total_loss",
        "epsilon_loss",
        "grad_norm",
        "learning_rate",
        "elapsed_seconds",
        "cumulative_elapsed_seconds",
    ):
        _positive(final.get(field), label=f"final_metrics.{field}")
    segment_elapsed = _positive(
        report.get("segment_elapsed_seconds"),
        label="segment_elapsed_seconds",
    )
    elapsed = _positive(report.get("elapsed_seconds"), label="report elapsed")
    if elapsed <= segment_elapsed:
        raise ValueError("capacity completion cumulative elapsed accounting differs")
    _positive(report.get("peak_vram_bytes"), label="peak VRAM")
    manifest = _read(run_dir / "run_manifest.json")
    for key, value in manifest.items():
        if report.get(key) != value:
            raise ValueError(f"capacity completion run manifest differs: {key}")
    metrics_path = run_dir / "train_metrics.jsonl"
    rows = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(
                    f"capacity completion metric row {line_number} is not an object"
                )
            rows.append(row)
    steps = [int(row.get("step", -1)) for row in rows]
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    expected_events = (
        CAPACITY_COMPLETION_TARGET_STEP // config.runtime.evaluation_interval
    )
    if (
        not steps
        or steps[0] != 1
        or steps[-1] != CAPACITY_COMPLETION_TARGET_STEP
        or any(left >= right for left, right in zip(steps, steps[1:]))
        or rows[-1] != final
        or len(validation_rows) != expected_events
        or [int(row.get("validation_event_index", -1)) for row in validation_rows]
        != list(range(expected_events))
    ):
        raise ValueError("capacity completion metric history differs")
    archive_path = Path(source_archive_path).resolve()
    if file_sha256(archive_path) != expected_source_archive_sha256:
        raise ValueError("capacity completion source archive SHA256 differs")
    archive_report = _read(archive_path)
    archive_row = archive_report.get("methods", {}).get(method)
    if (
        archive_report.get("status") != "pass"
        or archive_report.get("role")
        != "capacity_completion_50k_source_checkpoint_archive"
        or not isinstance(archive_row, Mapping)
    ):
        raise ValueError("capacity completion source archive contract differs")
    source_checkpoint = _checkpoint(
        Path(str(archive_row["archive_checkpoint"]["path"])).resolve(),
        step=CAPACITY_SCALING_TARGET_STEP,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_environment_sha=environment_sha,
        expected_dataset_sha=dataset_sha,
    )
    if (
        source_checkpoint["bytes"] != archive_row["source_checkpoint"]["bytes"]
        or source_checkpoint["sha256"]
        != archive_row["source_checkpoint"]["sha256"]
        or source_checkpoint["integrity_manifest"]["bytes"]
        != archive_row["source_integrity_manifest"]["bytes"]
        or source_checkpoint["integrity_manifest"]["sha256"]
        != archive_row["source_integrity_manifest"]["sha256"]
    ):
        raise ValueError("capacity completion archived source identity differs")
    checkpoint = _checkpoint(
        run_dir / "checkpoint_step_00100000.pt",
        step=CAPACITY_COMPLETION_TARGET_STEP,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_environment_sha=environment_sha,
        expected_dataset_sha=dataset_sha,
    )
    latest = _read(run_dir / "latest.json")
    if (
        report.get("latest_checkpoint") != latest
        or latest.get("checkpoint") != Path(checkpoint["path"]).name
        or int(latest.get("step", -1)) != CAPACITY_COMPLETION_TARGET_STEP
        or latest.get("checkpoint_sha256") != checkpoint["sha256"]
        or latest.get("integrity_manifest")
        != Path(checkpoint["integrity_manifest"]["path"]).name
    ):
        raise ValueError("capacity completion latest checkpoint binding differs")
    later = [
        path.name
        for path in run_dir.glob("checkpoint_step_*.pt")
        if int(path.stem.rsplit("_", 1)[-1]) > CAPACITY_COMPLETION_TARGET_STEP
    ]
    if later:
        raise ValueError("capacity completion run contains a later checkpoint")
    return {
        "schema_version": CAPACITY_COMPLETION_TRAINING_SCHEMA_VERSION,
        "status": "pass",
        "role": CAPACITY_COMPLETION_TRAINING_ROLE,
        "method": method,
        "run_dir": run_dir.as_posix(),
        "training_report": report_file.as_posix(),
        "config": config_file.as_posix(),
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        },
        "configured_steps": CAPACITY_COMPLETION_TARGET_STEP,
        "source_step": CAPACITY_SCALING_TARGET_STEP,
        "completed_steps": CAPACITY_COMPLETION_TARGET_STEP,
        "training_complete": True,
        "exact_resume": True,
        "last_resume_step": resume_step,
        "recovery_resume_used": resume_step > CAPACITY_SCALING_TARGET_STEP,
        "micro_batch_size": expected_micro_batch_size,
        "gradient_accumulation_steps": expected_gradient_accumulation_steps,
        "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
        "images_seen": CAPACITY_COMPLETION_TARGET_STEP
        * CAPACITY_PROBE_EFFECTIVE_BATCH,
        "parameter_count": expected_parameter_count,
        "dataset_identity_sha256": dataset_sha,
        "runtime_environment_sha256": environment_sha,
        "metric_row_count": len(rows),
        "validation_event_count": len(validation_rows),
        "source_checkpoint_archive": file_identity(archive_path),
        "source_checkpoint": source_checkpoint,
        "checkpoint": checkpoint,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_TRAINING_BOUNDARY
        ),
    }
