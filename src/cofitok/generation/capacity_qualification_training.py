from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import runtime_environment_sha256
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    checkpoint_integrity_path,
    resolve_latest_checkpoint,
    verify_training_checkpoint,
)


PARTIAL_TRAINING_SCHEMA_VERSION = 1
PARTIAL_TRAINING_ROLE = "generation_capacity_qualification_partial_training_validation"
CAPACITY_QUALIFICATION_STAGE = "stability_capacity_qualification"
CAPACITY_REFERENCE_STAGE = "stability_capacity_reference"
CAPACITY_QUALIFICATION_STAGES = {
    CAPACITY_REFERENCE_STAGE: 128,
    CAPACITY_QUALIFICATION_STAGE: 256,
}
CAPACITY_QUALIFICATION_STOP_STEP = 10_000
CAPACITY_QUALIFICATION_CONFIGURED_STEPS = 100_000
CAPACITY_QUALIFICATION_EFFECTIVE_BATCH = 64
CAPACITY_QUALIFICATION_DATASET = "imagenet_256"


def _read(path: Path, *, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _canonical_git(
    *,
    revision: str,
    branch: str,
) -> dict[str, Any]:
    if len(revision) != 40:
        raise ValueError("capacity qualification revision must contain 40 characters")
    try:
        int(revision, 16)
    except ValueError as error:
        raise ValueError("capacity qualification revision must be hexadecimal") from error
    if not branch:
        raise ValueError("capacity qualification branch is missing")
    return {"revision": revision, "branch": branch, "dirty": False}


def _checkpoint_step(path: Path, *, label: str) -> int:
    prefix = "checkpoint_step_"
    if path.suffix != ".pt" or not path.stem.startswith(prefix):
        raise ValueError(f"{label} filename is malformed")
    try:
        return int(path.stem[len(prefix) :])
    except ValueError as error:
        raise ValueError(f"{label} filename is malformed") from error


def _validate_exact_resume_state(
    report: dict[str, Any],
    *,
    run_dir: Path,
    metrics_path: Path,
    expected_revision: str,
    expected_branch: str,
    expected_runtime_environment_sha256: str,
    expected_dataset_identity_sha256: str,
    expected_stop_step: int,
    allow_exact_resume: bool,
) -> dict[str, Any]:
    resume_value = report.get("resume")
    transition = report.get("resume_revision_transition")
    reconciliation = report.get("metrics_resume_reconciliation")
    horizon_extension = report.get("horizon_extension")
    if resume_value is None:
        if transition is not None or reconciliation is not None:
            raise ValueError("fresh capacity qualification has unexpected resume state")
        if horizon_extension is not None:
            raise ValueError("capacity qualification must not extend its horizon")
        return {
            "resumed": False,
            "resume_checkpoint": None,
            "resume_step": None,
            "metrics_reconciliation": None,
        }
    if not allow_exact_resume:
        raise ValueError("capacity qualification resume requires explicit validation")
    if transition is not None:
        raise ValueError("capacity qualification cannot change revision while resuming")
    if horizon_extension is not None:
        raise ValueError("capacity qualification must not extend its horizon")
    resume_path = reject_symlink_chain(
        Path(str(resume_value)), name="capacity qualification resume checkpoint"
    ).resolve()
    if not resume_path.is_file() or resume_path.parent != run_dir:
        raise ValueError("capacity qualification resume checkpoint is outside its run")
    resume_step = _checkpoint_step(
        resume_path, label="capacity qualification resume checkpoint"
    )
    if not 0 < resume_step < expected_stop_step:
        raise ValueError("capacity qualification resume step is outside the bounded run")
    integrity = verify_training_checkpoint(resume_path)
    if (
        int(integrity.get("step", -1)) != resume_step
        or integrity.get("git_revision") != expected_revision
        or integrity.get("git_branch") != expected_branch
        or integrity.get("git_dirty") is not False
        or integrity.get("runtime_environment_sha256")
        != expected_runtime_environment_sha256
        or integrity.get("dataset_identity_sha256")
        != expected_dataset_identity_sha256
    ):
        raise ValueError("capacity qualification resume checkpoint identity differs")
    if not isinstance(reconciliation, dict):
        raise ValueError("capacity qualification metrics reconciliation is missing")
    status = reconciliation.get("status")
    if (
        int(reconciliation.get("schema_version", -1)) != 1
        or status not in {"unchanged", "reconciled"}
        or int(reconciliation.get("resume_step", -1)) != resume_step
        or Path(str(reconciliation.get("metrics", ""))).resolve() != metrics_path
        or int(reconciliation.get("retained_rows", -1)) < 1
        or int(reconciliation.get("orphaned_rows", -1)) < 0
    ):
        raise ValueError("capacity qualification metrics reconciliation differs")
    if status == "unchanged":
        if (
            int(reconciliation.get("orphaned_rows", -1)) != 0
            or reconciliation.get("orphan_archive") is not None
            or reconciliation.get("orphan_sha256") is not None
            or "report" in reconciliation
        ):
            raise ValueError("unchanged capacity metrics reconciliation is malformed")
    else:
        archive = reject_symlink_chain(
            Path(str(reconciliation.get("orphan_archive", ""))),
            name="capacity qualification metrics orphan archive",
        ).resolve()
        reconciliation_report = reject_symlink_chain(
            Path(str(reconciliation.get("report", ""))),
            name="capacity qualification metrics reconciliation report",
        ).resolve()
        orphan_sha = reconciliation.get("orphan_sha256")
        if (
            int(reconciliation.get("orphaned_rows", 0)) < 1
            or not archive.is_file()
            or not reconciliation_report.is_file()
            or not isinstance(orphan_sha, str)
            or len(orphan_sha) != 64
            or file_sha256(archive) != orphan_sha
        ):
            raise ValueError("reconciled capacity metrics archive differs")
        persisted = _read(
            reconciliation_report,
            name="capacity qualification metrics reconciliation report",
        )
        if persisted != {
            key: value for key, value in reconciliation.items() if key != "report"
        }:
            raise ValueError("capacity metrics reconciliation report differs")
    return {
        "resumed": True,
        "resume_checkpoint": resume_path.as_posix(),
        "resume_step": resume_step,
        "metrics_reconciliation": dict(reconciliation),
    }


def validate_capacity_qualification_partial_training(
    *,
    report_path: str | Path,
    config_path: str | Path,
    expected_revision: str,
    expected_branch: str,
    expected_parameter_count: int,
    expected_micro_batch_size: int,
    expected_gradient_accumulation_steps: int,
    expected_stage: str = CAPACITY_QUALIFICATION_STAGE,
    expected_base_channels: int = 256,
    expected_stop_step: int = CAPACITY_QUALIFICATION_STOP_STEP,
    expected_configured_steps: int = CAPACITY_QUALIFICATION_CONFIGURED_STEPS,
    expected_dataset: str = CAPACITY_QUALIFICATION_DATASET,
    allow_exact_resume: bool = False,
) -> dict[str, Any]:
    if (
        expected_stop_step < 1
        or expected_configured_steps <= expected_stop_step
        or expected_parameter_count < 1
        or expected_micro_batch_size < 1
        or expected_gradient_accumulation_steps < 1
    ):
        raise ValueError("capacity qualification expectation is invalid")
    if CAPACITY_QUALIFICATION_STAGES.get(expected_stage) != expected_base_channels:
        raise ValueError("capacity qualification stage/base-channel expectation differs")
    expected_git = _canonical_git(
        revision=expected_revision,
        branch=expected_branch,
    )
    effective_batch = (
        expected_micro_batch_size * expected_gradient_accumulation_steps
    )
    if effective_batch != CAPACITY_QUALIFICATION_EFFECTIVE_BATCH:
        raise ValueError("capacity qualification must use effective batch 64")

    report_file = reject_symlink_chain(
        report_path,
        name="capacity qualification training report",
    ).resolve()
    config_file = reject_symlink_chain(
        config_path,
        name="capacity qualification config",
    ).resolve()
    run_dir = reject_symlink_chain(
        report_file.parent,
        name="capacity qualification run directory",
    ).resolve()
    report = _read(report_file, name="capacity qualification training report")
    if (
        report.get("training_complete") is not False
        or int(report.get("completed_steps", -1)) != expected_stop_step
        or int(report.get("target_steps", -1)) != expected_configured_steps
        or report.get("stop_requested") is not False
        or report.get("stop_signal") is not None
    ):
        raise ValueError("capacity qualification report is not the exact clean stop")

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
        raise ValueError("capacity qualification resolved training config differs")
    if (
        config.runtime.steps != expected_configured_steps
        or config.data.dataset != expected_dataset
        or config.runtime.protected_checkpoint_steps != [expected_stop_step]
        or config.model.base_channels != expected_base_channels
    ):
        raise ValueError(
            "capacity qualification configured horizon/dataset/protection/capacity differs"
        )

    if report.get("git") != expected_git:
        raise ValueError("capacity qualification training Git identity differs")
    if report.get("training_authorization") is not None:
        raise ValueError("bounded capacity qualification must not consume a full-training gate")
    if (
        int(report.get("parameter_count", -1)) != expected_parameter_count
        or int(report.get("trainable_parameter_count", -1))
        != expected_parameter_count
    ):
        raise ValueError("capacity qualification parameter count differs")
    if Path(str(report.get("config_path", ""))).resolve() != config_file:
        raise ValueError("capacity qualification report config path differs")
    if Path(str(report.get("output_dir", ""))).resolve() != run_dir:
        raise ValueError("capacity qualification report output directory differs")

    environment = report.get("runtime_environment")
    if not isinstance(environment, dict):
        raise ValueError("capacity qualification runtime environment is missing")
    environment_sha = runtime_environment_sha256(environment)
    if report.get("runtime_environment_sha256") != environment_sha:
        raise ValueError("capacity qualification runtime environment SHA256 differs")
    provenance = report.get("dataset_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("capacity qualification dataset provenance is missing")
    validated_dataset = validate_dataset_provenance(
        provenance,
        expected_dataset=expected_dataset,
    )
    dataset_sha = validated_dataset["identity_sha256"]

    final = report.get("final_metrics")
    if not isinstance(final, dict):
        raise ValueError("capacity qualification final metrics are missing")
    if (
        int(final.get("step", -1)) != expected_stop_step
        or int(final.get("samples_seen", -1))
        != expected_stop_step * effective_batch
    ):
        raise ValueError("capacity qualification sample/step accounting differs")
    for field in (
        "total",
        "epsilon",
        "grad_norm",
        "learning_rate",
        "elapsed_seconds",
        "cumulative_elapsed_seconds",
    ):
        _finite(final.get(field), label=f"final_metrics.{field}", positive=True)
    _finite(report.get("segment_elapsed_seconds"), label="segment_elapsed_seconds", positive=True)
    _finite(report.get("elapsed_seconds"), label="elapsed_seconds", positive=True)
    if int(report.get("peak_vram_bytes", 0)) < 1:
        raise ValueError("capacity qualification peak VRAM accounting is missing")

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read(manifest_path, name="capacity qualification run manifest")
    for key, value in manifest.items():
        if report.get(key) != value:
            raise ValueError(f"capacity qualification run manifest differs: {key}")

    metrics_path = reject_symlink_chain(
        run_dir / "train_metrics.jsonl",
        name="capacity qualification metrics",
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"capacity qualification metrics are missing: {metrics_path}"
        )
    resume = _validate_exact_resume_state(
        report,
        run_dir=run_dir,
        metrics_path=metrics_path.resolve(),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_runtime_environment_sha256=environment_sha,
        expected_dataset_identity_sha256=dataset_sha,
        expected_stop_step=expected_stop_step,
        allow_exact_resume=allow_exact_resume,
    )
    rows: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"capacity qualification metric row {line_number} is invalid JSON"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"capacity qualification metric row {line_number} is not an object"
                )
            rows.append(row)
    steps = [int(row.get("step", -1)) for row in rows]
    if (
        not steps
        or steps[0] != 1
        or steps[-1] != expected_stop_step
        or any(left >= right for left, right in zip(steps, steps[1:]))
    ):
        raise ValueError(
            "capacity qualification metric steps are not canonical and increasing"
        )
    if rows[-1] != final:
        raise ValueError(
            "capacity qualification final metric row differs from training report"
        )
    for index, row in enumerate(rows):
        if int(row.get("samples_seen", -1)) != int(row["step"]) * effective_batch:
            raise ValueError(
                f"capacity qualification metric sample accounting differs at row {index}"
            )
        for field in ("total", "epsilon"):
            _finite(row.get(field), label=f"metrics[{index}].{field}")
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    expected_validation_events = (
        expected_stop_step // config.runtime.evaluation_interval
    )
    if (
        len(validation_rows) != expected_validation_events
        or [int(row.get("validation_event_index", -1)) for row in validation_rows]
        != list(range(expected_validation_events))
        or [int(row.get("step", -1)) for row in validation_rows]
        != list(
            range(
                config.runtime.evaluation_interval,
                expected_stop_step + 1,
                config.runtime.evaluation_interval,
            )
        )
    ):
        raise ValueError("capacity qualification scheduled validation events differ")
    for index, row in enumerate(validation_rows):
        _finite(
            row.get("validation_epsilon_mse"),
            label=f"validation[{index}].validation_epsilon_mse",
        )

    latest_path = run_dir / "latest.json"
    latest = _read(latest_path, name="capacity qualification latest pointer")
    if report.get("latest_checkpoint") != latest:
        raise ValueError("capacity qualification report and latest.json differ")
    checkpoint_name = f"checkpoint_step_{expected_stop_step:08d}.pt"
    checkpoint = resolve_latest_checkpoint(run_dir)
    if checkpoint.name != checkpoint_name or checkpoint.parent.resolve() != run_dir:
        raise ValueError("capacity qualification latest checkpoint differs")
    integrity = verify_training_checkpoint(checkpoint)
    if (
        int(integrity.get("step", -1)) != expected_stop_step
        or int(integrity.get("checkpoint_format_version", -1))
        != CHECKPOINT_FORMAT_VERSION
        or integrity.get("git_revision") != expected_revision
        or integrity.get("git_branch") != expected_branch
        or integrity.get("git_dirty") is not False
        or integrity.get("runtime_environment_sha256") != environment_sha
        or integrity.get("dataset_identity_sha256") != dataset_sha
        or latest.get("checkpoint_sha256") != integrity.get("checkpoint_sha256")
        or latest.get("checkpoint_bytes") != integrity.get("checkpoint_bytes")
        or latest.get("integrity_manifest")
        != checkpoint_integrity_path(checkpoint).name
    ):
        raise ValueError("capacity qualification checkpoint integrity identity differs")
    if any(key.startswith("authorization_") for key in integrity):
        raise ValueError(
            "capacity qualification checkpoint unexpectedly carries full-training authorization"
        )
    later = sorted(
        path.name
        for path in run_dir.glob("checkpoint_step_*.pt")
        if int(path.stem.rsplit("_", 1)[-1]) > expected_stop_step
    )
    if later:
        raise ValueError(
            "capacity qualification run contains checkpoints after the stop step"
        )

    integrity_path = checkpoint_integrity_path(checkpoint)
    return {
        "schema_version": PARTIAL_TRAINING_SCHEMA_VERSION,
        "status": "pass",
        "role": PARTIAL_TRAINING_ROLE,
        "stage": expected_stage,
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
        "fresh_initialization_verified": True,
        "exact_resume": resume,
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
            "step": expected_stop_step,
            "integrity_manifest": {
                "path": integrity_path.as_posix(),
                "bytes": integrity_path.stat().st_size,
                "sha256": file_sha256(integrity_path),
            },
        },
        "authorization_boundary": {
            "capacity_qualification_training_complete": True,
            "capacity_screen_allowed": False,
            "capacity_confirmation_allowed": False,
            "additional_training_allowed": False,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "formal_generation_claim_allowed": False,
            "release_allowed": False,
        },
    }


__all__ = [
    "CAPACITY_QUALIFICATION_CONFIGURED_STEPS",
    "CAPACITY_QUALIFICATION_DATASET",
    "CAPACITY_QUALIFICATION_EFFECTIVE_BATCH",
    "CAPACITY_QUALIFICATION_STAGE",
    "CAPACITY_REFERENCE_STAGE",
    "CAPACITY_QUALIFICATION_STAGES",
    "CAPACITY_QUALIFICATION_STOP_STEP",
    "PARTIAL_TRAINING_ROLE",
    "PARTIAL_TRAINING_SCHEMA_VERSION",
    "validate_capacity_qualification_partial_training",
]
