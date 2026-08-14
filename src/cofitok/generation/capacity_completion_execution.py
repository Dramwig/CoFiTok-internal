from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_completion_decision import (
    CAPACITY_COMPLETION_DECISION_BOUNDARY,
    CAPACITY_COMPLETION_TARGET_STEP,
    validate_capacity_completion_decision,
)
from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_PARAMETER_COUNTS,
)
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_METHODS,
    CAPACITY_SCALING_TARGET_STEP,
)
from cofitok.generation.capacity_scaling_execution import (
    validate_capacity_scaling_launch_receipt,
)
from cofitok.generation.capacity_scaling_result import (
    CAPACITY_SCALING_RESULT_BOUNDARY,
)


CAPACITY_COMPLETION_LAUNCH_SCHEMA_VERSION = 1
CAPACITY_COMPLETION_LAUNCH_ROLE = (
    "stability_full_data_capacity_completion_100k_launch_receipt"
)
CAPACITY_COMPLETION_STORAGE_STAGE = (
    "stability_full_data_capacity_completion_250m_100k_execution"
)
CAPACITY_COMPLETION_SOURCE_NAMES = {
    "capacity_completion_decision",
    "capacity_scaling_50k_result",
    "capacity_scaling_50k_launch_receipt",
    "source_checkpoint_archive",
    "cofitok_config",
    "dense_config",
    "storage_capacity",
}
CAPACITY_COMPLETION_LAUNCH_BOUNDARY = {
    "matched_250m_resume_authorized": True,
    "fresh_training_allowed": False,
    "resume_from_exact_step_50000_required": True,
    "stop_at_step_100000_required": True,
    "terminal_10000_sample_evaluation_required": True,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_decision_required_after_result": True,
}


def _hex(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _hex(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    expected = {
        "revision": value.get("revision"),
        "tree": value.get("tree"),
        "branch": value.get("branch"),
        "tracked_dirty": False,
    }
    if (
        not _hex(expected["revision"], length=40)
        or not _hex(expected["tree"], length=40)
        or not isinstance(expected["branch"], str)
        or not expected["branch"]
        or dict(value) != expected
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return expected


def _archive(
    report: Mapping[str, Any],
    *,
    decision_identity: Mapping[str, Any],
    resume_sources: Mapping[str, Mapping[str, Any]],
    output_root: str,
) -> dict[str, Any]:
    methods = report.get("methods")
    boundary = report.get("authorization_boundary")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "pass"
        or report.get("role")
        != "capacity_completion_50k_source_checkpoint_archive"
        or report.get("capacity_completion_decision") != dict(decision_identity)
        or not isinstance(methods, Mapping)
        or set(methods) != set(CAPACITY_SCALING_METHODS)
        or not isinstance(boundary, Mapping)
        or boundary.get("training_launch_allowed") is not False
        or boundary.get("source_checkpoint_mutation_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("capacity completion source archive differs")
    normalized = {}
    archive_root = PurePosixPath(output_root) / "reports/capacity_completion_100k/source_checkpoints"
    for method in CAPACITY_SCALING_METHODS:
        row = methods[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"capacity completion {method} archive row is malformed")
        source_checkpoint = _identity(
            row.get("source_checkpoint", {}),
            label=f"capacity completion {method} source checkpoint",
        )
        source_integrity = _identity(
            row.get("source_integrity_manifest", {}),
            label=f"capacity completion {method} source integrity",
        )
        archive_checkpoint = _identity(
            row.get("archive_checkpoint", {}),
            label=f"capacity completion {method} archive checkpoint",
        )
        archive_integrity = _identity(
            row.get("archive_integrity_manifest", {}),
            label=f"capacity completion {method} archive integrity",
        )
        expected = resume_sources[method]
        expected_dir = archive_root / method
        if (
            source_checkpoint != expected["checkpoint"]
            or source_integrity != expected["checkpoint_integrity_manifest"]
            or PurePosixPath(archive_checkpoint["path"]).parent != expected_dir
            or PurePosixPath(archive_integrity["path"]).parent != expected_dir
            or PurePosixPath(archive_checkpoint["path"]).name
            != "checkpoint_step_00050000.pt"
            or archive_integrity["path"]
            != f"{archive_checkpoint['path']}.integrity.json"
            or archive_checkpoint["bytes"] != source_checkpoint["bytes"]
            or archive_checkpoint["sha256"] != source_checkpoint["sha256"]
            or archive_integrity["bytes"] != source_integrity["bytes"]
            or archive_integrity["sha256"] != source_integrity["sha256"]
            or row.get("checkpoint_same_file") is not True
            or row.get("integrity_same_file") is not True
        ):
            raise ValueError(f"capacity completion {method} archive identity differs")
        normalized[method] = {
            "source_checkpoint": source_checkpoint,
            "source_integrity_manifest": source_integrity,
            "archive_checkpoint": archive_checkpoint,
            "archive_integrity_manifest": archive_integrity,
            "checkpoint_same_file": True,
            "integrity_same_file": True,
        }
    return normalized


def _storage(
    report: Mapping[str, Any],
    *,
    execution_git: Mapping[str, Any],
    storage_path: str,
) -> dict[str, Any]:
    filesystem = report.get("filesystem")
    plan = report.get("plan")
    expected_report_git = {
        "revision": execution_git["revision"],
        "branch": execution_git["branch"],
        "tracked_dirty": False,
    }
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != CAPACITY_COMPLETION_STORAGE_STAGE
        or report.get("status") != "pass"
        or report.get("git") != expected_report_git
        or not isinstance(filesystem, Mapping)
        or str(filesystem.get("path", "")) != storage_path
        or not isinstance(plan, Mapping)
        or int(plan.get("sample_count", -1)) < 24_096
        or int(plan.get("checkpoint_count", -1)) < 8
        or float(plan.get("checkpoint_size_multiplier", 0.0)) < 1.0
        or int(report.get("headroom_bytes", -1)) < 0
    ):
        raise ValueError("capacity completion storage evidence differs")
    return {
        "storage_path": storage_path,
        "free_bytes": int(filesystem["free_bytes"]),
        "required_free_bytes": int(plan["required_free_bytes"]),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def build_capacity_completion_launch_receipt(
    *,
    decision: Mapping[str, Any],
    capacity_scaling_result: Mapping[str, Any],
    capacity_scaling_launch_receipt: Mapping[str, Any],
    source_checkpoint_archive: Mapping[str, Any],
    storage_capacity: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    execution_git: Mapping[str, Any],
    training_git: Mapping[str, Any],
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    expected_scaling_execution_revision: str,
    expected_scaling_execution_tree: str,
    expected_scaling_execution_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    output_root: str,
    storage_path: str,
    training_project: str,
    gpu_idle_at_launch: bool,
    relevant_processes_absent_at_launch: bool,
) -> dict[str, Any]:
    if set(source_identities) != CAPACITY_COMPLETION_SOURCE_NAMES:
        raise ValueError("capacity completion launch source set differs")
    if not PurePosixPath(output_root).is_absolute():
        raise ValueError("capacity completion output root must be absolute")
    if (
        not PurePosixPath(storage_path).is_absolute()
        or not PurePosixPath(training_project).is_absolute()
    ):
        raise ValueError("capacity completion execution paths must be absolute")
    sources = {
        name: _identity(identity, label=f"capacity completion launch {name}")
        for name, identity in source_identities.items()
    }
    execution = _git(execution_git, label="capacity completion execution")
    training = _git(training_git, label="capacity completion training")
    if training != {
        "revision": expected_training_revision,
        "tree": expected_training_tree,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion training checkout identity differs")
    decision_evidence = validate_capacity_completion_decision(
        decision,
        expected_decision_revision=expected_decision_revision,
        expected_decision_tree=expected_decision_tree,
        expected_decision_branch=expected_decision_branch,
    )
    if (
        decision_evidence["execution_authorized"] is not True
        or decision.get("authorization_boundary")
        != CAPACITY_COMPLETION_DECISION_BOUNDARY
        or decision.get("source_evidence", {}).get(
            "capacity_scaling_50k_result"
        )
        != sources["capacity_scaling_50k_result"]
        or decision_evidence["output_root"] != output_root
    ):
        raise ValueError("capacity completion launch decision differs")
    if (
        capacity_scaling_result.get("authorization_boundary")
        != CAPACITY_SCALING_RESULT_BOUNDARY
        or capacity_scaling_result.get("decision", {}).get(
            "capacity_completion_supported"
        )
        is not True
        or capacity_scaling_result.get("output_root") != output_root
    ):
        raise ValueError("capacity completion source result differs")
    scaling_launch = validate_capacity_scaling_launch_receipt(
        capacity_scaling_launch_receipt,
        expected_execution_revision=expected_scaling_execution_revision,
        expected_execution_tree=expected_scaling_execution_tree,
        expected_execution_branch=expected_scaling_execution_branch,
        expected_training_revision=expected_training_revision,
        expected_training_tree=expected_training_tree,
        expected_training_branch=expected_training_branch,
        expected_output_root=output_root,
    )
    if (
        sources["cofitok_config"]
        != capacity_scaling_launch_receipt["source_reports"]["cofitok_config"]
        or sources["dense_config"]
        != capacity_scaling_launch_receipt["source_reports"]["dense_config"]
    ):
        raise ValueError("capacity completion config identity differs")
    archive = _archive(
        source_checkpoint_archive,
        decision_identity=sources["capacity_completion_decision"],
        resume_sources=decision_evidence["resume_sources"],
        output_root=output_root,
    )
    storage = _storage(
        storage_capacity,
        execution_git=execution,
        storage_path=storage_path,
    )
    if gpu_idle_at_launch is not True or relevant_processes_absent_at_launch is not True:
        raise ValueError("capacity completion launch requires an idle exclusive GPU")
    runtime = capacity_scaling_launch_receipt["selection"]["runtime_selection"]
    if int(runtime.get("effective_batch_size", -1)) != CAPACITY_PROBE_EFFECTIVE_BATCH:
        raise ValueError("capacity completion runtime selection differs")
    return {
        "schema_version": CAPACITY_COMPLETION_LAUNCH_SCHEMA_VERSION,
        "status": "pass",
        "role": CAPACITY_COMPLETION_LAUNCH_ROLE,
        "git": execution,
        "training_git": training,
        "training_project": training_project,
        "source_reports": sources,
        "selection": {
            "output_root": output_root,
            "dataset": "imagenet_256",
            "methods": list(CAPACITY_SCALING_METHODS),
            "model_base_channels": 256,
            "parameter_counts": copy.deepcopy(
                CAPACITY_PROBE_PARAMETER_COUNTS["base256"]
            ),
            "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
            "runtime_selection": copy.deepcopy(dict(runtime)),
            "resume_sources": archive,
            "resume_from_step": CAPACITY_SCALING_TARGET_STEP,
            "stop_after_step": CAPACITY_COMPLETION_TARGET_STEP,
            "configured_training_horizon": CAPACITY_COMPLETION_TARGET_STEP,
            "milestone_evaluation": copy.deepcopy(
                decision["selection"]["milestone_evaluation"]
            ),
            "terminal_evaluation": copy.deepcopy(
                decision_evidence["terminal_evaluation"]
            ),
        },
        "storage_capacity": storage,
        "launch_checks": {
            "gpu_idle": True,
            "relevant_processes_absent": True,
            "source_step_50000_archives_verified": True,
            "previous_scaling_launch_replayed": bool(scaling_launch),
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_LAUNCH_BOUNDARY
        ),
    }


def validate_capacity_completion_launch_receipt(
    report: Mapping[str, Any],
    *,
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    selection = report.get("selection")
    sources = report.get("source_reports")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_COMPLETION_LAUNCH_SCHEMA_VERSION
        or report.get("status") != "pass"
        or report.get("role") != CAPACITY_COMPLETION_LAUNCH_ROLE
        or report.get("authorization_boundary")
        != CAPACITY_COMPLETION_LAUNCH_BOUNDARY
        or not isinstance(selection, Mapping)
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_COMPLETION_SOURCE_NAMES
    ):
        raise ValueError("capacity completion launch receipt contract differs")
    execution = _git(report.get("git", {}), label="capacity completion launch")
    training = _git(
        report.get("training_git", {}),
        label="capacity completion launch training",
    )
    if execution != {
        "revision": expected_execution_revision,
        "tree": expected_execution_tree,
        "branch": expected_execution_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion launch Git identity differs")
    if training != {
        "revision": expected_training_revision,
        "tree": expected_training_tree,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity completion launch training Git differs")
    for name, identity in sources.items():
        _identity(identity, label=f"capacity completion launch source {name}")
    if (
        selection.get("output_root") != expected_output_root
        or selection.get("dataset") != "imagenet_256"
        or selection.get("methods") != list(CAPACITY_SCALING_METHODS)
        or int(selection.get("model_base_channels", -1)) != 256
        or selection.get("parameter_counts")
        != CAPACITY_PROBE_PARAMETER_COUNTS["base256"]
        or int(selection.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(selection.get("resume_from_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(selection.get("stop_after_step", -1))
        != CAPACITY_COMPLETION_TARGET_STEP
        or int(selection.get("configured_training_horizon", -1))
        != CAPACITY_COMPLETION_TARGET_STEP
        or selection.get("terminal_evaluation", {}).get(
            "sample_count_per_method"
        )
        != 10_000
        or selection.get("terminal_evaluation", {}).get(
            "precision_recall_required"
        )
        is not True
    ):
        raise ValueError("capacity completion launch selection differs")
    return {
        "execution_authorized": True,
        "output_root": expected_output_root,
        "resume_from_step": CAPACITY_SCALING_TARGET_STEP,
        "stop_after_step": CAPACITY_COMPLETION_TARGET_STEP,
        "runtime_selection": copy.deepcopy(selection["runtime_selection"]),
        "resume_sources": copy.deepcopy(selection["resume_sources"]),
        "terminal_evaluation": copy.deepcopy(selection["terminal_evaluation"]),
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_LAUNCH_BOUNDARY
        ),
    }
