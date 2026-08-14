from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_STOP_STEP,
)
from cofitok.generation.capacity_probe_execution import (
    validate_capacity_probe_launch_receipt_contract,
)
from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
)
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_DECISION_BOUNDARY,
    CAPACITY_SCALING_DECISION_ROLE,
    CAPACITY_SCALING_METHODS,
    CAPACITY_SCALING_RECOMMENDATION_ID,
    CAPACITY_SCALING_TARGET_STEP,
    validate_capacity_scaling_decision,
)


CAPACITY_SCALING_LAUNCH_SCHEMA_VERSION = 1
CAPACITY_SCALING_LAUNCH_ROLE = (
    "stability_full_data_capacity_scaling_50k_launch_receipt"
)
CAPACITY_SCALING_STAGE = "stability_capacity_scaling_50k"
CAPACITY_SCALING_STORAGE_STAGE = (
    "stability_full_data_capacity_scaling_250m_50k_execution"
)
CAPACITY_SCALING_SOURCE_NAMES = {
    "capacity_scaling_decision",
    "capacity_probe_result",
    "capacity_probe_launch_receipt",
    "cofitok_config",
    "dense_config",
    "storage_capacity",
}
CAPACITY_SCALING_MIN_CHECKPOINT_RESERVE = 6
CAPACITY_SCALING_MIN_SAMPLE_RESERVE = 4_096
CAPACITY_SCALING_LAUNCH_BOUNDARY = {
    "matched_250m_resume_authorized": True,
    "fresh_training_allowed": False,
    "resume_from_exact_step_10000_required": True,
    "stop_at_step_50000_required": True,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
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


def _git(value: Mapping[str, Any], *, with_tree: bool, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    result: dict[str, Any] = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if (
        not _hex(revision, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    if with_tree:
        tree = value.get("tree")
        if not _hex(tree, length=40):
            raise ValueError(f"{label} Git tree is malformed")
        result["tree"] = tree
    return result


def _validate_storage(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_path: str,
) -> dict[str, Any]:
    filesystem = report.get("filesystem")
    plan = report.get("plan")
    git_without_tree = {
        key: expected_git[key]
        for key in ("revision", "branch", "tracked_dirty")
    }
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != CAPACITY_SCALING_STORAGE_STAGE
        or report.get("status") != "pass"
        or report.get("git") != git_without_tree
        or not isinstance(filesystem, Mapping)
        or not isinstance(plan, Mapping)
        or filesystem.get("path") != expected_path
        or int(plan.get("checkpoint_count", -1))
        < CAPACITY_SCALING_MIN_CHECKPOINT_RESERVE
        or int(plan.get("sample_count", -1))
        < CAPACITY_SCALING_MIN_SAMPLE_RESERVE
        or float(plan.get("checkpoint_size_multiplier", 0.0)) < 1.0
        or int(report.get("headroom_bytes", -1)) < 0
    ):
        raise ValueError("capacity scaling storage evidence differs")
    return {
        "storage_path": expected_path,
        "free_bytes": int(filesystem["free_bytes"]),
        "required_free_bytes": int(plan["required_free_bytes"]),
        "headroom_bytes": int(report["headroom_bytes"]),
        "checkpoint_count": int(plan["checkpoint_count"]),
        "sample_count": int(plan["sample_count"]),
    }


def _validate_capacity_result(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_output_root: str,
) -> None:
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "completed"
        or report.get("role") != CAPACITY_PROBE_RESULT_ROLE
        or report.get("git") != {
            key: expected_git[key]
            for key in ("revision", "branch", "tracked_dirty")
        }
        or report.get("output_root") != expected_output_root
        or report.get("authorization_boundary") != CAPACITY_PROBE_RESULT_BOUNDARY
        or report.get("decision", {}).get("capacity_supported") is not True
    ):
        raise ValueError("capacity scaling requires the supported canonical probe result")


def build_capacity_scaling_launch_receipt(
    *,
    decision: Mapping[str, Any],
    capacity_probe_result: Mapping[str, Any],
    capacity_probe_launch_receipt: Mapping[str, Any],
    storage_capacity: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    resume_checkpoints: Mapping[str, Mapping[str, Mapping[str, Any]]],
    execution_git: Mapping[str, Any],
    training_git: Mapping[str, Any],
    expected_decision_revision: str,
    expected_decision_branch: str,
    output_root: str,
    storage_path: str,
    training_project: str,
    gpu_idle_at_launch: bool,
    relevant_processes_absent_at_launch: bool,
) -> dict[str, Any]:
    if set(source_identities) != CAPACITY_SCALING_SOURCE_NAMES:
        raise ValueError("capacity scaling launch source set differs")
    sources = {
        name: _identity(identity, label=f"capacity scaling launch source {name}")
        for name, identity in source_identities.items()
    }
    if set(resume_checkpoints) != set(CAPACITY_SCALING_METHODS):
        raise ValueError("capacity scaling resume checkpoint set differs")
    if not PurePosixPath(output_root).is_absolute():
        raise ValueError("capacity scaling output root must be absolute")
    if (
        not PurePosixPath(storage_path).is_absolute()
        or not PurePosixPath(training_project).is_absolute()
    ):
        raise ValueError("capacity scaling execution paths must be absolute")
    execution = _git(execution_git, with_tree=True, label="capacity scaling launch")
    training = _git(training_git, with_tree=True, label="capacity scaling training")
    decision_evidence = validate_capacity_scaling_decision(
        decision,
        expected_decision_revision=expected_decision_revision,
        expected_decision_branch=expected_decision_branch,
    )
    if (
        decision.get("role") != CAPACITY_SCALING_DECISION_ROLE
        or decision.get("authorization_boundary")
        != CAPACITY_SCALING_DECISION_BOUNDARY
        or decision_evidence["execution_authorized"] is not True
        or decision.get("recommended_next_stage", {}).get("id")
        != CAPACITY_SCALING_RECOMMENDATION_ID
        or decision.get("selection", {}).get("output_root") != output_root
    ):
        raise ValueError("capacity scaling launch decision does not authorize execution")
    result_source = decision.get("source_evidence", {}).get(
        "capacity_probe_result"
    )
    if result_source != sources["capacity_probe_result"]:
        raise ValueError("capacity scaling decision binds another capacity result")
    _validate_capacity_result(
        capacity_probe_result,
        expected_git=training,
        expected_output_root=output_root,
    )
    if capacity_probe_result.get("source_reports", {}).get("launch_receipt") != (
        sources["capacity_probe_launch_receipt"]
    ):
        raise ValueError("capacity scaling result binds another launch receipt")
    launch = validate_capacity_probe_launch_receipt_contract(
        capacity_probe_launch_receipt,
        expected_revision=str(training["revision"]),
        expected_branch=str(training["branch"]),
    )
    if launch["output_root"] != output_root:
        raise ValueError("capacity scaling source launch output root differs")
    launch_sources = capacity_probe_launch_receipt.get("source_reports", {})
    if (
        launch_sources.get("cofitok_config") != sources["cofitok_config"]
        or launch_sources.get("dense_config") != sources["dense_config"]
    ):
        raise ValueError("capacity scaling config identities differ from source launch")
    decision_resume = decision.get("selection", {}).get("resume_sources")
    if not isinstance(decision_resume, Mapping):
        raise ValueError("capacity scaling decision resume sources are missing")
    normalized_resume: dict[str, Any] = {}
    for method in CAPACITY_SCALING_METHODS:
        observed = resume_checkpoints[method]
        if not isinstance(observed, Mapping):
            raise ValueError(f"capacity scaling {method} checkpoint is malformed")
        checkpoint = _identity(
            observed.get("checkpoint", {}),
            label=f"capacity scaling {method} checkpoint",
        )
        integrity = _identity(
            observed.get("checkpoint_integrity_manifest", {}),
            label=f"capacity scaling {method} integrity",
        )
        expected = decision_resume.get(method)
        if (
            not isinstance(expected, Mapping)
            or checkpoint != expected.get("checkpoint")
            or integrity != expected.get("checkpoint_integrity_manifest")
            or not checkpoint["path"].endswith(
                f"checkpoint_step_{CAPACITY_PROBE_STOP_STEP:08d}.pt"
            )
            or integrity["path"] != f"{checkpoint['path']}.integrity.json"
        ):
            raise ValueError(f"capacity scaling {method} exact resume source differs")
        normalized_resume[method] = {
            "run_dir": str(PurePosixPath(checkpoint["path"]).parent),
            "checkpoint": checkpoint,
            "checkpoint_integrity_manifest": integrity,
        }
    storage = _validate_storage(
        storage_capacity,
        expected_git=execution,
        expected_path=storage_path,
    )
    if gpu_idle_at_launch is not True or relevant_processes_absent_at_launch is not True:
        raise ValueError("capacity scaling launch requires an idle GPU and no duplicate work")
    runtime = capacity_probe_launch_receipt.get("runtime_selection")
    if (
        not isinstance(runtime, Mapping)
        or int(runtime.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
    ):
        raise ValueError("capacity scaling source runtime selection differs")
    return {
        "schema_version": CAPACITY_SCALING_LAUNCH_SCHEMA_VERSION,
        "status": "pass",
        "role": CAPACITY_SCALING_LAUNCH_ROLE,
        "stage": CAPACITY_SCALING_STAGE,
        "git": execution,
        "training_git": training,
        "training_project": training_project,
        "source_reports": sources,
        "selection": {
            "output_root": output_root,
            "dataset": "imagenet_256",
            "methods": list(CAPACITY_SCALING_METHODS),
            "model_base_channels": 256,
            "resume_from_step": CAPACITY_PROBE_STOP_STEP,
            "stop_after_step": CAPACITY_SCALING_TARGET_STEP,
            "configured_training_horizon": 100_000,
            "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
            "runtime_selection": copy.deepcopy(dict(runtime)),
            "resume_sources": normalized_resume,
        },
        "storage_capacity": storage,
        "launch_observation": {
            "gpu_idle": True,
            "relevant_processes_absent": True,
            "exact_step_10000_resume_sources_present": True,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_LAUNCH_BOUNDARY
        ),
    }


def validate_capacity_scaling_launch_receipt(
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
    observation = report.get("launch_observation")
    if (
        int(report.get("schema_version", -1))
        != CAPACITY_SCALING_LAUNCH_SCHEMA_VERSION
        or report.get("status") != "pass"
        or report.get("role") != CAPACITY_SCALING_LAUNCH_ROLE
        or report.get("stage") != CAPACITY_SCALING_STAGE
        or report.get("authorization_boundary")
        != CAPACITY_SCALING_LAUNCH_BOUNDARY
        or not isinstance(selection, Mapping)
        or not isinstance(observation, Mapping)
    ):
        raise ValueError("capacity scaling launch receipt contract differs")
    execution = _git(report.get("git", {}), with_tree=True, label="launch receipt")
    training = _git(
        report.get("training_git", {}),
        with_tree=True,
        label="launch training",
    )
    if execution != {
        "revision": expected_execution_revision,
        "tree": expected_execution_tree,
        "branch": expected_execution_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity scaling launch execution Git differs")
    if training != {
        "revision": expected_training_revision,
        "tree": expected_training_tree,
        "branch": expected_training_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("capacity scaling launch training Git differs")
    sources = report.get("source_reports")
    if not isinstance(sources, Mapping) or set(sources) != CAPACITY_SCALING_SOURCE_NAMES:
        raise ValueError("capacity scaling launch source set differs")
    for name, identity in sources.items():
        _identity(identity, label=f"capacity scaling launch source {name}")
    if (
        selection.get("output_root") != expected_output_root
        or selection.get("methods") != list(CAPACITY_SCALING_METHODS)
        or int(selection.get("resume_from_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or int(selection.get("stop_after_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(selection.get("configured_training_horizon", -1)) != 100_000
        or int(selection.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or observation.get("gpu_idle") is not True
        or observation.get("relevant_processes_absent") is not True
        or observation.get("exact_step_10000_resume_sources_present") is not True
    ):
        raise ValueError("capacity scaling launch selection differs")
    return {
        "git": execution,
        "training_git": training,
        "output_root": expected_output_root,
        "resume_from_step": CAPACITY_PROBE_STOP_STEP,
        "stop_after_step": CAPACITY_SCALING_TARGET_STEP,
        "effective_batch_size": CAPACITY_PROBE_EFFECTIVE_BATCH,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_LAUNCH_BOUNDARY
        ),
    }
