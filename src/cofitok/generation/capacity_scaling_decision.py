"""Source-bound scientific decision after capacity confirmation.

The decision replays the completed four-arm confirmation and selects whether a
separate exact-resume scaling stage may be prepared. It is deliberately not
an execution authorization: no GPU work, training, 300K launch, promotion, or
release can be enabled by this report.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from cofitok.generation.capacity_confirmation_arm import (
    validate_capacity_confirmation_arm_validation,
)
from cofitok.generation.capacity_confirmation_result import (
    RESULT_BOUNDARY as CONFIRMATION_RESULT_BOUNDARY,
    RESULT_ROLE as CONFIRMATION_RESULT_ROLE,
    RESULT_SCHEMA as CONFIRMATION_RESULT_SCHEMA,
    validate_capacity_confirmation_result_contract,
)
from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    CONFIGURED_STEPS,
    EFFECTIVE_BATCH,
    STOP_STEP,
)


CAPACITY_SCALING_DECISION_SCHEMA = (
    "cofitok_generation_capacity_scaling_readiness_decision_v2"
)
CAPACITY_SCALING_DECISION_ROLE = (
    "source_bound_capacity_scaling_readiness_decision"
)
CAPACITY_SCALING_RECOMMENDATION_ID = (
    "prepare_exact_base256_10k_to_50k_capacity_scaling"
)
CAPACITY_SCALING_HOLD_ID = "hold_capacity_scaling_after_failed_confirmation"
CAPACITY_SCALING_TARGET_STEP = 50_000
CAPACITY_SCALING_CAPACITY = "base256"
CAPACITY_SCALING_ARMS = (
    "base256_cofitok",
    "base256_dense_identity",
)
CAPACITY_SCALING_METHODS = ("cofitok", "dense_identity")
CAPACITY_SCALING_DIRNAME = "capacity_scaling_50000"

CAPACITY_SCALING_DECISION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "capacity_scaling_preparation_allowed": "derived_from_confirmation",
    "capacity_scaling_launch_allowed": False,
    "configured_100k_completion_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    if (
        not isinstance(row.get("path"), str)
        or not row["path"]
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or row["bytes"] < 1
        or not _hex(row.get("sha256"), 64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not _hex(row.get("revision"), 40)
        or not _hex(row.get("tree"), 40)
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return copy.deepcopy(row)


def _absolute(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not PurePosixPath(value).is_absolute()
    ):
        raise ValueError(f"{name} must be an absolute POSIX path")
    return value


def _checkpoint(value: Any, *, arm: str) -> dict[str, Any]:
    row = _object(value, f"{arm} frozen checkpoint")
    if set(row) != {"path", "bytes", "sha256", "step", "integrity_manifest"}:
        raise ValueError(f"{arm} frozen checkpoint fields differ")
    checkpoint = _identity(
        {key: row[key] for key in ("path", "bytes", "sha256")},
        f"{arm} frozen checkpoint",
    )
    integrity = _identity(
        row.get("integrity_manifest"), f"{arm} checkpoint integrity"
    )
    expected_name = f"checkpoint_step_{STOP_STEP:08d}.pt"
    if (
        int(row.get("step", -1)) != STOP_STEP
        or not checkpoint["path"].endswith(f"/{expected_name}")
        or integrity["path"] != f"{checkpoint['path']}.integrity.json"
    ):
        raise ValueError(f"{arm} frozen checkpoint is not the exact step-10K source")
    return {
        **checkpoint,
        "step": STOP_STEP,
        "integrity_manifest": integrity,
    }


def _resume_source(
    report: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    arm: str,
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    row = validate_capacity_confirmation_arm_validation(report)
    spec = ARM_SPECS[arm]
    if (
        row.get("arm") != arm
        or row.get("capacity") != spec["capacity"]
        or row.get("method") != spec["method"]
        or row.get("execution_git") != expected_git
    ):
        raise ValueError(f"{arm} confirmation arm identity differs")
    frozen = _object(row.get("frozen_training"), f"{arm} frozen training")
    if frozen.get("training_performed") is not False:
        raise ValueError(f"{arm} confirmation unexpectedly performed training")
    checkpoint = _checkpoint(frozen.get("checkpoint"), arm=arm)
    config = _identity(frozen.get("config"), f"{arm} frozen config")
    training_report = _identity(
        frozen.get("training_report"), f"{arm} frozen training report"
    )
    screen = _object(frozen.get("screen_training"), f"{arm} screen training")
    run_dir = _absolute(screen.get("run_dir"), f"{arm} source run")
    if (
        screen.get("checkpoint") != checkpoint
        or screen.get("config") != config["path"]
        or screen.get("training_report") != training_report["path"]
        or screen.get("stage") != spec["recipe_stage"]
        or int(screen.get("configured_steps", -1)) != CONFIGURED_STEPS
        or int(screen.get("completed_steps", -1)) != STOP_STEP
        or screen.get("training_complete") is not False
        or screen.get("intentional_partial_stop") is not True
        or int(screen.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or int(screen.get("images_seen", -1)) != STOP_STEP * EFFECTIVE_BATCH
        or int(screen.get("parameter_count", -1)) != spec["parameter_count"]
        or not _hex(screen.get("runtime_environment_sha256"), 64)
        or not _hex(screen.get("dataset_identity_sha256"), 64)
        or PurePosixPath(checkpoint["path"]).parent.as_posix() != run_dir
        or PurePosixPath(run_dir).name != arm
    ):
        raise ValueError(f"{arm} frozen resume source differs")
    sources = _object(row.get("sources"), f"{arm} confirmation sources")
    return {
        "arm": arm,
        "method": spec["method"],
        "capacity": spec["capacity"],
        "base_channels": spec["base_channels"],
        "parameter_count": spec["parameter_count"],
        "effective_batch_size": EFFECTIVE_BATCH,
        "images_seen": STOP_STEP * EFFECTIVE_BATCH,
        "source_run_dir": run_dir,
        "config": config,
        "training_report": training_report,
        "checkpoint": checkpoint,
        "screen_arm_validation": _identity(
            sources.get("screen_arm_validation"),
            f"{arm} screen arm validation",
        ),
        "confirmation_arm_validation": _identity(
            identity, f"{arm} confirmation arm validation"
        ),
        "runtime_environment_sha256": screen["runtime_environment_sha256"],
        "dataset_identity_sha256": screen["dataset_identity_sha256"],
    }


def build_capacity_scaling_decision(
    *,
    capacity_confirmation_result: Mapping[str, Any],
    capacity_confirmation_result_identity: Mapping[str, Any],
    arm_validations: Mapping[str, Mapping[str, Any]],
    arm_validation_identities: Mapping[str, Mapping[str, Any]],
    confirmation_checkout: Mapping[str, Any],
    decision_git: Mapping[str, Any],
) -> dict[str, Any]:
    confirmation = validate_capacity_confirmation_result_contract(
        capacity_confirmation_result
    )
    result_id = _identity(
        capacity_confirmation_result_identity, "capacity confirmation result"
    )
    source_git = _git(confirmation_checkout, "capacity confirmation checkout")
    if confirmation.get("result_git") != source_git:
        raise ValueError("capacity confirmation result Git identity differs")
    if (
        confirmation.get("schema_version") != CONFIRMATION_RESULT_SCHEMA
        or confirmation.get("role") != CONFIRMATION_RESULT_ROLE
        or confirmation.get("authorization_boundary")
        != CONFIRMATION_RESULT_BOUNDARY
        or set(arm_validations) != set(ARM_NAMES)
        or set(arm_validation_identities) != set(ARM_NAMES)
    ):
        raise ValueError("capacity scaling requires the canonical confirmation result")
    expected_ids = _object(
        _object(
            confirmation.get("source_evidence"), "capacity confirmation sources"
        ).get("arm_validations"),
        "capacity confirmation arm identities",
    )
    normalized_ids: dict[str, dict[str, Any]] = {}
    validated_arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_id = _identity(
            arm_validation_identities[arm], f"{arm} confirmation validation"
        )
        if expected_ids.get(arm) != arm_id:
            raise ValueError(f"{arm} validation differs from confirmation result")
        validated_arms[arm] = validate_capacity_confirmation_arm_validation(
            arm_validations[arm]
        )
        normalized_ids[arm] = arm_id

    resume_sources: dict[str, dict[str, Any]] = {}
    for arm in CAPACITY_SCALING_ARMS:
        method = str(ARM_SPECS[arm]["method"])
        resume_sources[method] = _resume_source(
            validated_arms[arm],
            normalized_ids[arm],
            arm=arm,
            expected_git=source_git,
        )
    source_roots = {
        PurePosixPath(row["source_run_dir"]).parent.as_posix()
        for row in resume_sources.values()
    }
    if len(source_roots) != 1:
        raise ValueError("base256 resume sources do not share one screen root")
    source_root = next(iter(source_roots))
    runtime_shas = {
        row["runtime_environment_sha256"] for row in resume_sources.values()
    }
    dataset_shas = {row["dataset_identity_sha256"] for row in resume_sources.values()}
    if len(runtime_shas) != 1 or len(dataset_shas) != 1:
        raise ValueError("base256 matched resume provenance differs")

    passed = (
        confirmation.get("scientific_status") == "confirmation_pass"
        and confirmation.get("failed_checks") == []
        and _object(
            confirmation.get("next_stage"), "capacity confirmation next stage"
        ).get("support_collapse_resolved")
        is True
    )
    next_stage = {
        "route": "capacity_scaling_preparation" if passed else "hold",
        "reason": (
            "capacity_confirmation_passed_all_support_class_and_mechanism_checks"
            if passed
            else "capacity_confirmation_did_not_clear_all_predeclared_checks"
        ),
        "capacity_scaling_preparation_allowed": passed,
        "separate_stage_authorization_required": True,
        "execution_ready": False,
        "remote_mutation_allowed": False,
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
        "configured_100k_completion_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "release_allowed": False,
        "required_next_evidence": (
            "A source-bound preparation, user stage authorization, execution "
            "authorization, idle live snapshot, storage preflight, and immutable "
            "launch receipt for only the exact base256 step-10K to step-50K pair."
            if passed
            else "A new matched intervention justified by the named failed checks."
        ),
    }
    continuation_root = (
        PurePosixPath(source_root) / CAPACITY_SCALING_DIRNAME
    ).as_posix()
    output_dirs = {
        method: (PurePosixPath(continuation_root) / row["arm"]).as_posix()
        for method, row in resume_sources.items()
    }
    return {
        "schema_version": CAPACITY_SCALING_DECISION_SCHEMA,
        "role": CAPACITY_SCALING_DECISION_ROLE,
        "status": "completed",
        "scientific_status": "scaling_preparation_selected" if passed else "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "decision_git": _git(decision_git, "capacity scaling decision Git"),
        "source_evidence": {
            "capacity_confirmation_result": result_id,
            "capacity_confirmation_git": source_git,
            "capacity_confirmation_arm_validations": normalized_ids,
        },
        "qualification": {
            "confirmation_scientific_status": confirmation["scientific_status"],
            "support_collapse_resolved": passed,
            "failed_checks": copy.deepcopy(confirmation["failed_checks"]),
            "check_count": len(confirmation["checks"]),
        },
        "selection": {
            "dataset": "imagenet_256",
            "capacity": CAPACITY_SCALING_CAPACITY,
            "base_channels": 256,
            "methods": list(CAPACITY_SCALING_METHODS),
            "configured_training_horizon": CONFIGURED_STEPS,
            "resume_from_step": STOP_STEP,
            "stop_after_step": CAPACITY_SCALING_TARGET_STEP,
            "effective_batch_size": EFFECTIVE_BATCH,
            "controlled_change": "training_exposure_only",
            "fresh_initialization_allowed": False,
            "source_screen_root": source_root,
            "output_root": continuation_root,
            "output_dirs": output_dirs,
            "resume_sources": resume_sources,
            "milestone_evaluation": {
                "sample_count_per_method": 2_048,
                "weights": "ema",
                "sampler": "ddim",
                "sample_steps": 50,
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "fixed_random_stream_across_methods": True,
                "cofitok_mechanism_images": 256,
                "rollout_stability_required": True,
            },
        },
        "next_stage": next_stage,
        "claim_policy": {
            "role": "non_claim_capacity_scaling_readiness",
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
            "cross_stage_numeric_ranking_allowed": False,
            "step_50000_requires_new_source_bound_result_and_decision": True,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_DECISION_BOUNDARY
        ),
    }


def validate_capacity_scaling_decision(
    report: Mapping[str, Any],
    *,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
) -> dict[str, Any]:
    row = _object(report, "capacity scaling decision")
    selection = _object(row.get("selection"), "capacity scaling selection")
    qualification = _object(
        row.get("qualification"), "capacity scaling qualification"
    )
    next_stage = _object(row.get("next_stage"), "capacity scaling next stage")
    passed = qualification.get("support_collapse_resolved") is True
    expected_git = {
        "revision": expected_decision_revision,
        "tree": expected_decision_tree,
        "branch": expected_decision_branch,
        "tracked_dirty": False,
    }
    if (
        row.get("schema_version") != CAPACITY_SCALING_DECISION_SCHEMA
        or row.get("role") != CAPACITY_SCALING_DECISION_ROLE
        or row.get("status") != "completed"
        or row.get("scientific_status")
        != ("scaling_preparation_selected" if passed else "hold")
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("decision_git") != _git(expected_git, "expected decision Git")
        or row.get("authorization_boundary")
        != CAPACITY_SCALING_DECISION_BOUNDARY
        or "execution_authorization" in row
        or next_stage.get("capacity_scaling_preparation_allowed") is not passed
        or next_stage.get("separate_stage_authorization_required") is not True
        or next_stage.get("execution_ready") is not False
        or next_stage.get("remote_mutation_allowed") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("configured_100k_completion_allowed") is not False
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or next_stage.get("promotion_allowed") is not False
        or next_stage.get("release_allowed") is not False
        or selection.get("dataset") != "imagenet_256"
        or selection.get("capacity") != CAPACITY_SCALING_CAPACITY
        or int(selection.get("base_channels", -1)) != 256
        or selection.get("methods") != list(CAPACITY_SCALING_METHODS)
        or int(selection.get("configured_training_horizon", -1))
        != CONFIGURED_STEPS
        or int(selection.get("resume_from_step", -1)) != STOP_STEP
        or int(selection.get("stop_after_step", -1))
        != CAPACITY_SCALING_TARGET_STEP
        or int(selection.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
        or selection.get("controlled_change") != "training_exposure_only"
        or selection.get("fresh_initialization_allowed") is not False
    ):
        raise ValueError("capacity scaling decision contract differs")
    source_root = _absolute(
        selection.get("source_screen_root"), "capacity scaling source root"
    )
    output_root = _absolute(
        selection.get("output_root"), "capacity scaling output root"
    )
    expected_output_root = (
        PurePosixPath(source_root) / CAPACITY_SCALING_DIRNAME
    ).as_posix()
    output_dirs = _object(selection.get("output_dirs"), "capacity scaling outputs")
    resume_sources = _object(
        selection.get("resume_sources"), "capacity scaling resume sources"
    )
    if (
        output_root != expected_output_root
        or set(output_dirs) != set(CAPACITY_SCALING_METHODS)
        or set(resume_sources) != set(CAPACITY_SCALING_METHODS)
    ):
        raise ValueError("capacity scaling output selection differs")
    runtime_shas: set[str] = set()
    dataset_shas: set[str] = set()
    for arm, method in zip(CAPACITY_SCALING_ARMS, CAPACITY_SCALING_METHODS):
        resume = _object(resume_sources[method], f"{method} resume source")
        checkpoint = _checkpoint(resume.get("checkpoint"), arm=arm)
        if (
            resume.get("arm") != arm
            or resume.get("method") != method
            or resume.get("capacity") != CAPACITY_SCALING_CAPACITY
            or int(resume.get("base_channels", -1)) != 256
            or int(resume.get("parameter_count", -1))
            != ARM_SPECS[arm]["parameter_count"]
            or int(resume.get("effective_batch_size", -1)) != EFFECTIVE_BATCH
            or int(resume.get("images_seen", -1)) != STOP_STEP * EFFECTIVE_BATCH
            or PurePosixPath(checkpoint["path"]).parent.as_posix()
            != resume.get("source_run_dir")
            or output_dirs[method]
            != (PurePosixPath(output_root) / arm).as_posix()
        ):
            raise ValueError(f"{method} capacity scaling resume source differs")
        _identity(resume.get("config"), f"{method} config")
        _identity(resume.get("training_report"), f"{method} training report")
        _identity(
            resume.get("screen_arm_validation"), f"{method} screen validation"
        )
        _identity(
            resume.get("confirmation_arm_validation"),
            f"{method} confirmation validation",
        )
        if not _hex(resume.get("runtime_environment_sha256"), 64) or not _hex(
            resume.get("dataset_identity_sha256"), 64
        ):
            raise ValueError(f"{method} capacity scaling provenance differs")
        runtime_shas.add(resume["runtime_environment_sha256"])
        dataset_shas.add(resume["dataset_identity_sha256"])
    if len(runtime_shas) != 1 or len(dataset_shas) != 1:
        raise ValueError("capacity scaling matched provenance differs")
    sources = _object(row.get("source_evidence"), "capacity scaling sources")
    _identity(
        sources.get("capacity_confirmation_result"),
        "capacity confirmation result",
    )
    _git(sources.get("capacity_confirmation_git"), "capacity confirmation Git")
    arm_ids = _object(
        sources.get("capacity_confirmation_arm_validations"),
        "capacity confirmation arm validations",
    )
    if set(arm_ids) != set(ARM_NAMES):
        raise ValueError("capacity scaling confirmation arm set differs")
    for arm in ARM_NAMES:
        _identity(arm_ids[arm], f"{arm} confirmation validation")
    return {
        "decision_git": copy.deepcopy(expected_git),
        "capacity_scaling_preparation_allowed": passed,
        "next_stage": copy.deepcopy(next_stage),
        "output_root": output_root,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_DECISION_BOUNDARY
        ),
    }


__all__ = [
    "CAPACITY_SCALING_ARMS",
    "CAPACITY_SCALING_DECISION_BOUNDARY",
    "CAPACITY_SCALING_DECISION_ROLE",
    "CAPACITY_SCALING_DECISION_SCHEMA",
    "CAPACITY_SCALING_DIRNAME",
    "CAPACITY_SCALING_HOLD_ID",
    "CAPACITY_SCALING_METHODS",
    "CAPACITY_SCALING_RECOMMENDATION_ID",
    "CAPACITY_SCALING_TARGET_STEP",
    "build_capacity_scaling_decision",
    "validate_capacity_scaling_decision",
]
