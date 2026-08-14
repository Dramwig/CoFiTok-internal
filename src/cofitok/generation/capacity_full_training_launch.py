from __future__ import annotations

import copy
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_full_readiness import (
    CAPACITY_FULL_READINESS_BOUNDARY,
    EXPECTED_EFFECTIVE_BATCH,
    validate_capacity_full_readiness,
    validate_capacity_full_storage,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)


CAPACITY_FULL_TRAINING_LAUNCH_SCHEMA_VERSION = 1
CAPACITY_FULL_TRAINING_LAUNCH_ROLE = (
    "stability_capacity_full_300k_training_launch_receipt"
)
CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE = "capacity_full_experimental"
CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION = (
    "authorize_fresh_matched_300k_training"
)
CAPACITY_FULL_TRAINING_MILESTONES = [50_000, 100_000, 200_000, 300_000]

CAPACITY_FULL_TRAINING_LAUNCH_BOUNDARY = {
    "fresh_matched_300k_training_launch_allowed": True,
    "capacity_100k_checkpoint_resume_allowed": False,
    "fresh_start_step_required": 0,
    "target_step_required": 300_000,
    "exact_resume_allowed_only_from_target_config_checkpoints": True,
    "readiness_artifact_is_training_precondition": True,
    "training_authorization_is_quality_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "frozen_promotion_gate_replaced": False,
    "release_authorization_allowed": False,
    "separate_final_quality_gate_required": True,
    "unrelated_gpu_process_modification_allowed": False,
}

READINESS_SUPERVISOR_BOUNDARY = {
    "readiness_gpu_runtime_benchmark_allowed": True,
    "readiness_artifact_build_allowed": True,
    "training_launch_allowed_by_supervisor": False,
    "full_300k_launch_allowed_by_supervisor": False,
    "process_signaling_allowed": False,
    "unrelated_gpu_process_modification_allowed": False,
    "capacity_100k_checkpoint_resume_allowed": False,
    "promotion_or_release_allowed": False,
}

READINESS_DEPLOYMENT_BOUNDARY = {
    "capacity_100k_checkpoint_resume_allowed": False,
    "full_300k_launch_allowed_by_supervisor": False,
    "process_signaling_allowed": False,
    "promotion_or_release_allowed": False,
    "readiness_artifact_build_allowed_after_exact_decision": True,
    "readiness_gpu_runtime_benchmark_allowed_after_exact_decision": True,
    "separate_training_launch_receipt_required_after_readiness": True,
    "training_launch_allowed_by_supervisor": False,
    "unrelated_gpu_process_modification_allowed": False,
}

# The readiness benchmark was qualified at the parent revision.  Only these
# authorization, checkpoint-metadata, orchestration, tests, and record paths may
# differ in the training revision that consumes that benchmark.
ALLOWED_READINESS_TO_TRAINING_TRANSITION_PATHS = frozenset(
    {
        "artifacts/runbooks/generation_capacity_full_300k_execute.sh",
        "artifacts/runbooks/generation_capacity_full_300k_training_supervisor.sh",
        "docs/records/2026-08-14_generation_capacity_full_300k_readiness_supervisor.md",
        "scripts/build_generation_capacity_full_300k_training_launch_receipt.py",
        "scripts/run_generation_capacity_full_300k_training_supervisor.py",
        "scripts/verify_generation_capacity_full_300k_training_launch_receipt.py",
        "src/cofitok/generation/capacity_full_training_launch.py",
        "src/cofitok/training/authorization.py",
        "src/cofitok/training/checkpointing.py",
        "tests/test_generation_capacity_full_training_execution.py",
        "tests/test_generation_capacity_full_training_launch.py",
        "tests/test_generation_runbook_entrypoints.py",
    }
)

CAPACITY_FULL_TRAINING_SOURCE_NAMES = frozenset(
    {
        "capacity_full_readiness",
        "readiness_supervisor_status",
        "readiness_supervisor_deployment_receipt",
        "readiness_deployment_validation_clarification",
        "standing_authorization",
        "target_cofitok_config",
        "target_dense_identity_config",
        "launch_storage_capacity",
    }
)


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


def _git(
    value: Mapping[str, Any],
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if not _hex(revision, length=40) or not _hex(tree, length=40):
        raise ValueError(f"{label} expected Git identity is malformed")
    if dict(value) != expected:
        raise ValueError(f"{label} Git identity differs")
    return expected


def _same_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def _validate_transition(
    transition: Mapping[str, Any],
    *,
    readiness_revision: str,
    training_revision: str,
) -> dict[str, Any]:
    changed = transition.get("changed_paths")
    if not isinstance(changed, list) or any(not isinstance(path, str) for path in changed):
        raise ValueError("capacity-full readiness-to-training path set is malformed")
    normalized = sorted(set(changed))
    if (
        normalized != changed
        or set(normalized) != ALLOWED_READINESS_TO_TRAINING_TRANSITION_PATHS
        or transition.get("source_revision") != readiness_revision
        or transition.get("target_revision") != training_revision
        or transition.get("source_is_ancestor") is not True
        or transition.get("target_configs_byte_identical") is not True
        or transition.get("model_data_diffusion_compute_paths_changed") != []
        or transition.get("runtime_requalification_required") is not False
        or transition.get("authorization_metadata_only_runtime_change") is not True
    ):
        raise ValueError("capacity-full readiness-to-training transition differs")
    return {
        "source_revision": readiness_revision,
        "target_revision": training_revision,
        "source_is_ancestor": True,
        "changed_paths": normalized,
        "target_configs_byte_identical": True,
        "model_data_diffusion_compute_paths_changed": [],
        "runtime_requalification_required": False,
        "authorization_metadata_only_runtime_change": True,
    }


def _validate_readiness_supervisor_status(
    report: Mapping[str, Any],
    *,
    readiness_identity: Mapping[str, Any],
    readiness_git: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    full_output_root: str,
) -> None:
    detail = report.get("detail")
    readiness = report.get("readiness")
    expected = report.get("expected")
    if (
        report.get("schema_version") != 1
        or report.get("role") != "capacity_full_300k_readiness_supervisor"
        or report.get("status") != "pass"
        or detail
        not in {
            "capacity_full_readiness_completed_without_training_launch",
            "existing_capacity_full_readiness_replayed",
        }
        or report.get("authorization_boundary") != READINESS_SUPERVISOR_BOUNDARY
        or report.get("readiness_execution_only") is not True
        or report.get("training_launch_performed") is not False
        or report.get("full_300k_launch_performed") is not False
        or not isinstance(readiness, Mapping)
        or readiness.get("identity") != readiness_identity
        or readiness.get("full_training_launch_allowed_by_artifact") is not True
        or readiness.get("training_launch_performed_by_supervisor") is not False
        or readiness.get("full_300k_launch_performed_by_supervisor") is not False
        or not isinstance(expected, Mapping)
        or expected.get("readiness_git") != readiness_git
        or expected.get("standing_authorization") != standing_identity
        or expected.get("full_output_root") != full_output_root
        or expected.get("capacity_100k_checkpoint_resume_allowed") is not False
        or expected.get("training_launch_allowed_by_supervisor") is not False
        or expected.get("full_300k_launch_allowed_by_supervisor") is not False
    ):
        raise ValueError("capacity-full readiness supervisor status differs")


def _validate_readiness_deployment(
    report: Mapping[str, Any],
    *,
    readiness_git: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    full_output_root: str,
) -> None:
    checkout = report.get("checkout")
    target = report.get("target_state_at_deployment")
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "capacity_full_300k_readiness_supervisor_deployment"
        or report.get("status") != "active"
        or report.get("git") != readiness_git
        or not isinstance(checkout, Mapping)
        or checkout.get("git") != readiness_git
        or report.get("standing_authorization") != standing_identity
        or report.get("full_output_root") != full_output_root
        or report.get("authorization_boundary") != READINESS_DEPLOYMENT_BOUNDARY
        or not isinstance(target, Mapping)
        or target.get("capacity_100k_checkpoint_resume_allowed") is not False
        or target.get("cofitok_training_state_absent") is not True
        or target.get("dense_identity_training_state_absent") is not True
        or target.get("full_output_root_absent") is not True
    ):
        raise ValueError("capacity-full readiness deployment receipt differs")


def _validate_deployment_clarification(
    report: Mapping[str, Any],
    *,
    deployment_identity: Mapping[str, Any],
    readiness_git: Mapping[str, Any],
) -> None:
    effect = report.get("authorization_effect")
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "capacity_full_300k_readiness_supervisor_deployment_validation_clarification"
        or report.get("status") != "clarified"
        or report.get("deployment_receipt") != deployment_identity
        or report.get("deployed_git") != readiness_git
        or not isinstance(effect, Mapping)
        or effect.get("changes_authorization") is not False
        or effect.get("training_launch_allowed_by_supervisor") is not False
        or effect.get("full_300k_launch_allowed_by_supervisor") is not False
    ):
        raise ValueError("capacity-full readiness deployment clarification differs")


def capacity_full_training_launch_identity_sha256(
    report: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_capacity_full_training_launch(
    *,
    readiness: Mapping[str, Any],
    readiness_identity: Mapping[str, Any],
    readiness_supervisor_status: Mapping[str, Any],
    readiness_supervisor_status_identity: Mapping[str, Any],
    readiness_deployment_receipt: Mapping[str, Any],
    readiness_deployment_receipt_identity: Mapping[str, Any],
    readiness_deployment_clarification: Mapping[str, Any],
    readiness_deployment_clarification_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    target_config_identities: Mapping[str, Mapping[str, Any]],
    launch_storage: Mapping[str, Any],
    launch_storage_identity: Mapping[str, Any],
    readiness_to_training_transition: Mapping[str, Any],
    execution_git: Mapping[str, Any],
    training_git: Mapping[str, Any],
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_readiness_revision: str,
    expected_readiness_tree: str,
    expected_readiness_branch: str,
) -> dict[str, Any]:
    readiness_id = _identity(readiness_identity, label="capacity-full readiness")
    status_id = _identity(
        readiness_supervisor_status_identity,
        label="capacity-full readiness supervisor status",
    )
    deployment_id = _identity(
        readiness_deployment_receipt_identity,
        label="capacity-full readiness deployment receipt",
    )
    clarification_id = _identity(
        readiness_deployment_clarification_identity,
        label="capacity-full readiness deployment clarification",
    )
    standing_id = _identity(
        standing_authorization_identity,
        label="standing experiment authorization",
    )
    storage_id = _identity(
        launch_storage_identity,
        label="capacity-full launch storage capacity",
    )
    execution = _git(
        execution_git,
        revision=expected_execution_revision,
        tree=expected_execution_tree,
        branch=expected_execution_branch,
        label="capacity-full execution",
    )
    training = _git(
        training_git,
        revision=expected_training_revision,
        tree=expected_training_tree,
        branch=expected_training_branch,
        label="capacity-full training",
    )
    readiness_git = {
        "revision": expected_readiness_revision,
        "tree": expected_readiness_tree,
        "branch": expected_readiness_branch,
        "tracked_dirty": False,
    }
    readiness_evidence = validate_capacity_full_readiness(
        readiness,
        expected_readiness_revision=expected_readiness_revision,
        expected_readiness_tree=expected_readiness_tree,
        expected_readiness_branch=expected_readiness_branch,
    )
    if readiness.get("authorization_boundary") != CAPACITY_FULL_READINESS_BOUNDARY:
        raise ValueError("capacity-full readiness launch boundary differs")
    standing = validate_standing_experiment_authorization(standing_authorization)
    if (
        standing.get("preserved_safety_boundaries")
        != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment authorization safety boundary differs")
    full_root = readiness_evidence["full_output_root"]
    _validate_readiness_supervisor_status(
        readiness_supervisor_status,
        readiness_identity=readiness_id,
        readiness_git=readiness_git,
        standing_identity=standing_id,
        full_output_root=full_root,
    )
    _validate_readiness_deployment(
        readiness_deployment_receipt,
        readiness_git=readiness_git,
        standing_identity=standing_id,
        full_output_root=full_root,
    )
    _validate_deployment_clarification(
        readiness_deployment_clarification,
        deployment_identity=deployment_id,
        readiness_git=readiness_git,
    )
    if set(target_config_identities) != {"cofitok", "dense_identity"}:
        raise ValueError("capacity-full target config set differs")
    configs = {
        name: _identity(identity, label=f"capacity-full {name} config")
        for name, identity in target_config_identities.items()
    }
    readiness_sources = readiness.get("source_reports", {})
    if (
        not _same_content(
            configs["cofitok"], readiness_sources.get("target_cofitok_config", {})
        )
        or not _same_content(
            configs["dense_identity"],
            readiness_sources.get("target_dense_identity_config", {}),
        )
    ):
        raise ValueError("capacity-full launch configs differ from readiness")
    transition = _validate_transition(
        readiness_to_training_transition,
        readiness_revision=expected_readiness_revision,
        training_revision=expected_training_revision,
    )
    root = PurePosixPath(full_root)
    storage = validate_capacity_full_storage(
        launch_storage,
        expected_revision=expected_training_revision,
        expected_branch=expected_training_branch,
        expected_path=str(root.parent),
    )
    run_dirs = readiness_evidence["training_run_dirs"]
    runtime = readiness_evidence["runtime_selection"]
    if (
        run_dirs
        != {
            "cofitok": str(root / "cofitok"),
            "dense_identity": str(root / "dense_identity"),
        }
        or runtime.get("effective_batch_size") != EXPECTED_EFFECTIVE_BATCH
        or int(runtime.get("micro_batch_size", -1))
        * int(runtime.get("gradient_accumulation_steps", -1))
        != EXPECTED_EFFECTIVE_BATCH
    ):
        raise ValueError("capacity-full training target/runtime differs")
    sources = {
        "capacity_full_readiness": readiness_id,
        "readiness_supervisor_status": status_id,
        "readiness_supervisor_deployment_receipt": deployment_id,
        "readiness_deployment_validation_clarification": clarification_id,
        "standing_authorization": standing_id,
        "target_cofitok_config": configs["cofitok"],
        "target_dense_identity_config": configs["dense_identity"],
        "launch_storage_capacity": storage_id,
    }
    return {
        "schema_version": CAPACITY_FULL_TRAINING_LAUNCH_SCHEMA_VERSION,
        "status": "authorized",
        "role": CAPACITY_FULL_TRAINING_LAUNCH_ROLE,
        "stage": CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE,
        "decision": CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION,
        "git": {
            "execution": execution,
            "training": training,
            "readiness": readiness_git,
        },
        "source_reports": sources,
        "standing_authorization": {
            "source": standing_id,
            "validated_record": standing,
        },
        "readiness_to_training_transition": transition,
        "runtime_selection": copy.deepcopy(runtime),
        "storage_capacity": storage,
        "training_plan": {
            "dataset": "imagenet_256",
            "recipe_stage": "stability_full",
            "methods": ["cofitok", "dense_identity"],
            "configured_steps": 300_000,
            "start_step": 0,
            "milestone_steps": list(CAPACITY_FULL_TRAINING_MILESTONES),
            "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
            "training_run_dirs": copy.deepcopy(run_dirs),
            "target_configs": copy.deepcopy(configs),
            "fresh_initialization_required": True,
            "capacity_100k_checkpoint_resume_allowed": False,
            "exact_resume_allowed_only_from_target_config_checkpoints": True,
        },
        "full_output_root": full_root,
        "training_state_absent_at_authorization": True,
        "formal_generation_completion_claimed": False,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_FULL_TRAINING_LAUNCH_BOUNDARY
        ),
    }


def validate_capacity_full_training_launch(
    report: Mapping[str, Any],
    *,
    expected_execution_revision: str,
    expected_execution_tree: str,
    expected_execution_branch: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_readiness_revision: str,
    expected_readiness_tree: str,
    expected_readiness_branch: str,
) -> dict[str, Any]:
    git = report.get("git")
    sources = report.get("source_reports")
    plan = report.get("training_plan")
    standing = report.get("standing_authorization")
    runtime = report.get("runtime_selection")
    storage = report.get("storage_capacity")
    if (
        report.get("schema_version") != CAPACITY_FULL_TRAINING_LAUNCH_SCHEMA_VERSION
        or report.get("status") != "authorized"
        or report.get("role") != CAPACITY_FULL_TRAINING_LAUNCH_ROLE
        or report.get("stage") != CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE
        or report.get("decision") != CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION
        or report.get("authorization_boundary")
        != CAPACITY_FULL_TRAINING_LAUNCH_BOUNDARY
        or report.get("training_state_absent_at_authorization") is not True
        or report.get("formal_generation_completion_claimed") is not False
        or not isinstance(git, Mapping)
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_FULL_TRAINING_SOURCE_NAMES
        or not isinstance(plan, Mapping)
        or not isinstance(standing, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(storage, Mapping)
    ):
        raise ValueError("capacity-full training launch contract differs")
    execution = _git(
        git.get("execution", {}),
        revision=expected_execution_revision,
        tree=expected_execution_tree,
        branch=expected_execution_branch,
        label="capacity-full execution",
    )
    training = _git(
        git.get("training", {}),
        revision=expected_training_revision,
        tree=expected_training_tree,
        branch=expected_training_branch,
        label="capacity-full training",
    )
    readiness = _git(
        git.get("readiness", {}),
        revision=expected_readiness_revision,
        tree=expected_readiness_tree,
        branch=expected_readiness_branch,
        label="capacity-full readiness",
    )
    for name, identity in sources.items():
        _identity(identity, label=f"capacity-full training source {name}")
    validated_standing = validate_standing_experiment_authorization(
        standing.get("validated_record", {})
    )
    if (
        standing.get("source") != sources["standing_authorization"]
        or validated_standing.get("preserved_safety_boundaries")
        != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("capacity-full launch standing authorization differs")
    transition = _validate_transition(
        report.get("readiness_to_training_transition", {}),
        readiness_revision=expected_readiness_revision,
        training_revision=expected_training_revision,
    )
    full_root = PurePosixPath(str(report.get("full_output_root", "")))
    run_dirs = plan.get("training_run_dirs")
    configs = plan.get("target_configs")
    if (
        full_root.name != "stability_capacity_full_300k_v1"
        or plan.get("dataset") != "imagenet_256"
        or plan.get("recipe_stage") != "stability_full"
        or plan.get("methods") != ["cofitok", "dense_identity"]
        or plan.get("configured_steps") != 300_000
        or plan.get("start_step") != 0
        or plan.get("milestone_steps") != CAPACITY_FULL_TRAINING_MILESTONES
        or plan.get("effective_batch_size") != EXPECTED_EFFECTIVE_BATCH
        or plan.get("fresh_initialization_required") is not True
        or plan.get("capacity_100k_checkpoint_resume_allowed") is not False
        or plan.get("exact_resume_allowed_only_from_target_config_checkpoints")
        is not True
        or run_dirs
        != {
            "cofitok": str(full_root / "cofitok"),
            "dense_identity": str(full_root / "dense_identity"),
        }
        or not isinstance(configs, Mapping)
        or set(configs) != {"cofitok", "dense_identity"}
        or configs.get("cofitok") != sources["target_cofitok_config"]
        or configs.get("dense_identity")
        != sources["target_dense_identity_config"]
        or runtime.get("effective_batch_size") != EXPECTED_EFFECTIVE_BATCH
        or int(runtime.get("micro_batch_size", -1))
        * int(runtime.get("gradient_accumulation_steps", -1))
        != EXPECTED_EFFECTIVE_BATCH
        or int(storage.get("checkpoint_count", -1)) < 16
        or int(storage.get("sample_count", -1)) < 116_640
        or float(storage.get("checkpoint_size_multiplier", -1.0)) != 1.0
        or int(storage.get("free_bytes", -1))
        < int(storage.get("required_free_bytes", -1))
    ):
        raise ValueError("capacity-full training plan/runtime/storage differs")
    return {
        "status": "authorized",
        "stage": CAPACITY_FULL_TRAINING_AUTHORIZATION_STAGE,
        "decision": CAPACITY_FULL_TRAINING_AUTHORIZATION_DECISION,
        "git": {
            "execution": execution,
            "training": training,
            "readiness": readiness,
        },
        "full_output_root": report["full_output_root"],
        "training_run_dirs": copy.deepcopy(run_dirs),
        "runtime_selection": copy.deepcopy(dict(runtime)),
        "source_reports": copy.deepcopy(dict(sources)),
        "readiness_to_training_transition": transition,
        "authorization_boundary": copy.deepcopy(
            CAPACITY_FULL_TRAINING_LAUNCH_BOUNDARY
        ),
    }


def validate_capacity_full_training_launch_self_bound(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    git = report.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("capacity-full training launch Git binding is missing")
    execution = git.get("execution")
    training = git.get("training")
    readiness = git.get("readiness")
    if not all(isinstance(value, Mapping) for value in (execution, training, readiness)):
        raise ValueError("capacity-full training launch Git binding is malformed")
    return validate_capacity_full_training_launch(
        report,
        expected_execution_revision=str(execution.get("revision", "")),
        expected_execution_tree=str(execution.get("tree", "")),
        expected_execution_branch=str(execution.get("branch", "")),
        expected_training_revision=str(training.get("revision", "")),
        expected_training_tree=str(training.get("tree", "")),
        expected_training_branch=str(training.get("branch", "")),
        expected_readiness_revision=str(readiness.get("revision", "")),
        expected_readiness_tree=str(readiness.get("tree", "")),
        expected_readiness_branch=str(readiness.get("branch", "")),
    )


def capacity_full_training_authorization_thresholds(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = validate_capacity_full_training_launch_self_bound(report)
    runtime = evidence["runtime_selection"]
    return {
        "target_start_step": 0,
        "target_steps": 300_000,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "micro_batch_size": int(runtime["micro_batch_size"]),
        "gradient_accumulation_steps": int(
            runtime["gradient_accumulation_steps"]
        ),
        "capacity_100k_checkpoint_resume_allowed": False,
        "formal_generation_claim_allowed": False,
        "release_authorization_allowed": False,
    }
