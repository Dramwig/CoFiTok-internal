from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_full_readiness_decision import (
    CAPACITY_FULL_READINESS_DECISION_BOUNDARY,
    CAPACITY_FULL_ROOT_ID,
    EXPECTED_PARAMETER_COUNTS,
    validate_capacity_full_readiness_decision,
)


CAPACITY_FULL_READINESS_SCHEMA_VERSION = 1
CAPACITY_FULL_READINESS_ROLE = "stability_capacity_full_300k_training_readiness"
CAPACITY_FULL_READINESS_SOURCE_NAMES = {
    "readiness_decision",
    "target_cofitok_config",
    "target_dense_identity_config",
    "config_validation",
    "storage_capacity",
    "runtime_selection",
}
CAPACITY_FULL_READINESS_BOUNDARY = {
    "readiness_evidence_complete": True,
    "fresh_full_training_launch_allowed": True,
    "capacity_100k_checkpoint_resume_allowed": False,
    "fresh_full_training_required": True,
    "formal_generation_claim_allowed": False,
    "promotion_or_release_allowed": False,
    "separate_training_launch_receipt_required": True,
}
RUNTIME_CANDIDATES = [[1, 64], [2, 32], [4, 16], [8, 8], [16, 4]]
RUNTIME_BASELINE = [1, 64]
EXPECTED_EFFECTIVE_BATCH = 64
READINESS_SAMPLE_RESERVE = 16_384
FORMAL_POSTEVAL_SAMPLE_RESERVE = 100_256
FULL_COMPLETION_SAMPLE_RESERVE = (
    READINESS_SAMPLE_RESERVE + FORMAL_POSTEVAL_SAMPLE_RESERVE
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


def validate_capacity_full_config_contract(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    cofitok = report.get("cofitok")
    dense = report.get("dense")
    recipe = report.get("training_recipe")
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or not isinstance(cofitok, Mapping)
        or not isinstance(dense, Mapping)
        or not isinstance(recipe, Mapping)
        or cofitok.get("parameter_count") != EXPECTED_PARAMETER_COUNTS["cofitok"]
        or dense.get("parameter_count")
        != EXPECTED_PARAMETER_COUNTS["dense_identity"]
        or report.get("matched_backbone", {}).get("base_channels") != 256
        or recipe.get("stage") != "stability_full"
        or recipe.get("valid") is not True
    ):
        raise ValueError("capacity-full readiness config validation differs")
    return {
        "cofitok_parameter_count": EXPECTED_PARAMETER_COUNTS["cofitok"],
        "dense_parameter_count": EXPECTED_PARAMETER_COUNTS["dense_identity"],
        "relative_parameter_gap": float(report["relative_parameter_gap"]),
        "base_channels": 256,
        "recipe_stage": "stability_full",
        "recipe_schema": recipe.get("schema"),
    }


def validate_capacity_full_storage(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    expected_path: str,
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != "full_training"
        or report.get("status") != "pass"
        or report.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("capacity-full storage report differs")
    filesystem = report.get("filesystem")
    plan = report.get("plan")
    if not isinstance(filesystem, Mapping) or not isinstance(plan, Mapping):
        raise ValueError("capacity-full storage evidence is malformed")
    if PurePosixPath(str(filesystem.get("path", ""))) != PurePosixPath(
        expected_path
    ):
        raise ValueError("capacity-full storage path differs")
    minimums = {
        "checkpoint_count": 16,
        "sample_count": FULL_COMPLETION_SAMPLE_RESERVE,
        "estimated_sample_bytes_each": 256 * 1024,
        "additional_bytes": 16 * 1024**3,
        "safety_margin_bytes": 64 * 1024**3,
    }
    for key, minimum in minimums.items():
        if int(plan.get(key, -1)) < minimum:
            raise ValueError(f"capacity-full storage reserve weakened: {key}")
    reference_bytes = int(plan.get("reference_checkpoint_bytes_each", -1))
    multiplier = float(plan.get("checkpoint_size_multiplier", math.nan))
    checkpoint_bytes = int(plan.get("checkpoint_bytes_each", -1))
    if (
        reference_bytes < 1
        or not math.isfinite(multiplier)
        or multiplier != 1.0
        or checkpoint_bytes != reference_bytes
    ):
        raise ValueError("capacity-full actual-checkpoint storage scaling differs")
    checkpoint_reserve = int(plan["checkpoint_count"]) * checkpoint_bytes
    sample_reserve = int(plan["sample_count"]) * int(
        plan["estimated_sample_bytes_each"]
    )
    required = (
        checkpoint_reserve
        + sample_reserve
        + int(plan["additional_bytes"])
        + int(plan["safety_margin_bytes"])
    )
    if (
        int(plan.get("checkpoint_reserve_bytes", -1)) != checkpoint_reserve
        or int(plan.get("sample_reserve_bytes", -1)) != sample_reserve
        or int(plan.get("required_free_bytes", -1)) != required
    ):
        raise ValueError("capacity-full storage arithmetic differs")
    total = int(filesystem.get("total_bytes", -1))
    used = int(filesystem.get("used_bytes", -1))
    free = int(filesystem.get("free_bytes", -1))
    if total < 1 or used < 0 or free < required or used + free > total:
        raise ValueError("capacity-full storage headroom is insufficient")
    if int(report.get("headroom_bytes", -1)) != free - required:
        raise ValueError("capacity-full storage headroom arithmetic differs")
    return {
        "reference_checkpoint_bytes_each": reference_bytes,
        "checkpoint_size_multiplier": 1.0,
        "checkpoint_bytes_each": checkpoint_bytes,
        "checkpoint_count": int(plan["checkpoint_count"]),
        "sample_count": int(plan["sample_count"]),
        "required_free_bytes": required,
        "free_bytes": free,
        "headroom_bytes": free - required,
    }


def build_capacity_full_readiness(
    *,
    readiness_decision: Mapping[str, Any],
    readiness_decision_identity: Mapping[str, Any],
    target_config_identities: Mapping[str, Mapping[str, Any]],
    config_validation: Mapping[str, Any],
    config_validation_identity: Mapping[str, Any],
    storage_report: Mapping[str, Any],
    storage_report_identity: Mapping[str, Any],
    runtime_selection: Mapping[str, Any],
    runtime_selection_identity: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    readiness_git: Mapping[str, Any],
    expected_readiness_revision: str,
    expected_readiness_tree: str,
    expected_readiness_branch: str,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    expected_result_revision: str,
    expected_result_tree: str,
    expected_result_branch: str,
) -> dict[str, Any]:
    decision_identity = _identity(
        readiness_decision_identity,
        label="capacity-full readiness decision",
    )
    readiness_builder_git = _git(
        readiness_git,
        revision=expected_readiness_revision,
        tree=expected_readiness_tree,
        branch=expected_readiness_branch,
        label="capacity-full readiness builder",
    )
    decision_evidence = validate_capacity_full_readiness_decision(
        readiness_decision,
        expected_decision_revision=expected_decision_revision,
        expected_decision_tree=expected_decision_tree,
        expected_decision_branch=expected_decision_branch,
        expected_result_revision=expected_result_revision,
        expected_result_tree=expected_result_tree,
        expected_result_branch=expected_result_branch,
    )
    if (
        readiness_decision.get("authorization_boundary")
        != CAPACITY_FULL_READINESS_DECISION_BOUNDARY
        or decision_evidence["execution_authorization"].get(
            "readiness_gpu_runtime_benchmark_allowed"
        )
        is not True
        or decision_evidence["execution_authorization"].get(
            "training_launch_allowed"
        )
        is not False
    ):
        raise ValueError("capacity-full readiness decision authorization differs")
    methods = {"cofitok", "dense_identity"}
    if set(target_config_identities) != methods:
        raise ValueError("capacity-full readiness target config set differs")
    target_identities = {
        method: _identity(
            target_config_identities[method],
            label=f"capacity-full readiness {method} config",
        )
        for method in methods
    }
    decision_sources = readiness_decision.get("source_reports", {})
    if (
        not _same_content(
            target_identities["cofitok"],
            decision_sources.get("target_cofitok_config", {}),
        )
        or not _same_content(
            target_identities["dense_identity"],
            decision_sources.get("target_dense_identity_config", {}),
        )
    ):
        raise ValueError("capacity-full readiness configs differ from decision")
    config_identity = _identity(
        config_validation_identity,
        label="capacity-full config validation",
    )
    storage_identity = _identity(
        storage_report_identity,
        label="capacity-full storage report",
    )
    runtime_identity = _identity(
        runtime_selection_identity,
        label="capacity-full runtime selection",
    )
    config_contract = validate_capacity_full_config_contract(config_validation)
    plan = decision_evidence["readiness_plan"]
    storage = validate_capacity_full_storage(
        storage_report,
        expected_revision=expected_readiness_revision,
        expected_branch=expected_readiness_branch,
        expected_path=plan["storage_path"],
    )
    micro_batch_size = runtime_evidence.get("micro_batch_size")
    gradient_accumulation_steps = runtime_evidence.get(
        "gradient_accumulation_steps"
    )
    if (
        type(micro_batch_size) is not int
        or micro_batch_size < 1
        or type(gradient_accumulation_steps) is not int
        or gradient_accumulation_steps < 1
    ):
        raise ValueError("capacity-full runtime batch selection is malformed")
    expected_runtime = {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "baseline": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 64,
        },
        "candidate_count": len(RUNTIME_CANDIDATES),
        "runtime_environment_sha256": runtime_evidence.get(
            "runtime_environment_sha256"
        ),
        "dataset_identity_sha256": runtime_evidence.get(
            "dataset_identity_sha256"
        ),
        "estimated_speedup_over_baseline": runtime_evidence.get(
            "estimated_speedup_over_baseline"
        ),
    }
    if (
        runtime_evidence != expected_runtime
        or not isinstance(runtime_selection, Mapping)
        or runtime_selection.get("schema_version") != 3
        or runtime_selection.get("baseline") is None
        or micro_batch_size * gradient_accumulation_steps
        != EXPECTED_EFFECTIVE_BATCH
    ):
        raise ValueError("capacity-full runtime selection evidence differs")
    fresh = readiness_decision["fresh_training_contract"]
    if (
        fresh.get("training_state_absent_at_decision") is not True
        or fresh.get("capacity_100k_checkpoint_resume_allowed") is not False
        or fresh.get("target_start_step") != 0
    ):
        raise ValueError("capacity-full fresh-training boundary differs")
    sources = {
        "readiness_decision": decision_identity,
        "target_cofitok_config": target_identities["cofitok"],
        "target_dense_identity_config": target_identities["dense_identity"],
        "config_validation": config_identity,
        "storage_capacity": storage_identity,
        "runtime_selection": runtime_identity,
    }
    return {
        "schema_version": CAPACITY_FULL_READINESS_SCHEMA_VERSION,
        "status": "pass",
        "role": CAPACITY_FULL_READINESS_ROLE,
        "stage": "stability_full",
        "git": readiness_builder_git,
        "source_reports": sources,
        "decision_authorization": copy.deepcopy(
            decision_evidence["execution_authorization"]
        ),
        "config_contract": config_contract,
        "storage_capacity": storage,
        "runtime_selection": expected_runtime,
        "training_run_dirs": copy.deepcopy(fresh["training_run_dirs"]),
        "training_state_absent_at_build": True,
        "fresh_training_contract": {
            "source_capacity_checkpoints_are_evidence_only": True,
            "capacity_100k_checkpoint_resume_allowed": False,
            "target_start_step": 0,
            "target_steps": 300_000,
            "exact_resume_allowed_only_from_target_config_checkpoints": True,
        },
        "full_output_root": plan["full_output_root"],
        "benchmark_root": plan["benchmark_root"],
        "full_training_launch_allowed": True,
        "formal_generation_completion_claimed": False,
        "authorization_boundary": copy.deepcopy(CAPACITY_FULL_READINESS_BOUNDARY),
    }


def validate_capacity_full_readiness(
    report: Mapping[str, Any],
    *,
    expected_readiness_revision: str,
    expected_readiness_tree: str,
    expected_readiness_branch: str,
) -> dict[str, Any]:
    sources = report.get("source_reports")
    runtime = report.get("runtime_selection")
    fresh = report.get("fresh_training_contract")
    storage = report.get("storage_capacity")
    config = report.get("config_contract")
    decision_authorization = report.get("decision_authorization")
    if (
        report.get("schema_version") != CAPACITY_FULL_READINESS_SCHEMA_VERSION
        or report.get("status") != "pass"
        or report.get("role") != CAPACITY_FULL_READINESS_ROLE
        or report.get("stage") != "stability_full"
        or report.get("authorization_boundary") != CAPACITY_FULL_READINESS_BOUNDARY
        or report.get("full_training_launch_allowed") is not True
        or report.get("formal_generation_completion_claimed") is not False
        or report.get("training_state_absent_at_build") is not True
        or not isinstance(sources, Mapping)
        or set(sources) != CAPACITY_FULL_READINESS_SOURCE_NAMES
        or not isinstance(runtime, Mapping)
        or not isinstance(fresh, Mapping)
        or not isinstance(storage, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(decision_authorization, Mapping)
    ):
        raise ValueError("capacity-full readiness contract differs")
    git = _git(
        report.get("git", {}),
        revision=expected_readiness_revision,
        tree=expected_readiness_tree,
        branch=expected_readiness_branch,
        label="capacity-full readiness",
    )
    for name, identity in sources.items():
        _identity(identity, label=f"capacity-full readiness source {name}")
    micro_batch = runtime.get("micro_batch_size")
    accumulation = runtime.get("gradient_accumulation_steps")
    speedup = runtime.get("estimated_speedup_over_baseline")
    if (
        type(micro_batch) is not int
        or micro_batch < 1
        or type(accumulation) is not int
        or accumulation < 1
        or type(speedup) not in {int, float}
        or not math.isfinite(float(speedup))
        or float(speedup) <= 0.0
    ):
        raise ValueError("capacity-full readiness runtime evidence is malformed")
    full_root = PurePosixPath(str(report.get("full_output_root", "")))
    benchmark_root = PurePosixPath(str(report.get("benchmark_root", "")))
    training_runs = report.get("training_run_dirs")
    if (
        runtime.get("effective_batch_size") != EXPECTED_EFFECTIVE_BATCH
        or micro_batch * accumulation != EXPECTED_EFFECTIVE_BATCH
        or runtime.get("baseline")
        != {"micro_batch_size": 1, "gradient_accumulation_steps": 64}
        or runtime.get("candidate_count") != len(RUNTIME_CANDIDATES)
        or not _hex(runtime.get("runtime_environment_sha256"), length=64)
        or not _hex(runtime.get("dataset_identity_sha256"), length=64)
        or full_root.name != CAPACITY_FULL_ROOT_ID
        or benchmark_root != full_root / "reports/runtime_benchmark"
        or training_runs
        != {
            "cofitok": str(full_root / "cofitok"),
            "dense_identity": str(full_root / "dense_identity"),
        }
        or decision_authorization
        != {
            "readiness_gpu_runtime_benchmark_allowed": True,
            "readiness_artifact_build_allowed": True,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        }
        or config.get("cofitok_parameter_count")
        != EXPECTED_PARAMETER_COUNTS["cofitok"]
        or config.get("dense_parameter_count")
        != EXPECTED_PARAMETER_COUNTS["dense_identity"]
        or config.get("base_channels") != 256
        or config.get("recipe_stage") != "stability_full"
        or storage.get("checkpoint_size_multiplier") != 1.0
        or int(storage.get("checkpoint_count", -1)) < 16
        or int(storage.get("sample_count", -1))
        < FULL_COMPLETION_SAMPLE_RESERVE
        or int(storage.get("free_bytes", -1))
        < int(storage.get("required_free_bytes", -1))
        or fresh
        != {
            "source_capacity_checkpoints_are_evidence_only": True,
            "capacity_100k_checkpoint_resume_allowed": False,
            "target_start_step": 0,
            "target_steps": 300_000,
            "exact_resume_allowed_only_from_target_config_checkpoints": True,
        }
    ):
        raise ValueError("capacity-full readiness runtime/fresh contract differs")
    return {
        "status": "pass",
        "git": git,
        "full_output_root": report["full_output_root"],
        "training_run_dirs": copy.deepcopy(report["training_run_dirs"]),
        "runtime_selection": copy.deepcopy(dict(runtime)),
        "storage_capacity": copy.deepcopy(report["storage_capacity"]),
        "full_training_launch_allowed": True,
        "authorization_boundary": copy.deepcopy(CAPACITY_FULL_READINESS_BOUNDARY),
    }
