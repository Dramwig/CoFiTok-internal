"""A bounded, source-bound gate for the exposure/capacity follow-up.

The 2026-08-31 preparation intentionally leaves the next arm undecided.  This
module turns one arm into a reproducible *candidate* gate, but it never grants
execution authority.  A future, separately authorized runbook may consume the
contract after creating its own stage authorization.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from cofitok.generation.exposure_capacity import validate_preparation


EXECUTION_GATE_SCHEMA = "cofitok_generation_exposure_capacity_execution_gate_v1"
EXECUTION_GATE_ROLE = "source_bound_bounded_exposure_capacity_execution_gate"
SOURCE_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
SOURCE_TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
SOURCE_BRANCH = "scale/generation-stability-quality-bridge-100k"
SOURCE_STEP = 100_000
SOURCE_IMAGES_SEEN_PER_METHOD = 6_400_000
EXPOSURE_TARGET_STEP = 110_000
CAPACITY_QUALIFICATION_STEPS = 10_000
EFFECTIVE_BATCH_SIZE = 64
MIN_FREE_BYTES = 120 * 1024**3
MAX_GPU_MEMORY_USED_MIB = 16
MAX_GPU_UTILIZATION_PERCENT = 5
ARM_IDS = ("exposure_continuation", "capacity_qualification")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


# This artifact is a contract for a later authorization step, not that step.
GATE_AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "execution_gate_authorized": False,
    "remote_mutation_allowed": False,
    "gpu_execution_authorized": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "terminal_hold_replacement_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _hex(value: Any, *, pattern: re.Pattern[str], name: str) -> str:
    text = str(value)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{name} is not a lowercase hexadecimal identity")
    return text


def _identity(value: Any, *, name: str) -> dict[str, Any]:
    row = _object(value, name)
    path = row.get("path")
    size = row.get("bytes")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} path is missing")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{name} byte count is invalid")
    digest = _hex(row.get("sha256"), pattern=SHA256_PATTERN, name=f"{name} SHA256")
    return {"path": path, "bytes": size, "sha256": digest}


def _portable_path(value: str) -> PurePath:
    """Interpret serialized POSIX paths even when the validator runs on Windows."""
    return PurePosixPath(value) if value.startswith("/") else PureWindowsPath(value)


def _checkpoint_binding(value: Any, *, required: bool) -> dict[str, Any] | None:
    """Validate the immutable identity triplet needed for an exact resume.

    The gate builder supplies descriptors after physically verifying the files;
    this function keeps the serialized contract strict for later consumers.
    """
    if value is None:
        if required:
            raise ValueError("exposure continuation requires a source checkpoint binding")
        return None
    row = _object(value, "source checkpoint binding")
    if set(row) != {"run_dir", "step", "checkpoint", "integrity_manifest", "latest"}:
        raise ValueError("source checkpoint binding fields differ")
    run_dir = row.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir or not _portable_path(run_dir).is_absolute():
        raise ValueError("source checkpoint run directory is invalid")
    step = row.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step != SOURCE_STEP:
        raise ValueError("source checkpoint step must be the completed 100K step")
    checkpoint = _identity(row.get("checkpoint"), name="source checkpoint")
    integrity_manifest = _identity(
        row.get("integrity_manifest"), name="source checkpoint integrity manifest"
    )
    latest = _identity(row.get("latest"), name="source latest pointer")
    checkpoint_path = _portable_path(checkpoint["path"])
    integrity_path = _portable_path(integrity_manifest["path"])
    latest_path = _portable_path(latest["path"])
    run_path = _portable_path(run_dir)
    if checkpoint_path.parent != run_path:
        raise ValueError("source checkpoint is outside its declared run directory")
    if integrity_path.parent != run_path:
        raise ValueError("source checkpoint integrity manifest is outside its run directory")
    if latest_path.parent != run_path:
        raise ValueError("source latest pointer is outside its declared run directory")
    if integrity_path.name != f"{checkpoint_path.name}.integrity.json":
        raise ValueError("source checkpoint integrity manifest names another checkpoint")
    if latest_path.name != "latest.json":
        raise ValueError("source latest pointer must be latest.json")
    return {
        "run_dir": run_dir,
        "step": SOURCE_STEP,
        "checkpoint": checkpoint,
        "integrity_manifest": integrity_manifest,
        "latest": latest,
    }


def _validate_exposure_source_checkpoint(
    binding: Mapping[str, Any], preparation: Mapping[str, Any]
) -> None:
    """Ensure an exposure resume names the exact CoFiTok payload in preparation."""
    evidence = _object(preparation.get("evidence"), "preparation evidence")
    training = _object(evidence.get("training"), "preparation training evidence")
    summaries = _object(
        training.get("source_checkpoints"), "preparation source checkpoint summaries"
    )
    expected = _object(summaries.get("cofitok"), "preparation CoFiTok source checkpoint")
    if binding.get("run_dir") != expected.get("run_dir"):
        raise ValueError("exposure source checkpoint run directory differs from preparation")
    if binding.get("step") != expected.get("step"):
        raise ValueError("exposure source checkpoint step differs from preparation")
    checkpoint = _object(binding.get("checkpoint"), "exposure source checkpoint")
    expected_checkpoint = _object(expected.get("checkpoint"), "expected CoFiTok source checkpoint")
    for field in ("bytes", "sha256"):
        if checkpoint.get(field) != expected_checkpoint.get(field):
            raise ValueError(f"exposure source checkpoint {field} differs from preparation")
    checkpoint_name = _portable_path(str(checkpoint.get("path", ""))).name
    if checkpoint_name != expected_checkpoint.get("name"):
        raise ValueError("exposure source checkpoint filename differs from preparation")
    integrity = _object(binding.get("integrity_manifest"), "exposure source checkpoint sidecar")
    if _portable_path(str(integrity.get("path", ""))).name != expected.get(
        "integrity_manifest_name"
    ):
        raise ValueError("exposure source checkpoint sidecar differs from preparation")


def _git(value: Any, *, name: str, require_tree: bool = False) -> dict[str, Any]:
    row = _object(value, name)
    revision = _hex(
        row.get("revision"), pattern=GIT_REVISION_PATTERN, name=f"{name} revision"
    )
    branch = row.get("branch")
    if not isinstance(branch, str) or not branch:
        raise ValueError(f"{name} branch is missing")
    if row.get("tracked_dirty") is not False:
        raise ValueError(f"{name} is dirty")
    result = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if require_tree or "tree" in row:
        result["tree"] = _hex(
            row.get("tree"), pattern=GIT_REVISION_PATTERN, name=f"{name} tree"
        )
    return result


def _source_checkout(value: Any) -> dict[str, Any]:
    source = _git(value, name="source checkout", require_tree=True)
    expected = {
        "revision": SOURCE_REVISION,
        "tree": SOURCE_TREE,
        "branch": SOURCE_BRANCH,
        "tracked_dirty": False,
    }
    if source != expected:
        raise ValueError("source checkout is not the locked 100K bridge checkout")
    return expected


def _validate_live_prelaunch(
    *,
    runtime_environment_sha256: Any,
    dataset_identity_sha256: Any,
    gpu_inventory: Any,
    gpu_compute_processes: Any,
    conflicting_processes: Any,
    output_root_absent: Any,
    execution_lock_free: Any,
    free_bytes: Any,
) -> dict[str, Any]:
    runtime_sha = _hex(
        runtime_environment_sha256,
        pattern=SHA256_PATTERN,
        name="runtime environment SHA256",
    )
    dataset_sha = _hex(
        dataset_identity_sha256,
        pattern=SHA256_PATTERN,
        name="dataset identity SHA256",
    )
    if not isinstance(gpu_inventory, list) or len(gpu_inventory) != 1:
        raise ValueError("exactly one target GPU is required")
    normalized_gpu: list[dict[str, Any]] = []
    for index, value in enumerate(gpu_inventory):
        row = _object(value, f"GPU inventory row {index}")
        try:
            used = int(row["memory_used_mib"])
            utilization = int(row["utilization_percent"])
            total = int(row["memory_total_mib"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("GPU inventory fields are malformed") from exc
        if (
            used < 0
            or total < 1
            or utilization < 0
            or used > MAX_GPU_MEMORY_USED_MIB
            or utilization > MAX_GPU_UTILIZATION_PERCENT
        ):
            raise ValueError("target GPU is not idle")
        normalized = copy.deepcopy(row)
        normalized["memory_used_mib"] = used
        normalized["memory_total_mib"] = total
        normalized["utilization_percent"] = utilization
        normalized_gpu.append(normalized)
    if gpu_compute_processes != []:
        raise ValueError("GPU compute processes are present")
    if conflicting_processes != []:
        raise ValueError("conflicting project processes are present")
    if output_root_absent is not True:
        raise ValueError("candidate output root must be absent")
    if execution_lock_free is not True:
        raise ValueError("candidate execution lock is held")
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool):
        raise ValueError("free storage byte count is invalid")
    if free_bytes < MIN_FREE_BYTES:
        raise ValueError("insufficient free storage for bounded qualification")
    return {
        "runtime_environment_sha256": runtime_sha,
        "dataset_identity_sha256": dataset_sha,
        "gpu_inventory": normalized_gpu,
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": int(free_bytes),
        "minimum_free_bytes": MIN_FREE_BYTES,
    }


def _arm_contract(preparation: Mapping[str, Any], arm_id: str) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown exposure/capacity arm: {arm_id}")
    arms = _object(preparation.get("candidate_arms"), "preparation candidate arms")
    arm = _object(arms.get(arm_id), f"preparation {arm_id} arm")
    output_root = arm.get("output_root")
    if not isinstance(output_root, str) or not output_root.startswith(
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ):
        raise ValueError(f"{arm_id} output root is not project scoped")
    common = {
        "arm_id": arm_id,
        "source_stage": "stability_quality_bridge",
        "controlled_change": arm.get("controlled_change"),
        "methods": ["cofitok", "dense_identity"],
        "dataset": "imagenet_256",
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "source_step": SOURCE_STEP,
        "source_images_seen_per_method": SOURCE_IMAGES_SEEN_PER_METHOD,
        "matched_pair_required": True,
        "objective_change_allowed": False,
        "conditioning_change_allowed": False,
        "formal_quality_claim_allowed": False,
        "evaluation": {
            "sampler": "ddim",
            "sample_steps": 100,
            "weights": "ema",
            "samples_per_method": 10_000,
            "class_fidelity_required": True,
            "mechanism_diagnostics_required": True,
            "shared_random_stream_required": True,
        },
        "output_root": output_root,
    }
    if arm_id == "exposure_continuation":
        if arm.get("initialization") != "exact_100k_checkpoint_resume_only":
            raise ValueError("exposure arm initialization contract differs")
        if arm.get("model_layout_change_allowed") is not False:
            raise ValueError("exposure arm permits a model-layout change")
        common.update(
            {
                "initialization": "exact_100k_checkpoint_resume_only",
                "target_step": EXPOSURE_TARGET_STEP,
                "additional_steps": EXPOSURE_TARGET_STEP - SOURCE_STEP,
                "resume_checkpoint_step": SOURCE_STEP,
                "resume_checkpoint_required": True,
                "source_checkpoint_binding_required": True,
                "new_output_root_required": True,
                "automatic_300k_escalation_allowed": False,
            }
        )
    else:
        if arm.get("initialization") != "fresh_matched_initialization_required":
            raise ValueError("capacity arm initialization contract differs")
        if arm.get("candidate_base_channels") != 256:
            raise ValueError("capacity arm must use the prepared 256-channel candidate")
        common.update(
            {
                "initialization": "fresh_matched_initialization_required",
                "source_base_channels": 128,
                "candidate_base_channels": 256,
                "qualification_steps": CAPACITY_QUALIFICATION_STEPS,
                "screen_samples_per_method": 1_000,
                "confirmation_samples_per_method": 10_000,
                "confirmation_requires_new_stage_decision": True,
                "source_checkpoint_binding_required": False,
                "new_output_root_required": True,
                "automatic_300k_escalation_allowed": False,
            }
        )
    return common


def _validate_stage_authorization(value: Any) -> dict[str, Any]:
    expected = {
        "status": "not_authorized",
        "required": True,
        "scope": "bounded_qualification_only",
        "reason": "preparation_and_prelaunch_snapshot_do_not_authorize_execution",
    }
    if _object(value, "stage authorization") != expected:
        raise ValueError("stage authorization must remain absent")
    return expected


def build_execution_gate(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    arm_id: str,
    source_checkout: Mapping[str, Any],
    gate_builder_git: Mapping[str, Any],
    runtime_environment_sha256: str,
    dataset_identity_sha256: str,
    gpu_inventory: list[Mapping[str, Any]],
    gpu_compute_processes: list[Mapping[str, Any]],
    conflicting_processes: list[Mapping[str, Any]],
    output_root_absent: bool,
    execution_lock_free: bool,
    free_bytes: int,
    source_checkpoint_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = validate_preparation(dict(preparation))
    identity = _identity(preparation_identity, name="preparation")
    source = _source_checkout(source_checkout)
    builder = _git(gate_builder_git, name="gate builder")
    live = _validate_live_prelaunch(
        runtime_environment_sha256=runtime_environment_sha256,
        dataset_identity_sha256=dataset_identity_sha256,
        gpu_inventory=gpu_inventory,
        gpu_compute_processes=gpu_compute_processes,
        conflicting_processes=conflicting_processes,
        output_root_absent=output_root_absent,
        execution_lock_free=execution_lock_free,
        free_bytes=free_bytes,
    )
    contract = _arm_contract(preparation, arm_id)
    if arm_id == "capacity_qualification" and source_checkpoint_binding is not None:
        raise ValueError("capacity qualification must not provide a source checkpoint binding")
    checkpoint_binding = _checkpoint_binding(
        source_checkpoint_binding, required=arm_id == "exposure_continuation"
    )
    if checkpoint_binding is not None:
        _validate_exposure_source_checkpoint(checkpoint_binding, preparation)
    return {
        "schema_version": EXECUTION_GATE_SCHEMA,
        "role": EXECUTION_GATE_ROLE,
        "status": "prepared",
        "execution_ready": False,
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "selected_arm": arm_id,
        "preparation": identity,
        "source_checkout": source,
        "gate_builder_git": builder,
        "qualification_contract": contract,
        "source_checkpoint": checkpoint_binding,
        "live_prelaunch": live,
        "stage_authorization": {
            "status": "not_authorized",
            "required": True,
            "scope": "bounded_qualification_only",
            "reason": "preparation_and_prelaunch_snapshot_do_not_authorize_execution",
        },
        "selection_policy": {
            "single_arm_only": True,
            "exact_stage_authorization_required": True,
            "automatic_300k_escalation_allowed": False,
            "terminal_hold_replacement_allowed": False,
            "formal_quality_claim_allowed": False,
            "failure_policy": "fail_closed",
        },
        "authorization_boundary": copy.deepcopy(GATE_AUTHORIZATION_BOUNDARY),
    }


def validate_execution_gate(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_preparation(dict(preparation))
    identity = _identity(preparation_identity, name="preparation")
    gate = _object(report, "exposure/capacity execution gate")
    if (
        gate.get("schema_version") != EXECUTION_GATE_SCHEMA
        or gate.get("role") != EXECUTION_GATE_ROLE
        or gate.get("status") != "prepared"
        or gate.get("execution_ready") is not False
        or gate.get("terminal_status") != "hold"
        or gate.get("generation_advantage_proven") is not False
        or gate.get("preparation") != identity
        or gate.get("authorization_boundary") != GATE_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("exposure/capacity execution gate contract differs")
    arm_id = gate.get("selected_arm")
    contract = _arm_contract(preparation, str(arm_id))
    if gate.get("qualification_contract") != contract:
        raise ValueError("qualification arm contract differs from preparation")
    raw_checkpoint_binding = gate.get("source_checkpoint")
    if arm_id == "capacity_qualification" and raw_checkpoint_binding is not None:
        raise ValueError("capacity qualification must not provide a source checkpoint binding")
    checkpoint_binding = _checkpoint_binding(
        raw_checkpoint_binding, required=arm_id == "exposure_continuation"
    )
    if checkpoint_binding is not None:
        _validate_exposure_source_checkpoint(checkpoint_binding, preparation)
    if gate.get("source_checkpoint") != checkpoint_binding:
        raise ValueError("source checkpoint binding differs")
    if gate.get("source_checkout") != {
        "revision": SOURCE_REVISION,
        "tree": SOURCE_TREE,
        "branch": SOURCE_BRANCH,
        "tracked_dirty": False,
    }:
        raise ValueError("execution gate source checkout differs")
    _git(gate.get("gate_builder_git"), name="gate builder")
    _validate_stage_authorization(gate.get("stage_authorization"))
    live = _object(gate.get("live_prelaunch"), "live prelaunch")
    checked_live = _validate_live_prelaunch(
        runtime_environment_sha256=live.get("runtime_environment_sha256"),
        dataset_identity_sha256=live.get("dataset_identity_sha256"),
        gpu_inventory=live.get("gpu_inventory"),
        gpu_compute_processes=live.get("gpu_compute_processes"),
        conflicting_processes=live.get("conflicting_processes"),
        output_root_absent=live.get("output_root_absent"),
        execution_lock_free=live.get("execution_lock_free"),
        free_bytes=live.get("free_bytes"),
    )
    if live != checked_live or live.get("minimum_free_bytes") != MIN_FREE_BYTES:
        raise ValueError("live prelaunch snapshot differs")
    selection = _object(gate.get("selection_policy"), "selection policy")
    expected_selection = {
        "single_arm_only": True,
        "exact_stage_authorization_required": True,
        "automatic_300k_escalation_allowed": False,
        "terminal_hold_replacement_allowed": False,
        "formal_quality_claim_allowed": False,
        "failure_policy": "fail_closed",
    }
    if selection != expected_selection:
        raise ValueError("selection policy weakens the bounded gate")
    return copy.deepcopy(gate)


__all__ = [
    "ARM_IDS",
    "CAPACITY_QUALIFICATION_STEPS",
    "EFFECTIVE_BATCH_SIZE",
    "EXECUTION_GATE_ROLE",
    "EXECUTION_GATE_SCHEMA",
    "EXPOSURE_TARGET_STEP",
    "GATE_AUTHORIZATION_BOUNDARY",
    "MIN_FREE_BYTES",
    "SOURCE_BRANCH",
    "SOURCE_IMAGES_SEEN_PER_METHOD",
    "SOURCE_REVISION",
    "SOURCE_STEP",
    "SOURCE_TREE",
    "build_execution_gate",
    "validate_execution_gate",
]
