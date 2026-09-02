"""Source-bound authorization for the bounded exposure continuation."""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.configs import config_from_dict, config_to_dict, load_config
from cofitok.generation.exposure_capacity import validate_source_checkpoint_bindings
from cofitok.generation.exposure_capacity_gate import (
    EXPOSURE_SCHEDULER_CONTRACT,
    SOURCE_BRANCH,
    SOURCE_REVISION,
    SOURCE_STEP,
    SOURCE_TREE,
    validate_execution_gate,
)
from cofitok.inference_replay import file_identity, reject_symlink_chain
from cofitok.reporting import git_provenance
from cofitok.training.checkpointing import verify_training_checkpoint


AUTHORIZATION_SCHEMA = "cofitok_generation_exposure_capacity_execution_authorization_v1"
AUTHORIZATION_ROLE = "source_bound_bounded_exposure_capacity_execution_authorization"
AUTHORIZATION_DECISION = "authorize_exposure_continuation_100k_to_110k_only"
TARGET_STEP = 110_000
EFFECTIVE_BATCH_SIZE = 64

SOURCE_CHECKOUT = {
    "revision": SOURCE_REVISION,
    "tree": SOURCE_TREE,
    "branch": SOURCE_BRANCH,
    "tracked_dirty": False,
}

EVALUATION_CONTRACT = {
    "sampler": "ddim",
    "sample_steps": 100,
    "samples_per_method": 10_000,
    "weights": "ema",
    "class_fidelity_required": True,
    "mechanism_diagnostics_required": True,
    "shared_random_stream_required": True,
}

AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": True,
    "execution_gate_authorized": True,
    "remote_mutation_allowed": True,
    "gpu_execution_authorized": True,
    "training_launch_allowed": True,
    "sampling_launch_allowed": True,
    "evaluation_launch_allowed": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "terminal_hold_replacement_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
}

# A stage authorization is a user-created, content-addressed sentinel.  The
# standing authorization only supplies safety policy; it must never become an
# implicit approval for a new GPU stage.
STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_exposure_capacity_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = "user_created_exposure_continuation_execution_approval"
STAGE_AUTHORIZATION_SCOPE = "exposure_continuation_100k_to_110k_execution_only"
STAGE_AUTHORIZATION_DECISION = AUTHORIZATION_DECISION
STAGE_AUTHORIZATION_BOUNDARY = copy.deepcopy(AUTHORIZATION_BOUNDARY)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def read_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return _object(value, name)


def identity(path: str | Path) -> dict[str, Any]:
    return file_identity(reject_symlink_chain(path, name="authorization source"))


def _serialized_identity(value: Any, *, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = row.get("path")
    size = row.get("bytes")
    digest = row.get("sha256")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{name} identity path is missing")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{name} identity byte count is invalid")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ValueError(f"{name} identity SHA256 is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def checkout_identity(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    provenance = git_provenance(root)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "revision": provenance["revision"],
        "tree": tree,
        "branch": provenance["branch"],
        "tracked_dirty": bool(provenance["tracked_dirty"]),
    }


def _config_mismatch_paths(
    expected: Any,
    actual: Any,
    *,
    path: str = "config",
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected or key not in actual:
                mismatches.append(child)
                continue
            mismatches.extend(_config_mismatch_paths(expected[key], actual[key], path=child))
        return mismatches
    return [] if expected == actual else [path]


def _source_summary(preparation: Mapping[str, Any], method: str) -> dict[str, Any]:
    evidence = _object(preparation.get("evidence"), "preparation evidence")
    training = _object(evidence.get("training"), "preparation training evidence")
    summaries = _object(training.get("source_checkpoints"), "preparation source checkpoints")
    return _object(summaries.get(method), f"preparation {method} source checkpoint")


def _source_binding(
    run_dir: str | Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    root = reject_symlink_chain(run_dir, name="source checkpoint run directory").resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"source checkpoint run directory is not a real directory: {root}")
    summary = _object(expected, "source checkpoint summary")
    if int(summary.get("step", -1)) != SOURCE_STEP:
        raise ValueError("source checkpoint summary is not the 100K checkpoint")
    checkpoint_summary = _object(summary.get("checkpoint"), "source checkpoint summary")
    checkpoint_name = str(checkpoint_summary.get("name", ""))
    sidecar_name = str(summary.get("integrity_manifest_name", ""))
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("source checkpoint summary filename is invalid")
    if not sidecar_name or Path(sidecar_name).name != sidecar_name:
        raise ValueError("source checkpoint sidecar filename is invalid")
    checkpoint = root / checkpoint_name
    sidecar = root / sidecar_name
    latest = root / "latest.json"
    integrity = verify_training_checkpoint(checkpoint)
    if int(integrity.get("step", -1)) != SOURCE_STEP:
        raise ValueError("source checkpoint payload is not step 100000")
    checkpoint_id = identity(checkpoint)
    if checkpoint_id["bytes"] != checkpoint_summary.get("bytes"):
        raise ValueError("source checkpoint byte count differs from preparation")
    if checkpoint_id["sha256"] != checkpoint_summary.get("sha256"):
        raise ValueError("source checkpoint SHA256 differs from preparation")
    sidecar_id = identity(sidecar)
    latest_id = identity(latest)
    if integrity.get("checkpoint") != checkpoint.name:
        raise ValueError("source checkpoint sidecar names another payload")
    if int(integrity.get("checkpoint_bytes", -1)) != checkpoint_id["bytes"]:
        raise ValueError("source checkpoint sidecar byte count differs")
    if integrity.get("checkpoint_sha256") != checkpoint_id["sha256"]:
        raise ValueError("source checkpoint sidecar SHA256 differs")
    if int(integrity.get("step", -1)) != SOURCE_STEP:
        raise ValueError("source checkpoint sidecar step differs")
    latest_payload = read_object(latest, name="source latest pointer")
    if latest_payload.get("checkpoint") != checkpoint.name or int(
        latest_payload.get("step", -1)
    ) != SOURCE_STEP:
        raise ValueError("source latest pointer does not identify the 100K checkpoint")
    for key, expected in {
        "checkpoint_bytes": checkpoint_id["bytes"],
        "checkpoint_sha256": checkpoint_id["sha256"],
        "integrity_manifest": sidecar.name,
    }.items():
        if latest_payload.get(key) != expected:
            raise ValueError(f"source latest pointer {key} differs from payload")
    return {
        "run_dir": root.as_posix(),
        "step": SOURCE_STEP,
        "checkpoint": checkpoint_id,
        "integrity_manifest": sidecar_id,
        "latest": latest_id,
    }


def _physical_source_bindings(preparation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        method: _source_binding(
            _source_summary(preparation, method)["run_dir"],
            expected=_source_summary(preparation, method),
        )
        for method in ("cofitok", "dense_identity")
    }


def _preparation_sources(preparation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = _object(preparation.get("sources"), "preparation sources")
    for name, descriptor in sources.items():
        expected = _object(descriptor, f"preparation source {name}")
        if identity(str(expected.get("path", ""))) != _serialized_identity(
            expected, name=f"preparation source {name}"
        ):
            raise ValueError(f"preparation source changed: {name}")
    source_reports = {
        name: read_object(str(sources[name]["path"]), name=name)
        for name in ("cofitok_training_report", "dense_training_report")
    }
    validate_source_checkpoint_bindings(
        dict(preparation),
        cofitok_training_report=source_reports["cofitok_training_report"],
        dense_training_report=source_reports["dense_training_report"],
    )
    return {
        "cofitok": source_reports["cofitok_training_report"],
        "dense_identity": source_reports["dense_training_report"],
    }


def _validate_config_bindings(
    config_identities: Mapping[str, Any],
    *,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(config_identities) != {"cofitok", "dense_identity"}:
        raise ValueError("continuation config identities differ")
    validated: dict[str, dict[str, Any]] = {}
    for method in ("cofitok", "dense_identity"):
        descriptor = _serialized_identity(
            config_identities[method], name=f"{method} continuation config"
        )
        actual = identity(descriptor["path"])
        if actual != descriptor:
            raise ValueError(f"{method} continuation config changed")
        # Compare the fully resolved configs used by the trainer.  Target
        # JSON files may omit dataclass defaults, while source reports store
        # the resolved form; comparing raw JSON would reject an otherwise
        # exact continuation for irrelevant omitted defaults.
        config = config_to_dict(load_config(descriptor["path"]))
        source_config = config_to_dict(
            config_from_dict(
                _object(reports[method].get("config"), f"{method} source config")
            )
        )
        mismatches = _config_mismatch_paths(source_config, config)
        allowed = {"config.name", "config.runtime.steps"}
        if set(mismatches) - allowed:
            raise ValueError(
                f"{method} continuation config changes fields outside name/runtime.steps: "
                + ", ".join(mismatches[:8])
            )
        if config.get("runtime", {}).get("steps") != TARGET_STEP:
            raise ValueError(f"{method} continuation config target horizon differs")
        if source_config.get("runtime", {}).get("steps") != SOURCE_STEP:
            raise ValueError(f"{method} source config horizon differs")
        if config.get("data", {}).get("batch_size") != EFFECTIVE_BATCH_SIZE:
            raise ValueError(f"{method} continuation effective batch differs")
        if config.get("data", {}).get("dataset") != "imagenet_256":
            raise ValueError(f"{method} continuation dataset differs")
        validated[method] = descriptor
    return validated


def _expected_target(contract: Mapping[str, Any]) -> dict[str, Any]:
    output_root = str(contract.get("output_root", ""))
    output_path = PurePosixPath(output_root)
    return {
        "output_root": output_root,
        # Keep the lock outside the candidate root so it can be acquired
        # before the root is created.  This makes the output-root absence
        # check atomic with controller ownership.
        "execution_lock": (
            output_path.parent / f".{output_path.name}.exposure_execution.lock"
        ).as_posix(),
        "source_step": SOURCE_STEP,
        "target_step": TARGET_STEP,
        "additional_steps": TARGET_STEP - SOURCE_STEP,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "methods": ["cofitok", "dense_identity"],
        "objective_change_allowed": False,
        "conditioning_change_allowed": False,
        "model_layout_change_allowed": False,
    }


def _validate_standing_authorization(standing_identity: Mapping[str, Any]) -> None:
    descriptor = _serialized_identity(standing_identity, name="standing authorization")
    standing = read_object(descriptor["path"], name="standing authorization")
    if standing.get("status") != "active":
        raise ValueError("standing authorization is not active")
    boundaries = _object(standing.get("preserved_safety_boundaries"), "standing safety boundaries")
    for key in (
        "unrelated_project_processes_must_not_be_modified",
        "formal_remote_checkout_must_not_be_modified",
        "locked_evidence_must_not_be_overwritten",
        "independent_clean_checkout_required",
        "exact_revision_stage_and_output_binding_required",
        "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing",
    ):
        if boundaries.get(key) is not True:
            raise ValueError(f"standing authorization safety boundary {key} is missing")


def validate_live_snapshot(snapshot: Mapping[str, Any], *, output_root: str) -> dict[str, Any]:
    live = _object(snapshot, "authorization live prelaunch")
    gpu = live.get("gpu_inventory")
    if not isinstance(gpu, list) or len(gpu) != 1:
        raise ValueError("authorization requires exactly one target GPU")
    row = _object(gpu[0], "authorization GPU")
    try:
        memory_used = int(row["memory_used_mib"])
        utilization = int(row["utilization_percent"])
        memory_total = int(row["memory_total_mib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("authorization GPU inventory is malformed") from exc
    if (
        memory_used < 0
        or utilization < 0
        or memory_total < 1
        or memory_used > 16
        or utilization > 5
    ):
        raise ValueError("authorization target GPU is not idle")
    if live.get("gpu_compute_processes") != []:
        raise ValueError("authorization recorded GPU compute processes")
    if live.get("conflicting_processes") != []:
        raise ValueError("authorization recorded conflicting project processes")
    if live.get("output_root") != output_root or live.get("output_root_absent") is not True:
        raise ValueError("authorization output-root prelaunch binding differs")
    if live.get("execution_lock_free") is not True:
        raise ValueError("authorization execution lock was not free")
    if int(live.get("free_bytes", 0)) < 120 * 1024**3:
        raise ValueError("authorization storage preflight is below the minimum")
    for key in ("runtime_environment_sha256", "dataset_identity_sha256"):
        value = str(live.get(key, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"authorization {key} is malformed")
    return copy.deepcopy(live)


def _validate_execution_checkout(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(value, "execution checkout")
    required = {"revision", "tree", "branch", "tracked_dirty"}
    if set(row) != required:
        raise ValueError("execution checkout fields differ")
    revision = row.get("revision")
    tree = row.get("tree")
    branch = row.get("branch")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision
    ):
        raise ValueError("execution checkout revision is malformed")
    if not isinstance(tree, str) or len(tree) != 40 or any(
        char not in "0123456789abcdef" for char in tree
    ):
        raise ValueError("execution checkout tree is malformed")
    if not isinstance(branch, str) or not branch or row.get("tracked_dirty") is not False:
        raise ValueError("execution checkout is dirty or malformed")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def validate_stage_authorization(
    value: Mapping[str, Any],
    *,
    expected_output_root: str,
    execution_checkout: Mapping[str, Any],
    source_checkout: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the user-created approval for this exact bounded stage."""

    approval = _object(value, "stage authorization")
    if (
        approval.get("schema_version") != STAGE_AUTHORIZATION_SCHEMA
        or approval.get("role") != STAGE_AUTHORIZATION_ROLE
        or approval.get("status") != "approved"
        or approval.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or approval.get("decision") != AUTHORIZATION_DECISION
    ):
        raise ValueError("stage authorization schema or scope differs")
    expected_selection = {
        "source_revision": SOURCE_REVISION,
        "source_tree": SOURCE_TREE,
        "source_branch": SOURCE_BRANCH,
        "source_step": SOURCE_STEP,
        "target_step": TARGET_STEP,
        "execution_revision": execution_checkout["revision"],
        "execution_tree": execution_checkout["tree"],
        "execution_branch": execution_checkout["branch"],
        "output_root": Path(expected_output_root).resolve().as_posix(),
    }
    if approval.get("selection") != expected_selection:
        raise ValueError("stage authorization selection differs")
    record = _object(approval.get("approval_record"), "stage approval record")
    if not str(record.get("approved_by", "")).strip() or not str(
        record.get("approved_at", "")
    ).strip():
        raise ValueError("stage approval record is incomplete")
    if approval.get("source_checkout") != dict(source_checkout):
        raise ValueError("stage authorization source checkout differs")
    if approval.get("execution_checkout") != dict(execution_checkout):
        raise ValueError("stage authorization execution checkout differs")
    if approval.get("authorization_boundary") != STAGE_AUTHORIZATION_BOUNDARY:
        raise ValueError("stage authorization boundary differs")
    return {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "decision": AUTHORIZATION_DECISION,
        "selection": copy.deepcopy(expected_selection),
        "approval_record": copy.deepcopy(record),
        "source_checkout": copy.deepcopy(dict(source_checkout)),
        "execution_checkout": copy.deepcopy(dict(execution_checkout)),
        "authorization_boundary": copy.deepcopy(STAGE_AUTHORIZATION_BOUNDARY),
    }


def validate_source_checkout(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable checkout that produced the 100K source pair."""
    checkout = _validate_execution_checkout(value)
    if checkout != SOURCE_CHECKOUT:
        raise ValueError("source checkout does not match the locked 100K bridge")
    return checkout


def _validate_contract_fields(contract: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(_object(contract, "exposure qualification contract"))
    if normalized.get("arm_id") != "exposure_continuation":
        raise ValueError("authorization qualification arm differs")
    if normalized.get("target_step") != TARGET_STEP:
        raise ValueError("authorization qualification target differs")
    if normalized.get("source_step") != SOURCE_STEP:
        raise ValueError("authorization qualification source step differs")
    if normalized.get("additional_steps") != TARGET_STEP - SOURCE_STEP:
        raise ValueError("authorization qualification additional steps differ")
    if normalized.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE:
        raise ValueError("authorization qualification effective batch differs")
    if normalized.get("methods") != ["cofitok", "dense_identity"]:
        raise ValueError("authorization qualification methods differ")
    if normalized.get("dataset") != "imagenet_256":
        raise ValueError("authorization qualification dataset differs")
    if normalized.get("objective_change_allowed") is not False:
        raise ValueError("authorization qualification permits an objective change")
    if normalized.get("conditioning_change_allowed") is not False:
        raise ValueError("authorization qualification permits a conditioning change")
    if normalized.get("formal_quality_claim_allowed") is not False:
        raise ValueError("authorization qualification permits a quality claim")
    if normalized.get("initialization") != "exact_100k_checkpoint_resume_only":
        raise ValueError("authorization qualification initialization differs")
    if normalized.get("source_checkpoint_binding_required") is not True:
        raise ValueError("authorization qualification does not require source binding")
    if normalized.get("scheduler") != EXPOSURE_SCHEDULER_CONTRACT:
        raise ValueError("authorization scheduler horizon contract differs")
    if normalized.get("automatic_300k_escalation_allowed") is not False:
        raise ValueError("authorization qualification permits automatic escalation")
    if normalized.get("evaluation") != EVALUATION_CONTRACT:
        raise ValueError("authorization evaluation contract differs")
    output_root = normalized.get("output_root")
    if not isinstance(output_root, str) or not output_root.startswith(
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ):
        raise ValueError("authorization output root is not project scoped")
    return normalized


def _validate_source_bindings(
    source_checkpoints: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source_checkpoints = _object(source_checkpoints, "authorization source checkpoints")
    if set(source_checkpoints) != {"cofitok", "dense_identity"}:
        raise ValueError("authorization source checkpoint methods differ")
    physical = _physical_source_bindings(preparation)
    normalized: dict[str, dict[str, Any]] = {}
    for method in ("cofitok", "dense_identity"):
        supplied = _object(
            source_checkpoints[method], f"authorization {method} source checkpoint"
        )
        if supplied != physical[method]:
            raise ValueError(
                f"authorization {method} source payload/sidecar/latest differs from physical source"
            )
        normalized[method] = physical[method]
    if normalized["cofitok"] != gate.get("source_checkpoint"):
        raise ValueError("authorization CoFiTok source differs from candidate gate")
    return normalized


def build_authorization(
    *,
    gate: Mapping[str, Any],
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    gate_identity: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    config_identities: Mapping[str, Any],
    source_checkpoints: Mapping[str, Any],
    live_prelaunch: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    source_checkout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_execution_gate(
        dict(gate),
        preparation=dict(preparation),
        preparation_identity=dict(preparation_identity),
    )
    if gate.get("selected_arm") != "exposure_continuation":
        raise ValueError("execution authorization must select exposure_continuation")
    if gate.get("execution_ready") is not False:
        raise ValueError("candidate gate must remain non-authorizing")
    if gate.get("terminal_status") != "hold":
        raise ValueError("exposure authorization must preserve terminal hold")
    if gate.get("generation_advantage_proven") is not False:
        raise ValueError("exposure authorization cannot prove generation advantage")
    contract = _validate_contract_fields(gate.get("qualification_contract"))
    live = validate_live_snapshot(live_prelaunch, output_root=str(contract["output_root"]))
    execution = _validate_execution_checkout(execution_checkout)
    source = validate_source_checkout(source_checkout or SOURCE_CHECKOUT)
    stage = validate_stage_authorization(
        stage_authorization,
        expected_output_root=str(contract["output_root"]),
        execution_checkout=execution,
        source_checkout=source,
    )
    if stage != dict(stage_authorization):
        raise ValueError("stage authorization contains unsupported fields")
    stage_identity = _serialized_identity(
        stage_authorization_identity,
        name="stage authorization identity",
    )
    if identity(stage_identity["path"]) != stage_identity:
        raise ValueError("stage authorization changed while building execution authorization")
    _validate_standing_authorization(standing_identity)
    reports = _preparation_sources(preparation)
    configs = _validate_config_bindings(config_identities, reports=reports)
    sources = _validate_source_bindings(
        source_checkpoints, preparation=preparation, gate=gate
    )
    target = _expected_target(contract)
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "role": AUTHORIZATION_ROLE,
        "status": "authorized",
        "decision": AUTHORIZATION_DECISION,
        "scope": "bounded_qualification_only",
        "selected_arm": "exposure_continuation",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation": dict(preparation_identity),
        "candidate_gate": dict(gate_identity),
        "standing_authorization": dict(standing_identity),
        "stage_authorization_identity": stage_identity,
        "user_stage_authorization": stage,
        "source_checkout": source,
        "execution_checkout": execution,
        "source_checkpoints": sources,
        "configs": configs,
        "qualification_contract": copy.deepcopy(contract),
        "target": target,
        "evaluation": copy.deepcopy(EVALUATION_CONTRACT),
        "live_prelaunch": live,
        "stage_authorization": {
            "status": "authorized",
            "required": True,
            "scope": "bounded_qualification_only",
            "decision": AUTHORIZATION_DECISION,
            "reason": "standing authorization permits necessary bounded CoFiTok experiments",
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_authorization_contract(
    authorization: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    gate_identity: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    config_identities: Mapping[str, Any],
    source_checkout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    auth = _object(authorization, "exposure execution authorization")
    if auth.get("schema_version") != AUTHORIZATION_SCHEMA or auth.get("role") != AUTHORIZATION_ROLE:
        raise ValueError("exposure execution authorization schema differs")
    if auth.get("status") != "authorized" or auth.get("decision") != AUTHORIZATION_DECISION:
        raise ValueError("exposure execution authorization status differs")
    if auth.get("scope") != "bounded_qualification_only" or auth.get("selected_arm") != "exposure_continuation":
        raise ValueError("exposure execution authorization scope differs")
    if auth.get("terminal_status") != "hold" or auth.get("generation_advantage_proven") is not False:
        raise ValueError("exposure execution authorization weakens the terminal hold")
    if auth.get("authorization_boundary") != AUTHORIZATION_BOUNDARY:
        raise ValueError("exposure execution authorization boundary differs")
    validate_execution_gate(
        dict(gate), preparation=dict(preparation), preparation_identity=dict(preparation_identity)
    )
    if auth.get("candidate_gate") != dict(gate_identity):
        raise ValueError("execution authorization binds another candidate gate")
    if auth.get("preparation") != dict(preparation_identity):
        raise ValueError("execution authorization binds another preparation")
    if auth.get("standing_authorization") != dict(standing_identity):
        raise ValueError("execution authorization binds another standing authorization")
    stage_identity = _serialized_identity(
        auth.get("stage_authorization_identity"),
        name="stage authorization identity",
    )
    if identity(stage_identity["path"]) != stage_identity:
        raise ValueError("stage authorization changed")
    if auth.get("execution_checkout") != _validate_execution_checkout(execution_checkout):
        raise ValueError("execution authorization binds another execution checkout")
    _validate_standing_authorization(standing_identity)
    reports = _preparation_sources(preparation)
    configs = _validate_config_bindings(config_identities, reports=reports)
    if auth.get("configs") != configs:
        raise ValueError("execution authorization binds another config")
    expected_source_checkout = validate_source_checkout(source_checkout or SOURCE_CHECKOUT)
    if auth.get("source_checkout") != expected_source_checkout:
        raise ValueError("execution authorization source checkout differs")
    contract = _validate_contract_fields(auth.get("qualification_contract"))
    stage_report = read_object(stage_identity["path"], name="stage authorization")
    normalized_stage = validate_stage_authorization(
        stage_report,
        expected_output_root=str(contract["output_root"]),
        execution_checkout=_validate_execution_checkout(execution_checkout),
        source_checkout=expected_source_checkout,
    )
    if auth.get("user_stage_authorization") != normalized_stage:
        raise ValueError("execution authorization embeds a different user stage authorization")
    gate_contract = _validate_contract_fields(gate.get("qualification_contract"))
    if contract != gate_contract:
        raise ValueError("authorization qualification contract differs from candidate gate")
    if auth.get("evaluation") != EVALUATION_CONTRACT:
        raise ValueError("authorization evaluation contract differs")
    if auth.get("target") != _expected_target(contract):
        raise ValueError("authorization target contract differs")
    expected_stage = {
        "status": "authorized",
        "required": True,
        "scope": "bounded_qualification_only",
        "decision": AUTHORIZATION_DECISION,
        "reason": "standing authorization permits necessary bounded CoFiTok experiments",
    }
    if auth.get("stage_authorization") != expected_stage:
        raise ValueError("authorization stage binding differs")
    _validate_source_bindings(
        _object(auth.get("source_checkpoints"), "authorization source checkpoints"),
        preparation=preparation,
        gate=gate,
    )
    live = validate_live_snapshot(
        auth.get("live_prelaunch", {}), output_root=str(contract["output_root"])
    )
    if live != auth.get("live_prelaunch"):
        raise ValueError("authorization live prelaunch snapshot is malformed")
    return copy.deepcopy(auth)


__all__ = [
    "AUTHORIZATION_BOUNDARY",
    "AUTHORIZATION_DECISION",
    "AUTHORIZATION_ROLE",
    "AUTHORIZATION_SCHEMA",
    "EVALUATION_CONTRACT",
    "EFFECTIVE_BATCH_SIZE",
    "STAGE_AUTHORIZATION_BOUNDARY",
    "STAGE_AUTHORIZATION_DECISION",
    "STAGE_AUTHORIZATION_ROLE",
    "STAGE_AUTHORIZATION_SCOPE",
    "STAGE_AUTHORIZATION_SCHEMA",
    "TARGET_STEP",
    "SOURCE_CHECKOUT",
    "build_authorization",
    "checkout_identity",
    "file_identity",
    "identity",
    "read_object",
    "validate_authorization_contract",
    "validate_stage_authorization",
    "validate_source_checkout",
    "validate_live_snapshot",
]
