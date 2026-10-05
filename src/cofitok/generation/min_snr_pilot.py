from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from cofitok.generation_pair import generation_pair_contract


PREPARATION_SCHEMA = "cofitok_matched_min_snr_pilot_preparation_v1"
EXECUTION_GATE_SCHEMA = "cofitok_matched_min_snr_pilot_execution_gate_v1"
RESULT_SCHEMA = "cofitok_matched_min_snr_pilot_result_v1"
LEGACY_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
LEGACY_BRANCH = "scale/generation-stability-quality-bridge-100k"
PILOT_BRANCH = "scale/generation-min-snr-matched-pilot-v1-20260825"
DIRECT_EXECUTION_USER_INSTRUCTION = "之后不要我授权你直接运行需要的实验"
DATASET_IDENTITY_SHA256 = (
    "6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659"
)
LEGACY_RUNTIME_ENVIRONMENT_SHA256 = (
    "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"
)
METHODS = ("cofitok", "dense_identity")
GAMMA = 5.0
PILOT_STEP = 50_000
SCHEDULER_HORIZON = 100_000
EFFECTIVE_BATCH_SIZE = 64
IMAGES_PER_METHOD = PILOT_STEP * EFFECTIVE_BATCH_SIZE
SAMPLE_COUNT = 10_000
SAMPLE_STEPS = 100
SAMPLE_SEED = 20_260_825
MIN_FREE_BYTES = 120 * 1024**3
TRAINING_SEMANTIC_FILES = (
    "scripts/train_generation.py",
    "src/cofitok/configs.py",
    "src/cofitok/diffusion/schedule.py",
    "src/cofitok/generation_pair.py",
    "src/cofitok/training/losses.py",
)

PREPARATION_BOUNDARY = {
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "gpu_execution_allowed": False,
    "continuation_beyond_50000_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "new_source_bound_execution_gate_required": True,
}

EXECUTION_BOUNDARY = {
    "matched_min_snr_pilot_training_allowed": True,
    "same_config_restart_resume_allowed": True,
    "legacy_50k_control_sampling_allowed": True,
    "pilot_50k_sampling_allowed": True,
    "matched_ddim100_evaluation_allowed": True,
    "gpu_execution_allowed": True,
    "scope_limited_to_output_root_and_legacy_50k_read_only_controls": True,
    "changed_config_resume_allowed": False,
    "continuation_beyond_50000_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}

RESULT_BOUNDARY = {
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "gpu_execution_allowed": False,
    "continuation_beyond_50000_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "new_gate_required_for_any_followup": True,
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _hex(value: Any, *, length: int) -> str:
    text = str(value)
    if len(text) != length or any(c not in "0123456789abcdef" for c in text):
        raise ValueError("invalid hexadecimal identity")
    return text


def normalize_identity(value: Any, *, label: str) -> dict[str, Any]:
    row = _mapping(value, label=label)
    path = str(row.get("path", ""))
    size = int(row.get("bytes", 0))
    sha256 = _hex(row.get("sha256", ""), length=64)
    if not path or size < 1:
        raise ValueError(f"{label} identity is incomplete")
    return {"path": path, "bytes": size, "sha256": sha256}


def normalize_git(
    value: Any,
    *,
    label: str,
    expected_branch: str | None = None,
) -> dict[str, Any]:
    row = _mapping(value, label=label)
    revision = _hex(row.get("revision", ""), length=40)
    tree = _hex(row.get("tree", ""), length=40) if "tree" in row else None
    branch = str(row.get("branch", ""))
    if not branch or row.get("tracked_dirty") is not False:
        raise ValueError(f"{label} Git identity is not clean")
    if expected_branch is not None and branch != expected_branch:
        raise ValueError(f"{label} Git branch differs")
    result = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if tree is not None:
        result["tree"] = tree
    return result


def _require_false(value: Any, *, fields: tuple[str, ...], label: str) -> None:
    row = _mapping(value, label=label)
    for field in fields:
        if row.get(field) is not False:
            raise ValueError(f"{label} permits {field}")


def _validate_execution_authorization(value: Any) -> dict[str, Any]:
    authorization = _mapping(value, label="authorization record")
    if (
        authorization.get("scope") != "matched_min_snr_50k_pilot_only"
        or authorization.get("approved_by") != "user"
        or authorization.get("instruction") != DIRECT_EXECUTION_USER_INSTRUCTION
        or authorization.get("direct_execution_without_repeated_prompt") is not True
        or authorization.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("scoped user authorization record differs")
    return copy.deepcopy(dict(authorization))


def _validate_post_diagnostic_decision(value: Any) -> None:
    report = _mapping(value, label="post-diagnostic decision")
    stage = _mapping(report.get("recommended_next_stage"), label="recommended stage")
    controlled = _mapping(stage.get("controlled_change"), label="controlled change")
    requirements = _mapping(
        stage.get("preparation_requirements"), label="preparation requirements"
    )
    if (
        report.get("schema")
        != "cofitok_epsilon_stability_post_diagnostic_decision_v1"
        or report.get("status") != "completed"
        or report.get("terminal_status") != "hold"
        or report.get("decision") != "prepare_fresh_matched_min_snr_training_pilot"
        or report.get("generation_advantage_proven") is not False
        or stage.get("id") != "prepare_fresh_matched_min_snr_training_pilot"
        or stage.get("execution_ready") is not False
        or controlled.get("field") != "loss.min_snr_gamma"
        or float(controlled.get("legacy_value", -1.0)) != 0.0
        or controlled.get("prediction_target") != "epsilon"
        or requirements.get("fresh_initialization_required") is not True
        or requirements.get("changed_config_resume_allowed") is not False
        or requirements.get("exact_data_initialization_and_random_stream_binding_required")
        is not True
        or requirements.get("physical_checkpoint_integrity_required") is not True
        or requirements.get("matched_ddim100_quality_and_class_fidelity_evaluation_required")
        is not True
    ):
        raise ValueError("post-diagnostic decision does not authorize preparation")
    _require_false(
        report.get("authorization_boundary"),
        fields=(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "inference_export_allowed",
            "release_allowed",
            "process_signals_allowed",
            "gpu_execution_allowed",
        ),
        label="post-diagnostic authorization boundary",
    )


def _validate_pair_monitor(value: Any) -> None:
    report = _mapping(value, label="pair monitor")
    git = _mapping(report.get("git"), label="pair monitor Git")
    runs = _mapping(report.get("runs"), label="pair monitor runs")
    if (
        report.get("schema_version") != 2
        or report.get("status") != "pass"
        or report.get("stage") != "complete"
        or report.get("issues") != []
        or set(runs) != set(METHODS)
        or git.get("revision") != LEGACY_REVISION
        or git.get("branch") != LEGACY_BRANCH
        or git.get("tracked_dirty", git.get("dirty")) is not False
    ):
        raise ValueError("legacy matched pair is not complete and clean")


def _loss_with_default(config: Mapping[str, Any]) -> dict[str, Any]:
    loss = copy.deepcopy(dict(_mapping(config.get("loss"), label="loss config")))
    loss.setdefault("min_snr_gamma", 0.0)
    return loss


def _without_treatment(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    normalized.pop("name", None)
    normalized["loss"] = _loss_with_default(normalized)
    normalized["loss"].pop("min_snr_gamma", None)
    return normalized


def _validate_configs(
    legacy_configs: Mapping[str, Mapping[str, Any]],
    pilot_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(legacy_configs) != set(METHODS) or set(pilot_configs) != set(METHODS):
        raise ValueError("matched config set differs")
    for method in METHODS:
        legacy = _mapping(legacy_configs[method], label=f"legacy {method} config")
        pilot = _mapping(pilot_configs[method], label=f"pilot {method} config")
        legacy_gamma = float(_loss_with_default(legacy)["min_snr_gamma"])
        pilot_gamma = float(_loss_with_default(pilot)["min_snr_gamma"])
        runtime = _mapping(pilot.get("runtime"), label=f"pilot {method} runtime")
        diffusion = _mapping(
            pilot.get("diffusion"), label=f"pilot {method} diffusion"
        )
        data = _mapping(pilot.get("data"), label=f"pilot {method} data")
        if (
            legacy_gamma != 0.0
            or pilot_gamma != GAMMA
            or _without_treatment(legacy) != _without_treatment(pilot)
            or int(runtime.get("steps", -1)) != SCHEDULER_HORIZON
            or int(runtime.get("seed", -1)) != 2027
            or int(runtime.get("checkpoint_interval", -1)) != 5_000
            or runtime.get("protected_checkpoint_steps") != [50_000, 100_000]
            or diffusion.get("prediction_target") != "epsilon"
            or diffusion.get("schedule_type") != "cosine"
            or data.get("dataset") != "imagenet_256"
            or float(data.get("random_horizontal_flip_prob", -1.0)) != 0.5
        ):
            raise ValueError(f"pilot {method} changes more than Min-SNR")
    contract = generation_pair_contract(
        copy.deepcopy(dict(pilot_configs["cofitok"])),
        copy.deepcopy(dict(pilot_configs["dense_identity"])),
    )
    if contract.get("valid") is not True or contract.get("issues") != []:
        raise ValueError("pilot matched pair contract fails")
    return contract


def _validate_legacy_training_report(value: Any, *, method: str) -> dict[str, Any]:
    report = _mapping(value, label=f"legacy {method} training report")
    git = _mapping(report.get("git"), label=f"legacy {method} training Git")
    config = _mapping(report.get("config"), label=f"legacy {method} resolved config")
    data = _mapping(config.get("data"), label=f"legacy {method} data")
    optimization = _mapping(
        config.get("optimization"), label=f"legacy {method} optimization"
    )
    runtime = _mapping(config.get("runtime"), label=f"legacy {method} runtime")
    provenance = _mapping(
        report.get("dataset_provenance"), label=f"legacy {method} dataset"
    )
    if (
        report.get("training_complete") is not True
        or int(report.get("completed_steps", -1)) != SCHEDULER_HORIZON
        or git.get("revision") != LEGACY_REVISION
        or git.get("branch") != LEGACY_BRANCH
        or git.get("tracked_dirty", git.get("dirty")) is not False
        or int(data.get("batch_size", -1)) != EFFECTIVE_BATCH_SIZE
        or int(optimization.get("gradient_accumulation_steps", -1)) != 1
        or int(runtime.get("steps", -1)) != SCHEDULER_HORIZON
        or int(runtime.get("seed", -1)) != 2027
        or float(_loss_with_default(config)["min_snr_gamma"]) != 0.0
        or provenance.get("identity_sha256") != DATASET_IDENTITY_SHA256
        or report.get("runtime_environment_sha256")
        != LEGACY_RUNTIME_ENVIRONMENT_SHA256
    ):
        raise ValueError(f"legacy {method} training identity differs")
    return {
        "completed_steps": SCHEDULER_HORIZON,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "seed": 2027,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "runtime_environment_sha256": LEGACY_RUNTIME_ENVIRONMENT_SHA256,
    }


def _validate_legacy_audit(value: Any, *, method: str) -> dict[str, Any]:
    report = _mapping(value, label=f"legacy {method} checkpoint audit")
    checkpoint = _mapping(report.get("checkpoint"), label=f"legacy {method} checkpoint")
    payload = _mapping(checkpoint.get("payload"), label=f"legacy {method} payload")
    integrity = _mapping(
        checkpoint.get("integrity"), label=f"legacy {method} integrity"
    )
    metrics = _mapping(report.get("metrics"), label=f"legacy {method} metrics")
    target = _mapping(metrics.get("target_row"), label=f"legacy {method} target row")
    checkout = _mapping(
        report.get("training_checkout"), label=f"legacy {method} checkout"
    )
    if (
        report.get("status") != "pass"
        or int(checkpoint.get("step", -1)) != PILOT_STEP
        or checkpoint.get("physical_sha256_verified") is not True
        or payload.get("sha256") != integrity.get("checkpoint_sha256")
        or int(integrity.get("step", -1)) != PILOT_STEP
        or integrity.get("git_revision") != LEGACY_REVISION
        or integrity.get("git_branch") != LEGACY_BRANCH
        or integrity.get("git_dirty") is not False
        or integrity.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256
        or integrity.get("runtime_environment_sha256")
        != LEGACY_RUNTIME_ENVIRONMENT_SHA256
        or metrics.get("strictly_increasing") is not True
        or metrics.get("samples_seen_binding_verified") is not True
        or int(metrics.get("last_step", -1)) != PILOT_STEP
        or int(target.get("step", -1)) != PILOT_STEP
        or int(target.get("samples_seen", -1)) != IMAGES_PER_METHOD
        or checkout.get("revision") != LEGACY_REVISION
        or checkout.get("branch") != LEGACY_BRANCH
        or checkout.get("tracked_dirty") is not False
    ):
        raise ValueError(f"legacy {method} 50K checkpoint audit differs")
    return {
        "checkpoint": normalize_identity(payload, label=f"legacy {method} payload"),
        "step": PILOT_STEP,
        "samples_seen": IMAGES_PER_METHOD,
        "physical_sha256_verified": True,
    }


def build_preparation(
    *,
    post_diagnostic_decision: Mapping[str, Any],
    pair_monitor: Mapping[str, Any],
    legacy_configs: Mapping[str, Mapping[str, Any]],
    pilot_configs: Mapping[str, Mapping[str, Any]],
    legacy_training_reports: Mapping[str, Mapping[str, Any]],
    legacy_checkpoint_audits: Mapping[str, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    builder_git: Mapping[str, Any],
    training_source_delta: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    _validate_post_diagnostic_decision(post_diagnostic_decision)
    _validate_pair_monitor(pair_monitor)
    contract = _validate_configs(legacy_configs, pilot_configs)
    legacy_reports = {
        method: _validate_legacy_training_report(
            legacy_training_reports[method], method=method
        )
        for method in METHODS
    }
    legacy_controls = {
        method: _validate_legacy_audit(legacy_checkpoint_audits[method], method=method)
        for method in METHODS
    }
    normalized_sources = {
        str(name): normalize_identity(identity, label=str(name))
        for name, identity in source_identities.items()
    }
    required_sources = {
        "post_diagnostic_decision",
        "pair_monitor",
        "legacy_cofitok_config",
        "legacy_dense_config",
        "pilot_cofitok_config",
        "pilot_dense_config",
        "legacy_cofitok_training_report",
        "legacy_dense_training_report",
        "legacy_cofitok_checkpoint_audit_50k",
        "legacy_dense_checkpoint_audit_50k",
    }
    if set(normalized_sources) != required_sources:
        raise ValueError("preparation source set differs")
    git = normalize_git(builder_git, label="preparation builder", expected_branch=PILOT_BRANCH)
    delta = _mapping(training_source_delta, label="training source delta")
    changed = tuple(delta.get("changed_semantic_files", ()))
    if (
        delta.get("base_revision") != LEGACY_REVISION
        or changed != TRAINING_SEMANTIC_FILES
        or delta.get("base_is_ancestor") is not True
    ):
        raise ValueError("training source delta is not the constrained Min-SNR patch")
    if not output_root.startswith(
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ):
        raise ValueError("pilot output root is outside CoFiTok generation storage")
    return {
        "schema": PREPARATION_SCHEMA,
        "role": "generation_matched_min_snr_training_pilot_preparation",
        "status": "prepared",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "builder_git": git,
        "source_identities": normalized_sources,
        "training_source_delta": copy.deepcopy(dict(delta)),
        "controlled_change": {
            "field": "loss.min_snr_gamma",
            "legacy_value": 0.0,
            "pilot_value": GAMMA,
            "weighting": "min(SNR, gamma) / SNR",
            "prediction_target": "epsilon",
            "cofitok_representation_changed": False,
            "only_training_semantic_change": True,
        },
        "training_contract": {
            "methods": list(METHODS),
            "fresh_initialization": True,
            "changed_config_resume_allowed": False,
            "same_config_restart_resume_allowed": True,
            "dataset": "imagenet_256",
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "seed": 2027,
            "data_order_and_augmentation_stream_seed_matched": True,
            "scheduler_horizon_steps": SCHEDULER_HORIZON,
            "pilot_stop_step": PILOT_STEP,
            "optimizer_steps_per_method": PILOT_STEP,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "images_seen_per_method": IMAGES_PER_METHOD,
            "micro_batch_size": EFFECTIVE_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "checkpoint_steps": [PILOT_STEP],
            "continuation_beyond_50000_requires_new_gate": True,
            "pair_contract": contract,
        },
        "legacy_control_policy": {
            "status": "exact_frozen_50k_controls_accepted",
            "fresh_gamma_zero_control_required": False,
            "reason": (
                "pilot branches directly from the legacy training revision and "
                "changes only the standard matched Min-SNR loss path"
            ),
            "training_reports": legacy_reports,
            "controls": legacy_controls,
        },
        "evaluation_contract": {
            "arms": [
                f"{recipe}_{method}"
                for recipe in ("legacy_gamma0", "pilot_gamma5")
                for method in METHODS
            ],
            "checkpoint_step": PILOT_STEP,
            "weights": "ema",
            "sampler": "ddim",
            "sample_steps": SAMPLE_STEPS,
            "samples_per_arm": SAMPLE_COUNT,
            "seed": SAMPLE_SEED,
            "start_index": 0,
            "fixed_random_stream_shared_across_arms": True,
            "batch_size": 32,
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "precision": "bf16",
            "class_schedule": "balanced_modulo",
            "real_set_image_count": 50_000,
            "checkpoint_mechanism_images": 256,
            "cofitok_random_orders": 4,
        },
        "precommitted_success_thresholds": {
            "both_methods_min_relative_fid_improvement": 0.05,
            "both_methods_max_precision_regression": 0.01,
            "both_methods_max_recall_regression": 0.01,
            "both_methods_max_class_top1_regression": 0.01,
            "both_methods_max_class_top5_regression": 0.02,
            "min_class_top1": 0.01,
            "min_class_top5": 0.05,
            "min_predicted_class_fraction": 0.25,
            "min_normalized_predicted_entropy": 0.50,
            "cofitok_zero_token_max_abs": 1e-8,
            "cofitok_min_shuffled_to_ordered_endpoint_ratio": 10.0,
            "cofitok_ordered_rank_by_path_auc": 1,
            "cofitok_min_coarse_token_energy_ratio": 0.10,
        },
        "output_root": output_root,
        "claim_policy": {
            "pilot_is_single_seed": True,
            "pilot_success_proves_generation_advantage": False,
            "pilot_success_replaces_terminal_hold": False,
            "cross_tier_numeric_ranking_allowed": False,
            "new_gate_required_for_100k_continuation": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }


def validate_preparation(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema") != PREPARATION_SCHEMA
        or report.get("status") != "prepared"
        or report.get("terminal_status") != "hold"
        or report.get("generation_advantage_proven") is not False
        or report.get("authorization_boundary") != PREPARATION_BOUNDARY
    ):
        raise ValueError("Min-SNR pilot preparation contract differs")
    training = _mapping(report.get("training_contract"), label="training contract")
    evaluation = _mapping(report.get("evaluation_contract"), label="evaluation contract")
    control = _mapping(report.get("controlled_change"), label="controlled change")
    legacy = _mapping(report.get("legacy_control_policy"), label="legacy controls")
    thresholds = _mapping(
        report.get("precommitted_success_thresholds"), label="success thresholds"
    )
    if (
        control.get("field") != "loss.min_snr_gamma"
        or float(control.get("pilot_value", -1.0)) != GAMMA
        or control.get("only_training_semantic_change") is not True
        or training.get("methods") != list(METHODS)
        or training.get("fresh_initialization") is not True
        or training.get("changed_config_resume_allowed") is not False
        or int(training.get("scheduler_horizon_steps", -1)) != SCHEDULER_HORIZON
        or int(training.get("pilot_stop_step", -1)) != PILOT_STEP
        or int(training.get("images_seen_per_method", -1)) != IMAGES_PER_METHOD
        or int(training.get("effective_batch_size", -1)) != EFFECTIVE_BATCH_SIZE
        or training.get("pair_contract", {}).get("valid") is not True
        or legacy.get("status") != "exact_frozen_50k_controls_accepted"
        or legacy.get("fresh_gamma_zero_control_required") is not False
        or int(evaluation.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(evaluation.get("samples_per_arm", -1)) != SAMPLE_COUNT
        or int(evaluation.get("seed", -1)) != SAMPLE_SEED
        or evaluation.get("fixed_random_stream_shared_across_arms") is not True
        or float(thresholds.get("both_methods_min_relative_fid_improvement", -1.0))
        != 0.05
    ):
        raise ValueError("Min-SNR pilot preparation details differ")
    normalize_git(report.get("builder_git"), label="preparation builder", expected_branch=PILOT_BRANCH)
    for name, identity in _mapping(
        report.get("source_identities"), label="preparation sources"
    ).items():
        normalize_identity(identity, label=str(name))
    if not str(report.get("output_root", "")).startswith(
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    ):
        raise ValueError("Min-SNR pilot output root differs")
    return copy.deepcopy(dict(report))


def build_execution_gate(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    gate_builder_git: Mapping[str, Any],
    runtime_environment_sha256: str,
    dataset_identity_sha256: str,
    gpu_inventory: list[Mapping[str, Any]],
    gpu_compute_processes: list[Mapping[str, Any]],
    conflicting_processes: list[Mapping[str, Any]],
    output_root_absent: bool,
    execution_lock_free: bool,
    free_bytes: int,
    authorization_record: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_preparation(preparation)
    identity = normalize_identity(preparation_identity, label="preparation")
    git = normalize_git(
        gate_builder_git, label="execution gate builder", expected_branch=PILOT_BRANCH
    )
    if git != prepared["builder_git"]:
        raise ValueError("execution gate and preparation use different source")
    if runtime_environment_sha256 != LEGACY_RUNTIME_ENVIRONMENT_SHA256:
        raise ValueError("pilot runtime environment differs from legacy controls")
    if dataset_identity_sha256 != DATASET_IDENTITY_SHA256:
        raise ValueError("pilot dataset identity differs")
    if (
        len(gpu_inventory) != 1
        or int(gpu_inventory[0].get("memory_used_mib", -1)) > 16
        or int(gpu_inventory[0].get("utilization_percent", -1)) > 5
        or gpu_compute_processes != []
        or conflicting_processes != []
        or output_root_absent is not True
        or execution_lock_free is not True
        or int(free_bytes) < MIN_FREE_BYTES
    ):
        raise ValueError("live prelaunch exclusivity or capacity gate failed")
    authorization = _validate_execution_authorization(authorization_record)
    return {
        "schema": EXECUTION_GATE_SCHEMA,
        "role": "generation_matched_min_snr_training_pilot_execution_gate",
        "status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation": identity,
        "gate_builder_git": git,
        "authorization_record": copy.deepcopy(dict(authorization)),
        "live_prelaunch": {
            "runtime_environment_sha256": runtime_environment_sha256,
            "dataset_identity_sha256": dataset_identity_sha256,
            "gpu_inventory": [copy.deepcopy(dict(row)) for row in gpu_inventory],
            "gpu_compute_processes": [],
            "conflicting_processes": [],
            "output_root_absent": True,
            "execution_lock_free": True,
            "free_bytes": int(free_bytes),
            "minimum_free_bytes": MIN_FREE_BYTES,
        },
        "execution_contract": {
            "output_root": prepared["output_root"],
            "methods": list(METHODS),
            "gamma": GAMMA,
            "stop_step": PILOT_STEP,
            "scheduler_horizon_steps": SCHEDULER_HORIZON,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "images_seen_per_method": IMAGES_PER_METHOD,
            "evaluation": copy.deepcopy(prepared["evaluation_contract"]),
            "success_thresholds": copy.deepcopy(
                prepared["precommitted_success_thresholds"]
            ),
        },
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }


def validate_execution_gate(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_preparation(preparation)
    expected_identity = normalize_identity(preparation_identity, label="preparation")
    if (
        report.get("schema") != EXECUTION_GATE_SCHEMA
        or report.get("status") != "pass"
        or report.get("terminal_status") != "hold"
        or report.get("generation_advantage_proven") is not False
        or report.get("preparation") != expected_identity
        or report.get("authorization_boundary") != EXECUTION_BOUNDARY
    ):
        raise ValueError("Min-SNR pilot execution gate contract differs")
    contract = _mapping(report.get("execution_contract"), label="execution contract")
    live = _mapping(report.get("live_prelaunch"), label="live prelaunch")
    _validate_execution_authorization(report.get("authorization_record"))
    if (
        contract.get("output_root") != prepared["output_root"]
        or contract.get("methods") != list(METHODS)
        or float(contract.get("gamma", -1.0)) != GAMMA
        or int(contract.get("stop_step", -1)) != PILOT_STEP
        or int(contract.get("scheduler_horizon_steps", -1)) != SCHEDULER_HORIZON
        or int(contract.get("effective_batch_size", -1)) != EFFECTIVE_BATCH_SIZE
        or contract.get("evaluation") != prepared["evaluation_contract"]
        or contract.get("success_thresholds")
        != prepared["precommitted_success_thresholds"]
        or live.get("runtime_environment_sha256")
        != LEGACY_RUNTIME_ENVIRONMENT_SHA256
        or live.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256
        or live.get("gpu_compute_processes") != []
        or live.get("conflicting_processes") != []
        or live.get("output_root_absent") is not True
        or live.get("execution_lock_free") is not True
        or int(live.get("free_bytes", -1)) < MIN_FREE_BYTES
    ):
        raise ValueError("Min-SNR pilot execution gate details differ")
    normalize_git(
        report.get("gate_builder_git"),
        label="execution gate builder",
        expected_branch=PILOT_BRANCH,
    )
    return copy.deepcopy(dict(report))


def _finite_metric(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_training_audit(value: Any, *, method: str) -> dict[str, Any]:
    report = _mapping(value, label=f"pilot {method} training audit")
    training = _mapping(report.get("training"), label=f"pilot {method} training")
    checkpoint = _mapping(
        report.get("checkpoint"), label=f"pilot {method} checkpoint"
    )
    if (
        report.get("schema_version") != 1
        or report.get("role")
        != "generation_matched_min_snr_pilot_training_physical_audit"
        or report.get("status") != "pass"
        or report.get("method") != method
        or int(training.get("completed_steps", -1)) != PILOT_STEP
        or int(training.get("target_steps", -1)) != SCHEDULER_HORIZON
        or training.get("training_complete") is not False
        or int(training.get("effective_batch_size", -1)) != EFFECTIVE_BATCH_SIZE
        or int(training.get("samples_seen", -1)) != IMAGES_PER_METHOD
        or training.get("strictly_increasing_metrics") is not True
        or training.get("samples_seen_binding_verified") is not True
        or training.get("min_snr_metrics_verified") is not True
        or checkpoint.get("physical_sha256_verified") is not True
        or checkpoint.get("latest_exact_binding") is not True
    ):
        raise ValueError(f"pilot {method} training audit differs")
    return {
        "completed_steps": PILOT_STEP,
        "samples_seen": IMAGES_PER_METHOD,
        "checkpoint": normalize_identity(checkpoint, label=f"pilot {method} checkpoint"),
    }


def validate_sampling_preflight(
    value: Any,
    *,
    arm: str,
    expected_prefix_budget: int,
) -> dict[str, Any]:
    preflight = _mapping(value, label=f"{arm} sampling preflight")
    preflight_request = _mapping(
        preflight.get("request"), label=f"{arm} sampling preflight request"
    )
    preflight_result = _mapping(
        preflight.get("result"), label=f"{arm} sampling preflight result"
    )
    preflight_git = normalize_git(
        preflight.get("git"),
        label=f"{arm} sampling preflight Git",
        expected_branch=PILOT_BRANCH,
    )
    if (
        preflight.get("schema_version") != 1
        or preflight.get("status") != "passed"
        or int(preflight.get("checkpoint_step", -1)) != PILOT_STEP
        or preflight.get("weights") != "ema"
        or preflight.get("requested_weights") != "ema"
        or int(preflight_request.get("batch_size", -1)) != 32
        or int(preflight_request.get("prefix_budget", -1))
        != expected_prefix_budget
        or preflight_request.get("precision") != "bf16"
        or float(preflight_request.get("guidance_scale", -1.0)) != 1.5
        or float(preflight_request.get("guidance_rescale", -1.0)) != 0.0
        or preflight_request.get("cfg_batch_mode") != "batched"
        or int(preflight_request.get("warmup_forwards", -1)) != 0
        or int(preflight_request.get("measured_forwards", -1)) != 1
        or preflight_result.get("output_finite") is not True
        or preflight_result.get("output_shape") != [32, 3, 256, 256]
    ):
        raise ValueError(f"{arm} sampling preflight protocol differs")
    return {
        "checkpoint_sha256": _hex(
            preflight.get("checkpoint_sha256", ""), length=64
        ),
        "runtime_environment_sha256": _hex(
            preflight.get("runtime_environment_sha256", ""), length=64
        ),
        "git": preflight_git,
    }


def validate_evaluation_arm(
    value: Any,
    *,
    arm: str,
    expected_prefix_budget: int,
) -> dict[str, Any]:
    sources = _mapping(value, label=f"{arm} arm")
    preflight = validate_sampling_preflight(
        sources.get("sampling_preflight"),
        arm=arm,
        expected_prefix_budget=expected_prefix_budget,
    )
    generation = _mapping(sources.get("generation"), label=f"{arm} generation")
    class_fidelity = _mapping(
        sources.get("class_fidelity"), label=f"{arm} class fidelity"
    )
    checkpoint_eval = _mapping(
        sources.get("checkpoint_eval"), label=f"{arm} checkpoint evaluation"
    )
    generation_metrics = _mapping(
        generation.get("metrics"), label=f"{arm} generation metrics"
    )
    counts = _mapping(generation.get("counts"), label=f"{arm} counts")
    provenance = _mapping(
        generation.get("sample_provenance"), label=f"{arm} sample provenance"
    )
    sampling = _mapping(provenance.get("sampling"), label=f"{arm} sampling")
    class_metrics = _mapping(
        class_fidelity.get("metrics"), label=f"{arm} class metrics"
    )
    if (
        generation.get("schema_version") != 3
        or generation.get("status") != "completed"
        or int(counts.get("generated_image_count", -1)) != SAMPLE_COUNT
        or int(counts.get("real_image_count", -1)) != 50_000
        or int(provenance.get("checkpoint_step", -1)) != PILOT_STEP
        or provenance.get("weights") != "ema"
        or int(provenance.get("selected_prefix_budget", -1))
        != expected_prefix_budget
        or sampling.get("sampler") != "ddim"
        or int(sampling.get("sample_steps", -1)) != SAMPLE_STEPS
        or int(sampling.get("num_samples", -1)) != SAMPLE_COUNT
        or int(sampling.get("seed", -1)) != SAMPLE_SEED
        or int(sampling.get("start_index", -1)) != 0
        or float(sampling.get("guidance_scale", -1.0)) != 1.5
        or float(sampling.get("guidance_rescale", -1.0)) != 0.0
        or sampling.get("cfg_batch_mode") != "batched"
        or sampling.get("precision") != "bf16"
        or sampling.get("class_schedule") != "balanced_modulo"
        or class_fidelity.get("status") != "completed"
        or int(class_metrics.get("sample_count", -1)) != SAMPLE_COUNT
        or checkpoint_eval.get("status") != "completed"
        or int(checkpoint_eval.get("checkpoint_step", -1)) != PILOT_STEP
        or checkpoint_eval.get("weights") != "ema"
    ):
        raise ValueError(f"{arm} evaluation protocol differs")
    row = {
        "fid": _finite_metric(
            generation_metrics.get("frechet_inception_distance"), label=f"{arm} FID"
        ),
        "precision": _finite_metric(
            generation_metrics.get("precision"), label=f"{arm} precision"
        ),
        "recall": _finite_metric(
            generation_metrics.get("recall"), label=f"{arm} recall"
        ),
        "top1": _finite_metric(
            class_metrics.get("top1_accuracy"), label=f"{arm} top1"
        ),
        "top5": _finite_metric(
            class_metrics.get("top5_accuracy"), label=f"{arm} top5"
        ),
        "predicted_class_fraction": _finite_metric(
            class_metrics.get("predicted_class_fraction"),
            label=f"{arm} predicted-class fraction",
        ),
        "normalized_predicted_class_entropy": _finite_metric(
            class_metrics.get("normalized_predicted_class_entropy"),
            label=f"{arm} normalized entropy",
        ),
        "checkpoint_sha256": str(provenance.get("checkpoint_sha256", "")),
        "sample_set_sha256": str(provenance.get("sample_set_sha256", "")),
        "real_set_sha256": str(
            _mapping(generation.get("real_set"), label=f"{arm} real set").get(
                "sha256", ""
            )
        ),
    }
    _hex(row["checkpoint_sha256"], length=64)
    _hex(row["sample_set_sha256"], length=64)
    _hex(row["real_set_sha256"], length=64)
    if preflight["checkpoint_sha256"] != row["checkpoint_sha256"]:
        raise ValueError(f"{arm} sampling preflight checkpoint differs")
    row["sampling_preflight"] = preflight
    if row["fid"] <= 0.0 or any(
        not 0.0 <= row[field] <= 1.0
        for field in (
            "precision",
            "recall",
            "top1",
            "top5",
            "predicted_class_fraction",
            "normalized_predicted_class_entropy",
        )
    ):
        raise ValueError(f"{arm} evaluation metrics are out of range")
    if expected_prefix_budget > 1:
        metrics = _mapping(
            checkpoint_eval.get("metrics"), label=f"{arm} mechanism metrics"
        )
        config = _mapping(
            checkpoint_eval.get("config"), label=f"{arm} checkpoint config"
        )
        model = _mapping(config.get("model"), label=f"{arm} model config")
        ratios = [
            _finite_metric(value, label=f"{arm} component energy")
            for value in metrics.get("component_energy_ratio_per_sample_mean", [])
        ]
        strides = [int(value) for value in model.get("token_spatial_strides", [])]
        token_count = int(model.get("token_count", -1))
        if (
            token_count != expected_prefix_budget
            or len(ratios) != token_count
            or len(strides) != token_count
            or 1 not in strides
            or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-5)
        ):
            raise ValueError(f"{arm} mechanism energy evidence differs")
        coarse_count = strides.index(1)
        row["mechanism"] = {
            "zero_token_max_abs": _finite_metric(
                metrics.get("zero_token_max_abs"), label=f"{arm} zero token"
            ),
            "shuffled_to_ordered_endpoint_ratio": _finite_metric(
                metrics.get("shuffled_to_ordered_endpoint_ratio"),
                label=f"{arm} shuffle ratio",
            ),
            "ordered_rank_by_path_auc": int(
                metrics.get("ordered_rank_by_path_auc", -1)
            ),
            "coarse_token_energy_ratio": sum(ratios[:coarse_count]),
            "token_spatial_strides": strides,
        }
    return row


def build_result(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_gate: Mapping[str, Any],
    execution_gate_identity: Mapping[str, Any],
    pilot_training_audits: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    builder_git: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_preparation(preparation)
    prep_identity = normalize_identity(preparation_identity, label="preparation")
    validate_execution_gate(
        execution_gate,
        preparation=preparation,
        preparation_identity=prep_identity,
    )
    gate_identity = normalize_identity(execution_gate_identity, label="execution gate")
    if execution_gate.get("preparation") != prep_identity:
        raise ValueError("execution gate does not bind the result preparation")
    if set(pilot_training_audits) != set(METHODS):
        raise ValueError("pilot training audit set differs")
    training = {
        method: _validate_training_audit(pilot_training_audits[method], method=method)
        for method in METHODS
    }
    expected_arms = {
        f"{recipe}_{method}"
        for recipe in ("legacy_gamma0", "pilot_gamma5")
        for method in METHODS
    }
    if set(arms) != expected_arms:
        raise ValueError("Min-SNR pilot evaluation arm set differs")
    rows = {
        name: validate_evaluation_arm(
            arms[name],
            arm=name,
            expected_prefix_budget=8 if name.endswith("cofitok") else 1,
        )
        for name in sorted(expected_arms)
    }
    real_sets = {str(row["real_set_sha256"]) for row in rows.values()}
    if len(real_sets) != 1:
        raise ValueError("Min-SNR pilot arms use different real sets")
    if any(
        row["sampling_preflight"]["git"].get("revision")
        != prepared["builder_git"].get("revision")
        or row["sampling_preflight"]["git"].get("branch")
        != prepared["builder_git"].get("branch")
        or row["sampling_preflight"]["git"].get("tracked_dirty") is not False
        for row in rows.values()
    ):
        raise ValueError("Min-SNR pilot sampling preflight source differs")
    normalized_sources = {
        str(name): normalize_identity(identity, label=str(name))
        for name, identity in source_identities.items()
    }
    expected_source_names = {
        "preparation",
        "execution_gate",
        "pilot_cofitok_training_audit",
        "pilot_dense_training_audit",
        *{
            f"{arm}_{kind}"
            for arm in expected_arms
            for kind in (
                "generation",
                "class_fidelity",
                "checkpoint_eval",
                "sampling_preflight",
            )
        },
    }
    if set(normalized_sources) != expected_source_names:
        raise ValueError("Min-SNR pilot result source set differs")
    if normalized_sources["preparation"] != prep_identity:
        raise ValueError("result preparation identity differs")
    if normalized_sources["execution_gate"] != gate_identity:
        raise ValueError("result execution gate identity differs")
    thresholds = _mapping(
        prepared.get("precommitted_success_thresholds"), label="success thresholds"
    )
    comparisons: dict[str, Any] = {}
    all_method_gates_pass = True
    for method in METHODS:
        legacy = rows[f"legacy_gamma0_{method}"]
        pilot = rows[f"pilot_gamma5_{method}"]
        fid_improvement = (legacy["fid"] - pilot["fid"]) / legacy["fid"]
        gates = {
            "relative_fid_improvement": fid_improvement
            >= float(thresholds["both_methods_min_relative_fid_improvement"]),
            "precision_non_regression": pilot["precision"] - legacy["precision"]
            >= -float(thresholds["both_methods_max_precision_regression"]),
            "recall_non_regression": pilot["recall"] - legacy["recall"]
            >= -float(thresholds["both_methods_max_recall_regression"]),
            "class_top1_non_regression": pilot["top1"] - legacy["top1"]
            >= -float(thresholds["both_methods_max_class_top1_regression"]),
            "class_top5_non_regression": pilot["top5"] - legacy["top5"]
            >= -float(thresholds["both_methods_max_class_top5_regression"]),
            "absolute_class_top1": pilot["top1"]
            >= float(thresholds["min_class_top1"]),
            "absolute_class_top5": pilot["top5"]
            >= float(thresholds["min_class_top5"]),
            "predicted_class_fraction": pilot["predicted_class_fraction"]
            >= float(thresholds["min_predicted_class_fraction"]),
            "predicted_class_entropy": pilot["normalized_predicted_class_entropy"]
            >= float(thresholds["min_normalized_predicted_entropy"]),
        }
        passed = all(gates.values())
        all_method_gates_pass = all_method_gates_pass and passed
        comparisons[method] = {
            "legacy_gamma0": copy.deepcopy(legacy),
            "pilot_gamma5": copy.deepcopy(pilot),
            "deltas": {
                "relative_fid_improvement": fid_improvement,
                "precision": pilot["precision"] - legacy["precision"],
                "recall": pilot["recall"] - legacy["recall"],
                "top1": pilot["top1"] - legacy["top1"],
                "top5": pilot["top5"] - legacy["top5"],
            },
            "gates": gates,
            "status": "pass" if passed else "fail",
        }
    pilot_mechanism = rows["pilot_gamma5_cofitok"]["mechanism"]
    mechanism_gates = {
        "zero_token": pilot_mechanism["zero_token_max_abs"]
        <= float(thresholds["cofitok_zero_token_max_abs"]),
        "shuffle": pilot_mechanism["shuffled_to_ordered_endpoint_ratio"]
        >= float(thresholds["cofitok_min_shuffled_to_ordered_endpoint_ratio"]),
        "ordered_rank": pilot_mechanism["ordered_rank_by_path_auc"]
        == int(thresholds["cofitok_ordered_rank_by_path_auc"]),
        "coarse_energy": pilot_mechanism["coarse_token_energy_ratio"]
        >= float(thresholds["cofitok_min_coarse_token_energy_ratio"]),
    }
    mechanism_pass = all(mechanism_gates.values())
    candidate = all_method_gates_pass and mechanism_pass
    git = normalize_git(builder_git, label="result builder", expected_branch=PILOT_BRANCH)
    if git != prepared["builder_git"]:
        raise ValueError("result builder differs from preparation source")
    return {
        "schema": RESULT_SCHEMA,
        "role": "generation_matched_min_snr_training_pilot_result",
        "status": "completed",
        "scientific_status": "screening_only",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "selection_status": (
            "shared_min_snr_candidate_for_separately_gated_100k_continuation"
            if candidate
            else "no_shared_min_snr_candidate_at_50k"
        ),
        "builder_git": git,
        "source_identities": normalized_sources,
        "training": training,
        "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        "precommitted_success_thresholds": copy.deepcopy(dict(thresholds)),
        "comparisons": comparisons,
        "cofitok_mechanism": {
            "metrics": copy.deepcopy(pilot_mechanism),
            "gates": mechanism_gates,
            "status": "pass" if mechanism_pass else "fail",
        },
        "recommended_next_stage": {
            "id": (
                "prepare_separately_gated_matched_100k_min_snr_continuation"
                if candidate
                else "stop_min_snr_route_and_reassess_training_objective"
            ),
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "continuation_beyond_50000_allowed": False,
        },
        "claim_policy": copy.deepcopy(prepared["claim_policy"]),
        "authorization_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }


def validate_result(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema") != RESULT_SCHEMA
        or report.get("status") != "completed"
        or report.get("scientific_status") != "screening_only"
        or report.get("terminal_status") != "hold"
        or report.get("generation_advantage_proven") is not False
        or report.get("selection_status")
        not in {
            "shared_min_snr_candidate_for_separately_gated_100k_continuation",
            "no_shared_min_snr_candidate_at_50k",
        }
        or report.get("authorization_boundary") != RESULT_BOUNDARY
    ):
        raise ValueError("Min-SNR pilot result contract differs")
    stage = _mapping(report.get("recommended_next_stage"), label="next stage")
    if (
        stage.get("execution_ready") is not False
        or stage.get("gpu_execution_allowed") is not False
        or stage.get("continuation_beyond_50000_allowed") is not False
    ):
        raise ValueError("Min-SNR pilot result authorizes continuation")
    return copy.deepcopy(dict(report))
