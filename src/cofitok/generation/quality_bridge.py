from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    validate_dataset_provenance,
)
from cofitok.generation.protocol import sampling_protocol_contract
from cofitok.generation_class_fidelity import (
    validate_class_fidelity_qualification,
)


QUALITY_BRIDGE_SCHEMA_VERSION = 1
QUALITY_BRIDGE_ROLE = "stability_full_data_quality_bridge_preparation"
QUALITY_BRIDGE_RECIPE_STAGE = "stability_quality_bridge"
QUALITY_BRIDGE_DATASET = "imagenet_256"
QUALITY_BRIDGE_STEPS = 100_000
QUALITY_BRIDGE_EFFECTIVE_BATCH = 64
QUALITY_BRIDGE_MILESTONES = (50_000, 100_000)
QUALITY_BRIDGE_TERMINAL_SAMPLES = 10_000
QUALITY_BRIDGE_SOURCE_DATASET = "imagenet_256_10pct"
QUALITY_BRIDGE_SOURCE_STEPS = 50_000
QUALITY_BRIDGE_APPROVAL_SCHEMA_VERSION = 1
QUALITY_BRIDGE_APPROVAL_ROLE = (
    "stability_full_data_quality_bridge_execution_approval"
)
QUALITY_BRIDGE_APPROVAL_SCOPE = "stability_quality_bridge_100k_execution_only"
QUALITY_BRIDGE_LAUNCH_RECEIPT_SCHEMA_VERSION = 1
QUALITY_BRIDGE_LAUNCH_RECEIPT_ROLE = (
    "stability_full_data_quality_bridge_launch_receipt"
)
QUALITY_BRIDGE_RESULT_SCHEMA_VERSION = 1
QUALITY_BRIDGE_RESULT_ROLE = "stability_full_data_quality_bridge_result"
QUALITY_BRIDGE_STORAGE_STAGE = "stability_full_data_quality_bridge_100k_execution"
QUALITY_BRIDGE_MIN_STORAGE_SAMPLES = 28_192
QUALITY_BRIDGE_MIN_CHECKPOINT_COUNT = 10
QUALITY_BRIDGE_SOURCE_NAMES = {
    "preparation",
    "execution_approval",
    "cofitok_config",
    "dense_config",
    "config_validation",
    "storage_capacity",
    "runtime_selection",
}
QUALITY_BRIDGE_THRESHOLD_NAMES = {
    "max_absolute_fid",
    "max_endpoint_regression",
    "max_fid_regression",
    "max_precision_regression",
    "max_recall_regression",
    "min_coarse_token_energy_ratio",
    "min_precision",
    "min_recall",
    "min_samples",
}


PREPARATION_AUTHORIZATION_BOUNDARY = {
    "quality_bridge_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "new_gate_required": True,
    "explicit_execution_approval_required": True,
}


EXECUTION_APPROVAL_BOUNDARY = {
    "quality_bridge_execution_allowed": True,
    "scope_limited_to_quality_bridge": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "scaling_authorization_created": False,
    "report_is_promotion_gate": False,
}


LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY = {
    "quality_bridge_execution_authorized": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "scaling_authorization_created": False,
    "report_is_promotion_gate": False,
    "new_gate_required": True,
}


RESULT_AUTHORIZATION_BOUNDARY = {
    "quality_bridge_evidence_complete": True,
    "quality_bridge_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "new_gate_required": True,
}


def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _valid_hex_digest(value: Any, *, length: int = 64) -> bool:
    text = str(value)
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text
    )


def _source_identity(identity: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = str(identity.get("path", ""))
    bytes_count = int(identity.get("bytes", 0))
    sha256 = str(identity.get("sha256", ""))
    if (
        not path
        or bytes_count < 1
        or not _valid_hex_digest(sha256)
    ):
        raise ValueError(f"{label} identity is invalid")
    return {"path": path, "bytes": bytes_count, "sha256": sha256}


def _git_identity(
    value: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    label: str,
) -> dict[str, Any]:
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if dict(value) != expected:
        raise ValueError(f"{label} Git identity differs")
    return expected


def _finite_metric(
    report: Mapping[str, Any],
    name: str,
    *,
    label: str,
) -> float:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} metrics are missing")
    return _finite_float(metrics.get(name), label=f"{label} {name}")


def validate_quality_bridge_preparation(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", -1)) != QUALITY_BRIDGE_SCHEMA_VERSION
        or report.get("status") != "prepared"
        or report.get("role") != QUALITY_BRIDGE_ROLE
    ):
        raise ValueError("quality bridge preparation contract differs")
    selection = report.get("selection")
    evaluation = report.get("evaluation_contract")
    matched = report.get("matched_training_contract")
    hold = report.get("source_quality_hold")
    if not all(
        isinstance(value, Mapping)
        for value in (selection, evaluation, matched, hold)
    ):
        raise ValueError("quality bridge preparation evidence is incomplete")
    if (
        selection.get("dataset") != QUALITY_BRIDGE_DATASET
        or int(selection.get("steps", -1)) != QUALITY_BRIDGE_STEPS
        or int(selection.get("effective_batch_size", -1))
        != QUALITY_BRIDGE_EFFECTIVE_BATCH
        or int(selection.get("base_channels", -1)) != 128
        or selection.get("milestone_steps") != list(QUALITY_BRIDGE_MILESTONES)
        or selection.get("capacity_change_allowed") is not False
        or selection.get("qualified_model_and_loss_recipe_preserved") is not True
    ):
        raise ValueError("quality bridge preparation selection differs")
    milestones = evaluation.get("milestones")
    terminal = evaluation.get("terminal")
    if not isinstance(milestones, Mapping) or not isinstance(terminal, Mapping):
        raise ValueError("quality bridge evaluation contract is incomplete")
    if (
        milestones.get("steps") != list(QUALITY_BRIDGE_MILESTONES)
        or milestones.get("weights") != "ema"
        or milestones.get("sampler") != "ddim"
        or int(milestones.get("sample_steps", -1)) != 50
        or int(milestones.get("samples_per_method", -1)) != 2_048
        or int(milestones.get("mechanism_images", -1)) != 256
        or terminal.get("step") != QUALITY_BRIDGE_STEPS
        or terminal.get("weights") != "ema"
        or terminal.get("sampler") != "ddim"
        or int(terminal.get("sample_steps", -1)) != 100
        or int(terminal.get("samples_per_method", -1))
        != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or terminal.get("skip_precision_recall_allowed") is not False
        or terminal.get("matched_fixed_random_stream_required") is not True
    ):
        raise ValueError("quality bridge evaluation protocol differs")
    if (
        hold.get("status") != "validated_hold"
        or hold.get("failed_gates") != ["absolute_fid_quality"]
        or hold.get("source_dataset") != QUALITY_BRIDGE_SOURCE_DATASET
        or int(hold.get("source_steps", -1)) != QUALITY_BRIDGE_SOURCE_STEPS
    ):
        raise ValueError("quality bridge source hold differs")
    recipe = matched.get("config_validation", {}).get("training_recipe", {})
    if (
        matched.get("recipe_stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or recipe.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or recipe.get("valid") is not True
    ):
        raise ValueError("quality bridge matched training contract differs")
    if report.get("authorization_boundary") != PREPARATION_AUTHORIZATION_BOUNDARY:
        raise ValueError("quality bridge preparation authorization boundary differs")
    return {
        "dataset": QUALITY_BRIDGE_DATASET,
        "steps": QUALITY_BRIDGE_STEPS,
        "effective_batch_size": QUALITY_BRIDGE_EFFECTIVE_BATCH,
        "milestone_steps": list(QUALITY_BRIDGE_MILESTONES),
        "terminal_samples": QUALITY_BRIDGE_TERMINAL_SAMPLES,
        "source_hold": copy.deepcopy(dict(hold)),
    }


def validate_quality_bridge_execution_approval(
    approval: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    if (
        int(approval.get("schema_version", -1))
        != QUALITY_BRIDGE_APPROVAL_SCHEMA_VERSION
        or approval.get("role") != QUALITY_BRIDGE_APPROVAL_ROLE
        or approval.get("status") != "approved"
        or approval.get("scope") != QUALITY_BRIDGE_APPROVAL_SCOPE
    ):
        raise ValueError("quality bridge execution approval contract differs")
    if approval.get("preparation") != _source_identity(
        preparation_identity,
        label="quality bridge preparation",
    ):
        raise ValueError("quality bridge execution approval binds another preparation")
    git = approval.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("quality bridge execution approval lacks Git identity")
    _git_identity(
        git,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        label="quality bridge execution approval",
    )
    selection = approval.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("dataset") != QUALITY_BRIDGE_DATASET
        or int(selection.get("steps", -1)) != QUALITY_BRIDGE_STEPS
        or selection.get("milestone_steps") != list(QUALITY_BRIDGE_MILESTONES)
        or int(selection.get("effective_batch_size", -1))
        != QUALITY_BRIDGE_EFFECTIVE_BATCH
    ):
        raise ValueError("quality bridge execution approval selection differs")
    record = approval.get("approval_record")
    if (
        not isinstance(record, Mapping)
        or not str(record.get("approved_by", "")).strip()
        or not str(record.get("approved_at", "")).strip()
    ):
        raise ValueError("quality bridge execution approval record is incomplete")
    if str(approval.get("output_root", "")) != expected_output_root:
        raise ValueError("quality bridge execution approval output root differs")
    if approval.get("authorization_boundary") != EXECUTION_APPROVAL_BOUNDARY:
        raise ValueError("quality bridge execution approval boundary differs")
    return {
        "scope": QUALITY_BRIDGE_APPROVAL_SCOPE,
        "git": dict(git),
        "output_root": expected_output_root,
        "approval_record": copy.deepcopy(dict(record)),
        "authorization_boundary": copy.deepcopy(EXECUTION_APPROVAL_BOUNDARY),
    }


def _validate_storage_capacity(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_storage_path: str,
) -> dict[str, Any]:
    filesystem = report.get("filesystem")
    plan = report.get("plan")
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != QUALITY_BRIDGE_STORAGE_STAGE
        or report.get("status") != "pass"
        or report.get("git") != dict(expected_git)
        or not isinstance(filesystem, Mapping)
        or not isinstance(plan, Mapping)
        or str(filesystem.get("path", "")) != expected_storage_path
        or int(plan.get("sample_count", -1)) < QUALITY_BRIDGE_MIN_STORAGE_SAMPLES
        or int(plan.get("checkpoint_count", -1))
        < QUALITY_BRIDGE_MIN_CHECKPOINT_COUNT
        or int(report.get("headroom_bytes", -1)) < 0
    ):
        raise ValueError("quality bridge launch storage evidence differs")
    return {
        "storage_path": expected_storage_path,
        "free_bytes": int(filesystem["free_bytes"]),
        "required_free_bytes": int(plan["required_free_bytes"]),
        "headroom_bytes": int(report["headroom_bytes"]),
    }


def _validate_runtime_selection(
    report: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    expected_config_sha256: Mapping[str, str],
    expected_run_dirs: list[str],
    expected_benchmark_root: str,
) -> dict[str, Any]:
    selected = report.get("selected")
    lock = report.get("selection_lock")
    if not isinstance(selected, Mapping) or not isinstance(lock, Mapping):
        raise ValueError("quality bridge runtime selection is incomplete")
    micro_batch = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    if (
        report.get("status") != "selected"
        or report.get("git_revision") != expected_git["revision"]
        or report.get("config_sha256") != dict(expected_config_sha256)
        or str(report.get("benchmark_root", "")) != expected_benchmark_root
        or micro_batch < 1
        or accumulation < 1
        or micro_batch * accumulation != QUALITY_BRIDGE_EFFECTIVE_BATCH
        or lock.get("training_run_dirs") != expected_run_dirs
        or int(lock.get("training_target_steps", -1)) != QUALITY_BRIDGE_STEPS
        or int(lock.get("expected_effective_batch_size", -1))
        != QUALITY_BRIDGE_EFFECTIVE_BATCH
        or lock.get("git") != dict(expected_git)
        or lock.get("config_sha256") != dict(expected_config_sha256)
        or str(lock.get("benchmark_root", "")) != expected_benchmark_root
    ):
        raise ValueError("quality bridge runtime selection contract differs")
    environment_sha = str(report.get("runtime_environment_sha256", ""))
    if len(environment_sha) != 64:
        raise ValueError("quality bridge runtime environment identity is malformed")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "runtime_environment_sha256": environment_sha,
    }


def build_quality_bridge_launch_receipt(
    *,
    preparation: Mapping[str, Any],
    execution_approval: Mapping[str, Any],
    config_validation: Mapping[str, Any],
    storage_capacity: Mapping[str, Any],
    runtime_selection: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
    output_root: str,
    storage_path: str,
    training_run_dirs: list[str],
    benchmark_root: str,
    training_state_absent_at_launch: bool,
) -> dict[str, Any]:
    if set(source_identities) != QUALITY_BRIDGE_SOURCE_NAMES:
        raise ValueError("quality bridge launch receipt source set differs")
    normalized_sources = {
        name: _source_identity(identity, label=name)
        for name, identity in source_identities.items()
    }
    validate_quality_bridge_preparation(preparation)
    expected_git = _git_identity(
        {"revision": expected_revision, "branch": expected_branch, "tracked_dirty": False},
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        label="quality bridge launch",
    )
    approval = validate_quality_bridge_execution_approval(
        execution_approval,
        preparation_identity=normalized_sources["preparation"],
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_output_root=output_root,
    )
    if config_validation != preparation.get("matched_training_contract", {}).get(
        "config_validation"
    ):
        raise ValueError("quality bridge launch config validation differs from preparation")
    if (
        config_validation.get("status") != "pass"
        or config_validation.get("mismatches") != []
        or config_validation.get("training_recipe", {}).get("stage")
        != QUALITY_BRIDGE_RECIPE_STAGE
        or config_validation.get("training_recipe", {}).get("valid") is not True
    ):
        raise ValueError("quality bridge launch config validation is invalid")
    expected_config_sha256 = {
        "cofitok": normalized_sources["cofitok_config"]["sha256"],
        "dense_identity": normalized_sources["dense_config"]["sha256"],
    }
    if (
        preparation.get("matched_training_contract", {}).get("cofitok_config")
        != normalized_sources["cofitok_config"]
        or preparation.get("matched_training_contract", {}).get("dense_config")
        != normalized_sources["dense_config"]
    ):
        raise ValueError("quality bridge launch config identities differ from preparation")
    if len(training_run_dirs) != 2 or len(set(training_run_dirs)) != 2:
        raise ValueError("quality bridge requires exactly two distinct run directories")
    storage = _validate_storage_capacity(
        storage_capacity,
        expected_git=expected_git,
        expected_storage_path=storage_path,
    )
    runtime = _validate_runtime_selection(
        runtime_selection,
        expected_git=expected_git,
        expected_config_sha256=expected_config_sha256,
        expected_run_dirs=training_run_dirs,
        expected_benchmark_root=benchmark_root,
    )
    if training_state_absent_at_launch is not True:
        raise ValueError("quality bridge launch receipt requires absent initial training state")
    return {
        "schema_version": QUALITY_BRIDGE_LAUNCH_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "role": QUALITY_BRIDGE_LAUNCH_RECEIPT_ROLE,
        "stage": QUALITY_BRIDGE_RECIPE_STAGE,
        "git": expected_git,
        "source_reports": normalized_sources,
        "approval": approval,
        "selection": validate_quality_bridge_preparation(preparation),
        "runtime_selection": runtime,
        "storage_capacity": storage,
        "output_root": output_root,
        "storage_path": storage_path,
        "training_run_dirs": training_run_dirs,
        "benchmark_root": benchmark_root,
        "training_state_absent_at_launch": True,
        "authorization_boundary": copy.deepcopy(
            LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
        ),
    }


def validate_quality_bridge_launch_receipt_contract(
    report: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", -1))
        != QUALITY_BRIDGE_LAUNCH_RECEIPT_SCHEMA_VERSION
        or report.get("status") != "pass"
        or report.get("role") != QUALITY_BRIDGE_LAUNCH_RECEIPT_ROLE
        or report.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or report.get("training_state_absent_at_launch") is not True
    ):
        raise ValueError("quality bridge launch receipt contract differs")
    git = report.get("git")
    sources = report.get("source_reports")
    if not isinstance(git, Mapping) or not isinstance(sources, Mapping):
        raise ValueError("quality bridge launch receipt evidence is incomplete")
    if set(sources) != QUALITY_BRIDGE_SOURCE_NAMES:
        raise ValueError("quality bridge launch receipt source set differs")
    _git_identity(
        git,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        label="quality bridge launch receipt",
    )
    for name, identity in sources.items():
        _source_identity(identity, label=f"launch source {name}")
    boundary = report.get("authorization_boundary")
    if boundary != LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY:
        raise ValueError("quality bridge launch receipt authorization boundary differs")
    selection = report.get("selection")
    runtime = report.get("runtime_selection")
    storage = report.get("storage_capacity")
    approval = report.get("approval")
    training_run_dirs = report.get("training_run_dirs")
    output_root = str(report.get("output_root", ""))
    storage_path = str(report.get("storage_path", ""))
    benchmark_root = str(report.get("benchmark_root", ""))
    if (
        not isinstance(selection, Mapping)
        or selection.get("dataset") != QUALITY_BRIDGE_DATASET
        or int(selection.get("steps", -1)) != QUALITY_BRIDGE_STEPS
        or int(selection.get("effective_batch_size", -1))
        != QUALITY_BRIDGE_EFFECTIVE_BATCH
        or selection.get("milestone_steps") != list(QUALITY_BRIDGE_MILESTONES)
        or int(selection.get("terminal_samples", -1))
        != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or not isinstance(runtime, Mapping)
        or int(runtime.get("effective_batch_size", -1))
        != QUALITY_BRIDGE_EFFECTIVE_BATCH
        or not _valid_hex_digest(runtime.get("runtime_environment_sha256"))
        or not isinstance(storage, Mapping)
        or int(storage.get("headroom_bytes", -1)) < 0
        or str(storage.get("storage_path", "")) != storage_path
        or not isinstance(approval, Mapping)
        or approval.get("scope") != QUALITY_BRIDGE_APPROVAL_SCOPE
        or str(approval.get("output_root", "")) != output_root
        or approval.get("authorization_boundary") != EXECUTION_APPROVAL_BOUNDARY
        or not isinstance(training_run_dirs, list)
        or len(training_run_dirs) != 2
        or len(set(str(path) for path in training_run_dirs)) != 2
        or any(not PurePosixPath(str(path)).is_absolute() for path in training_run_dirs)
        or not PurePosixPath(output_root).is_absolute()
        or not PurePosixPath(storage_path).is_absolute()
        or not PurePosixPath(benchmark_root).is_absolute()
        or any(
            not PurePosixPath(str(path)).is_relative_to(PurePosixPath(output_root))
            for path in training_run_dirs
        )
        or not PurePosixPath(benchmark_root).is_relative_to(
            PurePosixPath(output_root)
        )
    ):
        raise ValueError("quality bridge launch receipt execution contract differs")
    return {
        "git": dict(git),
        "source_reports": copy.deepcopy(dict(sources)),
        "runtime_selection": copy.deepcopy(dict(runtime)),
        "training_run_dirs": list(training_run_dirs),
        "output_root": output_root,
        "storage_path": storage_path,
        "benchmark_root": benchmark_root,
        "authorization_boundary": copy.deepcopy(dict(boundary)),
    }


def _matched_sampling_protocol(sampling: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"prefix_budgets", "sample_set_digest"}
    return {
        key: copy.deepcopy(value)
        for key, value in sampling.items()
        if key not in ignored
    }


def _coarse_token_utilization(
    checkpoint_eval: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = checkpoint_eval.get("metrics")
    model = checkpoint_eval.get("config", {}).get("model", {})
    if not isinstance(metrics, Mapping) or not isinstance(model, Mapping):
        raise ValueError("CoFiTok terminal token-utilization evidence is missing")
    raw_ratios = metrics.get("component_energy_ratio_per_sample_mean")
    raw_strides = model.get("token_spatial_strides")
    try:
        ratios = [float(value) for value in raw_ratios]
        token_count = int(model.get("token_count", -1))
        strides = [int(value) for value in raw_strides]
    except (TypeError, ValueError):
        ratios = []
        strides = []
        token_count = -1
    if (
        token_count < 3
        or len(ratios) != token_count
        or len(strides) != token_count
        or any(value < 1 for value in strides)
        or 1 not in strides
        or any(not math.isfinite(value) or value < 0.0 for value in ratios)
        or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("CoFiTok terminal token-utilization evidence is invalid")
    coarse_token_count = strides.index(1)
    if (
        coarse_token_count < 1
        or any(value <= 1 for value in strides[:coarse_token_count])
        or any(value != 1 for value in strides[coarse_token_count:])
    ):
        raise ValueError("CoFiTok terminal stride partition is invalid")
    return {
        "source_metric": "component_energy_ratio_per_sample_mean",
        "partition_schema": "token_spatial_stride_suffix_v1",
        "token_count": token_count,
        "coarse_token_count": coarse_token_count,
        "full_resolution_tail_token_count": token_count - coarse_token_count,
        "token_spatial_strides": strides,
        "component_energy_ratios": ratios,
        "coarse_token_energy_ratio": sum(ratios[:coarse_token_count]),
    }


def _terminal_method_row(
    generation: Mapping[str, Any],
    checkpoint_eval: Mapping[str, Any],
    *,
    label: str,
    expected_prefix_budget: int,
    expected_random_orders: int,
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        int(generation.get("schema_version", -1)) != 3
        or generation.get("role") != "generation_directory_metrics_report"
        or generation.get("protocol") != "torch_fidelity_directory_metrics"
        or generation.get("status") != "completed"
        or generation.get("git") != dict(expected_git)
    ):
        raise ValueError(f"{label} terminal generation report contract differs")
    provenance = generation.get("sample_provenance")
    counts = generation.get("counts")
    parameters = generation.get("parameters")
    real_set = generation.get("real_set")
    if not all(
        isinstance(value, Mapping)
        for value in (provenance, counts, parameters, real_set)
    ):
        raise ValueError(f"{label} terminal generation provenance is incomplete")
    sampling = provenance.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(f"{label} terminal sampling protocol is missing")
    contract = sampling_protocol_contract(
        dict(sampling),
        stage="scaling",
        expected_num_train_timesteps=1_000,
    )
    if contract["valid"] is not True:
        raise ValueError(
            f"{label} terminal sampling protocol differs: "
            + ", ".join(contract["issues"])
        )
    if (
        int(counts.get("generated_image_count", -1))
        != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or int(provenance.get("checkpoint_step", -1)) != QUALITY_BRIDGE_STEPS
        or provenance.get("weights") != "ema"
        or int(provenance.get("selected_prefix_budget", -1))
        != expected_prefix_budget
        or provenance.get("git") != dict(expected_git)
        or sampling.get("prefix_budgets") != [expected_prefix_budget]
        or sampling.get("image_shape") != [3, 256, 256]
        or parameters.get("precision_recall_enabled") is not True
    ):
        raise ValueError(f"{label} terminal generation identity differs")
    if (
        not PurePosixPath(str(real_set.get("root", ""))).is_absolute()
        or not _valid_hex_digest(real_set.get("sha256"))
        or int(real_set.get("image_count", -1)) < QUALITY_BRIDGE_TERMINAL_SAMPLES
    ):
        raise ValueError(f"{label} terminal real-set identity is malformed")
    checkpoint_sha256 = str(provenance.get("checkpoint_sha256", ""))
    sample_set_sha256 = str(provenance.get("sample_set_sha256", ""))
    checkpoint_path = str(provenance.get("checkpoint", ""))
    integrity_manifest = str(provenance.get("checkpoint_integrity_manifest", ""))
    sampling_report_identity = _source_identity(
        provenance.get("report_identity", {}),
        label=f"{label} sampling report",
    )
    sampling_manifest_identity = _source_identity(
        provenance.get("manifest_identity", {}),
        label=f"{label} sampling manifest",
    )
    sampling_progress = provenance.get("sampling_progress")
    if not isinstance(sampling_progress, Mapping):
        raise ValueError(f"{label} terminal sampling progress is missing")
    sampling_progress_identity = _source_identity(
        sampling_progress.get("identity", {}),
        label=f"{label} sampling progress",
    )
    if (
        sampling_progress.get("status") != "completed"
        or int(sampling_progress.get("completed_samples", -1))
        != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or _finite_float(
            sampling_progress.get("cumulative_elapsed_seconds"),
            label=f"{label} sampling elapsed time",
        )
        <= 0.0
    ):
        raise ValueError(f"{label} terminal sampling progress is incomplete")
    runtime_environment_sha256 = str(
        generation.get("runtime_environment_sha256", "")
    )
    sampling_runtime_environment_sha256 = str(
        provenance.get("runtime_environment_sha256", "")
    )
    if (
        not _valid_hex_digest(checkpoint_sha256)
        or not _valid_hex_digest(sample_set_sha256)
        or not _valid_hex_digest(runtime_environment_sha256)
        or not _valid_hex_digest(sampling_runtime_environment_sha256)
        or not PurePosixPath(checkpoint_path).is_absolute()
        or not PurePosixPath(integrity_manifest).is_absolute()
        or not integrity_manifest.endswith(".integrity.json")
    ):
        raise ValueError(f"{label} terminal checkpoint/sample identity is malformed")
    fid = _finite_metric(generation, "frechet_inception_distance", label=label)
    inception = _finite_metric(generation, "inception_score_mean", label=label)
    precision = _finite_metric(generation, "precision", label=label)
    recall = _finite_metric(generation, "recall", label=label)
    if fid < 0.0 or inception <= 0.0 or not 0.0 <= precision <= 1.0 or not 0.0 <= recall <= 1.0:
        raise ValueError(f"{label} terminal quality metrics are outside their domains")

    request = checkpoint_eval.get("request")
    mechanism = checkpoint_eval.get("metrics")
    if (
        int(checkpoint_eval.get("schema_version", -1)) != 2
        or checkpoint_eval.get("role")
        != "generation_checkpoint_evaluation_report"
        or checkpoint_eval.get("status") != "completed"
        or checkpoint_eval.get("git") != dict(expected_git)
        or not isinstance(request, Mapping)
        or not isinstance(mechanism, Mapping)
        or int(request.get("num_images", -1)) != 256
        or int(request.get("timestep", -1)) != 500
        or int(request.get("random_orders", -1)) != expected_random_orders
        or request.get("weights") != "ema"
        or request.get("precision") != "bf16"
        or int(checkpoint_eval.get("checkpoint_step", -1))
        != QUALITY_BRIDGE_STEPS
        or checkpoint_eval.get("checkpoint_sha256") != checkpoint_sha256
        or checkpoint_eval.get("checkpoint_integrity_manifest")
        != integrity_manifest
        or checkpoint_eval.get("weights") != "ema"
        or int(mechanism.get("evaluated_images", -1)) != 256
    ):
        raise ValueError(f"{label} terminal mechanism evaluation differs")
    ordered = mechanism.get("orders", {}).get("ordered")
    if not isinstance(ordered, Mapping):
        raise ValueError(f"{label} terminal ordered mechanism row is missing")
    endpoint = _finite_float(
        ordered.get("endpoint_clean_mse"),
        label=f"{label} endpoint_clean_mse",
    )
    path_auc = _finite_float(
        ordered.get("prefix_path_mse_auc"),
        label=f"{label} prefix_path_mse_auc",
    )
    zero = _finite_float(
        mechanism.get("zero_token_max_abs"),
        label=f"{label} zero_token_max_abs",
    )
    shuffle_ratio = _finite_float(
        mechanism.get("shuffled_to_ordered_endpoint_ratio"),
        label=f"{label} shuffled_to_ordered_endpoint_ratio",
    )
    order_count = int(mechanism.get("order_count", -1))
    ordered_rank = int(mechanism.get("ordered_rank_by_path_auc", -1))
    if order_count < 1 or ordered_rank < 1 or ordered_rank > order_count:
        raise ValueError(f"{label} terminal order evidence is invalid")
    row = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_integrity_manifest": integrity_manifest,
        "checkpoint_step": QUALITY_BRIDGE_STEPS,
        "sample_set_sha256": sample_set_sha256,
        "sample_count": QUALITY_BRIDGE_TERMINAL_SAMPLES,
        "selected_prefix_budget": expected_prefix_budget,
        "weights": "ema",
        "sampling": copy.deepcopy(dict(sampling)),
        "sampling_runtime_environment_sha256": (
            sampling_runtime_environment_sha256
        ),
        "sampling_report": sampling_report_identity,
        "sampling_manifest": sampling_manifest_identity,
        "sampling_progress": sampling_progress_identity,
        "fid": fid,
        "inception_score": inception,
        "precision": precision,
        "recall": recall,
        "real_set": copy.deepcopy(dict(real_set)),
        "metrics_evaluator_git": copy.deepcopy(dict(generation["git"])),
        "metrics_runtime_environment_sha256": generation.get(
            "runtime_environment_sha256"
        ),
        "endpoint_clean_mse": endpoint,
        "prefix_path_mse_auc": path_auc,
        "ordered_rank_by_path_auc": ordered_rank,
        "order_count": order_count,
        "zero_token_max_abs": zero,
        "shuffled_to_ordered_endpoint_ratio": shuffle_ratio,
    }
    if expected_prefix_budget > 1:
        row["coarse_token_utilization"] = _coarse_token_utilization(
            checkpoint_eval
        )
    return row


def _validate_training_audit(
    audit: Mapping[str, Any],
    *,
    label: str,
    expected_run_dir: str,
) -> dict[str, Any]:
    checkpoint = audit.get("checkpoint")
    training_report = audit.get("training_report")
    latest = checkpoint.get("latest") if isinstance(checkpoint, Mapping) else None
    latest_integrity = (
        checkpoint.get("latest_integrity")
        if isinstance(checkpoint, Mapping)
        else None
    )
    expected_checkpoint = "checkpoint_step_00100000.pt"
    if (
        int(audit.get("schema_version", -1)) != 2
        or audit.get("status") != "complete"
        or audit.get("issues") != []
        or str(audit.get("run_dir", "")) != expected_run_dir
        or int(audit.get("expected_steps", -1)) != QUALITY_BRIDGE_STEPS
        or int(audit.get("last_step", -1)) != QUALITY_BRIDGE_STEPS
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("required_steps") != list(QUALITY_BRIDGE_MILESTONES)
        or checkpoint.get("missing_required_steps") != []
        or checkpoint.get("status") != "available"
        or not isinstance(training_report, Mapping)
        or training_report.get("status") != "current"
        or training_report.get("training_complete") is not True
        or int(training_report.get("completed_steps", -1))
        != QUALITY_BRIDGE_STEPS
        or str(training_report.get("path", ""))
        != (PurePosixPath(expected_run_dir) / "training_report.json").as_posix()
        or not isinstance(latest, Mapping)
        or latest.get("checkpoint") != expected_checkpoint
        or int(latest.get("step", -1)) != QUALITY_BRIDGE_STEPS
        or not isinstance(latest_integrity, Mapping)
        or latest_integrity.get("policy") != "required"
        or latest_integrity.get("status") != "verified"
        or latest_integrity.get("checkpoint") != expected_checkpoint
        or int(latest_integrity.get("step", -1)) != QUALITY_BRIDGE_STEPS
        or latest_integrity.get("integrity_manifest")
        != f"{expected_checkpoint}.integrity.json"
        or not _valid_hex_digest(latest_integrity.get("checkpoint_sha256"))
        or int(latest_integrity.get("checkpoint_bytes", 0)) < 1
    ):
        raise ValueError(f"{label} quality bridge training audit differs")
    return {
        "last_step": QUALITY_BRIDGE_STEPS,
        "metric_row_count": int(audit.get("metric_row_count", -1)),
        "validation_event_count": int(audit.get("validation_event_count", -1)),
        "checkpoint_steps": list(checkpoint.get("steps", [])),
        "latest_integrity": copy.deepcopy(dict(latest_integrity)),
        "warnings": list(audit.get("warnings", [])),
    }


def _validate_terminal_preflight(
    report: Mapping[str, Any],
    *,
    label: str,
    expected_git: Mapping[str, Any],
    expected_prefix_budget: int,
) -> dict[str, Any]:
    request = report.get("request")
    result = report.get("result")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "passed"
        or report.get("git") != dict(expected_git)
        or not isinstance(request, Mapping)
        or not isinstance(result, Mapping)
        or int(report.get("checkpoint_step", -1)) != QUALITY_BRIDGE_STEPS
        or report.get("weights") != "ema"
        or report.get("requested_weights") != "ema"
        or int(request.get("batch_size", -1)) != 32
        or int(request.get("prefix_budget", -1)) != expected_prefix_budget
        or request.get("precision") != "bf16"
        or float(request.get("guidance_scale", math.nan)) != 1.5
        or float(request.get("guidance_rescale", math.nan)) != 0.0
        or request.get("cfg_batch_mode") != "batched"
        or request.get("image_shape") != [3, 256, 256]
        or result.get("output_finite") is not True
        or result.get("output_shape") != [32, 3, 256, 256]
    ):
        raise ValueError(f"{label} terminal sampling preflight differs")
    checkpoint_sha256 = str(report.get("checkpoint_sha256", ""))
    checkpoint_integrity_manifest = str(
        report.get("checkpoint_integrity_manifest", "")
    )
    environment_sha256 = str(report.get("runtime_environment_sha256", ""))
    if (
        not _valid_hex_digest(checkpoint_sha256)
        or not _valid_hex_digest(environment_sha256)
        or not checkpoint_integrity_manifest.endswith(".integrity.json")
    ):
        raise ValueError(f"{label} terminal preflight identity is malformed")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_integrity_manifest": checkpoint_integrity_manifest,
        "runtime_environment_sha256": environment_sha256,
        "mean_forward_seconds": _finite_float(
            result.get("mean_forward_seconds"),
            label=f"{label} preflight mean_forward_seconds",
        ),
        "peak_memory": copy.deepcopy(result.get("cuda_memory_after_forward")),
    }


def _validate_physical_evidence(
    evidence: Mapping[str, Any],
    *,
    method: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    checkpoint = _source_identity(
        evidence.get("checkpoint", {}),
        label=f"{label} physical checkpoint",
    )
    integrity = _source_identity(
        evidence.get("checkpoint_integrity_manifest", {}),
        label=f"{label} physical checkpoint integrity manifest",
    )
    sampling_report = _source_identity(
        evidence.get("sampling_report", {}),
        label=f"{label} physical sampling report",
    )
    sampling_manifest = _source_identity(
        evidence.get("sampling_manifest", {}),
        label=f"{label} physical sampling manifest",
    )
    sampling_progress = _source_identity(
        evidence.get("sampling_progress", {}),
        label=f"{label} physical sampling progress",
    )
    real_set = evidence.get("real_set")
    if (
        checkpoint["path"] != method["checkpoint"]
        or checkpoint["sha256"] != method["checkpoint_sha256"]
        or integrity["path"] != method["checkpoint_integrity_manifest"]
        or int(evidence.get("checkpoint_step", -1))
        != method["checkpoint_step"]
        or str(evidence.get("sample_set_sha256", ""))
        != method["sample_set_sha256"]
        or int(evidence.get("sample_count", -1)) != method["sample_count"]
        or sampling_report != method["sampling_report"]
        or sampling_manifest != method["sampling_manifest"]
        or sampling_progress != method["sampling_progress"]
        or not isinstance(real_set, Mapping)
        or str(real_set.get("root", "")) != str(method["real_set"].get("root", ""))
        or str(real_set.get("sha256", ""))
        != str(method["real_set"].get("sha256", ""))
        or int(real_set.get("image_count", -1))
        != int(method["real_set"].get("image_count", -1))
    ):
        raise ValueError(f"{label} physical terminal evidence differs")
    return {
        "checkpoint": checkpoint,
        "checkpoint_integrity_manifest": integrity,
        "checkpoint_step": method["checkpoint_step"],
        "sampling_report": sampling_report,
        "sampling_manifest": sampling_manifest,
        "sampling_progress": sampling_progress,
        "sample_set_sha256": method["sample_set_sha256"],
        "sample_count": method["sample_count"],
        "real_set": copy.deepcopy(dict(real_set)),
    }


def _quality_screen(
    *,
    cofitok: Mapping[str, Any],
    dense: Mapping[str, Any],
    class_fidelity: Mapping[str, Any],
    source_hold: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = {
        name: source_hold[name] for name in QUALITY_BRIDGE_THRESHOLD_NAMES
    }
    fid_relative_change = (
        cofitok["fid"] / dense["fid"] - 1.0 if dense["fid"] > 0.0 else None
    )
    endpoint_relative_change = (
        cofitok["endpoint_clean_mse"] / dense["endpoint_clean_mse"] - 1.0
        if dense["endpoint_clean_mse"] > 0.0
        else None
    )
    precision_regression = dense["precision"] - cofitok["precision"]
    recall_regression = dense["recall"] - cofitok["recall"]
    utilization = cofitok.get("coarse_token_utilization")
    if not isinstance(utilization, Mapping):
        raise ValueError("quality bridge CoFiTok token utilization is missing")
    checks = [
        {
            "name": "cofitok_absolute_fid",
            "passed": cofitok["fid"] <= thresholds["max_absolute_fid"],
            "observed": cofitok["fid"],
            "comparison": "<=",
            "threshold": thresholds["max_absolute_fid"],
        },
        {
            "name": "matched_fid_tolerance",
            "passed": (
                dense["fid"] > 0.0
                and cofitok["fid"]
                <= dense["fid"] * (1.0 + thresholds["max_fid_regression"])
            ),
            "observed": fid_relative_change,
            "comparison": "<=",
            "threshold": thresholds["max_fid_regression"],
        },
        {
            "name": "cofitok_precision_floor",
            "passed": cofitok["precision"] >= thresholds["min_precision"],
            "observed": cofitok["precision"],
            "comparison": ">=",
            "threshold": thresholds["min_precision"],
        },
        {
            "name": "cofitok_recall_floor",
            "passed": cofitok["recall"] >= thresholds["min_recall"],
            "observed": cofitok["recall"],
            "comparison": ">=",
            "threshold": thresholds["min_recall"],
        },
        {
            "name": "matched_precision_tolerance",
            "passed": precision_regression
            <= thresholds["max_precision_regression"],
            "observed": precision_regression,
            "comparison": "<=",
            "threshold": thresholds["max_precision_regression"],
        },
        {
            "name": "matched_recall_tolerance",
            "passed": recall_regression <= thresholds["max_recall_regression"],
            "observed": recall_regression,
            "comparison": "<=",
            "threshold": thresholds["max_recall_regression"],
        },
        {
            "name": "matched_endpoint_tolerance",
            "passed": cofitok["endpoint_clean_mse"]
            <= dense["endpoint_clean_mse"]
            * (1.0 + thresholds["max_endpoint_regression"]),
            "observed": endpoint_relative_change,
            "comparison": "<=",
            "threshold": thresholds["max_endpoint_regression"],
        },
        {
            "name": "ordered_prefix_rank",
            "passed": cofitok["ordered_rank_by_path_auc"] == 1
            and cofitok["order_count"] >= 6,
            "observed": {
                "rank": cofitok["ordered_rank_by_path_auc"],
                "order_count": cofitok["order_count"],
            },
            "comparison": "rank == 1 and order_count >=",
            "threshold": 6,
        },
        {
            "name": "coarse_token_utilization",
            "passed": utilization["coarse_token_energy_ratio"]
            >= thresholds["min_coarse_token_energy_ratio"],
            "observed": utilization["coarse_token_energy_ratio"],
            "comparison": ">=",
            "threshold": thresholds["min_coarse_token_energy_ratio"],
        },
        {
            "name": "restricted_synthesis_zero_token",
            "passed": cofitok["zero_token_max_abs"] == 0.0,
            "observed": cofitok["zero_token_max_abs"],
            "comparison": "==",
            "threshold": 0.0,
        },
        {
            "name": "shuffle_mismatch",
            "passed": cofitok["shuffled_to_ordered_endpoint_ratio"] > 1.0,
            "observed": cofitok["shuffled_to_ordered_endpoint_ratio"],
            "comparison": ">",
            "threshold": 1.0,
        },
        {
            "name": "class_fidelity",
            "passed": class_fidelity.get("valid") is True,
            "observed": class_fidelity.get("status"),
            "comparison": "==",
            "threshold": "pass",
        },
    ]
    failed_checks = [row["name"] for row in checks if row["passed"] is not True]
    source_cofitok_fid = float(source_hold["cofitok_fid"])
    source_dense_fid = float(source_hold["dense_fid"])
    source_cofitok_recall = float(source_hold["cofitok_recall"])
    source_dense_recall = float(source_hold["dense_recall"])
    return {
        "status": "pass" if not failed_checks else "hold",
        "non_authorizing": True,
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": failed_checks,
        "source_to_bridge": {
            "cofitok_fid_delta": cofitok["fid"] - source_cofitok_fid,
            "cofitok_fid_relative_change": (
                cofitok["fid"] - source_cofitok_fid
            )
            / max(source_cofitok_fid, 1e-12),
            "dense_fid_delta": dense["fid"] - source_dense_fid,
            "dense_fid_relative_change": (dense["fid"] - source_dense_fid)
            / max(source_dense_fid, 1e-12),
            "cofitok_recall_delta": cofitok["recall"] - source_cofitok_recall,
            "dense_recall_delta": dense["recall"] - source_dense_recall,
        },
        "interpretation": (
            "This is a non-authorizing full-data quality screen. Every absolute, "
            "matched-distribution, mechanism, and class-fidelity check must pass; "
            "a separate source-compatible gate and explicit user decision are "
            "required before any larger training stage."
        ),
    }


def build_quality_bridge_result(
    *,
    preparation: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    training_pair_validation: Mapping[str, Any],
    cofitok_training_audit: Mapping[str, Any],
    dense_training_audit: Mapping[str, Any],
    milestone_evidence: Mapping[int, Mapping[str, Any]],
    cofitok_sampling_preflight: Mapping[str, Any],
    dense_sampling_preflight: Mapping[str, Any],
    cofitok_generation: Mapping[str, Any],
    dense_generation: Mapping[str, Any],
    cofitok_checkpoint_eval: Mapping[str, Any],
    dense_checkpoint_eval: Mapping[str, Any],
    class_fidelity_qualification: Mapping[str, Any],
    physical_evidence: Mapping[str, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    expected_source_names = {
        "preparation",
        "launch_receipt",
        "cofitok_training",
        "dense_training",
        "training_pair_validation",
        "cofitok_training_audit",
        "dense_training_audit",
        "milestone_50000",
        "milestone_100000",
        "cofitok_sampling_preflight",
        "dense_sampling_preflight",
        "cofitok_generation",
        "dense_generation",
        "cofitok_checkpoint_eval",
        "dense_checkpoint_eval",
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
    }
    if set(source_identities) != expected_source_names:
        raise ValueError("quality bridge result source set differs")
    sources = {
        name: _source_identity(identity, label=name)
        for name, identity in source_identities.items()
    }
    preparation_evidence = validate_quality_bridge_preparation(preparation)
    launch = validate_quality_bridge_launch_receipt_contract(
        launch_receipt,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    expected_git = launch["git"]
    if (
        launch_receipt.get("source_reports", {}).get("preparation")
        != sources["preparation"]
    ):
        raise ValueError("quality bridge result uses another preparation")
    training_run_dirs = launch["training_run_dirs"]
    expected_training_paths = {
        "cofitok_training": (
            PurePosixPath(training_run_dirs[0]) / "training_report.json"
        ).as_posix(),
        "dense_training": (
            PurePosixPath(training_run_dirs[1]) / "training_report.json"
        ).as_posix(),
    }
    if any(
        sources[name]["path"] != expected_path
        for name, expected_path in expected_training_paths.items()
    ):
        raise ValueError("quality bridge result training paths differ from launch")
    if (
        training_pair_validation.get("status") != "pass"
        or int(training_pair_validation.get("expected_steps", -1))
        != QUALITY_BRIDGE_STEPS
        or training_pair_validation.get("expected_revision") != expected_revision
        or training_pair_validation.get("expected_branch") != expected_branch
        or training_pair_validation.get("expected_dataset")
        != QUALITY_BRIDGE_DATASET
        or training_pair_validation.get("authorization_gate_identity_sha256")
        is not None
        or training_pair_validation.get("training_recipe", {}).get("stage")
        != QUALITY_BRIDGE_RECIPE_STAGE
        or training_pair_validation.get("training_recipe", {}).get("valid")
        is not True
        or training_pair_validation.get("cofitok", {}).get("formal_full_training")
        is not False
        or training_pair_validation.get("dense", {}).get("formal_full_training")
        is not False
    ):
        raise ValueError("quality bridge matched training validation differs")
    audits = {
        "cofitok": _validate_training_audit(
            cofitok_training_audit,
            label="CoFiTok",
            expected_run_dir=training_run_dirs[0],
        ),
        "dense_identity": _validate_training_audit(
            dense_training_audit,
            label="dense",
            expected_run_dir=training_run_dirs[1],
        ),
    }
    if set(milestone_evidence) != set(QUALITY_BRIDGE_MILESTONES):
        raise ValueError("quality bridge result milestone set differs")
    milestones: dict[str, Any] = {}
    for step in QUALITY_BRIDGE_MILESTONES:
        evidence = milestone_evidence[step]
        if evidence.get("status") != "verified":
            raise ValueError(f"quality bridge milestone {step} is not verified")
        milestones[str(step)] = copy.deepcopy(dict(evidence))

    methods = {
        "cofitok": _terminal_method_row(
            cofitok_generation,
            cofitok_checkpoint_eval,
            label="CoFiTok",
            expected_prefix_budget=8,
            expected_random_orders=4,
            expected_git=expected_git,
        ),
        "dense_identity": _terminal_method_row(
            dense_generation,
            dense_checkpoint_eval,
            label="dense",
            expected_prefix_budget=1,
            expected_random_orders=0,
            expected_git=expected_git,
        ),
    }
    cofitok = methods["cofitok"]
    dense = methods["dense_identity"]
    preflights = {
        "cofitok": _validate_terminal_preflight(
            cofitok_sampling_preflight,
            label="CoFiTok",
            expected_git=expected_git,
            expected_prefix_budget=8,
        ),
        "dense_identity": _validate_terminal_preflight(
            dense_sampling_preflight,
            label="dense",
            expected_git=expected_git,
            expected_prefix_budget=1,
        ),
    }
    if (
        preflights["cofitok"]["checkpoint_sha256"]
        != cofitok["checkpoint_sha256"]
        or preflights["dense_identity"]["checkpoint_sha256"]
        != dense["checkpoint_sha256"]
        or preflights["cofitok"]["checkpoint_integrity_manifest"]
        != cofitok["checkpoint_integrity_manifest"]
        or preflights["dense_identity"]["checkpoint_integrity_manifest"]
        != dense["checkpoint_integrity_manifest"]
        or preflights["cofitok"]["runtime_environment_sha256"]
        != cofitok["sampling_runtime_environment_sha256"]
        or preflights["dense_identity"]["runtime_environment_sha256"]
        != dense["sampling_runtime_environment_sha256"]
    ):
        raise ValueError("quality bridge terminal preflight uses another checkpoint")
    if (
        _matched_sampling_protocol(cofitok["sampling"])
        != _matched_sampling_protocol(dense["sampling"])
        or cofitok["real_set"] != dense["real_set"]
        or cofitok["metrics_evaluator_git"]
        != dense["metrics_evaluator_git"]
        or cofitok["metrics_runtime_environment_sha256"]
        != dense["metrics_runtime_environment_sha256"]
        or cofitok["sampling_runtime_environment_sha256"]
        != dense["sampling_runtime_environment_sha256"]
    ):
        raise ValueError("quality bridge terminal pair is not evaluator/protocol matched")
    if set(physical_evidence) != {"cofitok", "dense_identity"}:
        raise ValueError("quality bridge physical terminal evidence set differs")
    physical = {
        "cofitok": _validate_physical_evidence(
            physical_evidence["cofitok"],
            method=cofitok,
            label="CoFiTok",
        ),
        "dense_identity": _validate_physical_evidence(
            physical_evidence["dense_identity"],
            method=dense,
            label="dense",
        ),
    }

    class_fidelity = validate_class_fidelity_qualification(
        dict(class_fidelity_qualification),
        expected_stage="scaling",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        require_pass=False,
    )
    expected_class_sources = {
        "cofitok": sources["cofitok_class_fidelity"],
        "dense_identity": sources["dense_class_fidelity"],
    }
    if class_fidelity.get("sources") != expected_class_sources:
        raise ValueError("quality bridge class-fidelity sources differ")
    class_sampling = class_fidelity["sampling_contract"]
    if (
        class_sampling.get("cofitok_checkpoint_sha256")
        != cofitok["checkpoint_sha256"]
        or class_sampling.get("dense_checkpoint_sha256")
        != dense["checkpoint_sha256"]
        or class_sampling.get("cofitok_sample_set_sha256")
        != cofitok["sample_set_sha256"]
        or class_sampling.get("dense_sample_set_sha256")
        != dense["sample_set_sha256"]
    ):
        raise ValueError("quality bridge class fidelity uses another terminal sample pair")

    quality_screen = _quality_screen(
        cofitok=cofitok,
        dense=dense,
        class_fidelity=class_fidelity,
        source_hold=preparation_evidence["source_hold"],
    )
    return {
        "schema_version": QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "role": QUALITY_BRIDGE_RESULT_ROLE,
        "stage": QUALITY_BRIDGE_RECIPE_STAGE,
        "git": expected_git,
        "source_reports": sources,
        "training": {
            "pair_validation": copy.deepcopy(dict(training_pair_validation)),
            "audits": audits,
        },
        "milestones": milestones,
        "terminal": {
            "sampling_preflights": preflights,
            "physical_evidence": physical,
            "methods": methods,
            "matched_comparison": {
                "fid_relative_change": (cofitok["fid"] - dense["fid"])
                / max(dense["fid"], 1e-12),
                "precision_delta": cofitok["precision"] - dense["precision"],
                "recall_delta": cofitok["recall"] - dense["recall"],
                "endpoint_clean_mse_relative_change": (
                    cofitok["endpoint_clean_mse"]
                    - dense["endpoint_clean_mse"]
                )
                / max(dense["endpoint_clean_mse"], 1e-12),
            },
            "class_fidelity": class_fidelity,
        },
        "quality_screen": quality_screen,
        "authorization_boundary": copy.deepcopy(RESULT_AUTHORIZATION_BOUNDARY),
    }


def _gate_rows(gate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = gate.get("gates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source promotion gate has no gate rows")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("source promotion gate has a malformed gate row")
        name = str(row["name"])
        if name in by_name:
            raise ValueError(f"source promotion gate has duplicate gate row: {name}")
        if row.get("passed") not in {True, False}:
            raise ValueError(f"source promotion gate row lacks a boolean result: {name}")
        by_name[name] = row
    return by_name


def validate_quality_hold(gate: Mapping[str, Any]) -> dict[str, Any]:
    if gate.get("stage") != "scaling":
        raise ValueError("quality bridge source is not a scaling gate")
    if gate.get("source_profile") != "stability_scaling":
        raise ValueError("quality bridge source is not the stability scaling profile")
    if gate.get("status") != "fail" or gate.get("decision") != "hold":
        raise ValueError("quality bridge requires a source-valid scientific hold")

    rows = _gate_rows(gate)
    failed = sorted(name for name, row in rows.items() if row["passed"] is False)
    if failed != ["absolute_fid_quality"]:
        raise ValueError(
            "quality bridge requires absolute_fid_quality to be the only failed gate"
        )
    if rows.get("fid_within_tolerance", {}).get("passed") is not True:
        raise ValueError("source gate does not preserve matched FID competitiveness")
    if rows.get("ordered_prefix_path", {}).get("passed") is not True:
        raise ValueError("source gate does not preserve ordered prefix behavior")
    if rows.get("coarse_token_utilization", {}).get("passed") is not True:
        raise ValueError("source gate does not preserve compressed-token utilization")

    thresholds = gate.get("thresholds")
    summary = gate.get("summary")
    if not isinstance(thresholds, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("source promotion gate lacks thresholds or summary")
    if not QUALITY_BRIDGE_THRESHOLD_NAMES.issubset(thresholds):
        raise ValueError("source promotion gate quality thresholds are incomplete")
    raw_min_samples = thresholds["min_samples"]
    if type(raw_min_samples) is not int:
        raise ValueError("source promotion gate min_samples must be an integer")
    normalized_thresholds = {
        name: _finite_float(thresholds[name], label=name)
        for name in QUALITY_BRIDGE_THRESHOLD_NAMES
        if name != "min_samples"
    }
    normalized_thresholds["min_samples"] = raw_min_samples
    if (
        normalized_thresholds["max_absolute_fid"] <= 0.0
        or normalized_thresholds["min_samples"]
        != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or any(
            not 0.0 <= normalized_thresholds[name] <= 1.0
            for name in QUALITY_BRIDGE_THRESHOLD_NAMES
            if name not in {"max_absolute_fid", "min_samples"}
        )
    ):
        raise ValueError("source promotion gate quality thresholds are invalid")
    max_absolute_fid = float(normalized_thresholds["max_absolute_fid"])
    cofitok_fid = _finite_float(summary.get("cofitok_fid"), label="cofitok_fid")
    dense_fid = _finite_float(summary.get("dense_fid"), label="dense_fid")
    cofitok_recall = _finite_float(
        summary.get("cofitok_recall"), label="cofitok_recall"
    )
    dense_recall = _finite_float(summary.get("dense_recall"), label="dense_recall")
    if cofitok_fid <= max_absolute_fid:
        raise ValueError("source gate FID does not reproduce the quality hold")
    if max(cofitok_recall, dense_recall) >= 0.1:
        raise ValueError("source gate does not show the shared low-recall coverage signal")

    return {
        "status": "validated_hold",
        "failed_gates": failed,
        **normalized_thresholds,
        "cofitok_fid": cofitok_fid,
        "dense_fid": dense_fid,
        "cofitok_recall": cofitok_recall,
        "dense_recall": dense_recall,
        "cofitok_fid_relative_to_dense": cofitok_fid / dense_fid,
    }


def _effective_batch(config: Mapping[str, Any]) -> int:
    return int(config.get("data", {}).get("batch_size", 0)) * int(
        config.get("optimization", {}).get("gradient_accumulation_steps", 0)
    )


def _validate_source_training(
    report: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if report.get("training_complete") is not True:
        raise ValueError(f"{label} source training is incomplete")
    if (
        int(report.get("completed_steps", -1)) != QUALITY_BRIDGE_SOURCE_STEPS
        or int(report.get("target_steps", -1)) != QUALITY_BRIDGE_SOURCE_STEPS
    ):
        raise ValueError(f"{label} source training horizon is not exact 50K")
    config = report.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} source training config is missing")
    if config.get("data", {}).get("dataset") != QUALITY_BRIDGE_SOURCE_DATASET:
        raise ValueError(f"{label} source training dataset is not the 10% split")
    provenance = report.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} source training lacks dataset provenance")
    validated_provenance = validate_dataset_provenance(
        provenance,
        expected_dataset=QUALITY_BRIDGE_SOURCE_DATASET,
    )
    effective_batch = _effective_batch(config)
    final_metrics = report.get("final_metrics")
    if not isinstance(final_metrics, Mapping):
        raise ValueError(f"{label} source training lacks final metrics")
    samples_seen = int(final_metrics.get("samples_seen", -1))
    expected_samples = QUALITY_BRIDGE_SOURCE_STEPS * effective_batch
    if effective_batch != QUALITY_BRIDGE_EFFECTIVE_BATCH or samples_seen != expected_samples:
        raise ValueError(f"{label} source sample accounting is invalid")
    parameter_count = int(report.get("parameter_count", 0))
    if parameter_count < 1:
        raise ValueError(f"{label} source parameter count is missing")
    return {
        "config": copy.deepcopy(dict(config)),
        "parameter_count": parameter_count,
        "effective_batch_size": effective_batch,
        "samples_seen": samples_seen,
        "dataset_provenance": validated_provenance,
    }


def _without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(mapping))
    for key in keys:
        result.pop(key, None)
    return result


def _validate_preserved_recipe(
    source: Mapping[str, Any],
    bridge: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for section in ("model", "loss", "diffusion"):
        if source.get(section) != bridge.get(section):
            raise ValueError(f"{label} bridge changed the qualified {section} recipe")
    if _without(source.get("data", {}), "dataset", "batch_size") != _without(
        bridge.get("data", {}), "dataset", "batch_size"
    ):
        raise ValueError(f"{label} bridge changed data preprocessing beyond dataset coverage")
    if _without(
        source.get("optimization", {}), "gradient_accumulation_steps"
    ) != _without(bridge.get("optimization", {}), "gradient_accumulation_steps"):
        raise ValueError(f"{label} bridge changed optimization hyperparameters")
    if _without(
        source.get("runtime", {}), "steps", "protected_checkpoint_steps"
    ) != _without(bridge.get("runtime", {}), "steps", "protected_checkpoint_steps"):
        raise ValueError(f"{label} bridge changed runtime behavior beyond the horizon")
    if _effective_batch(source) != _effective_batch(bridge):
        raise ValueError(f"{label} bridge changed the effective batch size")


def build_quality_bridge_preparation(
    *,
    promotion_gate: Mapping[str, Any],
    promotion_gate_identity: Mapping[str, Any],
    gate_source_verification: Mapping[str, Any],
    source_cofitok_training: Mapping[str, Any],
    source_dense_training: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    config_validation: Mapping[str, Any],
) -> dict[str, Any]:
    hold = validate_quality_hold(promotion_gate)
    gate_identity = _source_identity(
        promotion_gate_identity, label="promotion gate"
    )
    cofitok_identity = _source_identity(
        cofitok_config_identity, label="CoFiTok bridge config"
    )
    dense_identity = _source_identity(
        dense_config_identity, label="dense bridge config"
    )

    if gate_source_verification.get("status") != "verified":
        raise ValueError("promotion gate source reports were not independently verified")
    if gate_source_verification.get("source_profile") != "stability_scaling":
        raise ValueError("promotion gate source verification profile is invalid")
    verified_sources = gate_source_verification.get("source_reports")
    if not isinstance(verified_sources, Mapping):
        raise ValueError("promotion gate source verification is incomplete")
    if dict(verified_sources) != dict(promotion_gate.get("source_reports", {})):
        raise ValueError("promotion gate source verification differs from the gate")

    if config_validation.get("status") != "pass":
        raise ValueError("quality bridge matched-config validation did not pass")
    recipe = config_validation.get("training_recipe")
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or recipe.get("valid") is not True
    ):
        raise ValueError("quality bridge training recipe contract did not pass")

    source_cofitok = _validate_source_training(
        source_cofitok_training, label="CoFiTok"
    )
    source_dense = _validate_source_training(source_dense_training, label="dense")
    if (
        source_cofitok["dataset_provenance"]["identity_sha256"]
        != source_dense["dataset_provenance"]["identity_sha256"]
    ):
        raise ValueError("source pair used different 10% dataset identities")

    if cofitok_config.get("data", {}).get("dataset") != QUALITY_BRIDGE_DATASET:
        raise ValueError("CoFiTok bridge does not use full ImageNet-256")
    if dense_config.get("data", {}).get("dataset") != QUALITY_BRIDGE_DATASET:
        raise ValueError("dense bridge does not use full ImageNet-256")
    if int(cofitok_config.get("runtime", {}).get("steps", -1)) != QUALITY_BRIDGE_STEPS:
        raise ValueError("CoFiTok bridge horizon is not exact 100K")
    if int(dense_config.get("runtime", {}).get("steps", -1)) != QUALITY_BRIDGE_STEPS:
        raise ValueError("dense bridge horizon is not exact 100K")
    if _effective_batch(cofitok_config) != QUALITY_BRIDGE_EFFECTIVE_BATCH:
        raise ValueError("CoFiTok bridge effective batch is not 64")
    if _effective_batch(dense_config) != QUALITY_BRIDGE_EFFECTIVE_BATCH:
        raise ValueError("dense bridge effective batch is not 64")

    _validate_preserved_recipe(
        source_cofitok["config"], cofitok_config, label="CoFiTok"
    )
    _validate_preserved_recipe(source_dense["config"], dense_config, label="dense")
    if source_cofitok["parameter_count"] != int(
        config_validation.get("cofitok", {}).get("parameter_count", -1)
    ):
        raise ValueError("CoFiTok bridge changed model capacity")
    if source_dense["parameter_count"] != int(
        config_validation.get("dense", {}).get("parameter_count", -1)
    ):
        raise ValueError("dense bridge changed model capacity")

    source_spec = FORMAL_GENERATION_DATASETS[QUALITY_BRIDGE_SOURCE_DATASET]
    bridge_spec = FORMAL_GENERATION_DATASETS[QUALITY_BRIDGE_DATASET]
    source_images = QUALITY_BRIDGE_SOURCE_STEPS * QUALITY_BRIDGE_EFFECTIVE_BATCH
    bridge_images = QUALITY_BRIDGE_STEPS * QUALITY_BRIDGE_EFFECTIVE_BATCH
    milestone_epochs = {
        str(step): step * QUALITY_BRIDGE_EFFECTIVE_BATCH / bridge_spec.train_images
        for step in QUALITY_BRIDGE_MILESTONES
    }

    return {
        "schema_version": QUALITY_BRIDGE_SCHEMA_VERSION,
        "status": "prepared",
        "role": QUALITY_BRIDGE_ROLE,
        "source_quality_hold": {
            **hold,
            "promotion_gate": gate_identity,
            "verified_source_reports": dict(verified_sources),
            "source_dataset": QUALITY_BRIDGE_SOURCE_DATASET,
            "source_train_images": source_spec.train_images,
            "source_steps": QUALITY_BRIDGE_SOURCE_STEPS,
            "source_effective_batch_size": QUALITY_BRIDGE_EFFECTIVE_BATCH,
            "source_images_seen_per_method": source_images,
            "source_equivalent_epochs": source_images / source_spec.train_images,
        },
        "selection": {
            "dataset": QUALITY_BRIDGE_DATASET,
            "train_images": bridge_spec.train_images,
            "steps": QUALITY_BRIDGE_STEPS,
            "effective_batch_size": QUALITY_BRIDGE_EFFECTIVE_BATCH,
            "images_seen_per_method": bridge_images,
            "equivalent_epochs": bridge_images / bridge_spec.train_images,
            "base_channels": 128,
            "milestone_steps": list(QUALITY_BRIDGE_MILESTONES),
            "milestone_equivalent_epochs": milestone_epochs,
            "budget_analysis": {
                "50000": {
                    "equivalent_epochs": milestone_epochs["50000"],
                    "decision": "retain_as_intermediate_milestone",
                    "reason": (
                        "clean early full-data quality point, but insufficient as the "
                        "only terminal decision for continued optimization"
                    ),
                },
                "100000": {
                    "equivalent_epochs": milestone_epochs["100000"],
                    "decision": "selected_terminal_budget",
                    "reason": (
                        "brackets the approximately four-epoch repetition inflection "
                        "while retaining a matched 50K slope point"
                    ),
                },
                "150000": {
                    "equivalent_epochs": (
                        150_000 * QUALITY_BRIDGE_EFFECTIVE_BATCH
                        / bridge_spec.train_images
                    ),
                    "decision": "rejected",
                    "reason": (
                        "adds fifty percent training cost after the main unique-data "
                        "coverage window and weakens diagnostic efficiency"
                    ),
                },
            },
            "capacity_change_allowed": False,
            "qualified_model_and_loss_recipe_preserved": True,
        },
        "matched_training_contract": {
            "recipe_stage": QUALITY_BRIDGE_RECIPE_STAGE,
            "config_validation": copy.deepcopy(dict(config_validation)),
            "cofitok_config": cofitok_identity,
            "dense_config": dense_identity,
            "source_parameter_counts": {
                "cofitok": source_cofitok["parameter_count"],
                "dense_identity": source_dense["parameter_count"],
            },
        },
        "evaluation_contract": {
            "milestones": {
                "steps": list(QUALITY_BRIDGE_MILESTONES),
                "weights": "ema",
                "sampler": "ddim",
                "sample_steps": 50,
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "samples_per_method": 2_048,
                "mechanism_images": 256,
                "role": "non_claim_trend_diagnostic",
            },
            "terminal": {
                "step": QUALITY_BRIDGE_STEPS,
                "weights": "ema",
                "sampler": "ddim",
                "sample_steps": 100,
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "samples_per_method": QUALITY_BRIDGE_TERMINAL_SAMPLES,
                "metrics": ["fid", "inception_score", "precision", "recall"],
                "skip_precision_recall_allowed": False,
                "matched_fixed_random_stream_required": True,
                "role": "quality_bridge_decision_evidence",
            },
        },
        "authorization_boundary": {
            "quality_bridge_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "new_gate_required": True,
            "explicit_execution_approval_required": True,
        },
    }
