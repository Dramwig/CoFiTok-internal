from __future__ import annotations

import copy
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_CONFIGURED_STEPS,
    CAPACITY_PROBE_EFFECTIVE_BATCH,
    CAPACITY_PROBE_MIN_COARSE_TOKEN_ENERGY_RATIO,
    CAPACITY_PROBE_PARAMETER_COUNTS,
    CAPACITY_PROBE_SAMPLES_PER_ARM,
    CAPACITY_PROBE_STOP_STEP,
    validate_capacity_probe_preparation_contract,
)
from cofitok.generation.capacity_probe_execution import (
    LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY,
    validate_capacity_probe_launch_receipt_contract,
)
from cofitok.generation.capacity_probe_training import PARTIAL_TRAINING_ROLE
from cofitok.generation.protocol import sampling_protocol_contract


CAPACITY_PROBE_RESULT_SCHEMA_VERSION = 1
CAPACITY_PROBE_RESULT_ROLE = "stability_full_data_capacity_probe_result"
CAPACITY_PROBE_ARM_NAMES = (
    "base128_cofitok",
    "base128_dense_identity",
    "base256_cofitok",
    "base256_dense_identity",
)
CAPACITY_PROBE_RESULT_SOURCE_NAMES = {
    "preparation",
    "launch_receipt",
    "base256_cofitok_training_validation",
    "base256_dense_identity_training_validation",
    *{
        f"{arm}_{kind}"
        for arm in CAPACITY_PROBE_ARM_NAMES
        for kind in ("sampling_preflight", "generation", "checkpoint_eval")
    },
}
CAPACITY_PROBE_RESULT_BOUNDARY = {
    "capacity_probe_evidence_complete": True,
    "capacity_probe_execution_allowed": False,
    "additional_training_allowed": False,
    "configured_100k_completion_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "formal_generation_claim_allowed": False,
    "release_authorization_allowed": False,
    "new_source_compatible_decision_required": True,
}


def _hex(value: Any, *, length: int = 64) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _hex(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(
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


def _validate_partial_training(
    report: Mapping[str, Any],
    *,
    method: str,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    boundary = report.get("authorization_boundary")
    expected_parameters = CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method]
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "pass"
        or report.get("role") != PARTIAL_TRAINING_ROLE
        or report.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or int(report.get("configured_steps", -1))
        != CAPACITY_PROBE_CONFIGURED_STEPS
        or int(report.get("completed_steps", -1)) != CAPACITY_PROBE_STOP_STEP
        or report.get("training_complete") is not False
        or report.get("intentional_partial_stop") is not True
        or int(report.get("effective_batch_size", -1))
        != CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(report.get("images_seen", -1))
        != CAPACITY_PROBE_STOP_STEP * CAPACITY_PROBE_EFFECTIVE_BATCH
        or int(report.get("parameter_count", -1)) != expected_parameters
        or not isinstance(checkpoint, Mapping)
        or not isinstance(boundary, Mapping)
        or boundary.get("capacity_probe_training_complete") is not True
        or boundary.get("full_training_complete") is not False
        or boundary.get("full_300k_launch_allowed") is not False
        or boundary.get("formal_generation_claim_allowed") is not False
        or boundary.get("release_allowed") is not False
    ):
        raise ValueError(f"{method} partial training validation differs")
    checkpoint_identity = _identity(checkpoint, label=f"{method} checkpoint")
    integrity_identity = _identity(
        checkpoint.get("integrity_manifest", {}),
        label=f"{method} checkpoint integrity manifest",
    )
    integrity_path = integrity_identity["path"]
    if (
        not checkpoint_identity["path"].endswith(
            f"checkpoint_step_{CAPACITY_PROBE_STOP_STEP:08d}.pt"
        )
        or not integrity_path.endswith(".pt.integrity.json")
    ):
        raise ValueError(f"{method} partial checkpoint path differs")
    return {
        "checkpoint": checkpoint_identity,
        "checkpoint_integrity_manifest": integrity_identity,
        "runtime_environment_sha256": report.get("runtime_environment_sha256"),
        "dataset_identity_sha256": report.get("dataset_identity_sha256"),
        "images_seen": int(report["images_seen"]),
        "parameter_count": expected_parameters,
    }


def _sampling_without_arm_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"prefix_budgets", "sample_set_digest"}
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in ignored
    }


def _coarse_token_utilization(
    checkpoint_eval: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    metrics = checkpoint_eval.get("metrics")
    model = checkpoint_eval.get("config", {}).get("model", {})
    if not isinstance(metrics, Mapping) or not isinstance(model, Mapping):
        raise ValueError(f"{label} token-utilization evidence is missing")
    raw_ratios = metrics.get("component_energy_ratio_per_sample_mean")
    raw_strides = model.get("token_spatial_strides")
    try:
        ratios = [float(value) for value in raw_ratios]
        strides = [int(value) for value in raw_strides]
        token_count = int(model.get("token_count", -1))
    except (TypeError, ValueError):
        ratios, strides, token_count = [], [], -1
    if (
        token_count < 3
        or len(ratios) != token_count
        or len(strides) != token_count
        or any(value < 1 for value in strides)
        or 1 not in strides
        or any(not math.isfinite(value) or value < 0.0 for value in ratios)
        or not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(f"{label} token-utilization evidence is invalid")
    coarse_token_count = strides.index(1)
    if (
        coarse_token_count < 1
        or any(value <= 1 for value in strides[:coarse_token_count])
        or any(value != 1 for value in strides[coarse_token_count:])
    ):
        raise ValueError(f"{label} token-utilization stride partition is invalid")
    coarse_ratio = sum(ratios[:coarse_token_count])
    return {
        "source_metric": "component_energy_ratio_per_sample_mean",
        "partition_schema": "token_spatial_stride_suffix_v1",
        "token_count": token_count,
        "coarse_token_count": coarse_token_count,
        "full_resolution_tail_token_count": token_count - coarse_token_count,
        "token_spatial_strides": strides,
        "component_energy_ratios": ratios,
        "coarse_token_energy_ratio": coarse_ratio,
        "minimum_coarse_token_energy_ratio": (
            CAPACITY_PROBE_MIN_COARSE_TOKEN_ENERGY_RATIO
        ),
        "passed": coarse_ratio >= CAPACITY_PROBE_MIN_COARSE_TOKEN_ENERGY_RATIO,
    }


def _validate_preflight(
    report: Mapping[str, Any],
    *,
    label: str,
    expected_checkpoint_sha256: str,
    expected_checkpoint_path: str,
    expected_integrity_path: str,
    expected_prefix_budget: int,
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    request = report.get("request")
    result = report.get("result")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("status") != "passed"
        or report.get("git") != dict(expected_git)
        or report.get("checkpoint_sha256") != expected_checkpoint_sha256
        or report.get("checkpoint") != expected_checkpoint_path
        or report.get("checkpoint_integrity_manifest") != expected_integrity_path
        or int(report.get("checkpoint_step", -1)) != CAPACITY_PROBE_STOP_STEP
        or report.get("weights") != "ema"
        or report.get("requested_weights") != "ema"
        or not isinstance(request, Mapping)
        or int(request.get("batch_size", -1)) != 32
        or int(request.get("prefix_budget", -1)) != expected_prefix_budget
        or request.get("precision") != "bf16"
        or float(request.get("guidance_scale", -1.0)) != 1.5
        or float(request.get("guidance_rescale", -1.0)) != 0.0
        or request.get("cfg_batch_mode") != "batched"
        or not isinstance(result, Mapping)
        or result.get("output_finite") is not True
        or _finite(
            result.get("mean_forward_seconds"),
            label=f"{label} preflight mean forward seconds",
        )
        <= 0.0
    ):
        raise ValueError(f"{label} sampling preflight differs")
    environment_sha = str(report.get("runtime_environment_sha256", ""))
    if not _hex(environment_sha):
        raise ValueError(f"{label} preflight runtime identity is malformed")
    return {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_step": CAPACITY_PROBE_STOP_STEP,
        "prefix_budget": expected_prefix_budget,
        "runtime_environment_sha256": environment_sha,
        "mean_forward_seconds": float(result["mean_forward_seconds"]),
    }


def _validate_arm(
    *,
    arm: str,
    generation: Mapping[str, Any],
    checkpoint_eval: Mapping[str, Any],
    sampling_preflight: Mapping[str, Any],
    physical_evidence: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any],
    expected_integrity: Mapping[str, Any],
    expected_prefix_budget: int,
    expected_random_orders: int,
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    expected_checkpoint_identity = _identity(
        expected_checkpoint,
        label=f"{arm} expected checkpoint",
    )
    expected_integrity_identity = _identity(
        expected_integrity,
        label=f"{arm} expected checkpoint integrity",
    )
    expected_integrity_path = expected_integrity_identity["path"]
    preflight = _validate_preflight(
        sampling_preflight,
        label=arm,
        expected_checkpoint_sha256=expected_checkpoint_identity["sha256"],
        expected_checkpoint_path=expected_checkpoint_identity["path"],
        expected_integrity_path=expected_integrity_path,
        expected_prefix_budget=expected_prefix_budget,
        expected_git=expected_git,
    )
    provenance = generation.get("sample_provenance")
    counts = generation.get("counts")
    parameters = generation.get("parameters")
    real_set = generation.get("real_set")
    if (
        int(generation.get("schema_version", -1)) != 3
        or generation.get("role") != "generation_directory_metrics_report"
        or generation.get("status") != "completed"
        or generation.get("protocol") != "torch_fidelity_directory_metrics"
        or generation.get("git") != dict(expected_git)
        or not all(
            isinstance(value, Mapping)
            for value in (provenance, counts, parameters, real_set)
        )
    ):
        raise ValueError(f"{arm} generation report contract differs")
    sampling = provenance.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError(f"{arm} sampling protocol is missing")
    protocol = sampling_protocol_contract(
        dict(sampling),
        stage="milestone",
        expected_num_train_timesteps=1_000,
    )
    if protocol["valid"] is not True:
        raise ValueError(
            f"{arm} sampling protocol differs: " + ", ".join(protocol["issues"])
        )
    if (
        int(counts.get("generated_image_count", -1))
        != CAPACITY_PROBE_SAMPLES_PER_ARM
        or parameters.get("precision_recall_enabled") is not False
        or provenance.get("checkpoint") != expected_checkpoint_identity["path"]
        or provenance.get("checkpoint_sha256")
        != expected_checkpoint_identity["sha256"]
        or provenance.get("checkpoint_integrity_manifest")
        != expected_integrity_path
        or int(provenance.get("checkpoint_step", -1))
        != CAPACITY_PROBE_STOP_STEP
        or provenance.get("weights") != "ema"
        or provenance.get("git") != dict(expected_git)
        or int(provenance.get("selected_prefix_budget", -1))
        != expected_prefix_budget
        or sampling.get("prefix_budgets") != [expected_prefix_budget]
        or sampling.get("image_shape") != [3, 256, 256]
    ):
        raise ValueError(f"{arm} generation identity differs")
    checkpoint_sha = str(provenance.get("checkpoint_sha256", ""))
    sample_set_sha = str(provenance.get("sample_set_sha256", ""))
    evaluator_environment = str(generation.get("runtime_environment_sha256", ""))
    sampling_environment = str(provenance.get("runtime_environment_sha256", ""))
    if not all(
        _hex(value)
        for value in (
            checkpoint_sha,
            sample_set_sha,
            evaluator_environment,
            sampling_environment,
        )
    ):
        raise ValueError(f"{arm} checkpoint/sample/runtime identity is malformed")
    if preflight["runtime_environment_sha256"] != sampling_environment:
        raise ValueError(f"{arm} preflight and sampling runtime identities differ")
    progress = provenance.get("sampling_progress")
    if (
        not isinstance(progress, Mapping)
        or progress.get("status") != "completed"
        or int(progress.get("completed_samples", -1))
        != CAPACITY_PROBE_SAMPLES_PER_ARM
        or _finite(
            progress.get("cumulative_elapsed_seconds"),
            label=f"{arm} sampling elapsed",
        )
        <= 0.0
    ):
        raise ValueError(f"{arm} sampling progress is incomplete")
    sampling_report = _identity(
        provenance.get("report_identity", {}),
        label=f"{arm} sampling report",
    )
    sampling_manifest = _identity(
        provenance.get("manifest_identity", {}),
        label=f"{arm} sampling manifest",
    )
    sampling_progress = _identity(
        progress.get("identity", {}),
        label=f"{arm} sampling progress",
    )
    if (
        not PurePosixPath(str(real_set.get("root", ""))).is_absolute()
        or not _hex(real_set.get("sha256"))
        or int(real_set.get("image_count", -1)) < CAPACITY_PROBE_SAMPLES_PER_ARM
    ):
        raise ValueError(f"{arm} real-set identity is malformed")
    metrics = generation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{arm} quality metrics are missing")
    fid = _finite(metrics.get("frechet_inception_distance"), label=f"{arm} FID")
    inception = _finite(metrics.get("inception_score_mean"), label=f"{arm} IS")
    if fid < 0.0 or inception <= 0.0:
        raise ValueError(f"{arm} quality metric domain differs")

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
        or checkpoint_eval.get("checkpoint")
        != expected_checkpoint_identity["path"]
        or checkpoint_eval.get("checkpoint_sha256") != checkpoint_sha
        or checkpoint_eval.get("checkpoint_integrity_manifest")
        != expected_integrity_path
        or int(checkpoint_eval.get("checkpoint_step", -1))
        != CAPACITY_PROBE_STOP_STEP
        or checkpoint_eval.get("weights") != "ema"
        or int(mechanism.get("evaluated_images", -1)) != 256
    ):
        raise ValueError(f"{arm} mechanism evaluation differs")
    ordered = mechanism.get("orders", {}).get("ordered")
    if not isinstance(ordered, Mapping):
        raise ValueError(f"{arm} ordered mechanism row is missing")
    endpoint = _finite(
        ordered.get("endpoint_clean_mse"),
        label=f"{arm} endpoint MSE",
    )
    path_auc = _finite(
        ordered.get("prefix_path_mse_auc"),
        label=f"{arm} path AUC",
    )
    zero = _finite(mechanism.get("zero_token_max_abs"), label=f"{arm} zero token")
    shuffle = _finite(
        mechanism.get("shuffled_to_ordered_endpoint_ratio"),
        label=f"{arm} shuffle ratio",
    )
    order_count = int(mechanism.get("order_count", -1))
    ordered_rank = int(mechanism.get("ordered_rank_by_path_auc", -1))
    if order_count < 1 or not 1 <= ordered_rank <= order_count:
        raise ValueError(f"{arm} order evidence is invalid")

    physical_checkpoint = _identity(
        physical_evidence.get("checkpoint", {}),
        label=f"{arm} physical checkpoint",
    )
    physical_integrity = _identity(
        physical_evidence.get("checkpoint_integrity_manifest", {}),
        label=f"{arm} physical checkpoint integrity",
    )
    if (
        physical_checkpoint != expected_checkpoint_identity
        or physical_integrity != expected_integrity_identity
        or int(physical_evidence.get("checkpoint_step", -1))
        != CAPACITY_PROBE_STOP_STEP
        or physical_evidence.get("sampling_report") != sampling_report
        or physical_evidence.get("sampling_manifest") != sampling_manifest
        or physical_evidence.get("sampling_progress") != sampling_progress
        or physical_evidence.get("sample_set_sha256") != sample_set_sha
        or int(physical_evidence.get("sample_count", -1))
        != CAPACITY_PROBE_SAMPLES_PER_ARM
        or not isinstance(physical_evidence.get("real_set"), Mapping)
        or physical_evidence["real_set"].get("root") != real_set.get("root")
        or physical_evidence["real_set"].get("sha256") != real_set.get("sha256")
        or int(physical_evidence["real_set"].get("image_count", -1))
        != int(real_set.get("image_count", -1))
    ):
        raise ValueError(f"{arm} physical evidence differs")
    mechanism_valid = True
    utilization = None
    if "cofitok" in arm:
        utilization = _coarse_token_utilization(
            checkpoint_eval,
            label=arm,
        )
        mechanism_valid = bool(
            zero == 0.0
            and ordered_rank == 1
            and order_count >= 6
            and shuffle > 1.0
            and utilization["passed"] is True
        )
    return {
        "checkpoint": expected_checkpoint_identity,
        "checkpoint_integrity_manifest": physical_integrity,
        "checkpoint_step": CAPACITY_PROBE_STOP_STEP,
        "weights": "ema",
        "sample_count": CAPACITY_PROBE_SAMPLES_PER_ARM,
        "sample_set_sha256": sample_set_sha,
        "selected_prefix_budget": expected_prefix_budget,
        "sampling": copy.deepcopy(dict(sampling)),
        "sampling_report": sampling_report,
        "sampling_manifest": sampling_manifest,
        "sampling_progress": sampling_progress,
        "sampling_runtime_environment_sha256": sampling_environment,
        "metrics_runtime_environment_sha256": evaluator_environment,
        "sampling_preflight": preflight,
        "real_set": copy.deepcopy(dict(real_set)),
        "fid": fid,
        "inception_score": inception,
        "endpoint_clean_mse": endpoint,
        "prefix_path_mse_auc": path_auc,
        "ordered_rank_by_path_auc": ordered_rank,
        "order_count": order_count,
        "zero_token_max_abs": zero,
        "shuffled_to_ordered_endpoint_ratio": shuffle,
        "coarse_token_utilization": utilization,
        "mechanism_invariants_valid": mechanism_valid,
    }


def build_capacity_probe_result(
    *,
    preparation: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    partial_training_validations: Mapping[str, Mapping[str, Any]],
    sampling_preflights: Mapping[str, Mapping[str, Any]],
    generation_reports: Mapping[str, Mapping[str, Any]],
    checkpoint_evaluations: Mapping[str, Mapping[str, Any]],
    physical_evidence: Mapping[str, Mapping[str, Any]],
    source_identities: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if set(source_identities) != CAPACITY_PROBE_RESULT_SOURCE_NAMES:
        raise ValueError("capacity probe result source set differs")
    if set(partial_training_validations) != {"cofitok", "dense_identity"}:
        raise ValueError("capacity probe partial training set differs")
    arm_set = set(CAPACITY_PROBE_ARM_NAMES)
    for label, values in (
        ("sampling preflight", sampling_preflights),
        ("generation report", generation_reports),
        ("checkpoint evaluation", checkpoint_evaluations),
        ("physical evidence", physical_evidence),
    ):
        if set(values) != arm_set:
            raise ValueError(f"capacity probe {label} arm set differs")
    sources = {
        name: _identity(identity, label=f"capacity probe source {name}")
        for name, identity in source_identities.items()
    }
    launch = validate_capacity_probe_launch_receipt_contract(
        launch_receipt,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if launch_receipt.get("authorization_boundary") != (
        LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("capacity probe launch authorization boundary differs")
    output_root = launch["output_root"]
    selection = validate_capacity_probe_preparation_contract(
        preparation,
        expected_output_root=output_root,
    )
    if launch_receipt.get("source_reports", {}).get("preparation") != sources[
        "preparation"
    ]:
        raise ValueError("capacity probe result binds another preparation")
    expected_git = _git(
        launch_receipt.get("git", {}),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        label="capacity probe result",
    )
    training = {
        method: _validate_partial_training(
            partial_training_validations[method],
            method=method,
            expected_revision=expected_revision,
            expected_branch=expected_branch,
        )
        for method in ("cofitok", "dense_identity")
    }
    references = preparation.get("checkpoint_references", {})
    expected_checkpoints = {
        "base128_cofitok": references["cofitok"]["checkpoint"],
        "base128_dense_identity": references["dense_identity"]["checkpoint"],
        "base256_cofitok": training["cofitok"]["checkpoint"],
        "base256_dense_identity": training["dense_identity"]["checkpoint"],
    }
    expected_integrities = {
        "base128_cofitok": references["cofitok"]["integrity"],
        "base128_dense_identity": references["dense_identity"]["integrity"],
        "base256_cofitok": training["cofitok"][
            "checkpoint_integrity_manifest"
        ],
        "base256_dense_identity": training["dense_identity"][
            "checkpoint_integrity_manifest"
        ],
    }
    arms = {}
    for arm in CAPACITY_PROBE_ARM_NAMES:
        cofitok = arm.endswith("cofitok")
        arms[arm] = _validate_arm(
            arm=arm,
            generation=generation_reports[arm],
            checkpoint_eval=checkpoint_evaluations[arm],
            sampling_preflight=sampling_preflights[arm],
            physical_evidence=physical_evidence[arm],
            expected_checkpoint=expected_checkpoints[arm],
            expected_integrity=expected_integrities[arm],
            expected_prefix_budget=8 if cofitok else 1,
            expected_random_orders=4 if cofitok else 0,
            expected_git=expected_git,
        )
    normalized_sampling = [
        _sampling_without_arm_fields(arms[arm]["sampling"])
        for arm in CAPACITY_PROBE_ARM_NAMES
    ]
    if any(value != normalized_sampling[0] for value in normalized_sampling[1:]):
        raise ValueError("capacity probe random streams or sampling protocols differ")
    real_sets = [arms[arm]["real_set"] for arm in CAPACITY_PROBE_ARM_NAMES]
    if any(value != real_sets[0] for value in real_sets[1:]):
        raise ValueError("capacity probe evaluation real sets differ")
    evaluator_environments = {
        arms[arm]["metrics_runtime_environment_sha256"]
        for arm in CAPACITY_PROBE_ARM_NAMES
    }
    sampling_environments = {
        arms[arm]["sampling_runtime_environment_sha256"]
        for arm in CAPACITY_PROBE_ARM_NAMES
    }
    if len(evaluator_environments) != 1 or len(sampling_environments) != 1:
        raise ValueError("capacity probe evaluation runtime environments differ")

    cofitok_delta = arms["base256_cofitok"]["fid"] - arms[
        "base128_cofitok"
    ]["fid"]
    dense_delta = arms["base256_dense_identity"]["fid"] - arms[
        "base128_dense_identity"
    ]["fid"]
    interaction = cofitok_delta - dense_delta
    shared_improvement = cofitok_delta < 0.0 and dense_delta < 0.0
    mechanism_valid = bool(
        arms["base128_cofitok"]["mechanism_invariants_valid"]
        and arms["base256_cofitok"]["mechanism_invariants_valid"]
    )
    capacity_supported = shared_improvement and mechanism_valid
    if capacity_supported:
        recommendation = {
            "id": "prepare_source_compatible_capacity_scaling_decision",
            "category": "capacity_supported",
            "execution_ready": False,
            "full_300k_launch_allowed": False,
            "reason": (
                "Both matched methods improved strictly at base256 while CoFiTok "
                "mechanism invariants remained valid."
            ),
        }
    elif shared_improvement:
        recommendation = {
            "id": "hold_capacity_scaling_due_to_mechanism_failure",
            "category": "capacity_not_qualified",
            "execution_ready": False,
            "full_300k_launch_allowed": False,
            "reason": "Shared FID improvement did not preserve CoFiTok mechanism invariants.",
        }
    else:
        recommendation = {
            "id": "hold_capacity_scaling_and_revisit_training_objective",
            "category": "capacity_not_supported",
            "execution_ready": False,
            "full_300k_launch_allowed": False,
            "reason": "Base256 did not strictly improve both matched methods at step 10K.",
        }
    return {
        "schema_version": CAPACITY_PROBE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "role": CAPACITY_PROBE_RESULT_ROLE,
        "git": expected_git,
        "output_root": output_root,
        "source_reports": sources,
        "selection": selection,
        "partial_training": training,
        "evaluation": {
            "arms": arms,
            "matched_sampling_protocol": normalized_sampling[0],
            "real_set": real_sets[0],
            "metrics_runtime_environment_sha256": next(
                iter(evaluator_environments)
            ),
            "sampling_runtime_environment_sha256": next(
                iter(sampling_environments)
            ),
        },
        "estimands": {
            "cofitok_fid_delta_base256_minus_base128": cofitok_delta,
            "dense_fid_delta_base256_minus_base128": dense_delta,
            "capacity_by_factorization_fid_interaction": interaction,
            "cofitok_fid_relative_change": cofitok_delta
            / max(arms["base128_cofitok"]["fid"], 1e-12),
            "dense_fid_relative_change": dense_delta
            / max(arms["base128_dense_identity"]["fid"], 1e-12),
        },
        "decision": {
            "shared_strict_fid_improvement": shared_improvement,
            "cofitok_mechanism_invariants_valid": mechanism_valid,
            "capacity_supported": capacity_supported,
            "recommendation": recommendation,
        },
        "claim_policy": {
            "role": "non_claim_capacity_causal_diagnostic",
            "sample_count_per_arm": CAPACITY_PROBE_SAMPLES_PER_ARM,
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(CAPACITY_PROBE_RESULT_BOUNDARY),
    }
