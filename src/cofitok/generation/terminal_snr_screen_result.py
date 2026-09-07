"""Scientific decision and physical replay for the terminal-SNR screen."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from cofitok.generation.exposure_capacity_authorization import identity, read_object
from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
    SCREEN_THRESHOLDS,
    validate_terminal_snr_screen_preparation_contract,
)
from cofitok.generation.terminal_snr_screen_arm import (
    validate_terminal_snr_screen_arm_validation,
)
from cofitok.generation.terminal_snr_screen_execution import (
    validate_terminal_snr_screen_launch_receipt_contract,
)


RESULT_SCHEMA = "cofitok_generation_terminal_snr_screen_result_v1"
RESULT_ROLE = "source_bound_terminal_snr_screen_scientific_result"
VALIDATION_SCHEMA = "cofitok_generation_terminal_snr_screen_result_validation_v1"
VALIDATION_ROLE = "content_addressed_terminal_snr_screen_result_validation"
RESULT_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "frozen_confirmation_preparation_allowed_if_pass": True,
    "frozen_confirmation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
VALIDATION_BOUNDARY = {
    "receipt_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "frozen_confirmation_launch_allowed": False,
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


def _hex(value: Any, *, length: int) -> bool:
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
        or not _hex(row.get("sha256"), length=64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git fields differ")
    if (
        not _hex(row.get("revision"), length=40)
        or not _hex(row.get("tree"), length=40)
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify an exact clean checkout")
    return copy.deepcopy(row)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite range")
    return result


def _summary(report: Mapping[str, Any], arm: str) -> dict[str, Any]:
    row = _object(report, f"{arm} arm validation")
    distribution = _object(row.get("distribution"), f"{arm} distribution")
    metrics = _object(distribution.get("metrics"), f"{arm} distribution metrics")
    class_fidelity = _object(row.get("class_fidelity"), f"{arm} class fidelity")
    class_metrics = _object(
        class_fidelity.get("metrics"), f"{arm} class fidelity metrics"
    )
    rollout = _object(row.get("rollout"), f"{arm} rollout")
    clipping = _object(
        rollout.get("terminal_raw_x0_clipping"), f"{arm} terminal clipping"
    )
    training = _object(row.get("training"), f"{arm} training")
    training_validation = _object(
        training.get("validation"), f"{arm} training validation"
    )
    sampling = _object(row.get("sampling"), f"{arm} sampling")
    result: dict[str, Any] = {
        "arm": arm,
        "condition": row.get("condition"),
        "method": row.get("method"),
        "endpoint_fraction": _finite(
            row.get("endpoint_fraction"), f"{arm} endpoint fraction", minimum=0.0
        ),
        "checkpoint": copy.deepcopy(training.get("checkpoint")),
        "validation_epsilon_mse": _finite(
            training.get("validation_epsilon_mse"),
            f"{arm} validation epsilon MSE",
            minimum=0.0,
        ),
        "sample_count": int(sampling.get("sample_count", -1)),
        "sample_set_sha256": sampling.get("sample_set_sha256"),
        "real_set": copy.deepcopy(
            _object(distribution.get("real_set"), f"{arm} real set")
        ),
        "classifier": copy.deepcopy(
            _object(class_fidelity.get("classifier"), f"{arm} classifier")
        ),
        "fid": _finite(metrics.get("fid"), f"{arm} FID", minimum=0.0),
        "inception_score_mean": _finite(
            metrics.get("inception_score_mean"), f"{arm} IS", minimum=0.0
        ),
        "precision": _finite(
            metrics.get("precision"), f"{arm} precision", minimum=0.0
        ),
        "recall": _finite(metrics.get("recall"), f"{arm} recall", minimum=0.0),
        "class_top1": _finite(
            class_metrics.get("top1_accuracy"), f"{arm} top1", minimum=0.0
        ),
        "class_top5": _finite(
            class_metrics.get("top5_accuracy"), f"{arm} top5", minimum=0.0
        ),
        "terminal_raw_x0_clip_fraction": _finite(
            clipping.get("raw_x0_clip_fraction"),
            f"{arm} terminal raw-x0 clip fraction",
            minimum=0.0,
        ),
        "training_runtime_environment_sha256": training_validation.get(
            "runtime_environment_sha256"
        ),
        "sampling_runtime_environment_sha256": sampling.get(
            "runtime_environment_sha256"
        ),
        "distribution_runtime_environment_sha256": distribution.get(
            "runtime_environment_sha256"
        ),
        "class_fidelity_runtime_environment_sha256": class_fidelity.get(
            "runtime_environment_sha256"
        ),
        "execution_git": copy.deepcopy(row.get("execution_git")),
    }
    if result["sample_count"] != 1_000 or not _hex(
        result["sample_set_sha256"], length=64
    ):
        raise ValueError(f"{arm} sample evidence differs")
    if result["precision"] > 1.0 or result["recall"] > 1.0:
        raise ValueError(f"{arm} distribution probability exceeds one")
    if result["class_top1"] > 1.0 or result["class_top5"] > 1.0:
        raise ValueError(f"{arm} class probability exceeds one")
    if result["terminal_raw_x0_clip_fraction"] > 1.0:
        raise ValueError(f"{arm} clip fraction exceeds one")
    if row.get("method") == "cofitok":
        checkpoint = _object(
            row.get("checkpoint_evaluation"), f"{arm} checkpoint evaluation"
        )
        diagnostic = _object(checkpoint.get("summary"), f"{arm} diagnostics")
        utilization = _object(
            diagnostic.get("utilization"), f"{arm} utilization"
        )
        result["cofitok_diagnostics"] = {
            "ordered_rank_by_path_auc": int(
                diagnostic.get("ordered_rank_by_path_auc", -1)
            ),
            "zero_token_max_abs": _finite(
                diagnostic.get("zero_token_max_abs"),
                f"{arm} zero-token maximum",
                minimum=0.0,
            ),
            "shuffled_to_ordered_endpoint_ratio": _finite(
                diagnostic.get("shuffled_to_ordered_endpoint_ratio"),
                f"{arm} shuffle ratio",
                minimum=0.0,
            ),
            "coarse_token_energy_ratio": _finite(
                utilization.get("coarse_token_energy_ratio"),
                f"{arm} coarse energy",
                minimum=0.0,
            ),
            "tail_two_energy_ratio": _finite(
                utilization.get("tail_two_energy_ratio"),
                f"{arm} tail-two energy",
                minimum=0.0,
            ),
        }
    return result


def _cross_arm_evidence(
    summaries: Mapping[str, Mapping[str, Any]],
    expected_runtime_environment_sha256: str,
) -> dict[str, Any]:
    if not _hex(expected_runtime_environment_sha256, length=64):
        raise ValueError("terminal-SNR launch runtime identity is malformed")
    if set(summaries) != set(ARM_NAMES):
        raise ValueError("terminal-SNR cross-arm evidence arm set differs")

    real_sets: dict[str, dict[str, Any]] = {}
    classifiers: dict[str, dict[str, Any]] = {}
    sample_shas: dict[str, str] = {}
    checkpoint_shas: dict[str, str] = {}
    runtime_shas: dict[str, dict[str, str]] = {}
    generation_runtime_fields = (
        "training_runtime_environment_sha256",
        "sampling_runtime_environment_sha256",
    )
    evaluator_runtime_fields = (
        "distribution_runtime_environment_sha256",
        "class_fidelity_runtime_environment_sha256",
    )
    for arm in ARM_NAMES:
        summary = _object(summaries[arm], f"{arm} summary")
        real_sets[arm] = _object(summary.get("real_set"), f"{arm} real set")
        classifiers[arm] = _object(
            summary.get("classifier"), f"{arm} classifier"
        )
        sample_sha = summary.get("sample_set_sha256")
        if not _hex(sample_sha, length=64):
            raise ValueError(f"{arm} sample-set identity is malformed")
        sample_shas[arm] = sample_sha
        checkpoint = _object(summary.get("checkpoint"), f"{arm} checkpoint")
        checkpoint_sha = checkpoint.get("sha256")
        if not _hex(checkpoint_sha, length=64):
            raise ValueError(f"{arm} checkpoint identity is malformed")
        checkpoint_shas[arm] = checkpoint_sha
        runtime_shas[arm] = {}
        for field in generation_runtime_fields:
            runtime_sha = summary.get(field)
            if runtime_sha != expected_runtime_environment_sha256:
                raise ValueError(
                    f"{arm} {field} runtime differs from launch"
                )
            runtime_shas[arm][field] = runtime_sha
        evaluator_runtimes: list[str] = []
        for field in evaluator_runtime_fields:
            runtime_sha = summary.get(field)
            if not _hex(runtime_sha, length=64):
                raise ValueError(f"{arm} {field} runtime is malformed")
            runtime_shas[arm][field] = runtime_sha
            evaluator_runtimes.append(runtime_sha)
        if len(set(evaluator_runtimes)) != 1:
            raise ValueError(f"{arm} evaluator runtime identities differ")

    reference_real_set = real_sets[ARM_NAMES[0]]
    if not _hex(reference_real_set.get("sha256"), length=64) or any(
        real_sets[arm] != reference_real_set for arm in ARM_NAMES[1:]
    ):
        raise ValueError("terminal-SNR arms do not share one identical real set")
    reference_classifier = classifiers[ARM_NAMES[0]]
    if not reference_classifier or any(
        classifiers[arm] != reference_classifier for arm in ARM_NAMES[1:]
    ):
        raise ValueError("terminal-SNR arms do not share one identical classifier")
    if len(set(sample_shas.values())) != len(ARM_NAMES):
        raise ValueError("terminal-SNR sample sets are missing or duplicated")
    if len(set(checkpoint_shas.values())) != len(ARM_NAMES):
        raise ValueError("terminal-SNR checkpoints are missing or duplicated")
    evaluator_runtime_shas = {
        runtimes["distribution_runtime_environment_sha256"]
        for runtimes in runtime_shas.values()
    }
    if len(evaluator_runtime_shas) != 1:
        raise ValueError("terminal-SNR evaluator runtime differs across arms")

    return {
        "generation_runtime_environment_sha256": (
            expected_runtime_environment_sha256
        ),
        "evaluator_runtime_environment_sha256": next(
            iter(evaluator_runtime_shas)
        ),
        "runtime_environment_sha256_by_arm": runtime_shas,
        "real_set": copy.deepcopy(reference_real_set),
        "real_set_identity_sha256": _canonical_sha256(reference_real_set),
        "classifier": copy.deepcopy(reference_classifier),
        "classifier_identity_sha256": _canonical_sha256(reference_classifier),
        "sample_set_sha256_by_arm": sample_shas,
        "sample_sets_unique": True,
        "checkpoint_sha256_by_arm": checkpoint_shas,
        "checkpoints_unique": True,
    }


def _check(name: str, *, actual: float | int, operator: str, threshold: float | int) -> dict[str, Any]:
    if operator == ">=":
        passed = actual >= threshold
    elif operator == "<=":
        passed = actual <= threshold
    elif operator == "==":
        passed = actual == threshold
    else:  # pragma: no cover - all operators are internal constants
        raise ValueError(f"unknown terminal-SNR check operator: {operator}")
    return {
        "name": name,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _evaluate(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        control = _object(summaries[f"control_{method}"], f"control {method}")
        endpoint = _object(
            summaries[f"endpoint0975_{method}"], f"endpoint {method}"
        )
        if control["fid"] <= 0.0:
            raise ValueError(f"control {method} FID must be positive")
        values = {
            "relative_fid_improvement": (control["fid"] - endpoint["fid"])
            / control["fid"],
            "precision_regression": control["precision"] - endpoint["precision"],
            "recall_regression": control["recall"] - endpoint["recall"],
            "class_top1_regression": control["class_top1"]
            - endpoint["class_top1"],
            "class_top5_regression": control["class_top5"]
            - endpoint["class_top5"],
            "terminal_raw_x0_clip_fraction_reduction": control[
                "terminal_raw_x0_clip_fraction"
            ]
            - endpoint["terminal_raw_x0_clip_fraction"],
            "endpoint_terminal_raw_x0_clip_fraction": endpoint[
                "terminal_raw_x0_clip_fraction"
            ],
        }
        comparisons[method] = values
        checks.extend(
            [
                _check(
                    f"{method}.relative_fid_improvement",
                    actual=values["relative_fid_improvement"],
                    operator=">=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_min_relative_fid_improvement"
                    ],
                ),
                _check(
                    f"{method}.precision_regression",
                    actual=values["precision_regression"],
                    operator="<=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_max_precision_regression"
                    ],
                ),
                _check(
                    f"{method}.recall_regression",
                    actual=values["recall_regression"],
                    operator="<=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_max_recall_regression"
                    ],
                ),
                _check(
                    f"{method}.class_top1_regression",
                    actual=values["class_top1_regression"],
                    operator="<=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_max_class_top1_regression"
                    ],
                ),
                _check(
                    f"{method}.class_top5_regression",
                    actual=values["class_top5_regression"],
                    operator="<=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_max_class_top5_regression"
                    ],
                ),
                _check(
                    f"{method}.endpoint_terminal_raw_x0_clip_fraction",
                    actual=values["endpoint_terminal_raw_x0_clip_fraction"],
                    operator="<=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_max_terminal_raw_x0_clip_fraction"
                    ],
                ),
                _check(
                    f"{method}.terminal_raw_x0_clip_fraction_reduction",
                    actual=values[
                        "terminal_raw_x0_clip_fraction_reduction"
                    ],
                    operator=">=",
                    threshold=SCREEN_THRESHOLDS[
                        "both_methods_min_terminal_raw_x0_clip_fraction_reduction"
                    ],
                ),
            ]
        )
    endpoint_diagnostics = _object(
        summaries["endpoint0975_cofitok"].get("cofitok_diagnostics"),
        "endpoint CoFiTok diagnostics",
    )
    checks.extend(
        [
            _check(
                "cofitok.ordered_rank_by_path_auc",
                actual=endpoint_diagnostics["ordered_rank_by_path_auc"],
                operator="==",
                threshold=SCREEN_THRESHOLDS["cofitok_ordered_rank_by_path_auc"],
            ),
            _check(
                "cofitok.coarse_token_energy_ratio",
                actual=endpoint_diagnostics["coarse_token_energy_ratio"],
                operator=">=",
                threshold=SCREEN_THRESHOLDS[
                    "cofitok_min_coarse_token_energy_ratio"
                ],
            ),
            _check(
                "cofitok.tail_two_energy_ratio",
                actual=endpoint_diagnostics["tail_two_energy_ratio"],
                operator="<=",
                threshold=SCREEN_THRESHOLDS[
                    "cofitok_max_tail_two_energy_ratio"
                ],
            ),
            _check(
                "cofitok.zero_token_max_abs",
                actual=endpoint_diagnostics["zero_token_max_abs"],
                operator="<=",
                threshold=SCREEN_THRESHOLDS["cofitok_zero_token_max_abs"],
            ),
            _check(
                "cofitok.shuffled_to_ordered_endpoint_ratio",
                actual=endpoint_diagnostics[
                    "shuffled_to_ordered_endpoint_ratio"
                ],
                operator=">=",
                threshold=SCREEN_THRESHOLDS[
                    "cofitok_min_shuffled_to_ordered_endpoint_ratio"
                ],
            ),
        ]
    )
    failed = [row["name"] for row in checks if row["pass"] is not True]
    return {
        "comparisons": comparisons,
        "checks": checks,
        "failed_checks": failed,
        "screen_pass": not failed,
    }


def build_terminal_snr_screen_result(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    arm_validations: Mapping[str, Mapping[str, Any]],
    arm_validation_identities: Mapping[str, Mapping[str, Any]],
    result_git: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_terminal_snr_screen_preparation_contract(preparation)
    prep_id = _identity(preparation_identity, "terminal-SNR preparation")
    launch_id = _identity(launch_receipt_identity, "terminal-SNR launch receipt")
    execution = _object(
        launch_receipt.get("execution_checkout"), "terminal-SNR execution checkout"
    )
    launch = validate_terminal_snr_screen_launch_receipt_contract(
        launch_receipt, expected_execution_checkout=execution
    )
    result_checkout = _git(result_git, "terminal-SNR result checkout")
    if result_checkout != execution:
        raise ValueError("terminal-SNR result Git differs from launch checkout")
    if launch.get("source_evidence", {}).get("preparation") != prep_id:
        raise ValueError("terminal-SNR launch binds another preparation")
    if set(arm_validations) != set(ARM_NAMES) or set(
        arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("terminal-SNR result arm set differs")
    summaries: dict[str, dict[str, Any]] = {}
    arm_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        validation = validate_terminal_snr_screen_arm_validation(
            arm_validations[arm]
        )
        arm_id = _identity(
            arm_validation_identities[arm], f"{arm} arm validation"
        )
        if validation.get("sources", {}).get("launch_receipt") != launch_id:
            raise ValueError(f"{arm} validation binds another launch receipt")
        if validation.get("execution_git") != execution:
            raise ValueError(f"{arm} execution Git differs")
        summaries[arm] = _summary(validation, arm)
        arm_ids[arm] = arm_id
    runtime_environment_sha256 = str(launch["runtime_environment_sha256"])
    cross_arm = _cross_arm_evidence(summaries, runtime_environment_sha256)
    evaluated = _evaluate(summaries)
    passed = bool(evaluated["screen_pass"])
    report = {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "pass" if passed else "hold",
        "terminal_status": "hold",
        "screen_pass": passed,
        "generation_advantage_proven": False,
        "decision": (
            "prepare_separately_authorized_frozen_10k_confirmation"
            if passed
            else "hold_terminal_snr_intervention"
        ),
        "result_git": result_checkout,
        "runtime_environment_sha256": runtime_environment_sha256,
        "source_evidence": {
            "preparation": prep_id,
            "launch_receipt": launch_id,
            "arm_validations": arm_ids,
        },
        "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
        "arm_summaries": summaries,
        "cross_arm_evidence": cross_arm,
        "comparisons": evaluated["comparisons"],
        "checks": evaluated["checks"],
        "failed_checks": evaluated["failed_checks"],
        "next_stage": {
            "route": "frozen_10k_confirmation_preparation" if passed else "hold",
            "frozen_confirmation_preparation_allowed": passed,
            "frozen_confirmation_launch_allowed": False,
            "separate_exact_authorization_required": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }
    return validate_terminal_snr_screen_result_contract(report)


def validate_terminal_snr_screen_result_contract(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "terminal-SNR result")
    summaries = _object(row.get("arm_summaries"), "terminal-SNR arm summaries")
    if set(summaries) != set(ARM_NAMES):
        raise ValueError("terminal-SNR result summary arm set differs")
    runtime_environment_sha256 = row.get("runtime_environment_sha256")
    cross_arm = _cross_arm_evidence(
        summaries, str(runtime_environment_sha256)
    )
    evaluated = _evaluate(summaries)
    passed = bool(evaluated["screen_pass"])
    expected_status = "pass" if passed else "hold"
    expected_decision = (
        "prepare_separately_authorized_frozen_10k_confirmation"
        if passed
        else "hold_terminal_snr_intervention"
    )
    next_stage = _object(row.get("next_stage"), "terminal-SNR next stage")
    sources = _object(row.get("source_evidence"), "terminal-SNR result sources")
    _identity(sources.get("preparation"), "terminal-SNR preparation")
    _identity(sources.get("launch_receipt"), "terminal-SNR launch receipt")
    arm_ids = _object(sources.get("arm_validations"), "terminal-SNR arm sources")
    if set(arm_ids) != set(ARM_NAMES):
        raise ValueError("terminal-SNR result source arm set differs")
    for arm in ARM_NAMES:
        _identity(arm_ids[arm], f"{arm} result source")
    if (
        row.get("schema_version") != RESULT_SCHEMA
        or row.get("role") != RESULT_ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status") != expected_status
        or row.get("terminal_status") != "hold"
        or row.get("screen_pass") is not passed
        or row.get("generation_advantage_proven") is not False
        or row.get("decision") != expected_decision
        or row.get("thresholds") != SCREEN_THRESHOLDS
        or row.get("comparisons") != evaluated["comparisons"]
        or row.get("checks") != evaluated["checks"]
        or row.get("failed_checks") != evaluated["failed_checks"]
        or row.get("cross_arm_evidence") != cross_arm
        or row.get("authorization_boundary") != RESULT_BOUNDARY
        or next_stage.get("route")
        != ("frozen_10k_confirmation_preparation" if passed else "hold")
        or next_stage.get("frozen_confirmation_preparation_allowed") is not passed
        or next_stage.get("frozen_confirmation_launch_allowed") is not False
        or next_stage.get("separate_exact_authorization_required") is not True
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("terminal-SNR result contract differs")
    _git(row.get("result_git"), "terminal-SNR result Git")
    return copy.deepcopy(row)


def _load_bound(identity_row: Mapping[str, Any], name: str) -> dict[str, Any]:
    expected = _identity(identity_row, name)
    actual_id = identity(expected["path"])
    if actual_id != expected:
        raise ValueError(f"{name} physical identity differs")
    return read_object(expected["path"], name=name)


def replay_terminal_snr_screen_result(report: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_terminal_snr_screen_result_contract(report)
    sources = _object(row["source_evidence"], "terminal-SNR result sources")
    preparation = _load_bound(sources["preparation"], "terminal-SNR preparation")
    launch = _load_bound(sources["launch_receipt"], "terminal-SNR launch receipt")
    arm_ids = _object(sources["arm_validations"], "terminal-SNR arm sources")
    arms = {
        arm: _load_bound(arm_ids[arm], f"{arm} arm validation")
        for arm in ARM_NAMES
    }
    return build_terminal_snr_screen_result(
        preparation=preparation,
        preparation_identity=sources["preparation"],
        launch_receipt=launch,
        launch_receipt_identity=sources["launch_receipt"],
        arm_validations=arms,
        arm_validation_identities=arm_ids,
        result_git=row["result_git"],
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_terminal_snr_screen_validation_receipt(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_terminal_snr_screen_result_contract(result)
    replayed = replay_terminal_snr_screen_result(validated)
    if replayed != validated:
        raise ValueError("terminal-SNR result differs from physical replay")
    result_id = _identity(result_identity, "terminal-SNR result")
    validator = _git(validator_git, "terminal-SNR result validator")
    basis = {
        "result": result_id,
        "result_git": validated["result_git"],
        "validator_git": validator,
        "screen_pass": validated["screen_pass"],
        "failed_checks": validated["failed_checks"],
        "source_evidence": validated["source_evidence"],
    }
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": validated["scientific_status"],
        "screen_pass": validated["screen_pass"],
        "result": result_id,
        "result_git": copy.deepcopy(validated["result_git"]),
        "validator_git": validator,
        "failed_checks": copy.deepcopy(validated["failed_checks"]),
        "source_evidence": copy.deepcopy(validated["source_evidence"]),
        "validation_basis_sha256": _canonical_sha256(basis),
        "generation_advantage_proven": False,
        "authorization_boundary": copy.deepcopy(VALIDATION_BOUNDARY),
    }


def validate_terminal_snr_screen_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "terminal-SNR result validation")
    expected = build_terminal_snr_screen_validation_receipt(
        result=result,
        result_identity=result_identity,
        validator_git=validator_git,
    )
    if (
        row.get("schema_version") != VALIDATION_SCHEMA
        or row.get("role") != VALIDATION_ROLE
        or row.get("authorization_boundary") != VALIDATION_BOUNDARY
        or row != expected
    ):
        raise ValueError("terminal-SNR result validation receipt differs")
    return copy.deepcopy(row)


__all__ = [
    "RESULT_BOUNDARY",
    "RESULT_ROLE",
    "RESULT_SCHEMA",
    "VALIDATION_BOUNDARY",
    "VALIDATION_ROLE",
    "VALIDATION_SCHEMA",
    "build_terminal_snr_screen_result",
    "build_terminal_snr_screen_validation_receipt",
    "replay_terminal_snr_screen_result",
    "validate_terminal_snr_screen_result_contract",
    "validate_terminal_snr_screen_validation_receipt",
]
