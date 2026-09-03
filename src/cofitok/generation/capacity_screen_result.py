"""Scientific result for the fresh four-arm capacity screen.

The 1K screen is intentionally diagnostic.  It may prepare a separately
authorized 10K-per-arm confirmation only when larger capacity improves both
methods directionally, CoFiTok does not regress specifically relative to the
dense control, and both CoFiTok capacity arms retain the ordered restricted
factorization mechanism.  It never authorizes 300K training.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    CAPACITY_NAMES,
    METHOD_NAMES,
    validate_capacity_screen_preparation_contract,
)
from cofitok.generation.capacity_screen_arm import (
    ARM_VALIDATION_ROLE,
    ARM_VALIDATION_SCHEMA,
    validate_capacity_screen_arm_validation,
)
from cofitok.generation.capacity_screen_execution import (
    validate_capacity_screen_launch_receipt_contract,
)


RESULT_SCHEMA = "cofitok_generation_capacity_screen_result_v1"
RESULT_ROLE = "source_bound_four_arm_capacity_screen_scientific_result"
SCREEN_THRESHOLDS = {
    "max_precision_drop": 0.05,
    "max_top1_drop": 0.01,
    "max_top5_drop": 0.02,
    "max_predicted_class_fraction_drop": 0.05,
    "max_normalized_predicted_entropy_drop": 0.05,
    "max_cofitok_relative_fid_interaction": 0.05,
    "max_cofitok_recall_interaction_regression": 0.02,
    "max_cofitok_top1_interaction_regression": 0.01,
    "max_endpoint_ratio": 1.05,
    "max_validation_ratio": 1.05,
    "max_reconstruction_ratio": 1.05,
    "max_predicted_x0_high_frequency_ratio": 1.50,
    "max_tail_two_energy_ratio": 0.65,
    "max_single_token_energy_ratio": 0.35,
    "min_coarse_token_energy_ratio": 0.10,
    "required_ordered_rank": 1,
    "min_order_count": 6,
    "max_zero_token_abs": 1e-8,
    "min_shuffle_mismatch_ratio": 2.0,
}
RESULT_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "capacity_confirmation_preparation_allowed": "derived_from_result",
    "capacity_confirmation_launch_allowed": False,
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
        or not isinstance(row.get("sha256"), str)
        or len(row["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in row["sha256"])
    ):
        raise ValueError(f"{name} identity is malformed")
    return {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not isinstance(row.get("revision"), str)
        or len(row["revision"]) != 40
        or not isinstance(row.get("tree"), str)
        or len(row["tree"]) != 40
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return copy.deepcopy(row)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} is outside its finite domain")
    return result


def _ratio(numerator: float, denominator: float, name: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return numerator / denominator


def _check(
    name: str,
    passed: bool,
    observed: Any,
    comparison: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "comparison": comparison,
        "threshold": threshold,
    }


def _arm_summary(report: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if (
        report.get("schema_version") != ARM_VALIDATION_SCHEMA
        or report.get("role") != ARM_VALIDATION_ROLE
        or report.get("status") != "pass"
        or report.get("arm") != arm
    ):
        raise ValueError(f"{arm} arm validation contract differs")
    distribution = _object(report.get("distribution"), f"{arm} distribution")
    class_fidelity = _object(report.get("class_fidelity"), f"{arm} class fidelity")
    training = _object(report.get("training"), f"{arm} training")
    checkpoint = _object(
        report.get("checkpoint_evaluation"), f"{arm} checkpoint evaluation"
    )
    rollout = _object(report.get("rollout"), f"{arm} rollout")
    return {
        "distribution": copy.deepcopy(
            _object(distribution.get("metrics"), f"{arm} distribution metrics")
        ),
        "class_fidelity": copy.deepcopy(
            _object(class_fidelity.get("metrics"), f"{arm} class metrics")
        ),
        "validation_epsilon_mse": _finite(
            training.get("validation_epsilon_mse"),
            f"{arm} validation epsilon MSE",
            minimum=0.0,
        ),
        "checkpoint": copy.deepcopy(
            _object(checkpoint.get("summary"), f"{arm} checkpoint summary")
        ),
        "rollout": copy.deepcopy(
            _object(rollout.get("summary"), f"{arm} rollout summary")
        ),
        "sample_set_sha256": _object(
            report.get("sampling"), f"{arm} sampling"
        ).get("sample_set_sha256"),
        "real_set": copy.deepcopy(distribution.get("real_set")),
    }


def _method_values(summary: Mapping[str, Any]) -> dict[str, float]:
    distribution = _object(summary.get("distribution"), "arm distribution summary")
    class_fidelity = _object(
        summary.get("class_fidelity"), "arm class-fidelity summary"
    )
    return {
        "fid": _finite(distribution.get("fid"), "screen FID", minimum=0.0),
        "precision": _finite(
            distribution.get("precision"), "screen precision", minimum=0.0
        ),
        "recall": _finite(
            distribution.get("recall"), "screen recall", minimum=0.0
        ),
        "top1": _finite(
            class_fidelity.get("top1_accuracy"), "screen top1", minimum=0.0
        ),
        "top5": _finite(
            class_fidelity.get("top5_accuracy"), "screen top5", minimum=0.0
        ),
        "predicted_class_fraction": _finite(
            class_fidelity.get("predicted_class_fraction"),
            "screen predicted class fraction",
            minimum=0.0,
        ),
        "normalized_predicted_class_entropy": _finite(
            class_fidelity.get("normalized_predicted_class_entropy"),
            "screen normalized class entropy",
            minimum=0.0,
        ),
    }


def _capacity_pair_checks(
    summaries: Mapping[str, Mapping[str, Any]],
    capacity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cofitok = summaries[f"{capacity}_cofitok"]
    dense = summaries[f"{capacity}_dense_identity"]
    cofitok_checkpoint = _object(cofitok.get("checkpoint"), "CoFiTok checkpoint summary")
    dense_checkpoint = _object(dense.get("checkpoint"), "dense checkpoint summary")
    cofitok_rollout = _object(cofitok.get("rollout"), "CoFiTok rollout summary")
    dense_rollout = _object(dense.get("rollout"), "dense rollout summary")
    endpoint_ratio = _ratio(
        _finite(
            cofitok_checkpoint.get("ordered_endpoint_clean_mse"),
            f"{capacity} CoFiTok endpoint",
            minimum=0.0,
        ),
        _finite(
            dense_checkpoint.get("ordered_endpoint_clean_mse"),
            f"{capacity} dense endpoint",
            minimum=0.0,
        ),
        f"{capacity} endpoint",
    )
    validation_ratio = _ratio(
        _finite(cofitok.get("validation_epsilon_mse"), "CoFiTok validation", minimum=0.0),
        _finite(dense.get("validation_epsilon_mse"), "dense validation", minimum=0.0),
        f"{capacity} validation",
    )
    reconstruction_ratio = _ratio(
        _finite(
            cofitok_rollout.get("final_reconstruction_x0_mse"),
            "CoFiTok reconstruction",
            minimum=0.0,
        ),
        _finite(
            dense_rollout.get("final_reconstruction_x0_mse"),
            "dense reconstruction",
            minimum=0.0,
        ),
        f"{capacity} reconstruction",
    )
    cofitok_hf = _object(
        cofitok_rollout.get("predicted_x0_high_frequency_ratio"),
        f"{capacity} CoFiTok high-frequency rollout",
    )
    dense_hf = _object(
        dense_rollout.get("predicted_x0_high_frequency_ratio"),
        f"{capacity} dense high-frequency rollout",
    )
    if set(cofitok_hf) != set(dense_hf) or not cofitok_hf:
        raise ValueError(f"{capacity} rollout timestep set differs")
    high_frequency = {
        timestep: _ratio(
            _finite(cofitok_hf[timestep], f"{capacity} CoFiTok HF {timestep}", minimum=0.0),
            _finite(dense_hf[timestep], f"{capacity} dense HF {timestep}", minimum=0.0),
            f"{capacity} high-frequency {timestep}",
        )
        for timestep in sorted(cofitok_hf, key=int)
    }
    peak_hf = max(high_frequency.values())
    utilization = _object(
        cofitok_checkpoint.get("utilization"), f"{capacity} CoFiTok utilization"
    )
    mechanism = {
        "ordered_rank": int(cofitok_checkpoint.get("ordered_rank_by_path_auc", -1)),
        "order_count": int(cofitok_checkpoint.get("order_count", -1)),
        "coarse_token_energy_ratio": _finite(
            utilization.get("coarse_token_energy_ratio"),
            f"{capacity} coarse-token energy",
            minimum=0.0,
        ),
        "tail_two_energy_ratio": _finite(
            utilization.get("tail_two_energy_ratio"),
            f"{capacity} tail-two energy",
            minimum=0.0,
        ),
        "max_single_token_energy_ratio": _finite(
            utilization.get("max_single_token_energy_ratio"),
            f"{capacity} max token energy",
            minimum=0.0,
        ),
        "zero_token_max_abs": _finite(
            cofitok_checkpoint.get("zero_token_max_abs"),
            f"{capacity} zero token",
            minimum=0.0,
        ),
        "shuffle_mismatch_ratio": _finite(
            cofitok_checkpoint.get("shuffled_to_ordered_endpoint_ratio"),
            f"{capacity} shuffle mismatch",
            minimum=0.0,
        ),
    }
    checks = [
        _check(
            f"{capacity}_endpoint_non_regression",
            endpoint_ratio <= SCREEN_THRESHOLDS["max_endpoint_ratio"],
            endpoint_ratio,
            "<=",
            SCREEN_THRESHOLDS["max_endpoint_ratio"],
        ),
        _check(
            f"{capacity}_validation_non_regression",
            validation_ratio <= SCREEN_THRESHOLDS["max_validation_ratio"],
            validation_ratio,
            "<=",
            SCREEN_THRESHOLDS["max_validation_ratio"],
        ),
        _check(
            f"{capacity}_reconstruction_non_regression",
            reconstruction_ratio <= SCREEN_THRESHOLDS["max_reconstruction_ratio"],
            reconstruction_ratio,
            "<=",
            SCREEN_THRESHOLDS["max_reconstruction_ratio"],
        ),
        _check(
            f"{capacity}_predicted_x0_high_frequency",
            peak_hf <= SCREEN_THRESHOLDS["max_predicted_x0_high_frequency_ratio"],
            peak_hf,
            "<=",
            SCREEN_THRESHOLDS["max_predicted_x0_high_frequency_ratio"],
        ),
        _check(
            f"{capacity}_ordered_rank",
            mechanism["ordered_rank"] == SCREEN_THRESHOLDS["required_ordered_rank"],
            mechanism["ordered_rank"],
            "==",
            SCREEN_THRESHOLDS["required_ordered_rank"],
        ),
        _check(
            f"{capacity}_order_coverage",
            mechanism["order_count"] >= SCREEN_THRESHOLDS["min_order_count"],
            mechanism["order_count"],
            ">=",
            SCREEN_THRESHOLDS["min_order_count"],
        ),
        _check(
            f"{capacity}_coarse_token_utilization",
            mechanism["coarse_token_energy_ratio"]
            >= SCREEN_THRESHOLDS["min_coarse_token_energy_ratio"],
            mechanism["coarse_token_energy_ratio"],
            ">=",
            SCREEN_THRESHOLDS["min_coarse_token_energy_ratio"],
        ),
        _check(
            f"{capacity}_tail_two_energy",
            mechanism["tail_two_energy_ratio"]
            <= SCREEN_THRESHOLDS["max_tail_two_energy_ratio"],
            mechanism["tail_two_energy_ratio"],
            "<=",
            SCREEN_THRESHOLDS["max_tail_two_energy_ratio"],
        ),
        _check(
            f"{capacity}_single_token_energy",
            mechanism["max_single_token_energy_ratio"]
            <= SCREEN_THRESHOLDS["max_single_token_energy_ratio"],
            mechanism["max_single_token_energy_ratio"],
            "<=",
            SCREEN_THRESHOLDS["max_single_token_energy_ratio"],
        ),
        _check(
            f"{capacity}_zero_token",
            mechanism["zero_token_max_abs"]
            <= SCREEN_THRESHOLDS["max_zero_token_abs"],
            mechanism["zero_token_max_abs"],
            "<=",
            SCREEN_THRESHOLDS["max_zero_token_abs"],
        ),
        _check(
            f"{capacity}_shuffle_mismatch",
            mechanism["shuffle_mismatch_ratio"]
            >= SCREEN_THRESHOLDS["min_shuffle_mismatch_ratio"],
            mechanism["shuffle_mismatch_ratio"],
            ">=",
            SCREEN_THRESHOLDS["min_shuffle_mismatch_ratio"],
        ),
    ]
    return {
        "endpoint_ratio": endpoint_ratio,
        "validation_ratio": validation_ratio,
        "reconstruction_ratio": reconstruction_ratio,
        "predicted_x0_high_frequency_ratio": high_frequency,
        "peak_predicted_x0_high_frequency_ratio": peak_hf,
        "mechanism": mechanism,
    }, checks


def build_capacity_screen_result(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    arm_validations: Mapping[str, Mapping[str, Any]],
    arm_validation_identities: Mapping[str, Mapping[str, Any]],
    result_git: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_capacity_screen_preparation_contract(preparation)
    prep_id = _identity(preparation_identity, "capacity screen preparation")
    launch_id = _identity(launch_receipt_identity, "capacity screen launch receipt")
    execution = _git(
        launch_receipt.get("execution_checkout"), "capacity screen execution"
    )
    launch = validate_capacity_screen_launch_receipt_contract(
        launch_receipt,
        expected_execution_checkout=execution,
    )
    launch_sources = _object(
        launch.get("source_evidence"), "capacity launch source evidence"
    )
    if launch_sources.get("preparation") != prep_id:
        raise ValueError("capacity screen launch receipt binds another preparation")
    if set(arm_validations) != set(ARM_NAMES) or set(
        arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("capacity screen result arm set differs")
    normalized_arm_ids = {
        arm: _identity(arm_validation_identities[arm], f"{arm} validation")
        for arm in ARM_NAMES
    }
    validated_arms: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        report = validate_capacity_screen_arm_validation(arm_validations[arm])
        if report.get("execution_git") != execution:
            raise ValueError(f"{arm} validation uses another execution checkout")
        sources = _object(report.get("sources"), f"{arm} validation sources")
        if sources.get("launch_receipt") != launch_id:
            raise ValueError(f"{arm} validation uses another launch receipt")
        validated_arms[arm] = report
        summaries[arm] = _arm_summary(report, arm)

    real_set_shas = {
        _object(summary.get("real_set"), f"{arm} real set").get("sha256")
        for arm, summary in summaries.items()
    }
    if len(real_set_shas) != 1 or None in real_set_shas:
        raise ValueError("capacity screen arms do not share one real set")
    sample_shas = {summary.get("sample_set_sha256") for summary in summaries.values()}
    if len(sample_shas) != len(ARM_NAMES) or None in sample_shas:
        raise ValueError("capacity screen sample sets are missing or duplicated")

    values = {arm: _method_values(summary) for arm, summary in summaries.items()}
    deltas: dict[str, dict[str, float]] = {}
    checks: list[dict[str, Any]] = []
    for method in METHOD_NAMES:
        base = values[f"base128_{method}"]
        large = values[f"base256_{method}"]
        row = {metric: large[metric] - base[metric] for metric in base}
        row["relative_fid_change"] = large["fid"] / max(base["fid"], 1e-12) - 1.0
        deltas[method] = row
        checks.extend(
            [
                _check(
                    f"{method}_fid_improves_with_capacity",
                    row["relative_fid_change"] < 0.0,
                    row["relative_fid_change"],
                    "<",
                    0.0,
                ),
                _check(
                    f"{method}_recall_improves_with_capacity",
                    row["recall"] > 0.0,
                    row["recall"],
                    ">",
                    0.0,
                ),
                _check(
                    f"{method}_precision_non_regression",
                    row["precision"] >= -SCREEN_THRESHOLDS["max_precision_drop"],
                    row["precision"],
                    ">=",
                    -SCREEN_THRESHOLDS["max_precision_drop"],
                ),
                _check(
                    f"{method}_top1_non_regression",
                    row["top1"] >= -SCREEN_THRESHOLDS["max_top1_drop"],
                    row["top1"],
                    ">=",
                    -SCREEN_THRESHOLDS["max_top1_drop"],
                ),
                _check(
                    f"{method}_top5_non_regression",
                    row["top5"] >= -SCREEN_THRESHOLDS["max_top5_drop"],
                    row["top5"],
                    ">=",
                    -SCREEN_THRESHOLDS["max_top5_drop"],
                ),
                _check(
                    f"{method}_predicted_class_fraction_non_regression",
                    row["predicted_class_fraction"]
                    >= -SCREEN_THRESHOLDS["max_predicted_class_fraction_drop"],
                    row["predicted_class_fraction"],
                    ">=",
                    -SCREEN_THRESHOLDS["max_predicted_class_fraction_drop"],
                ),
                _check(
                    f"{method}_class_entropy_non_regression",
                    row["normalized_predicted_class_entropy"]
                    >= -SCREEN_THRESHOLDS[
                        "max_normalized_predicted_entropy_drop"
                    ],
                    row["normalized_predicted_class_entropy"],
                    ">=",
                    -SCREEN_THRESHOLDS[
                        "max_normalized_predicted_entropy_drop"
                    ],
                ),
            ]
        )

    interaction = {
        metric: deltas["cofitok"][metric] - deltas["dense_identity"][metric]
        for metric in deltas["cofitok"]
    }
    checks.extend(
        [
            _check(
                "cofitok_relative_fid_capacity_interaction",
                interaction["relative_fid_change"]
                <= SCREEN_THRESHOLDS["max_cofitok_relative_fid_interaction"],
                interaction["relative_fid_change"],
                "<=",
                SCREEN_THRESHOLDS["max_cofitok_relative_fid_interaction"],
            ),
            _check(
                "cofitok_recall_capacity_interaction",
                interaction["recall"]
                >= -SCREEN_THRESHOLDS[
                    "max_cofitok_recall_interaction_regression"
                ],
                interaction["recall"],
                ">=",
                -SCREEN_THRESHOLDS[
                    "max_cofitok_recall_interaction_regression"
                ],
            ),
            _check(
                "cofitok_top1_capacity_interaction",
                interaction["top1"]
                >= -SCREEN_THRESHOLDS[
                    "max_cofitok_top1_interaction_regression"
                ],
                interaction["top1"],
                ">=",
                -SCREEN_THRESHOLDS[
                    "max_cofitok_top1_interaction_regression"
                ],
            ),
        ]
    )

    pair_stability: dict[str, Any] = {}
    for capacity in CAPACITY_NAMES:
        pair_stability[capacity], pair_checks = _capacity_pair_checks(
            summaries, capacity
        )
        checks.extend(pair_checks)
    names = [check["name"] for check in checks]
    if len(names) != len(set(names)):
        raise ValueError("capacity screen check names are not unique")
    failed = [check["name"] for check in checks if check["passed"] is not True]
    passed = not failed
    scientific_status = "screen_pass" if passed else "hold"
    next_stage = {
        "route": "capacity_confirmation" if passed else "hold",
        "reason": (
            "both_methods_directionally_improve_with_capacity_and_mechanism_is_preserved"
            if passed
            else "capacity_screen_failed_one_or_more_predeclared_checks"
        ),
        "capacity_confirmation_preparation_allowed": passed,
        "capacity_confirmation_samples_per_arm": 10_000,
        "gpu_execution_allowed": False,
        "sampling_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "required_next_evidence": (
            "A separate exact authorization for 10K samples from all four frozen step-10K checkpoints."
            if passed
            else "A new matched intervention justified by the named failed checks."
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": scientific_status,
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "result_git": _git(result_git, "capacity screen result Git"),
        "source_evidence": {
            "preparation": prep_id,
            "launch_receipt": launch_id,
            "arm_validations": normalized_arm_ids,
        },
        "validated_preparation": {
            "scientific_status": prepared["scientific_status"],
            "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        },
        "arm_summaries": summaries,
        "within_method_capacity_deltas": deltas,
        "difference_in_differences": interaction,
        "pair_stability": pair_stability,
        "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
        "checks": checks,
        "failed_checks": failed,
        "next_stage": next_stage,
        "claim_policy": {
            "screen_is_diagnostic": True,
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
            "confirmation_required_before_large_capacity_readiness": True,
            "screen_cannot_authorize_300k": True,
        },
        "authorization_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }


def validate_capacity_screen_result_contract(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity screen result")
    checks = row.get("checks")
    failed = row.get("failed_checks")
    next_stage = _object(row.get("next_stage"), "capacity screen next stage")
    if not isinstance(checks, list) or not checks or not isinstance(failed, list):
        raise ValueError("capacity screen result checks are incomplete")
    computed_failed: list[str] = []
    names: set[str] = set()
    for value in checks:
        check = _object(value, "capacity screen check")
        name = check.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("capacity screen check names differ")
        if check.get("passed") not in {True, False}:
            raise ValueError(f"capacity screen check is not boolean: {name}")
        names.add(name)
        if check["passed"] is False:
            computed_failed.append(name)
    passed = not computed_failed
    expected_status = "screen_pass" if passed else "hold"
    if (
        row.get("schema_version") != RESULT_SCHEMA
        or row.get("role") != RESULT_ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status") != expected_status
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("thresholds") != SCREEN_THRESHOLDS
        or row.get("authorization_boundary") != RESULT_BOUNDARY
        or failed != computed_failed
        or next_stage.get("capacity_confirmation_preparation_allowed")
        is not passed
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("sampling_launch_allowed") is not False
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or set(_object(row.get("arm_summaries"), "capacity arm summaries"))
        != set(ARM_NAMES)
    ):
        raise ValueError("capacity screen result contract differs")
    _git(row.get("result_git"), "capacity screen result Git")
    sources = _object(row.get("source_evidence"), "capacity screen result sources")
    _identity(sources.get("preparation"), "capacity screen preparation")
    _identity(sources.get("launch_receipt"), "capacity screen launch receipt")
    arm_sources = _object(
        sources.get("arm_validations"), "capacity screen arm validations"
    )
    if set(arm_sources) != set(ARM_NAMES):
        raise ValueError("capacity screen result arm source set differs")
    for arm in ARM_NAMES:
        _identity(arm_sources[arm], f"{arm} validation identity")
    return copy.deepcopy(row)


__all__ = [
    "RESULT_BOUNDARY",
    "RESULT_ROLE",
    "RESULT_SCHEMA",
    "SCREEN_THRESHOLDS",
    "build_capacity_screen_result",
    "validate_capacity_screen_result_contract",
]
