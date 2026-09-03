"""Fail-closed scientific result for the four-arm 10K confirmation.

The confirmation must show that the base-256 pair clears scaling-grade
distribution-support and class-fidelity floors, that both methods benefit from
capacity, and that CoFiTok preserves its ordered restricted mechanism.  A pass
only permits preparation of a separate large-capacity readiness decision.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from cofitok.generation.capacity_confirmation import (
    validate_capacity_confirmation_preparation_contract,
)
from cofitok.generation.capacity_confirmation_arm import (
    validate_capacity_confirmation_arm_validation,
)
from cofitok.generation.capacity_confirmation_execution import (
    validate_capacity_confirmation_launch_receipt_contract,
)
from cofitok.generation.capacity_screen import ARM_NAMES, CAPACITY_NAMES, METHOD_NAMES
from cofitok.generation.capacity_screen_result import SCREEN_THRESHOLDS
from cofitok.generation.exposure_capacity_decision import THRESHOLDS as SUPPORT_THRESHOLDS


RESULT_SCHEMA = "cofitok_generation_capacity_confirmation_result_v1"
RESULT_ROLE = "source_bound_four_arm_capacity_confirmation_qualification"
RESULT_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": "derived_from_result",
    "large_capacity_readiness_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
CONFIRMATION_THRESHOLDS = {
    "max_absolute_fid": SUPPORT_THRESHOLDS["max_absolute_fid"],
    "min_precision": SUPPORT_THRESHOLDS["min_precision"],
    "min_recall": SUPPORT_THRESHOLDS["min_recall"],
    "min_top1": SUPPORT_THRESHOLDS["min_top1"],
    "min_top5": SUPPORT_THRESHOLDS["min_top5"],
    "min_predicted_class_fraction": SUPPORT_THRESHOLDS[
        "min_predicted_class_fraction"
    ],
    "min_normalized_predicted_entropy": SUPPORT_THRESHOLDS[
        "min_normalized_predicted_entropy"
    ],
    "max_fid_regression": SUPPORT_THRESHOLDS["max_fid_regression"],
    "max_precision_regression": SUPPORT_THRESHOLDS["max_precision_regression"],
    "max_recall_regression": SUPPORT_THRESHOLDS["max_recall_regression"],
    "max_top1_regression": SUPPORT_THRESHOLDS["max_top1_regression"],
    "max_top5_regression": SUPPORT_THRESHOLDS["max_top5_regression"],
    "max_capacity_precision_drop": SCREEN_THRESHOLDS["max_precision_drop"],
    "max_capacity_top1_drop": SCREEN_THRESHOLDS["max_top1_drop"],
    "max_capacity_top5_drop": SCREEN_THRESHOLDS["max_top5_drop"],
    "max_capacity_class_fraction_drop": SCREEN_THRESHOLDS[
        "max_predicted_class_fraction_drop"
    ],
    "max_capacity_entropy_drop": SCREEN_THRESHOLDS[
        "max_normalized_predicted_entropy_drop"
    ],
    "max_cofitok_relative_fid_interaction": SCREEN_THRESHOLDS[
        "max_cofitok_relative_fid_interaction"
    ],
    "max_cofitok_recall_interaction_regression": SCREEN_THRESHOLDS[
        "max_cofitok_recall_interaction_regression"
    ],
    "max_cofitok_top1_interaction_regression": SCREEN_THRESHOLDS[
        "max_cofitok_top1_interaction_regression"
    ],
    "max_endpoint_ratio": SCREEN_THRESHOLDS["max_endpoint_ratio"],
    "max_validation_ratio": SCREEN_THRESHOLDS["max_validation_ratio"],
    "max_reconstruction_ratio": SCREEN_THRESHOLDS["max_reconstruction_ratio"],
    "max_predicted_x0_high_frequency_ratio": SCREEN_THRESHOLDS[
        "max_predicted_x0_high_frequency_ratio"
    ],
    "max_tail_two_energy_ratio": SCREEN_THRESHOLDS["max_tail_two_energy_ratio"],
    "max_single_token_energy_ratio": SCREEN_THRESHOLDS[
        "max_single_token_energy_ratio"
    ],
    "min_coarse_token_energy_ratio": SCREEN_THRESHOLDS[
        "min_coarse_token_energy_ratio"
    ],
    "required_ordered_rank": SCREEN_THRESHOLDS["required_ordered_rank"],
    "min_order_count": SCREEN_THRESHOLDS["min_order_count"],
    "max_zero_token_abs": SCREEN_THRESHOLDS["max_zero_token_abs"],
    "min_shuffle_mismatch_ratio": SCREEN_THRESHOLDS[
        "min_shuffle_mismatch_ratio"
    ],
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
    return {key: row[key] for key in ("path", "bytes", "sha256")}


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


def _fraction(value: Any, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 1.0:
        raise ValueError(f"{name} is outside [0, 1]")
    return result


def _ratio(left: float, right: float, name: str) -> float:
    if right <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return left / right


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


def _summary(report: Mapping[str, Any], arm: str) -> dict[str, Any]:
    distribution = _object(report.get("distribution"), f"{arm} distribution")
    class_fidelity = _object(report.get("class_fidelity"), f"{arm} class fidelity")
    metrics = _object(distribution.get("metrics"), f"{arm} distribution metrics")
    class_metrics = _object(
        class_fidelity.get("metrics"), f"{arm} class fidelity metrics"
    )
    frozen = _object(report.get("frozen_training"), f"{arm} frozen training")
    screen_training = _object(frozen.get("screen_training"), f"{arm} screen training")
    checkpoint_eval = _object(
        _object(
            report.get("screen_checkpoint_evaluation"),
            f"{arm} screen checkpoint evaluation",
        ).get("summary"),
        f"{arm} checkpoint summary",
    )
    rollout = _object(
        _object(report.get("screen_rollout"), f"{arm} screen rollout").get(
            "summary"
        ),
        f"{arm} rollout summary",
    )
    sampling = _object(report.get("sampling"), f"{arm} sampling")
    return {
        "fid": _finite(metrics.get("fid"), f"{arm} FID", minimum=0.0),
        "precision": _fraction(metrics.get("precision"), f"{arm} precision"),
        "recall": _fraction(metrics.get("recall"), f"{arm} recall"),
        "top1": _fraction(class_metrics.get("top1_accuracy"), f"{arm} top1"),
        "top5": _fraction(class_metrics.get("top5_accuracy"), f"{arm} top5"),
        "predicted_class_fraction": _fraction(
            class_metrics.get("predicted_class_fraction"),
            f"{arm} predicted class fraction",
        ),
        "normalized_predicted_class_entropy": _fraction(
            class_metrics.get("normalized_predicted_class_entropy"),
            f"{arm} predicted class entropy",
        ),
        "validation_epsilon_mse": _finite(
            screen_training.get("validation_epsilon_mse"),
            f"{arm} validation epsilon MSE",
            minimum=0.0,
        ),
        "checkpoint": copy.deepcopy(checkpoint_eval),
        "rollout": copy.deepcopy(rollout),
        "sample_set_sha256": sampling.get("sample_set_sha256"),
        "checkpoint_sha256": sampling.get("checkpoint_sha256"),
        "real_set": copy.deepcopy(distribution.get("real_set")),
        "classifier": copy.deepcopy(class_fidelity.get("classifier")),
    }


def _capacity_pair_checks(
    summaries: Mapping[str, Mapping[str, Any]],
    capacity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cofitok = summaries[f"{capacity}_cofitok"]
    dense = summaries[f"{capacity}_dense_identity"]
    cofitok_checkpoint = _object(cofitok.get("checkpoint"), "CoFiTok checkpoint")
    dense_checkpoint = _object(dense.get("checkpoint"), "dense checkpoint")
    cofitok_rollout = _object(cofitok.get("rollout"), "CoFiTok rollout")
    dense_rollout = _object(dense.get("rollout"), "dense rollout")
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
        _finite(cofitok["validation_epsilon_mse"], "CoFiTok validation", minimum=0.0),
        _finite(dense["validation_epsilon_mse"], "dense validation", minimum=0.0),
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
        cofitok_rollout.get("predicted_x0_high_frequency_ratio"), "CoFiTok HF"
    )
    dense_hf = _object(
        dense_rollout.get("predicted_x0_high_frequency_ratio"), "dense HF"
    )
    if set(cofitok_hf) != set(dense_hf) or not cofitok_hf:
        raise ValueError(f"{capacity} rollout timestep set differs")
    hf_ratios = {
        timestep: _ratio(
            _finite(cofitok_hf[timestep], f"CoFiTok HF {timestep}", minimum=0.0),
            _finite(dense_hf[timestep], f"dense HF {timestep}", minimum=0.0),
            f"{capacity} HF {timestep}",
        )
        for timestep in sorted(cofitok_hf, key=int)
    }
    utilization = _object(cofitok_checkpoint.get("utilization"), "CoFiTok utilization")
    mechanism = {
        "ordered_rank": int(cofitok_checkpoint.get("ordered_rank_by_path_auc", -1)),
        "order_count": int(cofitok_checkpoint.get("order_count", -1)),
        "coarse_token_energy_ratio": _finite(
            utilization.get("coarse_token_energy_ratio"), "coarse energy", minimum=0.0
        ),
        "tail_two_energy_ratio": _finite(
            utilization.get("tail_two_energy_ratio"), "tail energy", minimum=0.0
        ),
        "max_single_token_energy_ratio": _finite(
            utilization.get("max_single_token_energy_ratio"),
            "single-token energy",
            minimum=0.0,
        ),
        "zero_token_max_abs": _finite(
            cofitok_checkpoint.get("zero_token_max_abs"), "zero token", minimum=0.0
        ),
        "shuffle_mismatch_ratio": _finite(
            cofitok_checkpoint.get("shuffled_to_ordered_endpoint_ratio"),
            "shuffle mismatch",
            minimum=0.0,
        ),
    }
    peak_hf = max(hf_ratios.values())
    checks = [
        _check(
            f"{capacity}_endpoint_non_regression",
            endpoint_ratio <= CONFIRMATION_THRESHOLDS["max_endpoint_ratio"],
            endpoint_ratio,
            "<=",
            CONFIRMATION_THRESHOLDS["max_endpoint_ratio"],
        ),
        _check(
            f"{capacity}_validation_non_regression",
            validation_ratio <= CONFIRMATION_THRESHOLDS["max_validation_ratio"],
            validation_ratio,
            "<=",
            CONFIRMATION_THRESHOLDS["max_validation_ratio"],
        ),
        _check(
            f"{capacity}_reconstruction_non_regression",
            reconstruction_ratio
            <= CONFIRMATION_THRESHOLDS["max_reconstruction_ratio"],
            reconstruction_ratio,
            "<=",
            CONFIRMATION_THRESHOLDS["max_reconstruction_ratio"],
        ),
        _check(
            f"{capacity}_predicted_x0_high_frequency",
            peak_hf
            <= CONFIRMATION_THRESHOLDS[
                "max_predicted_x0_high_frequency_ratio"
            ],
            peak_hf,
            "<=",
            CONFIRMATION_THRESHOLDS["max_predicted_x0_high_frequency_ratio"],
        ),
        _check(
            f"{capacity}_ordered_rank",
            mechanism["ordered_rank"]
            == CONFIRMATION_THRESHOLDS["required_ordered_rank"],
            mechanism["ordered_rank"],
            "==",
            CONFIRMATION_THRESHOLDS["required_ordered_rank"],
        ),
        _check(
            f"{capacity}_order_coverage",
            mechanism["order_count"] >= CONFIRMATION_THRESHOLDS["min_order_count"],
            mechanism["order_count"],
            ">=",
            CONFIRMATION_THRESHOLDS["min_order_count"],
        ),
        _check(
            f"{capacity}_coarse_token_utilization",
            mechanism["coarse_token_energy_ratio"]
            >= CONFIRMATION_THRESHOLDS["min_coarse_token_energy_ratio"],
            mechanism["coarse_token_energy_ratio"],
            ">=",
            CONFIRMATION_THRESHOLDS["min_coarse_token_energy_ratio"],
        ),
        _check(
            f"{capacity}_tail_two_energy",
            mechanism["tail_two_energy_ratio"]
            <= CONFIRMATION_THRESHOLDS["max_tail_two_energy_ratio"],
            mechanism["tail_two_energy_ratio"],
            "<=",
            CONFIRMATION_THRESHOLDS["max_tail_two_energy_ratio"],
        ),
        _check(
            f"{capacity}_single_token_energy",
            mechanism["max_single_token_energy_ratio"]
            <= CONFIRMATION_THRESHOLDS["max_single_token_energy_ratio"],
            mechanism["max_single_token_energy_ratio"],
            "<=",
            CONFIRMATION_THRESHOLDS["max_single_token_energy_ratio"],
        ),
        _check(
            f"{capacity}_zero_token",
            mechanism["zero_token_max_abs"]
            <= CONFIRMATION_THRESHOLDS["max_zero_token_abs"],
            mechanism["zero_token_max_abs"],
            "<=",
            CONFIRMATION_THRESHOLDS["max_zero_token_abs"],
        ),
        _check(
            f"{capacity}_shuffle_mismatch",
            mechanism["shuffle_mismatch_ratio"]
            >= CONFIRMATION_THRESHOLDS["min_shuffle_mismatch_ratio"],
            mechanism["shuffle_mismatch_ratio"],
            ">=",
            CONFIRMATION_THRESHOLDS["min_shuffle_mismatch_ratio"],
        ),
    ]
    return {
        "endpoint_ratio": endpoint_ratio,
        "validation_ratio": validation_ratio,
        "reconstruction_ratio": reconstruction_ratio,
        "predicted_x0_high_frequency_ratio": hf_ratios,
        "mechanism": mechanism,
    }, checks


def build_capacity_confirmation_result(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    arm_validations: Mapping[str, Mapping[str, Any]],
    arm_validation_identities: Mapping[str, Mapping[str, Any]],
    result_git: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_capacity_confirmation_preparation_contract(preparation)
    prep_id = _identity(preparation_identity, "capacity confirmation preparation")
    launch_id = _identity(
        launch_receipt_identity, "capacity confirmation launch receipt"
    )
    execution = _git(
        launch_receipt.get("execution_checkout"), "capacity confirmation execution"
    )
    launch = validate_capacity_confirmation_launch_receipt_contract(
        launch_receipt, expected_execution_checkout=execution
    )
    launch_sources = _object(
        launch.get("source_evidence"), "capacity confirmation launch sources"
    )
    if launch_sources.get("preparation") != prep_id:
        raise ValueError("capacity confirmation launch binds another preparation")
    if set(arm_validations) != set(ARM_NAMES) or set(
        arm_validation_identities
    ) != set(ARM_NAMES):
        raise ValueError("capacity confirmation result arm set differs")
    normalized_ids = {
        arm: _identity(arm_validation_identities[arm], f"{arm} confirmation arm")
        for arm in ARM_NAMES
    }
    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        validation = validate_capacity_confirmation_arm_validation(
            arm_validations[arm]
        )
        if validation.get("execution_git") != execution:
            raise ValueError(f"{arm} confirmation validation uses another checkout")
        sources = _object(validation.get("sources"), f"{arm} confirmation sources")
        if sources.get("launch_receipt") != launch_id:
            raise ValueError(f"{arm} confirmation uses another launch receipt")
        summaries[arm] = _summary(validation, arm)

    real_sets = {
        _object(summary.get("real_set"), f"{arm} real set").get("sha256")
        for arm, summary in summaries.items()
    }
    if len(real_sets) != 1 or None in real_sets:
        raise ValueError("capacity confirmation arms do not share one real set")
    sample_sets = {summary.get("sample_set_sha256") for summary in summaries.values()}
    if len(sample_sets) != len(ARM_NAMES) or None in sample_sets:
        raise ValueError("capacity confirmation sample sets are missing or duplicated")
    classifier_rows = [summaries[arm]["classifier"] for arm in ARM_NAMES]
    if any(row != classifier_rows[0] for row in classifier_rows[1:]):
        raise ValueError("capacity confirmation classifier identity differs")

    values = {
        arm: {
            key: summaries[arm][key]
            for key in (
                "fid",
                "precision",
                "recall",
                "top1",
                "top5",
                "predicted_class_fraction",
                "normalized_predicted_class_entropy",
            )
        }
        for arm in ARM_NAMES
    }
    checks: list[dict[str, Any]] = []
    for method in METHOD_NAMES:
        large = values[f"base256_{method}"]
        for metric, threshold_name in (
            ("fid", "max_absolute_fid"),
            ("precision", "min_precision"),
            ("recall", "min_recall"),
            ("top1", "min_top1"),
            ("top5", "min_top5"),
            ("predicted_class_fraction", "min_predicted_class_fraction"),
            (
                "normalized_predicted_class_entropy",
                "min_normalized_predicted_entropy",
            ),
        ):
            threshold = CONFIRMATION_THRESHOLDS[threshold_name]
            if metric == "fid":
                passed = large[metric] <= threshold
                comparison = "<="
            else:
                passed = large[metric] >= threshold
                comparison = ">="
            checks.append(
                _check(
                    f"base256_{method}_{metric}_absolute",
                    passed,
                    large[metric],
                    comparison,
                    threshold,
                )
            )

    base256_cofitok = values["base256_cofitok"]
    base256_dense = values["base256_dense_identity"]
    matched_checks = [
        (
            "base256_cofitok_fid_vs_dense",
            base256_cofitok["fid"]
            <= base256_dense["fid"]
            * (1.0 + CONFIRMATION_THRESHOLDS["max_fid_regression"]),
            base256_cofitok["fid"] / max(base256_dense["fid"], 1e-12) - 1.0,
            CONFIRMATION_THRESHOLDS["max_fid_regression"],
        ),
        (
            "base256_cofitok_precision_vs_dense",
            base256_dense["precision"] - base256_cofitok["precision"]
            <= CONFIRMATION_THRESHOLDS["max_precision_regression"],
            base256_dense["precision"] - base256_cofitok["precision"],
            CONFIRMATION_THRESHOLDS["max_precision_regression"],
        ),
        (
            "base256_cofitok_recall_vs_dense",
            base256_dense["recall"] - base256_cofitok["recall"]
            <= CONFIRMATION_THRESHOLDS["max_recall_regression"],
            base256_dense["recall"] - base256_cofitok["recall"],
            CONFIRMATION_THRESHOLDS["max_recall_regression"],
        ),
        (
            "base256_cofitok_top1_vs_dense",
            base256_dense["top1"] - base256_cofitok["top1"]
            <= CONFIRMATION_THRESHOLDS["max_top1_regression"],
            base256_dense["top1"] - base256_cofitok["top1"],
            CONFIRMATION_THRESHOLDS["max_top1_regression"],
        ),
        (
            "base256_cofitok_top5_vs_dense",
            base256_dense["top5"] - base256_cofitok["top5"]
            <= CONFIRMATION_THRESHOLDS["max_top5_regression"],
            base256_dense["top5"] - base256_cofitok["top5"],
            CONFIRMATION_THRESHOLDS["max_top5_regression"],
        ),
    ]
    for name, passed, observed, threshold in matched_checks:
        checks.append(_check(name, passed, observed, "<=", threshold))

    deltas: dict[str, dict[str, float]] = {}
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
                    row["precision"]
                    >= -CONFIRMATION_THRESHOLDS["max_capacity_precision_drop"],
                    row["precision"],
                    ">=",
                    -CONFIRMATION_THRESHOLDS["max_capacity_precision_drop"],
                ),
                _check(
                    f"{method}_top1_non_regression",
                    row["top1"]
                    >= -CONFIRMATION_THRESHOLDS["max_capacity_top1_drop"],
                    row["top1"],
                    ">=",
                    -CONFIRMATION_THRESHOLDS["max_capacity_top1_drop"],
                ),
                _check(
                    f"{method}_top5_non_regression",
                    row["top5"]
                    >= -CONFIRMATION_THRESHOLDS["max_capacity_top5_drop"],
                    row["top5"],
                    ">=",
                    -CONFIRMATION_THRESHOLDS["max_capacity_top5_drop"],
                ),
                _check(
                    f"{method}_predicted_class_fraction_non_regression",
                    row["predicted_class_fraction"]
                    >= -CONFIRMATION_THRESHOLDS[
                        "max_capacity_class_fraction_drop"
                    ],
                    row["predicted_class_fraction"],
                    ">=",
                    -CONFIRMATION_THRESHOLDS["max_capacity_class_fraction_drop"],
                ),
                _check(
                    f"{method}_class_entropy_non_regression",
                    row["normalized_predicted_class_entropy"]
                    >= -CONFIRMATION_THRESHOLDS["max_capacity_entropy_drop"],
                    row["normalized_predicted_class_entropy"],
                    ">=",
                    -CONFIRMATION_THRESHOLDS["max_capacity_entropy_drop"],
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
                <= CONFIRMATION_THRESHOLDS[
                    "max_cofitok_relative_fid_interaction"
                ],
                interaction["relative_fid_change"],
                "<=",
                CONFIRMATION_THRESHOLDS[
                    "max_cofitok_relative_fid_interaction"
                ],
            ),
            _check(
                "cofitok_recall_capacity_interaction",
                interaction["recall"]
                >= -CONFIRMATION_THRESHOLDS[
                    "max_cofitok_recall_interaction_regression"
                ],
                interaction["recall"],
                ">=",
                -CONFIRMATION_THRESHOLDS[
                    "max_cofitok_recall_interaction_regression"
                ],
            ),
            _check(
                "cofitok_top1_capacity_interaction",
                interaction["top1"]
                >= -CONFIRMATION_THRESHOLDS[
                    "max_cofitok_top1_interaction_regression"
                ],
                interaction["top1"],
                ">=",
                -CONFIRMATION_THRESHOLDS[
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
    names = [row["name"] for row in checks]
    if len(names) != len(set(names)):
        raise ValueError("capacity confirmation check names are not unique")
    failed = [row["name"] for row in checks if row["passed"] is not True]
    passed = not failed
    next_stage = {
        "route": "large_capacity_readiness" if passed else "hold",
        "reason": (
            "base256_cleared_absolute_support_class_and_mechanism_gates"
            if passed
            else "capacity_confirmation_failed_one_or_more_predeclared_checks"
        ),
        "support_collapse_resolved": passed,
        "large_capacity_readiness_preparation_allowed": passed,
        "large_capacity_readiness_launch_allowed": False,
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "required_next_evidence": (
            "A separate source-bound 250M full-training readiness and launch authorization."
            if passed
            else "A new matched intervention justified by the named failed checks."
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "confirmation_pass" if passed else "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "result_git": _git(result_git, "capacity confirmation result Git"),
        "source_evidence": {
            "preparation": prep_id,
            "launch_receipt": launch_id,
            "arm_validations": normalized_ids,
        },
        "validated_preparation": {
            "scientific_status": prepared["scientific_status"],
            "evaluation_contract": copy.deepcopy(prepared["evaluation_contract"]),
        },
        "arm_summaries": summaries,
        "within_method_capacity_deltas": deltas,
        "difference_in_differences": interaction,
        "pair_stability": pair_stability,
        "thresholds": copy.deepcopy(CONFIRMATION_THRESHOLDS),
        "checks": checks,
        "failed_checks": failed,
        "next_stage": next_stage,
        "claim_policy": {
            "confirmation_is_large_capacity_qualification_only": True,
            "support_collapse_resolved": passed,
            "formal_generation_claim_allowed": False,
            "generation_advantage_proven": False,
            "full_300k_requires_separate_readiness_and_authorization": True,
        },
        "authorization_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }


def validate_capacity_confirmation_result_contract(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "capacity confirmation result")
    checks = row.get("checks")
    failed = row.get("failed_checks")
    next_stage = _object(row.get("next_stage"), "capacity confirmation next stage")
    if not isinstance(checks, list) or not checks or not isinstance(failed, list):
        raise ValueError("capacity confirmation checks are incomplete")
    computed_failed: list[str] = []
    names: set[str] = set()
    for value in checks:
        check = _object(value, "capacity confirmation check")
        name = check.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("capacity confirmation check names differ")
        if check.get("passed") not in {True, False}:
            raise ValueError(f"capacity confirmation check is not boolean: {name}")
        names.add(name)
        if check["passed"] is False:
            computed_failed.append(name)
    passed = not computed_failed
    if (
        row.get("schema_version") != RESULT_SCHEMA
        or row.get("role") != RESULT_ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status")
        != ("confirmation_pass" if passed else "hold")
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("thresholds") != CONFIRMATION_THRESHOLDS
        or row.get("authorization_boundary") != RESULT_BOUNDARY
        or failed != computed_failed
        or next_stage.get("support_collapse_resolved") is not passed
        or next_stage.get("large_capacity_readiness_preparation_allowed")
        is not passed
        or next_stage.get("large_capacity_readiness_launch_allowed") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("full_training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
        or set(_object(row.get("arm_summaries"), "capacity confirmation arms"))
        != set(ARM_NAMES)
    ):
        raise ValueError("capacity confirmation result contract differs")
    _git(row.get("result_git"), "capacity confirmation result Git")
    sources = _object(row.get("source_evidence"), "capacity confirmation sources")
    _identity(sources.get("preparation"), "capacity confirmation preparation")
    _identity(sources.get("launch_receipt"), "capacity confirmation launch")
    arms = _object(sources.get("arm_validations"), "capacity confirmation arm sources")
    if set(arms) != set(ARM_NAMES):
        raise ValueError("capacity confirmation result arm source set differs")
    for arm in ARM_NAMES:
        _identity(arms[arm], f"{arm} confirmation validation")
    return copy.deepcopy(row)


__all__ = [
    "CONFIRMATION_THRESHOLDS",
    "RESULT_BOUNDARY",
    "RESULT_ROLE",
    "RESULT_SCHEMA",
    "build_capacity_confirmation_result",
    "validate_capacity_confirmation_result_contract",
]
