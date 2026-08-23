from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any


POST_RECONCILIATION_DECISION_SCHEMA_VERSION = 1
POST_RECONCILIATION_DECISION_ROLE = (
    "generation_100k_post_reconciliation_experiment_decision"
)
AUTHORITATIVE_DECISION_ROLE = (
    "stability_quality_bridge_followup_experiment_decision"
)
RECONCILIATION_ROLE = "generation_100k_cross_protocol_reconciliation"
QUALITY_RESULT_ROLE = "stability_full_data_quality_bridge_result"
TRAINING_EXPOSURE_ROLE = "generation_training_exposure_audit"

TRAINING_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TRAINING_BRANCH = "scale/generation-stability-quality-bridge-100k"
TRAINING_TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"

EXPECTED_FAILED_CHECKS = [
    "cofitok_absolute_fid",
    "cofitok_recall_floor",
    "class_fidelity",
]
MATCHED_QUALITY_CHECKS = {
    "matched_fid_tolerance",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
}
MECHANISM_CHECKS = {
    "matched_endpoint_tolerance",
    "ordered_prefix_rank",
    "coarse_token_utilization",
    "restricted_synthesis_zero_token",
    "shuffle_mismatch",
}
EXPECTED_CHECKS = {
    "cofitok_absolute_fid",
    "matched_fid_tolerance",
    "cofitok_precision_floor",
    "cofitok_recall_floor",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
    "matched_endpoint_tolerance",
    "ordered_prefix_rank",
    "coarse_token_utilization",
    "restricted_synthesis_zero_token",
    "shuffle_mismatch",
    "class_fidelity",
}

AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "training_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "legacy_factorization_route_allowed": False,
    "legacy_conditioning_route_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "new_source_bound_execution_gate_required": True,
}


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _hex(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


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


def _same_content(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if {
        "bytes": left.get("bytes"),
        "sha256": left.get("sha256"),
    } != {
        "bytes": right.get("bytes"),
        "sha256": right.get("sha256"),
    }:
        raise ValueError(f"{label} content identity differs")


def _clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    tree = value.get("tree")
    branch = value.get("branch")
    path = value.get("path")
    if (
        not _hex(revision, length=40)
        or (tree is not None and not _hex(tree, length=40))
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
        or (path is not None and (not isinstance(path, str) or not path))
    ):
        raise ValueError(f"{label} Git identity is not exact and clean")
    result = {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }
    if tree is not None:
        result["tree"] = tree
    if path is not None:
        result["path"] = path
    return result


def _require_false_boundary(
    value: Mapping[str, Any],
    *,
    keys: set[str],
    label: str,
) -> None:
    if any(value.get(key) is not False for key in keys):
        raise ValueError(f"{label} weakens the non-authorizing boundary")


def _check_index(screen: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = screen.get("checks")
    if not isinstance(rows, list):
        raise TypeError("quality screen checks are missing")
    indexed: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("name"), str)
            or row.get("passed") not in {True, False}
        ):
            raise ValueError("quality screen check is malformed")
        name = str(row["name"])
        if name in indexed:
            raise ValueError("quality screen checks are duplicated")
        indexed[name] = copy.deepcopy(dict(row))
        if row["passed"] is False:
            failed.append(name)
    if set(indexed) != EXPECTED_CHECKS:
        raise ValueError("quality screen check set differs")
    if screen.get("failed_checks") != failed:
        raise ValueError("quality screen failed-check summary differs")
    return indexed


def _validate_authoritative_decision(
    decision: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_identity = _identity(identity, label="authoritative decision")
    route = decision.get("recommended_next_stage")
    boundary = decision.get("authorization_boundary")
    if (
        int(decision.get("schema_version", -1)) != 2
        or decision.get("status") != "completed"
        or decision.get("role") != AUTHORITATIVE_DECISION_ROLE
        or not isinstance(route, Mapping)
        or route.get("id") != "reconcile_100k_cross_protocol_evidence"
        or route.get("category") != "evidence_conflict"
        or route.get("execution_ready") is not False
        or route.get("gpu_execution_allowed") is not False
        or not isinstance(boundary, Mapping)
    ):
        raise ValueError("authoritative pre-reconciliation decision differs")
    _require_false_boundary(
        boundary,
        keys={
            "recommended_stage_execution_allowed",
            "quality_bridge_execution_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "report_is_promotion_gate",
            "release_authorization_allowed",
        },
        label="authoritative decision",
    )
    return {
        "identity": source_identity,
        "superseded_route": copy.deepcopy(dict(route)),
    }


def _validate_reconciliation(
    report: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    quality_identity: Mapping[str, Any],
    exposure_identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_identity = _identity(identity, label="cross-protocol reconciliation")
    boundary = report.get("claim_boundary")
    comparison = report.get("comparison")
    sources = report.get("source_evidence")
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != RECONCILIATION_ROLE
        or report.get("status") != "completed"
        or report.get("operational_status") != "pass"
        or report.get("terminal_status") != "hold"
        or not isinstance(boundary, Mapping)
        or not isinstance(comparison, Mapping)
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("cross-protocol reconciliation contract differs")
    _require_false_boundary(
        boundary,
        keys={
            "generation_advantage_proven",
            "quality_or_generation_advantage_claim_allowed",
            "training_launch_allowed",
            "sampling_launch_allowed",
            "gpu_execution_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        },
        label="cross-protocol reconciliation",
    )
    for name, expected in (
        ("decision", decision_identity),
        ("quality_result", quality_identity),
        ("terminal_training_exposure", exposure_identity),
    ):
        claimed = sources.get(name)
        if not isinstance(claimed, Mapping):
            raise ValueError(f"reconciliation {name} binding is missing")
        _same_content(claimed, expected, label=f"reconciliation {name}")

    rows = comparison.get("protocol_rows")
    if not isinstance(rows, Mapping) or set(rows) != {
        "ddim50_2048",
        "ddim100_2048",
        "ddim100_10000",
    }:
        raise ValueError("reconciliation protocol rows differ")
    winners = {
        name: row.get("lower_fid_method")
        for name, row in rows.items()
        if isinstance(row, Mapping)
    }
    if winners != {
        "ddim50_2048": "dense_identity",
        "ddim100_2048": "cofitok",
        "ddim100_10000": "cofitok",
    }:
        raise ValueError("reconciled matched ranking differs")
    if (
        comparison.get("ranking_reversal_explanation")
        != "sampler_step_effect_dominates_observed_ranking_reversal"
        or comparison.get("sampler_step_changes_matched_ranking") is not True
        or comparison.get("sample_count_changes_matched_ranking") is not False
        or comparison.get("terminal_protocol_controls_quality_status") is not True
        or comparison.get("terminal_quality_status") != "hold"
        or comparison.get("generation_advantage_proven") is not False
    ):
        raise ValueError("cross-protocol reconciliation conclusion differs")
    return {
        "identity": source_identity,
        "status": "resolved",
        "ranking_reversal_explanation": comparison[
            "ranking_reversal_explanation"
        ],
        "protocol_rows": copy.deepcopy(dict(rows)),
        "per_method_effects": copy.deepcopy(
            dict(comparison.get("per_method_effects", {}))
        ),
        "sampler_step_changes_matched_ranking": True,
        "sample_count_changes_matched_ranking": False,
    }


def _validate_quality_result(
    result: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_identity = _identity(identity, label="quality bridge result")
    screen = result.get("quality_screen")
    terminal = result.get("terminal")
    boundary = result.get("authorization_boundary")
    git = result.get("git")
    source_reports = result.get("source_reports")
    if (
        int(result.get("schema_version", -1)) != 1
        or result.get("status") != "completed"
        or result.get("role") != QUALITY_RESULT_ROLE
        or result.get("stage") != "stability_quality_bridge"
        or not isinstance(screen, Mapping)
        or not isinstance(terminal, Mapping)
        or not isinstance(boundary, Mapping)
        or not isinstance(git, Mapping)
        or not isinstance(source_reports, Mapping)
    ):
        raise ValueError("quality bridge result contract differs")
    normalized_git = _clean_git(git, label="quality bridge execution")
    if normalized_git != {
        "revision": TRAINING_REVISION,
        "branch": TRAINING_BRANCH,
        "tracked_dirty": False,
    }:
        raise ValueError("quality bridge execution Git identity differs")
    _require_false_boundary(
        boundary,
        keys={
            "quality_bridge_execution_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "report_is_promotion_gate",
            "release_authorization_allowed",
        },
        label="quality bridge result",
    )
    checks = _check_index(screen)
    if (
        screen.get("status") != "hold"
        or screen.get("non_authorizing") is not True
        or screen.get("failed_checks") != EXPECTED_FAILED_CHECKS
        or any(checks[name]["passed"] is not True for name in MATCHED_QUALITY_CHECKS)
        or any(checks[name]["passed"] is not True for name in MECHANISM_CHECKS)
        or checks["cofitok_absolute_fid"]["passed"] is not False
        or checks["cofitok_recall_floor"]["passed"] is not False
        or checks["class_fidelity"]["passed"] is not False
    ):
        raise ValueError("quality bridge failure classification differs")
    methods = terminal.get("methods")
    class_fidelity = terminal.get("class_fidelity")
    if (
        not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
        or not isinstance(class_fidelity, Mapping)
        or class_fidelity.get("status") != "hold"
        or class_fidelity.get("valid") is not False
    ):
        raise ValueError("terminal quality evidence differs")
    thresholds = screen.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise TypeError("quality thresholds are missing")
    return {
        "identity": source_identity,
        "git": normalized_git,
        "failed_checks": list(EXPECTED_FAILED_CHECKS),
        "checks": copy.deepcopy(list(screen["checks"])),
        "methods": copy.deepcopy(dict(methods)),
        "class_fidelity": copy.deepcopy(dict(class_fidelity)),
        "thresholds": copy.deepcopy(dict(thresholds)),
        "source_reports": copy.deepcopy(dict(source_reports)),
    }


def _validate_training_exposure(
    report: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    quality_identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_identity = _identity(identity, label="terminal training exposure")
    rows = report.get("rows")
    comparison = report.get("comparison")
    terminal = report.get("terminal_binding")
    if (
        int(report.get("schema_version", -1)) != 1
        or int(report.get("exposure_schema_version", -1)) != 1
        or report.get("role") != TRAINING_EXPOSURE_ROLE
        or report.get("status") != "pass"
        or not isinstance(rows, Mapping)
        or set(rows) != {"cofitok", "dense_identity"}
        or not isinstance(comparison, Mapping)
        or not isinstance(terminal, Mapping)
    ):
        raise ValueError("terminal training exposure contract differs")
    normalized: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        row = rows[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"training exposure {method} row is malformed")
        git = row.get("git")
        if not isinstance(git, Mapping):
            raise ValueError(f"training exposure {method} Git identity is missing")
        row_git = {
            "revision": git.get("revision"),
            "branch": git.get("branch"),
            "tracked_dirty": git.get("dirty"),
        }
        if _clean_git(row_git, label=f"training exposure {method}") != {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        }:
            raise ValueError(f"training exposure {method} Git identity differs")
        epochs = _finite(
            row.get("completed_equivalent_epochs"),
            label=f"training exposure {method} epochs",
        )
        if (
            row.get("status") != "complete"
            or row.get("training_complete") is not True
            or row.get("dataset") != "imagenet_256"
            or int(row.get("target_steps", -1)) != 100_000
            or int(row.get("completed_steps", -1)) != 100_000
            or int(row.get("effective_batch_size", -1)) != 64
            or int(row.get("samples_seen", -1)) != 6_400_000
            or int(row.get("train_image_count", -1)) != 1_281_167
            or not math.isclose(
                epochs,
                6_400_000 / 1_281_167,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"training exposure {method} row differs")
        normalized[method] = {
            "equivalent_epochs": epochs,
            "samples_seen": int(row["samples_seen"]),
            "train_image_count": int(row["train_image_count"]),
            "dataset_identity_sha256": row.get("dataset_identity_sha256"),
        }
    if normalized["cofitok"] != normalized["dense_identity"]:
        raise ValueError("terminal training exposure is not exactly matched")
    if any(
        comparison.get(key) is not True
        for key in (
            "same_dataset",
            "same_dataset_identity",
            "same_effective_batch_size",
            "same_completed_steps",
            "same_images_seen",
            "same_equivalent_epochs",
            "same_dataset_normalized_exposure",
        )
    ):
        raise ValueError("terminal training exposure comparison differs")
    bound_result = terminal.get("terminal_result")
    if (
        terminal.get("terminal_result_binding_verified") is not True
        or terminal.get("active_runbook_verification_completed") is not True
        or not isinstance(bound_result, Mapping)
    ):
        raise ValueError("terminal training exposure result binding differs")
    _same_content(
        bound_result,
        quality_identity,
        label="training exposure terminal quality result",
    )
    epochs = normalized["cofitok"]["equivalent_epochs"]
    reference = 24.96859419012024
    return {
        "identity": source_identity,
        "dataset": "imagenet_256",
        "dataset_identity_sha256": normalized["cofitok"][
            "dataset_identity_sha256"
        ],
        "matched": True,
        "steps_per_method": 100_000,
        "images_seen_per_method": 6_400_000,
        "train_image_count": 1_281_167,
        "full_data_equivalent_epochs": epochs,
        "historical_10pct_reference_equivalent_epochs": reference,
        "full_to_reference_equivalent_epoch_ratio": epochs / reference,
        "below_historical_reference_exposure": epochs < reference,
        "insufficient_exposure_is_live_hypothesis": epochs < reference,
        "causal_status": "not_identified_by_exposure_alone",
    }


def _validate_terminal_evidence(
    evidence: Mapping[str, Any],
    *,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    if set(evidence) != {"cofitok", "dense_identity"}:
        raise ValueError("terminal evidence method set differs")
    terminal_methods = quality["methods"]
    class_methods = quality["class_fidelity"].get("metrics")
    class_sources = quality["class_fidelity"].get("sources")
    result_sources = quality["source_reports"]
    if not isinstance(class_methods, Mapping) or not isinstance(
        class_sources, Mapping
    ):
        raise ValueError("terminal class-fidelity evidence is incomplete")
    normalized: dict[str, Any] = {}
    protocol_without_prefix: dict[str, Any] | None = None
    for method in ("cofitok", "dense_identity"):
        row = evidence[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"terminal {method} evidence is malformed")
        expected_prefix = 8 if method == "cofitok" else 1
        for name in (
            "generation_report",
            "class_fidelity_report",
            "checkpoint_payload",
            "checkpoint_sidecar",
            "latest",
        ):
            if not isinstance(row.get(name), Mapping):
                raise ValueError(f"terminal {method} {name} identity is missing")
            _identity(row[name], label=f"terminal {method} {name}")
        terminal = terminal_methods[method]
        if not isinstance(terminal, Mapping):
            raise ValueError(f"terminal {method} quality row is missing")
        generation_metrics = row.get("generation_metrics")
        class_metrics = row.get("class_fidelity_metrics")
        sampling = row.get("sampling")
        if (
            row.get("physical_sha256_verified") is not True
            or row.get("sidecar_latest_reconciled") is not True
            or int(row.get("checkpoint_step", -1)) != 100_000
            or row.get("checkpoint_sha256") != terminal.get("checkpoint_sha256")
            or row.get("sample_set_sha256") != terminal.get("sample_set_sha256")
            or int(row.get("prefix_budget", -1)) != expected_prefix
            or not isinstance(generation_metrics, Mapping)
            or not isinstance(class_metrics, Mapping)
            or not isinstance(sampling, Mapping)
        ):
            raise ValueError(f"terminal {method} evidence contract differs")
        if (
            _finite(generation_metrics.get("fid"), label=f"{method} FID")
            != _finite(terminal.get("fid"), label=f"terminal {method} FID")
            or _finite(
                generation_metrics.get("precision"),
                label=f"{method} precision",
            )
            != _finite(terminal.get("precision"), label=f"terminal {method} precision")
            or _finite(generation_metrics.get("recall"), label=f"{method} recall")
            != _finite(terminal.get("recall"), label=f"terminal {method} recall")
            or dict(class_metrics) != dict(class_methods[method])
        ):
            raise ValueError(f"terminal {method} metrics differ")
        claimed_class_source = class_sources.get(method)
        claimed_generation_source = result_sources.get(
            "cofitok_generation" if method == "cofitok" else "dense_generation"
        )
        claimed_result_class_source = result_sources.get(
            "cofitok_class_fidelity"
            if method == "cofitok"
            else "dense_class_fidelity"
        )
        if not isinstance(claimed_class_source, Mapping):
            raise ValueError(f"terminal {method} class source is missing")
        if not isinstance(claimed_generation_source, Mapping) or not isinstance(
            claimed_result_class_source, Mapping
        ):
            raise ValueError(f"terminal {method} result source is missing")
        _same_content(
            row["generation_report"],
            claimed_generation_source,
            label=f"terminal {method} generation source",
        )
        _same_content(
            row["class_fidelity_report"],
            claimed_class_source,
            label=f"terminal {method} class-fidelity source",
        )
        _same_content(
            row["class_fidelity_report"],
            claimed_result_class_source,
            label=f"terminal {method} result class-fidelity source",
        )
        if (
            sampling.get("sampler") != "ddim"
            or int(sampling.get("sample_steps", -1)) != 100
            or int(sampling.get("num_samples", -1)) != 10_000
            or int(sampling.get("start_index", -1)) != 0
            or int(sampling.get("seed", -1)) != 0
            or sampling.get("class_schedule") != "balanced_modulo"
            or sampling.get("guidance_scale") != 1.5
            or sampling.get("guidance_rescale") != 0.0
            or sampling.get("clip_x0") is not True
            or sampling.get("eta") != 0.0
            or sampling.get("precision") != "bf16"
            or sampling.get("prefix_budgets") != [expected_prefix]
        ):
            raise ValueError(f"terminal {method} sampling protocol differs")
        shared = {
            key: copy.deepcopy(value)
            for key, value in sampling.items()
            if key != "prefix_budgets"
        }
        if protocol_without_prefix is None:
            protocol_without_prefix = shared
        elif shared != protocol_without_prefix:
            raise ValueError("terminal matched sampling protocol differs")
        normalized[method] = copy.deepcopy(dict(row))
    return normalized


def _asset_audit(exposure: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "matched_100k_epsilon_stability_sampling": {
            "selected": True,
            "status": "design_family_available_but_terminal_sources_unbound",
            "incremental_information": (
                "Separates shared DDIM epsilon-to-x0 instability from checkpoint "
                "capacity or exposure by changing frozen-checkpoint sampling controls only."
            ),
            "cost_class": "bounded_inference_only",
            "new_source_binding_required": True,
        },
        "matched_capacity_qualification": {
            "selected": False,
            "status": "deferred",
            "reason": (
                "Higher cost and capacity remains confounded with the live low-exposure "
                "hypothesis."
            ),
            "training_launch_allowed": False,
        },
        "matched_exposure_qualification": {
            "selected": False,
            "status": "scientifically_live_but_deferred",
            "full_data_equivalent_epochs": exposure[
                "full_data_equivalent_epochs"
            ],
            "historical_10pct_reference_equivalent_epochs": exposure[
                "historical_10pct_reference_equivalent_epochs"
            ],
            "reason": (
                "Exposure is a live alternative, but a frozen-checkpoint sampler "
                "discriminator has much lower cost and no training confound."
            ),
            "training_launch_allowed": False,
        },
        "matched_training_recipe_intervention": {
            "selected": False,
            "status": "deferred",
            "reason": (
                "A fresh recipe changes the optimization trajectory and is not the "
                "minimum discriminator for the shared terminal artifact pattern."
            ),
            "training_launch_allowed": False,
        },
    }


def _recommended_stage() -> dict[str, Any]:
    return {
        "id": "prepare_matched_100k_epsilon_stability_sampling_diagnostic",
        "category": "shared_sampling_system_discriminator",
        "execution_ready": False,
        "gpu_execution_allowed": False,
        "sampling_launch_allowed": False,
        "training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "objective": (
            "Use the immutable matched 100K EMA checkpoints to test whether shared "
            "epsilon-to-x0 sampling instability explains the held absolute quality, "
            "support, and conditioning evidence before spending a training budget."
        ),
        "screening_design": {
            "sample_count_per_method_case": 1_000,
            "class_count": 1_000,
            "one_sample_per_class_per_case": True,
            "checkpoint_step": 100_000,
            "weights": "ema",
            "sampler": "ddim",
            "sample_steps": 100,
            "matched_methods": ["cofitok", "dense_identity"],
            "shared_random_stream_per_case": True,
            "shared_case_selection_required": True,
            "per_method_protocol_selection_allowed": False,
            "candidate_control_family": [
                "legacy_terminal_start_and_hard_x0_clip",
                "nonterminal_start",
                "schedule_sigma_initialization",
                "dynamic_x0_threshold",
                "recompute_epsilon_after_x0_constraint",
            ],
            "small_sample_fid_is_formal": False,
            "precision_recall_is_formal_at_1000": False,
        },
        "required_next_evidence": (
            "A new versioned source-compatible preparation that binds this decision, "
            "the reconciliation, both 100K checkpoint payloads/sidecars/latest files, "
            "the real set, evaluator, classifier, exact repair revision/tree, fresh "
            "random stream, output root, matched case policy, and a separately scoped "
            "execution gate. No GPU work is permitted before that gate passes."
        ),
        "confirmation_policy": {
            "matched_10000_confirmation_required_for_any_candidate": True,
            "confirmation_must_use_independent_random_stream": True,
            "terminal_result_replacement_allowed": False,
            "new_formal_gate_required": True,
        },
    }


def build_post_reconciliation_decision(
    *,
    authoritative_decision: Mapping[str, Any],
    authoritative_decision_identity: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    reconciliation_identity: Mapping[str, Any],
    quality_result: Mapping[str, Any],
    quality_result_identity: Mapping[str, Any],
    training_exposure: Mapping[str, Any],
    training_exposure_identity: Mapping[str, Any],
    terminal_evidence: Mapping[str, Any],
    builder_git: Mapping[str, Any],
) -> dict[str, Any]:
    decision_identity = _identity(
        authoritative_decision_identity,
        label="authoritative decision",
    )
    quality_identity = _identity(
        quality_result_identity,
        label="quality bridge result",
    )
    exposure_identity = _identity(
        training_exposure_identity,
        label="terminal training exposure",
    )
    builder = _clean_git(builder_git, label="post-reconciliation builder")
    old_decision = _validate_authoritative_decision(
        authoritative_decision,
        identity=decision_identity,
    )
    quality = _validate_quality_result(
        quality_result,
        identity=quality_identity,
    )
    exposure = _validate_training_exposure(
        training_exposure,
        identity=exposure_identity,
        quality_identity=quality_identity,
    )
    reconciled = _validate_reconciliation(
        reconciliation,
        identity=reconciliation_identity,
        decision_identity=decision_identity,
        quality_identity=quality_identity,
        exposure_identity=exposure_identity,
    )
    terminal = _validate_terminal_evidence(
        terminal_evidence,
        quality=quality,
    )

    thresholds = quality["thresholds"]
    methods = quality["methods"]
    max_fid = _finite(thresholds.get("max_absolute_fid"), label="absolute FID")
    min_recall = _finite(thresholds.get("min_recall"), label="recall floor")
    both_fid_high = all(
        _finite(methods[method].get("fid"), label=f"{method} FID") > max_fid
        for method in ("cofitok", "dense_identity")
    )
    both_recall_low = all(
        _finite(methods[method].get("recall"), label=f"{method} recall")
        < min_recall
        for method in ("cofitok", "dense_identity")
    )
    if not both_fid_high or not both_recall_low:
        raise ValueError("shared absolute quality/support classification differs")

    return {
        "schema_version": POST_RECONCILIATION_DECISION_SCHEMA_VERSION,
        "role": POST_RECONCILIATION_DECISION_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "builder_git": builder,
        "training_git": {
            "revision": TRAINING_REVISION,
            "tree": TRAINING_TREE,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "source_evidence": {
            "authoritative_pre_reconciliation_decision": old_decision[
                "identity"
            ],
            "cross_protocol_reconciliation": reconciled["identity"],
            "quality_bridge_result": quality["identity"],
            "terminal_training_exposure": exposure["identity"],
            "terminal_methods": {
                method: {
                    key: copy.deepcopy(terminal[method][key])
                    for key in (
                        "generation_report",
                        "class_fidelity_report",
                        "checkpoint_payload",
                        "checkpoint_sidecar",
                        "latest",
                    )
                }
                for method in ("cofitok", "dense_identity")
            },
        },
        "source_replay": {
            "authoritative_decision_route_replayed": True,
            "cross_protocol_reconciliation_replayed": True,
            "terminal_metrics_and_class_fidelity_replayed": True,
            "checkpoint_payload_sha256_sidecar_latest_reconciled": True,
            "terminal_training_exposure_replayed": True,
        },
        "scientific_resolution": {
            "cross_protocol_conflict": {
                "status": reconciled["status"],
                "explanation": reconciled[
                    "ranking_reversal_explanation"
                ],
                "sampler_step_changes_matched_ranking": True,
                "sample_count_changes_matched_ranking": False,
                "ddim50_2048_winner": "dense_identity",
                "ddim100_2048_winner": "cofitok",
                "ddim100_10000_winner": "cofitok",
            },
            "terminal_quality": {
                "status": "hold",
                "failed_checks": list(EXPECTED_FAILED_CHECKS),
                "matched_quality_checks_pass": True,
                "factorization_mechanism_checks_pass": True,
                "class_fidelity_pass": False,
                "both_methods_absolute_fid_above_threshold": both_fid_high,
                "both_methods_recall_below_floor": both_recall_low,
                "methods": copy.deepcopy(methods),
            },
            "failure_classification": {
                "matched_quality_only_failure": False,
                "factorization_mechanism_failure": False,
                "class_only_failure": False,
                "mixed_shared_absolute_quality_support_and_class_failure": True,
                "insufficient_exposure_is_live_hypothesis": exposure[
                    "insufficient_exposure_is_live_hypothesis"
                ],
                "exposure_causal_status": exposure["causal_status"],
            },
            "training_exposure": exposure,
        },
        "legacy_route_disposition": {
            "factorization_quality_regression_supervisor": {
                "eligible": False,
                "reason": (
                    "All matched-quality and mechanism checks pass; the observed hold "
                    "is not a matched-quality-only regression."
                ),
            },
            "conditioning_only_supervisor": {
                "eligible": False,
                "reason": (
                    "Class fidelity is one of three failed checks, not the sole failed check."
                ),
            },
            "legacy_canonical_decision_paths_must_not_be_modified": True,
            "legacy_supervisor_execution_authorizations_must_not_be_created": True,
        },
        "candidate_asset_audit": _asset_audit(exposure),
        "recommended_next_stage": _recommended_stage(),
        "claim_policy": {
            "experiment_selection_only": True,
            "cross_tier_numeric_ranking_allowed": False,
            "terminal_result_remains_authoritative": True,
            "small_sample_screen_can_prove_generation_advantage": False,
            "interpretation": (
                "The reconciliation removes the DDIM-50 ranking conflict but does not "
                "remove the absolute-quality, recall, or class-fidelity hold. The next "
                "stage is a bounded matched sampler discriminator, not a claim, route "
                "upgrade, or training authorization."
            ),
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
