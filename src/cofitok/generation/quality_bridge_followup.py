from __future__ import annotations

import copy
import math
from typing import Any, Mapping


FOLLOWUP_DECISION_SCHEMA_VERSION = 1
FOLLOWUP_DECISION_ROLE = "stability_quality_bridge_followup_experiment_decision"
QUALITY_BRIDGE_RESULT_ROLE = "stability_full_data_quality_bridge_result"
QUALITY_BRIDGE_RESULT_STAGE = "stability_quality_bridge"
QUALITY_BRIDGE_EXECUTION_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
QUALITY_BRIDGE_EXECUTION_BRANCH = (
    "scale/generation-stability-quality-bridge-100k"
)

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

MECHANISM_CHECKS = {
    "matched_endpoint_tolerance",
    "ordered_prefix_rank",
    "coarse_token_utilization",
    "restricted_synthesis_zero_token",
    "shuffle_mismatch",
}
MATCHED_QUALITY_CHECKS = {
    "matched_fid_tolerance",
    "matched_precision_tolerance",
    "matched_recall_tolerance",
}
ABSOLUTE_QUALITY_CHECKS = {
    "cofitok_absolute_fid",
    "cofitok_precision_floor",
    "cofitok_recall_floor",
}

AUTHORIZATION_BOUNDARY = {
    "decision_evidence_complete": True,
    "recommended_stage_execution_allowed": False,
    "quality_bridge_execution_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "report_is_promotion_gate": False,
    "release_authorization_allowed": False,
    "new_source_compatible_gate_required": True,
}

_ALERT_TO_TERMINAL_CHECK = {
    "cofitok_fid_more_than_25pct_above_dense": "matched_fid_tolerance",
    "cofitok_ordered_prefix_not_rank1": "ordered_prefix_rank",
    "cofitok_zero_token_contract_failed": "restricted_synthesis_zero_token",
    "cofitok_shuffle_mismatch_not_detected": "shuffle_mismatch",
}


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _hex_digest(value: Any, *, length: int) -> bool:
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
        or not _hex_digest(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not _hex_digest(revision, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def _quality_checks(screen: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = screen.get("checks")
    if not isinstance(rows, list):
        raise ValueError("quality bridge result has no quality checks")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("quality bridge quality check is malformed")
        name = str(row["name"])
        if name in indexed or row.get("passed") not in {True, False}:
            raise ValueError("quality bridge quality checks are duplicated or invalid")
        indexed[name] = copy.deepcopy(row)
    if set(indexed) != EXPECTED_CHECKS:
        raise ValueError("quality bridge quality-check set differs")
    failed = [row["name"] for row in rows if row["passed"] is False]
    if screen.get("failed_checks") != failed:
        raise ValueError("quality bridge failed-check summary differs")
    expected_status = "pass" if not failed else "hold"
    if screen.get("status") != expected_status or screen.get("non_authorizing") is not True:
        raise ValueError("quality bridge quality-screen state differs")
    return indexed


def _trend_sampling_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(value))
    protocol.pop("sample_set_digest", None)
    return protocol


def _milestone_row(
    report: Mapping[str, Any],
    *,
    step: int,
    method: str,
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", -1)) != 2
        or report.get("status") != "completed"
        or report.get("role") != "training_quality_trend_only"
        or report.get("source_profile") != "quality_bridge"
        or int(report.get("milestone_step", -1)) != step
        or int(report.get("expected_samples", -1)) != 2_048
        or report.get("claim_policy", {}).get("formal_generation_claim_allowed")
        is not False
    ):
        raise ValueError(f"quality bridge milestone {step} contract differs")
    methods = report.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {
        "cofitok",
        "dense_identity",
    }:
        raise ValueError(f"quality bridge milestone {step} method set differs")
    raw = methods.get(method)
    if not isinstance(raw, Mapping):
        raise ValueError(f"quality bridge milestone {step} {method} row is missing")
    expected_budget = 8 if method == "cofitok" else 1
    if (
        int(raw.get("checkpoint_step", -1)) != step
        or int(raw.get("sample_count", -1)) != 2_048
        or int(raw.get("selected_prefix_budget", -1)) != expected_budget
        or raw.get("weights") != "ema"
        or not _hex_digest(raw.get("checkpoint_sha256"), length=64)
        or not _hex_digest(raw.get("sample_set_sha256"), length=64)
        or not str(raw.get("checkpoint_integrity_manifest", "")).endswith(
            ".integrity.json"
        )
        or not isinstance(raw.get("sampling"), Mapping)
    ):
        raise ValueError(f"quality bridge milestone {step} {method} identity differs")
    return {
        "checkpoint": str(raw.get("checkpoint", "")),
        "checkpoint_sha256": str(raw["checkpoint_sha256"]),
        "checkpoint_integrity_manifest": str(raw["checkpoint_integrity_manifest"]),
        "fid": _finite(raw.get("fid"), label=f"milestone {step} {method} FID"),
        "inception_score": _finite(
            raw.get("inception_score"),
            label=f"milestone {step} {method} inception score",
        ),
        "endpoint_clean_mse": _finite(
            raw.get("endpoint_clean_mse"),
            label=f"milestone {step} {method} endpoint MSE",
        ),
        "sampling": copy.deepcopy(dict(raw["sampling"])),
    }


def _trend(
    milestones: Mapping[int, Mapping[str, Any]],
    *,
    terminal_methods: Mapping[str, Any],
) -> dict[str, Any]:
    if set(milestones) != {50_000, 100_000}:
        raise ValueError("quality bridge follow-up requires exact 50K/100K milestones")
    rows: dict[int, dict[str, dict[str, Any]]] = {}
    for step in (50_000, 100_000):
        rows[step] = {
            method: _milestone_row(milestones[step], step=step, method=method)
            for method in ("cofitok", "dense_identity")
        }

    method_trends: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        earlier = rows[50_000][method]
        later = rows[100_000][method]
        if _trend_sampling_protocol(earlier["sampling"]) != _trend_sampling_protocol(
            later["sampling"]
        ):
            raise ValueError(f"{method} 50K/100K milestone sampling protocol differs")
        terminal = terminal_methods.get(method)
        if not isinstance(terminal, Mapping):
            raise ValueError(f"quality bridge terminal {method} row is missing")
        if (
            later["checkpoint_sha256"] != terminal.get("checkpoint_sha256")
            or later["checkpoint_integrity_manifest"]
            != terminal.get("checkpoint_integrity_manifest")
        ):
            raise ValueError(
                f"{method} 100K milestone and terminal evaluation use different checkpoints"
            )
        fid_delta = later["fid"] - earlier["fid"]
        inception_delta = later["inception_score"] - earlier["inception_score"]
        method_trends[method] = {
            "fid_50000": earlier["fid"],
            "fid_100000": later["fid"],
            "fid_delta": fid_delta,
            "fid_relative_change": fid_delta / max(earlier["fid"], 1e-12),
            "fid_strictly_improved": later["fid"] < earlier["fid"],
            "inception_score_50000": earlier["inception_score"],
            "inception_score_100000": later["inception_score"],
            "inception_score_delta": inception_delta,
            "inception_score_strictly_improved": (
                later["inception_score"] > earlier["inception_score"]
            ),
            "endpoint_clean_mse_50000": earlier["endpoint_clean_mse"],
            "endpoint_clean_mse_100000": later["endpoint_clean_mse"],
            "checkpoint_100000_matches_terminal": True,
        }

    shared_fid = all(
        method_trends[method]["fid_strictly_improved"]
        for method in method_trends
    )
    shared_is = all(
        method_trends[method]["inception_score_strictly_improved"]
        for method in method_trends
    )
    return {
        "role": "descriptive_non_claim_experiment_routing_evidence",
        "formal_quality_gate": False,
        "threshold_policy": (
            "No post-hoc magnitude threshold is used. Shared trend support requires "
            "strictly lower FID for both matched methods from 50K to 100K under the "
            "same milestone protocol. Inception Score direction is recorded as a "
            "sensitivity check but cannot independently veto experiment routing from "
            "a 2,048-sample early-warning milestone."
        ),
        "methods": method_trends,
        "shared_fid_strictly_improved": shared_fid,
        "shared_inception_score_strictly_improved": shared_is,
        "shared_quality_trend_strictly_improved": shared_fid,
        "shared_inception_score_corroborates_fid_trend": shared_is,
    }


def _route(
    *,
    checks: Mapping[str, Mapping[str, Any]],
    trend: Mapping[str, Any],
    terminal_alerts: list[str],
) -> dict[str, Any]:
    failed = {name for name, row in checks.items() if row["passed"] is False}
    conflicts = sorted(
        alert
        for alert in terminal_alerts
        if checks[_ALERT_TO_TERMINAL_CHECK[alert]]["passed"] is True
    )
    common = {
        "execution_ready": False,
        "gpu_execution_allowed": False,
        "full_300k_launch_allowed": False,
        "release_authorization_allowed": False,
    }
    if conflicts:
        return {
            **common,
            "id": "reconcile_100k_cross_protocol_evidence",
            "category": "evidence_conflict",
            "objective": (
                "Reconcile the 2,048-sample DDIM-50 milestone alert with the "
                "10,000-sample DDIM-100 terminal result on the same 100K checkpoint."
            ),
            "trigger": {"contradictory_terminal_alerts": conflicts},
            "required_next_evidence": (
                "A source-bound replay that isolates sample-count/sampler-step effects "
                "without training or changing the formal protocol."
            ),
        }
    if not failed:
        return {
            **common,
            "id": "build_source_compatible_formal_quality_gate",
            "category": "formal_gate_preparation",
            "objective": (
                "Build and independently verify a new source-compatible formal gate "
                "for the completed full-data 100K evidence."
            ),
            "trigger": {"terminal_quality_screen": "pass"},
            "required_next_evidence": (
                "A new gate that replays the physical bridge sources and preserves all "
                "frozen absolute, matched, mechanism, class-fidelity, and authorization rules."
            ),
        }

    mechanism = sorted(failed & MECHANISM_CHECKS)
    if mechanism:
        return {
            **common,
            "id": "run_matched_factorization_mechanism_recovery_probe",
            "category": "cofitok_mechanism_recovery",
            "objective": (
                "Diagnose and repair the ordered restricted factorization before any "
                "capacity or training-scale increase."
            ),
            "trigger": {"failed_checks": mechanism},
            "required_next_evidence": (
                "A fresh matched bounded probe that restores endpoint, ordering, coarse-token, "
                "exact-zero, and shuffle invariants as applicable."
            ),
        }

    matched = sorted(failed & MATCHED_QUALITY_CHECKS)
    if matched:
        return {
            **common,
            "id": "run_matched_factorization_quality_regression_probe",
            "category": "matched_quality_regression",
            "objective": (
                "Determine why CoFiTok loses matched distribution quality before increasing "
                "model capacity or training budget."
            ),
            "trigger": {"failed_checks": matched},
            "required_next_evidence": (
                "A source-matched diagnostic that keeps data, backbone, optimizer, and sampler "
                "fixed while isolating factorization-only losses or token layout."
            ),
        }

    if "class_fidelity" in failed:
        return {
            **common,
            "id": "run_class_conditioning_fidelity_diagnostic",
            "category": "class_conditioning_recovery",
            "objective": (
                "Localize class-conditioning failure in labels, CFG dropout, sampling, or "
                "class-fidelity evaluation before larger training."
            ),
            "trigger": {"failed_checks": ["class_fidelity"]},
            "required_next_evidence": (
                "A matched class-conditional confusion/support diagnostic on the existing "
                "100K samples and checkpoint before any retraining."
            ),
        }

    absolute = sorted(failed & ABSOLUTE_QUALITY_CHECKS)
    if failed == set(absolute) and absolute:
        if trend["shared_quality_trend_strictly_improved"] is True:
            return {
                **common,
                "id": "prepare_matched_250m_capacity_qualification_probe",
                "category": "capacity_qualification",
                "objective": (
                    "Prepare a bounded matched 250M-parameter capacity qualification probe; "
                    "do not extend the rejected base-128 recipe to 150K and do not launch 300K."
                ),
                "trigger": {
                    "failed_checks": absolute,
                    "shared_50k_to_100k_quality_trend": "strictly_improved",
                },
                "required_next_evidence": (
                    "A fresh source-bound CUDA/runtime qualification and small matched probe "
                    "that tests capacity without claiming or authorizing full training."
                ),
            }
        return {
            **common,
            "id": "diagnose_terminal_distribution_support_then_recipe_probe",
            "category": "recipe_or_objective_intervention",
            "objective": (
                "Reuse the physical 100K terminal samples for distribution-support diagnosis, "
                "then prepare a bounded matched recipe/objective intervention instead of more "
                "base-128 steps."
            ),
            "trigger": {
                "failed_checks": absolute,
                "shared_50k_to_100k_quality_trend": "not_strictly_improved",
            },
            "required_next_evidence": (
                "A source-bound support audit followed by the smallest matched probe that tests "
                "the diagnosed data/optimization/objective bottleneck."
            ),
        }

    return {
        **common,
        "id": "extend_followup_policy_before_execution",
        "category": "unclassified_fail_closed",
        "objective": "Extend and test the decision policy before executing another experiment.",
        "trigger": {"failed_checks": sorted(failed)},
        "required_next_evidence": (
            "An explicit source-bound policy rule and synthetic branch tests for this combination."
        ),
    }


def build_quality_bridge_followup_decision(
    *,
    quality_bridge_result: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    milestones: Mapping[int, Mapping[str, Any]],
    milestone_identities: Mapping[int, Mapping[str, Any]],
    milestone_verifications: Mapping[int, Mapping[str, Any]],
    decision_git: Mapping[str, Any],
) -> dict[str, Any]:
    result_identity = _identity(
        quality_bridge_result_identity,
        label="quality bridge result",
    )
    builder_git = _git(decision_git, label="follow-up decision builder")
    result_git = _git(
        quality_bridge_result.get("git", {}),
        label="quality bridge result",
    )
    if result_git != {
        "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
        "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
        "tracked_dirty": False,
    }:
        raise ValueError("follow-up decision requires the exact quality bridge execution target")
    if (
        int(quality_bridge_result.get("schema_version", -1)) != 1
        or quality_bridge_result.get("status") != "completed"
        or quality_bridge_result.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or quality_bridge_result.get("stage") != QUALITY_BRIDGE_RESULT_STAGE
        or quality_bridge_result.get("authorization_boundary")
        != {
            "quality_bridge_evidence_complete": True,
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_authorization_allowed": False,
            "new_gate_required": True,
        }
    ):
        raise ValueError("quality bridge terminal result boundary differs")
    screen = quality_bridge_result.get("quality_screen")
    if not isinstance(screen, Mapping):
        raise ValueError("quality bridge result has no quality screen")
    checks = _quality_checks(screen)

    if (
        set(milestone_identities) != {50_000, 100_000}
        or set(milestone_verifications) != {50_000, 100_000}
    ):
        raise ValueError("quality bridge follow-up milestone evidence set differs")
    normalized_identities = {
        str(step): _identity(identity, label=f"milestone {step}")
        for step, identity in milestone_identities.items()
    }
    source_reports = quality_bridge_result.get("source_reports")
    if not isinstance(source_reports, Mapping):
        raise ValueError("quality bridge result source reports are missing")
    for step in (50_000, 100_000):
        expected = source_reports.get(f"milestone_{step}")
        if normalized_identities[str(step)] != expected:
            raise ValueError(f"quality bridge result binds another milestone {step}")
        verification = milestone_verifications[step]
        if (
            verification.get("status") != "verified"
            or verification.get("source_profile") != "quality_bridge"
        ):
            raise ValueError(f"quality bridge milestone {step} was not verified")

    terminal = quality_bridge_result.get("terminal")
    if not isinstance(terminal, Mapping) or not isinstance(
        terminal.get("methods"), Mapping
    ):
        raise ValueError("quality bridge terminal method evidence is missing")
    if set(terminal["methods"]) != {"cofitok", "dense_identity"}:
        raise ValueError("quality bridge terminal method set differs")
    trend = _trend(milestones, terminal_methods=terminal["methods"])
    terminal_milestone = milestones[100_000]
    alerts = terminal_milestone.get("quality_alerts")
    if (
        not isinstance(alerts, list)
        or any(alert not in _ALERT_TO_TERMINAL_CHECK for alert in alerts)
    ):
        raise ValueError("quality bridge 100K milestone alerts are invalid")
    recommendation = _route(
        checks=checks,
        trend=trend,
        terminal_alerts=list(alerts),
    )

    failed_checks = list(screen["failed_checks"])
    terminal_methods = terminal["methods"]
    terminal_summary = {
        method: {
            "fid": _finite(row.get("fid"), label=f"terminal {method} FID"),
            "precision": _finite(
                row.get("precision"), label=f"terminal {method} precision"
            ),
            "recall": _finite(
                row.get("recall"), label=f"terminal {method} recall"
            ),
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "sample_set_sha256": str(row.get("sample_set_sha256", "")),
        }
        for method, row in terminal_methods.items()
    }
    return {
        "schema_version": FOLLOWUP_DECISION_SCHEMA_VERSION,
        "status": "completed",
        "role": FOLLOWUP_DECISION_ROLE,
        "decision_builder_git": builder_git,
        "quality_bridge_execution_git": result_git,
        "source_reports": {
            "quality_bridge_result": result_identity,
            "milestones": normalized_identities,
        },
        "source_replay": {
            "quality_bridge_result_rebuilt_byte_equivalent": True,
            "physical_checkpoint_sample_and_real_set_reverified": True,
            "milestone_source_reports_reverified": True,
            "milestone_verifications": {
                str(step): copy.deepcopy(dict(milestone_verifications[step]))
                for step in (50_000, 100_000)
            },
        },
        "terminal_quality": {
            "status": screen["status"],
            "failed_checks": failed_checks,
            "checks": copy.deepcopy(list(screen["checks"])),
            "methods": terminal_summary,
            "class_fidelity": copy.deepcopy(terminal.get("class_fidelity")),
        },
        "milestone_trend": trend,
        "recommended_next_stage": recommendation,
        "claim_policy": {
            "experiment_selection_only": True,
            "milestone_trend_is_formal_quality_evidence": False,
            "terminal_result_is_promotion_gate": False,
            "cross_tier_numeric_ranking_allowed": False,
            "interpretation": (
                "The terminal 10K result controls quality status. The matched 2,048-sample "
                "milestones only route the next experiment and cannot authorize scaling, "
                "release, or a formal generation claim."
            ),
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
