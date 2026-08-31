"""Source-bound preparation for separating exposure from capacity effects.

This module deliberately stops before execution authorization.  It records the
evidence required to prepare a future matched stage while keeping all compute,
sampling, promotion, and release permissions disabled.
"""

from __future__ import annotations

import math
import re
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA_VERSION = "cofitok_generation_exposure_capacity_preparation_v1"
DECISION = "preserve_qualified_objective_prepare_exposure_capacity_disambiguation"
ROUTE_ID = "prepare_source_bound_exposure_or_capacity_gate"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _portable_path(value: str) -> PurePath:
    """Interpret serialized POSIX paths even when validation runs on Windows."""
    return PurePosixPath(value) if value.startswith("/") else PureWindowsPath(value)

PREPARATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_authorized": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "terminal_hold_replacement_allowed": False,
}

_SOURCE_NAMES = (
    "quality_bridge_result",
    "post_reconciliation_decision",
    "cross_protocol_reconciliation",
    "sampling_recovery_result",
    "min_snr_result",
    "pair_monitor",
    "cofitok_training_report",
    "dense_training_report",
    "cofitok_metrics",
    "dense_metrics",
)

_TRAINING_RUN_SUFFIXES = {
    "cofitok": "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
    "dense_identity": "dense_rollout_x0_u2_ema_teacher",
}
_TRAINING_LAYOUTS = {
    "cofitok": ("fixed_basis", 8, True),
    "dense_identity": ("dense_identity", 1, False),
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _boundary_is_disabled(boundary: Any, name: str) -> None:
    values = _object(boundary, name)
    for key, value in values.items():
        if (
            (key.endswith("_allowed") or key.endswith("_authorized"))
            and value is True
        ) or (key == "decision_is_execution_authorization" and value is True):
            raise ValueError(f"{name} must keep every execution permission disabled")


def _source_descriptors(source_identities: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if set(source_identities) != set(_SOURCE_NAMES):
        missing = sorted(set(_SOURCE_NAMES) - set(source_identities))
        extra = sorted(set(source_identities) - set(_SOURCE_NAMES))
        raise ValueError(f"source identities differ; missing={missing}, extra={extra}")
    normalized: dict[str, dict[str, Any]] = {}
    for name in _SOURCE_NAMES:
        descriptor = _object(source_identities[name], f"source identity {name}")
        path = descriptor.get("path")
        byte_count = descriptor.get("bytes")
        digest = descriptor.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"source identity {name} path is missing")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 1
        ):
            raise ValueError(f"source identity {name} bytes are invalid")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"source identity {name} SHA256 is invalid")
        normalized[name] = {
            "path": path,
            "bytes": byte_count,
            "sha256": digest,
        }
    return normalized


def _checkpoint_summary(report: dict[str, Any], *, method: str) -> dict[str, Any]:
    """Capture the immutable 100K payload identity used by exposure resume."""
    run_dir = report.get("output_dir")
    if not isinstance(run_dir, str) or not run_dir or not _portable_path(run_dir).is_absolute():
        raise ValueError(f"{method} training report output directory is invalid")
    if _portable_path(run_dir).name != _TRAINING_RUN_SUFFIXES[method]:
        raise ValueError(f"{method} training report output directory identifies another run")
    config = _object(report.get("config"), f"{method} training config")
    model = _object(config.get("model"), f"{method} training model config")
    expected_mode, expected_tokens, expected_feedback = _TRAINING_LAYOUTS[method]
    if model.get("synthesis_mode") != expected_mode:
        raise ValueError(f"{method} training model layout differs")
    if model.get("token_count") != expected_tokens:
        raise ValueError(f"{method} training token count differs")
    if model.get("predictor_use_feedback") is not expected_feedback:
        raise ValueError(f"{method} training feedback layout differs")
    latest = _object(report.get("latest_checkpoint"), f"{method} latest checkpoint")
    if latest.get("step") != 100_000:
        raise ValueError(f"{method} latest checkpoint is not the completed 100K step")
    name = latest.get("checkpoint")
    if name != "checkpoint_step_00100000.pt":
        raise ValueError(f"{method} latest checkpoint filename differs")
    sidecar = latest.get("integrity_manifest")
    if sidecar != f"{name}.integrity.json":
        raise ValueError(f"{method} latest checkpoint sidecar name differs")
    byte_count = latest.get("checkpoint_bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 1:
        raise ValueError(f"{method} latest checkpoint byte count is invalid")
    digest = latest.get("checkpoint_sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{method} latest checkpoint SHA256 is invalid")
    return {
        "run_dir": run_dir,
        "step": 100_000,
        "checkpoint": {
            "name": name,
            "bytes": byte_count,
            "sha256": digest,
        },
        "integrity_manifest_name": sidecar,
    }


def _validate_checkpoint_summaries(value: Any) -> dict[str, dict[str, Any]]:
    summaries = _object(value, "preparation source checkpoint summaries")
    if set(summaries) != {"cofitok", "dense_identity"}:
        raise ValueError("preparation source checkpoint methods differ")
    normalized: dict[str, dict[str, Any]] = {}
    for method in ("cofitok", "dense_identity"):
        row = _object(summaries.get(method), f"{method} source checkpoint summary")
        run_dir = row.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir or not _portable_path(run_dir).is_absolute():
            raise ValueError(f"{method} source checkpoint run directory is invalid")
        if row.get("step") != 100_000:
            raise ValueError(f"{method} source checkpoint step differs")
        checkpoint = _object(row.get("checkpoint"), f"{method} source checkpoint")
        if checkpoint.get("name") != "checkpoint_step_00100000.pt":
            raise ValueError(f"{method} source checkpoint filename differs")
        byte_count = checkpoint.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 1:
            raise ValueError(f"{method} source checkpoint byte count is invalid")
        digest = checkpoint.get("sha256")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{method} source checkpoint SHA256 is invalid")
        if row.get("integrity_manifest_name") != "checkpoint_step_00100000.pt.integrity.json":
            raise ValueError(f"{method} source checkpoint sidecar name differs")
        normalized[method] = {
            "run_dir": run_dir,
            "step": 100_000,
            "checkpoint": {
                "name": checkpoint["name"],
                "bytes": byte_count,
                "sha256": digest,
            },
            "integrity_manifest_name": row["integrity_manifest_name"],
        }
    return normalized


def validate_source_checkpoint_bindings(
    preparation: dict[str, Any],
    *,
    cofitok_training_report: dict[str, Any],
    dense_training_report: dict[str, Any],
) -> None:
    """Recompute checkpoint summaries from the referenced reports.

    The preparation report is content-addressed, but its embedded summaries
    must also be checked against the content-addressed training reports. This
    prevents a consumer from changing both a summary and the preparation hash
    while silently selecting a different resume payload.
    """
    evidence = _object(preparation.get("evidence"), "preparation evidence")
    training = _object(evidence.get("training"), "preparation training evidence")
    actual = _validate_checkpoint_summaries(training.get("source_checkpoints"))
    expected = {
        "cofitok": _checkpoint_summary(cofitok_training_report, method="cofitok"),
        "dense_identity": _checkpoint_summary(
            dense_training_report, method="dense_identity"
        ),
    }
    if actual != expected:
        raise ValueError("preparation source checkpoint summaries do not match training reports")


def _validate_evidence(
    *,
    objective_reassessment: dict[str, Any],
    quality_bridge_result: dict[str, Any],
    post_reconciliation_decision: dict[str, Any],
    cross_protocol_reconciliation: dict[str, Any],
    sampling_recovery_result: dict[str, Any],
    min_snr_result: dict[str, Any],
    pair_monitor: dict[str, Any],
    cofitok_metrics: dict[str, Any],
    dense_metrics: dict[str, Any],
    cofitok_training_report: dict[str, Any],
    dense_training_report: dict[str, Any],
) -> dict[str, Any]:
    if objective_reassessment.get("schema") != "cofitok_generation_training_objective_reassessment_v1":
        raise ValueError("objective reassessment schema is not recognized")
    if objective_reassessment.get("status") != "completed":
        raise ValueError("objective reassessment is not completed")
    if objective_reassessment.get("decision") != "preserve_qualified_objective_defer_new_intervention":
        raise ValueError("objective reassessment decision is not the current qualified decision")
    if objective_reassessment.get("generation_advantage_proven") is not False:
        raise ValueError("objective reassessment must retain generation_advantage_proven=false")
    if objective_reassessment.get("execution_ready") is not False:
        raise ValueError("objective reassessment unexpectedly authorizes execution")
    if objective_reassessment.get("next_evidence", {}).get("id") != ROUTE_ID:
        raise ValueError("objective reassessment does not select the exposure/capacity route")
    _boundary_is_disabled(
        objective_reassessment.get("authorization_boundary"),
        "objective reassessment authorization boundary",
    )

    if post_reconciliation_decision.get("status") != "completed":
        raise ValueError("post-reconciliation decision is not completed")
    if post_reconciliation_decision.get("operational_status") != "pass":
        raise ValueError("post-reconciliation decision did not pass operational replay")
    if post_reconciliation_decision.get("terminal_status") != "hold":
        raise ValueError("post-reconciliation decision must preserve terminal hold")
    if post_reconciliation_decision.get("generation_advantage_proven") is not False:
        raise ValueError("post-reconciliation decision must retain generation_advantage_proven=false")
    _boundary_is_disabled(
        post_reconciliation_decision.get("authorization_boundary"),
        "post-reconciliation authorization boundary",
    )

    if quality_bridge_result.get("status") != "completed":
        raise ValueError("quality bridge result is not completed")
    terminal = _object(quality_bridge_result.get("terminal"), "quality bridge terminal")
    quality_screen = _object(
        quality_bridge_result.get("quality_screen"), "quality bridge quality screen"
    )
    terminal_status = terminal.get("status", quality_screen.get("status"))
    if terminal_status != "hold":
        raise ValueError("quality bridge terminal status is not hold")
    if quality_screen.get("status") != "hold":
        raise ValueError("quality bridge quality screen is not hold")
    if quality_screen.get("non_authorizing") is not True:
        raise ValueError("quality bridge quality screen must be non-authorizing")
    # The canonical quality-bridge schema derives the scientific hold from the
    # non-authorizing quality screen and does not carry a top-level claim flag.
    # If a newer source includes that flag, it must still explicitly remain false.
    if (
        "generation_advantage_proven" in quality_bridge_result
        and quality_bridge_result["generation_advantage_proven"] is not False
    ):
        raise ValueError("quality bridge must retain generation_advantage_proven=false")
    bridge_boundary = quality_bridge_result.get("authorization_boundary")
    if not isinstance(bridge_boundary, dict):
        raise ValueError("quality bridge authorization boundary is missing")
    for key in PREPARATION_BOUNDARY:
        if key in bridge_boundary and bridge_boundary[key] is True:
            raise ValueError(f"quality bridge enables forbidden permission {key}")

    if cross_protocol_reconciliation.get("status") != "completed":
        raise ValueError("cross-protocol reconciliation is not completed")
    if cross_protocol_reconciliation.get("operational_status") != "pass":
        raise ValueError("cross-protocol reconciliation did not pass operational replay")
    if cross_protocol_reconciliation.get("terminal_status") != "hold":
        raise ValueError("cross-protocol reconciliation must preserve terminal hold")

    if sampling_recovery_result.get("status") not in {"completed", "pass"}:
        raise ValueError("sampling-recovery result is not completed")
    if sampling_recovery_result.get("scientific_status") != "screening_only":
        raise ValueError("sampling-recovery result is not screening-only")
    if sampling_recovery_result.get("selection_status") != "no_shared_sampling_recovery_candidate":
        raise ValueError("sampling-recovery route is not the recorded no-candidate result")
    if sampling_recovery_result.get("generation_advantage_proven") is not False:
        raise ValueError("sampling-recovery must not prove a generation advantage")

    if min_snr_result.get("status") != "completed":
        raise ValueError("Min-SNR result is not completed")
    if min_snr_result.get("scientific_status") != "screening_only":
        raise ValueError("Min-SNR result is not screening-only")
    if min_snr_result.get("selection_status") != "no_shared_min_snr_candidate_at_50k":
        raise ValueError("Min-SNR route is not the recorded no-candidate result")
    if min_snr_result.get("generation_advantage_proven") is not False:
        raise ValueError("Min-SNR must not prove a generation advantage")

    if pair_monitor.get("status") != "pass" or pair_monitor.get("stage") != "complete":
        raise ValueError("pair monitor is not a completed pass")
    if pair_monitor.get("issues") != []:
        raise ValueError("pair monitor reports issues")
    runs = _object(pair_monitor.get("runs"), "pair monitor runs")
    for method in ("cofitok", "dense_identity"):
        run = _object(runs.get(method), f"pair monitor {method} run")
        if run.get("complete") is not True or run.get("last_step") != 100_000:
            raise ValueError(f"pair monitor {method} is not complete at 100K")
        if run.get("expected_steps") != 100_000:
            raise ValueError(f"pair monitor {method} expected step differs")
        last_metric = _object(run.get("last_metric"), f"pair monitor {method} last metric")
        if last_metric.get("step") != 100_000 or last_metric.get("samples_seen") != 6_400_000:
            raise ValueError(f"pair monitor {method} exposure binding differs")

    metric_tails = {"cofitok": cofitok_metrics, "dense_identity": dense_metrics}
    for method, tail in metric_tails.items():
        if tail.get("step") != 100_000 or tail.get("samples_seen") != 6_400_000:
            raise ValueError(f"{method} metrics tail is not the completed 100K row")
        value = tail.get("validation_epsilon_mse")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{method} validation epsilon MSE is not finite")

    reports = {"cofitok": cofitok_training_report, "dense_identity": dense_training_report}
    revisions: set[str] = set()
    branches: set[str] = set()
    for method, report in reports.items():
        if report.get("training_complete") is not True:
            raise ValueError(f"{method} training report is not complete")
        if report.get("completed_steps") != 100_000 or report.get("target_steps") != 100_000:
            raise ValueError(f"{method} training report is not the completed 100K bridge")
        git = _object(report.get("git"), f"{method} training Git")
        revision = git.get("revision")
        branch = git.get("branch")
        if not isinstance(revision, str) or GIT_REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError(f"{method} training revision is invalid")
        if not isinstance(branch, str) or not branch:
            raise ValueError(f"{method} training branch is missing")
        if git.get("dirty") is not False:
            raise ValueError(f"{method} training checkout is dirty")
        revisions.add(revision)
        branches.add(branch)
    if len(revisions) != 1 or len(branches) != 1:
        raise ValueError("matched training reports do not share one Git revision and branch")

    source_checkpoints = {
        method: _checkpoint_summary(report, method=method)
        for method, report in reports.items()
    }

    return {
        "quality_bridge": {
            "status": quality_bridge_result["status"],
            "terminal_status": terminal_status,
            "generation_advantage_proven": False,
        },
        "post_reconciliation_decision": {
            "status": post_reconciliation_decision["status"],
            "operational_status": post_reconciliation_decision["operational_status"],
            "terminal_status": post_reconciliation_decision["terminal_status"],
        },
        "cross_protocol_reconciliation": {
            "status": cross_protocol_reconciliation["status"],
            "operational_status": cross_protocol_reconciliation["operational_status"],
            "terminal_status": cross_protocol_reconciliation["terminal_status"],
        },
        "sampling_recovery": {
            "status": sampling_recovery_result["status"],
            "selection_status": sampling_recovery_result["selection_status"],
        },
        "min_snr": {
            "status": min_snr_result["status"],
            "scientific_status": min_snr_result["scientific_status"],
            "selection_status": min_snr_result["selection_status"],
        },
        "pair_monitor": {
            "status": pair_monitor["status"],
            "stage": pair_monitor["stage"],
            "issues": list(pair_monitor["issues"]),
        },
        "training": {
            "completed_steps": 100_000,
            "images_seen_per_method": 6_400_000,
            "git_revision": next(iter(revisions)),
            "git_branch": next(iter(branches)),
            "matched_primary_loss_divergence_is_large": False,
            "validation_epsilon_mse": {
                "cofitok": float(cofitok_metrics["validation_epsilon_mse"]),
                "dense_identity": float(dense_metrics["validation_epsilon_mse"]),
            },
            "source_checkpoints": source_checkpoints,
        },
    }


def _candidate_arms(
    *,
    exposure_output_root: str,
    capacity_output_root: str,
) -> dict[str, dict[str, Any]]:
    for name, value in {
        "exposure_output_root": exposure_output_root,
        "capacity_output_root": capacity_output_root,
    }.items():
        if not isinstance(value, str) or not value.startswith("/root/autodl-tmp/CoFiTok/checkpoints/generation/"):
            raise ValueError(f"{name} must be a new project generation output root")
        if value.endswith("/stability_full_data_100k_base128_quality_bridge_v1"):
            raise ValueError(f"{name} must not reuse the locked quality-bridge root")

    return {
        "exposure_continuation": {
            "role": "same_capacity_exposure_discriminator",
            "controlled_change": "training_exposure_only",
            "source_stage": "stability_quality_bridge",
            "source_base_channels": 128,
            "model_layout_change_allowed": False,
            "objective_change_allowed": False,
            "conditioning_change_allowed": False,
            "initialization": "exact_100k_checkpoint_resume_only",
            "target_step": "must_be_selected_by_a_new_stage_gate",
            "output_root": exposure_output_root,
            "qualification_protocol": "matched_10000_ddim100_plus_class_fidelity_and_mechanism",
            "formal_quality_claim_allowed": False,
        },
        "capacity_qualification": {
            "role": "matched_capacity_discriminator",
            "controlled_change": "model_capacity_only",
            "source_stage": "stability_quality_bridge",
            "source_base_channels": 128,
            "candidate_base_channels": 256,
            "token_layout_change_allowed": False,
            "objective_change_allowed": False,
            "conditioning_change_allowed": False,
            "initialization": "fresh_matched_initialization_required",
            "qualification_steps": 10_000,
            "output_root": capacity_output_root,
            "qualification_protocol": "matched_1000_screen_then_10000_confirmation_if_selected",
            "formal_quality_claim_allowed": False,
        },
    }


def build_preparation(
    *,
    objective_reassessment: dict[str, Any],
    quality_bridge_result: dict[str, Any],
    post_reconciliation_decision: dict[str, Any],
    cross_protocol_reconciliation: dict[str, Any],
    sampling_recovery_result: dict[str, Any],
    min_snr_result: dict[str, Any],
    pair_monitor: dict[str, Any],
    cofitok_metrics: dict[str, Any],
    dense_metrics: dict[str, Any],
    cofitok_training_report: dict[str, Any],
    dense_training_report: dict[str, Any],
    source_identities: dict[str, dict[str, Any]],
    builder_git: dict[str, Any],
    exposure_output_root: str,
    capacity_output_root: str,
) -> dict[str, Any]:
    evidence = _validate_evidence(
        objective_reassessment=objective_reassessment,
        quality_bridge_result=quality_bridge_result,
        post_reconciliation_decision=post_reconciliation_decision,
        cross_protocol_reconciliation=cross_protocol_reconciliation,
        sampling_recovery_result=sampling_recovery_result,
        min_snr_result=min_snr_result,
        pair_monitor=pair_monitor,
        cofitok_metrics=cofitok_metrics,
        dense_metrics=dense_metrics,
        cofitok_training_report=cofitok_training_report,
        dense_training_report=dense_training_report,
    )
    sources = _source_descriptors(source_identities)
    git = _object(builder_git, "builder Git provenance")
    if not isinstance(git.get("revision"), str):
        raise ValueError("builder Git revision is missing")
    if not isinstance(git.get("branch"), str) or not git["branch"]:
        raise ValueError("builder Git branch is missing")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": "source_bound_exposure_capacity_preparation",
        "status": "prepared",
        "decision": DECISION,
        "route_id": ROUTE_ID,
        "scientific_scope": (
            "Preparation only: distinguish insufficient full-data exposure from "
            "capacity limitation while preserving the qualified objective."
        ),
        "evidence": evidence,
        "candidate_arms": _candidate_arms(
            exposure_output_root=exposure_output_root,
            capacity_output_root=capacity_output_root,
        ),
        "selection_policy": {
            "execution_order": "undetermined_until_new_source_compatible_gate",
            "shared_matched_pair_required": True,
            "fresh_remote_rehash_required": True,
            "exact_stage_authorization_required": True,
            "terminal_hold_replacement_allowed": False,
            "selection_by_current_fid_allowed": False,
            "automatic_300k_escalation_allowed": False,
            "failure_policy": "fail_closed",
        },
        "required_successor_gate": {
            "must_bind": [
                "current_preparation_sha256",
                "current_quality_bridge_result_sha256",
                "cross_protocol_reconciliation_sha256",
                "sampling_recovery_result_sha256",
                "min_snr_result_sha256",
                "exact_source_checkout_revision_tree_branch",
                "dataset_and_runtime_identity",
                "new_versioned_output_roots",
                "checkpoint_sidecars_and_latest_binding",
                "matched_sampling_and_class_fidelity_protocol",
            ],
            "must_reverify_before_gpu": True,
        },
        "sources": sources,
        "builder_git": git,
        "authorization_boundary": dict(PREPARATION_BOUNDARY),
    }


def validate_preparation(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("exposure/capacity preparation schema differs")
    if report.get("status") != "prepared" or report.get("decision") != DECISION:
        raise ValueError("exposure/capacity preparation is not in prepared state")
    if report.get("route_id") != ROUTE_ID:
        raise ValueError("exposure/capacity route id differs")
    if report.get("authorization_boundary") != PREPARATION_BOUNDARY:
        raise ValueError("preparation authorization boundary is not canonical")

    evidence = _object(report.get("evidence"), "preparation evidence")
    training = _object(evidence.get("training"), "preparation training evidence")
    if training.get("completed_steps") != 100_000:
        raise ValueError("preparation must bind the completed 100K bridge")
    if training.get("images_seen_per_method") != 6_400_000:
        raise ValueError("preparation must bind exact bridge exposure")
    if training.get("matched_primary_loss_divergence_is_large") is not False:
        raise ValueError("preparation has an invalid primary-loss interpretation")
    monitor = _object(evidence.get("pair_monitor"), "preparation pair monitor evidence")
    if monitor.get("status") != "pass" or monitor.get("stage") != "complete":
        raise ValueError("preparation pair monitor is not a completed pass")
    if monitor.get("issues") != []:
        raise ValueError("preparation pair monitor reports issues")
    validation = _object(training.get("validation_epsilon_mse"), "preparation validation evidence")
    for method in ("cofitok", "dense_identity"):
        value = validation.get(method)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"preparation validation value for {method} is invalid")
    _validate_checkpoint_summaries(training.get("source_checkpoints"))

    for key, expected in {
        "quality_bridge": ("terminal_status", "hold"),
        "post_reconciliation_decision": ("terminal_status", "hold"),
        "cross_protocol_reconciliation": ("terminal_status", "hold"),
        "sampling_recovery": ("selection_status", "no_shared_sampling_recovery_candidate"),
        "min_snr": ("selection_status", "no_shared_min_snr_candidate_at_50k"),
    }.items():
        row = _object(evidence.get(key), f"preparation evidence {key}")
        if row.get(expected[0]) != expected[1]:
            raise ValueError(f"preparation evidence {key} does not preserve the recorded hold")

    arms = _object(report.get("candidate_arms"), "candidate arms")
    if set(arms) != {"exposure_continuation", "capacity_qualification"}:
        raise ValueError("candidate arm set differs")
    exposure = _object(arms["exposure_continuation"], "exposure arm")
    capacity = _object(arms["capacity_qualification"], "capacity arm")
    if exposure.get("controlled_change") != "training_exposure_only":
        raise ValueError("exposure arm changes more than exposure")
    if exposure.get("model_layout_change_allowed") is not False:
        raise ValueError("exposure arm permits a layout change")
    if capacity.get("controlled_change") != "model_capacity_only":
        raise ValueError("capacity arm does not isolate capacity")
    for arm_name, arm in arms.items():
        if arm.get("objective_change_allowed") is not False:
            raise ValueError(f"{arm_name} permits an objective change")
        if arm.get("conditioning_change_allowed") is not False:
            raise ValueError(f"{arm_name} permits a conditioning change")
        if arm.get("formal_quality_claim_allowed") is not False:
            raise ValueError(f"{arm_name} permits a formal quality claim")
        root = arm.get("output_root")
        if not isinstance(root, str) or not root.startswith(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        ):
            raise ValueError(f"{arm_name} output root is not project scoped")
        if root.endswith("/stability_full_data_100k_base128_quality_bridge_v1"):
            raise ValueError(
                f"{arm_name} output root must not reuse the locked quality-bridge root"
            )
    if exposure.get("output_root") == capacity.get("output_root"):
        raise ValueError("candidate arms must use distinct output roots")

    selection = _object(report.get("selection_policy"), "selection policy")
    if selection.get("exact_stage_authorization_required") is not True:
        raise ValueError("selection policy weakens exact stage authorization")
    if selection.get("automatic_300k_escalation_allowed") is not False:
        raise ValueError("selection policy permits automatic 300K escalation")
    if selection.get("failure_policy") != "fail_closed":
        raise ValueError("selection policy is not fail-closed")

    sources = _source_descriptors(_object(report.get("sources"), "preparation sources"))
    builder_git = _object(report.get("builder_git"), "preparation builder Git")
    if not isinstance(builder_git.get("revision"), str) or not builder_git["revision"]:
        raise ValueError("preparation builder Git revision is missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "route_id": ROUTE_ID,
        "source_count": len(sources),
        "candidate_arm_ids": sorted(arms),
        "authorization_boundary": dict(PREPARATION_BOUNDARY),
    }
