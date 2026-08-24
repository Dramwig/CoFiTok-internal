from __future__ import annotations

import copy
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    EXECUTION_BOUNDARY as LEGACY_EXECUTION_BOUNDARY,
    PROBE_ROLE,
    PROBE_SCOPE,
    validate_conditioning_ranking_probe_preparation,
    validate_standing_experiment_authorization,
)


SCHEMA_VERSION = 1
PREPARATION_ROLE = PROBE_ROLE
PREPARATION_SCOPE = PROBE_SCOPE
AUTHORIZATION_ROLE = (
    "generation_conditioning_ranking_terminal_rebind_execution_authorization"
)
AUTHORIZATION_SCOPE = (
    "imagenet256_10pct_four_arm_semantic_alignment_probe1k_terminal_rebind_v2"
)
STAGE = "conditioning_ranking_four_arm_probe1k_terminal_rebind_v2"
OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_probe1k_terminal_rebind_v2"
)
LEGACY_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_probe1k_v1"
)
LEGACY_PREPARATION_GIT = {
    "revision": "fc78b85ac78faa1498d749c6a24d78289cde8438",
    "tree": "587145f7962332ae74b89e177e29dbcf38b3139e",
    "branch": "analysis/generation-conditioning-classifier-integrity-v1-20260822",
    "tracked_dirty": False,
}
OFFICIAL_EPSILON_VALIDATOR_GIT = {
    "revision": "62b5c7a77052605dff579e17a7d9cc88397b4a31",
    "tree": "8482f1933ae4ac1ab8133bce70a5ddbf1c4686b3",
    "branch": "scale/generation-quality-repair-epsilon-stability-v1",
    "tracked_dirty": False,
}
OFFICIAL_EPSILON_VALIDATOR_SHA256 = (
    "aedc6cffc29b2b36844e964f5d4b4ce4c3dc6f7e9c5c292ec811cb297899845b"
)

EXPECTED_SOURCE_SHA256 = {
    "standing_authorization": (
        "5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df"
    ),
    "quality_bridge_result": (
        "15752e05611fa888e15352934c1627ccb95418df0339f9ad120e195bc088d165"
    ),
    "post_reconciliation_decision": (
        "45817b27c33866f9c48eddc557c921e48c69c708d9fd47ee09695e086a1c47f4"
    ),
    "post_reconciliation_verification": (
        "19512853989d8d001bacd063e20485ecaa57d96bdae75e40da069f8a0cb9352d"
    ),
    "epsilon_stability_result": (
        "d6b5c3190a205f5be886c01e8543b9e2511988d5f0b9f2e642a33d0e8be1249f"
    ),
    "epsilon_recovery_status": (
        "a1bac6233a8f23e8626e459f8e1953de88c9e9292d0615c8b6d2c03c9548642d"
    ),
    "conditioning_gain_comparison": (
        "15537338114718e42a8b74acca0768dbbca42385cf0e6526677e460d69191294"
    ),
    "supersession_receipt": (
        "ce7bd71d7ef4892a7c2eafb2d3c46c69d89f3e9c14c8a2ed7ede81bf40e9b860"
    ),
    "supersession_marker": (
        "a6388095e6a54cb350e375ac4efc319136035bce9aab38ab22de20bf8b4f8262"
    ),
    "legacy_preparation": (
        "02abb34b68666f3221a7d1111844b64b6ba13abdfc050116afc9b8d0c0c673d9"
    ),
}

POST_RECONCILIATION_AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
    "evaluation_launch_allowed": False,
    "export_allowed": False,
    "full_300k_launch_allowed": False,
    "full_training_launch_allowed": False,
    "legacy_conditioning_route_allowed": False,
    "legacy_factorization_route_allowed": False,
    "new_source_bound_execution_gate_required": True,
    "process_signals_allowed": False,
    "promotion_allowed": False,
    "release_allowed": False,
    "sampling_launch_allowed": False,
    "training_launch_allowed": False,
}
EPSILON_RESULT_AUTHORIZATION_BOUNDARY = {
    "associated_non_formal_evaluation_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "full_training_launch_allowed": False,
    "independent_matched_10000_confirmation_launch_allowed": False,
    "inference_export_allowed": False,
    "matched_1000_sample_sampling_launch_allowed": False,
    "process_signals_allowed": False,
    "promotion_allowed": False,
    "release_allowed": False,
    "training_launch_allowed": False,
}
EPSILON_RESULT_CLAIM_BOUNDARY = {
    "cofitok_generation_advantage_claim_allowed": False,
    "independent_matched_10000_confirmation_required": True,
    "min_snr_training_tested": False,
    "one_thousand_sample_screening_only": True,
    "selected_case_is_not_confirmed": False,
}
RECOVERY_AUTHORIZED_ACTIONS = {
    "associated_non_formal_evaluation": True,
    "full_300k": False,
    "independent_10000_confirmation": False,
    "inference_export": False,
    "matched_1000_sample_sampling": True,
    "process_signals": False,
    "promotion": False,
    "release": False,
    "training": False,
}
GAIN_CLAIM_BOUNDARY = {
    "authorizes_checkpoint_modification": False,
    "authorizes_sampling": False,
    "authorizes_training": False,
    "diagnostic_only": True,
    "replaces_formal_quality_gate": False,
}
SUPERSESSION_SCOPE = {
    "checkpoint_payload_loading_allowed": False,
    "cpu_only": True,
    "diagnostic_non_authorizing": True,
    "full_300k_launch_allowed": False,
    "full_training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "inference_export_authorization_allowed": False,
    "new_gpu_supervisor_launch_allowed": False,
    "process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "sampling_launch_allowed": False,
    "static_filesystem_interlock_only": True,
    "training_launch_allowed": False,
    "upstream_evidence_modified": False,
}
PREPARATION_CLAIM_BOUNDARY = {
    "semantic_alignment_screening_only": True,
    "absolute_fid_or_recall_recovery_tested": False,
    "generation_advantage_claim_allowed": False,
    "training_quality_claim_allowed": False,
    "sample_quality_claim_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
}
EXECUTION_BOUNDARY = {
    **copy.deepcopy(LEGACY_EXECUTION_BOUNDARY),
    "cpu_only_heldout_postevaluation_allowed": True,
    "max_training_run_count": 4,
    "max_optimizer_steps_total": 4_000,
    "full_300k_launch_allowed": False,
    "inference_export_allowed": False,
    "process_signals_allowed": False,
}
CLAIM_BOUNDARY = {
    **copy.deepcopy(PREPARATION_CLAIM_BOUNDARY),
    "diagnostic_non_authorizing": True,
    "terminal_result_replacement_allowed": False,
    "conditioning_only_failure_claim_allowed": False,
    "absolute_quality_recovery_claim_allowed": False,
}

EXPECTED_FAILED_CHECKS = [
    "cofitok_absolute_fid",
    "cofitok_recall_floor",
    "class_fidelity",
]
EPSILON_INPUT_NAMES = (
    "design",
    "execution_authorization",
    "observation_manifest",
    "real_artifact_reference",
)


def _is_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _is_hex(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _require_source(
    identity: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    source = _identity(identity, label=name)
    if source["sha256"] != EXPECTED_SOURCE_SHA256[name]:
        raise ValueError(f"{name} SHA256 differs from the terminal rebind contract")
    return source


def _clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    tree = value.get("tree")
    branch = value.get("branch")
    if (
        not _is_hex(revision, length=40)
        or not _is_hex(tree, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity is malformed")
    result = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    path = value.get("path")
    if path is not None:
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label} Git path is malformed")
        result["path"] = path
    return result


def _same_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("bytes") == right.get("bytes")
        and left.get("sha256") == right.get("sha256")
    )


def validate_post_reconciliation_sources(
    *,
    quality_result: Mapping[str, Any],
    quality_identity: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    verification: Mapping[str, Any],
    verification_identity: Mapping[str, Any],
) -> dict[str, Any]:
    quality_id = _require_source(quality_identity, name="quality_bridge_result")
    decision_id = _require_source(
        decision_identity,
        name="post_reconciliation_decision",
    )
    verification_id = _require_source(
        verification_identity,
        name="post_reconciliation_verification",
    )
    screen = quality_result.get("quality_screen")
    if (
        quality_result.get("schema_version") != 1
        or quality_result.get("role") != "stability_full_data_quality_bridge_result"
        or quality_result.get("status") != "completed"
        or not isinstance(screen, Mapping)
        or screen.get("status") != "hold"
        or screen.get("failed_checks") != EXPECTED_FAILED_CHECKS
    ):
        raise ValueError("quality-bridge mixed terminal hold differs")

    resolution = decision.get("scientific_resolution")
    failure = resolution.get("failure_classification") if isinstance(resolution, Mapping) else None
    terminal = resolution.get("terminal_quality") if isinstance(resolution, Mapping) else None
    legacy = decision.get("legacy_route_disposition")
    source_evidence = decision.get("source_evidence")
    source_replay = decision.get("source_replay")
    next_stage = decision.get("recommended_next_stage")
    if (
        decision.get("schema_version") != 1
        or decision.get("role")
        != "generation_100k_post_reconciliation_experiment_decision"
        or decision.get("status") != "completed"
        or decision.get("operational_status") != "pass"
        or decision.get("terminal_status") != "hold"
        or decision.get("generation_advantage_proven") is not False
        or decision.get("authorization_boundary")
        != POST_RECONCILIATION_AUTHORIZATION_BOUNDARY
        or not isinstance(failure, Mapping)
        or failure.get("class_only_failure") is not False
        or failure.get("factorization_mechanism_failure") is not False
        or failure.get("matched_quality_only_failure") is not False
        or failure.get("mixed_shared_absolute_quality_support_and_class_failure")
        is not True
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "hold"
        or terminal.get("failed_checks") != EXPECTED_FAILED_CHECKS
        or terminal.get("both_methods_absolute_fid_above_threshold") is not True
        or terminal.get("both_methods_recall_below_floor") is not True
        or terminal.get("class_fidelity_pass") is not False
        or not isinstance(legacy, Mapping)
        or legacy.get("conditioning_only_supervisor", {}).get("eligible") is not False
        or legacy.get("factorization_quality_regression_supervisor", {}).get("eligible")
        is not False
        or not isinstance(source_replay, Mapping)
        or not source_replay
        or any(value is not True for value in source_replay.values())
        or not isinstance(next_stage, Mapping)
        or next_stage.get("id")
        != "prepare_matched_100k_epsilon_stability_sampling_diagnostic"
        or next_stage.get("execution_ready") is not False
        or next_stage.get("gpu_execution_allowed") is not False
        or not isinstance(source_evidence, Mapping)
        or _identity(
            source_evidence.get("quality_bridge_result", {}),
            label="decision quality result",
        )
        != quality_id
    ):
        raise ValueError("post-reconciliation mixed-failure decision differs")

    if (
        verification.get("schema_version") != 1
        or verification.get("role")
        != "generation_100k_post_reconciliation_decision_verification"
        or verification.get("status") != "verified"
        or _identity(verification.get("decision", {}), label="verified decision")
        != decision_id
        or verification.get("authorization_boundary")
        != POST_RECONCILIATION_AUTHORIZATION_BOUNDARY
        or verification.get("scientific_resolution") != resolution
        or verification.get("recommended_next_stage") != next_stage
        or verification.get("source_replay") != source_replay
    ):
        raise ValueError("post-reconciliation verification differs")
    return {
        "quality_result": quality_id,
        "decision": decision_id,
        "verification": verification_id,
        "failure_classification": copy.deepcopy(dict(failure)),
        "failed_checks": copy.deepcopy(list(EXPECTED_FAILED_CHECKS)),
    }


def validate_epsilon_sampling_recovery(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    recovery_status: Mapping[str, Any],
    recovery_status_identity: Mapping[str, Any],
    expected_decision: Mapping[str, Any],
    expected_verification: Mapping[str, Any],
    official_replay: Mapping[str, Any],
) -> dict[str, Any]:
    result_id = _require_source(result_identity, name="epsilon_stability_result")
    recovery_id = _require_source(
        recovery_status_identity,
        name="epsilon_recovery_status",
    )
    decision_id = _identity(expected_decision, label="expected post decision")
    verification_id = _identity(
        expected_verification,
        label="expected post verification",
    )
    execution = result.get("execution")
    candidates = result.get("candidates")
    bindings = result.get("source_bindings")
    if (
        result.get("status") != "pass"
        or result.get("scientific_status") != "screening_only"
        or result.get("selection_status")
        != "no_shared_sampling_recovery_candidate"
        or result.get("selected_case_id") is not None
        or result.get("generation_advantage_proven") is not False
        or result.get("authorization_boundary")
        != EPSILON_RESULT_AUTHORIZATION_BOUNDARY
        or result.get("claim_boundary") != EPSILON_RESULT_CLAIM_BOUNDARY
        or not isinstance(candidates, list)
        or len(candidates) != 7
        or any(
            not isinstance(candidate, Mapping)
            or candidate.get("passes_shared_recovery_screen") is not False
            for candidate in candidates
        )
        or not isinstance(result.get("observations"), list)
        or len(result["observations"]) != 16
        or not isinstance(execution, Mapping)
        or execution.get("status") != "approved"
        or execution.get("scope")
        != "matched_1000_sample_epsilon_stability_sampling_diagnostic_only"
        or execution.get("output_root")
        != (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "stability_full_data_100k_epsilon_stability_sampling_recovery_v1"
        )
        or execution.get("output_root_non_overlapping") is not True
        or execution.get("git") != OFFICIAL_EPSILON_VALIDATOR_GIT
        or not isinstance(bindings, Mapping)
        or _identity(
            bindings.get("terminal_route_receipt", {}),
            label="epsilon terminal route receipt",
        )
        != decision_id
        or _identity(
            execution.get("source_bindings", {}).get(
                "post_reconciliation_verification", {}
            ),
            label="epsilon post-reconciliation verification",
        )
        != verification_id
    ):
        raise ValueError("epsilon-stability no-candidate result differs")

    replay_inputs = official_replay.get("input_graph")
    if (
        official_replay.get("status") != "verified"
        or official_replay.get("cuda_visible_devices") != "-1"
        or official_replay.get("returncode") != 0
        or official_replay.get("checkout_git") != OFFICIAL_EPSILON_VALIDATOR_GIT
        or _identity(official_replay.get("validator", {}), label="epsilon validator")
        ["sha256"]
        != OFFICIAL_EPSILON_VALIDATOR_SHA256
        or _identity(official_replay.get("result", {}), label="replayed epsilon result")
        != result_id
        or not isinstance(replay_inputs, Mapping)
        or set(replay_inputs) != set(EPSILON_INPUT_NAMES)
    ):
        raise ValueError("official epsilon result replay differs")
    for name in EPSILON_INPUT_NAMES:
        expected = _identity(bindings.get(name, {}), label=f"epsilon {name}")
        if _identity(replay_inputs.get(name, {}), label=f"replayed epsilon {name}") != expected:
            raise ValueError(f"official epsilon replay input differs: {name}")

    if (
        recovery_status.get("schema_version") != 1
        or recovery_status.get("role")
        != "generation_epsilon_stability_pre_gpu_recovery_controller"
        or recovery_status.get("status") != "completed"
        or recovery_status.get("stage") != "completed"
        or recovery_status.get("detail")
        != "matched_1000_sample_sampling_recovery_diagnostic_completed"
        or recovery_status.get("completed_arms") != 16
        or recovery_status.get("total_arms") != 16
        or recovery_status.get("child_pid") is not None
        or recovery_status.get("case_id") is not None
        or recovery_status.get("method") is not None
        or recovery_status.get("generation_advantage_proven") is not False
        or recovery_status.get("authorized_actions") != RECOVERY_AUTHORIZED_ACTIONS
        or recovery_status.get("result_authorization_boundary")
        != EPSILON_RESULT_AUTHORIZATION_BOUNDARY
        or _identity(recovery_status.get("result", {}), label="recovery result")
        != result_id
    ):
        raise ValueError("epsilon-stability recovery completion differs")
    return {
        "result": result_id,
        "recovery_status": recovery_id,
        "completed_arms": 16,
        "candidate_count": 7,
        "selection_status": "no_shared_sampling_recovery_candidate",
        "official_replay": copy.deepcopy(dict(official_replay)),
    }


def validate_gain_diagnostic(
    *,
    report: Mapping[str, Any],
    report_identity: Mapping[str, Any],
    replayed_sources: Mapping[str, Any],
) -> dict[str, Any]:
    report_id = _require_source(
        report_identity,
        name="conditioning_gain_comparison",
    )
    interpretation = report.get("diagnostic_interpretation")
    sources = report.get("sources")
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_conditioning_gain_sweep_comparison"
        or report.get("status") != "completed"
        or report.get("claim_boundary") != GAIN_CLAIM_BOUNDARY
        or not isinstance(interpretation, Mapping)
        or interpretation.get("shared_inference_gain_recovery_supported") is not False
        or interpretation.get("gain_only_amplifies_without_semantic_recovery")
        is not True
        or interpretation.get("recommended_next_action")
        != "develop_matched_training_time_label_ranking_or_contrastive_denoising_loss"
        or not isinstance(sources, Mapping)
        or set(sources) != {"cofitok_k8", "dense_identity"}
        or set(replayed_sources) != set(sources)
    ):
        raise ValueError("conditioning-gain diagnostic differs")
    for name, descriptor in sources.items():
        if _identity(replayed_sources[name], label=f"replayed gain source {name}") != _identity(
            descriptor,
            label=f"gain source {name}",
        ):
            raise ValueError(f"conditioning-gain source changed: {name}")
    return {
        "report": report_id,
        "shared_inference_gain_recovery_supported": False,
        "gain_only_amplifies_without_semantic_recovery": True,
        "recommended_next_action": interpretation["recommended_next_action"],
        "replayed_sources": copy.deepcopy(dict(replayed_sources)),
    }


def validate_supersession_interlock(
    *,
    receipt: Mapping[str, Any],
    receipt_identity: Mapping[str, Any],
    marker: Mapping[str, Any],
    marker_identity: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_id = _require_source(
        receipt_identity,
        name="supersession_receipt",
    )
    marker_id = _require_source(marker_identity, name="supersession_marker")
    interlock = receipt.get("interlocks", {}).get("conditioning_ranking_v1")
    guarantees = receipt.get("guarantees")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("role") != "generation_terminal_route_supersession_interlock"
        or receipt.get("status") != "pass"
        or receipt.get("generation_advantage_proven") is not False
        or receipt.get("scope") != SUPERSESSION_SCOPE
        or not isinstance(interlock, Mapping)
        or interlock.get("active") is not True
        or interlock.get("legacy_output_root") != LEGACY_OUTPUT_ROOT
        or interlock.get("interlock_kind") != "directory_with_marker"
        or _identity(interlock.get("marker", {}), label="receipt marker") != marker_id
        or not isinstance(guarantees, Mapping)
        or guarantees.get("corrected_route_must_use_new_versioned_output_roots")
        is not True
        or guarantees.get("legacy_conditioning_child_launch_blocked_by_preexisting_lock_directory")
        is not True
        or guarantees.get("no_existing_process_was_signaled") is not True
        or guarantees.get("no_gpu_process_was_started") is not True
    ):
        raise ValueError("terminal-route supersession receipt differs")
    if (
        marker.get("schema_version") != 1
        or marker.get("role") != "generation_legacy_gpu_route_supersession_marker"
        or marker.get("status") != "active"
        or marker.get("route") != "conditioning_ranking_v1"
        or marker.get("legacy_output_root") != LEGACY_OUTPUT_ROOT
        or marker.get("reason")
        != "legacy_consumer_binds_stale_runtime_v1_terminal_guard"
        or marker.get("scope") != SUPERSESSION_SCOPE
        or marker.get("canonical_receipt") != receipt_id["path"]
    ):
        raise ValueError("legacy conditioning supersession marker differs")
    return {"receipt": receipt_id, "marker": marker_id, "legacy_route_blocked": True}


def validate_preparation_continuity(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    legacy_preparation: Mapping[str, Any],
    legacy_preparation_identity: Mapping[str, Any],
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
) -> dict[str, Any]:
    preparation_id = _identity(preparation_identity, label="rebind preparation")
    legacy_id = _require_source(
        legacy_preparation_identity,
        name="legacy_preparation",
    )
    if expected_output_root != OUTPUT_ROOT:
        raise ValueError("terminal-rebind output root differs")
    validate_conditioning_ranking_probe_preparation(
        preparation,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    validate_conditioning_ranking_probe_preparation(
        legacy_preparation,
        expected_revision=LEGACY_PREPARATION_GIT["revision"],
        expected_branch=LEGACY_PREPARATION_GIT["branch"],
        expected_output_root=LEGACY_OUTPUT_ROOT,
    )
    if (
        preparation.get("tree") != expected_tree
        or preparation.get("claim_boundary") != PREPARATION_CLAIM_BOUNDARY
        or _identity(
            preparation.get("legacy_candidate_source", {}),
            label="preparation legacy source",
        )
        != legacy_id
        or preparation.get("parameter_counts")
        != legacy_preparation.get("parameter_counts")
    ):
        raise ValueError("terminal-rebind preparation differs")
    current_configs = preparation.get("configs")
    legacy_configs = legacy_preparation.get("configs")
    if (
        not isinstance(current_configs, Mapping)
        or not isinstance(legacy_configs, Mapping)
        or set(current_configs) != set(legacy_configs)
    ):
        raise ValueError("terminal-rebind config set differs")
    for name in current_configs:
        if not _same_content(current_configs[name], legacy_configs[name]):
            raise ValueError(f"terminal-rebind config changed: {name}")
    return {
        "preparation": preparation_id,
        "legacy_preparation": legacy_id,
        "config_content_continuity_verified": True,
        "parameter_counts": copy.deepcopy(dict(preparation["parameter_counts"])),
    }


def build_terminal_rebind_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    legacy_preparation: Mapping[str, Any],
    legacy_preparation_identity: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    standing_authorization_identity: Mapping[str, Any],
    quality_result: Mapping[str, Any],
    quality_result_identity: Mapping[str, Any],
    post_decision: Mapping[str, Any],
    post_decision_identity: Mapping[str, Any],
    post_verification: Mapping[str, Any],
    post_verification_identity: Mapping[str, Any],
    epsilon_result: Mapping[str, Any],
    epsilon_result_identity: Mapping[str, Any],
    epsilon_recovery_status: Mapping[str, Any],
    epsilon_recovery_status_identity: Mapping[str, Any],
    gain_report: Mapping[str, Any],
    gain_report_identity: Mapping[str, Any],
    gain_replayed_sources: Mapping[str, Any],
    supersession_receipt: Mapping[str, Any],
    supersession_receipt_identity: Mapping[str, Any],
    supersession_marker: Mapping[str, Any],
    supersession_marker_identity: Mapping[str, Any],
    official_epsilon_replay: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_output_root: str,
    control_plane: Mapping[str, Any],
) -> dict[str, Any]:
    preparation_result = validate_preparation_continuity(
        preparation=preparation,
        preparation_identity=preparation_identity,
        legacy_preparation=legacy_preparation,
        legacy_preparation_identity=legacy_preparation_identity,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
        expected_output_root=expected_output_root,
    )
    standing_id = _require_source(
        standing_authorization_identity,
        name="standing_authorization",
    )
    validate_standing_experiment_authorization(standing_authorization)
    terminal = validate_post_reconciliation_sources(
        quality_result=quality_result,
        quality_identity=quality_result_identity,
        decision=post_decision,
        decision_identity=post_decision_identity,
        verification=post_verification,
        verification_identity=post_verification_identity,
    )
    epsilon = validate_epsilon_sampling_recovery(
        result=epsilon_result,
        result_identity=epsilon_result_identity,
        recovery_status=epsilon_recovery_status,
        recovery_status_identity=epsilon_recovery_status_identity,
        expected_decision=terminal["decision"],
        expected_verification=terminal["verification"],
        official_replay=official_epsilon_replay,
    )
    gain = validate_gain_diagnostic(
        report=gain_report,
        report_identity=gain_report_identity,
        replayed_sources=gain_replayed_sources,
    )
    supersession = validate_supersession_interlock(
        receipt=supersession_receipt,
        receipt_identity=supersession_receipt_identity,
        marker=supersession_marker,
        marker_identity=supersession_marker_identity,
    )
    git = _clean_git(authorization_git, label="terminal-rebind builder")
    if git != {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("terminal-rebind builder Git differs")
    runbook_id = _identity(runbook_identity, label="terminal-rebind runbook")
    if expected_output_root != OUTPUT_ROOT:
        raise ValueError("terminal-rebind output root differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": AUTHORIZATION_ROLE,
        "status": "authorized",
        "scope": AUTHORIZATION_SCOPE,
        "stage": STAGE,
        "authorized_git": git,
        "output_root": expected_output_root,
        "source_reports": {
            "preparation": preparation_result["preparation"],
            "legacy_preparation": preparation_result["legacy_preparation"],
            "standing_authorization": standing_id,
            "quality_bridge_result": terminal["quality_result"],
            "post_reconciliation_decision": terminal["decision"],
            "post_reconciliation_verification": terminal["verification"],
            "epsilon_stability_result": epsilon["result"],
            "epsilon_recovery_status": epsilon["recovery_status"],
            "conditioning_gain_comparison": gain["report"],
            "supersession_receipt": supersession["receipt"],
            "supersession_marker": supersession["marker"],
            "runbook": runbook_id,
        },
        "source_control": copy.deepcopy(dict(control_plane)),
        "scientific_route": {
            "terminal_status": "hold",
            "failure_classification": (
                "mixed_shared_absolute_quality_support_and_class_failure"
            ),
            "failed_checks": copy.deepcopy(list(EXPECTED_FAILED_CHECKS)),
            "sampling_recovery_completed_arms": epsilon["completed_arms"],
            "sampling_recovery_selection_status": epsilon["selection_status"],
            "shared_inference_gain_recovery_supported": gain[
                "shared_inference_gain_recovery_supported"
            ],
            "gain_only_amplifies_without_semantic_recovery": gain[
                "gain_only_amplifies_without_semantic_recovery"
            ],
            "recommended_hypothesis": gain["recommended_next_action"],
            "probe_interpretation": (
                "matched training-time semantic-alignment screening only"
            ),
            "absolute_fid_or_recall_recovery_tested": False,
            "generation_advantage_proven": False,
        },
        "continuity": {
            "legacy_candidate_config_content_verified": preparation_result[
                "config_content_continuity_verified"
            ],
            "legacy_v1_route_blocked": supersession["legacy_route_blocked"],
            "official_epsilon_result_replayed": True,
            "official_epsilon_replay": epsilon["official_replay"],
            "gain_sources_replayed": gain["replayed_sources"],
        },
        "execution_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "generation_advantage_proven": False,
    }


def validate_terminal_rebind_authorization(
    authorization: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    expected = build_terminal_rebind_authorization(**kwargs)
    if dict(authorization) != expected:
        raise ValueError("conditioning-ranking terminal-rebind authorization differs")
    return expected
