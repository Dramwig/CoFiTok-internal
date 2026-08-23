from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import build_generation_runtime_strict_route_selection as builder
from scripts import wait_generation_runtime_strict_route_selection as waiter


def write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def identity(path: Path) -> dict[str, object]:
    return builder.file_identity(path)


def safe_boundary() -> dict[str, object]:
    return {
        "diagnostic_non_authorizing": True,
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
        "sampling_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
        "inference_export_authorization_allowed": False,
        "process_signals_allowed": False,
        "upstream_decisions_modified": False,
        "cross_tier_numeric_ranking_allowed": False,
        "broad_generation_superiority_claim_allowed": False,
        "sota_claim_allowed": False,
    }


def recommendation(failed: list[str]) -> dict[str, object]:
    common = {
        "execution_ready": False,
        "gpu_execution_allowed": False,
        "full_300k_launch_allowed": False,
        "release_authorization_allowed": False,
    }
    failed_set = set(failed)
    if failed_set and failed_set <= builder.MATCHED_QUALITY_CHECKS:
        return {
            **common,
            "id": "run_matched_factorization_quality_regression_probe",
            "category": "matched_quality_regression",
            "trigger": {"failed_checks": sorted(failed_set)},
        }
    if failed_set == {"class_fidelity"}:
        return {
            **common,
            "id": "run_class_conditioning_fidelity_diagnostic",
            "category": "class_conditioning_recovery",
            "trigger": {"failed_checks": ["class_fidelity"]},
        }
    return {
        **common,
        "id": "extend_followup_policy_before_execution",
        "category": "unclassified_fail_closed",
        "trigger": {"failed_checks": failed},
    }


def build_fixture(
    tmp_path: Path,
    *,
    failed: list[str],
) -> tuple[builder.RouteSourcePaths, dict[str, str]]:
    generation = tmp_path / "generation"
    quality = generation / "stability_full_data_100k_base128_quality_bridge_v1"
    factorization_marker = (
        generation
        / "stability_full_data_100k_factorization_quality_regression_v1.lock"
        / "supersession_marker.json"
    )
    conditioning_marker = (
        generation
        / "conditioning_ranking_four_arm_probe1k_v1.lock"
        / "supersession_marker.json"
    )
    random_marker = (
        generation / "stability_full_data_100k_random_token_semantic_visual_v1"
    )
    standing = tmp_path / "standing_authorization.json"
    paths = builder.RouteSourcePaths(
        quality_root=quality,
        standing_authorization=standing,
        factorization_marker=factorization_marker,
        conditioning_marker=conditioning_marker,
        random_token_marker=random_marker,
    )
    json_paths = paths.json_paths()

    write(
        standing,
        {
            "schema_version": 1,
            "role": "cofitok_standing_experiment_authorization_record",
            "status": "active",
            "preserved_safety_boundaries": {
                "exact_revision_stage_and_output_binding_required": True,
                "formal_remote_checkout_must_not_be_modified": True,
                "independent_clean_checkout_required": True,
                "locked_evidence_must_not_be_overwritten": True,
                "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing": True,
                "unrelated_project_processes_must_not_be_modified": True,
            },
        },
    )
    write(
        json_paths["pair_monitor"],
        {"status": "pass", "stage": "complete", "issues": []},
    )
    pair_identity = identity(json_paths["pair_monitor"])
    write(
        json_paths["runtime_guard"],
        {
            "schema_version": 1,
            "role": "generation_runtime_compute_claim_guard",
            "status": "pass",
            "decision": "runtime_cost_claims_observational_only",
            "sources": {"terminal_pair_monitor": pair_identity},
            "claim_policy": {
                "training_wall_clock_direct_comparison_allowed": False,
                "training_throughput_direct_comparison_allowed": False,
                "cost_efficiency_ranking_allowed": False,
                "quality_or_generation_advantage_claim_allowed": False,
            },
            "claim_boundary": {
                "diagnostic_non_authorizing": True,
                "gpu_execution_allowed": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "inference_export_authorization_allowed": False,
                "release_authorization_allowed": False,
                "process_signals_allowed": False,
                "quality_claim_allowed": False,
                "broad_generation_superiority_claim_allowed": False,
            },
        },
    )
    checks = [
        {"name": name, "passed": name not in failed}
        for name in sorted(builder.EXPECTED_CHECKS)
    ]
    failed_in_order = [row["name"] for row in checks if row["passed"] is False]
    write(
        json_paths["quality_bridge_result"],
        {
            "schema_version": 1,
            "role": builder.QUALITY_RESULT_ROLE,
            "status": "completed",
            "stage": "stability_quality_bridge",
            "git": {
                "revision": builder.TRAINING_GIT["revision"],
                "branch": builder.TRAINING_GIT["branch"],
                "tracked_dirty": False,
            },
            "authorization_boundary": builder.QUALITY_RESULT_BOUNDARY,
            "quality_screen": {
                "status": "hold" if failed_in_order else "pass",
                "non_authorizing": True,
                "failed_checks": failed_in_order,
                "checks": checks,
            },
        },
    )
    quality_identity = identity(json_paths["quality_bridge_result"])
    selected_recommendation = recommendation(failed_in_order)
    write(
        json_paths["followup_decision"],
        {
            "schema_version": 2,
            "role": builder.FOLLOWUP_ROLE,
            "status": "completed",
            "decision_builder_git": builder.EXPOSURE_GIT,
            "quality_bridge_execution_git": {
                "revision": builder.TRAINING_GIT["revision"],
                "branch": builder.TRAINING_GIT["branch"],
                "tracked_dirty": False,
            },
            "source_reports": {"quality_bridge_result": quality_identity},
            "source_replay": {
                "quality_bridge_result_rebuilt_byte_equivalent": True,
                "physical_checkpoint_sample_and_real_set_reverified": True,
                "milestone_source_reports_reverified": True,
            },
            "terminal_quality": {
                "status": "hold" if failed_in_order else "pass",
                "failed_checks": failed_in_order,
            },
            "training_exposure": {
                "terminal_result_content_bound": True,
                "terminal_checkpoint_binding_verified": True,
            },
            "recommended_next_stage": selected_recommendation,
            "authorization_boundary": builder.FOLLOWUP_BOUNDARY,
        },
    )
    decision_identity = identity(json_paths["followup_decision"])
    write(
        json_paths["exposure_waiter_status"],
        {
            "schema_version": 1,
            "role": builder.EXPOSURE_WAITER_ROLE,
            "status": "completed",
            "detail": "exposure_aware_followup_decision_verified",
            "git": builder.EXPOSURE_GIT,
            "decision_sha256": decision_identity["sha256"],
            "recommended_next_stage": {
                "id": selected_recommendation["id"],
                "category": selected_recommendation["category"],
                "execution_ready": False,
            },
            "scope": {
                "gpu_use_allowed": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "evaluation_launch_allowed": False,
                "promotion_allowed": False,
                "release_allowed": False,
                "full_300k_launch_allowed": False,
                "process_signals_allowed": False,
            },
        },
    )
    write(
        json_paths["exposure_deployment_receipt"],
        {
            "schema_version": 1,
            "role": "generation_quality_bridge_exposure_aware_followup_waiter_deployment",
            "status": "pass",
        },
    )
    terminal_status = "hold" if failed_in_order else "pass"
    terminal_decision = builder.TERMINAL_DECISIONS[terminal_status]
    runtime_identity = identity(json_paths["runtime_guard"])
    write(
        json_paths["terminal_guard"],
        {
            "schema_version": 1,
            "role": builder.TERMINAL_GUARD_ROLE,
            "status": terminal_status,
            "decision": terminal_decision,
            "scope": {
                "training_git": {
                    "revision": builder.TRAINING_GIT["revision"],
                    "branch": builder.TRAINING_GIT["branch"],
                    "tracked_dirty": False,
                }
            },
            "sources": {
                "quality_bridge_result": quality_identity,
                "runtime_compute_claim_guard": runtime_identity,
            },
            "evidence": {
                "quality_screen": {
                    "status": terminal_status,
                    "failed_checks": failed_in_order,
                }
            },
            "claim_policy": {
                "larger_training_launch_allowed": False,
                "inference_export_authorization_allowed": False,
                "release_authorization_allowed": False,
                "cross_tier_numeric_ranking_allowed": False,
                "broad_generation_superiority_claim_allowed": False,
                "sota_claim_allowed": False,
            },
            "claim_boundary": {
                "diagnostic_non_authorizing": True,
                "training_launch_allowed": False,
                "gpu_execution_allowed": False,
                "sampling_launch_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "inference_export_authorization_allowed": False,
                "release_authorization_allowed": False,
                "process_signals_allowed": False,
            },
        },
    )
    guard_identity = identity(json_paths["terminal_guard"])
    write(
        json_paths["terminal_guard_status"],
        {
            "schema_version": 1,
            "role": builder.TERMINAL_GUARD_WAITER_ROLE,
            "status": "completed",
            "detail": "terminal_system_claim_guard_source_revalidated",
            "git": builder.TERMINAL_GUARD_GIT,
            "guard": guard_identity,
            "guard_status": terminal_status,
            "guard_decision": terminal_decision,
            "authorization_boundary": {
                "full_300k_launch_allowed": False,
                "gpu_execution_allowed": False,
                "inference_export_authorization_allowed": False,
                "process_signals_allowed": False,
                "promotion_or_release_allowed": False,
                "sampling_launch_allowed": False,
                "training_launch_allowed": False,
                "upstream_decisions_modified": False,
            },
        },
    )
    for name, role in (
        ("comparison_deployment_receipt", "comparison_deployment"),
        ("completion_deployment_receipt", "completion_deployment"),
        ("conjunct_deployment_receipt", "conjunct_deployment"),
    ):
        write(json_paths[name], {"schema_version": 1, "role": role, "status": "pass"})
    comparison_deployment = identity(json_paths["comparison_deployment_receipt"])
    write(
        json_paths["comparison"],
        {
            "schema_version": 1,
            "role": builder.COMPARISON_ROLE,
            "status": terminal_status,
            "decision": terminal_decision,
            "source_reports": {
                "terminal_system_claim_guard": guard_identity,
                "quality_bridge_result": quality_identity,
            },
            "comparison_policy": {
                "primary_direct_tier": "matched_training_direct",
                "external_context_tier": "official_pretrained_contextual",
                "cross_tier_numeric_ranking_allowed": False,
                "compute_matched_claim_allowed": False,
                "broad_generation_superiority_claim_allowed": False,
                "sota_claim_allowed": False,
            },
            "authorization_boundary": safe_boundary(),
            "matched_training_rows": [
                {"comparison_tier": "matched_training_direct"},
                {"comparison_tier": "matched_training_direct"},
            ],
            "official_context_rows": [
                {"comparison_tier": "official_pretrained_contextual"},
                {"comparison_tier": "official_pretrained_contextual"},
                {"comparison_tier": "official_pretrained_contextual"},
            ],
        },
    )
    comparison_identity = identity(json_paths["comparison"])
    write(
        json_paths["comparison_status"],
        {
            "schema_version": 1,
            "role": builder.COMPARISON_WAITER_ROLE,
            "status": "pass",
            "terminal_status": terminal_status,
            "expected": {"project_git": builder.COMPARISON_GIT},
            "deployment_receipt": comparison_deployment,
            "comparison": {"json": comparison_identity},
            "scope": {
                "cpu_only": True,
                "diagnostic_non_authorizing": True,
                "gpu_execution_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "inference_export_authorization_allowed": False,
                "process_signals_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "sampling_launch_allowed": False,
                "training_launch_allowed": False,
            },
        },
    )
    completion_deployment = identity(json_paths["completion_deployment_receipt"])
    write(
        json_paths["completion"],
        {
            "schema_version": 1,
            "role": builder.COMPLETION_ROLE,
            "status": "pass",
            "terminal_status": terminal_status,
            "terminal_decision": terminal_decision,
            "generation_advantage_proven": terminal_status == "pass",
            "scope": {"training_git": builder.TRAINING_GIT},
            "sources": {
                "terminal_system_claim_guard": guard_identity,
                "quality_bridge_comparison": comparison_identity,
                "quality_bridge_result": quality_identity,
            },
            "claim_policy": {
                "full_300k_launch_allowed": False,
                "promotion_or_release_allowed": False,
            },
            "authorization_boundary": safe_boundary(),
        },
    )
    completion_identity = identity(json_paths["completion"])
    write(
        json_paths["completion_status"],
        {
            "schema_version": 1,
            "role": builder.COMPLETION_WAITER_ROLE,
            "status": "pass",
            "detail": "terminal_completion_audit_revalidated",
            "terminal_status": terminal_status,
            "generation_advantage_proven": terminal_status == "pass",
            "git": builder.COMPLETION_GIT,
            "deployment_receipt": completion_deployment,
            "audit": {"identity": completion_identity},
            "scope": {
                "cpu_only_evidence_replay": True,
                "diagnostic_non_authorizing": True,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "inference_export_authorization_allowed": False,
                "process_signals_allowed": False,
                "upstream_decisions_modified": False,
            },
        },
    )
    completion_status_identity = identity(json_paths["completion_status"])
    conjunct_deployment = identity(json_paths["conjunct_deployment_receipt"])
    write(
        json_paths["conjunct"],
        {
            "schema_version": 1,
            "role": builder.CONJUNCT_ROLE,
            "status": "pass",
            "terminal_status": terminal_status,
            "terminal_decision": terminal_decision,
            "generation_advantage_proven": terminal_status == "pass",
            "training_identity": {
                "revision": builder.TRAINING_GIT["revision"],
                "branch": builder.TRAINING_GIT["branch"],
            },
            "sources": {
                "terminal_completion_audit": completion_identity,
                "terminal_completion_waiter_status": completion_status_identity,
            },
            "claim_policy": {
                "runtime_strict_comparator_passed": True,
                "strict_recovery_binding_verified": True,
                "training_wall_clock_direct_comparison_allowed": False,
                "training_throughput_direct_comparison_allowed": False,
                "cost_efficiency_ranking_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_or_release_allowed": False,
            },
            "authorization_boundary": {
                **safe_boundary(),
                "runtime_direct_ranking_allowed": False,
            },
        },
    )
    conjunct_identity = identity(json_paths["conjunct"])
    write(
        json_paths["conjunct_status"],
        {
            "schema_version": 1,
            "role": builder.CONJUNCT_WAITER_ROLE,
            "status": "pass",
            "git": builder.CONJUNCT_GIT,
            "deployment_receipt": conjunct_deployment,
            "conjunct": {
                "identity": conjunct_identity,
                "terminal_status": terminal_status,
                "generation_advantage_proven": terminal_status == "pass",
            },
            "generation_advantage_proven": terminal_status == "pass",
            "scope": {
                "cpu_only": True,
                "diagnostic_non_authorizing": True,
                "gpu_execution_allowed": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "inference_export_authorization_allowed": False,
                "training_process_signals_allowed": False,
                "unrelated_process_signals_allowed": False,
                "source_waiter_signals_allowed": False,
                "upstream_decisions_modified": False,
            },
        },
    )
    write(
        factorization_marker, {"role": "supersession_marker", "route": "factorization"}
    )
    write(conditioning_marker, {"role": "supersession_marker", "route": "conditioning"})
    write(random_marker, {"role": "supersession_marker", "route": "random_token"})
    write(
        json_paths["interlock_receipt"],
        {
            "schema_version": 1,
            "role": builder.INTERLOCK_ROLE,
            "status": "pass",
            "decision": "legacy_v1_gpu_route_consumers_superseded_fail_closed",
            "generation_advantage_proven": False,
            "scope": {
                "cpu_only": True,
                "diagnostic_non_authorizing": True,
                "static_filesystem_interlock_only": True,
                "gpu_execution_allowed": False,
                "new_gpu_supervisor_launch_allowed": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "inference_export_authorization_allowed": False,
                "process_signals_allowed": False,
                "upstream_evidence_modified": False,
            },
        },
    )
    if os.name == "posix":
        factorization_marker.parent.chmod(0o555)
        conditioning_marker.parent.chmod(0o555)
        factorization_marker.chmod(0o444)
        conditioning_marker.chmod(0o444)
        random_marker.chmod(0o444)
        json_paths["interlock_receipt"].chmod(0o444)
    hashes = {
        name: str(identity(path)["sha256"])
        for name, path in json_paths.items()
        if name in waiter.HASH_ARGUMENTS
    }
    return paths, hashes


@pytest.mark.parametrize(
    ("failed", "expected_route"),
    [
        (
            ["matched_fid_tolerance"],
            "factorization_quality_regression_diagnostic_runtime_strict",
        ),
        (["class_fidelity"], "conditioning_fidelity_diagnostic_runtime_strict"),
        (
            ["cofitok_absolute_fid", "class_fidelity"],
            "no_gpu_route",
        ),
    ],
)
def test_builder_selects_only_supported_exact_routes(
    tmp_path: Path,
    failed: list[str],
    expected_route: str,
) -> None:
    paths, hashes = build_fixture(tmp_path, failed=failed)
    report = builder.build_route_selection(
        paths=paths,
        expected_hashes=hashes,
        selector_git={"revision": "a" * 40},
        selector_sources={
            "waiter": {"path": "/waiter", "bytes": 1, "sha256": "b" * 64},
            "builder": {"path": "/builder", "bytes": 1, "sha256": "c" * 64},
        },
    )
    assert report["status"] == "pass"
    assert report["selection"]["id"] == expected_route
    assert report["authorization_boundary"]["route_execution_authorized"] is False
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_builder_rejects_unsafe_followup_boundary(tmp_path: Path) -> None:
    paths, hashes = build_fixture(tmp_path, failed=["matched_fid_tolerance"])
    decision_path = paths.json_paths()["followup_decision"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["authorization_boundary"]["recommended_stage_execution_allowed"] = True
    write(decision_path, decision)
    with pytest.raises(ValueError, match="follow-up authorization boundary differs"):
        builder.build_route_selection(
            paths=paths,
            expected_hashes={
                **hashes,
                "followup_decision": str(identity(decision_path)["sha256"]),
            },
            selector_git={},
            selector_sources={},
        )


def test_builder_rejects_stale_pair_monitor_binding(tmp_path: Path) -> None:
    paths, hashes = build_fixture(tmp_path, failed=["class_fidelity"])
    runtime_path = paths.json_paths()["runtime_guard"]
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["sources"]["terminal_pair_monitor"]["sha256"] = "0" * 64
    write(runtime_path, runtime)
    hashes["runtime_guard"] = str(identity(runtime_path)["sha256"])
    with pytest.raises(ValueError, match="binds another pair monitor"):
        builder.build_route_selection(
            paths=paths,
            expected_hashes=hashes,
            selector_git={},
            selector_sources={},
        )


def test_observer_fails_closed_on_upstream_failure(tmp_path: Path) -> None:
    paths, _ = build_fixture(tmp_path, failed=["class_fidelity"])
    status_path = paths.json_paths()["completion_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "failed"
    write(status_path, status)
    observation = waiter.observe_upstreams({"json_paths": paths.json_paths()})
    assert observation["state"] == "failed"
    assert observation["detail"] == "completion_waiter_failed"


def test_waiter_and_receipt_are_permanently_non_authorizing() -> None:
    assert waiter.SCOPE["cpu_only"] is True
    assert waiter.SCOPE["permanently_non_authorizing"] is True
    assert waiter.SCOPE["gpu_execution_allowed"] is False
    assert waiter.SCOPE["process_signals_allowed"] is False
    assert builder.AUTHORIZATION_BOUNDARY["route_execution_authorized"] is False
    assert builder.AUTHORIZATION_BOUNDARY["corrected_executor_launch_allowed"] is False
    assert builder.AUTHORIZATION_BOUNDARY["full_300k_launch_allowed"] is False
