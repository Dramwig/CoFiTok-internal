from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from cofitok.generation.quality_repair_decision import (
    POST_DIAGNOSTIC_DECISION_SCHEMA,
    SOURCE_NAMES,
    build_epsilon_stability_post_diagnostic_decision,
    validate_epsilon_stability_post_diagnostic_decision,
)
from cofitok.generation.quality_repair_result import (
    EPSILON_STABILITY_RESULT_SCHEMA,
)


REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
BRANCH = "scale/generation-stability-quality-bridge-100k"


def _identity(name: str) -> dict[str, Any]:
    character = hex((sum(name.encode("utf-8")) % 15) + 1)[2:]
    return {
        "path": f"/evidence/{name}.json",
        "bytes": len(name) + 1,
        "sha256": character * 64,
    }


def _training_git() -> dict[str, Any]:
    return {"revision": REVISION, "branch": BRANCH, "tracked_dirty": False}


def _false_boundary(*fields: str) -> dict[str, bool]:
    return {field: False for field in fields}


def _sources() -> dict[str, Any]:
    cases = [f"case_{index}" for index in range(8)]
    diagnostic = {
        "schema": EPSILON_STABILITY_RESULT_SCHEMA,
        "status": "pass",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_sampling_recovery_candidate",
        "selected_case_id": None,
        "generation_advantage_proven": False,
        "claim_boundary": {
            "one_thousand_sample_screening_only": True,
            "min_snr_training_tested": False,
            "cofitok_generation_advantage_claim_allowed": False,
        },
        "authorization_boundary": _false_boundary(
            "matched_1000_sample_sampling_launch_allowed",
            "independent_matched_10000_confirmation_launch_allowed",
            "training_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "inference_export_allowed",
            "release_allowed",
            "process_signals_allowed",
        ),
        "observations": [
            {"case_id": case, "method": method, "status": "pass"}
            for case in cases
            for method in ("cofitok", "dense_identity")
        ],
    }
    post = {
        "schema_version": 1,
        "role": "generation_100k_post_reconciliation_experiment_decision",
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "training_git": _training_git(),
        "recommended_next_stage": {
            "id": "prepare_matched_100k_epsilon_stability_sampling_diagnostic",
            "execution_ready": False,
            "gpu_execution_allowed": False,
        },
        "authorization_boundary": {
            **_false_boundary(
                "training_launch_allowed",
                "sampling_launch_allowed",
                "full_training_launch_allowed",
                "full_300k_launch_allowed",
                "promotion_allowed",
                "export_allowed",
                "release_allowed",
                "process_signals_allowed",
            ),
            "new_source_bound_execution_gate_required": True,
        },
    }
    cross = {
        "schema_version": 1,
        "role": "generation_100k_cross_protocol_reconciliation",
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "comparison": {
            "sampler_step_changes_matched_ranking": True,
            "sample_count_changes_matched_ranking": False,
            "generation_advantage_proven": False,
        },
        "claim_boundary": {
            **_false_boundary(
                "training_launch_allowed",
                "sampling_launch_allowed",
                "full_training_launch_allowed",
                "full_300k_launch_allowed",
                "promotion_authorization_allowed",
                "export_authorization_allowed",
                "release_authorization_allowed",
                "process_signals_allowed",
            ),
            "generation_advantage_proven": False,
        },
    }
    pair = {
        "schema_version": 2,
        "status": "pass",
        "stage": "complete",
        "issues": [],
        "git": _training_git(),
        "runs": {"cofitok": {}, "dense_identity": {}},
    }
    exposure_rows = {
        method: {
            "status": "complete",
            "training_complete": True,
            "completed_steps": 100_000,
            "samples_seen": 6_400_000,
            "effective_batch_size": 64,
        }
        for method in ("cofitok", "dense_identity")
    }
    exposure = {
        "schema_version": 1,
        "role": "generation_training_exposure_audit",
        "status": "pass",
        "rows": exposure_rows,
        "comparison": {
            "same_completed_steps": True,
            "same_images_seen": True,
            "same_dataset_normalized_exposure": True,
        },
        "milestone_binding": {
            "expected_step": 100_000,
            "matched_training_exposure_verified": True,
            "training_git": _training_git(),
        },
        "quality_bridge_plan": {
            "training_exposure": {
                "bridge": {"equivalent_epochs": 4.995445558619602},
                "source": {"equivalent_epochs": 24.96859419012024},
            }
        },
    }
    runtime = {
        "schema_version": 1,
        "status": "pass",
        "observed_runtime_parity": {"status": "proven"},
        "methods": {
            "cofitok": {
                "resume_compute_adjustment_verification": {
                    "status": "verified",
                    "summary": {
                        "discovered_orphan_archive_count": 2,
                        "covered_orphan_archive_count": 2,
                        "event_count": 2,
                        "issues": [],
                    },
                }
            }
        },
        "claim_boundary": {"full_300k_launch_allowed": False},
    }
    terminal_guard = {
        "schema_version": 1,
        "role": "generation_terminal_system_claim_guard",
        "status": "hold",
        "decision": (
            "terminal_system_evidence_complete_without_qualified_matched_advantage"
        ),
        "claim_boundary": _false_boundary(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        ),
    }
    completion = {
        "schema_version": 1,
        "role": "generation_quality_bridge_terminal_completion_audit",
        "status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "detail": "terminal_quality_bridge_evidence_physically_replayed",
        "authorization_boundary": _false_boundary(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
        ),
    }
    supersession = {
        "schema_version": 1,
        "role": "generation_terminal_route_supersession_interlock",
        "status": "pass",
        "decision": "legacy_v1_gpu_route_consumers_superseded_fail_closed",
        "generation_advantage_proven": False,
        "scope": _false_boundary(
            "training_launch_allowed",
            "sampling_launch_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_authorization_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
            "process_signals_allowed",
            "gpu_execution_allowed",
        ),
    }
    return {
        "epsilon_stability_sampling_result": diagnostic,
        "post_reconciliation_decision": post,
        "cross_protocol_reconciliation": cross,
        "pair_monitor": pair,
        "training_exposure_report": exposure,
        "runtime_compute_fairness": runtime,
        "terminal_system_claim_guard": terminal_guard,
        "terminal_completion_audit": completion,
        "terminal_route_supersession_receipt": supersession,
        "source_identities": {name: _identity(name) for name in SOURCE_NAMES},
        "builder_git": {
            "revision": "1" * 40,
            "tree": "2" * 40,
            "branch": "analysis/post-diagnostic-test",
            "tracked_dirty": False,
        },
    }


def test_post_diagnostic_decision_recommends_only_fresh_matched_preparation() -> None:
    sources = _sources()
    decision = build_epsilon_stability_post_diagnostic_decision(**sources)

    assert decision["schema"] == POST_DIAGNOSTIC_DECISION_SCHEMA
    assert decision["status"] == "completed"
    assert decision["terminal_status"] == "hold"
    assert decision["generation_advantage_proven"] is False
    assert decision["decision"] == "prepare_fresh_matched_min_snr_training_pilot"
    stage = decision["recommended_next_stage"]
    assert stage["controlled_change"]["field"] == "loss.min_snr_gamma"
    assert stage["attribution_policy"]["may_be_described_as_exposure_only"] is False
    assert stage["preparation_requirements"]["changed_config_resume_allowed"] is False
    assert stage["execution_ready"] is False
    for field in (
        "training_launch_allowed",
        "sampling_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "inference_export_allowed",
        "release_allowed",
        "process_signals_allowed",
        "gpu_execution_allowed",
    ):
        assert decision["authorization_boundary"][field] is False


def test_post_diagnostic_decision_rejects_a_sampling_candidate() -> None:
    sources = _sources()
    diagnostic = sources["epsilon_stability_sampling_result"]
    diagnostic["selection_status"] = (
        "candidate_identified_for_separately_authorized_10k_confirmation"
    )
    diagnostic["selected_case_id"] = "case_1"

    with pytest.raises(ValueError, match="diagnostic outcome differs"):
        build_epsilon_stability_post_diagnostic_decision(**sources)


def test_post_diagnostic_decision_rejects_permission_drift() -> None:
    sources = _sources()
    sources["terminal_completion_audit"]["authorization_boundary"][
        "training_launch_allowed"
    ] = True

    with pytest.raises(ValueError, match="permits training_launch_allowed"):
        build_epsilon_stability_post_diagnostic_decision(**sources)


def test_post_diagnostic_decision_rejects_incomplete_pair() -> None:
    sources = _sources()
    sources["pair_monitor"]["issues"] = ["trainer_missing"]

    with pytest.raises(ValueError, match="clean completed pair"):
        build_epsilon_stability_post_diagnostic_decision(**sources)


def test_post_diagnostic_decision_replays_exactly() -> None:
    sources = _sources()
    decision = build_epsilon_stability_post_diagnostic_decision(**sources)

    assert (
        validate_epsilon_stability_post_diagnostic_decision(
            decision, **sources
        )
        == decision
    )
    drifted = deepcopy(decision)
    drifted["recommended_next_stage"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="does not replay exactly"):
        validate_epsilon_stability_post_diagnostic_decision(
            drifted, **sources
        )
