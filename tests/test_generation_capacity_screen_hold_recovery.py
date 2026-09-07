from __future__ import annotations

import copy
from hashlib import sha256

import pytest

from cofitok.generation.capacity_screen import ARM_NAMES
from cofitok.generation.capacity_screen_hold_recovery import (
    AUTHORIZATION_BOUNDARY,
    CAPACITY_EXECUTION_GIT,
    CAPACITY_FAILED_CHECKS,
    EXPOSURE_EXECUTION_GIT,
    ROLLOUT_NAMES,
    build_hold_recovery_decision,
    build_hold_recovery_validation,
    validate_hold_recovery_decision,
    validate_hold_recovery_validation,
)


DECISION_GIT = {
    "revision": "d" * 40,
    "tree": "e" * 40,
    "branch": "analysis/capacity-hold-recovery",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": sha256(name.encode("utf-8")).hexdigest(),
    }


def _disabled() -> dict[str, bool]:
    return {
        "decision_is_execution_authorization": False,
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
        "sampling_launch_allowed": False,
        "evaluation_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "release_allowed": False,
        "process_signals_allowed": False,
    }


def _fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_hold_recovery.validate_exposure_result_receipt",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_hold_recovery.validate_exposure_decision_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_hold_recovery.validate_capacity_screen_result_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_hold_recovery.validate_capacity_screen_validation_receipt",
        lambda value, **_: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_hold_recovery.validate_capacity_screen_arm_validation",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_hold_recovery.validate_sampling_recovery_result",
        lambda *_, **__: {
            "status": "pass",
            "selection_status": "no_shared_sampling_recovery_candidate",
            "generation_advantage_proven": False,
            "case_count": 8,
            "candidate_count": 7,
            "physical_identity_count": 99,
        },
    )

    exposure_result_id = _identity("exposure_result")
    exposure_receipt_id = _identity("exposure_result_validation")
    exposure_decision_id = _identity("exposure_decision")
    exposure_decision_validation_id = _identity("exposure_decision_validation")
    capacity_result_id = _identity("capacity_result")
    capacity_receipt_id = _identity("capacity_result_validation")
    sampling_id = _identity("sampling_recovery")
    min_snr_id = _identity("min_snr_result")
    min_snr_guard_id = _identity("min_snr_guard")
    arm_ids = {arm: _identity(f"{arm}_validation") for arm in ARM_NAMES}
    rollout_ids = {name: _identity(f"{name}_rollout") for name in ROLLOUT_NAMES}

    exposure_result = {
        "execution_checkout": copy.deepcopy(EXPOSURE_EXECUTION_GIT),
        "rollout": {
            "cofitok": {"report": rollout_ids["exposure_cofitok"]},
            "dense_identity": {
                "report": rollout_ids["exposure_dense_identity"]
            },
        },
    }
    exposure_decision = {
        "decision": "capacity_screen",
        "source_evidence": {
            "exposure_result": exposure_result_id,
            "exposure_result_validation_receipt": exposure_receipt_id,
            "execution_git": copy.deepcopy(EXPOSURE_EXECUTION_GIT),
        },
    }
    exposure_decision_validation = {
        "schema_version": "cofitok_generation_exposure_capacity_decision_validation_v1",
        "role": "content_addressed_exposure_capacity_scientific_decision_validation",
        "status": "pass",
        "scientific_route": "capacity_screen",
        "generation_advantage_proven": False,
        "decision": exposure_decision_id,
        "verified_sources": {
            "exposure_result": exposure_result_id,
            "exposure_result_validation_receipt": exposure_receipt_id,
        },
        "authorization_boundary": _disabled(),
    }
    arms = {
        arm: {"rollout": {"report": rollout_ids[f"capacity_{arm}"]}}
        for arm in ARM_NAMES
    }
    capacity_result = {
        "result_git": copy.deepcopy(CAPACITY_EXECUTION_GIT),
        "scientific_status": "hold",
        "failed_checks": list(CAPACITY_FAILED_CHECKS),
        "next_stage": {"capacity_confirmation_preparation_allowed": False},
        "source_evidence": {"arm_validations": arm_ids},
        "validated_preparation": {
            "evaluation_contract": {"samples_per_arm": 1_000}
        },
        "arm_summaries": {
            arm: {"distribution": {"recall": 0.0}} for arm in ARM_NAMES
        },
        "checks": [
            {
                "name": "cofitok_precision_non_regression",
                "passed": False,
                "observed": -0.145,
                "threshold": -0.05,
            },
            {
                "name": "dense_identity_fid_improves_with_capacity",
                "passed": False,
                "observed": 0.0296,
                "threshold": 0.0,
            },
        ],
    }
    rollouts = {}
    for index, name in enumerate(ROLLOUT_NAMES):
        git = (
            EXPOSURE_EXECUTION_GIT
            if name.startswith("exposure_")
            else CAPACITY_EXECUTION_GIT
        )
        rollouts[name] = {
            "schema_version": 1,
            "status": "completed",
            "weights": "ema",
            "git": {
                "revision": git["revision"],
                "branch": git["branch"],
                "tracked_dirty": False,
            },
            "free_sampling_rollout": {
                "timesteps": [999],
                "steps": [
                    {
                        "timestep": 999,
                        "raw_x0_clip_fraction": 0.991 + index * 0.001,
                        "predicted_x0_rms": 0.99,
                    }
                ],
            },
        }
    min_snr_result = {
        "schema": "cofitok_matched_min_snr_pilot_result_v1",
        "role": "generation_matched_min_snr_training_pilot_result",
        "status": "completed",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_min_snr_candidate_at_50k",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": _disabled(),
    }
    min_snr_guard = {
        "schema": "cofitok_matched_min_snr_terminal_physical_guard_v1",
        "role": "generation_matched_min_snr_terminal_physical_guard",
        "status": "pass",
        "scientific_status": "physical_evidence_replayed_non_authorizing",
        "selection_status": "no_shared_min_snr_candidate_at_50k",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": _disabled(),
        "double_replay": {"identical": True, "pass_count": 2},
        "source_code_replay": {
            "identical": True,
            "physical_identities_verified": True,
            "pass_count": 3,
        },
        "evidence": {
            "result": min_snr_id,
            "selection_status": "no_shared_min_snr_candidate_at_50k",
            "all_checkpoint_payloads_physically_hashed": True,
            "all_checkpoint_payloads_rehashed_after_replay": True,
            "all_classifier_weights_physically_hashed": True,
            "all_sample_sets_physically_hashed": True,
            "all_source_reports_physically_hashed": True,
            "all_terminal_sources_rehashed_after_replay": True,
            "all_training_control_checkpoints_physically_bound": True,
            "result_logical_replay_exact": True,
        },
    }
    return {
        "exposure_result": exposure_result,
        "exposure_result_identity": exposure_result_id,
        "exposure_result_validation": {},
        "exposure_result_validation_identity": exposure_receipt_id,
        "exposure_decision": exposure_decision,
        "exposure_decision_identity": exposure_decision_id,
        "exposure_decision_validation": exposure_decision_validation,
        "exposure_decision_validation_identity": exposure_decision_validation_id,
        "capacity_result": capacity_result,
        "capacity_result_identity": capacity_result_id,
        "capacity_result_validation": {},
        "capacity_result_validation_identity": capacity_receipt_id,
        "capacity_arm_validations": arms,
        "capacity_arm_validation_identities": arm_ids,
        "sampling_recovery_result": {},
        "sampling_recovery_identity": sampling_id,
        "min_snr_result": min_snr_result,
        "min_snr_result_identity": min_snr_id,
        "min_snr_guard": min_snr_guard,
        "min_snr_guard_identity": min_snr_guard_id,
        "rollout_reports": rollouts,
        "rollout_identities": rollout_ids,
        "decision_git": DECISION_GIT,
        "allowed_sampling_output_prefix": "/evidence/",
    }


def test_builds_non_authorizing_hold_with_separated_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_hold_recovery_decision(**_fixture(monkeypatch))

    assert report["scientific_status"] == "hold"
    assert report["next_stage"]["selected_intervention"] is None
    assert report["next_stage"]["preparation_allowed"] is False
    assert report["evidence_interpretation"]["recall_power"][
        "may_upgrade_screen_to_pass"
    ] is False
    assert report["evidence_interpretation"]["independent_adverse_evidence"][
        "hold_remains_required_without_recall_checks"
    ] is True
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert not any(AUTHORIZATION_BOUNDARY.values())
    assert validate_hold_recovery_decision(report) == report


def test_rejects_rollout_identity_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _fixture(monkeypatch)
    inputs["rollout_identities"]["capacity_base256_cofitok"] = _identity(
        "different_rollout"
    )

    with pytest.raises(ValueError, match="identity differs"):
        build_hold_recovery_decision(**inputs)


def test_rejects_nonterminal_first_rollout_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    rollout = inputs["rollout_reports"]["exposure_cofitok"]
    rollout["free_sampling_rollout"]["timesteps"] = [998]
    rollout["free_sampling_rollout"]["steps"][0]["timestep"] = 998

    with pytest.raises(ValueError, match="protocol differs"):
        build_hold_recovery_decision(**inputs)


def test_rejects_min_snr_physical_replay_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    inputs["min_snr_guard"]["source_code_replay"][
        "physical_identities_verified"
    ] = False

    with pytest.raises(ValueError, match="physical replay guard differs"):
        build_hold_recovery_decision(**inputs)


def test_rejects_upstream_boundary_that_omits_a_core_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    del inputs["min_snr_result"]["authorization_boundary"][
        "training_launch_allowed"
    ]

    with pytest.raises(ValueError, match="omits disabled permissions"):
        build_hold_recovery_decision(**inputs)


def test_contract_rejects_permission_and_causal_claim_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_hold_recovery_decision(**_fixture(monkeypatch))
    permission = copy.deepcopy(report)
    permission["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_hold_recovery_decision(permission)

    causal = copy.deepcopy(report)
    causal["evidence_interpretation"]["terminal_start"][
        "causal_root_cause_is_proven"
    ] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_hold_recovery_decision(causal)


def test_validation_receipt_is_content_addressed_and_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_hold_recovery_decision(**_fixture(monkeypatch))
    receipt = build_hold_recovery_validation(
        decision=report,
        decision_identity=_identity("hold_decision"),
        validator_git=DECISION_GIT,
    )

    assert receipt["status"] == "pass"
    assert len(receipt["validation_basis_sha256"]) == 64
    assert receipt["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert (
        validate_hold_recovery_validation(
            receipt,
            decision=report,
            decision_identity=_identity("hold_decision"),
        )
        == receipt
    )


def test_validation_receipt_rejects_decision_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_hold_recovery_decision(**_fixture(monkeypatch))
    receipt = build_hold_recovery_validation(
        decision=report,
        decision_identity=_identity("hold_decision"),
        validator_git=DECISION_GIT,
    )

    with pytest.raises(ValueError, match="not reproducible"):
        validate_hold_recovery_validation(
            receipt,
            decision=report,
            decision_identity=_identity("changed_decision"),
        )
