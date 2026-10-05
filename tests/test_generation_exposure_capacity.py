from __future__ import annotations

from copy import deepcopy

import pytest

from cofitok.generation.exposure_capacity import (
    PREPARATION_BOUNDARY,
    build_preparation,
    validate_preparation,
)


def _identity(index: int) -> dict[str, object]:
    return {
        "path": f"/evidence/source_{index}.json",
        "bytes": 100 + index,
        "sha256": f"{index + 1:064x}",
    }


def _objective() -> dict[str, object]:
    return {
        "schema": "cofitok_generation_training_objective_reassessment_v1",
        "status": "completed",
        "decision": "preserve_qualified_objective_defer_new_intervention",
        "generation_advantage_proven": False,
        "execution_ready": False,
        "next_evidence": {"id": "prepare_source_bound_exposure_or_capacity_gate"},
        "authorization_boundary": deepcopy(PREPARATION_BOUNDARY),
    }


def _quality() -> dict[str, object]:
    return {
        "status": "completed",
        "terminal": {"status": "hold"},
        "quality_screen": {"status": "hold", "non_authorizing": True},
        "authorization_boundary": {key: False for key in PREPARATION_BOUNDARY},
    }


def _cross_protocol() -> dict[str, object]:
    return {"status": "completed", "operational_status": "pass", "terminal_status": "hold"}


def _post_reconciliation() -> dict[str, object]:
    return {
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": {key: False for key in PREPARATION_BOUNDARY},
    }


def _sampling() -> dict[str, object]:
    return {
        "status": "completed",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_sampling_recovery_candidate",
        "generation_advantage_proven": False,
    }


def _min_snr() -> dict[str, object]:
    return {
        "status": "completed",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_min_snr_candidate_at_50k",
        "generation_advantage_proven": False,
    }


def _pair_monitor() -> dict[str, object]:
    last_metric = {"step": 100_000, "samples_seen": 6_400_000}
    return {
        "status": "pass",
        "stage": "complete",
        "issues": [],
        "runs": {
            method: {
                "complete": True,
                "last_step": 100_000,
                "expected_steps": 100_000,
                "last_metric": dict(last_metric),
            }
            for method in ("cofitok", "dense_identity")
        },
    }


def _metrics(value: float) -> dict[str, object]:
    return {"step": 100_000, "samples_seen": 6_400_000, "validation_epsilon_mse": value}


def _training(
    revision: str = "a" * 40, *, method: str = "cofitok"
) -> dict[str, object]:
    run_name = (
        "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
        if method == "cofitok"
        else "dense_rollout_x0_u2_ema_teacher"
    )
    return {
        "training_complete": True,
        "completed_steps": 100_000,
        "target_steps": 100_000,
        "config": {
            "model": {
                "synthesis_mode": "fixed_basis" if method == "cofitok" else "dense_identity",
                "token_count": 8 if method == "cofitok" else 1,
                "predictor_use_feedback": method == "cofitok",
            }
        },
        "git": {"revision": revision, "branch": "scale/generation-stability-quality-bridge-100k", "dirty": False},
        "output_dir": f"/evidence/{run_name}",
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00100000.pt",
            "checkpoint_bytes": 1000 if method == "cofitok" else 1001,
            "checkpoint_sha256": ("1" if method == "cofitok" else "2") * 64,
            "integrity_manifest": "checkpoint_step_00100000.pt.integrity.json",
            "step": 100_000,
        },
    }


def _report() -> dict[str, object]:
    names = (
        "objective_reassessment",
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
    return build_preparation(
        objective_reassessment=_objective(),
        quality_bridge_result=_quality(),
        post_reconciliation_decision=_post_reconciliation(),
        cross_protocol_reconciliation=_cross_protocol(),
        sampling_recovery_result=_sampling(),
        min_snr_result=_min_snr(),
        pair_monitor=_pair_monitor(),
        cofitok_metrics=_metrics(0.0292560),
        dense_metrics=_metrics(0.0292407),
        cofitok_training_report=_training(method="cofitok"),
        dense_training_report=_training(method="dense_identity"),
        source_identities={name: _identity(index) for index, name in enumerate(names)},
        builder_git={
            "revision": "b" * 40,
            "tree": "c" * 40,
            "branch": "scale/generation-stability-quality-bridge-100k",
            "tracked_dirty": True,
        },
        exposure_output_root="/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure_capacity_disambiguation_v1/exposure",
        capacity_output_root="/root/autodl-tmp/CoFiTok/checkpoints/generation/exposure_capacity_disambiguation_v1/capacity",
    )


def test_preparation_is_replayable_and_non_authorizing() -> None:
    report = _report()
    validated = validate_preparation(report)
    assert validated["status"] == "pass"
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY
    assert report["candidate_arms"]["exposure_continuation"]["controlled_change"] == "training_exposure_only"
    assert report["candidate_arms"]["capacity_qualification"]["controlled_change"] == "model_capacity_only"


def test_preparation_rejects_terminal_hold_replacement() -> None:
    report = _report()
    report["authorization_boundary"]["terminal_hold_replacement_allowed"] = True
    with pytest.raises(ValueError, match="authorization boundary is not canonical"):
        validate_preparation(report)


def test_preparation_rejects_explicit_quality_advantage_claim() -> None:
    with pytest.raises(ValueError, match="quality bridge must retain"):
        quality = _quality()
        quality["generation_advantage_proven"] = True
        build_preparation(
            objective_reassessment=_objective(),
            quality_bridge_result=quality,
            post_reconciliation_decision=_post_reconciliation(),
            cross_protocol_reconciliation=_cross_protocol(),
            sampling_recovery_result=_sampling(),
            min_snr_result=_min_snr(),
            pair_monitor=_pair_monitor(),
            cofitok_metrics=_metrics(0.0292560),
            dense_metrics=_metrics(0.0292407),
            cofitok_training_report=_training(),
            dense_training_report=_training(),
            source_identities={name: _identity(index) for index, name in enumerate((
                "objective_reassessment",
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
            ))},
            builder_git={"revision": "c" * 40, "branch": "test"},
            exposure_output_root="/root/autodl-tmp/CoFiTok/checkpoints/generation/new/exposure",
            capacity_output_root="/root/autodl-tmp/CoFiTok/checkpoints/generation/new/capacity",
        )


def test_preparation_rejects_objective_change_in_candidate() -> None:
    report = _report()
    report["candidate_arms"]["capacity_qualification"]["objective_change_allowed"] = True
    with pytest.raises(ValueError, match="objective change"):
        validate_preparation(report)


def test_preparation_rejects_mismatched_training_revisions() -> None:
    with pytest.raises(ValueError, match="one Git revision"):
        build_preparation(
            objective_reassessment=_objective(),
            quality_bridge_result=_quality(),
            post_reconciliation_decision=_post_reconciliation(),
            cross_protocol_reconciliation=_cross_protocol(),
            sampling_recovery_result=_sampling(),
            min_snr_result=_min_snr(),
            pair_monitor=_pair_monitor(),
            cofitok_metrics=_metrics(0.0292560),
            dense_metrics=_metrics(0.0292407),
            cofitok_training_report=_training("a" * 40),
            dense_training_report=_training("b" * 40),
            source_identities={name: _identity(index) for index, name in enumerate((
                "objective_reassessment",
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
            ))},
            builder_git={"revision": "c" * 40, "branch": "test"},
            exposure_output_root="/root/autodl-tmp/CoFiTok/checkpoints/generation/new/exposure",
            capacity_output_root="/root/autodl-tmp/CoFiTok/checkpoints/generation/new/capacity",
        )


def test_preparation_rejects_locked_quality_bridge_output_root() -> None:
    with pytest.raises(ValueError, match="must not reuse"):
        report = _report()
        report["candidate_arms"]["exposure_continuation"]["output_root"] = (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "stability_full_data_100k_base128_quality_bridge_v1"
        )
        validate_preparation(report)


def test_preparation_rejects_unbound_objective_reassessment() -> None:
    report = _report()
    report["sources"].pop("objective_reassessment")
    with pytest.raises(ValueError, match="source identities differ"):
        validate_preparation(report)
