from __future__ import annotations

import copy

import pytest

from scripts.build_generation_conditioning_ranking_sampling_validation import (
    CLAIM_BOUNDARY,
    RUN_NAMES,
    build_sampling_validation,
    method_sampling_decision,
)
from scripts.evaluate_generation_conditioning_ranking_samples import (
    paired_class_fidelity_summary,
)


def _row(index: int, *, delta: float = 0.1) -> dict:
    control_log = -7.0
    ranked_log = control_log + delta
    return {
        "sample_index": index,
        "requested_class": index % 1000,
        "control": {
            "target_log_probability": control_log,
            "target_probability": 0.001,
            "predicted_class": index % 1000,
            "top1_correct": index % 4 == 0,
            "top5_correct": index % 2 == 0,
        },
        "ranked": {
            "target_log_probability": ranked_log,
            "target_probability": 0.0011,
            "predicted_class": index % 1000,
            "top1_correct": index % 4 == 0,
            "top5_correct": index % 2 == 0,
        },
        "ranked_minus_control_target_log_probability": delta,
    }


def _identity(name: str, character: str = "a") -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 123,
        "sha256": character * 64,
    }


def _git() -> dict:
    return {
        "revision": "b" * 40,
        "branch": "scale/generation-conditioning-ranking-sampling5k-v1",
        "tracked_dirty": False,
    }


def _inputs() -> dict:
    paired_metrics = paired_class_fidelity_summary(
        [_row(index) for index in range(8)]
    )
    generation = {
        "control_cofitok": {"fid": 100.0},
        "ranked_cofitok": {"fid": 105.0},
        "control_dense_identity": {"fid": 90.0},
        "ranked_dense_identity": {"fid": 95.0},
    }
    return {
        "postevaluation": {
            "decision": {
                "method_passes": {
                    "cofitok": True,
                    "dense_identity": True,
                },
                "shared_semantic_alignment_recovery_supported": True,
                "cofitok_specific_advantage_claim_allowed": False,
                "recommended_next_action": (
                    "consider_separately_authorized_matched_sampling_validation"
                ),
            }
        },
        "postevaluation_identity": _identity("postevaluation"),
        "checkpoint_evidence": {run: {"step": 1000} for run in RUN_NAMES},
        "sampling_provenance": {run: {"run": run} for run in RUN_NAMES},
        "sampling_sources": {
            run: {"sampling_report": _identity(f"{run}-sampling")}
            for run in RUN_NAMES
        },
        "generation_reports": generation,
        "generation_sources": {
            run: _identity(f"{run}-metrics") for run in RUN_NAMES
        },
        "paired_reports": {
            "cofitok": {"metrics": copy.deepcopy(paired_metrics)},
            "dense_identity": {"metrics": copy.deepcopy(paired_metrics)},
        },
        "paired_sources": {
            "cofitok": _identity("cofitok-paired"),
            "dense_identity": _identity("dense-paired"),
        },
        "git": _git(),
        "output_root": (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "conditioning_ranking_four_arm_sampling5k_v1"
        ),
    }


def test_shared_sampling_recovery_remains_non_authorizing() -> None:
    report = build_sampling_validation(**_inputs())

    assert report["decision"] == {
        "method_passes": {"cofitok": True, "dense_identity": True},
        "shared_generated_class_alignment_recovery_supported": True,
        "cofitok_specific_advantage_claim_allowed": False,
        "recommended_next_action": (
            "prepare_separately_bound_matched_5k_training_recipe_confirmation"
        ),
    }
    assert report["methods"]["cofitok"]["gates"]["fid_within_tolerance"] is True
    assert report["methods"]["dense_identity"]["pass"] is True
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert all(
        report["claim_boundary"][name] is False
        for name in (
            "authorizes_training",
            "authorizes_sampling",
            "authorizes_checkpoint_promotion",
            "authorizes_full_training",
            "authorizes_full_100k_or_300k",
            "authorizes_release",
            "broad_generation_superiority_claim_allowed",
            "cofitok_specific_advantage_claim_allowed",
        )
    )


def test_fid_regression_creates_method_asymmetry_without_advantage_claim() -> None:
    inputs = _inputs()
    inputs["generation_reports"]["ranked_cofitok"]["fid"] = 111.0

    report = build_sampling_validation(**inputs)

    assert report["methods"]["cofitok"]["gates"]["fid_within_tolerance"] is False
    assert report["methods"]["cofitok"]["pass"] is False
    assert report["methods"]["dense_identity"]["pass"] is True
    assert report["decision"] == {
        "method_passes": {"cofitok": False, "dense_identity": True},
        "shared_generated_class_alignment_recovery_supported": False,
        "cofitok_specific_advantage_claim_allowed": False,
        "recommended_next_action": "reject_shared_repair_due_method_asymmetry",
    }


def test_shared_class_failure_recommends_objective_revision() -> None:
    inputs = _inputs()
    failing = paired_class_fidelity_summary(
        [_row(index, delta=0.001) for index in range(8)]
    )
    inputs["paired_reports"] = {
        "cofitok": {"metrics": copy.deepcopy(failing)},
        "dense_identity": {"metrics": copy.deepcopy(failing)},
    }

    report = build_sampling_validation(**inputs)

    assert report["decision"]["method_passes"] == {
        "cofitok": False,
        "dense_identity": False,
    }
    assert report["decision"]["recommended_next_action"] == (
        "revise_training_time_semantic_alignment_objective"
    )


def test_method_decision_recomputes_exact_gate_set_and_fid_domain() -> None:
    metrics = paired_class_fidelity_summary([_row(index) for index in range(8)])
    drifted = copy.deepcopy(metrics)
    drifted["gates"]["unexpected"] = True
    with pytest.raises(ValueError, match="gates differ"):
        method_sampling_decision(
            method="cofitok",
            paired_metrics=drifted,
            control_generation={"fid": 100.0},
            ranked_generation={"fid": 100.0},
        )

    with pytest.raises(ValueError, match="outside its domain"):
        method_sampling_decision(
            method="cofitok",
            paired_metrics=metrics,
            control_generation={"fid": -1.0},
            ranked_generation={"fid": 1.0},
        )

    malformed = copy.deepcopy(metrics)
    malformed["ranked_minus_control"] = None
    with pytest.raises(ValueError, match="ranked_minus_control metrics are missing"):
        method_sampling_decision(
            method="cofitok",
            paired_metrics=malformed,
            control_generation={"fid": 1.0},
            ranked_generation={"fid": 1.0},
        )


def test_zero_fid_pair_has_a_finite_identity_ratio() -> None:
    metrics = paired_class_fidelity_summary([_row(index) for index in range(8)])
    decision = method_sampling_decision(
        method="cofitok",
        paired_metrics=metrics,
        control_generation={"fid": 0.0},
        ranked_generation={"fid": 0.0},
    )

    assert decision["ranked_minus_control"]["fid_ratio"] == 1.0
    assert decision["gates"]["fid_within_tolerance"] is True
