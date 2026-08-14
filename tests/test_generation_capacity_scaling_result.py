from __future__ import annotations

import copy

import pytest

from cofitok.generation.capacity_probe import CAPACITY_PROBE_PARAMETER_COUNTS
from cofitok.generation.capacity_scaling_result import (
    CAPACITY_COMPLETION_MECHANISM_HOLD_ID,
    CAPACITY_COMPLETION_PREPARATION_ID,
    CAPACITY_COMPLETION_QUALITY_HOLD_ID,
    CAPACITY_SCALING_RESULT_BOUNDARY,
    build_capacity_scaling_50k_result,
    validate_capacity_scaling_50k_result,
)
from cofitok.generation.capacity_scaling_execution import (
    build_capacity_scaling_launch_receipt,
)
from cofitok.generation.capacity_scaling_training import (
    CAPACITY_SCALING_TRAINING_BOUNDARY,
    CAPACITY_SCALING_TRAINING_ROLE,
)
from test_generation_capacity_probe_execution import _identity
from test_generation_capacity_scaling_execution import (
    DECISION_GIT,
    EXECUTION_GIT,
    TRAINING_GIT,
    _inputs as launch_inputs,
)


RESULT_GIT = {
    "revision": "9" * 40,
    "tree": "8" * 40,
    "branch": "scale/generation-capacity-scaling-result-v1",
    "tracked_dirty": False,
}


def _target_checkpoint(method: str, source: dict) -> dict:
    character = "c" if method == "cofitok" else "d"
    path = source["checkpoint"]["path"].replace(
        "checkpoint_step_00010000.pt",
        "checkpoint_step_00050000.pt",
    )
    return {
        "path": path,
        "bytes": 4_100_000_000,
        "sha256": character * 64,
        "step": 50_000,
        "integrity_manifest": _identity(
            f"{path}.integrity.json",
            "1" if method == "cofitok" else "2",
        ),
    }


def _training(method: str, launch: dict) -> dict:
    source = launch["selection"]["resume_sources"][method]
    source_checkpoint = {
        **copy.deepcopy(source["checkpoint"]),
        "step": 10_000,
        "integrity_manifest": copy.deepcopy(
            source["checkpoint_integrity_manifest"]
        ),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "role": CAPACITY_SCALING_TRAINING_ROLE,
        "git": {
            "revision": TRAINING_GIT["revision"],
            "branch": TRAINING_GIT["branch"],
            "tracked_dirty": False,
        },
        "configured_steps": 100_000,
        "source_step": 10_000,
        "completed_steps": 50_000,
        "training_complete": False,
        "exact_resume": True,
        "effective_batch_size": 64,
        "images_seen": 3_200_000,
        "parameter_count": CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method],
        "last_resume_step": 10_000,
        "recovery_resume_used": False,
        "runtime_environment_sha256": "3" * 64,
        "dataset_identity_sha256": "4" * 64,
        "source_checkpoint": source_checkpoint,
        "checkpoint": _target_checkpoint(method, source),
        "authorization_boundary": copy.deepcopy(
            CAPACITY_SCALING_TRAINING_BOUNDARY
        ),
    }


def _milestone_row(method: str, training: dict, *, fid: float) -> dict:
    checkpoint = training["checkpoint"]
    return {
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_integrity_manifest": checkpoint["integrity_manifest"]["path"],
        "checkpoint_step": 50_000,
        "weights": "ema",
        "sample_set_sha256": ("5" if method == "cofitok" else "6") * 64,
        "selected_prefix_budget": 8 if method == "cofitok" else 1,
        "sample_count": 2_048,
        "fid": fid,
        "inception_score": 4.0,
        "endpoint_clean_mse": 0.1,
        "prefix_path_mse_auc": 0.2,
        "ordered_rank_by_path_auc": 1,
        "order_count": 6 if method == "cofitok" else 1,
        "zero_token_max_abs": 0.0,
        "shuffled_to_ordered_endpoint_ratio": 1.5,
    }


def _kwargs(
    *,
    cofitok_fid: float = 140.0,
    dense_fid: float = 145.0,
) -> dict:
    launch_source = launch_inputs()
    capacity_result = launch_source["capacity_probe_result"]
    decision = launch_source["decision"]
    launch = build_capacity_scaling_launch_receipt(**launch_source)
    training = {
        method: _training(method, launch)
        for method in ("cofitok", "dense_identity")
    }
    cofitok_eval_identity = _identity("/evidence/step50/cofitok_eval.json", "7")
    milestone_sources = {
        "cofitok_generation": _identity(
            "/evidence/step50/cofitok_generation.json", "8"
        ),
        "dense_generation": _identity(
            "/evidence/step50/dense_generation.json", "9"
        ),
        "cofitok_checkpoint_eval": cofitok_eval_identity,
        "dense_checkpoint_eval": _identity(
            "/evidence/step50/dense_eval.json", "a"
        ),
    }
    methods = {
        "cofitok": _milestone_row(
            "cofitok", training["cofitok"], fid=cofitok_fid
        ),
        "dense_identity": _milestone_row(
            "dense_identity", training["dense_identity"], fid=dense_fid
        ),
    }
    alerts = []
    if (cofitok_fid - dense_fid) / max(dense_fid, 1e-12) > 0.25:
        alerts.append("cofitok_fid_more_than_25pct_above_dense")
    milestone = {
        "schema_version": 2,
        "status": "completed",
        "role": "training_quality_trend_only",
        "source_profile": "capacity_scaling",
        "claim_policy": {"formal_generation_claim_allowed": False},
        "milestone_step": 50_000,
        "expected_samples": 2_048,
        "source_reports": milestone_sources,
        "methods": methods,
        "quality_alerts": alerts,
    }
    milestone_verification = {
        "status": "verified",
        "source_profile": "capacity_scaling",
        "source_reports": milestone_sources,
    }
    milestone_evidence = {
        "status": "verified",
        "source_profile": "capacity_scaling",
        "quality_alerts": alerts,
        "cofitok_fid": cofitok_fid,
        "dense_fid": dense_fid,
    }
    cofitok_checkpoint = training["cofitok"]["checkpoint"]
    cofitok_eval = {
        "schema_version": 2,
        "status": "completed",
        "role": "generation_checkpoint_evaluation_report",
        "git": {
            "revision": TRAINING_GIT["revision"],
            "branch": TRAINING_GIT["branch"],
            "tracked_dirty": False,
        },
        "request": {
            "num_images": 256,
            "timestep": 500,
            "random_orders": 4,
            "weights": "ema",
            "precision": "bf16",
        },
        "checkpoint": cofitok_checkpoint["path"],
        "checkpoint_sha256": cofitok_checkpoint["sha256"],
        "checkpoint_integrity_manifest": cofitok_checkpoint[
            "integrity_manifest"
        ]["path"],
        "checkpoint_step": 50_000,
        "weights": "ema",
        "config": {
            "model": {
                "token_count": 8,
                "token_spatial_strides": [16, 16, 8, 8, 4, 1, 1, 1],
            }
        },
        "metrics": {
            "evaluated_images": 256,
            "component_energy_ratio_per_sample_mean": [
                0.02,
                0.02,
                0.02,
                0.02,
                0.02,
                0.3,
                0.3,
                0.3,
            ],
        },
    }
    identities = {
        "capacity_probe_result": launch["source_reports"]["capacity_probe_result"],
        "capacity_scaling_decision": launch["source_reports"][
            "capacity_scaling_decision"
        ],
        "capacity_scaling_launch_receipt": _identity(
            "/evidence/capacity_scaling_launch.json", "b"
        ),
        "capacity_scaling_execution_status": _identity(
            "/evidence/capacity_scaling_execution.json", "c"
        ),
        "cofitok_training_validation": _identity(
            "/evidence/cofitok_training.json", "d"
        ),
        "dense_identity_training_validation": _identity(
            "/evidence/dense_training.json", "e"
        ),
        "milestone_50000": _identity("/evidence/milestone_50000.json", "f"),
        "cofitok_checkpoint_eval": cofitok_eval_identity,
    }
    execution = {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_full_data_capacity_scaling_50k_execution",
        "stage": "complete",
        "git": copy.deepcopy(EXECUTION_GIT),
        "launch_receipt": identities["capacity_scaling_launch_receipt"],
        "training_validations": {
            "cofitok": identities["cofitok_training_validation"],
            "dense_identity": identities["dense_identity_training_validation"],
        },
        "milestone": identities["milestone_50000"],
        "authorization_boundary": {
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_or_release_allowed": False,
        },
    }
    return {
        "capacity_probe_result": capacity_result,
        "capacity_scaling_decision": decision,
        "launch_receipt": launch,
        "execution_status": execution,
        "training_validations": training,
        "milestone_report": milestone,
        "milestone_source_verification": milestone_verification,
        "milestone_evidence": milestone_evidence,
        "cofitok_checkpoint_evaluation": cofitok_eval,
        "source_identities": identities,
        "result_builder_git": RESULT_GIT,
        "expected_capacity_revision": TRAINING_GIT["revision"],
        "expected_capacity_branch": TRAINING_GIT["branch"],
        "expected_decision_revision": DECISION_GIT["revision"],
        "expected_decision_branch": DECISION_GIT["branch"],
        "expected_execution_revision": EXECUTION_GIT["revision"],
        "expected_execution_tree": EXECUTION_GIT["tree"],
        "expected_execution_branch": EXECUTION_GIT["branch"],
        "expected_training_tree": TRAINING_GIT["tree"],
    }


def _validate(report: dict) -> dict:
    return validate_capacity_scaling_50k_result(
        report,
        expected_execution_revision=EXECUTION_GIT["revision"],
        expected_execution_tree=EXECUTION_GIT["tree"],
        expected_execution_branch=EXECUTION_GIT["branch"],
        expected_result_revision=RESULT_GIT["revision"],
        expected_result_tree=RESULT_GIT["tree"],
        expected_result_branch=RESULT_GIT["branch"],
    )


def test_shared_50k_improvement_supports_only_a_new_decision() -> None:
    report = build_capacity_scaling_50k_result(**_kwargs())
    assert report["decision"]["capacity_completion_supported"] is True
    assert report["decision"]["recommendation"]["id"] == (
        CAPACITY_COMPLETION_PREPARATION_ID
    )
    assert report["authorization_boundary"] == CAPACITY_SCALING_RESULT_BOUNDARY
    assert report["authorization_boundary"]["additional_training_allowed"] is False
    assert _validate(report)["capacity_completion_supported"] is True


def test_missing_shared_improvement_holds_capacity_completion() -> None:
    report = build_capacity_scaling_50k_result(
        **_kwargs(cofitok_fid=140.0, dense_fid=170.0)
    )
    assert report["decision"]["shared_strict_fid_improvement"] is False
    assert report["decision"]["capacity_completion_supported"] is False
    assert report["decision"]["recommendation"]["id"] == (
        CAPACITY_COMPLETION_QUALITY_HOLD_ID
    )


def test_coarse_token_failure_holds_for_mechanism_recovery() -> None:
    kwargs = _kwargs()
    kwargs["cofitok_checkpoint_evaluation"]["metrics"][
        "component_energy_ratio_per_sample_mean"
    ] = [0.005, 0.005, 0.005, 0.005, 0.005, 0.325, 0.325, 0.325]
    report = build_capacity_scaling_50k_result(**kwargs)
    assert report["decision"]["cofitok_mechanism_invariants_valid"] is False
    assert report["decision"]["recommendation"]["id"] == (
        CAPACITY_COMPLETION_MECHANISM_HOLD_ID
    )


def test_result_rejects_training_milestone_checkpoint_drift() -> None:
    kwargs = _kwargs()
    kwargs["training_validations"]["cofitok"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="milestone checkpoint differs"):
        build_capacity_scaling_50k_result(**kwargs)


def test_result_validator_rejects_authorization_and_metric_tampering() -> None:
    report = build_capacity_scaling_50k_result(**_kwargs())
    escalated = copy.deepcopy(report)
    escalated["authorization_boundary"]["additional_training_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        _validate(escalated)

    tampered = copy.deepcopy(report)
    tampered["evaluation"]["step_50000"]["methods"]["cofitok"]["fid"] -= 1.0
    with pytest.raises(ValueError, match="estimands differ"):
        _validate(tampered)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.build_generation_capacity_scaling_50k_result",
        "scripts.verify_generation_capacity_scaling_50k_result",
    ],
)
def test_capacity_scaling_result_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)
