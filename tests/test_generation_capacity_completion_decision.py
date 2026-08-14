from __future__ import annotations

import copy

import pytest

from cofitok.generation.capacity_completion_decision import (
    CAPACITY_COMPLETION_DECISION_BOUNDARY,
    CAPACITY_COMPLETION_RECOMMENDATION_ID,
    build_capacity_completion_decision,
    validate_capacity_completion_decision,
)
from cofitok.generation.capacity_scaling_result import (
    CAPACITY_COMPLETION_MECHANISM_HOLD_ID,
    CAPACITY_COMPLETION_QUALITY_HOLD_ID,
    build_capacity_scaling_50k_result,
)
from test_generation_capacity_probe_execution import _identity, _standing
from test_generation_capacity_scaling_execution import EXECUTION_GIT, TRAINING_GIT
from test_generation_capacity_scaling_result import RESULT_GIT, _kwargs


DECISION_GIT = {
    "revision": "7" * 40,
    "tree": "6" * 40,
    "branch": "scale/generation-capacity-completion-decision-v1",
    "tracked_dirty": False,
}


def _decision_kwargs(*, result: dict | None = None) -> dict:
    if result is None:
        result = build_capacity_scaling_50k_result(**_kwargs())
    return {
        "capacity_scaling_result": result,
        "capacity_scaling_result_identity": _identity(
            "/evidence/capacity_scaling_50k_result.json", "1"
        ),
        "standing_authorization": _standing(),
        "standing_authorization_identity": _identity(
            "/evidence/standing_authorization.json", "2"
        ),
        "decision_git": DECISION_GIT,
        "expected_execution_revision": EXECUTION_GIT["revision"],
        "expected_execution_tree": EXECUTION_GIT["tree"],
        "expected_execution_branch": EXECUTION_GIT["branch"],
        "expected_result_revision": RESULT_GIT["revision"],
        "expected_result_tree": RESULT_GIT["tree"],
        "expected_result_branch": RESULT_GIT["branch"],
        "expected_training_revision": TRAINING_GIT["revision"],
        "expected_training_tree": TRAINING_GIT["tree"],
        "expected_training_branch": TRAINING_GIT["branch"],
    }


def _validate(report: dict) -> dict:
    return validate_capacity_completion_decision(
        report,
        expected_decision_revision=DECISION_GIT["revision"],
        expected_decision_tree=DECISION_GIT["tree"],
        expected_decision_branch=DECISION_GIT["branch"],
    )


def test_supported_50k_result_authorizes_only_exact_50k_to_100k_completion() -> None:
    report = build_capacity_completion_decision(**_decision_kwargs())
    authorization = report["execution_authorization"]
    assert report["recommended_next_stage"]["id"] == (
        CAPACITY_COMPLETION_RECOMMENDATION_ID
    )
    assert authorization["matched_250m_resume_allowed"] is True
    assert authorization["resume_from_step"] == 50_000
    assert authorization["stop_after_step"] == 100_000
    assert authorization["terminal_evaluation"]["sample_count_per_method"] == 10_000
    assert authorization["terminal_evaluation"]["precision_recall_required"] is True
    assert report["authorization_boundary"] == CAPACITY_COMPLETION_DECISION_BOUNDARY
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert _validate(report)["execution_authorized"] is True


def test_failed_shared_improvement_produces_non_authorizing_quality_hold() -> None:
    result = build_capacity_scaling_50k_result(
        **_kwargs(cofitok_fid=140.0, dense_fid=170.0)
    )
    report = build_capacity_completion_decision(
        **_decision_kwargs(result=result)
    )
    assert report["execution_authorization"]["matched_250m_resume_allowed"] is False
    assert report["recommended_next_stage"]["id"] == (
        CAPACITY_COMPLETION_QUALITY_HOLD_ID
    )
    assert _validate(report)["execution_authorized"] is False


def test_failed_coarse_token_utilization_produces_mechanism_hold() -> None:
    kwargs = _kwargs()
    kwargs["cofitok_checkpoint_evaluation"]["metrics"][
        "component_energy_ratio_per_sample_mean"
    ] = [0.005, 0.005, 0.005, 0.005, 0.005, 0.325, 0.325, 0.325]
    result = build_capacity_scaling_50k_result(**kwargs)
    report = build_capacity_completion_decision(
        **_decision_kwargs(result=result)
    )
    assert report["recommended_next_stage"]["id"] == (
        CAPACITY_COMPLETION_MECHANISM_HOLD_ID
    )
    assert report["execution_authorization"]["training_launch_allowed"] is False


def test_completion_decision_rejects_result_or_training_git_drift() -> None:
    kwargs = _decision_kwargs()
    kwargs["capacity_scaling_result"]["training_git"]["tree"] = "0" * 40
    with pytest.raises(ValueError, match="training Git identity differs"):
        build_capacity_completion_decision(**kwargs)

    kwargs = _decision_kwargs()
    kwargs["capacity_scaling_result"]["authorization_boundary"][
        "additional_training_allowed"
    ] = True
    with pytest.raises(ValueError, match="contract differs"):
        build_capacity_completion_decision(**kwargs)


def test_completion_decision_validator_rejects_scope_escalation() -> None:
    report = build_capacity_completion_decision(**_decision_kwargs())
    escalated = copy.deepcopy(report)
    escalated["execution_authorization"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="execution scope differs"):
        _validate(escalated)

    drifted = copy.deepcopy(report)
    drifted["selection"]["resume_from_step"] = 49_999
    with pytest.raises(ValueError, match="selection differs"):
        _validate(drifted)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.build_generation_capacity_completion_decision",
        "scripts.verify_generation_capacity_completion_decision",
    ],
)
def test_capacity_completion_decision_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)
