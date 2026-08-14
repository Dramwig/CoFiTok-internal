from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.generation.capacity_completion_result import (
    CAPACITY_COMPLETION_RESULT_BOUNDARY,
    CAPACITY_COMPLETION_RESULT_ROLE,
    CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION,
    CAPACITY_COMPLETION_RESULT_SOURCE_NAMES,
    _policy,
    validate_capacity_completion_100k_result,
)
from scripts.build_generation_capacity_completion_100k_result import source_paths


REVISION = "1" * 40
TREE = "2" * 40
BRANCH = "scale/generation-capacity-completion-100k-result-waiter-v1"
SHA256 = "3" * 64


def _recommendation(
    *,
    quality_status: str = "pass",
    failed_checks: list[str] | None = None,
    milestone_alerts: list[str] | None = None,
    shared_improvement: bool = True,
) -> dict[str, object]:
    return _policy(
        quality_status=quality_status,
        failed_checks=[] if failed_checks is None else failed_checks,
        milestone_alerts=[] if milestone_alerts is None else milestone_alerts,
        shared_strict_fid_improvement=shared_improvement,
    )


def _report() -> dict[str, object]:
    source_reports = {
        name: {
            "path": f"/tmp/{name}.json",
            "bytes": 1,
            "sha256": SHA256,
        }
        for name in CAPACITY_COMPLETION_RESULT_SOURCE_NAMES
    }
    recommendation = _recommendation()
    method = {
        "checkpoint_step": 100_000,
        "sample_count": 10_000,
        "weights": "ema",
        "checkpoint_sha256": SHA256,
        "sample_set_sha256": SHA256,
    }
    git = {
        "revision": REVISION,
        "tree": TREE,
        "branch": BRANCH,
        "tracked_dirty": False,
    }
    return {
        "schema_version": CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "role": CAPACITY_COMPLETION_RESULT_ROLE,
        "output_root": "/tmp/capacity-completion",
        "execution_git": copy.deepcopy(git),
        "training_git": copy.deepcopy(git),
        "result_builder_git": copy.deepcopy(git),
        "source_reports": source_reports,
        "milestones": {
            "trend_50000_to_100000": {
                "protocol_matched": True,
                "shared_strict_fid_improvement": True,
                "step_100000_quality_alerts": [],
            }
        },
        "terminal": {
            "methods": {
                "cofitok": copy.deepcopy(method),
                "dense_identity": copy.deepcopy(method),
            }
        },
        "quality_screen": {
            "status": "pass",
            "failed_checks": [],
            "checks": [],
        },
        "decision_support": {
            "terminal_quality_pass": True,
            "failed_checks": [],
            "step_50000_to_100000_shared_strict_fid_improvement": True,
            "step_100000_quality_alerts": [],
            "recommended_next_stage": recommendation,
        },
        "claim_policy": {
            "role": "non_claim_capacity_completion_terminal_diagnostic",
            "terminal_sample_count_per_method": 10_000,
            "milestone_sample_count_per_method": 2_048,
            "cross_protocol_numeric_ranking_allowed": False,
            "formal_generation_claim_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(
            CAPACITY_COMPLETION_RESULT_BOUNDARY
        ),
    }


def _validate(report: dict[str, object]) -> dict[str, object]:
    return validate_capacity_completion_100k_result(
        report,
        expected_execution_revision=REVISION,
        expected_execution_tree=TREE,
        expected_execution_branch=BRANCH,
        expected_training_revision=REVISION,
        expected_training_tree=TREE,
        expected_training_branch=BRANCH,
        expected_result_revision=REVISION,
        expected_result_tree=TREE,
        expected_result_branch=BRANCH,
    )


def test_capacity_completion_result_paths_cover_exact_source_contract(
    tmp_path: Path,
) -> None:
    paths = source_paths(tmp_path / "capacity")
    assert set(paths) == CAPACITY_COMPLETION_RESULT_SOURCE_NAMES - {
        "quality_bridge_preparation"
    }
    assert paths["milestone_100000"].as_posix().endswith(
        "reports/capacity_completion_100k/milestone_step_00100000.json"
    )
    assert paths["dense_identity_class_fidelity"].as_posix().endswith(
        "base256_dense_identity/terminal_100k/samples_10000_ddim100_cfg15/"
        "class_fidelity/class_fidelity_report.json"
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_id"),
    (
        ({}, "build_source_compatible_formal_quality_gate"),
        (
            {"milestone_alerts": ["fid_regression"]},
            "reconcile_50k_100k_milestone_terminal_evidence",
        ),
        (
            {
                "quality_status": "hold",
                "failed_checks": ["ordered_prefix_rank"],
            },
            "run_factorization_mechanism_recovery_diagnostic",
        ),
        (
            {
                "quality_status": "hold",
                "failed_checks": ["matched_fid_tolerance"],
            },
            "run_matched_factorization_quality_regression_probe",
        ),
        (
            {
                "quality_status": "hold",
                "failed_checks": ["class_fidelity"],
            },
            "run_class_conditioning_fidelity_diagnostic",
        ),
        (
            {
                "quality_status": "hold",
                "failed_checks": ["cofitok_absolute_fid"],
            },
            "build_source_compatible_250m_full_300k_readiness_decision",
        ),
        (
            {
                "quality_status": "hold",
                "failed_checks": ["cofitok_absolute_fid"],
                "shared_improvement": False,
            },
            "revisit_matched_training_objective_before_more_scale",
        ),
        (
            {"quality_status": "hold", "failed_checks": ["unknown"]},
            "extend_capacity_completion_policy_before_execution",
        ),
    ),
)
def test_capacity_completion_result_policy_is_fail_closed(
    kwargs: dict[str, object],
    expected_id: str,
) -> None:
    recommendation = _recommendation(**kwargs)
    assert recommendation["id"] == expected_id
    assert recommendation["execution_ready"] is False
    assert recommendation["gpu_execution_allowed"] is False
    assert recommendation["training_launch_allowed"] is False
    assert recommendation["full_300k_launch_allowed"] is False


def test_capacity_completion_result_validator_preserves_non_authorizing_boundary(
) -> None:
    report = _report()
    evidence = _validate(report)
    assert evidence["terminal_quality_pass"] is True
    assert evidence["authorization_boundary"]["full_300k_launch_allowed"] is False

    tampered = copy.deepcopy(report)
    tampered["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        _validate(tampered)


def test_capacity_completion_result_validator_recomputes_policy() -> None:
    report = _report()
    report["decision_support"]["recommended_next_stage"][
        "training_launch_allowed"
    ] = True
    with pytest.raises(ValueError, match="next-stage policy differs"):
        _validate(report)


def test_capacity_completion_result_entrypoints_import() -> None:
    for module in (
        "scripts.build_generation_capacity_completion_100k_result",
        "scripts.verify_generation_capacity_completion_100k_result",
        "scripts.wait_for_generation_capacity_completion_100k_result",
    ):
        __import__(module)
