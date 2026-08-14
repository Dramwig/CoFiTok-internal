from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.generation.capacity_probe_result import build_capacity_probe_result
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_DECISION_BOUNDARY,
    CAPACITY_SCALING_HOLD_ID,
    CAPACITY_SCALING_MECHANISM_HOLD_ID,
    CAPACITY_SCALING_RECOMMENDATION_ID,
    build_capacity_scaling_decision,
    validate_capacity_scaling_decision,
)
from cofitok.reporting import file_sha256
from scripts import build_generation_capacity_scaling_decision as builder
from test_generation_capacity_probe_execution import BRANCH, REVISION, _standing
from test_generation_capacity_probe_result import _kwargs


ROOT = Path(__file__).resolve().parents[1]
DECISION_GIT = {
    "revision": "d" * 40,
    "branch": "scale/generation-capacity-scaling-decision-v1",
    "tracked_dirty": False,
}


def _result(*, supported: bool) -> dict:
    return build_capacity_probe_result(
        **_kwargs(
            fids={
                "base128_cofitok": 180.0,
                "base128_dense_identity": 175.0,
                "base256_cofitok": 160.0,
                "base256_dense_identity": 165.0 if supported else 176.0,
            }
        )
    )


def _identity(path: str, character: str) -> dict[str, object]:
    return {"path": path, "bytes": 100, "sha256": character * 64}


def _decision(*, supported: bool) -> dict:
    return build_capacity_scaling_decision(
        capacity_probe_result=_result(supported=supported),
        capacity_probe_result_identity=_identity("/evidence/result.json", "e"),
        standing_authorization=_standing(),
        standing_authorization_identity=_identity(
            "/evidence/standing_authorization.json",
            "f",
        ),
        decision_git=DECISION_GIT,
        expected_capacity_revision=REVISION,
        expected_capacity_branch=BRANCH,
    )


def test_supported_capacity_authorizes_only_exact_10k_to_50k_resume() -> None:
    report = _decision(supported=True)
    assert report["recommended_next_stage"]["id"] == (
        CAPACITY_SCALING_RECOMMENDATION_ID
    )
    authorization = report["execution_authorization"]
    assert authorization["matched_250m_resume_allowed"] is True
    assert authorization["resume_from_step"] == 10_000
    assert authorization["stop_after_step"] == 50_000
    assert authorization["configured_100k_completion_allowed"] is False
    assert authorization["full_300k_launch_allowed"] is False
    assert report["authorization_boundary"] == CAPACITY_SCALING_DECISION_BOUNDARY
    evidence = validate_capacity_scaling_decision(
        report,
        expected_decision_revision=DECISION_GIT["revision"],
        expected_decision_branch=DECISION_GIT["branch"],
    )
    assert evidence["execution_authorized"] is True


def test_unsupported_capacity_holds_without_gpu_authorization() -> None:
    report = _decision(supported=False)
    assert report["recommended_next_stage"]["id"] == CAPACITY_SCALING_HOLD_ID
    authorization = report["execution_authorization"]
    assert authorization["matched_250m_resume_allowed"] is False
    assert authorization["gpu_execution_allowed"] is False
    assert authorization["training_launch_allowed"] is False


def test_shared_improvement_with_mechanism_failure_routes_to_mechanism_recovery() -> None:
    result = _result(supported=True)
    result["decision"]["cofitok_mechanism_invariants_valid"] = False
    result["decision"]["capacity_supported"] = False
    result["decision"]["recommendation"] = {
        "id": "hold_capacity_scaling_due_to_mechanism_failure",
        "category": "capacity_not_qualified",
        "execution_ready": False,
        "full_300k_launch_allowed": False,
        "reason": "mechanism failed",
    }
    report = build_capacity_scaling_decision(
        capacity_probe_result=result,
        capacity_probe_result_identity=_identity("/evidence/result.json", "e"),
        standing_authorization=_standing(),
        standing_authorization_identity=_identity("/evidence/standing.json", "f"),
        decision_git=DECISION_GIT,
        expected_capacity_revision=REVISION,
        expected_capacity_branch=BRANCH,
    )
    assert report["recommended_next_stage"]["id"] == (
        CAPACITY_SCALING_MECHANISM_HOLD_ID
    )
    assert report["execution_authorization"]["training_launch_allowed"] is False


def test_capacity_scaling_rejects_result_or_standing_authorization_drift() -> None:
    result = _result(supported=True)
    result["authorization_boundary"] = copy.deepcopy(
        result["authorization_boundary"]
    )
    result["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="canonical capacity result"):
        build_capacity_scaling_decision(
            capacity_probe_result=result,
            capacity_probe_result_identity=_identity("/evidence/result.json", "e"),
            standing_authorization=_standing(),
            standing_authorization_identity=_identity("/evidence/standing.json", "f"),
            decision_git=DECISION_GIT,
            expected_capacity_revision=REVISION,
            expected_capacity_branch=BRANCH,
        )

    standing = _standing()
    standing["status"] = "revoked"
    with pytest.raises(ValueError, match="standing experiment authorization"):
        build_capacity_scaling_decision(
            capacity_probe_result=_result(supported=True),
            capacity_probe_result_identity=_identity("/evidence/result.json", "e"),
            standing_authorization=standing,
            standing_authorization_identity=_identity("/evidence/standing.json", "f"),
            decision_git=DECISION_GIT,
            expected_capacity_revision=REVISION,
            expected_capacity_branch=BRANCH,
        )


def test_capacity_scaling_result_replay_rejects_nonreproducible_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result(supported=True)
    path = tmp_path / "capacity_probe_result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(builder, "build_from_args", lambda args: copy.deepcopy(result))
    actual, identity = builder.replay_capacity_probe_result(
        path,
        expected_sha256=file_sha256(path),
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    assert actual == result
    assert identity["sha256"] == file_sha256(path)

    changed = copy.deepcopy(result)
    changed["status"] = "tampered"
    monkeypatch.setattr(builder, "build_from_args", lambda args: changed)
    with pytest.raises(ValueError, match="not reproducible"):
        builder.replay_capacity_probe_result(
            path,
            expected_sha256=file_sha256(path),
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


@pytest.mark.parametrize(
    "entrypoint",
    (
        "scripts/build_generation_capacity_scaling_decision.py",
        "scripts/verify_generation_capacity_scaling_decision.py",
    ),
)
def test_capacity_scaling_entrypoints_import_under_runbook_pythonpath(
    entrypoint: str,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "capacity" in result.stdout.lower()
    assert "50k" in result.stdout.lower()
