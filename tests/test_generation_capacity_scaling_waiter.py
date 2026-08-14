from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_RECOMMENDATION_ID,
)
from cofitok.reporting import file_sha256
from scripts import wait_for_generation_capacity_scaling_decision as waiter


ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        project=ROOT,
        capacity_probe_result=tmp_path / "capacity_probe_result.json",
        standing_authorization=tmp_path / "standing_authorization.json",
        expected_standing_authorization_sha256="a" * 64,
        decision=tmp_path / "capacity_scaling_decision.json",
        status=tmp_path / "waiter_status.json",
        expected_self_revision="d" * 40,
        expected_self_tree="e" * 40,
        expected_self_branch="scale/capacity-decision",
        expected_capacity_revision="c" * 40,
        expected_capacity_branch="scale/capacity-probe",
        poll_seconds=1,
    )


def _decision(*, authorized: bool) -> dict:
    return {
        "recommended_next_stage": {
            "id": (
                CAPACITY_SCALING_RECOMMENDATION_ID
                if authorized
                else "hold_250m_capacity_scaling_and_prepare_recipe_intervention"
            ),
            "execution_ready": authorized,
        },
        "execution_authorization": {
            "matched_250m_resume_allowed": authorized,
        },
    }


@pytest.mark.parametrize(
    ("authorized", "expected_status"),
    ((True, "completed"), (False, "not_selected")),
)
def test_waiter_builds_once_and_records_selected_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorized: bool,
    expected_status: str,
) -> None:
    args = _args(tmp_path)
    result = {"status": "completed"}
    args.capacity_probe_result.write_text(json.dumps(result), encoding="utf-8")
    args.standing_authorization.write_text("{}", encoding="utf-8")
    args.expected_standing_authorization_sha256 = file_sha256(
        args.standing_authorization
    )
    expected = _decision(authorized=authorized)
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(
        waiter,
        "_git_identity",
        lambda project: {
            "revision": args.expected_self_revision,
            "tree": args.expected_self_tree,
            "branch": args.expected_self_branch,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        waiter,
        "build_from_sources",
        lambda **kwargs: copy.deepcopy(expected),
    )
    monkeypatch.setattr(
        waiter,
        "validate_capacity_scaling_decision",
        lambda *values, **kwargs: {},
    )
    waiter.main()
    actual = json.loads(args.decision.read_text(encoding="utf-8"))
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert actual == expected
    assert status["status"] == expected_status
    assert status["recommended_next_stage"]["id"] == (
        expected["recommended_next_stage"]["id"]
    )
    assert status["authorization_boundary"]["training_launch_allowed"] is False


def test_waiter_refuses_existing_nonreproducible_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.capacity_probe_result.write_text("{}", encoding="utf-8")
    args.standing_authorization.write_text("{}", encoding="utf-8")
    args.expected_standing_authorization_sha256 = file_sha256(
        args.standing_authorization
    )
    args.decision.write_text(json.dumps({"different": True}), encoding="utf-8")
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(
        waiter,
        "_git_identity",
        lambda project: {
            "revision": args.expected_self_revision,
            "tree": args.expected_self_tree,
            "branch": args.expected_self_branch,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        waiter,
        "build_from_sources",
        lambda **kwargs: _decision(authorized=True),
    )
    with pytest.raises(ValueError, match="not reproducible"):
        waiter.main()


def test_capacity_scaling_waiter_runbook_is_cpu_only() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/generation_capacity_scaling_decision_after_probe.sh"
    ).read_text(encoding="utf-8")
    assert "wait_for_generation_capacity_scaling_decision.py" in source
    assert "train_generation.py" not in source
    assert "nvidia-smi" not in source
    assert "full_matched_300k" not in source
