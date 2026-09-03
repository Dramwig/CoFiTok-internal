from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.exposure_capacity_authorization import identity
from scripts import wait_for_generation_capacity_scaling_decision as waiter


ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        project=ROOT,
        capacity_confirmation_result=tmp_path / "capacity_confirmation_result.json",
        decision=tmp_path / "capacity_scaling_decision.json",
        status=tmp_path / "waiter_status.json",
        expected_self_revision="a" * 40,
        expected_self_tree="b" * 40,
        expected_self_branch="scale/capacity-decision",
        expected_confirmation_revision="c" * 40,
        expected_confirmation_tree="d" * 40,
        expected_confirmation_branch="scale/capacity-confirmation",
        poll_seconds=1,
    )


def _decision(*, selected: bool) -> dict[str, object]:
    return {
        "scientific_status": "scaling_preparation_selected" if selected else "hold",
        "next_stage": {
            "route": "capacity_scaling_preparation" if selected else "hold",
            "capacity_scaling_preparation_allowed": selected,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


@pytest.mark.parametrize(
    ("selected", "expected_status"),
    ((True, "completed"), (False, "not_selected")),
)
def test_waiter_builds_once_and_records_non_authorizing_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
    expected_status: str,
) -> None:
    args = _args(tmp_path)
    args.capacity_confirmation_result.write_text("{}", encoding="utf-8")
    expected = _decision(selected=selected)
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
    assert status["next_stage"] == expected["next_stage"]
    assert status["authorization_boundary"]["training_launch_allowed"] is False
    assert status["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_waiter_refuses_existing_nonreproducible_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    args.capacity_confirmation_result.write_text("{}", encoding="utf-8")
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
        lambda **kwargs: _decision(selected=True),
    )
    with pytest.raises(ValueError, match="not reproducible"):
        waiter.main()
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "not reproducible" in status["error"]


def test_capacity_scaling_waiter_runbook_is_cpu_only() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/generation_capacity_scaling_decision_after_confirmation.sh"
    ).read_text(encoding="utf-8")
    assert "capacity-confirmation-result" in source
    assert "wait_for_generation_capacity_scaling_decision.py" in source
    assert "capacity-probe-result" not in source
    assert "standing-authorization" not in source
    assert "train_generation.py" not in source
    assert "nvidia-smi" not in source
    assert "full_matched_300k" not in source


def test_waiter_status_binds_physical_confirmation_result(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.capacity_confirmation_result.write_text("{}", encoding="utf-8")
    result_id = identity(args.capacity_confirmation_result)
    waiter._status(
        args,
        state="waiting",
        detail="test",
        polls=1,
        self_git={
            "revision": args.expected_self_revision,
            "tree": args.expected_self_tree,
            "branch": args.expected_self_branch,
            "tracked_dirty": False,
        },
        result_identity=result_id,
        decision_identity=None,
        next_stage=None,
    )
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["capacity_confirmation_result"] == result_id
