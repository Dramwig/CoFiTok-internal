from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import wait_for_generation_capacity_probe_preparation as waiter


def _args(tmp_path: Path, decision: Path) -> list[str]:
    values = {
        "base-cofitok-config": tmp_path / "base_cofitok.json",
        "base-dense-config": tmp_path / "base_dense.json",
        "capacity-cofitok-config": tmp_path / "capacity_cofitok.json",
        "capacity-dense-config": tmp_path / "capacity_dense.json",
        "cofitok-reference-receipt": tmp_path / "cofitok_reference.json",
        "dense-reference-receipt": tmp_path / "dense_reference.json",
        "expected-preparation-revision": "a" * 40,
        "expected-preparation-branch": "scale/capacity",
        "output-root": tmp_path / "output",
    }
    arguments = [
        "waiter",
        "--followup-decision",
        str(decision),
        "--preparation",
        str(tmp_path / "preparation.json"),
        "--status",
        str(tmp_path / "status.json"),
        "--expected-self-revision",
        "a" * 40,
        "--expected-self-tree",
        "b" * 40,
        "--expected-self-branch",
        "scale/capacity",
        "--poll-seconds",
        "1",
    ]
    for name, value in values.items():
        arguments.extend(("--build-argument", f"{name}={value}"))
    return arguments


def _decision(path: Path, recommendation: str) -> None:
    path.write_text(
        json.dumps(
            {"recommended_next_stage": {"id": recommendation}},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_waiter_builds_only_selected_capacity_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = tmp_path / "decision.json"
    _decision(decision, waiter.CAPACITY_RECOMMENDATION)
    monkeypatch.setattr(waiter.sys, "argv", _args(tmp_path, decision))
    built = {
        "status": "prepared",
        "authorization_boundary": {
            "capacity_probe_execution_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }
    monkeypatch.setattr(waiter, "build_from_paths", lambda args: built)

    waiter.main()

    preparation = json.loads(
        (tmp_path / "preparation.json").read_text(encoding="utf-8")
    )
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert preparation == built
    assert status["status"] == "completed"
    assert status["observed_recommendation"] == waiter.CAPACITY_RECOMMENDATION
    assert status["authorization_boundary"]["training_launch_allowed"] is False


def test_waiter_exits_without_preparation_for_another_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = tmp_path / "decision.json"
    _decision(decision, "run_matched_factorization_mechanism_recovery_probe")
    monkeypatch.setattr(waiter.sys, "argv", _args(tmp_path, decision))
    monkeypatch.setattr(
        waiter,
        "build_from_paths",
        lambda args: pytest.fail("non-capacity branch must not build a preparation"),
    )

    waiter.main()

    assert not (tmp_path / "preparation.json").exists()
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "not_selected"
    assert status["authorization_boundary"]["capacity_probe_execution_allowed"] is False


def test_waiter_replays_existing_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = tmp_path / "decision.json"
    _decision(decision, waiter.CAPACITY_RECOMMENDATION)
    arguments = _args(tmp_path, decision)
    monkeypatch.setattr(waiter.sys, "argv", arguments)
    expected = {"status": "prepared", "value": 1}
    (tmp_path / "preparation.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    monkeypatch.setattr(waiter, "build_from_paths", lambda args: expected)

    waiter.main()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"


def test_waiter_rejects_existing_nonreproducible_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = tmp_path / "decision.json"
    _decision(decision, waiter.CAPACITY_RECOMMENDATION)
    monkeypatch.setattr(waiter.sys, "argv", _args(tmp_path, decision))
    (tmp_path / "preparation.json").write_text(
        json.dumps({"status": "tampered"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        waiter,
        "build_from_paths",
        lambda args: {"status": "prepared"},
    )

    with pytest.raises(ValueError, match="not reproducible"):
        waiter.main()
