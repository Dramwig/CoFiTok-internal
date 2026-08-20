from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
)
from cofitok.generation.terminal_distribution_support import (
    TERMINAL_DISTRIBUTION_SUPPORT_ROUTE,
)
from cofitok.inference_replay import file_identity
from scripts import wait_for_generation_terminal_distribution_support as waiter


ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path: Path, *, max_polls: int = 0) -> argparse.Namespace:
    script = Path(waiter.__file__).resolve()
    return argparse.Namespace(
        project=tmp_path / "project",
        quality_bridge_root=tmp_path / "quality",
        expected_revision="a" * 40,
        expected_tree="b" * 40,
        expected_branch="analysis/generation-terminal-distribution-support-v1",
        expected_self_sha256=waiter._sha256(script),
        expected_followup_revision="c" * 40,
        expected_followup_branch="analysis/generation-quality-bridge-exposure-routing-v1",
        python=tmp_path / "python",
        status=tmp_path / "status.json",
        log=tmp_path / "waiter.log",
        poll_seconds=1,
        max_polls=max_polls,
        nearest_chunk_size=16,
    )


def _decision(route: str, result_identity: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": FOLLOWUP_DECISION_SCHEMA_VERSION,
        "status": "completed",
        "role": FOLLOWUP_DECISION_ROLE,
        "source_reports": {"quality_bridge_result": result_identity},
        "recommended_next_stage": {
            "id": route,
            "category": "recipe_or_objective_intervention",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": FOLLOWUP_AUTHORIZATION_BOUNDARY,
    }


def _checkout(args: argparse.Namespace) -> dict[str, object]:
    return {
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
        "runbook_sha256": "d" * 64,
        "diagnostic_script_sha256": "e" * 64,
    }


def test_terminal_distribution_support_waiter_bounded_wait_is_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, max_polls=1)
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(waiter, "_validate_checkout", lambda unused: _checkout(args))
    monkeypatch.setattr(waiter.time, "sleep", lambda seconds: None)

    assert waiter.main() == 78
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["status"] == "stopped"
    assert status["scope"]["gpu_use_allowed"] is False
    assert status["scope"]["recipe_probe_launch_allowed"] is False
    assert status["scope"]["process_signals_allowed"] is False


def test_terminal_distribution_support_waiter_skips_unselected_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    reports = args.quality_bridge_root / "reports"
    reports.mkdir(parents=True)
    result = reports / waiter.RESULT_NAME
    result.write_text("{}", encoding="utf-8")
    decision = reports / waiter.DECISION_NAME
    decision.write_text(
        json.dumps(
            _decision(
                "prepare_matched_250m_capacity_qualification_probe",
                file_identity(result),
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(waiter, "_validate_checkout", lambda unused: _checkout(args))
    monkeypatch.setattr(
        waiter.subprocess,
        "run",
        lambda *unused, **kwargs: (_ for _ in ()).throw(
            AssertionError("unselected route must not execute a child")
        ),
    )

    assert waiter.main() == 0
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["detail"] == "terminal_distribution_support_route_not_selected"
    assert status["diagnostic_executed"] is False


def test_terminal_distribution_support_waiter_runs_only_selected_cpu_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    reports = args.quality_bridge_root / "reports"
    reports.mkdir(parents=True)
    result = reports / waiter.RESULT_NAME
    result.write_text("{}", encoding="utf-8")
    decision = reports / waiter.DECISION_NAME
    decision.write_text(
        json.dumps(
            _decision(TERMINAL_DISTRIBUTION_SUPPORT_ROUTE, file_identity(result))
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(waiter, "_validate_checkout", lambda unused: _checkout(args))
    monkeypatch.setattr(
        waiter,
        "_postcondition",
        lambda output, **kwargs: {
            "status": "verified",
            "route": TERMINAL_DISTRIBUTION_SUPPORT_ROUTE,
            "gpu_execution_allowed": False,
            "training_launch_allowed": False,
        },
    )

    def run_once(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert "--expected-quality-bridge-result-sha256" in command
        output = args.quality_bridge_root / waiter.OUTPUT_RELATIVE_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(waiter.subprocess, "run", run_once)

    assert waiter.main() == 0
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["detail"] == "terminal_distribution_support_diagnostic_verified"
    assert status["diagnostic_executed"] is True
    assert status["verification"]["training_launch_allowed"] is False


def test_terminal_distribution_support_waiter_rejects_selected_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    reports = args.quality_bridge_root / "reports"
    reports.mkdir(parents=True)
    result = reports / waiter.RESULT_NAME
    result.write_text("{}", encoding="utf-8")
    decision = reports / waiter.DECISION_NAME
    wrong_identity = file_identity(result)
    wrong_identity["sha256"] = "f" * 64
    decision.write_text(
        json.dumps(_decision(TERMINAL_DISTRIBUTION_SUPPORT_ROUTE, wrong_identity)),
        encoding="utf-8",
    )
    monkeypatch.setattr(waiter, "parse_args", lambda: args)
    monkeypatch.setattr(waiter, "_validate_checkout", lambda unused: _checkout(args))

    assert waiter.main() == 84
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["detail"] == "selected_route_quality_bridge_binding_failed"


def test_terminal_distribution_support_runbook_hides_cuda_and_lowers_priority() -> None:
    source = (
        ROOT
        / "artifacts"
        / "runbooks"
        / "generation_quality_bridge_terminal_distribution_support_waiter.sh"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in source
    assert "nice -n 10 ionice -c 3" in source
    assert "kill " not in source
    assert "scripts/wait_for_generation_terminal_distribution_support.py" in source
