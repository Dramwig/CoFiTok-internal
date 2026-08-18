from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

from scripts import run_generation_terminal_system_claim_guard_waiter as waiter


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
    ).strip()


def _args(tmp_path: Path) -> Namespace:
    quality_root = tmp_path / "quality"
    statistical_root = tmp_path / "statistical"
    output_root = quality_root / "reports" / "terminal_system_claim_guard_v1"
    return Namespace(
        project=ROOT,
        quality_output_root=quality_root,
        quality_result=quality_root / "reports" / "quality_bridge_result.json",
        statistical_claim_guard=(
            statistical_root / "statistical_claim_language_guard.json"
        ),
        statistical_waiter_status=statistical_root / "waiter_status.json",
        visual_audit_waiter_status=(
            quality_root / "reports" / "requested_class_visual_audit_waiter_status.json"
        ),
        runtime_claim_guard=(
            quality_root
            / "reports"
            / "runtime_compute_claim_guard_v1"
            / "runtime_compute_claim_guard.json"
        ),
        runtime_waiter_status=(
            quality_root
            / "reports"
            / "runtime_compute_claim_guard_v1"
            / "waiter_status.json"
        ),
        output=output_root / "terminal_system_claim_guard.json",
        status_output=output_root / "waiter_status.json",
        lock=output_root / "waiter.lock",
        expected_revision=_git("rev-parse", "HEAD"),
        expected_tree=_git("rev-parse", "HEAD^{tree}"),
        expected_branch=_git("branch", "--show-current"),
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )


def test_waiter_builds_cpu_only_guard_after_all_sources_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    for path in (
        args.quality_result,
        args.statistical_claim_guard,
        args.runtime_claim_guard,
    ):
        _write(path, {"status": "completed"})
    _write(args.statistical_waiter_status, {"status": "completed"})
    _write(args.runtime_waiter_status, {"status": "pass", "phase": "completed"})
    _write(args.visual_audit_waiter_status, {"status": "completed"})
    monkeypatch.setattr(
        waiter.terminal_guard,
        "build_guard",
        lambda **_kwargs: {
            "schema_version": 1,
            "role": "generation_terminal_system_claim_guard",
            "status": "hold",
            "decision": "terminal_system_evidence_complete_without_qualified_matched_advantage",
        },
    )

    assert waiter.run(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    guard = json.loads(args.output.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["guard_status"] == "hold"
    assert status["authorization_boundary"] == waiter.AUTHORIZATION_BOUNDARY
    assert status["authorization_boundary"]["gpu_execution_allowed"] is False
    assert status["authorization_boundary"]["process_signals_allowed"] is False
    assert guard["status"] == "hold"


def test_waiter_fails_closed_when_an_upstream_waiter_fails(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _write(
        args.statistical_waiter_status,
        {"status": "failed", "detail": "source_contract_drift"},
    )

    assert waiter.run(args) == 1
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["detail"] == "upstream_terminal_evidence_failed"
    assert "source_contract_drift" in status["failures"][0]
    assert not args.output.exists()
