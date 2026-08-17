from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import run_generation_quality_bridge_claim_qualification_waiter as waiter


CONTROL_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/quality-claim-waiter",
    "tracked_dirty": False,
}
TERMINAL_CONTROL_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "analysis/terminal-uncertainty",
    "tracked_dirty": False,
}
QUALITY_GIT = {
    "revision": "e" * 40,
    "branch": "scale/quality-bridge",
    "tracked_dirty": False,
}
EVALUATOR_GIT = {
    "revision": "f" * 40,
    "tree": "0" * 40,
    "branch": "",
    "tracked_dirty": False,
}


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _terminal_status(
    *,
    pid: int,
    output_root: Path,
    quality_identity: dict[str, Any],
    manifest_identity: dict[str, Any],
    uncertainty_identity: dict[str, Any],
    status: str = "pass",
) -> dict[str, Any]:
    return {
        "schema_version": waiter.terminal_waiter.WAITER_SCHEMA_VERSION,
        "role": waiter.terminal_waiter.WAITER_ROLE,
        "status": status,
        "detail": "matched_relative_generation_advantage_supported",
        "phase": "completed" if status in {"pass", "hold"} else "source",
        "pid": pid,
        "expected": {
            "output_root": output_root.resolve().as_posix(),
            "control_git": TERMINAL_CONTROL_GIT,
            "quality_git": QUALITY_GIT,
            "evaluator_git": EVALUATOR_GIT,
        },
        "quality_bridge": {"result": quality_identity},
        "execution_manifest": {"identity": manifest_identity},
        "audit": {"source": uncertainty_identity},
        "claim_boundary": waiter.terminal_waiter.CLAIM_BOUNDARY,
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def test_terminal_status_binds_exact_sources(tmp_path: Path) -> None:
    output_root = tmp_path / "terminal"
    identities = [
        _write(tmp_path / f"source-{index}.json", {"index": index})
        for index in range(3)
    ]
    report = _terminal_status(
        pid=123,
        output_root=output_root,
        quality_identity=identities[0],
        manifest_identity=identities[1],
        uncertainty_identity=identities[2],
    )

    validated = waiter.validate_terminal_status(
        report,
        expected_pid=123,
        expected_output_root=output_root.resolve(),
        expected_control_git=TERMINAL_CONTROL_GIT,
        expected_quality_git=QUALITY_GIT,
        expected_evaluator_git=EVALUATOR_GIT,
    )

    assert validated["terminal"] is True
    assert validated["status"] == "pass"


def test_terminal_status_rejects_authorizing_boundary(tmp_path: Path) -> None:
    output_root = tmp_path / "terminal"
    identities = [
        _write(tmp_path / f"source-{index}.json", {"index": index})
        for index in range(3)
    ]
    report = _terminal_status(
        pid=123,
        output_root=output_root,
        quality_identity=identities[0],
        manifest_identity=identities[1],
        uncertainty_identity=identities[2],
    )
    report["claim_boundary"] = dict(report["claim_boundary"])
    report["claim_boundary"]["full_300k_launch_allowed"] = True

    with pytest.raises(ValueError, match="contract differs"):
        waiter.validate_terminal_status(
            report,
            expected_pid=123,
            expected_output_root=output_root.resolve(),
            expected_control_git=TERMINAL_CONTROL_GIT,
            expected_quality_git=QUALITY_GIT,
            expected_evaluator_git=EVALUATOR_GIT,
        )


@pytest.mark.parametrize("scientific_status", ["pass", "hold"])
def test_waiter_builds_one_non_authorizing_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_status: str,
) -> None:
    project = (tmp_path / "project").resolve()
    terminal_root = (tmp_path / "terminal").resolve()
    output_root = (tmp_path / "claim").resolve()
    project.mkdir(parents=True)
    quality_path = tmp_path / "quality_bridge_result.json"
    manifest_path = terminal_root / "reports" / "execution_manifest.json"
    uncertainty_path = terminal_root / "terminal_100k" / "matched_uncertainty.json"
    quality_identity = _write(quality_path, {"quality": True})
    manifest_identity = _write(manifest_path, {"manifest": True})
    uncertainty_identity = _write(uncertainty_path, {"uncertainty": True})
    terminal_status_path = terminal_root / "reports" / "waiter_status.json"
    _write(
        terminal_status_path,
        _terminal_status(
            pid=999_999,
            output_root=terminal_root,
            quality_identity=quality_identity,
            manifest_identity=manifest_identity,
            uncertainty_identity=uncertainty_identity,
            status=scientific_status,
        ),
    )
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    monkeypatch.setattr(waiter, "_terminal_process_active", lambda **_kwargs: False)
    monkeypatch.setattr(
        waiter.qualification_builder,
        "build_qualification",
        lambda **_kwargs: {
            "schema_version": 1,
            "role": "test_qualification",
            "status": scientific_status,
            "decision": (
                "matched_quality_bridge_fid_advantage_statistically_qualified"
                if scientific_status == "pass"
                else "matched_quality_bridge_fid_advantage_not_statistically_qualified"
            ),
            "claim_policy": {
                "matched_relative_fid_advantage_claim_allowed": (
                    scientific_status == "pass"
                ),
                "full_300k_launch_allowed": False,
            },
        },
    )
    args = argparse.Namespace(
        project=project,
        terminal_output_root=terminal_root,
        terminal_status=terminal_status_path,
        terminal_pid_file=terminal_root / "reports" / "waiter.pid",
        quality_result=quality_path,
        execution_manifest=manifest_path,
        uncertainty_report=uncertainty_path,
        output_root=output_root,
        status_output=output_root / "reports" / "waiter_status.json",
        pid_file=output_root / "reports" / "waiter.pid",
        qualification_output=(
            output_root / "reports" / "statistical_claim_qualification.json"
        ),
        expected_terminal_pid=999_999,
        expected_control_revision=CONTROL_GIT["revision"],
        expected_control_tree=CONTROL_GIT["tree"],
        expected_control_branch=CONTROL_GIT["branch"],
        expected_terminal_control_revision=TERMINAL_CONTROL_GIT["revision"],
        expected_terminal_control_tree=TERMINAL_CONTROL_GIT["tree"],
        expected_terminal_control_branch=TERMINAL_CONTROL_GIT["branch"],
        expected_quality_revision=QUALITY_GIT["revision"],
        expected_quality_branch=QUALITY_GIT["branch"],
        expected_evaluator_revision=EVALUATOR_GIT["revision"],
        expected_evaluator_tree=EVALUATOR_GIT["tree"],
        expected_evaluator_branch=EVALUATOR_GIT["branch"],
        poll_seconds=0.001,
        timeout_seconds=1.0,
    )

    assert waiter.run_waiter(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    qualification_report = json.loads(
        args.qualification_output.read_text(encoding="utf-8")
    )

    assert status["status"] == scientific_status
    assert status["phase"] == "completed"
    assert qualification_report["status"] == scientific_status
    assert status["claim_boundary"]["training_process_signals_allowed"] is False
    assert not args.pid_file.exists()


def test_waiter_rejects_terminal_source_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    terminal_root = (tmp_path / "terminal").resolve()
    output_root = (tmp_path / "claim").resolve()
    project.mkdir(parents=True)
    quality_path = tmp_path / "quality_bridge_result.json"
    manifest_path = terminal_root / "reports" / "execution_manifest.json"
    uncertainty_path = terminal_root / "terminal_100k" / "matched_uncertainty.json"
    quality_identity = _write(quality_path, {"quality": True})
    manifest_identity = _write(manifest_path, {"manifest": True})
    uncertainty_identity = _write(uncertainty_path, {"uncertainty": True})
    terminal_status_path = terminal_root / "reports" / "waiter_status.json"
    _write(
        terminal_status_path,
        _terminal_status(
            pid=999_998,
            output_root=terminal_root,
            quality_identity=quality_identity,
            manifest_identity=manifest_identity,
            uncertainty_identity=uncertainty_identity,
        ),
    )
    _write(quality_path, {"quality": "changed"})
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    monkeypatch.setattr(waiter, "_terminal_process_active", lambda **_kwargs: False)
    args = argparse.Namespace(
        project=project,
        terminal_output_root=terminal_root,
        terminal_status=terminal_status_path,
        terminal_pid_file=terminal_root / "reports" / "waiter.pid",
        quality_result=quality_path,
        execution_manifest=manifest_path,
        uncertainty_report=uncertainty_path,
        output_root=output_root,
        status_output=output_root / "reports" / "waiter_status.json",
        pid_file=output_root / "reports" / "waiter.pid",
        qualification_output=(
            output_root / "reports" / "statistical_claim_qualification.json"
        ),
        expected_terminal_pid=999_998,
        expected_control_revision=CONTROL_GIT["revision"],
        expected_control_tree=CONTROL_GIT["tree"],
        expected_control_branch=CONTROL_GIT["branch"],
        expected_terminal_control_revision=TERMINAL_CONTROL_GIT["revision"],
        expected_terminal_control_tree=TERMINAL_CONTROL_GIT["tree"],
        expected_terminal_control_branch=TERMINAL_CONTROL_GIT["branch"],
        expected_quality_revision=QUALITY_GIT["revision"],
        expected_quality_branch=QUALITY_GIT["branch"],
        expected_evaluator_revision=EVALUATOR_GIT["revision"],
        expected_evaluator_tree=EVALUATOR_GIT["tree"],
        expected_evaluator_branch=EVALUATOR_GIT["branch"],
        poll_seconds=0.001,
        timeout_seconds=1.0,
    )

    with pytest.raises(ValueError, match="terminal source identities differ"):
        waiter.run_waiter(args)
