from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import (
    run_generation_quality_bridge_claim_language_guard_waiter as waiter,
)


CONTROL_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/claim-language-guard-waiter",
    "tracked_dirty": False,
}
SOURCE_CONTROL_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "analysis/quality-claim-waiter",
    "tracked_dirty": False,
}


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _source_status(
    *,
    pid: int,
    output_root: Path,
    source_report: Path,
    source_identity: dict[str, Any],
    status: str = "pass",
) -> dict[str, Any]:
    return {
        "schema_version": waiter.source_waiter.WAITER_SCHEMA_VERSION,
        "role": waiter.source_waiter.WAITER_ROLE,
        "status": status,
        "detail": (
            "matched_quality_bridge_fid_advantage_statistically_qualified"
            if status == "pass"
            else "matched_quality_bridge_fid_advantage_not_statistically_qualified"
        ),
        "phase": "completed" if status in {"pass", "hold"} else "source",
        "pid": pid,
        "expected": {
            "control_git": SOURCE_CONTROL_GIT,
            "output_root": output_root.resolve().as_posix(),
            "qualification_output": source_report.resolve().as_posix(),
        },
        "statistical_claim_qualification": (
            {
                "identity": source_identity,
                "status": status,
                "decision": (
                    "matched_quality_bridge_fid_advantage_statistically_qualified"
                    if status == "pass"
                    else (
                        "matched_quality_bridge_fid_advantage_not_"
                        "statistically_qualified"
                    )
                ),
                "claim_policy": {
                    "matched_relative_fid_advantage_claim_allowed": (
                        status == "pass"
                    ),
                    "broad_generation_superiority_claim_allowed": False,
                },
            }
            if status in {"pass", "hold"}
            else None
        ),
        "claim_boundary": waiter.source_waiter.CLAIM_BOUNDARY,
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def _args(
    *,
    project: Path,
    source_root: Path,
    output_root: Path,
    source_report: Path,
    expected_source_pid: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        project=project,
        source_output_root=source_root,
        source_status=source_root / "reports" / "waiter_status.json",
        source_pid_file=source_root / "reports" / "waiter.pid",
        source_report=source_report,
        output_root=output_root,
        status_output=output_root / "reports" / "waiter_status.json",
        pid_file=output_root / "reports" / "waiter.pid",
        guard_output=output_root / "reports" / "claim_language_guard.json",
        expected_source_pid=expected_source_pid,
        expected_control_revision=CONTROL_GIT["revision"],
        expected_control_tree=CONTROL_GIT["tree"],
        expected_control_branch=CONTROL_GIT["branch"],
        expected_source_control_revision=SOURCE_CONTROL_GIT["revision"],
        expected_source_control_tree=SOURCE_CONTROL_GIT["tree"],
        expected_source_control_branch=SOURCE_CONTROL_GIT["branch"],
        poll_seconds=0.001,
        timeout_seconds=1.0,
    )


def test_source_status_binds_exact_report_and_control(tmp_path: Path) -> None:
    source_root = (tmp_path / "source").resolve()
    source_report = source_root / "reports" / "qualification.json"
    identity = _write(source_report, {"source": True})
    report = _source_status(
        pid=123,
        output_root=source_root,
        source_report=source_report,
        source_identity=identity,
    )

    validated = waiter.validate_source_status(
        report,
        expected_pid=123,
        expected_output_root=source_root,
        expected_source_report=source_report,
        expected_control_git=SOURCE_CONTROL_GIT,
    )

    assert validated["terminal"] is True
    assert validated["status"] == "pass"
    assert validated["qualification"]["identity"] == identity


def test_source_status_rejects_authorizing_boundary(tmp_path: Path) -> None:
    source_root = (tmp_path / "source").resolve()
    source_report = source_root / "reports" / "qualification.json"
    identity = _write(source_report, {"source": True})
    report = _source_status(
        pid=123,
        output_root=source_root,
        source_report=source_report,
        source_identity=identity,
    )
    report["claim_boundary"] = dict(report["claim_boundary"])
    report["claim_boundary"]["full_300k_launch_allowed"] = True

    with pytest.raises(ValueError, match="contract differs"):
        waiter.validate_source_status(
            report,
            expected_pid=123,
            expected_output_root=source_root,
            expected_source_report=source_report,
            expected_control_git=SOURCE_CONTROL_GIT,
        )


@pytest.mark.parametrize("scientific_status", ["pass", "hold"])
def test_waiter_builds_one_bound_non_authorizing_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_status: str,
) -> None:
    project = (tmp_path / "project").resolve()
    source_root = (tmp_path / "source").resolve()
    output_root = (tmp_path / "guard").resolve()
    project.mkdir(parents=True)
    source_report = source_root / "reports" / "qualification.json"
    source_identity = _write(
        source_report,
        {
            "status": scientific_status,
            "decision": (
                "matched_quality_bridge_fid_advantage_statistically_qualified"
                if scientific_status == "pass"
                else (
                    "matched_quality_bridge_fid_advantage_not_"
                    "statistically_qualified"
                )
            ),
            "claim_policy": {
                "matched_relative_fid_advantage_claim_allowed": (
                    scientific_status == "pass"
                ),
                "broad_generation_superiority_claim_allowed": False,
            },
        },
    )
    source_status_path = source_root / "reports" / "waiter_status.json"
    _write(
        source_status_path,
        _source_status(
            pid=999_999,
            output_root=source_root,
            source_report=source_report,
            source_identity=source_identity,
            status=scientific_status,
        ),
    )
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    monkeypatch.setattr(waiter, "_source_process_active", lambda **_kwargs: False)
    monkeypatch.setattr(
        waiter.guard_builder,
        "build_guard",
        lambda **_kwargs: {
            "schema_version": 1,
            "role": "test_claim_language_guard",
            "status": scientific_status,
            "decision": (
                "lower_fid_point_estimate_with_paired_kid_support"
                if scientific_status == "pass"
                else "matched_distribution_quality_claim_not_supported"
            ),
            "metric_roles": {
                "fid": {"statistical_significance_tested": False},
                "paired_block_kid": {"statistical_significance_tested": True},
            },
            "claim_policy": {
                "matched_distribution_quality_claim_allowed": (
                    scientific_status == "pass"
                ),
                "fid_statistical_significance_claim_allowed": False,
            },
            "claim_text": "strict metric-semantics claim",
            "claim_boundary": waiter.guard_builder.CLAIM_BOUNDARY,
        },
    )
    args = _args(
        project=project,
        source_root=source_root,
        output_root=output_root,
        source_report=source_report,
        expected_source_pid=999_999,
    )

    assert waiter.run_waiter(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    guard = json.loads(args.guard_output.read_text(encoding="utf-8"))

    assert status["status"] == scientific_status
    assert status["phase"] == "completed"
    assert guard["status"] == scientific_status
    assert status["source"]["claim_report"] == source_identity
    assert status["claim_language_guard"]["identity"] == file_identity(
        args.guard_output
    )
    assert (
        status["claim_language_guard"]["claim_policy"][
            "fid_statistical_significance_claim_allowed"
        ]
        is False
    )
    assert status["claim_boundary"]["training_process_signals_allowed"] is False
    assert not args.pid_file.exists()


def test_waiter_rejects_source_report_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    source_root = (tmp_path / "source").resolve()
    output_root = (tmp_path / "guard").resolve()
    project.mkdir(parents=True)
    source_report = source_root / "reports" / "qualification.json"
    source_identity = _write(source_report, {"source": True})
    _write(
        source_root / "reports" / "waiter_status.json",
        _source_status(
            pid=999_998,
            output_root=source_root,
            source_report=source_report,
            source_identity=source_identity,
        ),
    )
    _write(source_report, {"source": "changed"})
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    monkeypatch.setattr(waiter, "_source_process_active", lambda **_kwargs: False)
    args = _args(
        project=project,
        source_root=source_root,
        output_root=output_root,
        source_report=source_report,
        expected_source_pid=999_998,
    )

    with pytest.raises(ValueError, match="source report identity differs"):
        waiter.run_waiter(args)


def test_waiter_rejects_source_report_summary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    source_root = (tmp_path / "source").resolve()
    output_root = (tmp_path / "guard").resolve()
    project.mkdir(parents=True)
    source_report = source_root / "reports" / "qualification.json"
    source_payload = {
        "status": "pass",
        "decision": "report-decision",
        "claim_policy": {"matched_relative_fid_advantage_claim_allowed": True},
    }
    source_identity = _write(source_report, source_payload)
    status_payload = _source_status(
        pid=999_995,
        output_root=source_root,
        source_report=source_report,
        source_identity=source_identity,
    )
    _write(source_root / "reports" / "waiter_status.json", status_payload)
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    monkeypatch.setattr(waiter, "_source_process_active", lambda **_kwargs: False)
    args = _args(
        project=project,
        source_root=source_root,
        output_root=output_root,
        source_report=source_report,
        expected_source_pid=999_995,
    )

    with pytest.raises(ValueError, match="source report summary differs"):
        waiter.run_waiter(args)


def test_waiter_propagates_source_failure_without_building_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    source_root = (tmp_path / "source").resolve()
    output_root = (tmp_path / "guard").resolve()
    project.mkdir(parents=True)
    source_report = source_root / "reports" / "qualification.json"
    source_status = _source_status(
        pid=999_997,
        output_root=source_root,
        source_report=source_report,
        source_identity={},
        status="failed",
    )
    _write(source_root / "reports" / "waiter_status.json", source_status)
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    monkeypatch.setattr(waiter, "_source_process_active", lambda **_kwargs: False)
    args = _args(
        project=project,
        source_root=source_root,
        output_root=output_root,
        source_report=source_report,
        expected_source_pid=999_997,
    )

    assert waiter.run_waiter(args) == 1
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["detail"] == "quality_claim_source_waiter_failed"
    assert not args.guard_output.exists()
    assert not args.pid_file.exists()


def test_waiter_rejects_output_inside_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = (tmp_path / "project").resolve()
    source_root = (tmp_path / "source").resolve()
    project.mkdir(parents=True)
    source_report = source_root / "reports" / "qualification.json"
    monkeypatch.setattr(waiter, "_git_identity", lambda _project: CONTROL_GIT)
    args = _args(
        project=project,
        source_root=source_root,
        output_root=source_root,
        source_report=source_report,
        expected_source_pid=999_996,
    )

    with pytest.raises(ValueError, match="path scope differs"):
        waiter.run_waiter(args)
