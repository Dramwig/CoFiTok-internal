from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import monitor_generation_pair as monitor


def _raw_cmdline(argv: list[str]) -> bytes:
    return b"\0".join(value.encode("utf-8") for value in argv) + b"\0"


def _write_process(
    proc_root: Path,
    *,
    pid: int,
    start_ticks: int,
    argv: list[str],
) -> None:
    process = proc_root / str(pid)
    process.mkdir()
    process.joinpath("cmdline").write_bytes(_raw_cmdline(argv))
    stat_tail = ["S", "1"] + ["0"] * 17 + [str(start_ticks)] + ["0"] * 8
    process.joinpath("stat").write_text(
        f"{pid} (process worker) " + " ".join(stat_tail),
        encoding="utf-8",
    )


def _readlink(path: object) -> str:
    candidate = Path(path)
    pid = int(candidate.parent.name)
    if candidate.name == "exe":
        return "/usr/bin/bash" if pid == 618_821 else "/usr/bin/python"
    if candidate.name == "cwd":
        return (
            "/tmp/quality-bridge/CoFiTok-internal"
            if pid == 618_821
            else "/root/checkouts/runtime-waiter/CoFiTok-internal"
        )
    raise AssertionError(candidate)


def _expected(raw: bytes) -> dict[str, object]:
    return {
        "pid": 618_821,
        "start_ticks": 1_660_953_145,
        "executable": "/usr/bin/bash",
        "cwd": "/tmp/quality-bridge/CoFiTok-internal",
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_exact_identity_ignores_nonbound_pattern_candidate(tmp_path: Path) -> None:
    controller_argv = ["bash", "artifacts/runbooks/controller.sh"]
    waiter_argv = [
        "python",
        "waiter.py",
        "--active-runbook",
        "/tmp/quality-bridge/CoFiTok-internal/artifacts/runbooks/controller.sh",
    ]
    _write_process(
        tmp_path,
        pid=618_821,
        start_ticks=1_660_953_145,
        argv=controller_argv,
    )
    _write_process(
        tmp_path,
        pid=832_803,
        start_ticks=1_664_412_098,
        argv=waiter_argv,
    )
    raw = _raw_cmdline(controller_argv)

    with patch.object(monitor.os, "readlink", side_effect=_readlink):
        live, evidence = monitor.inspect_exact_runbook_identity(
            expected=_expected(raw),
            pattern_processes=[
                "618821 bash artifacts/runbooks/controller.sh",
                "832803 python waiter.py --active-runbook controller.sh",
            ],
            proc_root=tmp_path,
        )

    assert live == ["618821 bash artifacts/runbooks/controller.sh"]
    assert evidence["status"] == "active"
    assert evidence["mismatches"] == []
    assert evidence["pattern_candidates"] == {
        "pids": [618_821, 832_803],
        "bound_pid_reported": True,
        "nonbound_pids": [832_803],
        "substring_candidate_contamination_observed": True,
    }


def test_missing_bound_controller_cannot_be_replaced_by_pattern_waiter(
    tmp_path: Path,
) -> None:
    waiter_argv = ["python", "waiter.py", "--active-runbook", "controller.sh"]
    _write_process(
        tmp_path,
        pid=832_803,
        start_ticks=1_664_412_098,
        argv=waiter_argv,
    )
    expected = {
        "pid": 618_821,
        "start_ticks": 1_660_953_145,
        "executable": "/usr/bin/bash",
        "cwd": "/tmp/quality-bridge/CoFiTok-internal",
        "cmdline_sha256": "a" * 64,
    }

    with patch.object(monitor.os, "readlink", side_effect=_readlink):
        live, evidence = monitor.inspect_exact_runbook_identity(
            expected=expected,
            pattern_processes=[
                "832803 python waiter.py --active-runbook controller.sh"
            ],
            proc_root=tmp_path,
        )

    assert live == []
    assert evidence["status"] == "invalid"
    assert evidence["mismatches"] == ["process_missing"]
    assert evidence["pattern_candidates"]["nonbound_pids"] == [832_803]


def test_start_tick_change_invalidates_pid_reuse(tmp_path: Path) -> None:
    controller_argv = ["bash", "artifacts/runbooks/controller.sh"]
    _write_process(
        tmp_path,
        pid=618_821,
        start_ticks=1_660_953_146,
        argv=controller_argv,
    )

    with patch.object(monitor.os, "readlink", side_effect=_readlink):
        live, evidence = monitor.inspect_exact_runbook_identity(
            expected=_expected(_raw_cmdline(controller_argv)),
            pattern_processes=[
                "618821 bash artifacts/runbooks/controller.sh"
            ],
            proc_root=tmp_path,
        )

    assert live == []
    assert evidence["mismatches"] == ["start_ticks"]


def test_incomplete_pair_fails_when_exact_controller_is_lost() -> None:
    report = {
        "status": "waiting",
        "stage": "cofitok_transition",
        "issues": [],
    }
    evidence = {
        "status": "invalid",
        "mismatches": ["process_missing"],
    }

    observed = monitor.enforce_exact_runbook_identity(
        report,
        evidence=evidence,
    )

    assert observed["status"] == "failed"
    assert observed["runbook_identity"] == evidence
    assert observed["issues"] == [
        "exact runbook identity is unavailable before pair completion: "
        "process_missing"
    ]


def test_completed_pair_remains_pass_after_controller_exit() -> None:
    report = {"status": "pass", "stage": "complete", "issues": []}
    evidence = {
        "status": "invalid",
        "mismatches": ["process_missing"],
    }

    observed = monitor.enforce_exact_runbook_identity(
        report,
        evidence=evidence,
    )

    assert observed["status"] == "pass"
    assert observed["issues"] == []


def test_exact_binding_requires_every_identity_field() -> None:
    args = Namespace(
        runbook_process_pid=618_821,
        runbook_process_start_ticks=0,
        runbook_process_executable="/usr/bin/bash",
        runbook_process_cwd="/tmp/quality-bridge/CoFiTok-internal",
        runbook_process_cmdline_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="incomplete"):
        monitor._exact_runbook_binding(args)


def test_complete_exact_binding_is_normalized() -> None:
    args = Namespace(
        runbook_process_pid=618_821,
        runbook_process_start_ticks=1_660_953_145,
        runbook_process_executable="/usr/bin/bash",
        runbook_process_cwd="/tmp/quality-bridge/CoFiTok-internal",
        runbook_process_cmdline_sha256="A" * 64,
    )

    assert monitor._exact_runbook_binding(args) == {
        "pid": 618_821,
        "start_ticks": 1_660_953_145,
        "executable": "/usr/bin/bash",
        "cwd": "/tmp/quality-bridge/CoFiTok-internal",
        "cmdline_sha256": "a" * 64,
    }


def test_legacy_pattern_mode_remains_available() -> None:
    args = Namespace(
        runbook_process_pid=0,
        runbook_process_start_ticks=0,
        runbook_process_executable="",
        runbook_process_cwd="",
        runbook_process_cmdline_sha256="",
    )

    assert monitor._exact_runbook_binding(args) is None
