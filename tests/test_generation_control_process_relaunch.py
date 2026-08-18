from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cofitok.generation_control_continuity import git_state
from cofitok.generation_control_process_relaunch import (
    APPROVAL_SCOPE,
    READINESS_AUTHORIZATION_BOUNDARY,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_TEXT,
    assess_process_relaunch_readiness,
    build_process_relaunch_approval,
    execute_process_relaunch,
)
from cofitok.inference_replay import file_identity


ROOT = Path(__file__).resolve().parents[1]
STANDING_BOUNDARIES = {
    "unrelated_project_processes_must_not_be_modified": True,
    "formal_remote_checkout_must_not_be_modified": True,
    "locked_evidence_must_not_be_overwritten": True,
    "independent_clean_checkout_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "stage_must_remain_non_authorizing_when_protocol_declares_non_authorizing": True,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_repo(path: Path) -> dict:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "fixture"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "cofitok-tests@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CoFiTok tests"], cwd=path, check=True
    )
    (path / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True)
    return git_state(path)


def _row(tmp_path: Path, *, index: int, name: str, status: str | None) -> dict:
    cwd = tmp_path / f"checkout-{name}"
    script = cwd / "scripts" / f"{name}.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    status_path = tmp_path / "statuses" / f"{name}.json"
    if status is not None:
        _write_json(
            status_path,
            {
                "schema_version": 1,
                "role": f"role_{name}",
                "status": status,
                "detail": f"fixture_{status}",
                "pid": 10_000 + index,
            },
        )
    log = tmp_path / "logs" / f"{name}.log"
    return {
        "index": index,
        "name": name,
        "role": f"role_{name}",
        "blocking": True,
        "status_path": status_path.resolve().as_posix(),
        "runtime": {
            "argv": ["python", f"scripts/{name}.py"],
            "cwd": cwd.resolve().as_posix(),
            "environment": {"CUDA_VISIBLE_DEVICES": ""},
            "nice": 0,
        },
        "relaunch_io": {
            "stdin_target": "/dev/null",
            "stdout_target": log.resolve().as_posix(),
            "stderr_target": log.resolve().as_posix(),
            "stdout_open_mode": "append",
            "stderr_open_mode": "append",
        },
        "entrypoint": {"argument_index": 1, "argument": f"scripts/{name}.py"},
    }


def _fixture(
    tmp_path: Path, statuses: tuple[str | None, ...] = ("waiting", "completed")
) -> dict:
    formal_git = _git_repo(tmp_path / "formal")
    implementation_git = _git_repo(tmp_path / "implementation")
    rows = [
        _row(tmp_path, index=index, name=f"stage_{index}", status=status)
        for index, status in enumerate(statuses)
    ]
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, {"processes": rows})
    standing_path = tmp_path / "standing_authorization.json"
    _write_json(
        standing_path,
        {
            "schema_version": 1,
            "role": STANDING_AUTHORIZATION_ROLE,
            "status": "active",
            "instruction": {"exact_text": STANDING_AUTHORIZATION_TEXT},
            "preserved_safety_boundaries": STANDING_BOUNDARIES,
        },
    )
    return {
        "formal_git": formal_git,
        "implementation_git": implementation_git,
        "rows": rows,
        "manifest_path": manifest_path,
        "standing_path": standing_path,
    }


def _source_verifier(**_: object) -> dict:
    return {"status": "pass", "source_verified_process_count": 2}


def _readiness(fixture: dict, *, live: list[dict] | None = None) -> dict:
    return assess_process_relaunch_readiness(
        manifest_path=fixture["manifest_path"],
        expected_manifest_sha256=file_identity(fixture["manifest_path"])["sha256"],
        standing_authorization_path=fixture["standing_path"],
        expected_standing_authorization_sha256=file_identity(
            fixture["standing_path"]
        )["sha256"],
        formal_git=fixture["formal_git"],
        implementation_git=fixture["implementation_git"],
        expected_process_count=len(fixture["rows"]),
        source_verifier=_source_verifier,
        process_scanner=lambda _: list(live or []),
    )


def _approval(tmp_path: Path, fixture: dict) -> tuple[Path, str]:
    readiness_path = tmp_path / "readiness.json"
    _write_json(readiness_path, _readiness(fixture))
    approval = build_process_relaunch_approval(
        readiness_path=readiness_path,
        expected_readiness_sha256=file_identity(readiness_path)["sha256"],
    )
    approval_path = tmp_path / "approval.json"
    _write_json(approval_path, approval)
    return approval_path, file_identity(approval_path)["sha256"]


def test_readiness_refuses_while_any_matching_control_process_is_alive(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    readiness = _readiness(
        fixture,
        live=[
            {
                "name": "stage_0",
                "role": "role_stage_0",
                "pid": 123,
                "start_ticks": 456,
                "cwd": fixture["rows"][0]["runtime"]["cwd"],
                "entrypoint": "fixture",
            }
        ],
    )
    assert readiness["status"] == "not_ready"
    assert readiness["approval_allowed"] is False
    assert readiness["selected_process_names"] == ["stage_0"]
    assert readiness["issues"] == ["matching_control_processes_are_still_alive"]
    assert readiness["effects"]["processes_launched"] is False


def test_readiness_selects_only_absent_active_or_missing_processes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, statuses=("waiting", None, "completed"))
    readiness = _readiness(fixture)
    assert readiness["status"] == "ready"
    assert readiness["approval_allowed"] is True
    assert readiness["selected_process_names"] == ["stage_0", "stage_1"]
    assert readiness["successful_process_names"] == ["stage_2"]
    assert readiness["issues"] == []
    assert readiness["authorization_boundary"] == READINESS_AUTHORIZATION_BOUNDARY

    readiness_path = tmp_path / "readiness.json"
    _write_json(readiness_path, readiness)
    approval = build_process_relaunch_approval(
        readiness_path=readiness_path,
        expected_readiness_sha256=file_identity(readiness_path)["sha256"],
    )
    assert approval["status"] == "approved"
    assert approval["selected_process_names"] == ["stage_0", "stage_1"]
    assert approval["scope"] == APPROVAL_SCOPE


def test_terminal_hold_prevents_relaunch_approval(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, statuses=("waiting", "hold"))
    readiness = _readiness(fixture)
    assert readiness["status"] == "terminal"
    assert readiness["approval_allowed"] is False
    assert readiness["issues"] == ["terminal_hold:stage_1"]
    readiness_path = tmp_path / "readiness.json"
    _write_json(readiness_path, readiness)
    with pytest.raises(ValueError, match="does not authorize"):
        build_process_relaunch_approval(
            readiness_path=readiness_path,
            expected_readiness_sha256=file_identity(readiness_path)["sha256"],
        )


class _Handle:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> None:
        return None


def test_execution_rechecks_readiness_and_publishes_each_selected_process(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, statuses=("waiting", None, "completed"))
    approval_path, approval_sha = _approval(tmp_path, fixture)
    launched: list[str] = []
    published: list[dict] = []

    def launcher(row: dict) -> _Handle:
        launched.append(row["name"])
        return _Handle(20_000 + len(launched))

    def waiter(row: dict, process: _Handle, **_: object) -> dict:
        return {
            "name": row["name"],
            "role": row["role"],
            "pid": process.pid,
            "status": "waiting",
            "classification": "active",
            "process_alive": True,
            "exit_code": None,
        }

    report = execute_process_relaunch(
        approval_path=approval_path,
        expected_approval_sha256=approval_sha,
        publish=lambda value: published.append(dict(value)),
        source_verifier=_source_verifier,
        process_scanner=lambda _: [],
        launcher=launcher,
        status_waiter=waiter,
    )
    assert report["status"] == "pass"
    assert launched == ["stage_0", "stage_1"]
    assert [row["name"] for row in report["launched"]] == launched
    assert [value["status"] for value in published] == [
        "launching",
        "launching",
        "launching",
        "pass",
    ]
    assert report["rollback"] == []


def test_partial_launch_failure_rolls_back_only_owned_processes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, statuses=("waiting", "waiting"))
    approval_path, approval_sha = _approval(tmp_path, fixture)
    launched: list[tuple[dict, _Handle]] = []
    rollback_calls: list[list[str]] = []
    published: list[dict] = []

    def launcher(row: dict) -> _Handle:
        process = _Handle(30_000 + len(launched))
        launched.append((row, process))
        return process

    def waiter(row: dict, process: _Handle, **_: object) -> dict:
        if row["name"] == "stage_1":
            raise RuntimeError("fixture failure")
        return {"name": row["name"], "pid": process.pid, "status": "waiting"}

    def rollback(values: list[tuple[dict, _Handle]], **_: object) -> list[dict]:
        names = [row["name"] for row, _ in values]
        rollback_calls.append(names)
        return [
            {"name": row["name"], "pid": process.pid, "signaled": True}
            for row, process in values
        ]

    report = execute_process_relaunch(
        approval_path=approval_path,
        expected_approval_sha256=approval_sha,
        publish=lambda value: published.append(dict(value)),
        source_verifier=_source_verifier,
        process_scanner=lambda _: [],
        launcher=launcher,
        status_waiter=waiter,
        rollback=rollback,
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert rollback_calls == [["stage_0", "stage_1"]]
    assert [row["name"] for row in report["rollback"]] == ["stage_0", "stage_1"]
    assert published[-1] == report


def test_production_runbook_is_inert_without_explicit_relaunch_switch() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/generation_capacity_control_plane_process_relaunch_after_restart.sh"
    ).read_text(encoding="utf-8")
    assert 'CONTROL_PROCESS_RELAUNCH_ALLOWED:-' in source
    assert 'CONTROL_PROCESS_RELAUNCH_ALLOWED must equal true' in source
    assert "--require-ready" in source
    assert "--issue-approval" in source
    assert "nvidia-smi" not in source


_DUMMY_CONTROL_SOURCE = """\
import json
import os
import sys
import time
from pathlib import Path

status_path = Path(sys.argv[1])
role = sys.argv[2]
mode = sys.argv[3]
if mode == "exit_before_status":
    raise SystemExit(17)
payload = {
    "schema_version": 1,
    "role": role,
    "status": "waiting",
    "detail": "posix_dummy_control_process",
    "pid": os.getpid(),
}
temporary = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
temporary.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
os.replace(temporary, status_path)
print(f"dummy-started:{os.getpid()}", flush=True)
while True:
    time.sleep(1.0)
"""


def _configure_posix_dummy_rows(fixture: dict, modes: tuple[str, ...]) -> None:
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    for row, mode in zip(fixture["rows"], modes, strict=True):
        script = Path(row["runtime"]["cwd"]) / row["entrypoint"]["argument"]
        script.write_text(_DUMMY_CONTROL_SOURCE, encoding="utf-8")
        row["runtime"]["argv"] = [
            sys.executable,
            row["entrypoint"]["argument"],
            row["status_path"],
            row["role"],
            mode,
        ]
        row["runtime"]["nice"] = current_nice
        log_path = Path(row["relaunch_io"]["stdout_target"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"sentinel:{row['name']}\n", encoding="utf-8")
    _write_json(fixture["manifest_path"], {"processes": fixture["rows"]})


def _terminate_and_reap(pids: list[int]) -> None:
    remaining = set(pids)
    for pid in list(remaining):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while remaining and time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                observed, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                remaining.remove(pid)
                continue
            if observed == pid:
                remaining.remove(pid)
        if remaining:
            time.sleep(0.05)
    for pid in list(remaining):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="requires Linux process groups")
def test_posix_detached_launch_appends_logs_and_blocks_repeat_relaunch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, statuses=("waiting", "waiting"))
    _configure_posix_dummy_rows(fixture, ("wait", "wait"))
    approval_path, approval_sha = _approval(tmp_path, fixture)
    pids: list[int] = []
    try:
        report = execute_process_relaunch(
            approval_path=approval_path,
            expected_approval_sha256=approval_sha,
            publish=lambda _: None,
            status_timeout_seconds=5.0,
            rollback_grace_seconds=1.0,
            source_verifier=_source_verifier,
        )
        assert report["status"] == "pass"
        pids = [row["pid"] for row in report["launched"]]
        assert len(pids) == 2
        for row in fixture["rows"]:
            log = Path(row["relaunch_io"]["stdout_target"]).read_text(
                encoding="utf-8"
            )
            assert log.startswith(f"sentinel:{row['name']}\n")
            assert "dummy-started:" in log

        repeat = assess_process_relaunch_readiness(
            manifest_path=fixture["manifest_path"],
            expected_manifest_sha256=file_identity(fixture["manifest_path"])[
                "sha256"
            ],
            standing_authorization_path=fixture["standing_path"],
            expected_standing_authorization_sha256=file_identity(
                fixture["standing_path"]
            )["sha256"],
            formal_git=fixture["formal_git"],
            implementation_git=fixture["implementation_git"],
            expected_process_count=2,
            source_verifier=_source_verifier,
        )
        assert repeat["status"] == "not_ready"
        assert repeat["approval_allowed"] is False
        assert {row["pid"] for row in repeat["live_matches"]} == set(pids)
        assert {row["name"] for row in repeat["live_matches"]} == {
            "stage_0",
            "stage_1",
        }
    finally:
        _terminate_and_reap(pids)


@pytest.mark.skipif(os.name != "posix", reason="requires Linux process groups")
def test_posix_partial_failure_rolls_back_actual_owned_process_groups(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, statuses=("waiting", "waiting"))
    _configure_posix_dummy_rows(fixture, ("wait", "exit_before_status"))
    approval_path, approval_sha = _approval(tmp_path, fixture)
    cleanup_pids: list[int] = []
    try:
        report = execute_process_relaunch(
            approval_path=approval_path,
            expected_approval_sha256=approval_sha,
            publish=lambda _: None,
            status_timeout_seconds=5.0,
            rollback_grace_seconds=2.0,
            source_verifier=_source_verifier,
        )
        assert report["status"] == "failed"
        assert report["error_type"] == "RuntimeError"
        assert "stage_1:17" in report["detail"]
        assert [row["name"] for row in report["rollback"]] == [
            "stage_1",
            "stage_0",
        ]
        assert report["rollback"][0]["signaled"] is False
        assert report["rollback"][1]["signaled"] is True
        cleanup_pids = [row["pid"] for row in report["rollback"]]
    finally:
        _terminate_and_reap(cleanup_pids)
