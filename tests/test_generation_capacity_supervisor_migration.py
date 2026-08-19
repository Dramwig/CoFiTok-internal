from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

if os.name == "posix":
    import fcntl

from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_EXACT_TEXT,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
)
from cofitok.generation_control_process_migration import (
    APPROVAL_SCOPE,
    PLAN_SCOPE,
    SUPERVISOR_NAMES,
    SUPERVISOR_SPECS,
    _stop_old_process,
    build_supervisor_migration_approval,
    build_supervisor_migration_plan,
    execute_supervisor_migration,
)
from cofitok.inference_replay import file_identity
from cofitok.generation_control_continuity import git_state


ROOT = Path(__file__).resolve().parents[1]
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(path: Path, *, value: str, branch: str) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "revision": value * 40,
        "tree": value.upper() * 40,
        "branch": branch,
        "tracked_dirty": False,
        "porcelain_count": 0,
        "porcelain_sha256": EMPTY_SHA256,
    }


def _expected_git(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in ("path", "revision", "tree", "branch", "tracked_dirty")
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    target = tmp_path / "target"
    formal = tmp_path / "formal"
    target.mkdir()
    formal.mkdir()
    (target / "src/cofitok").mkdir(parents=True)
    (target / "src/cofitok/process_monitoring.py").write_text(
        "def publish_child_heartbeat(): pass\n", encoding="utf-8"
    )
    target_git = _git(target, value="a", branch="fix/heartbeat")
    formal_git = _git(formal, value="b", branch="scale/generative-system")

    standing = tmp_path / "standing_authorization.json"
    _write(
        standing,
        {
            "schema_version": 1,
            "role": STANDING_AUTHORIZATION_ROLE,
            "status": "active",
            "instruction": {
                "language": "zh-CN",
                "exact_text": STANDING_AUTHORIZATION_EXACT_TEXT,
                "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
                "received_at": "2026-08-19T00:00:00+00:00",
            },
            "preserved_safety_boundaries": STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
        },
    )

    runtimes: dict[int, dict[str, Any]] = {}
    old_git: dict[str, dict[str, Any]] = {}
    status_paths: dict[str, Path] = {}
    barrier_paths: dict[str, list[Path]] = {}
    lifetime_locks: dict[tuple[int, str], dict[str, Any]] = {}
    pids: dict[str, int] = {}
    for index, spec in enumerate(SUPERVISOR_SPECS):
        name = str(spec["name"])
        pid = 10_000 + index
        pids[name] = pid
        checkout = tmp_path / f"old-{index}"
        source = checkout / str(spec["entrypoint"])
        source.parent.mkdir(parents=True)
        source.write_text("# old supervisor\n", encoding="utf-8")
        target_source = target / str(spec["entrypoint"])
        target_source.parent.mkdir(parents=True, exist_ok=True)
        target_source.write_text(
            "from cofitok.process_monitoring import publish_child_heartbeat\n",
            encoding="utf-8",
        )
        status_path = tmp_path / "statuses" / f"{name}.json"
        status_paths[name] = status_path
        lock_filename = spec.get("lock_filename")
        if lock_filename is not None:
            lock_path = (status_path.parent.parent / str(lock_filename)).resolve()
            lock_path.touch()
            stat = lock_path.stat()
            lifetime_locks[(pid, lock_path.as_posix())] = {
                "path": lock_path.as_posix(),
                "identity": file_identity(lock_path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "owner_pid": pid,
                "owner_descriptors": [9],
                "exclusive_nonblocking_probe": "contended",
            }
        argv = [
            sys.executable,
            str(spec["entrypoint"]),
            "--status-output",
            status_path.resolve().as_posix(),
        ]
        barriers = []
        for option_index, option in enumerate(spec["barrier_options"]):
            barrier = tmp_path / "barriers" / f"{name}-{option_index}.json"
            barriers.append(barrier)
            argv.extend((str(option), barrier.resolve().as_posix()))
        barrier_paths[name] = barriers
        _write(
            status_path,
            {
                "role": spec["role"],
                "status": "waiting",
                "detail": spec["waiting_detail"],
                "pid": pid,
                "child_pid": None,
            },
        )
        log = tmp_path / "logs" / f"{name}.log"
        runtime = {
            "pid": pid,
            "ppid": 1,
            "process_group_id": pid,
            "session_id": pid,
            "nice": 0,
            "start_ticks": pid * 100,
            "umask": "0022",
            "cwd": checkout.resolve().as_posix(),
            "argv": argv,
            "executable_target": Path(sys.executable).resolve().as_posix(),
            "stdin_target": "/dev/null",
            "stdout_target": log.resolve().as_posix(),
            "stderr_target": log.resolve().as_posix(),
            "stdout_fd_flags": 1,
            "stderr_fd_flags": 1,
            "environment": {},
        }
        runtimes[pid] = runtime
        old_git[checkout.resolve().as_posix()] = _git(
            checkout,
            value=str((index + 1) % 10),
            branch=f"old/{index}",
        )

    states = {
        target.resolve().as_posix(): target_git,
        formal.resolve().as_posix(): formal_git,
        **old_git,
    }

    def inspect(pid: int) -> dict[str, Any]:
        try:
            return dict(runtimes[pid])
        except KeyError as error:
            raise ProcessLookupError(pid) from error

    def inspect_git(path: Path) -> dict[str, Any]:
        return dict(states[path.resolve().as_posix()])

    def inspect_lock(pid: int, path: Path) -> dict[str, Any]:
        return dict(lifetime_locks[(pid, path.resolve().as_posix())])

    return {
        "target": target,
        "formal": formal,
        "target_git": target_git,
        "formal_git": formal_git,
        "standing": standing,
        "pids": pids,
        "runtimes": runtimes,
        "inspect": inspect,
        "inspect_git": inspect_git,
        "inspect_lock": inspect_lock,
        "status_paths": status_paths,
        "barrier_paths": barrier_paths,
    }


def _plan(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    arguments = {
        "selected_pids": fixture["pids"],
        "target_project": fixture["target"],
        "expected_target_git": _expected_git(fixture["target_git"]),
        "formal_project": fixture["formal"],
        "expected_formal_git": _expected_git(fixture["formal_git"]),
        "standing_authorization_path": fixture["standing"],
        "expected_standing_authorization_sha256": file_identity(
            fixture["standing"]
        )["sha256"],
        "inspect_process": fixture["inspect"],
        "session_scanner": lambda _pids: [],
        "git_inspector": fixture["inspect_git"],
        "lock_inspector": fixture["inspect_lock"],
    }
    arguments.update(overrides)
    return build_supervisor_migration_plan(**arguments)


def _approval(tmp_path: Path, plan: dict[str, Any]) -> tuple[Path, str]:
    plan_path = tmp_path / "plan.json"
    approval_path = tmp_path / "approval.json"
    _write(plan_path, plan)
    approval = build_supervisor_migration_approval(
        plan_path=plan_path,
        expected_plan_sha256=file_identity(plan_path)["sha256"],
    )
    _write(approval_path, approval)
    return approval_path, file_identity(approval_path)["sha256"]


def test_plan_binds_exact_six_waiting_supervisors(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan = _plan(fixture)

    assert plan["status"] == "ready"
    assert plan["selected_process_names"] == list(SUPERVISOR_NAMES)
    assert [row["name"] for row in plan["old_processes"]] == list(
        SUPERVISOR_NAMES
    )
    assert [row["name"] for row in plan["new_processes"]] == list(
        SUPERVISOR_NAMES
    )
    assert all(row["status"]["child_pid"] is None for row in plan["old_processes"])
    assert all(
        row["runtime"]["cwd"] == fixture["target"].resolve().as_posix()
        for row in plan["new_processes"]
    )
    assert all(
        row["runtime"]["environment"]["PYTHONPATH"].split(os.pathsep)
        == [
            fixture["target"].resolve().as_posix(),
            (fixture["target"] / "src").resolve().as_posix(),
        ]
        for row in plan["new_processes"]
    )
    assert plan["scope"] == PLAN_SCOPE
    assert plan["effects"]["processes_signaled"] is False


def test_plan_rejects_any_nonwaiting_or_child_active_status(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    name = SUPERVISOR_NAMES[0]
    spec = SUPERVISOR_SPECS[0]
    _write(
        fixture["status_paths"][name],
        {
            "role": spec["role"],
            "status": "running",
            "detail": "capacity_scaling_child_running",
            "pid": fixture["pids"][name],
            "child_pid": 99_999,
        },
    )
    with pytest.raises(ValueError, match="not at the migration barrier"):
        _plan(fixture)


def test_plan_rejects_trigger_artifact_or_session_member(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    barrier = fixture["barrier_paths"][SUPERVISOR_NAMES[0]][0]
    _write(barrier, {"status": "appeared"})
    with pytest.raises(ValueError, match="trigger barrier already exists"):
        _plan(fixture)

    barrier.unlink()
    with pytest.raises(ValueError, match="session-member"):
        _plan(
            fixture,
            session_scanner=lambda _pids: [
                {
                    "pid": 88_888,
                    "selected_owner_pid": fixture["pids"][SUPERVISOR_NAMES[0]],
                }
            ],
        )


def test_plan_requires_exact_process_set_and_hardened_target(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    incomplete = dict(fixture["pids"])
    incomplete.pop(SUPERVISOR_NAMES[-1])
    with pytest.raises(ValueError, match="exactly the six"):
        _plan(fixture, selected_pids=incomplete)

    target = fixture["target"] / str(SUPERVISOR_SPECS[0]["entrypoint"])
    target.write_text("# missing heartbeat guard\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks heartbeat hardening"):
        _plan(fixture)


def test_stop_rechecks_waiting_barrier_before_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    plan = _plan(fixture)
    row = plan["old_processes"][0]
    spec = SUPERVISOR_SPECS[0]
    _write(
        fixture["status_paths"][str(spec["name"])],
        {
            "role": spec["role"],
            "status": "running",
            "detail": "child_started_after_plan",
            "pid": row["runtime"]["pid"],
            "child_pid": 99_999,
        },
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "cofitok.generation_control_process_migration.os.killpg",
        lambda pid, value: signals.append((pid, value)),
        raising=False,
    )

    with pytest.raises(ValueError, match="not at the migration barrier"):
        _stop_old_process(
            row,
            grace_seconds=0.1,
            inspect_process=fixture["inspect"],
            session_scanner=lambda _pids: [],
            lock_inspector=fixture["inspect_lock"],
        )

    assert signals == []


class _Handle:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0 if self.returncode is None else self.returncode


def _execute(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    waiter_failure_name: str | None = None,
    stopper_failure_name: str | None = None,
    publish=lambda _value: None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _plan(fixture)
    approval_path, approval_sha = _approval(tmp_path, plan)
    alive = {pid: dict(value) for pid, value in fixture["runtimes"].items()}
    stopped: list[str] = []
    launched: list[tuple[str, str, _Handle]] = []
    terminated: list[str] = []
    next_pid = 20_000

    def inspect(pid: int) -> dict[str, Any]:
        try:
            return dict(alive[pid])
        except KeyError as error:
            raise ProcessLookupError(pid) from error

    def stopper(row: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if row["name"] == stopper_failure_name:
            raise RuntimeError(f"stop failed:{row['name']}")
        stopped.append(row["name"])
        alive.pop(row["runtime"]["pid"])
        return {
            "name": row["name"],
            "pid": row["runtime"]["pid"],
            "signal": "SIGTERM",
            "forced": False,
        }

    def launcher(row: dict[str, Any]) -> _Handle:
        nonlocal next_pid
        next_pid += 1
        handle = _Handle(next_pid)
        kind = (
            "new"
            if row["runtime"]["cwd"] == fixture["target"].resolve().as_posix()
            else "old"
        )
        launched.append((kind, row["name"], handle))
        return handle

    def waiter(row: dict[str, Any], process: _Handle, **_kwargs: Any) -> dict[str, Any]:
        kind = (
            "new"
            if row["runtime"]["cwd"] == fixture["target"].resolve().as_posix()
            else "old"
        )
        if kind == "new" and row["name"] == waiter_failure_name:
            raise RuntimeError(f"status failed:{row['name']}")
        return {
            "name": row["name"],
            "role": row["role"],
            "pid": process.pid,
            "status": "waiting",
            "detail": row["waiting_detail"],
        }

    def terminate(row: dict[str, Any], process: _Handle, **_kwargs: Any) -> dict[str, Any]:
        process.returncode = -15
        terminated.append(row["name"])
        return {
            "name": row["name"],
            "pid": process.pid,
            "signaled": True,
            "forced": False,
        }

    report = execute_supervisor_migration(
        approval_path=approval_path,
        expected_approval_sha256=approval_sha,
        publish=publish,
        inspect_process=inspect,
        session_scanner=lambda _pids: [],
        source_validator=lambda _plan: None,
        old_stopper=stopper,
        launcher=launcher,
        status_waiter=waiter,
        terminator=terminate,
        lock_inspector=fixture["inspect_lock"],
    )
    return report, {
        "stopped": stopped,
        "launched": launched,
        "terminated": terminated,
    }


def test_execution_stops_downstream_first_and_launches_hardened_upstream_first(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report, observed = _execute(tmp_path, fixture)

    assert report["status"] == "pass"
    assert observed["stopped"] == list(reversed(SUPERVISOR_NAMES))
    assert [(kind, name) for kind, name, _ in observed["launched"]] == [
        ("new", name) for name in SUPERVISOR_NAMES
    ]
    assert observed["terminated"] == []
    assert report["scope"] == APPROVAL_SCOPE


def test_partial_new_launch_rolls_back_and_restores_all_old_commands(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    failed_name = SUPERVISOR_NAMES[2]
    report, observed = _execute(
        tmp_path,
        fixture,
        waiter_failure_name=failed_name,
    )

    assert report["status"] == "failed_restored"
    assert report["detail"] == f"status failed:{failed_name}"
    assert observed["terminated"] == list(reversed(SUPERVISOR_NAMES[:3]))
    assert [(kind, name) for kind, name, _ in observed["launched"]][-6:] == [
        ("old", name) for name in SUPERVISOR_NAMES
    ]
    assert [row["name"] for row in report["restored"]] == list(SUPERVISOR_NAMES)


def test_partial_stop_failure_restores_only_already_stopped_old_commands(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    failed_name = SUPERVISOR_NAMES[-3]
    report, observed = _execute(
        tmp_path,
        fixture,
        stopper_failure_name=failed_name,
    )

    assert report["status"] == "failed_restored"
    stopped = list(reversed(SUPERVISOR_NAMES))[:2]
    assert observed["stopped"] == stopped
    assert [(kind, name) for kind, name, _ in observed["launched"]] == [
        ("old", name) for name in SUPERVISOR_NAMES if name in stopped
    ]
    assert observed["terminated"] == []


def test_status_publication_failure_does_not_detach_migration(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def failed_publish(_value: object) -> None:
        raise OSError("status disk is full")

    report, _ = _execute(tmp_path, fixture, publish=failed_publish)
    assert report["status"] == "pass"
    assert report["publish_warnings"]
    assert report["publish_warnings"][0].endswith("status disk is full")


def test_runbook_is_explicit_exact_and_gpu_inert() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/generation_capacity_pipeline_supervisor_migration.sh"
    ).read_text(encoding="utf-8")
    assert "CONTROL_PROCESS_MIGRATION_ALLOWED must equal true" in source
    assert "generation_capacity_pipeline_supervisor_migration.py" in source
    assert "execute_generation_capacity_pipeline_supervisor_migration.py" in source
    assert "python3.10" in source
    assert "nvidia-smi" not in source
    for name in SUPERVISOR_NAMES:
        assert name in source


_DUMMY_SUPERVISOR = """\
from cofitok.process_monitoring import publish_child_heartbeat
import argparse
import json
import os
import time
from pathlib import Path

del publish_child_heartbeat
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--status-output", type=Path, required=True)
parser.add_argument("--fixture-role", required=True)
parser.add_argument("--fixture-detail", required=True)
args, _ = parser.parse_known_args()
payload = {
    "role": args.fixture_role,
    "status": "waiting",
    "detail": args.fixture_detail,
    "pid": os.getpid(),
    "child_pid": None,
}
temporary = args.status_output.with_name(f".{args.status_output.name}.{os.getpid()}.tmp")
args.status_output.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
os.replace(temporary, args.status_output)
while True:
    time.sleep(1.0)
"""


def _git_repo(path: Path, *, branch: str) -> dict[str, Any]:
    subprocess.run(["git", "init", "-b", branch], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "cofitok-tests@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CoFiTok tests"],
        cwd=path,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True)
    return git_state(path)


def _launch_detached(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Path,
    lock: Path | None = None,
) -> None:
    first = os.fork()
    if first == 0:
        second = os.fork()
        if second > 0:
            os._exit(0)
        os.setsid()
        os.chdir(cwd)
        stdin = os.open("/dev/null", os.O_RDONLY)
        output = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(stdin, 0)
        os.dup2(output, 1)
        os.dup2(output, 2)
        os.close(stdin)
        os.close(output)
        if lock is not None:
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock_descriptor = os.open(
                lock,
                os.O_WRONLY | os.O_CREAT,
                0o600,
            )
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.set_inheritable(lock_descriptor, True)
        os.execve(argv[0], argv, environment)
    os.waitpid(first, 0)


def _wait_for_pid(path: Path, *, timeout_seconds: float = 10.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = payload.get("pid")
            if type(pid) is int and pid > 0:
                return pid
        time.sleep(0.05)
    raise TimeoutError(f"fixture status did not appear: {path}")


def _terminate_groups(pids: set[int]) -> None:
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if not alive:
            return
        time.sleep(0.05)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="requires Linux process groups")
def test_posix_migrates_six_real_detached_waiting_supervisors(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    formal = tmp_path / "formal"
    target.mkdir()
    formal.mkdir()
    (formal / "tracked.txt").write_text("formal\n", encoding="utf-8")
    (target / "src/cofitok").mkdir(parents=True)
    (target / "src/cofitok/__init__.py").write_text("", encoding="utf-8")
    (target / "src/cofitok/process_monitoring.py").write_text(
        "def publish_child_heartbeat(): pass\n", encoding="utf-8"
    )
    for spec in SUPERVISOR_SPECS:
        source = target / str(spec["entrypoint"])
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(_DUMMY_SUPERVISOR, encoding="utf-8")
    target_git = _git_repo(target, branch="fix/heartbeat")
    formal_git = _git_repo(formal, branch="scale/generative-system")

    standing = tmp_path / "standing.json"
    _write(
        standing,
        {
            "schema_version": 1,
            "role": STANDING_AUTHORIZATION_ROLE,
            "status": "active",
            "instruction": {
                "language": "zh-CN",
                "exact_text": STANDING_AUTHORIZATION_EXACT_TEXT,
                "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
                "received_at": "2026-08-19T00:00:00+00:00",
            },
            "preserved_safety_boundaries": STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
        },
    )
    pids: dict[str, int] = {}
    cleanup: set[int] = set()
    environment = dict(os.environ)
    try:
        for index, spec in enumerate(SUPERVISOR_SPECS):
            checkout = tmp_path / f"old-{index}"
            checkout.mkdir()
            source = checkout / str(spec["entrypoint"])
            source.parent.mkdir(parents=True)
            source.write_text(_DUMMY_SUPERVISOR, encoding="utf-8")
            (checkout / "src/cofitok").mkdir(parents=True)
            (checkout / "src/cofitok/__init__.py").write_text("", encoding="utf-8")
            (checkout / "src/cofitok/process_monitoring.py").write_text(
                "def publish_child_heartbeat(): pass\n", encoding="utf-8"
            )
            _git_repo(checkout, branch=f"old-{index}")
            status = tmp_path / "statuses" / f"{spec['name']}.json"
            log = tmp_path / "logs" / f"{spec['name']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                str(spec["entrypoint"]),
                "--status-output",
                status.resolve().as_posix(),
                "--fixture-role",
                str(spec["role"]),
                "--fixture-detail",
                str(spec["waiting_detail"]),
            ]
            for barrier_index, option in enumerate(spec["barrier_options"]):
                argv.extend(
                    (
                        str(option),
                        (
                            tmp_path
                            / "barriers"
                            / f"{spec['name']}-{barrier_index}.json"
                        ).resolve().as_posix(),
                    )
                )
            process_environment = {
                **environment,
                "PYTHONPATH": os.pathsep.join(
                    (checkout.resolve().as_posix(), (checkout / "src").as_posix())
                ),
            }
            _launch_detached(
                argv,
                cwd=checkout,
                environment=process_environment,
                log=log,
                lock=(
                    status.parent.parent / str(spec["lock_filename"])
                    if spec.get("lock_filename") is not None
                    else None
                ),
            )
            pid = _wait_for_pid(status)
            pids[str(spec["name"])] = pid
            cleanup.add(pid)

        plan = build_supervisor_migration_plan(
            selected_pids=pids,
            target_project=target,
            expected_target_git=target_git,
            formal_project=formal,
            expected_formal_git=formal_git,
            standing_authorization_path=standing,
            expected_standing_authorization_sha256=file_identity(standing)["sha256"],
        )
        approval_path, approval_sha = _approval(tmp_path, plan)
        report = execute_supervisor_migration(
            approval_path=approval_path,
            expected_approval_sha256=approval_sha,
            publish=lambda _value: None,
            stop_grace_seconds=5.0,
            status_timeout_seconds=10.0,
        )

        assert report["status"] == "pass"
        migrated_pids = {row["pid"] for row in report["launched"]}
        cleanup.update(migrated_pids)
        assert len(migrated_pids) == len(SUPERVISOR_NAMES)
        assert not any(Path(f"/proc/{pid}").exists() for pid in pids.values())
        assert all(
            os.readlink(f"/proc/{pid}/cwd") == target.resolve().as_posix()
            for pid in migrated_pids
        )
        for spec in SUPERVISOR_SPECS:
            if spec.get("lock_filename") is None:
                continue
            lock_path = tmp_path / str(spec["lock_filename"])
            descriptor = os.open(lock_path, os.O_WRONLY)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(descriptor)
    finally:
        _terminate_groups(cleanup)
