from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cofitok.generation_control_continuity import git_state
from cofitok.generation_control_process_snapshot import (
    AUTHORIZATION_BOUNDARY,
    EFFECTS,
    build_process_relaunch_manifest,
    verify_process_relaunch_manifest,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import write_json_report


ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path: Path, files: dict[str, str], branch: str) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    for name, payload in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "fixture")
    _git(path, "branch", "-M", branch)
    return path


def _runtime(repo: Path, tmp_path: Path, pid: int, script: str) -> dict:
    log = tmp_path / f"{pid}.log"
    log.write_text("", encoding="utf-8")
    return {
        "pid": pid,
        "ppid": 1,
        "process_group_id": pid,
        "session_id": pid,
        "nice": 19,
        "start_ticks": pid * 100,
        "umask": "0022",
        "cwd": repo.resolve().as_posix(),
        "argv": [
            "/runtime/python",
            f"scripts/{script}",
            "--poll-seconds",
            "60",
        ],
        "executable_target": "/runtime/python3.10",
        "stdin_target": "/dev/null",
        "stdout_target": log.resolve().as_posix(),
        "stderr_target": log.resolve().as_posix(),
        "stdout_fd_flags": 0o102001,
        "stderr_fd_flags": 0o102001,
        "environment": {
            "PATH": "/usr/bin",
            "PYTHONPATH": repo.resolve().as_posix(),
        },
    }


def _fixture(tmp_path: Path) -> dict:
    process_repo = _repo(
        tmp_path / "process-repo",
        {
            "scripts/wait_a.py": "def main(): return 0\n",
            "scripts/run_b.py": "def main(): return 0\n",
            "scripts/observe.py": "def main(): return 0\n",
        },
        "scale/test-processes",
    )
    builder_repo = _repo(
        tmp_path / "builder-repo",
        {"builder.txt": "builder\n"},
        "scale/test-builder",
    )
    static_manifest = tmp_path / "static-manifest.json"
    write_json_report(static_manifest, {"schema_version": 1, "role": "static"})

    statuses = []
    specs = [
        (101, "stage_a", "role_a", "wait_a.py", True),
        (202, "stage_b", "role_b", "run_b.py", False),
    ]
    runtimes = {}
    stages = []
    for index, (pid, name, role, script, blocking) in enumerate(specs):
        status_path = tmp_path / f"{name}.json"
        write_json_report(
            status_path,
            {
                "schema_version": 1,
                "role": role,
                "status": "waiting",
                "detail": "fixture_wait",
                "pid": pid,
                "updated_at": "2026-08-15T00:00:00+00:00",
            },
        )
        statuses.append(status_path)
        runtimes[pid] = _runtime(process_repo, tmp_path, pid, script)
        stages.append(
            {
                "index": index,
                "name": name,
                "expected_role": role,
                "role": role,
                "blocking": blocking,
                "status_path": status_path.resolve().as_posix(),
                "pid": pid,
                "process_alive": True,
                "issues": [],
            }
        )

    observer_pid = 303
    runtimes[observer_pid] = _runtime(
        process_repo, tmp_path, observer_pid, "observe.py"
    )
    lineage = tmp_path / "lineage.json"
    write_json_report(
        lineage,
        {
            "schema_version": 1,
            "role": "capacity_generation_pipeline_lineage_observer",
            "status": "waiting",
            "detail": "waiting_at:stage_a",
            "observed_at": "2026-08-15T00:00:00+00:00",
            "issues": [],
            "observer": {"pid": observer_pid},
            "stages": stages,
        },
    )
    return {
        "process_repo": process_repo,
        "builder_repo": builder_repo,
        "static_manifest": static_manifest,
        "lineage": lineage,
        "runtimes": runtimes,
    }


def _build(tmp_path: Path) -> tuple[dict, Path, dict]:
    fixture = _fixture(tmp_path)
    inspector = lambda pid: dict(fixture["runtimes"][pid])
    manifest = build_process_relaunch_manifest(
        lineage_report_path=fixture["lineage"],
        static_continuity_manifest_path=fixture["static_manifest"],
        expected_static_continuity_sha256=file_identity(
            fixture["static_manifest"]
        )["sha256"],
        builder_git=git_state(fixture["builder_repo"]),
        expected_process_count=3,
        inspect_process=inspector,
        captured_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
        hostname="fixture-host",
    )
    path = tmp_path / "manifest.json"
    write_json_report(path, manifest)
    return fixture, path, manifest


def test_process_snapshot_captures_and_reverifies_exact_sources_and_runtime(
    tmp_path: Path,
) -> None:
    fixture, path, manifest = _build(tmp_path)
    assert manifest["process_count"] == 3
    assert manifest["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert manifest["effects"] == EFFECTS
    assert [row["name"] for row in manifest["processes"]] == [
        "stage_a",
        "stage_b",
        "capacity_pipeline_lineage_observer",
    ]
    report = verify_process_relaunch_manifest(
        manifest_path=path,
        expected_manifest_sha256=file_identity(path)["sha256"],
        expected_process_count=3,
        require_live=True,
        inspect_process=lambda pid: dict(fixture["runtimes"][pid]),
    )
    assert report["status"] == "pass"
    assert report["source_verified_process_count"] == 3
    assert report["live_verified_process_count"] == 3


def test_process_snapshot_source_verification_does_not_require_live_pids(
    tmp_path: Path,
) -> None:
    _, path, _ = _build(tmp_path)

    def unavailable(_: int) -> dict:
        raise AssertionError("source-only verification inspected a live process")

    report = verify_process_relaunch_manifest(
        manifest_path=path,
        expected_manifest_sha256=file_identity(path)["sha256"],
        expected_process_count=3,
        require_live=False,
        inspect_process=unavailable,
    )
    assert report["status"] == "pass"
    assert report["live_verified_process_count"] == 0


def test_process_snapshot_rejects_entrypoint_drift(tmp_path: Path) -> None:
    fixture, path, _ = _build(tmp_path)
    (fixture["process_repo"] / "scripts/wait_a.py").write_text(
        "changed = True\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="entrypoint identity differs"):
        verify_process_relaunch_manifest(
            manifest_path=path,
            expected_manifest_sha256=file_identity(path)["sha256"],
            expected_process_count=3,
            require_live=False,
        )


def test_process_snapshot_rejects_live_runtime_drift(tmp_path: Path) -> None:
    fixture, path, _ = _build(tmp_path)

    def changed(pid: int) -> dict:
        runtime = dict(fixture["runtimes"][pid])
        if pid == 202:
            runtime["nice"] = 0
        return runtime

    with pytest.raises(ValueError, match="live process runtime differs for stage_b"):
        verify_process_relaunch_manifest(
            manifest_path=path,
            expected_manifest_sha256=file_identity(path)["sha256"],
            expected_process_count=3,
            require_live=True,
            inspect_process=changed,
        )


def test_process_snapshot_preserves_live_flags_but_relaunches_logs_in_append_mode(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["runtimes"][101]["stdout_fd_flags"] = 0o100001
    manifest = build_process_relaunch_manifest(
        lineage_report_path=fixture["lineage"],
        static_continuity_manifest_path=fixture["static_manifest"],
        expected_static_continuity_sha256=file_identity(
            fixture["static_manifest"]
        )["sha256"],
        builder_git=git_state(fixture["builder_repo"]),
        expected_process_count=3,
        inspect_process=lambda pid: dict(fixture["runtimes"][pid]),
    )
    row = manifest["processes"][0]
    assert row["runtime"]["stdout_fd_flags"] == 0o100001
    assert row["relaunch_io"]["stdout_open_mode"] == "append"
    assert row["relaunch_io"]["stderr_open_mode"] == "append"


def test_process_snapshot_code_has_no_launch_signal_or_gpu_entrypoint() -> None:
    paths = [
        ROOT / "src/cofitok/generation_control_process_snapshot.py",
        ROOT / "scripts/capture_generation_control_plane_process_snapshot.py",
        ROOT / "scripts/verify_generation_control_plane_process_snapshot.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "subprocess.Popen" not in source
    assert "os.kill" not in source
    assert "signal." not in source
    assert "nvidia-smi" not in source
    assert "train_generation.py" not in source
    assert "generate_samples.py" not in source
