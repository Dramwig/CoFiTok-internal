from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
)
from scripts.run_generation_capacity_probe_execution_supervisor import (
    PREPARATION_WAITER_ROLE,
    _capacity_processes,
    _execution_child_environment,
    _git_identity,
    _execution_stage,
    _launch_capacity_probe_child,
    _status,
    _wait_for_capacity_probe_child,
    classify_execution_exit,
    validate_completed_execution,
    validate_preparation_waiter_status,
    verify_supervisor_checkout,
)


REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "scale/generation-capacity-probe-v1"
SUPERVISOR_REVISION = "c" * 40
SUPERVISOR_TREE = "d" * 40
SUPERVISOR_BRANCH = "fix/generation-capacity-probe-supervisor-v1"


def _waiter(preparation: Path, *, status: str) -> dict:
    identity = None
    recommendation = None
    if status == "completed":
        identity = {
            "path": preparation.resolve().as_posix(),
            "bytes": preparation.stat().st_size,
            "sha256": hashlib.sha256(preparation.read_bytes()).hexdigest(),
        }
        recommendation = "prepare_matched_250m_capacity_qualification_probe"
    return {
        "schema_version": 1,
        "role": PREPARATION_WAITER_ROLE,
        "status": status,
        "detail": "test",
        "polls": 3,
        "updated_at": "2026-08-14T00:00:00+00:00",
        "self_git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "preparation_path": preparation.resolve().as_posix(),
        "preparation": identity,
        "observed_recommendation": recommendation,
        "authorization_boundary": {
            "preparation_build_allowed": True,
            "capacity_probe_execution_allowed": False,
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_allowed": False,
        },
    }


def test_supervisor_accepts_only_source_bound_completed_preparation(
    tmp_path: Path,
) -> None:
    preparation = tmp_path / "preparation.json"
    preparation.write_text(json.dumps({"status": "prepared"}) + "\n")
    report = _waiter(preparation, status="completed")
    observed = validate_preparation_waiter_status(
        report,
        expected_revision=REVISION,
        expected_tree=TREE,
        expected_branch=BRANCH,
        expected_preparation_path=preparation,
    )
    assert observed["status"] == "completed"

    preparation.write_text(json.dumps({"status": "changed"}) + "\n")
    with pytest.raises(ValueError, match="not source-bound"):
        validate_preparation_waiter_status(
            report,
            expected_revision=REVISION,
            expected_tree=TREE,
            expected_branch=BRANCH,
            expected_preparation_path=preparation,
        )


def test_supervisor_classifies_only_recoverable_physical_stages_for_retry() -> None:
    assert classify_execution_exit(exit_code=9, stage="launch_guard") == "wait"
    assert classify_execution_exit(exit_code=75, stage="initialization") == "wait"
    assert classify_execution_exit(exit_code=1, stage="cofitok_training") == "retry"
    assert classify_execution_exit(exit_code=143, stage="cofitok_training") == "retry"
    assert (
        classify_execution_exit(exit_code=1, stage="base256_cofitok_evaluation")
        == "retry"
    )
    assert classify_execution_exit(exit_code=1, stage="authorization") == "fail"
    assert classify_execution_exit(exit_code=143, stage="authorization") == "fail"
    assert classify_execution_exit(exit_code=1, stage="storage_preflight") == "fail"
    assert (
        classify_execution_exit(
            exit_code=1,
            stage="cofitok_training_validation",
        )
        == "fail"
    )
    assert classify_execution_exit(exit_code=1, stage="result_build") == "fail"
    assert _execution_stage({"stage": "cofitok_training"}) == "cofitok_training"
    assert _execution_stage({"detail": "capacity probe stage failed: result_build"}) == (
        "initialization"
    )


def test_completed_execution_remains_non_authorizing(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "completed",
                "role": CAPACITY_PROBE_RESULT_ROLE,
                "git": {
                    "revision": REVISION,
                    "branch": BRANCH,
                    "tracked_dirty": False,
                },
                "claim_policy": {"formal_generation_claim_allowed": False},
                "authorization_boundary": CAPACITY_PROBE_RESULT_BOUNDARY,
            },
            sort_keys=True,
        )
        + "\n"
    )
    result_identity = {
        "path": result.resolve().as_posix(),
        "bytes": result.stat().st_size,
        "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
    }
    report = {
        "schema_version": 1,
        "role": "stability_full_data_capacity_probe_execution",
        "status": "completed",
        "stage": "completed",
        "detail": (
            "bounded capacity-probe evidence verified; no 100K/300K "
            "authorization created"
        ),
        "git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "capacity_probe_only": True,
        "intentional_training_stop_step": 10_000,
        "configured_100k_completion_allowed": False,
        "full_300k_launch_allowed": False,
        "report_is_promotion_gate": False,
        "release_authorization_allowed": False,
        "result": result_identity,
    }
    observed = validate_completed_execution(
        report,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        result_path=result,
    )
    assert observed["full_300k_launch_allowed"] is False
    drifted = copy.deepcopy(report)
    drifted["configured_100k_completion_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_completed_execution(
            drifted,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            result_path=result,
        )


def test_supervisor_status_preserves_safety_boundaries() -> None:
    execution_git = {
        "revision": REVISION,
        "tree": TREE,
        "branch": BRANCH,
        "tracked_dirty": False,
    }
    supervisor_git = {
        "revision": SUPERVISOR_REVISION,
        "tree": SUPERVISOR_TREE,
        "branch": SUPERVISOR_BRANCH,
        "tracked_dirty": False,
    }
    report = _status(
        status="waiting",
        detail="waiting_for_gpu_idle",
        expected={"required_idle_polls": 5, "execution_git": execution_git},
        supervisor_git=supervisor_git,
        gpu_rows=[{"pid": 123}],
    )
    assert report["supervisor_git"] == supervisor_git
    assert report["expected"]["execution_git"] == execution_git
    assert report["supervisor_git"]["revision"] != (
        report["expected"]["execution_git"]["revision"]
    )
    boundary = report["authorization_boundary"]
    assert boundary["unrelated_gpu_process_modification_allowed"] is False
    assert boundary["configured_100k_completion_allowed"] is False
    assert boundary["full_300k_launch_allowed"] is False


def test_capacity_probe_child_wait_refreshes_running_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    execution_status = tmp_path / "execution_status.json"
    execution_status.write_text(
        json.dumps({"stage": "cofitok_training"}) + "\n",
        encoding="utf-8",
    )
    reports: list[dict] = []

    class Child:
        pid = 456

        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, *, timeout: float) -> int:
            self.wait_calls += 1
            if self.wait_calls < 3:
                raise subprocess.TimeoutExpired(["child"], timeout)
            return 0

    monkeypatch.setattr(
        "scripts.run_generation_capacity_probe_execution_supervisor._capacity_processes",
        lambda output_root: [{"pid": 456, "command": "capacity probe"}],
    )
    monkeypatch.setattr(
        "scripts.run_generation_capacity_probe_execution_supervisor._gpu_compute_rows",
        lambda: [{"pid": 456, "used_memory_mib": 1024}],
    )
    child = Child()
    result = _wait_for_capacity_probe_child(
        child,  # type: ignore[arg-type]
        poll_seconds=120.0,
        execution_status_path=execution_status,
        output_root=tmp_path,
        publish=lambda **values: reports.append(values),
    )

    assert result == 0
    assert child.wait_calls == 3
    assert len(reports) == 2
    assert all(report["status"] == "running" for report in reports)
    assert all(report["child_pid"] == 456 for report in reports)
    assert all(report["execution_stage"] == "cofitok_training" for report in reports)
    assert reports[0]["capacity_processes"] == [
        {"pid": 456, "command": "capacity probe"}
    ]
    assert reports[0]["gpu_rows"] == [{"pid": 456, "used_memory_mib": 1024}]


def test_capacity_probe_supervisor_runbook_is_locked_and_non_preemptive() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/generation_stability_capacity_probe_250m_10k_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert (
        "exec flock --exclusive --nonblock --conflict-exit-code 75 --no-fork"
        in source
    )
    assert "--required-idle-polls 5" in source
    assert "run_generation_capacity_probe_execution_supervisor.py" in source
    assert '--supervisor-project "$SUPERVISOR_PROJECT"' in source
    assert '--execution-project "$EXECUTION_PROJECT"' in source
    assert (
        'export PYTHONPATH="$SUPERVISOR_PROJECT:$SUPERVISOR_PROJECT/src"'
        in source
    )
    assert (
        'RUNBOOK="$EXECUTION_PROJECT/artifacts/runbooks/'
        'generation_stability_capacity_probe_250m_10k_execute.sh"'
        in source
    )
    assert '--expected-supervisor-revision "$EXPECTED_SUPERVISOR_REVISION"' in source
    assert 'RUNTIME_TMP="$OUTPUT_ROOT/runtime_tmp"' in source
    assert 'export TMPDIR="$RUNTIME_TMP"' in source
    assert 'export TEMP="$RUNTIME_TMP"' in source
    assert 'export TMP="$RUNTIME_TMP"' in source
    assert "kill " not in source
    assert "pkill" not in source


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("flock") is None,
    reason="requires Linux flock",
)
def test_capacity_probe_supervisor_holds_lock_for_process_lifetime(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    runbook = (
        root
        / "artifacts/runbooks/generation_stability_capacity_probe_250m_10k_supervisor.sh"
    )
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nexec sleep 30\n", encoding="utf-8")
    fake_python.chmod(0o755)
    standing_authorization = tmp_path / "standing_authorization.json"
    standing_authorization.write_text("{}\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    execution_project = tmp_path / "execution-checkout"
    execution_project.mkdir()
    lock_path = (
        checkpoint_root
        / "stability_full_data_100k_capacity_probe_250m_10k_v1"
        / "capacity_probe_execution_supervisor.lock"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "SUPERVISOR_PROJECT": root.as_posix(),
            "EXECUTION_PROJECT": execution_project.as_posix(),
            "PREPARATION_PROJECT": root.as_posix(),
            "PYTHON": fake_python.as_posix(),
            "CHECKPOINT_ROOT": checkpoint_root.as_posix(),
            "STANDING_AUTHORIZATION": standing_authorization.as_posix(),
            "EXPECTED_STANDING_AUTHORIZATION_SHA256": "a" * 64,
            "EXPECTED_TARGET_REVISION": "b" * 40,
            "EXPECTED_TARGET_TREE": "c" * 40,
            "EXPECTED_TARGET_BRANCH": "test-branch",
            "EXPECTED_SUPERVISOR_REVISION": "f" * 40,
            "EXPECTED_SUPERVISOR_TREE": "1" * 40,
            "EXPECTED_SUPERVISOR_BRANCH": "test-supervisor-branch",
            "EXPECTED_PREPARATION_REVISION": "d" * 40,
            "EXPECTED_PREPARATION_TREE": "e" * 40,
            "EXPECTED_PREPARATION_BRANCH": "test-preparation-branch",
        }
    )
    child = subprocess.Popen(
        ["bash", str(runbook)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not lock_path.is_file() and child.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("capacity supervisor lock was not created")
            time.sleep(0.05)
        assert child.poll() is None
        competing = subprocess.run(
            [
                "flock",
                "--exclusive",
                "--nonblock",
                "--conflict-exit-code",
                "75",
                str(lock_path),
                "/bin/true",
            ],
            check=False,
        )
        assert competing.returncode == 75
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5.0)


def test_capacity_process_scan_ignores_supervisor_runbook_argument(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Result:
        stdout = "\n".join(
            [
                (
                    "100 1 3 /python scripts/"
                    "run_generation_capacity_probe_execution_supervisor.py "
                    "--runbook /project/artifacts/runbooks/"
                    "generation_stability_capacity_probe_250m_10k_execute.sh"
                ),
                (
                    "200 100 2 bash /project/artifacts/runbooks/"
                    "generation_stability_capacity_probe_250m_10k_execute.sh"
                ),
            ]
        )

    monkeypatch.setattr(
        "scripts.run_generation_capacity_probe_execution_supervisor.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    rows = _capacity_processes(tmp_path)
    assert [row["pid"] for row in rows] == [200]


def test_preparation_git_identity_can_ignore_only_untracked_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    class Result:
        stdout = ""

    def run(arguments, **kwargs):
        calls.append(tuple(arguments))
        result = Result()
        if arguments[-2:] == ["rev-parse", "HEAD"]:
            result.stdout = REVISION + "\n"
        elif arguments[-2:] == ["rev-parse", "HEAD^{tree}"]:
            result.stdout = TREE + "\n"
        elif arguments[-2:] == ["branch", "--show-current"]:
            result.stdout = BRANCH + "\n"
        elif "--untracked-files=no" not in arguments:
            result.stdout = "?? waiter.log\n"
        return result

    monkeypatch.setattr(
        "scripts.run_generation_capacity_probe_execution_supervisor.subprocess.run",
        run,
    )
    tracked = _git_identity(tmp_path, include_untracked=False)
    strict = _git_identity(tmp_path, include_untracked=True)
    assert tracked["tracked_dirty"] is False
    assert strict["tracked_dirty"] is True
    assert any("--untracked-files=no" in command for command in calls)


def test_supervisor_checkout_is_bound_to_its_own_script_and_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "supervisor"
    script = project / "scripts/run_generation_capacity_probe_execution_supervisor.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test\n", encoding="utf-8")
    identity = {
        "revision": SUPERVISOR_REVISION,
        "tree": SUPERVISOR_TREE,
        "branch": SUPERVISOR_BRANCH,
        "tracked_dirty": False,
    }
    monkeypatch.setattr(
        "scripts.run_generation_capacity_probe_execution_supervisor._git_identity",
        lambda root: identity,
    )

    assert verify_supervisor_checkout(
        project,
        expected_revision=SUPERVISOR_REVISION,
        expected_tree=SUPERVISOR_TREE,
        expected_branch=SUPERVISOR_BRANCH,
        supervisor_script=script,
    ) == identity
    with pytest.raises(ValueError, match="script is not checkout-bound"):
        verify_supervisor_checkout(
            project,
            expected_revision=SUPERVISOR_REVISION,
            expected_tree=SUPERVISOR_TREE,
            expected_branch=SUPERVISOR_BRANCH,
            supervisor_script=tmp_path / "another-supervisor.py",
        )


def test_execution_child_environment_keeps_training_on_target_checkout(
    tmp_path: Path,
) -> None:
    execution_project = tmp_path / "execution"
    environment = _execution_child_environment(
        base_environment={
            "KEEP": "yes",
            "PYTHONPATH": "/supervisor:/supervisor/src",
        },
        execution_project=execution_project,
        python_executable="/env/bin/python",
        values={
            "EXPECTED_TARGET_REVISION": REVISION,
            "EXPECTED_TARGET_TREE": TREE,
            "EXPECTED_TARGET_BRANCH": BRANCH,
        },
    )

    assert environment["KEEP"] == "yes"
    assert environment["PYTHON"] == "/env/bin/python"
    assert environment["PYTHONPATH"] == os.pathsep.join(
        [
            execution_project.as_posix(),
            (execution_project / "src").as_posix(),
        ]
    )
    assert "/supervisor" not in environment["PYTHONPATH"]
    assert environment["EXPECTED_TARGET_REVISION"] == REVISION
    assert environment["EXPECTED_TARGET_TREE"] == TREE
    assert environment["EXPECTED_TARGET_BRANCH"] == BRANCH


def test_capacity_probe_child_launch_uses_execution_cwd_and_runbook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    execution_project = tmp_path / "execution"
    runbook = execution_project / "artifacts/runbooks/execute.sh"
    captured: dict[str, object] = {}
    sentinel = object()

    def popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "scripts.run_generation_capacity_probe_execution_supervisor.subprocess.Popen",
        popen,
    )
    child = _launch_capacity_probe_child(
        runbook=runbook,
        execution_project=execution_project,
        environment={"PYTHONPATH": "target"},
    )

    assert child is sentinel
    assert captured["arguments"] == ["bash", str(runbook)]
    assert captured["cwd"] == execution_project
    assert captured["env"] == {"PYTHONPATH": "target"}
