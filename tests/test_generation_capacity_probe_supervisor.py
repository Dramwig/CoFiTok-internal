from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_ROLE,
)
from scripts.run_generation_capacity_probe_execution_supervisor import (
    PREPARATION_WAITER_ROLE,
    _capacity_processes,
    _git_identity,
    _status,
    classify_execution_exit,
    validate_completed_execution,
    validate_preparation_waiter_status,
)


REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "scale/generation-capacity-probe-v1"


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
    assert (
        classify_execution_exit(exit_code=1, stage="base256_cofitok_evaluation")
        == "retry"
    )
    assert classify_execution_exit(exit_code=1, stage="authorization") == "fail"
    assert classify_execution_exit(exit_code=1, stage="storage_preflight") == "fail"
    assert (
        classify_execution_exit(
            exit_code=1,
            stage="cofitok_training_validation",
        )
        == "fail"
    )
    assert classify_execution_exit(exit_code=1, stage="result_build") == "fail"


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
    report = _status(
        status="waiting",
        detail="waiting_for_gpu_idle",
        expected={"required_idle_polls": 5},
        gpu_rows=[{"pid": 123}],
    )
    boundary = report["authorization_boundary"]
    assert boundary["unrelated_gpu_process_modification_allowed"] is False
    assert boundary["configured_100k_completion_allowed"] is False
    assert boundary["full_300k_launch_allowed"] is False


def test_capacity_probe_supervisor_runbook_is_locked_and_non_preemptive() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/generation_stability_capacity_probe_250m_10k_supervisor.sh"
    ).read_text(encoding="utf-8")
    assert "flock -n" in source
    assert "--required-idle-polls 5" in source
    assert "run_generation_capacity_probe_execution_supervisor.py" in source
    assert "kill " not in source
    assert "pkill" not in source


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
