from __future__ import annotations

import copy
import signal
from pathlib import Path

import pytest

from scripts.run_generation_capacity_scaling_50k_supervisor import (
    DECISION_WAITER_ROLE,
    RUNBOOK_NAME,
    _child_activity,
    _is_relevant_command,
    _terminate_owned_child,
    validate_decision_waiter_status,
)


REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "scale/generation-capacity-scaling-decision-waiter-v1"


def _identity(path: Path) -> dict[str, object]:
    return {"path": path.resolve().as_posix(), "bytes": 2, "sha256": "c" * 64}


def _status(path: Path, *, status: str) -> dict[str, object]:
    terminal = status in {"completed", "not_selected"}
    return {
        "schema_version": 1,
        "role": DECISION_WAITER_ROLE,
        "status": status,
        "detail": "fixture",
        "self_git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "decision_path": path.resolve().as_posix(),
        "decision": _identity(path) if terminal else None,
        "recommended_next_stage": (
            {"id": "resume_matched_250m_capacity_bridge_to_50000"}
            if terminal
            else None
        ),
        "authorization_boundary": {
            "cpu_only_decision_build_allowed": True,
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "configured_100k_completion_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_or_release_allowed": False,
        },
    }


@pytest.mark.parametrize("state", ("waiting", "completed", "not_selected"))
def test_supervisor_accepts_only_canonical_decision_waiter_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text("{}")
    monkeypatch.setattr(
        "scripts.run_generation_capacity_scaling_50k_supervisor.file_identity",
        lambda path: _identity(decision),
    )
    evidence = validate_decision_waiter_status(
        _status(decision, status=state),
        expected_revision=REVISION,
        expected_tree=TREE,
        expected_branch=BRANCH,
        expected_decision_path=decision,
    )
    assert evidence["status"] == state


def test_supervisor_rejects_waiter_authorization_escalation(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text("{}")
    report = _status(decision, status="waiting")
    report["authorization_boundary"] = copy.deepcopy(
        report["authorization_boundary"]
    )
    report["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_decision_waiter_status(
            report,
            expected_revision=REVISION,
            expected_tree=TREE,
            expected_branch=BRANCH,
            expected_decision_path=decision,
        )


def test_supervisor_does_not_classify_its_own_runbook_argument_as_active() -> None:
    root = "/root/checkpoints/capacity"
    supervisor = (
        "python scripts/run_generation_capacity_scaling_50k_supervisor.py "
        f"--runbook /project/artifacts/runbooks/{RUNBOOK_NAME} --output-root {root}"
    )
    child = f"bash /project/artifacts/runbooks/{RUNBOOK_NAME}"
    trainer = f"python scripts/train_generation.py --output-dir {root}/base256_cofitok"
    assert _is_relevant_command(supervisor, output_root=root) is False
    assert _is_relevant_command(child, output_root=root) is True
    assert _is_relevant_command(trainer, output_root=root) is True


def test_supervisor_activity_uses_source_progress_not_its_own_status(
    tmp_path: Path,
) -> None:
    execution_status = tmp_path / "reports/capacity_scaling_50k_execution_status.json"
    execution_status.parent.mkdir(parents=True)
    execution_status.write_text("{}")
    metrics = tmp_path / "base256_cofitok/train_metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}\n")
    observation = _child_activity(
        tmp_path,
        execution_status=execution_status,
        child_started_at=1.0,
        now=metrics.stat().st_mtime + 10.0,
    )
    assert observation["newest_path"] == metrics.resolve().as_posix()
    assert observation["age_seconds"] == pytest.approx(10.0)


def test_supervisor_terminates_only_its_owned_child_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Child:
        pid = 4321
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    child = Child()
    calls: list[tuple[int, signal.Signals]] = []

    def killpg(pid: int, signum: signal.Signals) -> None:
        calls.append((pid, signum))
        child.returncode = -int(signum)

    monkeypatch.setattr(
        "scripts.run_generation_capacity_scaling_50k_supervisor.os.killpg",
        killpg,
        raising=False,
    )
    result = _terminate_owned_child(child, grace_seconds=0.1)  # type: ignore[arg-type]
    assert calls == [(child.pid, signal.SIGTERM)]
    assert result == {
        "signal": "SIGTERM",
        "forced_kill": False,
        "returncode": -int(signal.SIGTERM),
    }
