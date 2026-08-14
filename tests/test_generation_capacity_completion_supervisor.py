from __future__ import annotations

import copy
import json
from pathlib import Path
import signal

import pytest

from cofitok.inference_replay import file_identity
from scripts.run_generation_capacity_completion_100k_supervisor import (
    DECISION_WAITER_BOUNDARY,
    DECISION_WAITER_ROLE,
    EXECUTION_BOUNDARY,
    EXECUTION_ROLE,
    RUNBOOK_NAME,
    _child_activity,
    _completed_execution,
    _is_relevant_command,
    _terminate_owned_child,
    validate_decision_waiter_status,
)


REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "scale/generation-capacity-completion-decision-waiter-v1"
EXECUTION_REVISION = "c" * 40
EXECUTION_TREE = "d" * 40
EXECUTION_BRANCH = "scale/generation-capacity-completion-execution-v1"


def _waiter_status(
    *,
    output_root: Path,
    decision: Path,
    result: Path,
    status: str,
) -> dict[str, object]:
    terminal = status == "completed"
    source = output_root / "source.json"
    source.write_text("source", encoding="utf-8")
    return {
        "schema_version": 1,
        "role": DECISION_WAITER_ROLE,
        "status": status,
        "detail": "fixture",
        "git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "output_root": output_root.resolve().as_posix(),
        "source_identities": {"source": file_identity(source)} if terminal else None,
        "capacity_scaling_result": file_identity(result) if terminal else None,
        "capacity_completion_decision": file_identity(decision) if terminal else None,
        "execution_authorized": True if terminal else None,
        "error": "fixture failure" if status == "failed" else None,
        "authorization_boundary": copy.deepcopy(DECISION_WAITER_BOUNDARY),
    }


@pytest.mark.parametrize("state", ("waiting", "completed", "failed"))
def test_supervisor_accepts_only_canonical_completion_waiter_states(
    tmp_path: Path,
    state: str,
) -> None:
    decision = tmp_path / "decision.json"
    result = tmp_path / "result.json"
    decision.write_text("{}", encoding="utf-8")
    result.write_text("{}", encoding="utf-8")
    report = _waiter_status(
        output_root=tmp_path,
        decision=decision,
        result=result,
        status=state,
    )
    evidence = validate_decision_waiter_status(
        report,
        expected_revision=REVISION,
        expected_tree=TREE,
        expected_branch=BRANCH,
        expected_output_root=tmp_path,
        expected_decision_path=decision,
        expected_result_path=result,
    )
    assert evidence["status"] == state


def test_supervisor_rejects_completion_waiter_scope_escalation(
    tmp_path: Path,
) -> None:
    decision = tmp_path / "decision.json"
    result = tmp_path / "result.json"
    decision.write_text("{}", encoding="utf-8")
    result.write_text("{}", encoding="utf-8")
    report = _waiter_status(
        output_root=tmp_path,
        decision=decision,
        result=result,
        status="waiting",
    )
    report["authorization_boundary"] = copy.deepcopy(DECISION_WAITER_BOUNDARY)
    report["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_decision_waiter_status(
            report,
            expected_revision=REVISION,
            expected_tree=TREE,
            expected_branch=BRANCH,
            expected_output_root=tmp_path,
            expected_decision_path=decision,
            expected_result_path=result,
        )


def test_supervisor_does_not_classify_its_own_runbook_argument_as_active() -> None:
    root = "/root/checkpoints/capacity"
    supervisor = (
        "python scripts/run_generation_capacity_completion_100k_supervisor.py "
        f"--runbook /project/artifacts/runbooks/{RUNBOOK_NAME} --output-root {root}"
    )
    child = f"bash /project/artifacts/runbooks/{RUNBOOK_NAME}"
    trainer = f"python scripts/train_generation.py --output-dir {root}/base256_cofitok"
    class_eval = (
        "python scripts/evaluate_generation_class_fidelity.py "
        f"--output-dir {root}/base256_cofitok/terminal_100k/class_fidelity"
    )
    assert _is_relevant_command(supervisor, output_root=root) is False
    assert _is_relevant_command(child, output_root=root) is True
    assert _is_relevant_command(trainer, output_root=root) is True
    assert _is_relevant_command(class_eval, output_root=root) is True


def test_supervisor_activity_uses_terminal_sampling_progress(
    tmp_path: Path,
) -> None:
    execution_status = tmp_path / "reports/capacity_completion_100k/execution_status.json"
    execution_status.parent.mkdir(parents=True)
    execution_status.write_text("{}", encoding="utf-8")
    progress = (
        tmp_path
        / "base256_cofitok/terminal_100k/"
        "samples_10000_ddim100_cfg15/sampling_progress.json"
    )
    progress.parent.mkdir(parents=True)
    progress.write_text("{}", encoding="utf-8")
    observation = _child_activity(
        tmp_path,
        execution_status=execution_status,
        child_started_at=1.0,
        now=progress.stat().st_mtime + 10.0,
    )
    assert observation["newest_path"] == progress.resolve().as_posix()
    assert observation["age_seconds"] == pytest.approx(10.0)


def test_supervisor_verifies_every_completed_terminal_identity(
    tmp_path: Path,
) -> None:
    identities: list[dict[str, object]] = []

    def artifact(name: str) -> dict[str, object]:
        path = tmp_path / "evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        identity = file_identity(path)
        identities.append(identity)
        return identity

    methods = {}
    for method in ("cofitok", "dense_identity"):
        methods[method] = {
            "training_validation": artifact(f"{method}-training"),
            "sampling_preflight": artifact(f"{method}-preflight"),
            "generation_metrics": artifact(f"{method}-metrics"),
            "checkpoint_evaluation": artifact(f"{method}-mechanism"),
            "class_fidelity": artifact(f"{method}-class"),
        }
    report = {
        "schema_version": 1,
        "role": EXECUTION_ROLE,
        "status": "completed",
        "stage": "complete",
        "git": {
            "revision": EXECUTION_REVISION,
            "tree": EXECUTION_TREE,
            "branch": EXECUTION_BRANCH,
            "tracked_dirty": False,
        },
        "output_root": tmp_path.resolve().as_posix(),
        "launch_receipt": artifact("launch"),
        "source_checkpoint_archive": artifact("archive"),
        "milestone_100000": artifact("milestone"),
        "class_fidelity_qualification": artifact("qualification"),
        "methods": methods,
        "authorization_boundary": copy.deepcopy(EXECUTION_BOUNDARY),
    }
    status = tmp_path / "execution_status.json"
    status.write_text(json.dumps(report), encoding="utf-8")
    assert _completed_execution(
        status,
        expected_revision=EXECUTION_REVISION,
        expected_tree=EXECUTION_TREE,
        expected_branch=EXECUTION_BRANCH,
        expected_output_root=tmp_path,
    ) == report
    Path(identities[-1]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        _completed_execution(
            status,
            expected_revision=EXECUTION_REVISION,
            expected_tree=EXECUTION_TREE,
            expected_branch=EXECUTION_BRANCH,
            expected_output_root=tmp_path,
        )


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
        "scripts.run_generation_capacity_completion_100k_supervisor.os.killpg",
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
