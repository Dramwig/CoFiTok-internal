from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "operations"
    / "generation"
    / "stability_50k_milestone_waiter.py"
)
SPEC = importlib.util.spec_from_file_location("stability_50k_milestone_waiter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
waiter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(waiter)


CONFIG_SHA = "a" * 64
TRAINING_REVISION = "b" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"


def progress_report() -> dict:
    schedules = {
        name: {
            "verified_rows": 801,
            "active_rows": 100,
            "nonzero_loss_rows": 100,
        }
        for name in ("rollout_consistency", "ema_teacher_consistency")
    }
    return {
        "status": "healthy",
        "last_step": 40000,
        "metric_row_count": 801,
        "issues": [],
        "warnings": [],
        "validation": {
            "event_count": 40,
            "logging_complete": True,
            "provenance_metadata_status": "complete",
        },
        "checkpoint": {
            "steps": [35000, 36545, 40000],
            "missing_required_steps": [],
            "latest": {
                "step": 40000,
                "git_revision": TRAINING_REVISION,
                "git_branch": TRAINING_BRANCH,
                "git_dirty": False,
            },
            "latest_integrity": {
                "status": "verified",
                "step": 40000,
                "checkpoint_sha256": "c" * 64,
            },
        },
        "consistency_schedules": {
            "status": "verified",
            "config_sha256": CONFIG_SHA,
            "schedules": schedules,
        },
    }


def validate(report: dict) -> None:
    waiter.validate_progress_report(
        report,
        milestone_step=40000,
        evaluation_interval=1000,
        required_checkpoint_steps=(35000, 36545, 40000),
        expected_config_sha256=CONFIG_SHA,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )


def test_accepts_complete_schedule_and_verified_milestone() -> None:
    validate(progress_report())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("validation", "event_count"), 39, "validation evidence"),
        (("validation", "provenance_metadata_status"), "missing", "provenance"),
        (("checkpoint", "steps"), [36545, 40000], "required recovery"),
        (("checkpoint", "latest_integrity", "status"), "invalid", "integrity"),
        (("checkpoint", "latest_integrity", "step"), 35000, "not the milestone"),
        (("checkpoint", "latest", "git_revision"), "wrong", "revision"),
        (("consistency_schedules", "config_sha256"), "wrong", "config SHA256"),
    ],
)
def test_rejects_incomplete_or_mismatched_evidence(
    path: tuple[str, ...], value: object, message: str
) -> None:
    report = progress_report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate(report)


@pytest.mark.parametrize(
    "schedule_name", ["rollout_consistency", "ema_teacher_consistency"]
)
def test_rejects_schedule_loss_that_is_zero_after_activation(
    schedule_name: str,
) -> None:
    report = progress_report()
    report["consistency_schedules"]["schedules"][schedule_name][
        "nonzero_loss_rows"
    ] = 99
    with pytest.raises(ValueError, match="inactive loss rows"):
        validate(report)


def test_audit_command_binds_config_and_all_recovery_steps(tmp_path: Path) -> None:
    command = waiter.audit_command(
        project_root=tmp_path / "project",
        run_dir=tmp_path / "run",
        config=tmp_path / "config.json",
        output=tmp_path / "report.json",
        expected_steps=50000,
        checkpoint_interval=5000,
        evaluation_interval=1000,
        required_checkpoint_steps=(35000, 36545, 40000),
    )
    assert command[command.index("--config") + 1] == str(tmp_path / "config.json")
    assert command[command.index("--required-checkpoint-steps") + 1] == (
        "35000,36545,40000"
    )
    assert command[command.index("--integrity-policy") + 1] == "required"


@pytest.mark.parametrize("inherited_pythonpath", [None, "existing/pythonpath"])
def test_progress_audit_injects_project_src(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inherited_pythonpath: str | None,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    if inherited_pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", inherited_pythonpath)
    captured: dict = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)
    waiter.run_progress_audit(command=["python", "audit.py"], project_root=project_root)
    expected = str(project_root / "src")
    if inherited_pythonpath:
        expected = os.pathsep.join((expected, inherited_pythonpath))
    assert captured["kwargs"]["env"]["PYTHONPATH"] == expected
    assert captured["kwargs"]["cwd"] == project_root
    assert captured["kwargs"]["check"] is True


def test_project_identity_is_exact() -> None:
    waiter.validate_project_identity(
        {"revision": "r", "branch": "b", "tracked_dirty": False},
        expected_revision="r",
        expected_branch="b",
    )
    with pytest.raises(ValueError, match="revision"):
        waiter.validate_project_identity(
            {"revision": "x", "branch": "b", "tracked_dirty": False},
            expected_revision="r",
            expected_branch="b",
        )
