from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256
from scripts import run_generation_stability_posttraining_supervisor as supervisor


TRAINING_REVISION = "a" * 40
TRAINING_BRANCH = "scale/generation-large-capacity"
EVALUATION_REVISION = "b" * 40
EVALUATION_BRANCH = "scale/generation-stability-full-eval"


def _monitor(*, status: str = "pass") -> dict:
    complete = status == "pass"
    runs = {}
    for method in ("cofitok", "dense_identity"):
        runs[method] = {
            "complete": complete,
            "expected_steps": 300_000,
            "last_step": 300_000 if complete else 200_000,
            "health_issues": [],
        }
    return {
        "schema_version": 2,
        "monitor": supervisor.FULL_MONITOR_NAME,
        "status": status,
        "stage": "complete" if complete else "cofitok_training",
        "issues": [],
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "runs": runs,
        "updated_at": "2026-07-31T00:00:00+00:00",
    }


def test_full_monitor_requires_both_exact_300k_runs() -> None:
    observation = supervisor.validate_full_monitor(
        _monitor(),
        expected_revision=TRAINING_REVISION,
        expected_branch=TRAINING_BRANCH,
    )
    assert observation["complete"] is True

    report = _monitor()
    report["runs"]["dense_identity"]["last_step"] = 299_999
    with pytest.raises(ValueError, match="dense_identity"):
        supervisor.validate_full_monitor(
            report,
            expected_revision=TRAINING_REVISION,
            expected_branch=TRAINING_BRANCH,
        )


def test_full_launch_receipt_rehashes_every_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = tmp_path / "stability_full_300k_ema_teacher"
    sources = {}
    for name in (
        "deployment_receipt",
        "promotion_gate",
        "stability_supplemental",
        "scaling_class_fidelity",
        "full_readiness",
        "readiness_bridge",
        "cofitok_config",
        "dense_config",
        "config_validation",
        "storage_capacity",
        "runtime_selection",
        "launch_storage_capacity",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        sources[name] = supervisor._source_identity(path)
    sources["promotion_gate"]["sha256"] = "c" * 64
    sources["full_readiness"]["sha256"] = "d" * 64
    report = {
        "schema_version": 4,
        "status": "pass",
        "role": supervisor.FULL_LAUNCH_ROLE,
        "stage": "stability_full",
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "readiness_sha256": "d" * 64,
        "quality_prerequisites": {
            "frozen_stability_supplemental": {
                "report": sources["stability_supplemental"],
                "checks": {
                    "base_gate_passed": True,
                    "distribution_support_passed": True,
                    "ema_rollout_stability_passed": True,
                    "all_supplemental_quality_checks_passed": True,
                },
                "supplemental_non_authorizing": True,
                "required_for_full_training_launch": True,
                "full_training_launch_allowed": False,
            },
            "frozen_scaling_class_fidelity": {
                "report": sources["scaling_class_fidelity"],
                "promotion_gate": sources["promotion_gate"],
                "evaluator_git": {
                    "revision": EVALUATION_REVISION,
                    "branch": EVALUATION_BRANCH,
                    "tracked_dirty": False,
                },
                "class_fidelity_passed": True,
                "supplemental_non_authorizing": True,
                "required_for_full_training_launch": True,
                "full_training_launch_allowed": False,
            },
        },
        "full_training_launch_authorized": True,
        "formal_generation_completion_claimed": False,
        "source_reports": sources,
        "training_run_dirs": [
            (full_root / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher").as_posix(),
            (full_root / "dense_rollout_x0_u2_ema_teacher").as_posix(),
        ],
    }
    receipt = tmp_path / "full_training_launch_receipt.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")

    # Restore the two intentionally synthetic hashes through matching file identities.
    for name, digest in (("promotion_gate", "c" * 64), ("full_readiness", "d" * 64)):
        sources[name]["sha256"] = supervisor.file_sha256(Path(sources[name]["path"]))
    report["readiness_sha256"] = sources["full_readiness"]["sha256"]
    receipt.write_text(json.dumps(report), encoding="utf-8")
    supplemental_replays = []

    def replay_supplemental(*args, **kwargs):
        supplemental_replays.append(kwargs)
        return report["quality_prerequisites"][
            "frozen_stability_supplemental"
        ]

    monkeypatch.setattr(
        supervisor,
        "verify_frozen_supplemental_report",
        replay_supplemental,
    )
    class_fidelity_replays = []

    def replay_class_fidelity(*args, **kwargs):
        class_fidelity_replays.append(kwargs)
        return report["quality_prerequisites"][
            "frozen_scaling_class_fidelity"
        ]

    monkeypatch.setattr(
        supervisor,
        "verify_frozen_class_fidelity_qualification",
        replay_class_fidelity,
    )
    evidence = supervisor.validate_full_launch_receipt(
        report,
        receipt_path=receipt,
        expected_receipt_sha256=file_sha256(receipt),
        expected_readiness_sha256=sources["full_readiness"]["sha256"],
        expected_scaling_gate_sha256=sources["promotion_gate"]["sha256"],
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        checkpoint_root=tmp_path,
    )
    assert evidence["source_count"] == 12
    assert evidence["stability_supplemental_sha256"] == sources[
        "stability_supplemental"
    ]["sha256"]
    assert supplemental_replays == [
        {
            "report_path": Path(sources["stability_supplemental"]["path"]),
            "expected_report_sha256": sources["stability_supplemental"][
                "sha256"
            ],
            "promotion_gate_path": Path(sources["promotion_gate"]["path"]),
        }
    ]
    assert class_fidelity_replays == [
        {
            "report_path": Path(sources["scaling_class_fidelity"]["path"]),
            "expected_report_sha256": sources["scaling_class_fidelity"][
                "sha256"
            ],
            "promotion_gate_path": Path(sources["promotion_gate"]["path"]),
            "expected_evaluator_revision": EVALUATION_REVISION,
            "expected_evaluator_branch": EVALUATION_BRANCH,
        }
    ]

    Path(sources["dense_config"]["path"]).write_text("replaced", encoding="ascii")
    with pytest.raises(ValueError, match="dense_config"):
        supervisor.validate_full_launch_receipt(
            report,
            receipt_path=receipt,
            expected_receipt_sha256=file_sha256(receipt),
            expected_readiness_sha256=sources["full_readiness"]["sha256"],
            expected_scaling_gate_sha256=sources["promotion_gate"]["sha256"],
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            checkpoint_root=tmp_path,
        )


def test_completion_reuse_requires_exact_expectations() -> None:
    expected = {
        "decision_sha256": "a" * 64,
        "full_training_revision": TRAINING_REVISION,
    }
    report = {
        "profile": supervisor.COMPLETION_PROFILE,
        "status": "pass",
        "complete": True,
        "failed_checks": [],
        "missing_checks": [],
        "checks": [{"name": "all", "status": "pass"}],
        "expectations": dict(expected),
    }
    assert supervisor.validate_completion(report, expected=expected)["check_count"] == 1
    report["expectations"]["full_training_revision"] = "f" * 40
    with pytest.raises(ValueError, match="expectations differ"):
        supervisor.validate_completion(report, expected=expected)


def test_stage_runner_stops_on_nonretryable_audit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "completion.json"
    report.write_text(
        json.dumps({"status": "failed", "complete": False}),
        encoding="utf-8",
    )
    calls: list[dict] = []

    class Child:
        pid = 19

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 5.0
            return 1

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Child())
    with pytest.raises(RuntimeError, match="nonretryable report"):
        supervisor._run_stage(
            name="completion_audit",
            runbook=tmp_path / "audit.sh",
            project=tmp_path,
            environment={},
            max_attempts=3,
            poll_seconds=5.0,
            retry_seconds=1.0,
            write_status=lambda **kwargs: calls.append(kwargs),
            nonretryable_report=report,
        )
    assert len(calls) == 2
    assert calls[-1]["detail"] == "completion_audit_nonretryable_report"


def test_stage_runner_stops_immediately_when_replay_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    launches = 0

    class Child:
        pid = 23

        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 5.0
            return supervisor.STAGE_REPLAY_ERROR_EXIT_CODE

    def popen(*args, **kwargs):
        nonlocal launches
        del args, kwargs
        launches += 1
        return Child()

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="replay validation failed"):
        supervisor._run_stage(
            name="inference_export",
            runbook=tmp_path / "export.sh",
            project=tmp_path,
            environment={},
            max_attempts=3,
            poll_seconds=5.0,
            retry_seconds=1.0,
            write_status=lambda **kwargs: calls.append(kwargs),
        )

    assert launches == 1
    assert calls[-1]["detail"] == "inference_export_replay_rejected"


def test_stage_runner_refreshes_status_while_child_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class Child:
        pid = 29
        wait_calls = 0

        @classmethod
        def wait(cls, *, timeout: float) -> int:
            assert timeout == 3.0
            cls.wait_calls += 1
            if cls.wait_calls == 1:
                raise subprocess.TimeoutExpired(["posteval"], timeout)
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Child())
    supervisor._run_stage(
        name="full_postevaluation",
        runbook=tmp_path / "posteval.sh",
        project=tmp_path,
        environment={},
        max_attempts=1,
        poll_seconds=3.0,
        retry_seconds=1.0,
        write_status=lambda **kwargs: calls.append(kwargs),
    )

    assert Child.wait_calls == 2
    assert [call["detail"] for call in calls] == [
        "full_postevaluation_running",
        "full_postevaluation_running",
    ]
    assert all(call["child_pid"] == 29 for call in calls)


def test_posttraining_supervisor_runbook_cannot_launch_training() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/"
        "generation_stability_ema_teacher_posttraining_supervisor.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_FULL_LAUNCH_RECEIPT_SHA256=${" in source
    assert "run_generation_stability_posttraining_supervisor.py" in source
    assert "generation_stability_ema_teacher_full_posteval_50k.sh" in source
    assert "generation_stability_ema_teacher_export_inference_artifacts.sh" in source
    assert "generation_stability_ema_teacher_completion_audit.sh" in source
    assert "generation_stability_ema_teacher_full_matched_300k_after_gate.sh" not in source
    assert "train_generation.py" not in source


def test_supervisor_always_replays_posteval_before_accepting_existing_gate() -> None:
    source = Path(supervisor.__file__).read_text(encoding="utf-8")
    stage_call = source.index('name="full_postevaluation"')
    gate_check = source.index(
        'if not final_gate_path.is_file():',
        stage_call,
    )

    assert stage_call < gate_check
    assert 'if not final_gate_path.is_file():\n        _run_stage(' not in source
