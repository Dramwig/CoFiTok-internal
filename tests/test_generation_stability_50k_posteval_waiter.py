from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.run_generation_stability_50k_posteval_waiter as waiter
from cofitok.reporting import file_sha256
from scripts.run_generation_stability_50k_posteval_waiter import (
    prepare_pair_summary,
    prepare_pair_summary_if_authorized,
    validate_monitor,
    validate_pair_summary,
)


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
MONITOR_NAME = "generation_stability_ema_teacher_matched_50k"


def _monitor(*, status: str = "pass", stage: str = "complete") -> dict:
    return {
        "schema_version": 1,
        "monitor": MONITOR_NAME,
        "status": status,
        "stage": stage,
        "issues": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
    }


def _summary(sources: dict | None = None) -> dict:
    source = {"path": "/tmp/source.json", "bytes": 1, "sha256": "a" * 64}
    return {
        "schema_version": 1,
        "status": "completed",
        "stage": "stability_matched_50k",
        "completed_steps_per_method": 50_000,
        "images_seen_per_method": 3_200_000,
        "effective_batch_size": 64,
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
        },
        "training_pair": {"status": "pass"},
        "formal_300k_authorization_allowed": False,
        "formal_ema_sampling_gate_required": True,
        "sources": sources
        or {
            "cofitok_training": source,
            "dense_training": source,
            "decision_validation": source,
            "config_validation": source,
        },
    }


def _source(path: Path) -> dict:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _summary_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    project = tmp_path / "checkout"
    builder = project / "scripts" / "build_generation_stability_50k_summary.py"
    builder.parent.mkdir(parents=True)
    builder.write_text("# bound builder\n", encoding="utf-8")
    (project / "src").mkdir()
    source_paths = {
        "cofitok_training": tmp_path / "cofitok_training.json",
        "dense_training": tmp_path / "dense_training.json",
        "decision_validation": tmp_path / "decision_validation.json",
        "config_validation": tmp_path / "config_validation.json",
    }
    for index, path in enumerate(source_paths.values()):
        path.write_text(json.dumps({"index": index}), encoding="utf-8")
    return project, builder, source_paths


def test_waiter_accepts_bound_completed_monitor_and_pair() -> None:
    monitor = validate_monitor(
        _monitor(),
        expected_name=MONITOR_NAME,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )
    summary = validate_pair_summary(
        _summary(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )

    assert monitor["status"] == "pass"
    assert monitor["stage"] == "complete"
    assert summary["status"] == "verified"


def test_waiter_rejects_stale_running_monitor() -> None:
    payload = _monitor(status="running", stage="cofitok_training")
    now = datetime.now(timezone.utc)
    payload["updated_at"] = (now - timedelta(seconds=901)).isoformat()

    with pytest.raises(RuntimeError, match="stale"):
        validate_monitor(
            payload,
            expected_name=MONITOR_NAME,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            now=now,
            silence_seconds=900,
        )


def test_waiter_rejects_monitor_revision_mismatch() -> None:
    payload = _monitor()
    payload["git"]["revision"] = "2" * 40

    with pytest.raises(ValueError, match="Git identity"):
        validate_monitor(
            payload,
            expected_name=MONITOR_NAME,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "failed"),
        ("completed_steps_per_method", 49_999),
        ("formal_300k_authorization_allowed", True),
        ("formal_ema_sampling_gate_required", False),
    ],
)
def test_waiter_rejects_incomplete_or_promoting_pair(field: str, value: object) -> None:
    payload = _summary()
    payload[field] = value

    with pytest.raises(ValueError, match="contract mismatch"):
        validate_pair_summary(
            payload,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


def test_waiter_builds_missing_summary_from_exact_bound_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, builder, source_paths = _summary_fixture(tmp_path)
    output = tmp_path / "pair_summary.json"
    captured: dict[str, object] = {}
    monkeypatch.setattr(waiter, "_verify_evaluation_checkout", lambda *args, **kwargs: {})

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        sources = {name: _source(path) for name, path in source_paths.items()}
        output.write_text(json.dumps(_summary(sources)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)
    observation = prepare_pair_summary(
        project=project,
        pair_summary=output,
        pair_summary_builder=builder,
        source_paths=source_paths,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision="2" * 40,
        expected_evaluation_branch="scale/generation-stability-50k-posteval-v3",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1] == str(builder.resolve())
    assert command[command.index("--expected-revision") + 1] == TRAINING_REVISION
    assert command[command.index("--expected-branch") + 1] == TRAINING_BRANCH
    assert command[command.index("--output") + 1] == str(output.resolve())
    assert observation["preparation"] == "rebuilt_after_completed_training"
    assert observation["sources"] == {
        name: _source(path) for name, path in source_paths.items()
    }


def test_waiter_does_not_prepare_summary_before_monitor_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_called(**kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(waiter, "prepare_pair_summary", fail_if_called)
    observation = prepare_pair_summary_if_authorized(
        validate_monitor(
            _monitor(status="running", stage="cofitok_training"),
            expected_name=MONITOR_NAME,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )
    )

    assert observation is None
    assert called is False


def test_waiter_reuses_existing_source_bound_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, builder, source_paths = _summary_fixture(tmp_path)
    output = tmp_path / "pair_summary.json"
    sources = {name: _source(path) for name, path in source_paths.items()}
    output.write_text(json.dumps(_summary(sources)), encoding="utf-8")
    monkeypatch.setattr(
        waiter.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("existing summary must not be rebuilt"),
    )

    observation = prepare_pair_summary(
        project=project,
        pair_summary=output,
        pair_summary_builder=builder,
        source_paths=source_paths,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision="2" * 40,
        expected_evaluation_branch="scale/generation-stability-50k-posteval-v3",
    )

    assert observation["preparation"] == "reused_existing_source_bound_summary"


def test_waiter_fails_closed_when_summary_builder_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, builder, source_paths = _summary_fixture(tmp_path)
    monkeypatch.setattr(waiter, "_verify_evaluation_checkout", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        waiter.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="bound builder rejected source",
        ),
    )

    with pytest.raises(RuntimeError, match="bound builder rejected source"):
        prepare_pair_summary(
            project=project,
            pair_summary=tmp_path / "pair_summary.json",
            pair_summary_builder=builder,
            source_paths=source_paths,
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_evaluation_revision="2" * 40,
            expected_evaluation_branch="scale/generation-stability-50k-posteval-v3",
        )


def test_waiter_runbook_is_non_promoting_and_identity_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/"
        "generation_stability_ema_teacher_50k_posteval_waiter.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_TRAINING_REVISION" in source
    assert "EXPECTED_TRAINING_BRANCH" in source
    assert "EXPECTED_TARGET_REVISION" in source
    assert "EXPECTED_TARGET_BRANCH" in source
    assert "run_generation_stability_50k_posteval_waiter.py" in source
    assert "--pair-summary-builder" in source
    assert "--cofitok-training" in source
    assert "--dense-training" in source
    assert "--decision-validation" in source
    assert "--config-validation" in source
    assert "generation_full_matched_300k_after_gate.sh" not in source
    assert "train_generation.py" not in source
