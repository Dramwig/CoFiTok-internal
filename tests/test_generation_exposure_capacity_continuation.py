from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

import scripts.run_generation_exposure_capacity_continuation as controller


def _controller_status(root: Path, authorization: dict[str, object], *, status: str = "failed") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "controller_status.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": controller.ROLE,
                "status": status,
                "stage": "training_cofitok",
                "authorization": authorization,
                "terminal_status": "hold",
                "generation_advantage_proven": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_allowed": False,
                "release_allowed": False,
                "process_signals_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_claimed_output_root_is_created_inside_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock(_path: Path):
        events.append("lock_enter")
        yield
        events.append("lock_exit")

    monkeypatch.setattr(controller, "_execution_lock", fake_lock)
    output_root = tmp_path / "candidate"
    with controller._claimed_output_root(output_root, tmp_path / "candidate.lock"):
        assert output_root.is_dir()
        assert events == ["lock_enter"]
    assert events == ["lock_enter", "lock_exit"]


@pytest.mark.skipif(controller.fcntl is None, reason="POSIX flock is required")
def test_execution_lock_rejects_a_competing_lock(tmp_path: Path) -> None:
    lock = tmp_path / "candidate.lock"
    with controller._execution_lock(lock):
        with pytest.raises(RuntimeError, match="another bounded continuation"):
            with controller._execution_lock(lock):
                pass


def test_resume_reopens_only_an_incomplete_root_bound_to_the_authorization(
    tmp_path: Path,
) -> None:
    authorization = {"path": "/tmp/auth.json", "bytes": 12, "sha256": "a" * 64}
    root = tmp_path / "candidate"
    status = _controller_status(root, authorization)

    assert controller._validate_resumable_output_root(root, status, authorization)["status"] == "failed"

    tampered = json.loads(status.read_text(encoding="utf-8"))
    tampered["authorization"]["sha256"] = "b" * 64
    status.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization identity"):
        controller._validate_resumable_output_root(root, status, authorization)


def test_resume_rejects_terminal_or_completed_roots(tmp_path: Path) -> None:
    authorization = {"path": "/tmp/auth.json", "bytes": 12, "sha256": "a" * 64}
    root = tmp_path / "candidate"
    status = _controller_status(root, authorization, status="completed")
    with pytest.raises(ValueError, match="non-terminal"):
        controller._validate_resumable_output_root(root, status, authorization)

    status = _controller_status(root, authorization, status="failed")
    (root / "exposure_capacity_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains a result"):
        controller._validate_resumable_output_root(root, status, authorization)


def test_training_resume_action_requires_explicit_resume_for_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.json").write_text(json.dumps({"checkpoint": "checkpoint.pt"}), encoding="utf-8")
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(controller, "resolve_latest_checkpoint", lambda _run: checkpoint)
    monkeypatch.setattr(controller, "verify_training_checkpoint", lambda _checkpoint: {"step": 100_001})
    execution = {"revision": "b" * 40, "branch": "execution"}

    with pytest.raises(RuntimeError, match="with --resume"):
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=tmp_path / "config.json",
            execution_checkout=execution,
            resume_requested=False,
        )
    assert (
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=tmp_path / "config.json",
            execution_checkout=execution,
            resume_requested=True,
        )
        == "auto"
    )


def test_training_resume_action_skips_only_a_verified_completed_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "latest.json").write_text(json.dumps({"checkpoint": "checkpoint.pt"}), encoding="utf-8")
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(controller, "resolve_latest_checkpoint", lambda _run: checkpoint)
    monkeypatch.setattr(controller, "verify_training_checkpoint", lambda _checkpoint: {"step": controller.TARGET_STEP})
    config = tmp_path / "config.json"
    report = {
        "training_complete": True,
        "completed_steps": controller.TARGET_STEP,
        "target_steps": controller.TARGET_STEP,
        "config_path": config.resolve().as_posix(),
        "git": {"revision": "b" * 40, "branch": "execution", "dirty": False},
    }
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")

    assert (
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=config,
            execution_checkout={"revision": "b" * 40, "branch": "execution"},
            resume_requested=True,
        )
        == "skip"
    )

    report["git"]["revision"] = "c" * 40
    (run / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="execution identity"):
        controller._training_resume_action(
            run,
            source_checkpoint=tmp_path / "source.pt",
            config_path=config,
            execution_checkout={"revision": "b" * 40, "branch": "execution"},
            resume_requested=True,
        )


def test_runbook_resume_is_explicitly_opt_in() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_exposure_capacity_continuation_100k_to_110k.sh"
    )
    source = runbook.read_text(encoding="utf-8")
    assert 'RESUME="${EXPOSURE_CONTINUATION_RESUME:-false}"' in source
    assert "RESUME_ARGS+=(--resume)" in source
