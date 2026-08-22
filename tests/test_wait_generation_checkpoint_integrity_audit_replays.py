from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "wait_generation_checkpoint_integrity_audit_replays.py"
)
SOURCE_SCOPE = {
    "read_only_checkpoint_verification": True,
    "gpu_required": False,
    "training_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
}


def load_waiter() -> ModuleType:
    name = "cofitok_generation_checkpoint_audit_replay_waiter_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def source_fixture(tmp_path: Path, *, step: int = 90_000) -> dict[str, object]:
    module = load_waiter()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir()
    paths = module.milestone_paths(audit_dir, alias="dense", step=step)
    auditor = tmp_path / "wait_generation_checkpoint_integrity_audit.py"
    auditor.write_text("# auditor\n", encoding="utf-8")
    auditor_identity = identity(auditor)
    return {
        "module": module,
        "run_dir": run_dir,
        "audit_dir": audit_dir,
        "paths": paths,
        "auditor": auditor_identity,
        "step": step,
    }


def waiter_args(tmp_path: Path, *, once: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_steps=(90_000, 95_000, 100_000),
        alias="dense",
        expected_project_revision="a" * 40,
        expected_project_tree="b" * 40,
        expected_project_branch="replay",
        expected_waiter_source_sha256="1" * 64,
        expected_verifier_source_sha256="2" * 64,
        expected_training_revision="c" * 40,
        expected_training_tree="d" * 40,
        expected_training_branch="training",
        expected_auditor_revision="e" * 40,
        expected_auditor_tree="f" * 40,
        expected_auditor_branch="",
        expected_auditor_source_sha256="3" * 64,
        expected_dataset_sha256="4" * 64,
        expected_runtime_sha256="5" * 64,
        effective_batch=64,
        status_output=tmp_path / "replay_waiter_status.json",
        timeout_seconds=60.0,
        poll_seconds=0.01,
        source_status_silence_seconds=1_800.0,
        once=once,
    )


def test_checkpoint_steps_and_paths_are_canonical(tmp_path: Path) -> None:
    module = load_waiter()
    assert module.parse_checkpoint_steps([100_000, 90_000, 95_000]) == (
        90_000,
        95_000,
        100_000,
    )
    with pytest.raises(ValueError, match="unique"):
        module.parse_checkpoint_steps([90_000, 90_000])
    paths = module.milestone_paths(tmp_path, alias="dense", step=90_000)
    assert paths["audit_report"].name == (
        "dense_checkpoint_step_00090000_physical_integrity_audit.json"
    )
    assert paths["replay_output"].name.endswith(
        "physical_integrity_audit_replay.json"
    )


def test_source_observation_waits_then_accepts_bound_pass(tmp_path: Path) -> None:
    data = source_fixture(tmp_path)
    module = data["module"]
    verifier = module.load_verifier()
    waiting = module.observe_source_milestone(
        verifier,
        paths=data["paths"],
        run_dir=data["run_dir"],
        step=data["step"],
        auditor_source=data["auditor"],
        silence_seconds=1_800.0,
    )
    assert waiting["state"] == "waiting"

    checkpoint_sha = "a" * 64
    audit = {
        "checkpoint": {
            "step": data["step"],
            "payload": {"sha256": checkpoint_sha},
        }
    }
    write_json(data["paths"]["audit_report"], audit)
    status = {
        "schema_version": 1,
        "role": "generation_checkpoint_physical_integrity_milestone_waiter",
        "status": "pass",
        "detail": "checkpoint_physical_integrity_verified",
        "updated_at": module.utc_now(),
        "checkpoint_step": data["step"],
        "run_dir": data["run_dir"].resolve().as_posix(),
        "audit_output": data["paths"]["audit_report"].resolve().as_posix(),
        "audit_output_identity": identity(data["paths"]["audit_report"]),
        "waiter_source": data["auditor"],
        "scope": SOURCE_SCOPE,
    }
    write_json(data["paths"]["waiter_status"], status)
    ready = module.observe_source_milestone(
        verifier,
        paths=data["paths"],
        run_dir=data["run_dir"],
        step=data["step"],
        auditor_source=data["auditor"],
        silence_seconds=1_800.0,
    )
    assert ready["state"] == "ready"
    assert ready["checkpoint_sha256"] == checkpoint_sha


def test_source_observation_rejects_failed_waiter(tmp_path: Path) -> None:
    data = source_fixture(tmp_path)
    module = data["module"]
    verifier = module.load_verifier()
    write_json(
        data["paths"]["waiter_status"],
        {
            "schema_version": 1,
            "role": "generation_checkpoint_physical_integrity_milestone_waiter",
            "status": "failed",
            "detail": "source failure",
            "updated_at": module.utc_now(),
            "checkpoint_step": data["step"],
            "run_dir": data["run_dir"].resolve().as_posix(),
            "audit_output": data["paths"]["audit_report"].resolve().as_posix(),
            "waiter_source": data["auditor"],
            "scope": SOURCE_SCOPE,
        },
    )
    with pytest.raises(RuntimeError, match="source waiter failed"):
        module.observe_source_milestone(
            verifier,
            paths=data["paths"],
            run_dir=data["run_dir"],
            step=data["step"],
            auditor_source=data["auditor"],
            silence_seconds=1_800.0,
        )


def test_source_observation_reports_live_waiting_status(tmp_path: Path) -> None:
    data = source_fixture(tmp_path)
    module = data["module"]
    verifier = module.load_verifier()
    write_json(
        data["paths"]["waiter_status"],
        {
            "schema_version": 1,
            "role": "generation_checkpoint_physical_integrity_milestone_waiter",
            "status": "waiting",
            "detail": "checkpoint_missing",
            "updated_at": module.utc_now(),
            "checkpoint_step": data["step"],
            "run_dir": data["run_dir"].resolve().as_posix(),
            "audit_output": data["paths"]["audit_report"].resolve().as_posix(),
            "waiter_source": data["auditor"],
            "scope": SOURCE_SCOPE,
        },
    )
    waiting = module.observe_source_milestone(
        verifier,
        paths=data["paths"],
        run_dir=data["run_dir"],
        step=data["step"],
        auditor_source=data["auditor"],
        silence_seconds=1_800.0,
    )
    assert waiting["state"] == "waiting"
    assert waiting["detail"] == "checkpoint_missing"
    assert waiting["source_status"]["sha256"] == identity(
        data["paths"]["waiter_status"]
    )["sha256"]


def test_source_observation_rejects_stale_waiting_status(tmp_path: Path) -> None:
    data = source_fixture(tmp_path)
    module = data["module"]
    verifier = module.load_verifier()
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    write_json(
        data["paths"]["waiter_status"],
        {
            "schema_version": 1,
            "role": "generation_checkpoint_physical_integrity_milestone_waiter",
            "status": "waiting",
            "detail": "checkpoint_missing",
            "updated_at": stale,
            "checkpoint_step": data["step"],
            "run_dir": data["run_dir"].resolve().as_posix(),
            "audit_output": data["paths"]["audit_report"].resolve().as_posix(),
            "waiter_source": data["auditor"],
            "scope": SOURCE_SCOPE,
        },
    )
    with pytest.raises(RuntimeError, match="source waiter is stale"):
        module.observe_source_milestone(
            verifier,
            paths=data["paths"],
            run_dir=data["run_dir"],
            step=data["step"],
            auditor_source=data["auditor"],
            silence_seconds=60.0,
        )


def test_verifier_command_preserves_empty_detached_auditor_branch(tmp_path: Path) -> None:
    module = load_waiter()
    command = module.build_verifier_command(
        python="python",
        verifier_source=tmp_path / "verify.py",
        source={
            "source_audit": {"path": "/audit.json", "sha256": "a" * 64},
            "source_status": {"path": "/status.json", "sha256": "b" * 64},
            "checkpoint_step": 90_000,
            "checkpoint_sha256": "c" * 64,
        },
        replay_output=tmp_path / "replay.json",
        auditor_source_path=tmp_path / "auditor.py",
        auditor_source_sha256="d" * 64,
        auditor_checkout=tmp_path / "auditor",
        auditor_git={"revision": "e" * 40, "tree": "f" * 40, "branch": ""},
        training_checkout=tmp_path / "training",
        training_git={
            "revision": "1" * 40,
            "tree": "2" * 40,
            "branch": "training",
        },
        dataset_sha256="3" * 64,
        runtime_sha256="4" * 64,
        effective_batch=64,
        project_git={
            "revision": "5" * 40,
            "tree": "6" * 40,
            "branch": "replay",
        },
        verifier_source_sha256="7" * 64,
    )
    index = command.index("--expected-auditor-branch")
    assert command[index + 1] == ""
    assert command[-2:] == ["--output", str(tmp_path / "replay.json")]


def test_immutable_receipt_view_ignores_live_append_observations() -> None:
    module = load_waiter()
    base = {
        "schema_version": 1,
        "role": module.REPLAY_ROLE,
        "status": "pass",
        "verifier_git": {"revision": "a"},
        "verifier_source": {"sha256": "b"},
        "audit_report": {"sha256": "c"},
        "waiter_status": {"sha256": "d"},
        "auditor_source": {"sha256": "e"},
        "auditor_checkout": {"revision": "f"},
        "training_checkout": {"revision": "1"},
        "checkpoint": {"step": 90_000},
        "latest_pointer_replay": {
            "historical_identity": {"sha256": "2"},
            "historical_content": {"step": 90_000},
            "historical_byte_identity_reconstructed": True,
            "current_content": {"step": 90_000},
        },
        "metrics_replay": {
            "audit_time_prefix": {"sha256": "3"},
            "audit_time_prefix_byte_exact": True,
            "target_row": {"step": 90_000},
            "current_observed_prefix": {"last_step": 90_100},
        },
        "checks": {"checkpoint_payload_physically_rehashed": True},
        "scope": {"full_300k_launch_allowed": False},
        "verified_at": "first",
    }
    advanced = json.loads(json.dumps(base))
    advanced["verified_at"] = "second"
    advanced["latest_pointer_replay"]["current_content"] = {"step": 95_000}
    advanced["metrics_replay"]["current_observed_prefix"] = {"last_step": 95_200}
    assert module.immutable_receipt_view(base) == module.immutable_receipt_view(
        advanced
    )


def test_existing_replay_restart_revalidates_immutable_evidence(tmp_path: Path) -> None:
    module = load_waiter()
    receipt = {
        "schema_version": 1,
        "role": module.REPLAY_ROLE,
        "status": "pass",
        "verifier_git": {"revision": "a" * 40, "tree": "b" * 40, "branch": "replay"},
        "verifier_source": {"sha256": "1" * 64},
        "audit_report": {"sha256": "2" * 64},
        "waiter_status": {"sha256": "3" * 64},
        "auditor_source": {"sha256": "4" * 64},
        "auditor_checkout": {"revision": "c" * 40, "branch": ""},
        "training_checkout": {"revision": "d" * 40, "branch": "training"},
        "checkpoint": {
            "step": 90_000,
            "payload": {"sha256": "5" * 64},
        },
        "latest_pointer_replay": {
            "historical_identity": {"sha256": "6" * 64},
            "historical_content": {"step": 90_000},
            "historical_byte_identity_reconstructed": True,
            "current_content": {"step": 90_000},
        },
        "metrics_replay": {
            "audit_time_prefix": {"sha256": "7" * 64},
            "audit_time_prefix_byte_exact": True,
            "target_row": {"step": 90_000},
            "current_observed_prefix": {"last_step": 90_000},
        },
        "checks": {"checkpoint_payload_physically_rehashed": True},
        "scope": {"full_300k_launch_allowed": False},
        "verified_at": "original",
    }
    replay_output = tmp_path / "replay.json"
    write_json(replay_output, receipt)

    class FakeVerifier:
        def stable_json(self, path: Path, *, name: str):
            assert path == replay_output
            assert "replay receipt" in name
            return receipt, identity(replay_output)

        def verify_replay(self, **kwargs):
            assert kwargs["expected_auditor_branch"] == ""
            fresh = json.loads(json.dumps(receipt))
            fresh["verified_at"] = "restart"
            fresh["latest_pointer_replay"]["current_content"] = {
                "step": 95_000
            }
            fresh["metrics_replay"]["current_observed_prefix"] = {
                "last_step": 95_000
            }
            return fresh

    source = {
        "source_audit": {"path": "/audit.json", "sha256": "2" * 64},
        "source_status": {"path": "/status.json", "sha256": "3" * 64},
        "checkpoint_step": 90_000,
        "checkpoint_sha256": "5" * 64,
    }
    result = module.validate_existing_replay(
        FakeVerifier(),
        output=replay_output,
        source=source,
        auditor_source_path=tmp_path / "auditor.py",
        auditor_source_sha256="4" * 64,
        auditor_checkout=tmp_path / "auditor",
        auditor_git={"revision": "c" * 40, "tree": "8" * 40, "branch": ""},
        training_checkout=tmp_path / "training",
        training_git={
            "revision": "d" * 40,
            "tree": "9" * 40,
            "branch": "training",
        },
        dataset_sha256="a" * 64,
        runtime_sha256="b" * 64,
        effective_batch=64,
        project_git=receipt["verifier_git"],
    )
    assert result == identity(replay_output)


def test_once_returns_waiting_without_sleep_or_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_waiter()
    args = waiter_args(tmp_path, once=True)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = {
        "audit_dir": audit_dir,
        "run_dir": run_dir,
        "auditor_source": {"sha256": "3" * 64},
        "project_git": {
            "revision": args.expected_project_revision,
            "tree": args.expected_project_tree,
            "branch": args.expected_project_branch,
        },
        "waiter_source": {"sha256": args.expected_waiter_source_sha256},
        "verifier_source": {"sha256": args.expected_verifier_source_sha256},
    }
    monkeypatch.setattr(module, "static_context", lambda *_: context)
    monkeypatch.setattr(
        module,
        "observe_source_milestone",
        lambda *_, step, **__: {
            "state": "waiting",
            "detail": "checkpoint_missing",
            "checkpoint_step": step,
        },
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda *_: pytest.fail("--once must not sleep"),
    )
    monkeypatch.setattr(
        module,
        "execute_replay",
        lambda *_, **__: pytest.fail("waiting --once must not run a replay"),
    )
    assert module.run_locked(args, SimpleNamespace()) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "waiting"
    assert status["detail"] == "waiting_for_source_audit_step_00090000"
    assert status["completed_count"] == 0
    assert status["scope"] == module.READ_ONLY_SCOPE


def test_run_waiter_rejects_second_instance_for_same_status(tmp_path: Path) -> None:
    module = load_waiter()
    args = SimpleNamespace(status_output=tmp_path / "waiter_status.json")
    with module.exclusive_output_lock(args.status_output, role="test_holder"):
        assert module.run_waiter(args, verifier=SimpleNamespace()) == 75


def test_waiting_status_is_permanently_non_authorizing() -> None:
    module = load_waiter()
    args = SimpleNamespace(
        checkpoint_steps=(90_000, 95_000, 100_000),
        alias="dense",
        expected_project_revision="a",
        expected_project_tree="b",
        expected_project_branch="replay",
        expected_training_revision="c",
        expected_training_tree="d",
        expected_training_branch="training",
        expected_auditor_revision="e",
        expected_auditor_tree="f",
        expected_auditor_branch="",
        expected_dataset_sha256="1" * 64,
        expected_runtime_sha256="2" * 64,
        effective_batch=64,
    )
    status = module.status_payload(
        status="waiting",
        detail="waiting_for_source_audit_step_00090000",
        args=args,
        started_at=module.utc_now(),
    )
    assert status["scope"] == module.READ_ONLY_SCOPE
    assert status["expected"]["full_300k_launch_allowed"] is False
    assert status["expected"]["release_authorization_allowed"] is False
