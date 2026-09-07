from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import scripts.run_generation_terminal_snr_confirmation as controller
from cofitok.generation.terminal_snr_confirmation_execution import (
    terminal_snr_confirmation_execution_lock_path,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES


ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1/frozen_confirmation_10000"


def _identity(name: str) -> dict[str, object]:
    return {"path": f"/tmp/{name}.json", "bytes": 10, "sha256": "c" * 64}


def _status(root: Path, auth: dict[str, object], launch: dict[str, object]) -> Path:
    control = root.parent / ".confirmation-control"
    control.mkdir(parents=True)
    path = control / "status.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": controller.STATUS_SCHEMA,
                "role": controller.ROLE,
                "status": "failed",
                "authorization": auth,
                "launch_receipt": launch,
                "terminal_status": "hold",
                "generation_advantage_proven": False,
                "frozen_checkpoint_training_allowed": False,
                "large_capacity_readiness_launch_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_allowed": False,
                "export_allowed": False,
                "release_allowed": False,
                "process_signals_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_controller_exposes_all_terminal_screen_sources() -> None:
    argv = [
        "--project-root",
        "/tmp/project",
        "--preparation",
        "/tmp/preparation.json",
        "--expected-preparation-sha256",
        "a" * 64,
        "--authorization",
        "/tmp/authorization.json",
        "--expected-authorization-sha256",
        "b" * 64,
        "--launch-receipt",
        "/tmp/launch.json",
        "--expected-launch-receipt-sha256",
        "c" * 64,
        "--live-snapshot",
        "/tmp/live.json",
        "--expected-live-snapshot-sha256",
        "d" * 64,
        "--output-root",
        ROOT,
        "--real-dir",
        "/tmp/real",
        "--classifier-checkpoint",
        "/tmp/classifier.pth",
        "--cache-root",
        "/tmp/cache",
    ]
    for arm in ARM_NAMES:
        option = arm.replace("_", "-")
        argv.extend(
            [
                f"--{option}-screen-validation",
                f"/tmp/{arm}.json",
                f"--expected-{option}-screen-validation-sha256",
                "e" * 64,
            ]
        )
    parsed = controller.parse_args(argv)
    assert parsed.output_root == Path(ROOT)
    for arm in ARM_NAMES:
        assert getattr(parsed, f"{arm}_screen_validation") == Path(
            f"/tmp/{arm}.json"
        )


def test_resume_root_is_bound_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "confirmation"
    root.mkdir()
    auth = _identity("authorization")
    launch = _identity("launch")
    status = _status(root, auth, launch)
    controller._validate_root_for_resume(
        root,
        status_path=status,
        authorization_identity=auth,
        launch_identity=launch,
    )
    (root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="unexpected entries"):
        controller._validate_root_for_resume(
            root,
            status_path=status,
            authorization_identity=auth,
            launch_identity=launch,
        )


def test_common_environment_is_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setenv("PYTHONPATH", "/untracked")
    env = controller._common_env(Path("/tmp/project"))
    assert "CUBLAS_WORKSPACE_CONFIG" not in env
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "PYTHONHASHSEED" not in env
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert env["PYTHONPATH"].split(os.pathsep) == [
        "/tmp/project",
        "/tmp/project/src",
    ]


def test_frozen_checkpoint_step_comes_from_training_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "checkpoint_step_00010000.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    sidecar_path = tmp_path / "checkpoint_step_00010000.pt.integrity.json"
    sidecar_path.write_text("{}", encoding="utf-8")
    checkpoint_sha = "a" * 64
    screen_validation = {
        "training": {
            "validation": {"completed_steps": 10_000},
            "checkpoint": {
                "path": checkpoint_path.as_posix(),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": checkpoint_sha,
                "integrity_manifest": controller.identity(sidecar_path),
            },
        }
    }
    monkeypatch.setattr(
        controller,
        "verify_training_checkpoint",
        lambda _: {"checkpoint_sha256": checkpoint_sha, "step": 10_000},
    )
    assert controller._verify_frozen_checkpoint(
        screen_validation, arm="control_cofitok"
    ) == checkpoint_path.resolve()


def test_runbook_never_trains_or_builds_authorization() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_terminal_snr_confirmation_10k.sh"
    )
    source = path.read_text(encoding="utf-8")
    assert "train_generation.py" not in source
    assert "build_generation_terminal_snr_confirmation_stage_authorization.py" not in source
    assert "build_generation_terminal_snr_confirmation_execution_authorization.py" not in source
    assert 'RESUME="${TERMINAL_SNR_CONFIRMATION_RESUME:-false}"' in source
    assert "unset CUBLAS_WORKSPACE_CONFIG CUDA_VISIBLE_DEVICES PYTHONHASHSEED" in source
    assert terminal_snr_confirmation_execution_lock_path(ROOT).endswith(
        ".terminal_snr_confirmation_execution.lock"
    )
