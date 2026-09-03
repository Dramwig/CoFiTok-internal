from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

import scripts.run_generation_capacity_confirmation as controller
from cofitok.generation.capacity_confirmation_execution import (
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    capacity_confirmation_execution_lock_path,
    validate_capacity_confirmation_live_snapshot,
)
from cofitok.generation.capacity_screen import ARM_NAMES


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}
ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1/confirmation_10000"
LOCK = capacity_confirmation_execution_lock_path(ROOT)


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


def test_confirmation_controller_exposes_all_frozen_screen_sources() -> None:
    argv = [
        "--project-root", "/tmp/project",
        "--preparation", "/tmp/preparation.json",
        "--expected-preparation-sha256", "a" * 64,
        "--authorization", "/tmp/authorization.json",
        "--expected-authorization-sha256", "b" * 64,
        "--launch-receipt", "/tmp/launch.json",
        "--expected-launch-receipt-sha256", "c" * 64,
        "--live-snapshot", "/tmp/live.json",
        "--expected-live-snapshot-sha256", "d" * 64,
        "--output-root", ROOT,
        "--real-dir", "/tmp/real",
        "--classifier-checkpoint", "/tmp/classifier.pth",
        "--cache-root", "/tmp/cache",
    ]
    for arm in ARM_NAMES:
        option = arm.replace("_", "-")
        argv.extend(
            [
                f"--{option}-screen-validation", f"/tmp/{arm}.json",
                f"--expected-{option}-screen-validation-sha256", "e" * 64,
            ]
        )
    parsed = controller.parse_args(argv)
    assert parsed.output_root == Path(ROOT)
    for arm in ARM_NAMES:
        assert getattr(parsed, f"{arm}_screen_validation") == Path(f"/tmp/{arm}.json")


def test_confirmation_resume_root_is_bound_and_fail_closed(tmp_path: Path) -> None:
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


def test_confirmation_live_snapshot_binds_checkout_and_exact_lock() -> None:
    report = {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": copy.deepcopy(GIT),
        "gpu_inventory": [
            {
                "memory_used_mib": 0,
                "memory_total_mib": 97_887,
                "utilization_percent": 0,
            }
        ],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root": ROOT,
        "execution_lock": LOCK,
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 500 * 1024**3,
        "runtime_environment_sha256": "d" * 64,
        "dataset_identity_sha256": "e" * 64,
        "captured_at": "2026-09-03T00:00:00+00:00",
    }
    assert validate_capacity_confirmation_live_snapshot(
        report,
        expected_output_root=ROOT,
        expected_execution_checkout=GIT,
        expected_execution_lock=LOCK,
    ) == report

    report["execution_lock"] = f"{LOCK}.other"
    with pytest.raises(ValueError, match="live prelaunch snapshot differs"):
        validate_capacity_confirmation_live_snapshot(
            report,
            expected_output_root=ROOT,
            expected_execution_checkout=GIT,
            expected_execution_lock=LOCK,
        )


def test_confirmation_runbook_never_invokes_training_or_builds_authorization() -> None:
    path = Path(__file__).resolve().parents[1] / "artifacts/runbooks/generation_capacity_confirmation_10k.sh"
    source = path.read_text(encoding="utf-8")
    assert "train_generation.py" not in source
    assert "build_generation_capacity_confirmation_stage_authorization.py" not in source
    assert "build_generation_capacity_confirmation_execution_authorization.py" not in source
    assert 'RESUME="${CAPACITY_CONFIRMATION_RESUME:-false}"' in source
