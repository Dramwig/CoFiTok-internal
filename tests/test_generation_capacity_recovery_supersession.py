from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cofitok.generation_recovery_supersession import (
    CONTRACT_ROLE,
    inspect_recovery_supersession,
    load_recovery_supersession_contract,
)


NOW = datetime(2026, 8, 17, 12, 30, tzinfo=timezone.utc)
REVISION = "c" * 40
TREE = "6" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
DATASET = "d" * 64
RUNTIME = "e" * 64
ROOT = Path(__file__).resolve().parents[1]
LIVE_CONTRACT = (
    ROOT
    / "configs/generation/diagnostics/"
    "quality_bridge_recovery_supersession_20260817.json"
)
LIVE_CONTRACT_SHA256 = (
    "28318ff89c009be65899117bbb9d45f533bc116f63b6bd335e21463818d5f609"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reference(path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fixture(tmp_path: Path, *, failure_detail: str | None = None) -> dict:
    output_root = (tmp_path / "quality").resolve()
    run_dir = (output_root / "cofitok").resolve()
    failure_path = tmp_path / "recovery_supervisor_status.json"
    successor_path = tmp_path / "recovery_supervisor_v2_status.json"
    receipt_path = tmp_path / "recovery_supervisor_v2_deployment_receipt.json"
    execution_path = output_root / "reports/execution_status.json"
    pair_path = output_root / "pair_monitor.json"
    watchdog_50_path = run_dir / "training_watchdog_step_00050000.json"
    watchdog_100_path = run_dir / "training_watchdog_step_00100000.json"
    manifest_path = run_dir / "run_manifest.json"
    reconciliation_path = run_dir / "metrics_resume_reconciliation.json"
    checkpoint_path = output_root / "reports/checkpoint_audit.json"

    _write(
        failure_path,
        {
            "schema_version": 1,
            "role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor",
            "status": "failed",
            "detail": failure_detail
            or "nonretryable_recovery_controller_exit_1:exit_1_is_not_retryable",
            "supervisor_pid": 1,
            "updated_at": "2026-08-17T04:54:28+00:00",
        },
    )
    _write(
        receipt_path,
        {
            "schema_version": 1,
            "role": "cofitok_quality_bridge_ipc_recovery_v2_deployment_receipt",
            "status": "pass",
            "scope": {
                "full_300k_launch_allowed": False,
                "full_training_launch_allowed": False,
                "quality_bridge_exact_resume_allowed": True,
                "release_authorization_allowed": False,
            },
            "training": {
                "revision": REVISION,
                "tree": TREE,
                "branch": BRANCH,
                "output_root": output_root.as_posix(),
            },
            "deployment": {
                "deployed_source_path": "/tmp/recovery_supervisor_v2.py",
                "deployed_source_bytes": 10,
                "deployed_source_sha256": "a" * 64,
            },
        },
    )
    _write(
        successor_path,
        {
            "schema_version": 1,
            "role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor_v2",
            "status": "observing",
            "detail": "existing_quality_bridge_execution_is_active",
            "supervisor_pid": 101,
            "updated_at": NOW.isoformat(),
            "attempt": 0,
            "git": {
                "revision": REVISION,
                "tree": TREE,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "scope": {
                "full_300k_launch_allowed": False,
                "full_training_launch_allowed": False,
                "release_authorization_allowed": False,
            },
            "identities": {
                "recovery_supervisor": {
                    "path": "/tmp/recovery_supervisor_v2.py",
                    "bytes": 10,
                    "sha256": "a" * 64,
                }
            },
        },
    )
    _write(
        execution_path,
        {
            "schema_version": 1,
            "role": "stability_full_data_quality_bridge_execution",
            "status": "running",
            "pid": 102,
            "git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "quality_bridge_only": True,
            "full_300k_launch_allowed": False,
            "full_training_launch_allowed": False,
            "report_is_promotion_gate": False,
        },
    )
    _write(
        pair_path,
        {
            "schema_version": 2,
            "monitor": "generation_stability_full_data_quality_bridge_100k",
            "status": "running",
            "stage": "cofitok_training",
            "updated_at": NOW.isoformat(),
            "issues": [],
            "git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "runs": {
                "cofitok": {
                    "complete": False,
                    "last_step": 25_100,
                    "health_issues": [],
                    "run_manifest": {
                        "status": "verified",
                        "git_revision": REVISION,
                        "dataset_identity_sha256": DATASET,
                        "runtime_environment_sha256": RUNTIME,
                    },
                },
                "dense_identity": {
                    "complete": False,
                    "last_step": 0,
                    "health_issues": [],
                },
            },
        },
    )
    _write(
        watchdog_50_path,
        {
            "schema_version": 1,
            "role": "generation_training_watchdog",
            "status": "running",
            "pid": 103,
            "child_pid": 104,
            "monitor_pid": 105,
            "monitor_process_alive": True,
            "child_exit_code": None,
            "watchdog_exit_code": None,
            "reason": "child_and_monitor_active",
            "updated_at": NOW.isoformat(),
            "monitor_report_path": pair_path.resolve().as_posix(),
            "monitor": {
                "status": "running",
                "stage": "cofitok_training",
                "issues": [],
            },
            "command": [
                "python",
                "scripts/train_generation.py",
                "--output-dir",
                run_dir.as_posix(),
                "--resume",
                "auto",
            ],
        },
    )
    _write(
        reconciliation_path,
        {
            "schema_version": 1,
            "status": "reconciled",
            "resume_step": 20_000,
            "orphaned_rows": 4,
            "orphan_sha256": "f" * 64,
        },
    )
    _write(
        manifest_path,
        {
            "git": {
                "revision": REVISION,
                "branch": BRANCH,
                "dirty": False,
            },
            "output_dir": run_dir.as_posix(),
            "resume": f"{run_dir.as_posix()}/checkpoint_step_00020000.pt",
            "dataset_provenance": {"identity_sha256": DATASET},
            "runtime_environment_sha256": RUNTIME,
            "metrics_resume_reconciliation": {
                "status": "reconciled",
                "resume_step": 20_000,
                "report": reconciliation_path.resolve().as_posix(),
                "orphan_sha256": "f" * 64,
            },
        },
    )
    _write(
        checkpoint_path,
        {
            "schema_version": 1,
            "role": "generation_checkpoint_physical_integrity_milestone_audit",
            "status": "pass",
            "run_dir": run_dir.as_posix(),
            "training_checkout": {
                "revision": REVISION,
                "tree": TREE,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "checkpoint": {
                "step": 25_000,
                "physical_sha256_verified": True,
                "integrity": {
                    "step": 25_000,
                    "git_revision": REVISION,
                    "git_branch": BRANCH,
                    "git_dirty": False,
                    "dataset_identity_sha256": DATASET,
                    "runtime_environment_sha256": RUNTIME,
                },
            },
            "latest_pointer": {"exact_target_binding": True},
            "metrics": {
                "strictly_increasing": True,
                "samples_seen_binding_verified": True,
                "target_row": {"samples_seen": 25_000 * 64},
            },
        },
    )

    contract_path = tmp_path / "supersession_contract.json"
    _write(
        contract_path,
        {
            "schema_version": 1,
            "role": CONTRACT_ROLE,
            "superseded_failure": {
                "role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor",
                "detail_prefix": "nonretryable_recovery_controller_exit_1",
                "source": _reference(failure_path),
            },
            "successor": {
                "role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor_v2",
                "status_path": successor_path.resolve().as_posix(),
                "deployment_receipt": _reference(receipt_path),
            },
            "active_chain": {
                "execution_status_path": execution_path.resolve().as_posix(),
                "pair_monitor_path": pair_path.resolve().as_posix(),
                "watchdog_statuses": [
                    {
                        "path": watchdog_50_path.resolve().as_posix(),
                        "target_step": 50_000,
                    },
                    {
                        "path": watchdog_100_path.resolve().as_posix(),
                        "target_step": 100_000,
                    },
                ],
            },
            "exact_resume": {
                "run_manifest": _reference(manifest_path),
                "reconciliation": _reference(reconciliation_path),
                "resume_step": 20_000,
                "checkpoint_audit": _reference(checkpoint_path),
                "checkpoint_step": 25_000,
            },
            "expected_training": {
                "revision": REVISION,
                "tree": TREE,
                "branch": BRANCH,
                "output_root": output_root.as_posix(),
                "run_dir": run_dir.as_posix(),
                "dataset_identity_sha256": DATASET,
                "runtime_environment_sha256": RUNTIME,
                "effective_batch": 64,
            },
        },
    )
    failed_stage = {
        "index": 0,
        "name": "quality_bridge_100k_recovery",
        "expected_role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor",
        "blocking": True,
        "status_path": failure_path.resolve().as_posix(),
        "identity": {
            **_reference(failure_path),
            "bytes": failure_path.stat().st_size,
        },
        "role": "cofitok_quality_bridge_100k_bounded_recovery_supervisor",
        "status": "failed",
        "detail": failure_detail
        or "nonretryable_recovery_controller_exit_1:exit_1_is_not_retryable",
        "classification": "failure",
        "health": "failed",
        "pid": 1,
        "process_alive": False,
        "updated_at": "2026-08-17T04:54:28+00:00",
        "age_seconds": 1.0,
        "issues": [],
    }
    return {
        "contract_path": contract_path,
        "failed_stage": failed_stage,
        "receipt_path": receipt_path,
        "pair_path": pair_path,
        "watchdog_50_path": watchdog_50_path,
        "watchdog_100_path": watchdog_100_path,
        "checkpoint_path": checkpoint_path,
    }


def _inspect(fixture: dict, *, alive: set[int] | None = None) -> dict:
    contract = load_recovery_supersession_contract(fixture["contract_path"])
    live = {101, 102, 103, 104, 105} if alive is None else alive
    return inspect_recovery_supersession(
        fixture["failed_stage"],
        contract=contract,
        now=NOW,
        stale_seconds=240.0,
        process_exists=lambda pid: pid in live,
    )


def test_checked_in_recovery_supersession_contract_pins_live_evidence() -> None:
    contract = load_recovery_supersession_contract(LIVE_CONTRACT)
    assert contract["identity"]["sha256"] == LIVE_CONTRACT_SHA256
    assert contract["superseded_failure"]["source"]["sha256"] == (
        "035037c1e1c4cca78ebd0429c85218dce5ecf08435523ae3d89b6ee22e9e9a88"
    )
    assert contract["successor"]["deployment_receipt"]["sha256"] == (
        "c20cf6193db929adbecd2dbe2822deb907f6aedfb29095175317524b1a6cc075"
    )
    assert contract["exact_resume"]["run_manifest"]["sha256"] == (
        "c81bc1f1b75b1e09f257ee8177b144e12832f305ccdefda8c0715f8e145a061b"
    )
    assert contract["exact_resume"]["reconciliation"]["sha256"] == (
        "dfe268c1defbaaa424cb93df108deb1421080f9568c47b02f1607568d6f57aca"
    )
    assert contract["exact_resume"]["checkpoint_audit"]["sha256"] == (
        "3b89622a286d5ef6b7244daa47a1866f272da447b564887ad477c53138a6d1f9"
    )
    assert contract["exact_resume"]["resume_step"] == 20_000
    assert contract["exact_resume"]["checkpoint_step"] == 25_000
    assert [
        row["target_step"]
        for row in contract["active_chain"]["watchdog_statuses"]
    ] == [50_000, 100_000]
    assert contract["expected_training"]["dataset_identity_sha256"] == (
        "6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659"
    )
    assert contract["expected_training"]["runtime_environment_sha256"] == (
        "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"
    )


def test_source_bound_exact_resume_supersedes_the_old_recovery_failure(
    tmp_path: Path,
) -> None:
    stage = _inspect(_fixture(tmp_path))
    assert stage["status"] == "running"
    assert stage["classification"] == "active"
    assert stage["health"] == "running"
    assert stage["issues"] == []
    assert stage["supersession"]["status"] == "pass"
    assert stage["supersession"]["resume_step"] == 20_000
    assert stage["supersession"]["trusted_checkpoint_step"] == 25_000
    assert stage["supersession"]["claim_boundary"]["training_launch_allowed"] is False


def test_recovery_supersession_rejects_a_different_failure_signature(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, failure_detail="another_failure")
    with pytest.raises(ValueError, match="failure identity differs"):
        _inspect(fixture)


def test_recovery_supersession_rejects_tampered_immutable_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = json.loads(fixture["receipt_path"].read_text(encoding="utf-8"))
    receipt["scope"]["full_training_launch_allowed"] = True
    _write(fixture["receipt_path"], receipt)
    with pytest.raises(ValueError, match="SHA256 differs"):
        _inspect(fixture)


def test_recovery_supersession_rejects_an_unhealthy_live_pair_monitor(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["issues"] = ["stalled"]
    _write(fixture["pair_path"], pair)
    with pytest.raises(ValueError, match="pair monitor is unhealthy"):
        _inspect(fixture)


def test_recovery_supersession_requires_the_active_trainer_and_watchdog(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="trainer/watchdog binding differs"):
        _inspect(fixture, alive={101, 102, 103, 105})


def _complete_watchdog(path: Path) -> None:
    watchdog = json.loads(path.read_text(encoding="utf-8"))
    watchdog.update(
        {
            "status": "completed",
            "reason": "child_completed",
            "child_exit_code": 0,
            "watchdog_exit_code": 0,
        }
    )
    _write(path, watchdog)


def test_recovery_supersession_accepts_the_50k_milestone_handoff(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["runs"]["cofitok"]["last_step"] = 50_000
    _write(fixture["pair_path"], pair)
    _complete_watchdog(fixture["watchdog_50_path"])

    stage = _inspect(fixture, alive={101, 102})
    assert stage["supersession"]["live_cofitok_step"] == 50_000
    assert stage["supersession"]["watchdog_target_step"] == 50_000


def test_recovery_supersession_switches_to_the_100k_watchdog(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["runs"]["cofitok"]["last_step"] = 50_100
    _write(fixture["pair_path"], pair)
    _complete_watchdog(fixture["watchdog_50_path"])
    watchdog = json.loads(
        fixture["watchdog_50_path"].read_text(encoding="utf-8")
    )
    watchdog.update(
        {
            "status": "running",
            "reason": "child_and_monitor_active",
            "pid": 106,
            "child_pid": 107,
            "monitor_pid": 105,
            "child_exit_code": None,
            "watchdog_exit_code": None,
        }
    )
    _write(fixture["watchdog_100_path"], watchdog)

    stage = _inspect(fixture, alive={101, 102, 105, 106, 107})
    assert stage["supersession"]["live_cofitok_step"] == 50_100
    assert stage["supersession"]["watchdog_target_step"] == 100_000


def test_recovery_supersession_rejects_a_missing_next_watchdog(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["runs"]["cofitok"]["last_step"] = 50_100
    _write(fixture["pair_path"], pair)
    _complete_watchdog(fixture["watchdog_50_path"])

    with pytest.raises(ValueError, match="next watchdog status is missing"):
        _inspect(fixture, alive={101, 102})


def test_recovery_supersession_contract_requires_increasing_watchdogs(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    contract = json.loads(
        fixture["contract_path"].read_text(encoding="utf-8")
    )
    contract["active_chain"]["watchdog_statuses"][1]["target_step"] = 50_000
    _write(fixture["contract_path"], contract)

    with pytest.raises(ValueError, match="target steps are not increasing"):
        load_recovery_supersession_contract(fixture["contract_path"])


def test_recovery_supersession_requires_the_terminal_100k_watchdog(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["runs"]["cofitok"].update({"complete": True, "last_step": 100_000})
    _write(fixture["pair_path"], pair)
    _complete_watchdog(fixture["watchdog_50_path"])
    watchdog = json.loads(
        fixture["watchdog_50_path"].read_text(encoding="utf-8")
    )
    _write(fixture["watchdog_100_path"], watchdog)

    stage = _inspect(fixture, alive={101, 102})
    assert stage["supersession"]["live_cofitok_step"] == 100_000
    assert stage["supersession"]["watchdog_target_step"] == 100_000


def test_recovery_supersession_rejects_a_tampered_checkpoint_audit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = json.loads(
        fixture["checkpoint_path"].read_text(encoding="utf-8")
    )
    checkpoint["checkpoint"]["integrity"]["git_revision"] = "0" * 40
    _write(fixture["checkpoint_path"], checkpoint)
    with pytest.raises(ValueError, match="SHA256 differs"):
        _inspect(fixture)
