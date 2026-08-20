from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cofitok.environment import runtime_environment_sha256
from cofitok.generation_recovery_supersession import (
    CONTRACT_ROLE,
    MUTABLE_SEMANTIC_POLICY,
    inspect_recovery_supersession,
    load_recovery_supersession_contract,
)


NOW = datetime(2026, 8, 17, 12, 30, tzinfo=timezone.utc)
REVISION = "c" * 40
TREE = "6" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
DATASET = "d" * 64
RUNTIME_ENVIRONMENT = {
    "schema_version": 1,
    "python": {"implementation": "CPython", "version": "3.10.20"},
}
RUNTIME = runtime_environment_sha256(RUNTIME_ENVIRONMENT)
ROOT = Path(__file__).resolve().parents[1]
LIVE_CONTRACT = (
    ROOT
    / "configs/generation/diagnostics/"
    "quality_bridge_recovery_supersession_20260817.json"
)
LIVE_CONTRACT_V2 = (
    ROOT
    / "configs/generation/diagnostics/"
    "quality_bridge_recovery_supersession_20260820_v2.json"
)
LIVE_CONTRACT_V2_SHA256 = (
    "d42b3bca92087062eb8e2999ac9f1cd3be729bf72b9e3e4cb823d5544c67e379"
)
LIVE_CONTRACT_SHA256 = (
    "28318ff89c009be65899117bbb9d45f533bc116f63b6bd335e21463818d5f609"
)


def _tracked_blob_sha256(path: Path, *, revision: str = "HEAD") -> str:
    relative_path = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


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
    original_orphan_payload = '{"step": 20001}\n'
    original_orphan_sha = hashlib.sha256(
        original_orphan_payload.encode("utf-8")
    ).hexdigest()
    reconciliation_path = run_dir / (
        "metrics_resume_reconciliation_00020000_"
        f"{original_orphan_sha[:12]}.json"
    )
    original_orphan_path = run_dir / (
        "train_metrics_orphaned_at_resume_00020000_"
        f"{original_orphan_sha[:12]}.jsonl"
    )
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
    original_orphan_path.parent.mkdir(parents=True, exist_ok=True)
    original_orphan_path.write_bytes(original_orphan_payload.encode("utf-8"))
    _write(
        reconciliation_path,
        {
            "schema_version": 1,
            "status": "reconciled",
            "resume_step": 20_000,
            "metrics": (run_dir / "train_metrics.jsonl").as_posix(),
            "retained_rows": 401,
            "orphaned_rows": 4,
            "orphan_archive": original_orphan_path.as_posix(),
            "orphan_sha256": original_orphan_sha,
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
                "orphan_sha256": original_orphan_sha,
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
        "manifest_path": manifest_path,
        "reconciliation_path": reconciliation_path,
        "original_orphan_path": original_orphan_path,
        "watchdog_50_path": watchdog_50_path,
        "watchdog_100_path": watchdog_100_path,
        "checkpoint_path": checkpoint_path,
    }


def _upgrade_fixture_to_v2(
    fixture: dict,
    *,
    resume_step: int = 50_000,
    reconciliation_status: str = "unchanged",
) -> dict:
    contract = json.loads(fixture["contract_path"].read_text(encoding="utf-8"))
    contract["schema_version"] = 2
    contract["exact_resume"]["run_manifest"] = {
        "path": fixture["manifest_path"].resolve().as_posix(),
        "policy": MUTABLE_SEMANTIC_POLICY,
    }
    _write(fixture["contract_path"], contract)

    run_dir = fixture["manifest_path"].parent.resolve()
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest.update(
        {
            "config": {
                "data": {"batch_size": 64},
                "optimization": {"gradient_accumulation_steps": 1},
            },
            "runtime_environment": RUNTIME_ENVIRONMENT,
            "resume": (
                run_dir / f"checkpoint_step_{resume_step:08d}.pt"
            ).as_posix(),
            "resume_revision_transition": None,
        }
    )
    metrics_path = (run_dir / "train_metrics.jsonl").as_posix()
    if reconciliation_status == "unchanged":
        manifest["metrics_resume_reconciliation"] = {
            "schema_version": 1,
            "status": "unchanged",
            "resume_step": resume_step,
            "metrics": metrics_path,
            "retained_rows": 1002,
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        }
    elif reconciliation_status == "reconciled":
        orphan_payload = f'{{"step": {resume_step + 1}}}\n'
        orphan_sha = hashlib.sha256(orphan_payload.encode("utf-8")).hexdigest()
        orphan_path = run_dir / (
            f"train_metrics_orphaned_at_resume_{resume_step:08d}_"
            f"{orphan_sha[:12]}.jsonl"
        )
        orphan_path.write_bytes(orphan_payload.encode("utf-8"))
        report_path = run_dir / (
            f"metrics_resume_reconciliation_{resume_step:08d}_"
            f"{orphan_sha[:12]}.json"
        )
        report = {
            "schema_version": 1,
            "status": "reconciled",
            "resume_step": resume_step,
            "metrics": metrics_path,
            "retained_rows": 1002,
            "orphaned_rows": 1,
            "orphan_archive": orphan_path.as_posix(),
            "orphan_sha256": orphan_sha,
        }
        _write(report_path, report)
        manifest["metrics_resume_reconciliation"] = {
            **report,
            "report": report_path.as_posix(),
        }
        fixture["current_orphan_path"] = orphan_path
        fixture["current_reconciliation_path"] = report_path
    else:
        raise ValueError("unsupported test reconciliation status")
    _write(fixture["manifest_path"], manifest)

    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["runs"]["cofitok"]["last_step"] = max(resume_step + 100, 50_100)
    _write(fixture["pair_path"], pair)
    watchdog = json.loads(
        fixture["watchdog_50_path"].read_text(encoding="utf-8")
    )
    _complete_watchdog(fixture["watchdog_50_path"])
    watchdog.update(
        {
            "status": "running",
            "pid": 106,
            "child_pid": 107,
            "monitor_pid": 105,
            "child_exit_code": None,
            "watchdog_exit_code": None,
            "reason": "child_and_monitor_active",
            "command": [
                "python",
                "scripts/train_generation.py",
                "--output-dir",
                run_dir.as_posix(),
                "--resume",
                "auto",
            ],
        }
    )
    _write(fixture["watchdog_100_path"], watchdog)
    fixture["alive"] = {101, 102, 105, 106, 107}
    return fixture


def _inspect(fixture: dict, *, alive: set[int] | None = None) -> dict:
    contract = load_recovery_supersession_contract(fixture["contract_path"])
    live = fixture.get("alive", {101, 102, 103, 104, 105}) if alive is None else alive
    return inspect_recovery_supersession(
        fixture["failed_stage"],
        contract=contract,
        now=NOW,
        stale_seconds=240.0,
        process_exists=lambda pid: pid in live,
    )


def test_checked_in_recovery_supersession_contract_pins_live_evidence() -> None:
    contract = load_recovery_supersession_contract(LIVE_CONTRACT)
    assert contract["identity"]["sha256"] == hashlib.sha256(
        LIVE_CONTRACT.read_bytes()
    ).hexdigest()
    assert _tracked_blob_sha256(LIVE_CONTRACT) == LIVE_CONTRACT_SHA256
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


def test_checked_in_v2_contract_preserves_immutable_recovery_evidence() -> None:
    contract = load_recovery_supersession_contract(LIVE_CONTRACT_V2)
    assert contract["identity"]["sha256"] == LIVE_CONTRACT_V2_SHA256
    assert contract["schema_version"] == 2
    assert contract["exact_resume"]["run_manifest"]["policy"] == (
        MUTABLE_SEMANTIC_POLICY
    )
    assert "sha256" not in contract["exact_resume"]["run_manifest"]
    assert contract["exact_resume"]["reconciliation"]["sha256"] == (
        "dfe268c1defbaaa424cb93df108deb1421080f9568c47b02f1607568d6f57aca"
    )
    assert contract["exact_resume"]["checkpoint_audit"]["sha256"] == (
        "3b89622a286d5ef6b7244daa47a1866f272da447b564887ad477c53138a6d1f9"
    )
    assert contract["exact_resume"]["resume_step"] == 20_000


def test_v2_accepts_a_later_unchanged_exact_resume(tmp_path: Path) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    stage = _inspect(fixture)

    supersession = stage["supersession"]
    assert stage["status"] == "running"
    assert supersession["status"] == "pass"
    assert supersession["run_manifest"]["policy"] == MUTABLE_SEMANTIC_POLICY
    assert supersession["original_resume_step"] == 20_000
    assert supersession["current_resume_step"] == 50_000
    assert supersession["current_resume_checkpoint"].endswith(
        "/checkpoint_step_00050000.pt"
    )
    assert supersession["current_metrics_reconciliation"] == {
        "status": "unchanged",
        "resume_step": 50_000,
        "report": None,
        "orphan_archive": None,
    }
    assert supersession["original_orphan_archive"]["sha256"] == hashlib.sha256(
        fixture["original_orphan_path"].read_bytes()
    ).hexdigest()


def test_v2_accepts_a_content_addressed_current_reconciliation(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(
        _fixture(tmp_path), reconciliation_status="reconciled"
    )
    stage = _inspect(fixture)

    evidence = stage["supersession"]["current_metrics_reconciliation"]
    assert evidence["status"] == "reconciled"
    assert evidence["resume_step"] == 50_000
    assert evidence["report"]["path"] == fixture[
        "current_reconciliation_path"
    ].resolve().as_posix()
    assert evidence["orphan_archive"]["sha256"] == hashlib.sha256(
        fixture["current_orphan_path"].read_bytes()
    ).hexdigest()


def test_v2_rejects_a_manifest_policy_downgrade(tmp_path: Path) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    contract = json.loads(fixture["contract_path"].read_text(encoding="utf-8"))
    contract["exact_resume"]["run_manifest"]["policy"] = "immutable_sha256"
    _write(fixture["contract_path"], contract)

    with pytest.raises(ValueError, match="mutable-semantic policy differs"):
        load_recovery_supersession_contract(fixture["contract_path"])


def test_v2_rejects_a_resume_older_than_the_original_recovery(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path), resume_step=10_000)
    with pytest.raises(ValueError, match="predates original recovery"):
        _inspect(fixture)


def test_v2_rejects_a_resume_path_escape(tmp_path: Path) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["resume"] = (tmp_path / "checkpoint_step_00050000.pt").as_posix()
    _write(fixture["manifest_path"], manifest)

    with pytest.raises(ValueError, match="escapes the recovery run directory"):
        _inspect(fixture)


def test_v2_rejects_stable_manifest_identity_drift(tmp_path: Path) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["dataset_provenance"]["identity_sha256"] = "0" * 64
    _write(fixture["manifest_path"], manifest)
    with pytest.raises(ValueError, match="dataset identity differs"):
        _inspect(fixture)

    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path / "runtime"))
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["runtime_environment"]["python"]["version"] = "3.10.19"
    _write(fixture["manifest_path"], manifest)
    with pytest.raises(ValueError, match="runtime identity differs"):
        _inspect(fixture)

    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path / "batch"))
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["config"]["optimization"]["gradient_accumulation_steps"] = 2
    _write(fixture["manifest_path"], manifest)
    with pytest.raises(ValueError, match="effective batch differs"):
        _inspect(fixture)


def test_v2_rejects_inconsistent_unchanged_reconciliation(tmp_path: Path) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    manifest["metrics_resume_reconciliation"]["orphaned_rows"] = 1
    _write(fixture["manifest_path"], manifest)

    with pytest.raises(ValueError, match="unchanged reconciliation differs"):
        _inspect(fixture)


def test_v2_rejects_tampered_current_reconciliation_evidence(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(
        _fixture(tmp_path), reconciliation_status="reconciled"
    )
    fixture["current_orphan_path"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current orphan archive SHA256 differs"):
        _inspect(fixture)


def test_v2_rejects_tampered_original_reconciliation_evidence(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    fixture["original_orphan_path"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="original orphan archive SHA256 differs"):
        _inspect(fixture)


def test_v2_requires_the_live_step_to_cover_the_current_resume(
    tmp_path: Path,
) -> None:
    fixture = _upgrade_fixture_to_v2(_fixture(tmp_path))
    pair = json.loads(fixture["pair_path"].read_text(encoding="utf-8"))
    pair["runs"]["cofitok"]["last_step"] = 49_999
    _write(fixture["pair_path"], pair)

    with pytest.raises(ValueError, match="predates trusted checkpoint"):
        _inspect(fixture)


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
