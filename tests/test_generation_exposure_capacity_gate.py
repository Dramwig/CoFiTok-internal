from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_generation_exposure_capacity_execution_gate as build_cli
from scripts import validate_generation_exposure_capacity_preparation as preparation_cli
from scripts import validate_generation_exposure_capacity_execution_gate as validate_cli
from cofitok.generation.exposure_capacity import PREPARATION_BOUNDARY, build_preparation
from cofitok.generation.exposure_capacity_gate import (
    CAPACITY_QUALIFICATION_STEPS,
    EXPOSURE_TARGET_STEP,
    GATE_AUTHORIZATION_BOUNDARY,
    SOURCE_BRANCH,
    SOURCE_REVISION,
    SOURCE_TREE,
    build_execution_gate,
    validate_execution_gate,
)


def _identity(index: int) -> dict[str, object]:
    return {
        "path": f"/evidence/source_{index}.json",
        "bytes": index + 100,
        "sha256": f"{index + 1:064x}",
    }


def _preparation() -> dict[str, object]:
    quality = {
        "status": "completed",
        "terminal": {"status": "hold"},
        "quality_screen": {"status": "hold", "non_authorizing": True},
        "authorization_boundary": {key: False for key in PREPARATION_BOUNDARY},
    }
    objective = {
        "schema": "cofitok_generation_training_objective_reassessment_v1",
        "status": "completed",
        "decision": "preserve_qualified_objective_defer_new_intervention",
        "generation_advantage_proven": False,
        "execution_ready": False,
        "next_evidence": {"id": "prepare_source_bound_exposure_or_capacity_gate"},
        "authorization_boundary": deepcopy(PREPARATION_BOUNDARY),
    }
    post = {
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": {key: False for key in PREPARATION_BOUNDARY},
    }
    cross = {"status": "completed", "operational_status": "pass", "terminal_status": "hold"}
    sampling = {
        "status": "completed",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_sampling_recovery_candidate",
        "generation_advantage_proven": False,
    }
    min_snr = {
        "status": "completed",
        "scientific_status": "screening_only",
        "selection_status": "no_shared_min_snr_candidate_at_50k",
        "generation_advantage_proven": False,
    }
    pair = {
        "status": "pass",
        "stage": "complete",
        "issues": [],
        "runs": {
            method: {
                "complete": True,
                "last_step": 100_000,
                "expected_steps": 100_000,
                "last_metric": {"step": 100_000, "samples_seen": 6_400_000},
            }
            for method in ("cofitok", "dense_identity")
        },
    }
    training = lambda method="cofitok": {
        "output_dir": (
            "/evidence/cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
            if method == "cofitok"
            else "/evidence/dense_rollout_x0_u2_ema_teacher"
        ),
        "training_complete": True,
        "completed_steps": 100_000,
        "target_steps": 100_000,
        "config": {
            "model": {
                "synthesis_mode": "fixed_basis" if method == "cofitok" else "dense_identity",
                "token_count": 8 if method == "cofitok" else 1,
                "predictor_use_feedback": method == "cofitok",
            }
        },
        "git": {
            "revision": "a" * 40,
            "branch": SOURCE_BRANCH,
            "dirty": False,
        },
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00100000.pt",
            "checkpoint_bytes": 1000 if method == "cofitok" else 1001,
            "checkpoint_sha256": ("1" if method == "cofitok" else "2") * 64,
            "integrity_manifest": "checkpoint_step_00100000.pt.integrity.json",
            "step": 100_000,
        },
    }
    metric = lambda value: {
        "step": 100_000,
        "samples_seen": 6_400_000,
        "validation_epsilon_mse": value,
    }
    names = (
        "objective_reassessment",
        "quality_bridge_result",
        "post_reconciliation_decision",
        "cross_protocol_reconciliation",
        "sampling_recovery_result",
        "min_snr_result",
        "pair_monitor",
        "cofitok_training_report",
        "dense_training_report",
        "cofitok_metrics",
        "dense_metrics",
    )
    return build_preparation(
        objective_reassessment=objective,
        quality_bridge_result=quality,
        post_reconciliation_decision=post,
        cross_protocol_reconciliation=cross,
        sampling_recovery_result=sampling,
        min_snr_result=min_snr,
        pair_monitor=pair,
        cofitok_metrics=metric(0.029256),
        dense_metrics=metric(0.029241),
        cofitok_training_report=training("cofitok"),
        dense_training_report=training("dense_identity"),
        source_identities={name: _identity(index) for index, name in enumerate(names)},
        builder_git={
            "revision": "b" * 40,
            "tree": "c" * 40,
            "branch": SOURCE_BRANCH,
            "tracked_dirty": True,
        },
        exposure_output_root=(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "exposure_capacity_disambiguation_v1/exposure"
        ),
        capacity_output_root=(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "exposure_capacity_disambiguation_v1/capacity"
        ),
    )


def _kwargs(preparation: dict[str, object], arm_id: str) -> dict[str, object]:
    values = {
        "preparation": preparation,
        "preparation_identity": _identity(50),
        "arm_id": arm_id,
        "source_checkout": {
            "revision": SOURCE_REVISION,
            "tree": SOURCE_TREE,
            "branch": SOURCE_BRANCH,
            "tracked_dirty": False,
        },
        "gate_builder_git": {
            "revision": "d" * 40,
            "branch": "analysis/exposure-capacity-gate-v1",
            "tracked_dirty": False,
        },
        "runtime_environment_sha256": "1" * 64,
        "dataset_identity_sha256": "2" * 64,
        "gpu_inventory": [
            {
                "index": 0,
                "memory_used_mib": 0,
                "memory_total_mib": 97_887,
                "utilization_percent": 0,
            }
        ],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 300 * 1024**3,
    }
    if arm_id == "exposure_continuation":
        expected = preparation["evidence"]["training"]["source_checkpoints"]["cofitok"]
        expected_checkpoint = expected["checkpoint"]
        values["source_checkpoint_binding"] = {
            "run_dir": expected["run_dir"],
            "step": 100_000,
            "checkpoint": {
                "path": f"{expected['run_dir']}/{expected_checkpoint['name']}",
                "bytes": expected_checkpoint["bytes"],
                "sha256": expected_checkpoint["sha256"],
            },
            "integrity_manifest": {
                "path": f"{expected['run_dir']}/{expected['integrity_manifest_name']}",
                "bytes": 161,
                "sha256": "3" * 64,
            },
            "latest": {
                "path": f"{expected['run_dir']}/latest.json",
                "bytes": 162,
                "sha256": "4" * 64,
            },
        }
    return values


def test_capacity_gate_is_bounded_and_non_authorizing() -> None:
    preparation = _preparation()
    gate = build_execution_gate(**_kwargs(preparation, "capacity_qualification"))
    assert gate["status"] == "prepared"
    assert gate["execution_ready"] is False
    assert gate["authorization_boundary"] == GATE_AUTHORIZATION_BOUNDARY
    contract = gate["qualification_contract"]
    assert contract["qualification_steps"] == CAPACITY_QUALIFICATION_STEPS
    assert contract["screen_samples_per_method"] == 1_000
    assert contract["confirmation_samples_per_method"] == 10_000
    assert validate_execution_gate(
        gate,
        preparation=preparation,
        preparation_identity=_identity(50),
    ) == gate


def test_exposure_gate_has_fixed_first_bounded_target() -> None:
    preparation = _preparation()
    gate = build_execution_gate(**_kwargs(preparation, "exposure_continuation"))
    contract = gate["qualification_contract"]
    assert contract["target_step"] == EXPOSURE_TARGET_STEP
    assert contract["additional_steps"] == 10_000
    assert contract["resume_checkpoint_required"] is True
    assert contract["automatic_300k_escalation_allowed"] is False
    assert gate["source_checkpoint"]["step"] == 100_000
    assert gate["source_checkpoint"]["checkpoint"]["bytes"] == 1000


def test_capacity_gate_rejects_resume_source() -> None:
    preparation = _preparation()
    values = _kwargs(preparation, "capacity_qualification")
    values["source_checkpoint_binding"] = _kwargs(
        preparation, "exposure_continuation"
    )["source_checkpoint_binding"]
    with pytest.raises(ValueError, match="must not provide a source checkpoint binding"):
        build_execution_gate(**values)

    gate = build_execution_gate(**_kwargs(preparation, "capacity_qualification"))
    gate["source_checkpoint"] = values["source_checkpoint_binding"]
    with pytest.raises(ValueError, match="must not provide a source checkpoint binding"):
        validate_execution_gate(
            gate,
            preparation=preparation,
            preparation_identity=_identity(50),
        )


def test_gate_rejects_permission_or_stage_authorization_tampering() -> None:
    preparation = _preparation()
    gate = build_execution_gate(**_kwargs(preparation, "capacity_qualification"))
    tampered = deepcopy(gate)
    tampered["authorization_boundary"]["gpu_execution_authorized"] = True
    with pytest.raises(ValueError, match="gate contract differs"):
        validate_execution_gate(
            tampered,
            preparation=preparation,
            preparation_identity=_identity(50),
        )

    tampered = build_execution_gate(**_kwargs(preparation, "exposure_continuation"))
    tampered["source_checkpoint"]["latest"]["path"] = "/evidence/other/latest.json"
    with pytest.raises(ValueError, match="outside its declared run directory"):
        validate_execution_gate(
            tampered,
            preparation=preparation,
            preparation_identity=_identity(50),
        )

    tampered = build_execution_gate(**_kwargs(preparation, "exposure_continuation"))
    tampered["source_checkpoint"]["checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sha256 differs from preparation"):
        validate_execution_gate(
            tampered,
            preparation=preparation,
            preparation_identity=_identity(50),
        )

    tampered = build_execution_gate(**_kwargs(preparation, "exposure_continuation"))
    tampered["source_checkpoint"]["run_dir"] = "/evidence/dense_identity_run"
    tampered["source_checkpoint"]["checkpoint"]["path"] = (
        "/evidence/dense_identity_run/checkpoint_step_00100000.pt"
    )
    tampered["source_checkpoint"]["integrity_manifest"]["path"] = (
        "/evidence/dense_identity_run/checkpoint_step_00100000.pt.integrity.json"
    )
    tampered["source_checkpoint"]["latest"]["path"] = "/evidence/dense_identity_run/latest.json"
    with pytest.raises(ValueError, match="run directory differs from preparation"):
        validate_execution_gate(
            tampered,
            preparation=preparation,
            preparation_identity=_identity(50),
        )

    for key, value in (
        ("execution_ready", True),
        ("terminal_status", "pass"),
        ("generation_advantage_proven", True),
    ):
        tampered = deepcopy(gate)
        tampered[key] = value
        with pytest.raises(ValueError, match="gate contract differs"):
            validate_execution_gate(
                tampered,
                preparation=preparation,
                preparation_identity=_identity(50),
            )

    tampered = deepcopy(gate)
    tampered["stage_authorization"]["status"] = "authorized"
    with pytest.raises(ValueError, match="stage authorization"):
        validate_execution_gate(
            tampered,
            preparation=preparation,
            preparation_identity=_identity(50),
        )


def test_gate_rejects_busy_gpu_and_conflicting_process() -> None:
    preparation = _preparation()
    busy = _kwargs(preparation, "capacity_qualification")
    busy["gpu_inventory"] = [
        {
            "index": 0,
            "memory_used_mib": 32,
            "memory_total_mib": 97_887,
            "utilization_percent": 0,
        }
    ]
    with pytest.raises(ValueError, match="target GPU is not idle"):
        build_execution_gate(**busy)

    conflicting = _kwargs(preparation, "capacity_qualification")
    conflicting["conflicting_processes"] = [{"pid": 42, "command": "train_generation"}]
    with pytest.raises(ValueError, match="conflicting project processes"):
        build_execution_gate(**conflicting)


def test_conflicting_process_scan_ignores_parent_command_substrings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ps_output = "\n".join(
        (
            "100 1 Mon Jan 1 00:00:00 2026 bash -c python scripts/"
            "build_generation_exposure_capacity_execution_gate.py --preparation "
            "/root/autodl-tmp/CoFiTok/checkpoints/generation_exposure_capacity_"
            "disambiguation_v1/preparation.json",
            "101 1 Mon Jan 1 00:00:00 2026 python scripts/generation_exposure_capacity.py",
            "102 1 Mon Jan 1 00:00:00 2026 python -m cofitok.generation_exposure_capacity",
            "103 1 Mon Jan 1 00:00:00 2026 python scripts/train_generation.py",
        )
    )
    monkeypatch.setattr(
        build_cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=ps_output),
    )
    monkeypatch.setattr(build_cli.os, "getpid", lambda: 999)

    conflicts = build_cli._conflicting_processes()

    assert [row["pid"] for row in conflicts] == [101, 102, 103]


def test_gate_rejects_wrong_source_checkout_or_storage() -> None:
    preparation = _preparation()
    wrong_source = _kwargs(preparation, "capacity_qualification")
    wrong_source["source_checkout"] = {
        "revision": "e" * 40,
        "tree": SOURCE_TREE,
        "branch": SOURCE_BRANCH,
        "tracked_dirty": False,
    }
    with pytest.raises(ValueError, match="locked 100K bridge"):
        build_execution_gate(**wrong_source)

    low_storage = _kwargs(preparation, "capacity_qualification")
    low_storage["free_bytes"] = 1
    with pytest.raises(ValueError, match="insufficient free storage"):
        build_execution_gate(**low_storage)


def _materialize_preparation_sources(
    preparation: dict[str, object], tmp_path: Path
) -> tuple[dict[str, object], Path, str]:
    """Give the CLI replay real, content-addressed source files to rehash."""
    materialized = deepcopy(preparation)
    sources = materialized["sources"]
    assert isinstance(sources, dict)
    for name, descriptor in sources.items():
        assert isinstance(descriptor, dict)
        source = tmp_path / f"{name}.json"
        if name in {"cofitok_training_report", "dense_training_report"}:
            method = "cofitok" if name.startswith("cofitok") else "dense_identity"
            summary = materialized["evidence"]["training"]["source_checkpoints"][method]
            model = {
                "synthesis_mode": "fixed_basis" if method == "cofitok" else "dense_identity",
                "token_count": 8 if method == "cofitok" else 1,
                "predictor_use_feedback": method == "cofitok",
            }
            payload = {
                "output_dir": summary["run_dir"],
                "config": {"model": model},
                "latest_checkpoint": {
                    "checkpoint": summary["checkpoint"]["name"],
                    "checkpoint_bytes": summary["checkpoint"]["bytes"],
                    "checkpoint_sha256": summary["checkpoint"]["sha256"],
                    "integrity_manifest": summary["integrity_manifest_name"],
                    "step": summary["step"],
                },
            }
        else:
            payload = {"source": name}
        source.write_text(json.dumps(payload), encoding="utf-8")
        descriptor["path"] = str(source.resolve())
        descriptor["bytes"] = source.stat().st_size
        descriptor["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(
        json.dumps(materialized, sort_keys=True), encoding="utf-8"
    )
    return materialized, preparation_path, hashlib.sha256(preparation_path.read_bytes()).hexdigest()


def test_cli_build_and_validate_replay_is_source_bound_and_non_authorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    preparation, preparation_path, preparation_sha = _materialize_preparation_sources(
        _preparation(), tmp_path
    )
    output = tmp_path / "gate.json"
    lock = tmp_path / "locks" / "execution.lock"
    builder_git = {
        "revision": "d" * 40,
        "tree": "e" * 40,
        "branch": "analysis/exposure-capacity-gate-v1",
        "tracked_dirty": False,
    }
    monkeypatch.setattr(
        build_cli,
        "_source_checkout",
        lambda _path: {
            "revision": SOURCE_REVISION,
            "tree": SOURCE_TREE,
            "branch": SOURCE_BRANCH,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(build_cli, "_git", lambda _path: builder_git)
    monkeypatch.setattr(
        build_cli,
        "_gpu_inventory",
        lambda: (
            [
                {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "test-gpu",
                    "memory_used_mib": 0,
                    "memory_total_mib": 97_887,
                    "utilization_percent": 0,
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(build_cli, "_conflicting_processes", lambda: [])
    monkeypatch.setattr(build_cli, "_lock_is_free", lambda _path: True)
    monkeypatch.setattr(
        build_cli.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=300 * 1024**3),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_exposure_capacity_execution_gate.py",
            "--preparation",
            str(preparation_path),
            "--expected-preparation-sha256",
            preparation_sha,
            "--arm",
            "capacity_qualification",
            "--source-project-root",
            str(tmp_path),
            "--runtime-environment-sha256",
            "1" * 64,
            "--dataset-identity-sha256",
            "2" * 64,
            "--storage-path",
            str(tmp_path),
            "--execution-lock",
            str(lock),
            "--output",
            str(output),
        ],
    )
    build_cli.main()
    gate_sha = capsys.readouterr().out.strip()
    assert gate_sha == hashlib.sha256(output.read_bytes()).hexdigest()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_execution_gate.py",
            "--gate",
            str(output),
            "--expected-gate-sha256",
            gate_sha,
            "--preparation",
            str(preparation_path),
            "--expected-preparation-sha256",
            preparation_sha,
        ],
    )
    validate_cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["selected_arm"] == "capacity_qualification"
    assert result["execution_ready"] is False
    assert result["authorization_boundary"] == GATE_AUTHORIZATION_BOUNDARY

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_exposure_capacity_execution_gate.py",
            "--preparation",
            str(preparation_path),
            "--expected-preparation-sha256",
            preparation_sha,
            "--arm",
            "capacity_qualification",
            "--source-project-root",
            str(tmp_path),
            "--runtime-environment-sha256",
            "1" * 64,
            "--dataset-identity-sha256",
            "2" * 64,
            "--storage-path",
            str(tmp_path),
            "--execution-lock",
            str(lock),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError, match="gate output already exists"):
        build_cli.main()


def test_lock_probe_does_not_create_missing_lock_or_parent(tmp_path: Path) -> None:
    lock = tmp_path / "missing" / "execution.lock"
    assert build_cli._lock_is_free(lock) is True
    assert not lock.exists()
    assert not lock.parent.exists()


@pytest.mark.skipif(build_cli.fcntl is None, reason="real flock probing requires POSIX")
def test_lock_probe_reads_existing_lock_without_creation(tmp_path: Path) -> None:
    lock = tmp_path / "execution.lock"
    lock.write_bytes(b"")
    assert build_cli._lock_is_free(lock) is True
    assert lock.read_bytes() == b""


def test_exposure_resume_binding_rehashes_checkpoint_sidecar_and_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preparation, preparation_path, preparation_sha = _materialize_preparation_sources(
        _preparation(), tmp_path
    )
    run_dir = tmp_path / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step_00100000.pt"
    payload = b"synthetic-checkpoint-payload\n"
    checkpoint.write_bytes(payload)
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": checkpoint.name,
                "checkpoint_bytes": len(payload),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_format_version": 1,
                "step": 100_000,
            }
        ),
        encoding="utf-8",
    )
    latest = run_dir / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "step": 100_000,
                "checkpoint_bytes": len(payload),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_format_version": 1,
                "integrity_manifest": sidecar.name,
            }
        ),
        encoding="utf-8",
    )
    expected = preparation["evidence"]["training"]["source_checkpoints"]["cofitok"]
    expected["run_dir"] = str(run_dir.resolve())
    expected["checkpoint"]["bytes"] = len(payload)
    expected["checkpoint"]["sha256"] = checkpoint_sha
    report_path = Path(preparation["sources"]["cofitok_training_report"]["path"])
    report_payload = {
        "output_dir": expected["run_dir"],
        "config": {
            "model": {
                "synthesis_mode": "fixed_basis",
                "token_count": 8,
                "predictor_use_feedback": True,
            }
        },
        "latest_checkpoint": {
            "checkpoint": expected["checkpoint"]["name"],
            "checkpoint_bytes": len(payload),
            "checkpoint_sha256": checkpoint_sha,
            "integrity_manifest": expected["integrity_manifest_name"],
            "step": expected["step"],
        },
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    preparation["sources"]["cofitok_training_report"].update(
        {
            "bytes": report_path.stat().st_size,
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    )
    preparation_path.write_text(json.dumps(preparation, sort_keys=True), encoding="utf-8")
    preparation_sha = hashlib.sha256(preparation_path.read_bytes()).hexdigest()
    binding = build_cli._source_checkpoint_binding(run_dir, checkpoint, sidecar, latest)
    gate = build_execution_gate(
        **{
            **_kwargs(preparation, "exposure_continuation"),
            "source_checkpoint_binding": binding,
        }
    )
    gate["preparation"] = {
        "path": str(preparation_path.resolve()),
        "bytes": preparation_path.stat().st_size,
        "sha256": preparation_sha,
    }
    gate_path = tmp_path / "exposure-gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_execution_gate.py",
            "--gate",
            str(gate_path),
            "--expected-gate-sha256",
            gate_sha,
            "--preparation",
            str(preparation_path),
            "--expected-preparation-sha256",
            preparation_sha,
        ],
    )
    validate_cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["source_checkpoint_verified"] is True

    checkpoint.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(ValueError, match="source checkpoint binding checkpoint changed"):
        validate_cli.main()


def test_preparation_cli_rejects_checkpoint_summary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparation, preparation_path, _ = _materialize_preparation_sources(
        _preparation(), tmp_path
    )
    preparation["evidence"]["training"]["source_checkpoints"]["cofitok"]["checkpoint"][
        "sha256"
    ] = "f" * 64
    preparation_path.write_text(json.dumps(preparation, sort_keys=True), encoding="utf-8")
    tampered_sha = hashlib.sha256(preparation_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_preparation.py",
            "--preparation",
            str(preparation_path),
            "--expected-preparation-sha256",
            tampered_sha,
        ],
    )
    with pytest.raises(ValueError, match="summaries do not match training reports"):
        preparation_cli.main()


def test_preparation_cli_rehashes_sources_and_rejects_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _preparation_report, preparation_path, preparation_sha = _materialize_preparation_sources(
        _preparation(), tmp_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_preparation.py",
            "--preparation",
            str(preparation_path),
            "--expected-preparation-sha256",
            preparation_sha,
        ],
    )
    preparation_cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["source_count"] == 11

    source = tmp_path / "quality_bridge_result.json"
    replaced = bytearray(source.read_bytes())
    replaced[0] ^= 1
    source.write_bytes(replaced)
    with pytest.raises(ValueError, match="SHA256 changed"):
        preparation_cli.main()

    tampered_report = bytearray(preparation_path.read_bytes())
    tampered_report[0] ^= 1
    preparation_path.write_bytes(tampered_report)
    with pytest.raises(ValueError, match="preparation SHA256 differs"):
        preparation_cli.main()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix symlink creation requires elevated privileges on Windows",
)
def test_exposure_capacity_cli_source_identities_reject_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "source.json"
    target.write_text("{}\n", encoding="ascii")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="must not contain a symlink"):
        build_cli._identity(alias)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        validate_cli._read(alias)
