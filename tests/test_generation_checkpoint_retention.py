from __future__ import annotations

import json
from pathlib import Path

import pytest

from cofitok.checkpoint_retention import (
    build_checkpoint_retention_inventory,
    build_retention_runway_report,
    verify_checkpoint_retention_inventory,
)
from cofitok.reporting import file_sha256, write_json_report


def _checkpoint(run: Path, step: int, content: bytes) -> Path:
    path = run / f"checkpoint_step_{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    write_json_report(
        path.with_name(f"{path.name}.integrity.json"),
        {
            "schema_version": 1,
            "checkpoint": path.name,
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_format_version": 1,
            "step": step,
        },
    )
    return path


def _completed_run(root: Path) -> tuple[Path, Path, Path]:
    run = root / "pair" / "method"
    middle = _checkpoint(run, 500, b"middle")
    latest = _checkpoint(run, 1000, b"latest")
    write_json_report(
        run / "latest.json",
        {"checkpoint": latest.name, "step": 1000},
    )
    write_json_report(
        run / "training_report.json",
        {
            "training_complete": True,
            "completed_steps": 1000,
            "config": {"runtime": {"protected_checkpoint_steps": []}},
        },
    )
    return run, middle, latest


def test_retention_inventory_requires_latest_and_candidates_unreferenced_middle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probe"
    _, middle, latest = _completed_run(root)

    report = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    by_name = {Path(item["path"]).name: item for item in report["checkpoints"]}

    assert by_name[latest.name]["disposition"] == "required"
    assert by_name[middle.name]["disposition"] == "candidate_for_archive"
    assert not by_name[middle.name]["references"]
    assert report["summary"]["currently_reclaimable_bytes"] == 0
    assert report["summary"]["potential_archive_candidate_bytes"] == len(b"middle")
    assert report["archive_or_deletion_authorized"] is False
    assert verify_checkpoint_retention_inventory(report) == report


def test_operational_reference_blocks_archive_candidate(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    run, middle, _ = _completed_run(root)
    write_json_report(
        root / "pair_monitor.json",
        {"run_dir": run.as_posix(), "checkpoints": [{"name": middle.name}]},
    )

    report = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    entry = next(item for item in report["checkpoints"] if item["step"] == 500)

    assert entry["disposition"] == "indeterminate"
    assert "operational_reference" in entry["reasons"]
    assert report["archive_readiness"] == "blocked"


def test_authoritative_reference_makes_intermediate_required(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    _, middle, _ = _completed_run(root)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "milestone_report.json").write_text(
        json.dumps({"checkpoint": middle.as_posix()}), encoding="utf-8"
    )

    report = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[("tracked", evidence, "authoritative")],
        physical_hash=True,
    )
    entry = next(item for item in report["checkpoints"] if item["step"] == 500)

    assert entry["disposition"] == "required"
    assert "authoritative_reference" in entry["reasons"]


def test_protected_intermediate_is_required_without_external_reference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probe"
    run, _, _ = _completed_run(root)
    training = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
    training["config"]["runtime"]["protected_checkpoint_steps"] = [500]
    write_json_report(run / "training_report.json", training)

    report = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    entry = next(item for item in report["checkpoints"] if item["step"] == 500)

    assert entry["disposition"] == "required"
    assert "protected_checkpoint_step" in entry["reasons"]


def test_missing_physical_hash_and_integrity_drift_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    _, middle, _ = _completed_run(root)
    report = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=False,
    )
    entry = next(item for item in report["checkpoints"] if item["step"] == 500)
    assert entry["disposition"] == "indeterminate"

    hashed = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    middle.write_bytes(b"changed")
    with pytest.raises(ValueError, match="replay changed"):
        verify_checkpoint_retention_inventory(hashed)


def test_report_tampering_is_rejected_by_physical_replay(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    _completed_run(root)
    report = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    latest = next(item for item in report["checkpoints"] if item["step"] == 1000)
    latest["disposition"] = "candidate_for_archive"

    with pytest.raises(ValueError, match="replay changed"):
        verify_checkpoint_retention_inventory(report)


def test_runway_never_counts_unapproved_archive_candidates(tmp_path: Path) -> None:
    root = tmp_path / "probe"
    _completed_run(root)
    inventory = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    inventory_path = tmp_path / "retention.json"
    write_json_report(inventory_path, inventory)

    runway = build_retention_runway_report(
        retention_report=inventory,
        retention_report_path=inventory_path,
        expected_retention_report_sha256=file_sha256(inventory_path),
        filesystem_path=tmp_path,
        total_bytes=1000,
        used_bytes=400,
        free_bytes=600,
        required_free_bytes=650,
    )

    assert runway["status"] == "fail"
    assert runway["current_headroom_bytes"] == -50
    assert runway["currently_reclaimable_bytes"] == 0
    assert runway["potential_archive_bytes_counted_as_current_capacity"] is False
    assert runway["projected_headroom_after_unapproved_archive_bytes"] > -50
    assert runway["retention_inventory_verification"] == "expected_sha256_binding"
    assert runway["physical_checkpoint_hashes_replayed"] is False


def test_runway_binds_locked_inventory_without_rehashing_checkpoints(
    tmp_path: Path,
) -> None:
    root = tmp_path / "probe"
    _, middle, _ = _completed_run(root)
    inventory = build_checkpoint_retention_inventory(
        inventory_root=root,
        reference_roots=[],
        physical_hash=True,
    )
    inventory_path = tmp_path / "retention.json"
    write_json_report(inventory_path, inventory)
    expected_sha256 = file_sha256(inventory_path)

    middle.write_bytes(b"changed after the independently replayed inventory")
    runway = build_retention_runway_report(
        retention_report=inventory,
        retention_report_path=inventory_path,
        expected_retention_report_sha256=expected_sha256,
        filesystem_path=tmp_path,
        total_bytes=1000,
        used_bytes=400,
        free_bytes=600,
        required_free_bytes=500,
    )

    assert runway["status"] == "pass"
    assert runway["physical_checkpoint_hashes_replayed"] is False

    inventory_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        build_retention_runway_report(
            retention_report=inventory,
            retention_report_path=inventory_path,
            expected_retention_report_sha256=expected_sha256,
            filesystem_path=tmp_path,
            total_bytes=1000,
            used_bytes=400,
            free_bytes=600,
            required_free_bytes=500,
        )
