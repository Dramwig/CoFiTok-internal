from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


REFERENCE_SCHEMA_VERSION = 1
REFERENCE_ROLE = "generation_checkpoint_hardlink_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve a verified checkpoint through same-filesystem hardlinks so "
            "rolling retention cannot remove the bytes needed by a later comparison."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _source(path: Path, *, trusted_sha256: str = "") -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": resolved.as_posix(),
        "bytes": stat.st_size,
        "sha256": trusted_sha256 or file_sha256(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "hardlink_count_at_build": stat.st_nlink,
    }


def _validate_integrity(
    integrity: dict[str, Any],
    *,
    expected_step: int,
    expected_revision: str,
    expected_branch: str,
) -> None:
    if int(integrity.get("step", -1)) != expected_step:
        raise ValueError("checkpoint step differs")
    if (
        integrity.get("git_revision") != expected_revision
        or integrity.get("git_branch") != expected_branch
        or integrity.get("git_dirty") is not False
    ):
        raise ValueError("checkpoint Git identity differs")


def _atomic_hardlink(source: Path, target: Path) -> None:
    if target.exists():
        source_stat = source.stat()
        target_stat = target.stat()
        if (
            source_stat.st_dev != target_stat.st_dev
            or source_stat.st_ino != target_stat.st_ino
        ):
            raise ValueError(f"existing reference is not the same inode: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.link.",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def preserve_checkpoint_reference(
    *,
    checkpoint: Path,
    reference_dir: Path,
    expected_step: int,
    expected_revision: str,
    expected_branch: str,
    reason: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    reference_dir = reference_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if expected_step < 1 or not reason.strip():
        raise ValueError("expected step and reason must be set")

    integrity_path = checkpoint_integrity_path(checkpoint)
    integrity = verify_training_checkpoint(checkpoint)
    _validate_integrity(
        integrity,
        expected_step=expected_step,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    reference_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint.stat().st_dev != reference_dir.stat().st_dev:
        raise ValueError("reference directory is not on the checkpoint filesystem")

    target_checkpoint = reference_dir / checkpoint.name
    target_integrity = reference_dir / integrity_path.name
    _atomic_hardlink(checkpoint, target_checkpoint)
    _atomic_hardlink(integrity_path, target_integrity)

    checkpoint_sha256 = str(integrity["checkpoint_sha256"])
    source_checkpoint = _source(checkpoint, trusted_sha256=checkpoint_sha256)
    reference_checkpoint = _source(
        target_checkpoint,
        trusted_sha256=checkpoint_sha256,
    )
    source_integrity = _source(integrity_path)
    reference_integrity = _source(target_integrity)
    for source, reference, label in (
        (source_checkpoint, reference_checkpoint, "checkpoint"),
        (source_integrity, reference_integrity, "integrity"),
    ):
        if (
            source["device"] != reference["device"]
            or source["inode"] != reference["inode"]
            or source["bytes"] != reference["bytes"]
            or source["sha256"] != reference["sha256"]
        ):
            raise ValueError(f"{label} hardlink identity differs")
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "status": "preserved",
        "role": REFERENCE_ROLE,
        "reason": reason.strip(),
        "expected": {
            "step": expected_step,
            "git": {
                "revision": expected_revision,
                "branch": expected_branch,
                "tracked_dirty": False,
            },
        },
        "source": {
            "checkpoint": source_checkpoint,
            "integrity": source_integrity,
        },
        "reference": {
            "directory": reference_dir.as_posix(),
            "checkpoint": reference_checkpoint,
            "integrity": reference_integrity,
        },
        "storage": {
            "same_filesystem": True,
            "checkpoint_same_inode": True,
            "integrity_same_inode": True,
            "additional_checkpoint_data_blocks_required": False,
        },
        "authorization_boundary": {
            "training_launch_allowed": False,
            "gpu_use_allowed": False,
            "checkpoint_mutation_allowed": False,
            "checkpoint_deletion_allowed": False,
            "full_300k_launch_allowed": False,
            "release_allowed": False,
        },
    }


def verify_checkpoint_reference(
    report: dict[str, Any],
    *,
    expected_step: int,
    expected_revision: str,
    expected_branch: str,
    expected_reason: str,
) -> dict[str, Any]:
    if (
        report.get("schema_version") != REFERENCE_SCHEMA_VERSION
        or report.get("status") != "preserved"
        or report.get("role") != REFERENCE_ROLE
        or report.get("reason") != expected_reason
    ):
        raise ValueError("checkpoint reference receipt contract differs")
    expected = report.get("expected")
    if expected != {
        "step": expected_step,
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        },
    }:
        raise ValueError("checkpoint reference expected identity differs")
    boundary = report.get("authorization_boundary")
    if boundary != {
        "training_launch_allowed": False,
        "gpu_use_allowed": False,
        "checkpoint_mutation_allowed": False,
        "checkpoint_deletion_allowed": False,
        "full_300k_launch_allowed": False,
        "release_allowed": False,
    }:
        raise ValueError("checkpoint reference authorization boundary differs")
    storage = report.get("storage")
    if storage != {
        "same_filesystem": True,
        "checkpoint_same_inode": True,
        "integrity_same_inode": True,
        "additional_checkpoint_data_blocks_required": False,
    }:
        raise ValueError("checkpoint reference storage contract differs")

    source = report.get("source")
    reference = report.get("reference")
    if not isinstance(source, dict) or not isinstance(reference, dict):
        raise ValueError("checkpoint reference source descriptors are missing")
    reference_checkpoint = Path(str(reference.get("checkpoint", {}).get("path", "")))
    reference_integrity = Path(str(reference.get("integrity", {}).get("path", "")))
    if (
        not reference_checkpoint.is_file()
        or not reference_integrity.is_file()
        or checkpoint_integrity_path(reference_checkpoint) != reference_integrity
    ):
        raise FileNotFoundError("checkpoint reference files are missing")
    integrity = verify_training_checkpoint(reference_checkpoint)
    _validate_integrity(
        integrity,
        expected_step=expected_step,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    for label, path in (
        ("checkpoint", reference_checkpoint),
        ("integrity", reference_integrity),
    ):
        actual = _source(path)
        recorded = reference.get(label)
        if not isinstance(recorded, dict):
            raise ValueError(f"checkpoint reference {label} identity is missing")
        for key in ("path", "bytes", "sha256", "device", "inode"):
            if actual[key] != recorded.get(key):
                raise ValueError(f"checkpoint reference {label} identity changed: {key}")
        source_path = Path(str(source.get(label, {}).get("path", "")))
        if source_path.exists() and not os.path.samefile(source_path, path):
            raise ValueError(f"checkpoint reference {label} is no longer the source inode")
    return report


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        existing = _read_json(output)
        verify_checkpoint_reference(
            existing,
            expected_step=args.expected_step,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            expected_reason=args.reason,
        )
    else:
        report = preserve_checkpoint_reference(
            checkpoint=Path(args.checkpoint),
            reference_dir=Path(args.reference_dir),
            expected_step=args.expected_step,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            reason=args.reason,
        )
        write_json_report(output, report)
    print(f"checkpoint reference verified: {output}")


if __name__ == "__main__":
    main()
