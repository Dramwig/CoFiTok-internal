from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256, write_json_report
from scripts.preserve_generation_checkpoint_reference import (
    preserve_checkpoint_reference,
    verify_checkpoint_reference,
)


REVISION = "a" * 40
BRANCH = "scale/test"


def _checkpoint(root: Path, *, step: int = 10_000) -> Path:
    checkpoint = root / f"checkpoint_step_{step:08d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"verified checkpoint bytes")
    write_json_report(
        checkpoint.with_name(f"{checkpoint.name}.integrity.json"),
        {
            "schema_version": 1,
            "checkpoint": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_format_version": 1,
            "step": step,
            "git_revision": REVISION,
            "git_branch": BRANCH,
            "git_dirty": False,
        },
    )
    return checkpoint


def test_preserves_checkpoint_and_sidecar_as_same_inode(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    reference_dir = tmp_path / "references" / "cofitok"

    report = preserve_checkpoint_reference(
        checkpoint=checkpoint,
        reference_dir=reference_dir,
        expected_step=10_000,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        reason="matched capacity comparison",
    )

    target = reference_dir / checkpoint.name
    assert os.path.samefile(checkpoint, target)
    assert os.path.samefile(
        checkpoint.with_name(f"{checkpoint.name}.integrity.json"),
        reference_dir / f"{checkpoint.name}.integrity.json",
    )
    assert report["storage"]["additional_checkpoint_data_blocks_required"] is False
    assert report["authorization_boundary"]["training_launch_allowed"] is False


def test_replay_is_idempotent(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    kwargs = {
        "checkpoint": checkpoint,
        "reference_dir": tmp_path / "references" / "cofitok",
        "expected_step": 10_000,
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
        "reason": "matched capacity comparison",
    }

    assert preserve_checkpoint_reference(**kwargs) == preserve_checkpoint_reference(**kwargs)


def test_reference_verifies_after_original_source_is_unlinked(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    report = preserve_checkpoint_reference(
        checkpoint=checkpoint,
        reference_dir=tmp_path / "references" / "cofitok",
        expected_step=10_000,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        reason="matched capacity comparison",
    )
    checkpoint.unlink()
    checkpoint.with_name(f"{checkpoint.name}.integrity.json").unlink()

    assert verify_checkpoint_reference(
        report,
        expected_step=10_000,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_reason="matched capacity comparison",
    ) == report


def test_rejects_wrong_git_or_step(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    common = {
        "checkpoint": checkpoint,
        "reference_dir": tmp_path / "references" / "cofitok",
        "reason": "matched capacity comparison",
    }
    with pytest.raises(ValueError, match="step differs"):
        preserve_checkpoint_reference(
            **common,
            expected_step=9_999,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )
    with pytest.raises(ValueError, match="Git identity differs"):
        preserve_checkpoint_reference(
            **common,
            expected_step=10_000,
            expected_revision="b" * 40,
            expected_branch=BRANCH,
        )


def test_rejects_existing_non_hardlink_target(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    reference_dir = tmp_path / "references" / "cofitok"
    reference_dir.mkdir(parents=True)
    (reference_dir / checkpoint.name).write_bytes(checkpoint.read_bytes())

    with pytest.raises(ValueError, match="not the same inode"):
        preserve_checkpoint_reference(
            checkpoint=checkpoint,
            reference_dir=reference_dir,
            expected_step=10_000,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            reason="matched capacity comparison",
        )


def test_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        preserve_checkpoint_reference(
            checkpoint=checkpoint,
            reference_dir=tmp_path / "references" / "cofitok",
            expected_step=10_000,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            reason="matched capacity comparison",
        )
