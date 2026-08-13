from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from cofitok.reporting import file_sha256, write_json_report


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
BRANCH = "scale/test"


def _checkpoint(root: Path, name: str) -> Path:
    checkpoint = root / name / "checkpoint_step_00010000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(name.encode())
    write_json_report(
        checkpoint.with_name(f"{checkpoint.name}.integrity.json"),
        {
            "schema_version": 1,
            "checkpoint": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_format_version": 1,
            "step": 10_000,
            "git_revision": REVISION,
            "git_branch": BRANCH,
            "git_dirty": False,
        },
    )
    return checkpoint


def test_waiter_preserves_complete_matched_set(tmp_path: Path) -> None:
    cofitok = _checkpoint(tmp_path / "runs", "cofitok")
    dense = _checkpoint(tmp_path / "runs", "dense_identity")
    reference = tmp_path / "references"
    receipts = tmp_path / "receipts"
    status = tmp_path / "status.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/wait_for_generation_checkpoint_references.py",
            "--source",
            f"cofitok={cofitok}",
            "--source",
            f"dense_identity={dense}",
            "--reference-root",
            str(reference),
            "--receipt-root",
            str(receipts),
            "--expected-step",
            "10000",
            "--expected-revision",
            REVISION,
            "--expected-branch",
            BRANCH,
            "--reason",
            "matched capacity comparison",
            "--status",
            str(status),
            "--poll-seconds",
            "1",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((".", "src"))},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert os.path.samefile(
        cofitok,
        reference / "cofitok" / "checkpoint_step_00010000.pt",
    )
    assert os.path.samefile(
        dense,
        reference / "dense_identity" / "checkpoint_step_00010000.pt",
    )
    assert payload["authorization_boundary"]["gpu_use_allowed"] is False
