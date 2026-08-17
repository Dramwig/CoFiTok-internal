from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "artifacts"
    / "reports"
    / "generation"
    / "stability_full_data_quality_bridge_ipc_recovery_v2"
    / "deployment_receipt.json"
)
SUPERVISOR = (
    ROOT
    / "scripts"
    / "run_generation_quality_bridge_ipc_recovery_supervisor_v2.py"
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def load_receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_receipt_binds_the_exact_tracked_supervisor_and_git_chain() -> None:
    receipt = load_receipt()
    deployment = receipt["deployment"]
    git = receipt["git"]
    bundle = receipt["bundle"]
    training = receipt["training"]
    remote_checkout = receipt["remote_checkout"]

    assert receipt["schema_version"] == 1
    assert receipt["status"] == "pass"
    assert receipt["role"] == (
        "cofitok_quality_bridge_ipc_recovery_v2_deployment_receipt"
    )
    assert datetime.fromisoformat(str(receipt["created_at"])).utcoffset() is not None

    assert SUPERVISOR.stat().st_size == deployment["deployed_source_bytes"]
    assert file_sha256(SUPERVISOR) == deployment["deployed_source_sha256"]
    assert git["source_path"] == SUPERVISOR.relative_to(ROOT).as_posix()

    assert bundle["advertised_head"] == git["record_commit"]
    assert remote_checkout["head"] == bundle["advertised_head"]
    assert remote_checkout["branch"] == git["branch"]
    assert remote_checkout["tracked_dirty"] is False
    assert bundle["prerequisite"] == git["training_parent"]
    assert bundle["prerequisite"] == training["revision"]
    assert bundle["verify_status"] == "pass"

    for field in (
        "code_commit",
        "record_commit",
        "deployment_evidence_commit",
        "training_parent",
    ):
        assert GIT_COMMIT_RE.fullmatch(str(git[field]))


def test_receipt_preserves_the_non_authorizing_recovery_boundary() -> None:
    receipt = load_receipt()
    scope = receipt["scope"]

    assert scope == {
        "full_300k_launch_allowed": False,
        "full_training_launch_allowed": False,
        "quality_bridge_exact_resume_allowed": True,
        "release_authorization_allowed": False,
        "retryable_signature": (
            "watchdog_child_failed_exit_1_and_exact_pin_memory_"
            "resource_sharer_filenotfound"
        ),
        "unrelated_gpu_processes_must_not_be_signaled": True,
    }
    assert receipt["deployment"]["status_at_receipt"] == "observing"
    assert receipt["deployment"]["supervisor_attempt_at_receipt"] == 0

    for section, fields in {
        "authorization": ("standing_authorization_sha256",),
        "bundle": ("sha256",),
        "deployment": ("deployed_source_sha256",),
        "formal_checkout_observation": ("porcelain_sha256",),
        "runbook": (
            "execution_approval_sha256",
            "launch_receipt_sha256",
            "preparation_sha256",
            "runbook_sha256",
        ),
    }.items():
        values = receipt[section]
        for field in fields:
            assert SHA256_RE.fullmatch(str(values[field]))
