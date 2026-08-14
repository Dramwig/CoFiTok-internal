from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.generation.capacity_completion_decision import (
    build_capacity_completion_decision,
)
from cofitok.generation.capacity_completion_execution import (
    CAPACITY_COMPLETION_LAUNCH_BOUNDARY,
    CAPACITY_COMPLETION_SOURCE_NAMES,
    CAPACITY_COMPLETION_STORAGE_STAGE,
    build_capacity_completion_launch_receipt,
    validate_capacity_completion_launch_receipt,
)
from cofitok.generation.capacity_scaling_execution import (
    build_capacity_scaling_launch_receipt,
)
from cofitok.generation.capacity_scaling_result import (
    build_capacity_scaling_50k_result,
)
from test_generation_capacity_completion_decision import (
    DECISION_GIT,
    _decision_kwargs,
)
from test_generation_capacity_probe_execution import _identity
from test_generation_capacity_scaling_execution import (
    EXECUTION_GIT as SCALING_EXECUTION_GIT,
    TRAINING_GIT,
    _inputs as scaling_launch_inputs,
)
from test_generation_capacity_scaling_result import _kwargs as result_kwargs


EXECUTION_GIT = {
    "revision": "5" * 40,
    "tree": "4" * 40,
    "branch": "scale/generation-capacity-completion-execution-v1",
    "tracked_dirty": False,
}


def _storage() -> dict:
    return {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": CAPACITY_COMPLETION_STORAGE_STAGE,
        "status": "pass",
        "git": {
            "revision": EXECUTION_GIT["revision"],
            "branch": EXECUTION_GIT["branch"],
            "tracked_dirty": False,
        },
        "filesystem": {
            "path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
            "free_bytes": 200 * 1024**3,
        },
        "plan": {
            "sample_count": 24_096,
            "checkpoint_count": 8,
            "checkpoint_size_multiplier": 1.0,
            "required_free_bytes": 100 * 1024**3,
        },
        "headroom_bytes": 100 * 1024**3,
    }


def _inputs() -> dict:
    result_args = result_kwargs()
    result = build_capacity_scaling_50k_result(**result_args)
    decision = build_capacity_completion_decision(
        **_decision_kwargs(result=result)
    )
    scaling_launch = build_capacity_scaling_launch_receipt(
        **scaling_launch_inputs()
    )
    output_root = result["output_root"]
    archives = {}
    for method in ("cofitok", "dense_identity"):
        source = decision["selection"]["resume_sources"][method]
        checkpoint = source["checkpoint"]
        integrity = source["checkpoint_integrity_manifest"]
        archive_path = (
            f"{output_root}/reports/capacity_completion_100k/"
            f"source_checkpoints/{method}/checkpoint_step_00050000.pt"
        )
        archives[method] = {
            "source_checkpoint": copy.deepcopy(checkpoint),
            "source_integrity_manifest": copy.deepcopy(integrity),
            "archive_checkpoint": {
                "path": archive_path,
                "bytes": checkpoint["bytes"],
                "sha256": checkpoint["sha256"],
            },
            "archive_integrity_manifest": {
                "path": f"{archive_path}.integrity.json",
                "bytes": integrity["bytes"],
                "sha256": integrity["sha256"],
            },
            "checkpoint_same_file": True,
            "integrity_same_file": True,
            "checkpoint_link_count": 2,
            "integrity_link_count": 2,
        }
    archive = {
        "schema_version": 1,
        "status": "pass",
        "role": "capacity_completion_50k_source_checkpoint_archive",
        "capacity_completion_decision": _identity("/evidence/completion.json", "1"),
        "output_root": output_root,
        "methods": archives,
        "authorization_boundary": {
            "source_checkpoint_mutation_allowed": False,
            "training_launch_allowed": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_or_release_allowed": False,
        },
    }
    sources = {
        "capacity_completion_decision": archive[
            "capacity_completion_decision"
        ],
        "capacity_scaling_50k_result": decision["source_evidence"][
            "capacity_scaling_50k_result"
        ],
        "capacity_scaling_50k_launch_receipt": _identity(
            "/evidence/scaling_launch.json", "2"
        ),
        "source_checkpoint_archive": _identity("/evidence/archive.json", "3"),
        "cofitok_config": scaling_launch["source_reports"]["cofitok_config"],
        "dense_config": scaling_launch["source_reports"]["dense_config"],
        "storage_capacity": _identity("/evidence/storage.json", "4"),
    }
    assert set(sources) == CAPACITY_COMPLETION_SOURCE_NAMES
    return {
        "decision": decision,
        "capacity_scaling_result": result,
        "capacity_scaling_launch_receipt": scaling_launch,
        "source_checkpoint_archive": archive,
        "storage_capacity": _storage(),
        "source_identities": sources,
        "execution_git": EXECUTION_GIT,
        "training_git": TRAINING_GIT,
        "expected_decision_revision": DECISION_GIT["revision"],
        "expected_decision_tree": DECISION_GIT["tree"],
        "expected_decision_branch": DECISION_GIT["branch"],
        "expected_scaling_execution_revision": SCALING_EXECUTION_GIT["revision"],
        "expected_scaling_execution_tree": SCALING_EXECUTION_GIT["tree"],
        "expected_scaling_execution_branch": SCALING_EXECUTION_GIT["branch"],
        "expected_training_revision": TRAINING_GIT["revision"],
        "expected_training_tree": TRAINING_GIT["tree"],
        "expected_training_branch": TRAINING_GIT["branch"],
        "output_root": output_root,
        "storage_path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
        "training_project": "/execution/capacity-training",
        "gpu_idle_at_launch": True,
        "relevant_processes_absent_at_launch": True,
    }


def _validate(report: dict) -> dict:
    return validate_capacity_completion_launch_receipt(
        report,
        expected_execution_revision=EXECUTION_GIT["revision"],
        expected_execution_tree=EXECUTION_GIT["tree"],
        expected_execution_branch=EXECUTION_GIT["branch"],
        expected_training_revision=TRAINING_GIT["revision"],
        expected_training_tree=TRAINING_GIT["tree"],
        expected_training_branch=TRAINING_GIT["branch"],
        expected_output_root=_inputs()["output_root"],
    )


def test_capacity_completion_launch_binds_exact_archived_50k_sources() -> None:
    report = build_capacity_completion_launch_receipt(**_inputs())
    assert report["selection"]["resume_from_step"] == 50_000
    assert report["selection"]["stop_after_step"] == 100_000
    assert report["selection"]["terminal_evaluation"][
        "sample_count_per_method"
    ] == 10_000
    assert report["authorization_boundary"] == CAPACITY_COMPLETION_LAUNCH_BOUNDARY
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert _validate(report)["execution_authorized"] is True


def test_capacity_completion_launch_rejects_busy_gpu_or_archive_drift() -> None:
    inputs = _inputs()
    inputs["gpu_idle_at_launch"] = False
    with pytest.raises(ValueError, match="idle exclusive GPU"):
        build_capacity_completion_launch_receipt(**inputs)

    inputs = _inputs()
    inputs["source_checkpoint_archive"]["methods"]["cofitok"][
        "archive_checkpoint"
    ]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive identity differs"):
        build_capacity_completion_launch_receipt(**inputs)


def test_capacity_completion_launch_validator_rejects_scope_escalation() -> None:
    report = build_capacity_completion_launch_receipt(**_inputs())
    report["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        _validate(report)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.archive_generation_capacity_completion_sources",
        "scripts.restore_generation_capacity_completion_sources",
        "scripts.verify_generation_capacity_completion_source_archive",
        "scripts.build_generation_capacity_completion_launch_receipt",
        "scripts.verify_generation_capacity_completion_launch_receipt",
    ],
)
def test_capacity_completion_execution_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)
