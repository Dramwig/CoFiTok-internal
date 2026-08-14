from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.generation.capacity_probe_result import build_capacity_probe_result
from cofitok.generation.capacity_scaling_decision import (
    build_capacity_scaling_decision,
)
from cofitok.generation.capacity_scaling_execution import (
    CAPACITY_SCALING_LAUNCH_BOUNDARY,
    CAPACITY_SCALING_SOURCE_NAMES,
    CAPACITY_SCALING_STORAGE_STAGE,
    build_capacity_scaling_launch_receipt,
    validate_capacity_scaling_launch_receipt,
)
from test_generation_capacity_probe_execution import (
    BRANCH,
    OUTPUT_ROOT,
    REVISION,
    STORAGE_PATH,
    _identity,
    _launch_receipt,
    _standing,
)
from test_generation_capacity_probe_result import _kwargs
from scripts.build_generation_milestone_report import (
    expected_source_report_suffixes,
)


DECISION_GIT = {
    "revision": "d" * 40,
    "branch": "scale/generation-capacity-scaling-decision-v1",
    "tracked_dirty": False,
}
EXECUTION_GIT = {
    "revision": "e" * 40,
    "tree": "1" * 40,
    "branch": "scale/generation-capacity-scaling-execution-v1",
    "tracked_dirty": False,
}
TRAINING_GIT = {
    "revision": REVISION,
    "tree": "2" * 40,
    "branch": BRANCH,
    "tracked_dirty": False,
}
ROOT = Path(__file__).resolve().parents[1]


def _result() -> dict:
    return build_capacity_probe_result(
        **_kwargs(
            fids={
                "base128_cofitok": 180.0,
                "base128_dense_identity": 175.0,
                "base256_cofitok": 160.0,
                "base256_dense_identity": 165.0,
            }
        )
    )


def _decision(result: dict) -> dict:
    return build_capacity_scaling_decision(
        capacity_probe_result=result,
        capacity_probe_result_identity=_identity("/evidence/result.json", "e"),
        standing_authorization=_standing(),
        standing_authorization_identity=_identity("/evidence/standing.json", "f"),
        decision_git=DECISION_GIT,
        expected_capacity_revision=REVISION,
        expected_capacity_branch=BRANCH,
    )


def _storage() -> dict:
    return {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": CAPACITY_SCALING_STORAGE_STAGE,
        "status": "pass",
        "git": {
            "revision": EXECUTION_GIT["revision"],
            "branch": EXECUTION_GIT["branch"],
            "tracked_dirty": False,
        },
        "filesystem": {"path": STORAGE_PATH, "free_bytes": 200 * 1024**3},
        "plan": {
            "sample_count": 4_096,
            "checkpoint_count": 6,
            "checkpoint_size_multiplier": 1.0,
            "required_free_bytes": 100 * 1024**3,
        },
        "headroom_bytes": 100 * 1024**3,
    }


def _inputs() -> dict:
    result = _result()
    decision = _decision(result)
    launch = _launch_receipt()
    result_identity = decision["source_evidence"]["capacity_probe_result"]
    launch_identity = result["source_reports"]["launch_receipt"]
    sources = {
        "capacity_scaling_decision": _identity("/evidence/decision.json", "6"),
        "capacity_probe_result": result_identity,
        "capacity_probe_launch_receipt": launch_identity,
        "cofitok_config": launch["source_reports"]["cofitok_config"],
        "dense_config": launch["source_reports"]["dense_config"],
        "storage_capacity": _identity("/evidence/storage.json", "7"),
    }
    assert set(sources) == CAPACITY_SCALING_SOURCE_NAMES
    resume = {
        method: {
            "checkpoint": copy.deepcopy(
                decision["selection"]["resume_sources"][method]["checkpoint"]
            ),
            "checkpoint_integrity_manifest": copy.deepcopy(
                decision["selection"]["resume_sources"][method][
                    "checkpoint_integrity_manifest"
                ]
            ),
        }
        for method in ("cofitok", "dense_identity")
    }
    return {
        "decision": decision,
        "capacity_probe_result": result,
        "capacity_probe_launch_receipt": launch,
        "storage_capacity": _storage(),
        "source_identities": sources,
        "resume_checkpoints": resume,
        "execution_git": EXECUTION_GIT,
        "training_git": TRAINING_GIT,
        "expected_decision_revision": DECISION_GIT["revision"],
        "expected_decision_branch": DECISION_GIT["branch"],
        "output_root": OUTPUT_ROOT,
        "storage_path": STORAGE_PATH,
        "training_project": "/execution/capacity-probe-training",
        "gpu_idle_at_launch": True,
        "relevant_processes_absent_at_launch": True,
    }


def test_capacity_scaling_launch_receipt_binds_exact_10k_to_50k_resume() -> None:
    report = build_capacity_scaling_launch_receipt(**_inputs())
    assert report["authorization_boundary"] == CAPACITY_SCALING_LAUNCH_BOUNDARY
    assert report["selection"]["resume_from_step"] == 10_000
    assert report["selection"]["stop_after_step"] == 50_000
    assert report["authorization_boundary"][
        "configured_100k_completion_allowed"
    ] is False
    evidence = validate_capacity_scaling_launch_receipt(
        report,
        expected_execution_revision=EXECUTION_GIT["revision"],
        expected_execution_tree=EXECUTION_GIT["tree"],
        expected_execution_branch=EXECUTION_GIT["branch"],
        expected_training_revision=TRAINING_GIT["revision"],
        expected_training_tree=TRAINING_GIT["tree"],
        expected_training_branch=TRAINING_GIT["branch"],
        expected_output_root=OUTPUT_ROOT,
    )
    assert evidence["stop_after_step"] == 50_000


def test_capacity_scaling_launch_rejects_busy_gpu_or_resume_drift() -> None:
    inputs = _inputs()
    inputs["gpu_idle_at_launch"] = False
    with pytest.raises(ValueError, match="idle GPU"):
        build_capacity_scaling_launch_receipt(**inputs)

    inputs = _inputs()
    inputs["resume_checkpoints"]["cofitok"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact resume source differs"):
        build_capacity_scaling_launch_receipt(**inputs)


def test_capacity_scaling_launch_rejects_authorization_escalation() -> None:
    inputs = _inputs()
    inputs["decision"] = copy.deepcopy(inputs["decision"])
    inputs["decision"]["execution_authorization"][
        "configured_100k_completion_allowed"
    ] = True
    with pytest.raises(ValueError, match="execution scope differs"):
        build_capacity_scaling_launch_receipt(**inputs)

    report = build_capacity_scaling_launch_receipt(**_inputs())
    report["authorization_boundary"] = copy.deepcopy(
        report["authorization_boundary"]
    )
    report["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_scaling_launch_receipt(
            report,
            expected_execution_revision=EXECUTION_GIT["revision"],
            expected_execution_tree=EXECUTION_GIT["tree"],
            expected_execution_branch=EXECUTION_GIT["branch"],
            expected_training_revision=TRAINING_GIT["revision"],
            expected_training_tree=TRAINING_GIT["tree"],
            expected_training_branch=TRAINING_GIT["branch"],
            expected_output_root=OUTPUT_ROOT,
        )


@pytest.mark.parametrize(
    "entrypoint",
    (
        "scripts/build_generation_capacity_scaling_launch_receipt.py",
        "scripts/verify_generation_capacity_scaling_launch_receipt.py",
        "scripts/validate_generation_capacity_scaling_training.py",
        "scripts/verify_generation_capacity_scaling_training.py",
    ),
)
def test_capacity_scaling_execution_entrypoints_import(
    entrypoint: str,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "capacity" in result.stdout.lower()


def test_capacity_scaling_milestone_profile_binds_250m_run_paths() -> None:
    suffixes = expected_source_report_suffixes(
        50_000,
        source_profile="capacity_scaling",
    )
    assert "base256_cofitok/milestones/step_00050000" in suffixes[
        "cofitok_generation"
    ]
    assert "base256_dense_identity/milestones/step_00050000" in suffixes[
        "dense_generation"
    ]
