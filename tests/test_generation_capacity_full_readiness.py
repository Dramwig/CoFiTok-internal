from __future__ import annotations

import copy

import pytest

from cofitok.generation import capacity_full_readiness as readiness
from cofitok.generation.capacity_full_readiness_decision import (
    CAPACITY_FULL_READINESS_DECISION_BOUNDARY,
    EXPECTED_PARAMETER_COUNTS,
)


READINESS_REVISION = "1" * 40
READINESS_TREE = "2" * 40
READINESS_BRANCH = "scale/readiness"
DECISION_REVISION = "3" * 40
DECISION_TREE = "4" * 40
DECISION_BRANCH = "scale/decision"
RESULT_REVISION = "5" * 40
RESULT_TREE = "6" * 40
RESULT_BRANCH = "scale/result"
SHA256 = "7" * 64
FULL_ROOT = "/checkpoints/stability_capacity_full_300k_v1"


def _identity(path: str, *, digest: str = SHA256) -> dict[str, object]:
    return {"path": path, "bytes": 10, "sha256": digest}


def _storage() -> dict[str, object]:
    checkpoint_bytes = 1_100_000_000
    checkpoint_count = 16
    sample_count = readiness.FULL_COMPLETION_SAMPLE_RESERVE
    sample_bytes = 256 * 1024
    additional = 16 * 1024**3
    safety = 64 * 1024**3
    required = (
        checkpoint_bytes * checkpoint_count
        + sample_count * sample_bytes
        + additional
        + safety
    )
    free = required + 1_000_000_000
    used = 2_000_000_000
    return {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": "full_training",
        "status": "pass",
        "git": {
            "revision": READINESS_REVISION,
            "branch": READINESS_BRANCH,
            "tracked_dirty": False,
        },
        "filesystem": {
            "path": "/checkpoints",
            "total_bytes": used + free,
            "used_bytes": used,
            "free_bytes": free,
        },
        "plan": {
            "checkpoint_count": checkpoint_count,
            "reference_checkpoint_bytes_each": checkpoint_bytes,
            "checkpoint_size_multiplier": 1.0,
            "checkpoint_bytes_each": checkpoint_bytes,
            "checkpoint_reserve_bytes": checkpoint_count * checkpoint_bytes,
            "sample_count": sample_count,
            "estimated_sample_bytes_each": sample_bytes,
            "sample_reserve_bytes": sample_count * sample_bytes,
            "additional_bytes": additional,
            "safety_margin_bytes": safety,
            "required_free_bytes": required,
        },
        "headroom_bytes": free - required,
    }


def _kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    target_ids = {
        "cofitok": _identity("/previous/target_cofitok.json", digest="8" * 64),
        "dense_identity": _identity(
            "/previous/target_dense.json", digest="9" * 64
        ),
    }
    current_ids = {
        "cofitok": _identity("/current/target_cofitok.json", digest="8" * 64),
        "dense_identity": _identity(
            "/current/target_dense.json", digest="9" * 64
        ),
    }
    decision_evidence = {
        "readiness_plan": {
            "stage": "stability_full",
            "full_output_root": FULL_ROOT,
            "benchmark_root": f"{FULL_ROOT}/reports/runtime_benchmark",
            "storage_path": "/checkpoints",
        },
        "execution_authorization": {
            "readiness_gpu_runtime_benchmark_allowed": True,
            "readiness_artifact_build_allowed": True,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "training_run_dirs": {
            "cofitok": f"{FULL_ROOT}/cofitok",
            "dense_identity": f"{FULL_ROOT}/dense_identity",
        },
    }
    monkeypatch.setattr(
        readiness,
        "validate_capacity_full_readiness_decision",
        lambda *args, **kwargs: copy.deepcopy(decision_evidence),
    )
    decision = {
        "authorization_boundary": CAPACITY_FULL_READINESS_DECISION_BOUNDARY,
        "execution_authorization": copy.deepcopy(
            decision_evidence["execution_authorization"]
        ),
        "source_reports": {
            "target_cofitok_config": target_ids["cofitok"],
            "target_dense_identity_config": target_ids["dense_identity"],
        },
        "fresh_training_contract": {
            "training_state_absent_at_decision": True,
            "capacity_100k_checkpoint_resume_allowed": False,
            "target_start_step": 0,
            "target_steps": 300_000,
            "exact_resume_allowed_only_from_target_config_checkpoints": True,
            "training_run_dirs": copy.deepcopy(
                decision_evidence["training_run_dirs"]
            ),
        },
    }
    config_validation = {
        "status": "pass",
        "mismatches": [],
        "cofitok": {"parameter_count": EXPECTED_PARAMETER_COUNTS["cofitok"]},
        "dense": {
            "parameter_count": EXPECTED_PARAMETER_COUNTS["dense_identity"]
        },
        "matched_backbone": {"base_channels": 256},
        "training_recipe": {
            "stage": "stability_full",
            "valid": True,
            "schema": "stability_full_v1",
        },
        "relative_parameter_gap": 0.00007483957,
    }
    runtime_evidence = {
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": 64,
        "baseline": {"micro_batch_size": 1, "gradient_accumulation_steps": 64},
        "candidate_count": 5,
        "runtime_environment_sha256": "a" * 64,
        "dataset_identity_sha256": "b" * 64,
        "estimated_speedup_over_baseline": 1.25,
    }
    return {
        "readiness_decision": decision,
        "readiness_decision_identity": _identity("/reports/decision.json"),
        "target_config_identities": current_ids,
        "config_validation": config_validation,
        "config_validation_identity": _identity("/reports/config.json"),
        "storage_report": _storage(),
        "storage_report_identity": _identity("/reports/storage.json"),
        "runtime_selection": {"schema_version": 3, "baseline": {}},
        "runtime_selection_identity": _identity("/reports/runtime.json"),
        "runtime_evidence": runtime_evidence,
        "readiness_git": {
            "revision": READINESS_REVISION,
            "tree": READINESS_TREE,
            "branch": READINESS_BRANCH,
            "tracked_dirty": False,
        },
        "expected_readiness_revision": READINESS_REVISION,
        "expected_readiness_tree": READINESS_TREE,
        "expected_readiness_branch": READINESS_BRANCH,
        "expected_decision_revision": DECISION_REVISION,
        "expected_decision_tree": DECISION_TREE,
        "expected_decision_branch": DECISION_BRANCH,
        "expected_result_revision": RESULT_REVISION,
        "expected_result_tree": RESULT_TREE,
        "expected_result_branch": RESULT_BRANCH,
    }


def test_capacity_full_readiness_authorizes_only_fresh_separate_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = readiness.build_capacity_full_readiness(**_kwargs(monkeypatch))
    evidence = readiness.validate_capacity_full_readiness(
        report,
        expected_readiness_revision=READINESS_REVISION,
        expected_readiness_tree=READINESS_TREE,
        expected_readiness_branch=READINESS_BRANCH,
    )
    assert evidence["full_training_launch_allowed"] is True
    assert report["fresh_training_contract"][
        "capacity_100k_checkpoint_resume_allowed"
    ] is False
    assert report["fresh_training_contract"]["target_start_step"] == 0
    assert report["storage_capacity"]["checkpoint_size_multiplier"] == 1.0
    assert report["storage_capacity"]["sample_count"] == 116_640
    assert "reference_checkpoints" not in report["storage_capacity"]
    assert report["authorization_boundary"][
        "separate_training_launch_receipt_required"
    ] is True


def test_capacity_full_readiness_rejects_weakened_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _kwargs(monkeypatch)
    kwargs["storage_report"]["plan"]["checkpoint_size_multiplier"] = 1.01
    with pytest.raises(ValueError, match="storage scaling differs"):
        readiness.build_capacity_full_readiness(**kwargs)


def test_capacity_full_readiness_rejects_malformed_runtime_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _kwargs(monkeypatch)
    kwargs["runtime_evidence"]["micro_batch_size"] = "2"
    with pytest.raises(ValueError, match="batch selection is malformed"):
        readiness.build_capacity_full_readiness(**kwargs)


def test_capacity_full_readiness_validator_rejects_launch_field_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = readiness.build_capacity_full_readiness(**_kwargs(monkeypatch))
    report["decision_authorization"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="runtime/fresh contract differs"):
        readiness.validate_capacity_full_readiness(
            report,
            expected_readiness_revision=READINESS_REVISION,
            expected_readiness_tree=READINESS_TREE,
            expected_readiness_branch=READINESS_BRANCH,
        )
