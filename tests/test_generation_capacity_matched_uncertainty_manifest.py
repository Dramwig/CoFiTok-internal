from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.generation.capacity_completion_result import (
    CAPACITY_COMPLETION_RESULT_BOUNDARY,
    CAPACITY_COMPLETION_RESULT_ROLE,
    CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.inference_replay import file_identity
from scripts import audit_generation_matched_uncertainty as audit
from scripts import (
    build_generation_capacity_matched_uncertainty_manifest as builder,
)
from scripts import run_generation_matched_uncertainty_waiter as base_waiter


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
RUNTIME_SHA = "e" * 64
EXECUTION_GIT = {
    "revision": "1" * 40,
    "tree": "2" * 40,
    "branch": "scale/capacity-completion-execution",
    "tracked_dirty": False,
}
TRAINING_GIT = {
    "revision": "3" * 40,
    "tree": "4" * 40,
    "branch": "scale/generation-large-capacity",
    "tracked_dirty": False,
}
RESULT_GIT = {
    "revision": "5" * 40,
    "tree": "6" * 40,
    "branch": "analysis/capacity-completion-result",
    "tracked_dirty": False,
}
EVALUATION_GIT = {
    "revision": "7" * 40,
    "tree": "8" * 40,
    "branch": "scale/generative-system",
    "tracked_dirty": False,
}


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _sampling(prefix_budget: int, *, count: int) -> dict[str, Any]:
    return {
        "num_samples": count,
        "start_index": 0,
        "prefix_budgets": [prefix_budget],
        "image_shape": [3, 256, 256],
        "sample_steps": 250 if count == 50_000 else 100,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "weights": "ema",
        "precision": "bf16",
        "random_stream": {
            "batch_size_invariant": True,
            "prefix_budgets_share_stream": True,
            "resume_index_invariant": True,
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
        },
    }


def _metrics_report(
    tmp_path: Path,
    *,
    method: str,
    prefix_budget: int,
    count: int,
    checkpoint_step: int,
    fid: float,
    evaluation_git: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    real_dir = tmp_path / "real"
    generated_dir = tmp_path / method / f"prefix_{prefix_budget}"
    sampling_path = tmp_path / method / "sampling_report.json"
    sampling_identity = _write(sampling_path, {"method": method})
    real_set = {
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "root": real_dir.resolve().as_posix(),
        "image_count": 50_000,
        "sha256": SHA_A,
    }
    payload: dict[str, Any] = {
        "schema_version": 3,
        "role": "generation_directory_metrics_report",
        "protocol": "torch_fidelity_directory_metrics",
        "status": "completed",
        "paths": {
            "generated_dir": generated_dir.resolve().as_posix(),
            "real_dir": real_dir.resolve().as_posix(),
            "sampling_report": sampling_path.resolve().as_posix(),
        },
    }
    if evaluation_git is not None:
        sample_sha = SHA_C if method == "cofitok" else "9" * 64
        checkpoint_sha = SHA_B if method == "cofitok" else SHA_D
        payload.update(
            {
                "git": {
                    "revision": evaluation_git["revision"],
                    "branch": evaluation_git["branch"],
                    "tracked_dirty": False,
                },
                "counts": {
                    "generated_image_count": count,
                    "real_image_count": 50_000,
                },
                "parameters": {"precision_recall_enabled": True},
                "real_set": real_set,
                "runtime_environment_sha256": RUNTIME_SHA,
                "sample_provenance": {
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_sha256": checkpoint_sha,
                    "sample_set_sha256": sample_sha,
                    "weights": "ema",
                    "selected_prefix_budget": prefix_budget,
                    "sampling": _sampling(prefix_budget, count=count),
                    "report_identity": sampling_identity,
                },
                "metrics": {"frechet_inception_distance": fid},
            }
        )
    path = tmp_path / method / "generation_metrics.json"
    identity = _write(path, payload)
    return path, identity, {
        "real_set": real_set,
        "sampling_identity": sampling_identity,
    }


def _audit_args(manifest_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        execution_manifest=manifest_path,
        real_dir=None,
        cofitok_generated_dir=None,
        dense_generated_dir=None,
        cofitok_sampling_report=None,
        dense_sampling_report=None,
        cofitok_metrics_report=None,
        dense_metrics_report=None,
        output=None,
        cache_root=None,
        real_cache_name=None,
        batch_size=None,
        min_samples=None,
        block_size=None,
        real_fold_count=None,
        bootstrap_repetitions=None,
        seed=None,
        cpu=None,
        resume=False,
        require_advantage=False,
    )


def test_builds_capacity_completion_100k_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cofitok_metrics, cofitok_meta = _metrics_report(
        tmp_path,
        method="cofitok",
        prefix_budget=8,
        count=10_000,
        checkpoint_step=100_000,
        fid=70.0,
        evaluation_git=None,
    )
    _, dense_metrics, dense_meta = _metrics_report(
        tmp_path,
        method="dense_identity",
        prefix_budget=1,
        count=10_000,
        checkpoint_step=100_000,
        fid=80.0,
        evaluation_git=None,
    )
    methods = {
        "cofitok": {
            "checkpoint_step": 100_000,
            "sample_count": 10_000,
            "selected_prefix_budget": 8,
            "weights": "ema",
            "sampling": _sampling(8, count=10_000),
            "checkpoint_sha256": SHA_B,
            "sample_set_sha256": SHA_C,
            "metrics_runtime_environment_sha256": RUNTIME_SHA,
            "real_set": cofitok_meta["real_set"],
            "sampling_report": cofitok_meta["sampling_identity"],
            "fid": 70.0,
        },
        "dense_identity": {
            "checkpoint_step": 100_000,
            "sample_count": 10_000,
            "selected_prefix_budget": 1,
            "weights": "ema",
            "sampling": _sampling(1, count=10_000),
            "checkpoint_sha256": SHA_D,
            "sample_set_sha256": "9" * 64,
            "metrics_runtime_environment_sha256": RUNTIME_SHA,
            "real_set": dense_meta["real_set"],
            "sampling_report": dense_meta["sampling_identity"],
            "fid": 80.0,
        },
    }
    result = {
        "schema_version": CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION,
        "role": CAPACITY_COMPLETION_RESULT_ROLE,
        "status": "completed",
        "execution_git": EXECUTION_GIT,
        "training_git": TRAINING_GIT,
        "result_builder_git": RESULT_GIT,
        "source_reports": {
            "cofitok_generation": cofitok_metrics,
            "dense_generation": dense_metrics,
        },
        "terminal": {"methods": methods},
        "quality_screen": {"status": "pass", "failed_checks": []},
        "decision_support": {"recommended_next_stage": {"id": "review"}},
        "authorization_boundary": CAPACITY_COMPLETION_RESULT_BOUNDARY,
    }
    anchor_path = tmp_path / "reports" / "capacity_completion_100k_result.json"
    anchor = _write(anchor_path, result)
    monkeypatch.setattr(
        builder,
        "validate_capacity_completion_100k_result",
        lambda *_args, **_kwargs: {"status": "completed"},
    )

    output_root = tmp_path / "uncertainty"
    manifest = builder.build_manifest(
        source_kind="capacity_completion_100k",
        source_anchor_path=anchor_path,
        expected_source_anchor_sha256=anchor["sha256"],
        expected_execution_revision=EXECUTION_GIT["revision"],
        expected_execution_tree=EXECUTION_GIT["tree"],
        expected_execution_branch=EXECUTION_GIT["branch"],
        expected_training_revision=TRAINING_GIT["revision"],
        expected_training_tree=TRAINING_GIT["tree"],
        expected_training_branch=TRAINING_GIT["branch"],
        expected_result_revision=RESULT_GIT["revision"],
        expected_result_tree=RESULT_GIT["tree"],
        expected_result_branch=RESULT_GIT["branch"],
        output_root=output_root,
    )
    path = output_root / "reports" / "manifest.json"
    identity = _write(path, manifest)
    assert manifest["stream_id"] == (
        "full_data_capacity_completion_terminal_100k_00000000_00010000"
    )
    assert manifest["arguments"]["real_fold_count"] == 5
    assert manifest["capacity_source"]["anchor"] == anchor
    hydrated = audit.hydrate_execution_arguments(_audit_args(path))
    assert audit.validate_execution_manifest(hydrated) is not None
    validated = base_waiter.validate_execution_manifest(
        path,
        expected_sha256=identity["sha256"],
        output_root=output_root,
    )
    assert validated["checkpoint_step"] == 100_000


def test_builds_capacity_full_300k_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cofitok_metrics, _ = _metrics_report(
        tmp_path,
        method="cofitok",
        prefix_budget=8,
        count=50_000,
        checkpoint_step=300_000,
        fid=4.0,
        evaluation_git=EVALUATION_GIT,
    )
    _, dense_metrics, _ = _metrics_report(
        tmp_path,
        method="dense_identity",
        prefix_budget=1,
        count=50_000,
        checkpoint_step=300_000,
        fid=5.0,
        evaluation_git=EVALUATION_GIT,
    )
    gate = {
        "stage": "full",
        "source_profile": "capacity_full",
        "status": "pass",
        "decision": "large_scale_generation_ready",
        "provenance_contract": {
            "training_revision": TRAINING_GIT["revision"],
            "training_branch": TRAINING_GIT["branch"],
            "evaluation_revision": EVALUATION_GIT["revision"],
            "evaluation_branch": EVALUATION_GIT["branch"],
        },
        "gates": [{"name": "absolute_fid", "passed": True}],
        "source_reports": {
            "cofitok_generation": cofitok_metrics,
            "dense_generation": dense_metrics,
        },
    }
    anchor_path = tmp_path / "full" / "reports" / "final_generation_gate.json"
    anchor = _write(anchor_path, gate)
    monkeypatch.setattr(
        builder,
        "verify_generation_gate_source_reports",
        lambda _gate: {
            "status": "verified",
            "stage": "full",
            "source_profile": "capacity_full",
            "source_reports": copy.deepcopy(gate["source_reports"]),
        },
    )

    output_root = tmp_path / "full-uncertainty"
    manifest = builder.build_manifest(
        source_kind="capacity_full_300k",
        source_anchor_path=anchor_path,
        expected_source_anchor_sha256=anchor["sha256"],
        expected_training_revision=TRAINING_GIT["revision"],
        expected_training_tree=TRAINING_GIT["tree"],
        expected_training_branch=TRAINING_GIT["branch"],
        expected_evaluation_revision=EVALUATION_GIT["revision"],
        expected_evaluation_tree=EVALUATION_GIT["tree"],
        expected_evaluation_branch=EVALUATION_GIT["branch"],
        output_root=output_root,
    )
    path = output_root / "reports" / "manifest.json"
    identity = _write(path, manifest)
    assert manifest["stream_id"] == (
        "capacity_full_terminal_300k_00000000_00050000"
    )
    assert manifest["arguments"]["min_samples"] == 50_000
    assert manifest["arguments"]["real_fold_count"] == 1
    validated = base_waiter.validate_execution_manifest(
        path,
        expected_sha256=identity["sha256"],
        output_root=output_root,
    )
    assert validated["end_index_exclusive"] == 50_000
    assert validated["checkpoint_step"] == 300_000


def test_capacity_full_rejects_unmatched_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cofitok_metrics, _ = _metrics_report(
        tmp_path,
        method="cofitok",
        prefix_budget=8,
        count=50_000,
        checkpoint_step=300_000,
        fid=4.0,
        evaluation_git=EVALUATION_GIT,
    )
    dense_path, dense_metrics, _ = _metrics_report(
        tmp_path,
        method="dense_identity",
        prefix_budget=1,
        count=50_000,
        checkpoint_step=300_000,
        fid=5.0,
        evaluation_git=EVALUATION_GIT,
    )
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    dense["sample_provenance"]["sampling"]["guidance_scale"] = 2.0
    dense_metrics = _write(dense_path, dense)
    gate = {
        "stage": "full",
        "source_profile": "capacity_full",
        "status": "pass",
        "decision": "large_scale_generation_ready",
        "provenance_contract": {
            "training_revision": TRAINING_GIT["revision"],
            "training_branch": TRAINING_GIT["branch"],
            "evaluation_revision": EVALUATION_GIT["revision"],
            "evaluation_branch": EVALUATION_GIT["branch"],
        },
        "gates": [{"name": "absolute_fid", "passed": True}],
        "source_reports": {
            "cofitok_generation": cofitok_metrics,
            "dense_generation": dense_metrics,
        },
    }
    anchor_path = tmp_path / "full" / "reports" / "final_generation_gate.json"
    anchor = _write(anchor_path, gate)
    monkeypatch.setattr(
        builder,
        "verify_generation_gate_source_reports",
        lambda _gate: {
            "status": "verified",
            "stage": "full",
            "source_profile": "capacity_full",
            "source_reports": copy.deepcopy(gate["source_reports"]),
        },
    )
    with pytest.raises(ValueError, match="sampling protocols are not matched"):
        builder.build_manifest(
            source_kind="capacity_full_300k",
            source_anchor_path=anchor_path,
            expected_source_anchor_sha256=anchor["sha256"],
            expected_training_revision=TRAINING_GIT["revision"],
            expected_training_tree=TRAINING_GIT["tree"],
            expected_training_branch=TRAINING_GIT["branch"],
            expected_evaluation_revision=EVALUATION_GIT["revision"],
            expected_evaluation_tree=EVALUATION_GIT["tree"],
            expected_evaluation_branch=EVALUATION_GIT["branch"],
            output_root=tmp_path / "uncertainty",
        )
