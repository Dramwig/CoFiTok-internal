from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.inference_replay import file_identity
from scripts import audit_generation_matched_uncertainty as audit
from scripts import (
    build_generation_quality_bridge_matched_uncertainty_manifest as builder,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
RUNTIME_SHA = "e" * 64
REVISION = "f" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"


def _write(path: Path, payload: dict[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _sampling(prefix_budget: int) -> dict[str, object]:
    return {
        "num_samples": 10_000,
        "start_index": 0,
        "prefix_budgets": [prefix_budget],
        "sample_steps": 100,
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


def _case(tmp_path: Path) -> dict[str, object]:
    real_dir = tmp_path / "real"
    cofitok_dir = tmp_path / "cofitok" / "prefix_8"
    dense_dir = tmp_path / "dense" / "prefix_1"
    cofitok_sampling_path = tmp_path / "cofitok" / "sampling_report.json"
    dense_sampling_path = tmp_path / "dense" / "sampling_report.json"
    cofitok_sampling = _write(cofitok_sampling_path, {"method": "cofitok"})
    dense_sampling = _write(dense_sampling_path, {"method": "dense"})

    real_set = {
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "root": real_dir.resolve().as_posix(),
        "image_count": 50_000,
        "sha256": SHA_A,
    }
    cofitok_metrics_path = tmp_path / "cofitok" / "generation_metrics.json"
    dense_metrics_path = tmp_path / "dense" / "generation_metrics.json"
    cofitok_metrics = _write(
        cofitok_metrics_path,
        {
            "schema_version": 3,
            "role": "generation_directory_metrics_report",
            "protocol": "torch_fidelity_directory_metrics",
            "status": "completed",
            "paths": {
                "generated_dir": cofitok_dir.resolve().as_posix(),
                "real_dir": real_dir.resolve().as_posix(),
                "sampling_report": cofitok_sampling_path.resolve().as_posix(),
            },
        },
    )
    dense_metrics = _write(
        dense_metrics_path,
        {
            "schema_version": 3,
            "role": "generation_directory_metrics_report",
            "protocol": "torch_fidelity_directory_metrics",
            "status": "completed",
            "paths": {
                "generated_dir": dense_dir.resolve().as_posix(),
                "real_dir": real_dir.resolve().as_posix(),
                "sampling_report": dense_sampling_path.resolve().as_posix(),
            },
        },
    )

    methods = {
        "cofitok": {
            "checkpoint_step": 100_000,
            "sample_count": 10_000,
            "selected_prefix_budget": 8,
            "weights": "ema",
            "sampling": _sampling(8),
            "checkpoint_sha256": SHA_B,
            "sample_set_sha256": SHA_C,
            "metrics_runtime_environment_sha256": RUNTIME_SHA,
            "real_set": real_set,
            "sampling_report": cofitok_sampling,
            "fid": 72.0,
        },
        "dense_identity": {
            "checkpoint_step": 100_000,
            "sample_count": 10_000,
            "selected_prefix_budget": 1,
            "weights": "ema",
            "sampling": _sampling(1),
            "checkpoint_sha256": SHA_D,
            "sample_set_sha256": "1" * 64,
            "metrics_runtime_environment_sha256": RUNTIME_SHA,
            "real_set": real_set,
            "sampling_report": dense_sampling,
            "fid": 80.0,
        },
    }
    result_path = tmp_path / "reports" / "quality_bridge_result.json"
    result = {
        "schema_version": QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
        "role": QUALITY_BRIDGE_RESULT_ROLE,
        "status": "completed",
        "stage": QUALITY_BRIDGE_RECIPE_STAGE,
        "git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "source_reports": {
            "cofitok_generation": cofitok_metrics,
            "dense_generation": dense_metrics,
        },
        "terminal": {"methods": methods},
        "quality_screen": {"status": "pass", "decision": "advance"},
        "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
    }
    result_identity = _write(result_path, result)
    return {
        "result_path": result_path,
        "result": result,
        "result_identity": result_identity,
        "output_root": tmp_path / "uncertainty",
        "cofitok_metrics_path": cofitok_metrics_path,
    }


def _build(case: dict[str, object]) -> dict[str, object]:
    identity = case["result_identity"]
    assert isinstance(identity, dict)
    return builder.build_manifest(
        quality_result_path=case["result_path"],
        expected_quality_result_sha256=str(identity["sha256"]),
        expected_revision=REVISION,
        expected_branch=BRANCH,
        output_root=case["output_root"],
    )


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


def test_builds_source_bound_manifest_accepted_by_auditor(tmp_path: Path) -> None:
    case = _case(tmp_path)
    manifest = _build(case)
    manifest_path = tmp_path / "uncertainty" / "manifests" / "terminal.json"
    _write(manifest_path, manifest)

    arguments = manifest["arguments"]
    assert isinstance(arguments, dict)
    assert manifest["stream_id"] == (
        "full_data_quality_bridge_terminal_100k_00000000_00010000"
    )
    assert arguments["min_samples"] == 10_000
    assert arguments["block_size"] == 500
    assert manifest["expected"]["fid_point_estimates"] == {
        "cofitok": 72.0,
        "dense_identity": 80.0,
    }
    hydrated = audit.hydrate_execution_arguments(_audit_args(manifest_path))
    binding = audit.validate_execution_manifest(hydrated)
    assert binding is not None
    assert binding["stream_id"] == manifest["stream_id"]


def test_rejects_quality_result_sha_mismatch(tmp_path: Path) -> None:
    case = _case(tmp_path)
    with pytest.raises(ValueError, match="result SHA256 differs"):
        builder.build_manifest(
            quality_result_path=case["result_path"],
            expected_quality_result_sha256="0" * 64,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            output_root=case["output_root"],
        )


def test_rejects_quality_result_git_mismatch(tmp_path: Path) -> None:
    case = _case(tmp_path)
    identity = case["result_identity"]
    assert isinstance(identity, dict)
    with pytest.raises(ValueError, match="result contract differs"):
        builder.build_manifest(
            quality_result_path=case["result_path"],
            expected_quality_result_sha256=str(identity["sha256"]),
            expected_revision="0" * 40,
            expected_branch=BRANCH,
            output_root=case["output_root"],
        )


def test_rejects_unmatched_terminal_protocol(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = copy.deepcopy(case["result"])
    result["terminal"]["methods"]["dense_identity"]["sampling"][
        "guidance_scale"
    ] = 2.0
    identity = _write(case["result_path"], result)
    case["result_identity"] = identity
    with pytest.raises(ValueError, match="sampling protocols are not matched"):
        _build(case)


def test_rejects_mutated_bound_metrics_source(tmp_path: Path) -> None:
    case = _case(tmp_path)
    path = case["cofitok_metrics_path"]
    assert isinstance(path, Path)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source identity differs"):
        _build(case)


def test_rejects_real_set_mismatch(tmp_path: Path) -> None:
    case = _case(tmp_path)
    result = copy.deepcopy(case["result"])
    result["terminal"]["methods"]["dense_identity"]["real_set"] = dict(
        result["terminal"]["methods"]["dense_identity"]["real_set"]
    )
    result["terminal"]["methods"]["dense_identity"]["real_set"][
        "sha256"
    ] = "2" * 64
    identity = _write(case["result_path"], result)
    case["result_identity"] = identity
    with pytest.raises(ValueError, match="evaluator/real-set parity differs"):
        _build(case)


def test_prepare_manifest_resume_is_byte_strict(tmp_path: Path) -> None:
    case = _case(tmp_path)
    manifest = _build(case)
    path = tmp_path / "uncertainty" / "manifests" / "terminal.json"
    first = builder.prepare_manifest(path, manifest, resume=False, overwrite=False)
    resumed = builder.prepare_manifest(path, manifest, resume=True, overwrite=False)
    assert resumed == first
    changed = copy.deepcopy(manifest)
    changed["arguments"]["seed"] = 9
    with pytest.raises(ValueError, match="does not match"):
        builder.prepare_manifest(path, changed, resume=True, overwrite=False)
