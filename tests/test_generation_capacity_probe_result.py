from __future__ import annotations

import copy

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation.capacity_probe_execution import (
    LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY,
)
from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_ARM_NAMES,
    CAPACITY_PROBE_RESULT_BOUNDARY,
    CAPACITY_PROBE_RESULT_SOURCE_NAMES,
    build_capacity_probe_result,
)
from test_generation_capacity_probe_execution import (
    BRANCH,
    OUTPUT_ROOT,
    REVISION,
    _identity,
    _launch_receipt,
    _preparation,
)


def _git() -> dict[str, object]:
    return {
        "revision": REVISION,
        "branch": BRANCH,
        "tracked_dirty": False,
    }


def _training(method: str) -> dict[str, object]:
    run = f"{OUTPUT_ROOT}/base256_{method}"
    checkpoint = f"{run}/checkpoint_step_00010000.pt"
    sha = ("7" if method == "cofitok" else "8") * 64
    parameters = 250_153_763 if method == "cofitok" else 250_135_043
    return {
        "schema_version": 1,
        "status": "pass",
        "role": "generation_capacity_probe_partial_training_validation",
        "run_dir": run,
        "git": _git(),
        "configured_steps": 100_000,
        "completed_steps": 10_000,
        "training_complete": False,
        "intentional_partial_stop": True,
        "effective_batch_size": 64,
        "images_seen": 640_000,
        "parameter_count": parameters,
        "runtime_environment_sha256": "9" * 64,
        "dataset_identity_sha256": "a" * 64,
        "checkpoint": {
            "path": checkpoint,
            "bytes": 4_000_000_000,
            "sha256": sha,
            "integrity_manifest": _identity(
                f"{checkpoint}.integrity.json",
                "f" if method == "cofitok" else "0",
            ),
        },
        "authorization_boundary": {
            "capacity_probe_training_complete": True,
            "full_training_complete": False,
            "full_300k_launch_allowed": False,
            "formal_generation_claim_allowed": False,
            "release_allowed": False,
        },
    }


def _base128_reference(method: str) -> dict[str, object]:
    name = "cofitok" if method == "cofitok" else "dense_identity"
    checkpoint = (
        f"/root/autodl-tmp/CoFiTok/checkpoints/generation/quality_bridge/"
        f"capacity_probe_references/base128_step_00010000/{name}/"
        "checkpoint_step_00010000.pt"
    )
    character = "4" if method == "cofitok" else "5"
    return {
        "checkpoint": {
            "path": checkpoint,
            "bytes": 1_000_000_000,
            "sha256": character * 64,
        },
        "integrity": {
            "path": f"{checkpoint}.integrity.json",
            "bytes": 1_000,
            "sha256": ("a" if method == "cofitok" else "b") * 64,
        },
    }


def _expected_checkpoint(
    arm: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if arm.startswith("base128"):
        method = "cofitok" if arm.endswith("cofitok") else "dense_identity"
        reference = _base128_reference(method)
        return reference["checkpoint"], reference["integrity"]
    method = "cofitok" if arm.endswith("cofitok") else "dense_identity"
    checkpoint = _training(method)["checkpoint"]
    return (
        {key: checkpoint[key] for key in ("path", "bytes", "sha256")},
        checkpoint["integrity_manifest"],
    )


def _sampling(budget: int) -> dict[str, object]:
    return {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_samples": 2_048,
        "start_index": 0,
        "batch_size": 32,
        "sample_steps": 50,
        "num_train_timesteps": 1_000,
        "actual_timesteps": select_sampling_timesteps(1_000, 50),
        "prefix_budgets": [budget],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "image_shape": [3, 256, 256],
        "class_schedule": "balanced_modulo",
        "random_stream": {
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }


def _arm_reports(arm: str, *, fid: float) -> tuple[dict, dict, dict, dict]:
    budget = 8 if arm.endswith("cofitok") else 1
    checkpoint, integrity = _expected_checkpoint(arm)
    integrity_path = integrity["path"]
    sample_sha = str(CAPACITY_PROBE_ARM_NAMES.index(arm) + 1) * 64
    real_set = {
        "root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val",
        "sha256": "e" * 64,
        "image_count": 50_000,
    }
    report_identity = _identity(f"/{arm}/sampling_report.json", "b")
    manifest_identity = _identity(f"/{arm}/sampling_manifest.json", "c")
    progress_identity = _identity(f"/{arm}/sampling_progress.json", "d")
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "git": _git(),
        "runtime_environment_sha256": "f" * 64,
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_integrity_manifest": integrity_path,
        "checkpoint_step": 10_000,
        "weights": "ema",
        "requested_weights": "ema",
        "request": {
            "batch_size": 32,
            "prefix_budget": budget,
            "precision": "bf16",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
        },
        "result": {"output_finite": True, "mean_forward_seconds": 1.0},
    }
    generation = {
        "schema_version": 3,
        "role": "generation_directory_metrics_report",
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "git": _git(),
        "runtime_environment_sha256": "0" * 64,
        "counts": {"generated_image_count": 2_048},
        "parameters": {"precision_recall_enabled": False},
        "real_set": real_set,
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 4.0,
        },
        "sample_provenance": {
            "checkpoint": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_integrity_manifest": integrity_path,
            "checkpoint_step": 10_000,
            "weights": "ema",
            "git": _git(),
            "runtime_environment_sha256": "f" * 64,
            "selected_prefix_budget": budget,
            "sample_set_sha256": sample_sha,
            "report_identity": report_identity,
            "manifest_identity": manifest_identity,
            "sampling_progress": {
                "identity": progress_identity,
                "status": "completed",
                "completed_samples": 2_048,
                "cumulative_elapsed_seconds": 100.0,
            },
            "sampling": _sampling(budget),
        },
    }
    random_orders = 4 if budget == 8 else 0
    checkpoint_eval = {
        "schema_version": 2,
        "role": "generation_checkpoint_evaluation_report",
        "status": "completed",
        "git": _git(),
        "request": {
            "num_images": 256,
            "timestep": 500,
            "random_orders": random_orders,
            "weights": "ema",
            "precision": "bf16",
        },
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_integrity_manifest": integrity_path,
        "checkpoint_step": 10_000,
        "weights": "ema",
        "metrics": {
            "evaluated_images": 256,
            "orders": {
                "ordered": {
                    "endpoint_clean_mse": 0.1,
                    "prefix_path_mse_auc": 0.2,
                }
            },
            "order_count": 6 if budget == 8 else 1,
            "ordered_rank_by_path_auc": 1,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
        },
    }
    physical = {
        "checkpoint": checkpoint,
        "checkpoint_integrity_manifest": integrity,
        "checkpoint_step": 10_000,
        "sampling_report": report_identity,
        "sampling_manifest": manifest_identity,
        "sampling_progress": progress_identity,
        "sample_set_sha256": sample_sha,
        "sample_count": 2_048,
        "real_set": real_set,
    }
    return preflight, generation, checkpoint_eval, physical


def _source_identities() -> dict[str, dict[str, object]]:
    return {
        name: _identity(f"/evidence/{name}.json", hex(index + 1)[2:][-1])
        for index, name in enumerate(sorted(CAPACITY_PROBE_RESULT_SOURCE_NAMES))
    }


def _kwargs(*, fids: dict[str, float]) -> dict:
    preflights = {}
    generations = {}
    evaluations = {}
    physical = {}
    for arm in CAPACITY_PROBE_ARM_NAMES:
        preflight, generation, evaluation, physical_row = _arm_reports(
            arm,
            fid=fids[arm],
        )
        preflights[arm] = preflight
        generations[arm] = generation
        evaluations[arm] = evaluation
        physical[arm] = physical_row
    preparation = _preparation()
    preparation["checkpoint_references"] = {
        "cofitok": _base128_reference("cofitok"),
        "dense_identity": _base128_reference("dense_identity"),
    }
    launch = _launch_receipt()
    sources = _source_identities()
    sources["preparation"] = launch["source_reports"]["preparation"]
    return {
        "preparation": preparation,
        "launch_receipt": launch,
        "partial_training_validations": {
            "cofitok": _training("cofitok"),
            "dense_identity": _training("dense_identity"),
        },
        "sampling_preflights": preflights,
        "generation_reports": generations,
        "checkpoint_evaluations": evaluations,
        "physical_evidence": physical,
        "source_identities": sources,
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
    }


def test_capacity_probe_result_supports_capacity_only_on_shared_improvement() -> None:
    report = build_capacity_probe_result(
        **_kwargs(
            fids={
                "base128_cofitok": 180.0,
                "base128_dense_identity": 175.0,
                "base256_cofitok": 160.0,
                "base256_dense_identity": 165.0,
            }
        )
    )
    assert report["decision"]["shared_strict_fid_improvement"] is True
    assert report["decision"]["capacity_supported"] is True
    assert report["estimands"]["capacity_by_factorization_fid_interaction"] == -10.0
    assert report["authorization_boundary"] == CAPACITY_PROBE_RESULT_BOUNDARY
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert report["claim_policy"]["formal_generation_claim_allowed"] is False


def test_capacity_probe_result_holds_without_shared_improvement() -> None:
    report = build_capacity_probe_result(
        **_kwargs(
            fids={
                "base128_cofitok": 180.0,
                "base128_dense_identity": 175.0,
                "base256_cofitok": 160.0,
                "base256_dense_identity": 176.0,
            }
        )
    )
    assert report["decision"]["capacity_supported"] is False
    assert report["decision"]["recommendation"]["id"] == (
        "hold_capacity_scaling_and_revisit_training_objective"
    )


def test_capacity_probe_result_rejects_sampling_or_physical_drift() -> None:
    kwargs = _kwargs(
        fids={arm: 170.0 for arm in CAPACITY_PROBE_ARM_NAMES}
    )
    kwargs["generation_reports"]["base256_cofitok"]["sample_provenance"][
        "sampling"
    ]["seed"] = 1
    with pytest.raises(ValueError, match="sampling protocol differs"):
        build_capacity_probe_result(**kwargs)

    kwargs = _kwargs(
        fids={arm: 170.0 for arm in CAPACITY_PROBE_ARM_NAMES}
    )
    kwargs["physical_evidence"]["base128_cofitok"]["sample_set_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="physical evidence differs"):
        build_capacity_probe_result(**kwargs)


def test_capacity_probe_result_rejects_authorization_escalation() -> None:
    kwargs = _kwargs(fids={arm: 170.0 for arm in CAPACITY_PROBE_ARM_NAMES})
    kwargs["launch_receipt"]["authorization_boundary"] = copy.deepcopy(
        LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
    )
    kwargs["launch_receipt"]["authorization_boundary"][
        "full_300k_launch_allowed"
    ] = True
    with pytest.raises(ValueError, match="contract differs"):
        build_capacity_probe_result(**kwargs)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.build_generation_capacity_probe_result",
        "scripts.verify_generation_capacity_probe_result",
    ],
)
def test_capacity_probe_result_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)
