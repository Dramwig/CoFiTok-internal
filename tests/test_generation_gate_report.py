from __future__ import annotations

import copy

import pytest

from cofitok.environment import runtime_environment_sha256
from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation_gate import GENERATION_GATE_SCHEMA_VERSION
from scripts.build_generation_gate_report import build_report


def _training(parameters: int, token_count: int) -> dict:
    checkpoint_sha256 = ("a" if token_count > 1 else "b") * 64
    return {
        "training_complete": True,
        "completed_steps": 50_000,
        "target_steps": 50_000,
        "parameter_count": parameters,
        "elapsed_seconds": 1_000.0,
        "peak_vram_bytes": 24 * 1024**3,
        "final_metrics": {"samples_seen": 3_200_000},
        "git": {
            "dirty": False,
            "revision": "a" * 40,
            "branch": "scale/generative-system",
        },
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00050000.pt",
            "checkpoint_bytes": 1_000_000,
            "checkpoint_sha256": checkpoint_sha256,
            "integrity_manifest": "checkpoint_step_00050000.pt.integrity.json",
            "step": 50_000,
        },
        "config": {
            "data": {"dataset": "imagenet_256_10pct", "batch_size": 16},
            "diffusion": {"schedule": "cosine", "num_train_timesteps": 1000},
            "runtime": {"steps": 50_000, "device": "cuda"},
            "optimization": {"batch": 64, "gradient_accumulation_steps": 4},
            "model": {
                "token_count": token_count,
                "token_channels": 64 if token_count > 1 else 3,
                "image_channels": 3,
                "image_size": 256,
                "base_channels": 128,
                "predictor_type": "scalable_unet",
                "predictor_use_feedback": token_count > 1,
                "num_classes": 1000,
                "class_dropout_prob": 0.1,
                "synthesis_mode": "restricted" if token_count > 1 else "dense_identity",
            },
            "loss": {
                "epsilon_weight": 1.0,
                "denoise_path_component_weight": 0.1 if token_count > 1 else 0.0,
            },
        },
    }


def _generation(fid: float, token_count: int, sha: str) -> dict:
    runtime_environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "GPU"},
    }
    real_set_sha = "9" * 64
    return {
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "git": {
            "revision": "a" * 40,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "runtime_environment": copy.deepcopy(runtime_environment),
        "runtime_environment_sha256": runtime_environment_sha256(
            runtime_environment
        ),
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "paths": {
            "real_dir": "/datasets/imagenet_256/val",
            "generated_dir": f"/samples/prefix_{token_count}",
        },
        "counts": {"real_image_count": 50_000, "generated_image_count": 10_000},
        "real_set": {
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": real_set_sha,
            "root": "/datasets/imagenet_256/val",
            "image_count": 50_000,
        },
        "parameters": {
            "batch_size": 64,
            "seed": 2027,
            "real_cache_name": f"imagenet256_val__cofitok_{real_set_sha[:16]}",
        },
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 18.0,
            "inception_score_std": 0.2,
            "precision": 0.6,
            "recall": 0.4,
        },
        "sample_provenance": {
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": runtime_environment_sha256(
                runtime_environment
            ),
            "git": {
                "revision": "a" * 40,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
            "checkpoint": "/checkpoint.pt",
            "checkpoint_sha256": sha,
            "checkpoint_integrity_manifest": "/checkpoints/checkpoint.pt.integrity.json",
            "checkpoint_step": 50_000,
            "weights": "ema",
            "sample_set_sha256": ("c" if token_count > 1 else "d") * 64,
            "selected_prefix_budget": token_count,
            "sampling_progress": {
                "report": "/samples/sampling_progress.json",
                "status": "completed",
                "invocation": 1,
                "completed_samples": 10_000,
                "cumulative_elapsed_seconds": 100.0,
            },
            "sampling": {
                "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_samples": 10_000,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": 100,
                "num_train_timesteps": 1000,
                "actual_timesteps": select_sampling_timesteps(1000, 100),
                "image_shape": [3, 256, 256],
                "class_schedule": "balanced_modulo",
                "prefix_budgets": [token_count],
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "seed": 0,
                "precision": "bf16",
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
            },
        },
    }


def _full_generation(fid: float, token_count: int, sha: str) -> dict:
    report = _generation(fid, token_count, sha)
    report["counts"]["generated_image_count"] = 50_000
    report["sample_provenance"]["sampling_progress"]["completed_samples"] = 50_000
    sampling = report["sample_provenance"]["sampling"]
    sampling["num_samples"] = 50_000
    sampling["sample_steps"] = 250
    sampling["actual_timesteps"] = select_sampling_timesteps(1000, 250)
    return report


def _checkpoint(endpoint: float, sha: str, rank: int = 1) -> dict:
    token_count = 1 if sha == "b" * 64 else 8
    return {
        "git": {
            "revision": "a" * 40,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "checkpoint_sha256": sha,
        "checkpoint_integrity_manifest": "/checkpoints/checkpoint.pt.integrity.json",
        "checkpoint_step": 50_000,
        "config": {"model": {"token_count": token_count}},
        "metrics": {
            "evaluated_images": 1024,
            "orders": {"ordered": {"endpoint_clean_mse": endpoint}},
            "ordered_rank_by_path_auc": rank,
            "order_count": 18,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
            "component_energy_ratio_per_sample_mean": (
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.30, 0.64]
                if token_count > 1
                else [1.0]
            ),
        }
    }


def _rollout_stability_qualification(*, passed: bool = True) -> dict:
    failed = [] if passed else ["predicted_x0_high_frequency"]
    return {
        "schema_version": 2,
        "status": "pass" if passed else "fail",
        "protocol": {
            "weights": "ema",
            "checkpoint_step": 50_000,
            "checkpoint_evaluated_images": 1024,
            "rollout": {
                "num_images": 64,
                "batch_size": 8,
                "sample_steps": 100,
                "guidance_scale": 1.5,
                "teacher_guidance_scale": 1.0,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "clip_x0": True,
                "precision": "bf16",
                "seed": 2029,
            },
        },
        "identity": {
            "evaluation_git_revision": "a" * 40,
            "evaluation_git_branch": "scale/generative-system",
            "cofitok_checkpoint_sha256": "a" * 64,
            "dense_checkpoint_sha256": "b" * 64,
        },
        "gates": {
            "predicted_x0_high_frequency": {"passed": passed},
            "reconstruction_regression": {"passed": True},
        },
        "pair_contract": {"valid": True},
        "failed_gates_fixture": failed,
    }


def test_generation_gate_holds_on_collapsed_coarse_tokens() -> None:
    checkpoint = _checkpoint(0.1, "a" * 64)
    checkpoint["metrics"]["component_energy_ratio_per_sample_mean"] = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.004,
        0.335,
        0.66,
    ]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=checkpoint,
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(row for row in report["gates"] if row["name"] == "coarse_token_utilization")
    assert gate["passed"] is False
    assert gate["evidence"]["coarse_token_energy_ratio"] == pytest.approx(0.005)
    assert report["status"] == "fail"


def test_generation_gate_uses_rgbtail3_stride_partition() -> None:
    checkpoint = _checkpoint(0.1, "a" * 64)
    checkpoint["config"]["model"]["token_spatial_strides"] = [
        16,
        16,
        8,
        8,
        4,
        1,
        1,
        1,
    ]
    checkpoint["metrics"]["component_energy_ratio_per_sample_mean"] = [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.30,
        0.31,
        0.34,
    ]

    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=checkpoint,
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        require_stride_partition=True,
    )

    gate = next(
        row for row in report["gates"] if row["name"] == "coarse_token_utilization"
    )
    assert gate["passed"] is True
    assert gate["evidence"]["partition_schema"] == "token_spatial_stride_suffix_v1"
    assert gate["evidence"]["coarse_token_count"] == 5
    assert gate["evidence"]["full_resolution_tail_token_count"] == 3
    assert gate["evidence"]["coarse_token_energy_ratio"] == pytest.approx(0.05)


def test_stability_generation_gate_rejects_missing_stride_partition() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.102, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        require_stride_partition=True,
    )

    gate = next(
        row for row in report["gates"] if row["name"] == "coarse_token_utilization"
    )
    assert gate["passed"] is False
    assert (
        gate["evidence"]["partition_schema"]
        == "invalid_missing_token_spatial_stride_suffix"
    )


def test_generation_gate_passes_matched_quality_and_mechanism() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.102, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "pass"
    assert report["schema_version"] == GENERATION_GATE_SCHEMA_VERSION
    assert report["decision"] == "promote_to_full_imagenet256"
    assert report["summary"]["coarse_token_energy_ratio"] == pytest.approx(0.06)
    assert all(gate["passed"] for gate in report["gates"])


def test_generation_gate_binds_passing_ema_rollout_stability() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.102, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        rollout_stability_qualification=_rollout_stability_qualification(),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        row
        for row in report["gates"]
        if row["name"] == "rollout_stability_diagnostic"
    )
    assert report["status"] == "pass"
    assert gate["passed"] is True
    assert gate["evidence"]["weights"] == "ema"
    assert gate["evidence"]["failed_gates"] == []


def test_generation_gate_holds_on_failed_ema_rollout_stability() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.102, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        rollout_stability_qualification=_rollout_stability_qualification(
            passed=False
        ),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        row
        for row in report["gates"]
        if row["name"] == "rollout_stability_diagnostic"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False
    assert gate["evidence"]["failed_gates"] == [
        "predicted_x0_high_frequency"
    ]


def test_generation_gate_holds_on_fid_regression() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(25.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert not next(gate for gate in report["gates"] if gate["name"] == "fid_within_tolerance")[
        "passed"
    ]


def test_generation_gate_holds_on_unpaired_sampling_streams() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    dense["sample_provenance"]["sampling"]["random_stream"]["batch_size_invariant"] = False
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_rejects_matched_but_weakened_formal_sampling() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    cofitok["sample_provenance"]["sampling"]["clip_x0"] = False
    dense["sample_provenance"]["sampling"]["clip_x0"] = False

    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "formal_sampling_protocol"
    )
    assert gate["passed"] is False
    assert report["status"] == "fail"


def test_generation_gate_rejects_mismatched_sampling_environment() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    dense_environment = dense["sample_provenance"]["runtime_environment"]
    dense_environment["device"]["name"] = "another GPU"
    dense["sample_provenance"][
        "runtime_environment_sha256"
    ] = runtime_environment_sha256(dense_environment)
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_sampling_runtime_environment"
    )
    assert gate["passed"] is False


def test_generation_gate_rejects_mismatched_evaluator_environment() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    dense["runtime_environment"]["device"]["name"] = "another GPU"
    dense["runtime_environment_sha256"] = runtime_environment_sha256(
        dense["runtime_environment"]
    )
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_evaluator_runtime_environment"
    )
    assert gate["passed"] is False


def test_generation_gate_rejects_mismatched_real_set_content() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    dense["real_set"]["sha256"] = "8" * 64
    dense["parameters"]["real_cache_name"] = (
        "imagenet256_val__cofitok_" + "8" * 16
    )
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "matched_real_set_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_requires_completed_sampling_progress() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["sample_provenance"]["sampling_progress"]["status"] = "running"
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_rejects_dirty_sampling_code() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["sample_provenance"]["git"]["tracked_dirty"] = True
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_sampling_code_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_rejects_dirty_evaluator_code() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["git"]["tracked_dirty"] = True
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_evaluator_code_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_rejects_dirty_checkpoint_evaluator_code() -> None:
    cofitok_checkpoint = _checkpoint(0.1, "a" * 64)
    cofitok_checkpoint["git"]["tracked_dirty"] = True
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=cofitok_checkpoint,
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_checkpoint_evaluator_code_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_accepts_explicit_training_and_evaluation_identities() -> None:
    training_revision = "1" * 40
    evaluation_revision = "2" * 40
    training_branch = "scale/generation-stability-50k-preflight"
    evaluation_branch = "scale/generation-stability-50k-posteval"
    cofitok_training = _training(100_500, 8)
    dense_training = _training(100_000, 1)
    cofitok_training["git"].update(
        revision=training_revision,
        branch=training_branch,
    )
    dense_training["git"].update(
        revision=training_revision,
        branch=training_branch,
    )
    cofitok_generation = _generation(20.5, 8, "a" * 64)
    dense_generation = _generation(20.0, 1, "b" * 64)
    for generation in (cofitok_generation, dense_generation):
        generation["git"].update(
            revision=evaluation_revision,
            branch=evaluation_branch,
        )
        generation["sample_provenance"]["git"].update(
            revision=evaluation_revision,
            branch=evaluation_branch,
        )
    cofitok_checkpoint = _checkpoint(0.1, "a" * 64)
    dense_checkpoint = _checkpoint(0.1, "b" * 64)
    for checkpoint in (cofitok_checkpoint, dense_checkpoint):
        checkpoint["git"].update(
            revision=evaluation_revision,
            branch=evaluation_branch,
        )

    report = build_report(
        cofitok_training=cofitok_training,
        dense_training=dense_training,
        cofitok_generation=cofitok_generation,
        dense_generation=dense_generation,
        cofitok_checkpoint=cofitok_checkpoint,
        dense_checkpoint=dense_checkpoint,
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        expected_training_revision=training_revision,
        expected_training_branch=training_branch,
        expected_evaluation_revision=evaluation_revision,
        expected_evaluation_branch=evaluation_branch,
    )

    assert report["status"] == "pass"
    assert report["provenance_contract"] == {
        "training_revision": training_revision,
        "training_branch": training_branch,
        "evaluation_revision": evaluation_revision,
        "evaluation_branch": evaluation_branch,
    }


def test_generation_gate_rejects_wrong_explicit_evaluation_revision() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        expected_evaluation_revision="2" * 40,
    )

    for name in (
        "matched_evaluator_code_provenance",
        "matched_sampling_code_provenance",
        "matched_checkpoint_evaluator_code_provenance",
    ):
        gate = next(row for row in report["gates"] if row["name"] == name)
        assert gate["passed"] is False


def test_generation_gate_requires_positive_sampling_elapsed_time() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["sample_provenance"]["sampling_progress"][
        "cumulative_elapsed_seconds"
    ] = 0.0
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_on_mismatched_training_revision() -> None:
    dense_training = _training(100_000, 1)
    dense_training["git"]["revision"] = "b" * 40
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=dense_training,
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_training_revision")
    assert gate["passed"] is False


def test_generation_gate_holds_on_shared_backbone_drift() -> None:
    dense_training = _training(100_000, 1)
    dense_training["config"]["model"]["class_dropout_prob"] = 0.2
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=dense_training,
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "matched_training_protocol"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False


def test_generation_gate_requires_exact_sample_count_and_shared_real_set() -> None:
    dense = _generation(20.0, 1, "b" * 64)
    dense["counts"]["generated_image_count"] = 10_001
    dense["paths"]["real_dir"] = "/datasets/other/val"
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_generation_protocol")
    assert gate["passed"] is False


def test_generation_gate_requires_balanced_class_schedule() -> None:
    dense = _generation(20.0, 1, "b" * 64)
    dense["sample_provenance"]["sampling"]["class_schedule"] = None
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_sampling_shape_is_unproven() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    del dense["sample_provenance"]["sampling"]["image_shape"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_sample_set_digest_is_unproven() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    del dense["sample_provenance"]["sample_set_sha256"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_mechanism_eval_uses_another_checkpoint() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "c" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "checkpoint_evaluation_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_holds_when_mechanism_eval_uses_another_integrity_manifest() -> None:
    checkpoint = _checkpoint(0.1, "a" * 64)
    checkpoint["checkpoint_integrity_manifest"] = "/checkpoints/other.pt.integrity.json"
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=checkpoint,
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "checkpoint_evaluation_provenance"
    )
    assert gate["passed"] is False


def test_full_generation_gate_uses_ready_decision_and_absolute_fid() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_full_generation(19.0, 8, "a" * 64),
        dense_generation=_full_generation(18.5, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=50_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "large_scale_generation_ready"
    assert report["stage"] == "full"


def test_full_generation_gate_holds_above_absolute_fid_limit() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_full_generation(20.5, 8, "a" * 64),
        dense_generation=_full_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=50_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "absolute_fid_quality")
    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert gate["passed"] is False


def test_full_generation_gate_requires_training_checkpoint_integrity() -> None:
    cofitok_training = _training(100_500, 8)
    del cofitok_training["latest_checkpoint"]["checkpoint_sha256"]
    report = build_report(
        cofitok_training=cofitok_training,
        dense_training=_training(100_000, 1),
        cofitok_generation=_full_generation(19.0, 8, "a" * 64),
        dense_generation=_full_generation(18.5, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=50_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "full_training_checkpoint_integrity"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False


def test_generation_gate_holds_when_quality_metrics_are_incomplete() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    del cofitok["metrics"]["recall"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "generation_metrics_complete")
    assert gate["passed"] is False


def test_generation_gate_requires_complete_training_cost_accounting() -> None:
    dense_training = _training(100_000, 1)
    dense_training["final_metrics"]["samples_seen"] -= 1
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=dense_training,
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "training_cost_accounting")
    assert report["status"] == "fail"
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("frechet_inception_distance", -1.0),
        ("inception_score_mean", 0.0),
        ("inception_score_std", -0.1),
        ("precision", 1.1),
        ("recall", -0.1),
        ("recall", "not-a-number"),
    ],
)
def test_generation_gate_rejects_finite_but_invalid_metric_ranges(metric, value) -> None:
    cofitok = _generation(19.0, 8, "a" * 64)
    cofitok["metrics"][metric] = value
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(19.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "distribution_metric_ranges"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("precision", "recall"),
    [(0.29, 0.40), (0.60, 0.29), (0.54, 0.40), (0.60, 0.34)],
)
def test_full_generation_gate_enforces_precision_recall_floor_and_retention(
    precision, recall
) -> None:
    cofitok = _full_generation(19.0, 8, "a" * 64)
    dense = _full_generation(19.0, 1, "b" * 64)
    cofitok["metrics"]["precision"] = precision
    cofitok["metrics"]["recall"] = recall
    dense["metrics"]["precision"] = 0.60
    dense["metrics"]["recall"] = 0.40
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=50_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "full_precision_recall_quality"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False
