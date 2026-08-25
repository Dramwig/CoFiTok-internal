from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from cofitok.configs import LossConfig, load_config
from cofitok.diffusion import DiffusionSchedule
from cofitok.generation.min_snr_pilot import (
    DATASET_IDENTITY_SHA256,
    EFFECTIVE_BATCH_SIZE,
    EXECUTION_BOUNDARY,
    GAMMA,
    IMAGES_PER_METHOD,
    LEGACY_BRANCH,
    LEGACY_REVISION,
    LEGACY_RUNTIME_ENVIRONMENT_SHA256,
    METHODS,
    PILOT_BRANCH,
    PILOT_STEP,
    PREPARATION_BOUNDARY,
    RESULT_BOUNDARY,
    SCHEDULER_HORIZON,
    TRAINING_SEMANTIC_FILES,
    build_execution_gate,
    build_preparation,
    build_result,
    validate_execution_gate,
    validate_preparation,
    validate_result,
    validate_sampling_preflight,
)
from cofitok.models import CoFiTokOutput
from cofitok.training import compute_losses


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG_PATHS = {
    "cofitok": ROOT
    / "configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
    "dense_identity": ROOT
    / "configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json",
}
PILOT_CONFIG_PATHS = {
    "cofitok": ROOT
    / "configs/generation/imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k_horizon_50k_pilot.json",
    "dense_identity": ROOT
    / "configs/generation/imagenet256_min_snr_gamma5_quality_repair_rollout_x0_u2_ema_teacher_dense_100k_horizon_50k_pilot.json",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(name: str) -> dict[str, Any]:
    digit = hex((sum(name.encode()) % 15) + 1)[2:]
    return {"path": f"/evidence/{name}.json", "bytes": 10, "sha256": digit * 64}


def _post_decision() -> dict[str, Any]:
    return {
        "schema": "cofitok_epsilon_stability_post_diagnostic_decision_v1",
        "status": "completed",
        "terminal_status": "hold",
        "decision": "prepare_fresh_matched_min_snr_training_pilot",
        "generation_advantage_proven": False,
        "recommended_next_stage": {
            "id": "prepare_fresh_matched_min_snr_training_pilot",
            "execution_ready": False,
            "controlled_change": {
                "field": "loss.min_snr_gamma",
                "legacy_value": 0.0,
                "prediction_target": "epsilon",
            },
            "preparation_requirements": {
                "fresh_initialization_required": True,
                "changed_config_resume_allowed": False,
                "exact_data_initialization_and_random_stream_binding_required": True,
                "physical_checkpoint_integrity_required": True,
                "matched_ddim100_quality_and_class_fidelity_evaluation_required": True,
            },
        },
        "authorization_boundary": {
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "inference_export_allowed": False,
            "release_allowed": False,
            "process_signals_allowed": False,
            "gpu_execution_allowed": False,
        },
    }


def _pair_monitor() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "pass",
        "stage": "complete",
        "issues": [],
        "git": {
            "revision": LEGACY_REVISION,
            "branch": LEGACY_BRANCH,
            "tracked_dirty": False,
        },
        "runs": {method: {} for method in METHODS},
    }


def _legacy_report(path: Path) -> dict[str, Any]:
    config = load_config(path)
    resolved = replace(
        config,
        data=replace(config.data, batch_size=EFFECTIVE_BATCH_SIZE),
        optimization=replace(config.optimization, gradient_accumulation_steps=1),
    )
    return {
        "training_complete": True,
        "completed_steps": SCHEDULER_HORIZON,
        "git": {"revision": LEGACY_REVISION, "branch": LEGACY_BRANCH, "dirty": False},
        "config": asdict(resolved),
        "dataset_provenance": {"identity_sha256": DATASET_IDENTITY_SHA256},
        "runtime_environment_sha256": LEGACY_RUNTIME_ENVIRONMENT_SHA256,
    }


def _legacy_audit(method: str) -> dict[str, Any]:
    sha = ("a" if method == "cofitok" else "b") * 64
    return {
        "status": "pass",
        "checkpoint": {
            "step": PILOT_STEP,
            "physical_sha256_verified": True,
            "payload": {
                "path": f"/legacy/{method}/checkpoint_step_00050000.pt",
                "bytes": 100,
                "sha256": sha,
            },
            "integrity": {
                "step": PILOT_STEP,
                "checkpoint_sha256": sha,
                "git_revision": LEGACY_REVISION,
                "git_branch": LEGACY_BRANCH,
                "git_dirty": False,
                "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
                "runtime_environment_sha256": LEGACY_RUNTIME_ENVIRONMENT_SHA256,
            },
        },
        "metrics": {
            "strictly_increasing": True,
            "samples_seen_binding_verified": True,
            "last_step": PILOT_STEP,
            "target_row": {"step": PILOT_STEP, "samples_seen": IMAGES_PER_METHOD},
        },
        "training_checkout": {
            "revision": LEGACY_REVISION,
            "branch": LEGACY_BRANCH,
            "tracked_dirty": False,
        },
    }


def _preparation() -> dict[str, Any]:
    legacy_configs = {method: _read(path) for method, path in LEGACY_CONFIG_PATHS.items()}
    pilot_configs = {method: _read(path) for method, path in PILOT_CONFIG_PATHS.items()}
    source_names = {
        "post_diagnostic_decision",
        "pair_monitor",
        "legacy_cofitok_config",
        "legacy_dense_config",
        "pilot_cofitok_config",
        "pilot_dense_config",
        "legacy_cofitok_training_report",
        "legacy_dense_training_report",
        "legacy_cofitok_checkpoint_audit_50k",
        "legacy_dense_checkpoint_audit_50k",
    }
    return build_preparation(
        post_diagnostic_decision=_post_decision(),
        pair_monitor=_pair_monitor(),
        legacy_configs=legacy_configs,
        pilot_configs=pilot_configs,
        legacy_training_reports={
            method: _legacy_report(LEGACY_CONFIG_PATHS[method]) for method in METHODS
        },
        legacy_checkpoint_audits={method: _legacy_audit(method) for method in METHODS},
        source_identities={name: _identity(name) for name in source_names},
        builder_git={
            "revision": "c" * 40,
            "tree": "d" * 40,
            "branch": PILOT_BRANCH,
            "tracked_dirty": False,
        },
        training_source_delta={
            "base_revision": LEGACY_REVISION,
            "base_is_ancestor": True,
            "changed_semantic_files": list(TRAINING_SEMANTIC_FILES),
        },
        output_root=(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "min_snr_gamma5_matched_50k_pilot_v1"
        ),
    )


def test_min_snr_gamma_zero_keeps_exact_legacy_epsilon_reduction() -> None:
    epsilon = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(epsilon)
    output = CoFiTokOutput(
        tokens=[epsilon],
        components=[epsilon],
        prefix_epsilons=[epsilon],
        epsilon=epsilon,
    )
    schedule = DiffusionSchedule(load_config(LEGACY_CONFIG_PATHS["dense_identity"]).diffusion, "cpu")
    losses = compute_losses(
        LossConfig(epsilon_weight=1.0, min_snr_gamma=0.0),
        output,
        schedule,
        noisy_images=torch.randn_like(epsilon),
        clean_images=torch.randn_like(epsilon),
        noise=noise,
        timesteps=torch.tensor([0, 100, 500, 999]),
    )
    assert torch.equal(losses.epsilon, F.mse_loss(epsilon, noise))
    assert torch.equal(losses.epsilon_unweighted, losses.epsilon)
    assert losses.min_snr_weight_mean.item() == 1.0


def test_standard_epsilon_min_snr_weights_downweight_high_snr() -> None:
    config = load_config(PILOT_CONFIG_PATHS["dense_identity"])
    schedule = DiffusionSchedule(config.diffusion, "cpu")
    timesteps = torch.tensor([0, 10, 500, 999])
    weights = schedule.min_snr_loss_weights(timesteps, GAMMA)
    assert torch.all(weights > 0.0)
    assert torch.all(weights <= 1.0)
    assert weights[0] < weights[1] < weights[2]
    assert weights[-1] == 1.0


def test_preparation_fixes_only_matched_gamma_and_remains_non_authorizing() -> None:
    report = _preparation()
    assert validate_preparation(report) == report
    assert report["controlled_change"]["pilot_value"] == GAMMA
    assert report["training_contract"]["pilot_stop_step"] == PILOT_STEP
    assert report["training_contract"]["scheduler_horizon_steps"] == SCHEDULER_HORIZON
    assert report["legacy_control_policy"]["fresh_gamma_zero_control_required"] is False
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY


def test_preparation_rejects_any_second_config_change() -> None:
    legacy = {method: _read(path) for method, path in LEGACY_CONFIG_PATHS.items()}
    pilot = {method: _read(path) for method, path in PILOT_CONFIG_PATHS.items()}
    pilot["dense_identity"]["optimization"]["learning_rate"] = 2e-4
    base = _preparation()
    with pytest.raises(ValueError, match="changes more than Min-SNR"):
        build_preparation(
            post_diagnostic_decision=_post_decision(),
            pair_monitor=_pair_monitor(),
            legacy_configs=legacy,
            pilot_configs=pilot,
            legacy_training_reports={
                method: _legacy_report(LEGACY_CONFIG_PATHS[method]) for method in METHODS
            },
            legacy_checkpoint_audits={method: _legacy_audit(method) for method in METHODS},
            source_identities=base["source_identities"],
            builder_git=base["builder_git"],
            training_source_delta=base["training_source_delta"],
            output_root=base["output_root"],
        )


def test_execution_gate_is_bounded_and_rejects_gpu_contention() -> None:
    preparation = _preparation()
    identity = _identity("preparation")
    kwargs = {
        "preparation": preparation,
        "preparation_identity": identity,
        "gate_builder_git": preparation["builder_git"],
        "runtime_environment_sha256": LEGACY_RUNTIME_ENVIRONMENT_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "gpu_inventory": [
            {
                "index": 0,
                "memory_used_mib": 0,
                "memory_total_mib": 97887,
                "utilization_percent": 0,
            }
        ],
        "gpu_compute_processes": [],
        "conflicting_processes": [],
        "output_root_absent": True,
        "execution_lock_free": True,
        "free_bytes": 300 * 1024**3,
        "authorization_record": {
            "scope": "matched_min_snr_50k_pilot_only",
            "approved_by": "user",
            "instruction": "Run needed bounded experiments directly.",
            "direct_execution_without_repeated_prompt": True,
            "full_300k_launch_allowed": False,
        },
    }
    gate = build_execution_gate(**kwargs)
    assert validate_execution_gate(
        gate, preparation=preparation, preparation_identity=identity
    ) == gate
    assert gate["authorization_boundary"] == EXECUTION_BOUNDARY
    assert gate["authorization_boundary"]["continuation_beyond_50000_allowed"] is False
    busy = deepcopy(kwargs)
    busy["gpu_compute_processes"] = [{"pid": 42}]
    with pytest.raises(ValueError, match="exclusivity"):
        build_execution_gate(**busy)


def _execution_gate(preparation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity("preparation")
    gate = build_execution_gate(
        preparation=preparation,
        preparation_identity=identity,
        gate_builder_git=preparation["builder_git"],
        runtime_environment_sha256=LEGACY_RUNTIME_ENVIRONMENT_SHA256,
        dataset_identity_sha256=DATASET_IDENTITY_SHA256,
        gpu_inventory=[
            {
                "index": 0,
                "memory_used_mib": 0,
                "memory_total_mib": 97887,
                "utilization_percent": 0,
            }
        ],
        gpu_compute_processes=[],
        conflicting_processes=[],
        output_root_absent=True,
        execution_lock_free=True,
        free_bytes=300 * 1024**3,
        authorization_record={
            "scope": "matched_min_snr_50k_pilot_only",
            "approved_by": "user",
            "instruction": "Run needed bounded experiments directly.",
            "direct_execution_without_repeated_prompt": True,
            "full_300k_launch_allowed": False,
        },
    )
    return gate, identity


def _pilot_training_audit(method: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "generation_matched_min_snr_pilot_training_physical_audit",
        "status": "pass",
        "method": method,
        "training": {
            "completed_steps": PILOT_STEP,
            "target_steps": SCHEDULER_HORIZON,
            "training_complete": False,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "samples_seen": IMAGES_PER_METHOD,
            "strictly_increasing_metrics": True,
            "samples_seen_binding_verified": True,
            "min_snr_metrics_verified": True,
        },
        "checkpoint": {
            "path": f"/pilot/{method}/checkpoint_step_00050000.pt",
            "bytes": 100,
            "sha256": ("e" if method == "cofitok" else "f") * 64,
            "physical_sha256_verified": True,
            "latest_exact_binding": True,
        },
    }


def _arm(*, method: str, pilot: bool) -> dict[str, Any]:
    prefix = 8 if method == "cofitok" else 1
    checkpoint_sha = ("1" if pilot else "2") * 64
    sample_sha = ("3" if pilot else "4") * 64
    fid = 90.0 if pilot else 100.0
    checkpoint_metrics: dict[str, Any] = {}
    model: dict[str, Any] = {"token_count": prefix}
    if method == "cofitok":
        model["token_spatial_strides"] = [16, 16, 8, 8, 4, 1, 1, 1]
        checkpoint_metrics = {
            "component_energy_ratio_per_sample_mean": [
                0.03,
                0.03,
                0.03,
                0.03,
                0.03,
                0.28,
                0.28,
                0.29,
            ],
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 20.0,
            "ordered_rank_by_path_auc": 1,
        }
    return {
        "sampling_preflight": {
            "schema_version": 1,
            "status": "passed",
            "git": {
                "revision": "c" * 40,
                "branch": PILOT_BRANCH,
                "tracked_dirty": False,
            },
            "runtime_environment_sha256": "6" * 64,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": PILOT_STEP,
            "weights": "ema",
            "requested_weights": "ema",
            "request": {
                "batch_size": 32,
                "prefix_budget": prefix,
                "precision": "bf16",
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "warmup_forwards": 0,
                "measured_forwards": 1,
            },
            "result": {
                "output_finite": True,
                "output_shape": [32, 3, 256, 256],
            },
        },
        "generation": {
            "schema_version": 3,
            "status": "completed",
            "counts": {"generated_image_count": 10_000, "real_image_count": 50_000},
            "metrics": {
                "frechet_inception_distance": fid,
                "precision": 0.50,
                "recall": 0.10,
            },
            "real_set": {"sha256": "5" * 64},
            "sample_provenance": {
                "checkpoint_step": PILOT_STEP,
                "checkpoint_sha256": checkpoint_sha,
                "sample_set_sha256": sample_sha,
                "weights": "ema",
                "selected_prefix_budget": prefix,
                "sampling": {
                    "sampler": "ddim",
                    "sample_steps": 100,
                    "num_samples": 10_000,
                    "seed": 20_260_825,
                    "start_index": 0,
                    "guidance_scale": 1.5,
                    "guidance_rescale": 0.0,
                    "cfg_batch_mode": "batched",
                    "precision": "bf16",
                    "class_schedule": "balanced_modulo",
                },
            },
        },
        "class_fidelity": {
            "status": "completed",
            "metrics": {
                "sample_count": 10_000,
                "top1_accuracy": 0.02,
                "top5_accuracy": 0.08,
                "predicted_class_fraction": 0.60,
                "normalized_predicted_class_entropy": 0.70,
            },
        },
        "checkpoint_eval": {
            "status": "completed",
            "checkpoint_step": PILOT_STEP,
            "weights": "ema",
            "config": {"model": model},
            "metrics": checkpoint_metrics,
        },
    }


def test_result_requires_shared_quality_class_and_mechanism_gates() -> None:
    preparation = _preparation()
    gate, preparation_identity = _execution_gate(preparation)
    gate_identity = _identity("execution_gate")
    audits = {method: _pilot_training_audit(method) for method in METHODS}
    arms = {
        f"{recipe}_{method}": _arm(method=method, pilot=recipe == "pilot_gamma5")
        for recipe in ("legacy_gamma0", "pilot_gamma5")
        for method in METHODS
    }
    source_names = {
        "preparation",
        "execution_gate",
        "pilot_cofitok_training_audit",
        "pilot_dense_training_audit",
        *{
            f"{arm}_{kind}"
            for arm in arms
            for kind in (
                "generation",
                "class_fidelity",
                "checkpoint_eval",
                "sampling_preflight",
            )
        },
    }
    identities = {name: _identity(name) for name in source_names}
    identities["preparation"] = preparation_identity
    identities["execution_gate"] = gate_identity
    result = build_result(
        preparation=preparation,
        preparation_identity=preparation_identity,
        execution_gate=gate,
        execution_gate_identity=gate_identity,
        pilot_training_audits=audits,
        arms=arms,
        source_identities=identities,
        builder_git=preparation["builder_git"],
    )
    assert validate_result(result) == result
    assert result["selection_status"].startswith("shared_min_snr_candidate")
    assert result["generation_advantage_proven"] is False
    assert result["authorization_boundary"] == RESULT_BOUNDARY

    failed_arms = deepcopy(arms)
    failed_arms["pilot_gamma5_dense_identity"]["class_fidelity"]["metrics"][
        "top5_accuracy"
    ] = 0.0
    failed = build_result(
        preparation=preparation,
        preparation_identity=preparation_identity,
        execution_gate=gate,
        execution_gate_identity=gate_identity,
        pilot_training_audits=audits,
        arms=failed_arms,
        source_identities=identities,
        builder_git=preparation["builder_git"],
    )
    assert failed["selection_status"] == "no_shared_min_snr_candidate_at_50k"
    assert failed["recommended_next_stage"]["execution_ready"] is False


def test_sampling_preflight_must_be_passed_and_checkpoint_bound() -> None:
    arm = _arm(method="cofitok", pilot=True)
    report = arm["sampling_preflight"]
    row = validate_sampling_preflight(
        report,
        arm="pilot_gamma5_cofitok",
        expected_prefix_budget=8,
    )
    assert row["checkpoint_sha256"] == "1" * 64
    failed = deepcopy(report)
    failed["status"] = "failed"
    with pytest.raises(ValueError, match="preflight protocol differs"):
        validate_sampling_preflight(
            failed,
            arm="pilot_gamma5_cofitok",
            expected_prefix_budget=8,
        )
