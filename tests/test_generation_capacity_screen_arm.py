from __future__ import annotations

import json
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    ROLLOUT_BATCH_SIZE,
    ROLLOUT_IMAGES,
    ROLLOUT_SEED,
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
)
from cofitok.generation.capacity_screen_arm import (
    ARM_VALIDATION_BOUNDARY,
    build_capacity_screen_arm_validation,
    validate_capacity_screen_arm_validation,
)
from cofitok.generation.capacity_screen_execution import (
    LAUNCH_BOUNDARY,
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
    capacity_screen_benchmark_root,
    capacity_screen_execution_lock_path,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256


REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "scale/generation-capacity-source-compatible-v1"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _git() -> dict[str, object]:
    return {"revision": REVISION, "branch": BRANCH, "tracked_dirty": False}


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str = "base256_cofitok",
) -> dict[str, object]:
    spec = ARM_SPECS[arm]
    # The execution contract is POSIX-path based even on the local Windows
    # test host.  Keep the fixture layout identical to the production
    # ``output_root/training/<arm>`` layout.
    native_root = (tmp_path / "capacity").resolve()
    # ``//?/C:/...`` is both an absolute PurePosixPath (matching the formal
    # Linux contract) and a valid Windows extended path to the pytest temp
    # directory.
    root_text = f"//?/{native_root.as_posix()}"
    root = Path(root_text)
    run_dirs = {
        name: (root / "training" / name).as_posix() for name in ARM_NAMES
    }
    run_dir = Path(run_dirs[arm])
    config_path = tmp_path / f"{arm}.json"
    config = {
        "data": {"dataset": "imagenet_256"},
        "model": {
            "token_count": 8 if spec["method"] == "cofitok" else 1,
            "token_spatial_strides": (
                [16, 16, 8, 8, 4, 1, 1, 1]
                if spec["method"] == "cofitok"
                else None
            ),
            "base_channels": spec["base_channels"],
        },
        "runtime": {"steps": 100_000},
    }
    if spec["method"] != "cofitok":
        del config["model"]["token_spatial_strides"]
    _write(config_path, config)
    training_report = run_dir / "training_report.json"
    resolved_config = {
        "data": {"dataset": "imagenet_256"},
        "model": config["model"],
        "runtime": {"steps": 100_000},
    }
    _write(
        training_report,
        {
            "config": resolved_config,
            "final_metrics": {
                "step": 10_000,
                "validation_epsilon_mse": 0.03,
            },
        },
    )
    checkpoint = run_dir / "checkpoint_step_00010000.pt"
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    _write(sidecar, {"checkpoint": checkpoint.name})
    checkpoint_identity = {
        "path": checkpoint.resolve().as_posix(),
        "bytes": checkpoint.stat().st_size,
        "sha256": "c" * 64,
        "integrity_manifest": file_identity(sidecar),
    }

    launch_path = tmp_path / "launch.json"
    launch = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "pass",
        "stage": "capacity_screen",
        "execution_checkout": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "source_evidence": {
            "configs": {
                name: (
                    file_identity(config_path)
                    if name == arm
                    else {"path": f"/{name}.json", "bytes": 1, "sha256": "d" * 64}
                )
                for name in ARM_NAMES
            }
        },
        "runtime_selection": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 64,
            "effective_batch_size": 64,
            "runtime_environment_sha256": "e" * 64,
            "dataset_identity_sha256": "f" * 64,
            "selection_sha256": "a" * 64,
            "benchmark_root": capacity_screen_benchmark_root(root_text),
        },
        "output_root": root_text,
        "execution_lock": capacity_screen_execution_lock_path(root_text),
        "run_dirs": run_dirs,
        "benchmark_root": capacity_screen_benchmark_root(root_text),
        "runtime_environment_sha256": "e" * 64,
        "dataset_identity_sha256": "f" * 64,
        "live_snapshot": {
            "schema_version": "cofitok_generation_capacity_screen_live_snapshot_v1",
            "role": "capacity_screen_live_prelaunch_snapshot",
            "status": "pass",
            "execution_checkout": {
                "revision": REVISION,
                "tree": TREE,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "gpu_inventory": [
                {
                    "memory_used_mib": 0,
                    "memory_total_mib": 97_887,
                    "utilization_percent": 0,
                }
            ],
            "gpu_compute_processes": [],
            "conflicting_processes": [],
            "output_root": root_text,
            "execution_lock": capacity_screen_execution_lock_path(root_text),
            "output_root_absent": True,
            "execution_lock_free": True,
            "free_bytes": 500 * 1024**3,
            "runtime_environment_sha256": "e" * 64,
            "dataset_identity_sha256": "f" * 64,
            "captured_at": "2026-09-03T00:00:00+00:00",
        },
        "training_state_absent_at_launch": True,
        "authorization_boundary": LAUNCH_BOUNDARY,
    }
    _write(launch_path, launch)

    prefix = int(spec["prefix_budget"])
    sampling_dir = root / arm / "sampling"
    generated_dir = sampling_dir / f"prefix_{prefix}"
    generated_dir.mkdir(parents=True)
    sampling = {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {"name": "cofitok.generation.GenerationSession", "version": 1},
        "sampler": "ddim",
        "num_samples": SAMPLE_COUNT,
        "start_index": 0,
        "batch_size": SAMPLE_BATCH_SIZE,
        "sample_steps": SAMPLE_STEPS,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, SAMPLE_STEPS),
        "prefix_budgets": [prefix],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": SAMPLE_SEED,
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
        "sample_set_digest": {
            "algorithm": "sha256",
            "framing": "filename_utf8_nul_file_bytes_nul",
        },
    }
    environment: dict[str, object] = {}
    environment_sha = runtime_environment_sha256(environment)
    sample_sha = "f" * 64
    sampling_report = sampling_dir / "sampling_report.json"
    sampling_manifest = sampling_dir / "sampling_manifest.json"
    sampling_progress = sampling_dir / "sampling_progress.json"
    provenance = {
        "report": sampling_report.resolve().as_posix(),
        "report_identity": {"path": "sampling", "bytes": 1, "sha256": "1" * 64},
        "manifest_identity": {"path": "manifest", "bytes": 1, "sha256": "2" * 64},
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_integrity_manifest": sidecar.resolve().as_posix(),
        "checkpoint_step": 10_000,
        "weights": "ema",
        "git": _git(),
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "selected_prefix_budget": prefix,
        "image_shape": [3, 256, 256],
        "sample_set_sha256": sample_sha,
        "sampling_progress": {
            "report": sampling_progress.resolve().as_posix(),
            "identity": {"path": "progress", "bytes": 1, "sha256": "3" * 64},
            "status": "completed",
            "invocation": 1,
            "completed_samples": SAMPLE_COUNT,
            "cumulative_elapsed_seconds": 1.0,
        },
        "sampling": sampling,
        "sampling_protocol_contract": {"valid": True, "issues": []},
    }
    _write(
        sampling_report,
        {
            "schema_version": 6,
            "status": "completed",
            "git": _git(),
            "runtime_environment": environment,
            "runtime_environment_sha256": environment_sha,
            "checkpoint": checkpoint.resolve().as_posix(),
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_integrity_manifest": sidecar.resolve().as_posix(),
            "checkpoint_step": 10_000,
            "weights": "ema",
            "sampling": sampling,
            "output_dirs": {str(prefix): generated_dir.resolve().as_posix()},
        },
    )
    _write(sampling_manifest, {})
    _write(sampling_progress, {})
    fake_images = [generated_dir / f"{index:06d}.png" for index in range(SAMPLE_COUNT)]
    import scripts.evaluate_generation_metrics as generation_metrics

    monkeypatch.setattr(generation_metrics, "find_images", lambda _: fake_images)
    monkeypatch.setattr(
        generation_metrics,
        "validate_sampling_provenance",
        lambda *_: provenance,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_arm.sample_set_sha256",
        lambda _: sample_sha,
    )

    metrics_report = root / arm / "evaluation" / "metrics" / "generation_metrics_report.json"
    _write(
        metrics_report,
        {
            "schema_version": 3,
            "role": "generation_directory_metrics_report",
            "status": "completed",
            "protocol": "torch_fidelity_directory_metrics",
            "git": _git(),
            "runtime_environment": environment,
            "runtime_environment_sha256": environment_sha,
            "sample_provenance": provenance,
            "counts": {"generated_image_count": SAMPLE_COUNT, "real_image_count": 50_000},
            "parameters": {
                "precision_recall_enabled": True,
                "batch_size": 64,
                "prc_batch_size": SAMPLE_COUNT,
                "seed": SAMPLE_SEED,
            },
            "real_set": {
                "image_count": 50_000,
                "digest_schema": "cofitok_image_tree_sha256_v1",
                "sha256": "4" * 64,
            },
            "metrics": {
                "frechet_inception_distance": 100.0,
                "inception_score_mean": 2.0,
                "inception_score_std": 0.1,
                "precision": 0.5,
                "recall": 0.1,
            },
        },
    )
    class_report = root / arm / "evaluation" / "class_fidelity" / "class_fidelity_report.json"
    _write(
        class_report,
        {
            "git": _git(),
            "runtime_environment_sha256": environment_sha,
            "sample_provenance": provenance,
            "classifier": {"name": "fixture"},
            "metrics": {
                "sample_count": SAMPLE_COUNT,
                "num_classes": 1000,
                "requested_class_count": 1000,
                "requested_count_min": 1,
                "requested_count_max": 1,
                "top1_accuracy": 0.01,
                "top5_accuracy": 0.05,
                "mean_target_probability": 0.01,
                "predicted_class_fraction": 0.5,
                "normalized_predicted_class_entropy": 0.6,
                "target_negative_log_likelihood": 5.0,
            },
        },
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_arm.validate_class_fidelity_report",
        lambda _: None,
    )

    checkpoint_report = root / arm / "evaluation" / "mechanism" / "checkpoint_evaluation_report.json"
    checkpoint_manifest = checkpoint_report.parent / "checkpoint_evaluation_manifest.json"
    request = {
        "num_images": 256,
        "timestep": 500,
        "random_orders": 4 if spec["method"] == "cofitok" else 0,
        "seed": SAMPLE_SEED,
        "weights": "ema",
        "precision": "bf16",
    }
    checkpoint_manifest_value = {
        "schema_version": 1,
        "role": "generation_checkpoint_evaluation_manifest",
        "git": _git(),
        "checkpoint": {
            "path": checkpoint.resolve().as_posix(),
            "bytes": checkpoint_identity["bytes"],
            "sha256": checkpoint_identity["sha256"],
            "integrity_manifest": checkpoint_identity["integrity_manifest"],
            "step": 10_000,
        },
        "request": request,
        "output_dir": checkpoint_report.parent.resolve().as_posix(),
        "report": checkpoint_report.resolve().as_posix(),
    }
    _write(checkpoint_manifest, checkpoint_manifest_value)
    ratios = [0.12, 0.10, 0.09, 0.08, 0.11, 0.16, 0.17, 0.17]
    checkpoint_metrics = {
        "evaluated_images": 256,
        "timestep": 500,
        "order_count": 6 if spec["method"] == "cofitok" else 1,
        "orders": {"ordered": {"endpoint_clean_mse": 0.03}},
    }
    if spec["method"] == "cofitok":
        checkpoint_metrics.update(
            {
                "component_energy_ratio_per_sample_mean": ratios,
                "ordered_rank_by_path_auc": 1,
                "zero_token_max_abs": 0.0,
                "shuffled_to_ordered_endpoint_ratio": 10.0,
            }
        )
    _write(
        checkpoint_report,
        {
            "schema_version": 2,
            "role": "generation_checkpoint_evaluation_report",
            "status": "completed",
            "git": _git(),
            "manifest": file_identity(checkpoint_manifest),
            "request": request,
            "checkpoint": checkpoint.resolve().as_posix(),
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_integrity_manifest": sidecar.resolve().as_posix(),
            "checkpoint_step": 10_000,
            "weights": "ema",
            "precision": "bf16",
            "config": resolved_config,
            "metrics": checkpoint_metrics,
        },
    )

    rollout_report = root / arm / "evaluation" / "rollout" / "rollout_stability_report.json"
    rollout_protocol = {
        "num_images": ROLLOUT_IMAGES,
        "batch_size": ROLLOUT_BATCH_SIZE,
        "teacher_timesteps": [999, 900, 750, 500, 250, 100, 10],
        "sample_steps": SAMPLE_STEPS,
        "guidance_scale": 1.5,
        "teacher_guidance_scale": 1.0,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "clip_x0": True,
        "precision": "bf16",
        "seed": ROLLOUT_SEED,
    }
    timesteps = select_sampling_timesteps(1000, SAMPLE_STEPS)
    _write(
        rollout_report,
        {
            "schema_version": 1,
            "status": "completed",
            "git": _git(),
            "checkpoint": checkpoint.resolve().as_posix(),
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_integrity_manifest": sidecar.resolve().as_posix(),
            "checkpoint_step": 10_000,
            "weights": "ema",
            "config": resolved_config,
            "protocol": rollout_protocol,
            "reconstruction_rollout": {
                "summary": {
                    "final_clipped_x0_mse": 0.03,
                    "final_to_best_x0_mse_amplification": 1.1,
                }
            },
            "free_sampling_rollout": {
                "steps": [
                    {"timestep": timestep, "predicted_x0_high_frequency_ratio": 0.2}
                    for timestep in timesteps
                ]
            },
        },
    )

    training_validation = {
        "schema_version": 1,
        "status": "pass",
        "role": "generation_capacity_qualification_partial_training_validation",
        "stage": spec["recipe_stage"],
        "runtime_environment_sha256": "e" * 64,
        "checkpoint": checkpoint_identity,
    }
    monkeypatch.setattr(
        "cofitok.generation.capacity_screen_arm.validate_capacity_qualification_partial_training",
        lambda **_: training_validation,
    )
    return {
        "arm": arm,
        "launch_receipt_path": launch_path,
        "expected_launch_receipt_sha256": file_sha256(launch_path),
        "config_path": config_path,
        "training_report_path": training_report,
        "sampling_report_path": sampling_report,
        "metrics_report_path": metrics_report,
        "class_fidelity_report_path": class_report,
        "checkpoint_evaluation_report_path": checkpoint_report,
        "rollout_report_path": rollout_report,
    }


def test_builds_physical_cofitok_arm_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    report = build_capacity_screen_arm_validation(**kwargs)
    assert report["status"] == "pass"
    assert report["arm"] == "base256_cofitok"
    assert report["sampling"]["sample_count"] == 1000
    assert report["checkpoint_evaluation"]["summary"]["ordered_rank_by_path_auc"] == 1
    assert report["checkpoint_evaluation"]["summary"]["utilization"][
        "coarse_token_energy_ratio"
    ] == pytest.approx(0.5)
    assert report["authorization_boundary"] == ARM_VALIDATION_BOUNDARY
    assert validate_capacity_screen_arm_validation(report) == report


def test_builds_dense_arm_with_matched_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch, arm="base128_dense_identity")
    report = build_capacity_screen_arm_validation(**kwargs)
    summary = report["checkpoint_evaluation"]["summary"]
    assert summary["order_count"] == 1
    assert "utilization" not in summary
    assert report["rollout"]["protocol"]["num_images"] == 64


def test_rejects_sampling_protocol_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    path = Path(kwargs["sampling_report_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    report["sampling"]["sample_steps"] = 50
    _write(path, report)
    with pytest.raises(ValueError, match="sampling protocol differs"):
        build_capacity_screen_arm_validation(**kwargs)


def test_replay_detects_raw_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    report = build_capacity_screen_arm_validation(**kwargs)
    path = Path(kwargs["metrics_report_path"])
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["metrics"]["frechet_inception_distance"] = 90.0
    _write(path, metrics)
    with pytest.raises(ValueError, match="exact replay"):
        validate_capacity_screen_arm_validation(report)


def test_contract_rejects_direct_300k_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    report = build_capacity_screen_arm_validation(**kwargs)
    report["authorization_boundary"] = dict(report["authorization_boundary"])
    report["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_screen_arm_validation(report)
