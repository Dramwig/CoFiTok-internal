from __future__ import annotations

import json
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_confirmation import (
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
)
from cofitok.generation.capacity_confirmation_arm import (
    ARM_VALIDATION_BOUNDARY,
    build_capacity_confirmation_arm_validation,
    validate_capacity_confirmation_arm_validation,
)
from cofitok.generation.capacity_confirmation_execution import (
    LAUNCH_RECEIPT_ROLE,
    LAUNCH_RECEIPT_SCHEMA,
)
from cofitok.generation.capacity_screen import ARM_NAMES, ARM_SPECS
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
    root = tmp_path / "capacity" / "confirmation_10000"
    arm_root = root / arm
    checkpoint = tmp_path / arm / "checkpoint_step_00010000.pt"
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write(sidecar, {"checkpoint": checkpoint.name})
    checkpoint_row = {
        "path": checkpoint.resolve().as_posix(),
        "bytes": checkpoint.stat().st_size,
        "sha256": "c" * 64,
        "step": 10_000,
        "integrity_manifest": file_identity(sidecar),
    }
    screen_path = tmp_path / f"screen_{arm}.json"
    screen_validation = {
        "arm": arm,
        "training": {
            "validation_epsilon_mse": 0.03,
            "checkpoint": checkpoint_row,
        },
        "checkpoint_evaluation": {
            "summary": {
                "ordered_endpoint_clean_mse": 0.03,
                "order_count": 6 if spec["method"] == "cofitok" else 1,
                **(
                    {
                        "ordered_rank_by_path_auc": 1,
                        "zero_token_max_abs": 0.0,
                        "shuffled_to_ordered_endpoint_ratio": 10.0,
                        "utilization": {
                            "coarse_token_energy_ratio": 0.5,
                            "tail_two_energy_ratio": 0.34,
                            "max_single_token_energy_ratio": 0.17,
                        },
                    }
                    if spec["method"] == "cofitok"
                    else {}
                ),
            }
        },
        "rollout": {
            "summary": {
                "final_reconstruction_x0_mse": 0.03,
                "predicted_x0_high_frequency_ratio": {
                    "91": 0.2,
                    "192": 0.2,
                    "394": 0.2,
                    "595": 0.2,
                },
            }
        },
    }
    _write(screen_path, screen_validation)
    screen_id = file_identity(screen_path)

    environment: dict[str, object] = {}
    environment_sha = runtime_environment_sha256(environment)
    output_dirs = {name: (root / name).resolve().as_posix() for name in ARM_NAMES}
    launch_path = tmp_path / "confirmation_launch.json"
    launch = {
        "schema_version": LAUNCH_RECEIPT_SCHEMA,
        "role": LAUNCH_RECEIPT_ROLE,
        "execution_checkout": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "source_evidence": {
            "capacity_screen_arm_validations": {
                name: screen_id if name == arm else {
                    "path": f"/{name}.json",
                    "bytes": 1,
                    "sha256": "d" * 64,
                }
                for name in ARM_NAMES
            }
        },
        "frozen_arms": {
            name: {
                "screen_arm_validation": screen_id if name == arm else {
                    "path": f"/{name}.json",
                    "bytes": 1,
                    "sha256": "d" * 64,
                },
                "checkpoint": checkpoint_row if name == arm else {},
            }
            for name in ARM_NAMES
        },
        "output_dirs": output_dirs,
        "runtime_environment_sha256": environment_sha,
    }
    _write(launch_path, launch)

    prefix = int(spec["prefix_budget"])
    sampling_dir = arm_root / "samples_10000_ddim100_cfg15"
    generated_dir = sampling_dir / f"prefix_{prefix}"
    generated_dir.mkdir(parents=True)
    sampling = {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        },
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
    sampling_report = sampling_dir / "sampling_report.json"
    sampling_progress = sampling_dir / "sampling_progress.json"
    sample_sha = "f" * 64
    provenance = {
        "report": sampling_report.resolve().as_posix(),
        "report_identity": {"path": "sampling", "bytes": 1, "sha256": "1" * 64},
        "manifest_identity": {"path": "manifest", "bytes": 1, "sha256": "2" * 64},
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_row["sha256"],
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
            "checkpoint_sha256": checkpoint_row["sha256"],
            "checkpoint_integrity_manifest": sidecar.resolve().as_posix(),
            "checkpoint_step": 10_000,
            "weights": "ema",
            "sampling": sampling,
            "output_dirs": {str(prefix): generated_dir.resolve().as_posix()},
        },
    )
    _write(sampling_progress, {})
    fake_images = [generated_dir / f"{index:06d}.png" for index in range(SAMPLE_COUNT)]
    import scripts.evaluate_generation_metrics as generation_metrics

    monkeypatch.setattr(generation_metrics, "find_images", lambda _: fake_images)
    monkeypatch.setattr(
        generation_metrics, "validate_sampling_provenance", lambda *_: provenance
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_arm.sample_set_sha256",
        lambda _: sample_sha,
    )

    metrics_report = arm_root / "metrics" / "generation_metrics_report.json"
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
            "counts": {
                "generated_image_count": SAMPLE_COUNT,
                "real_image_count": 50_000,
            },
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
                "frechet_inception_distance": 90.0,
                "inception_score_mean": 3.0,
                "inception_score_std": 0.1,
                "precision": 0.5,
                "recall": 0.2,
            },
        },
    )
    class_report = arm_root / "class_fidelity" / "class_fidelity_report.json"
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
                "requested_count_min": 10,
                "requested_count_max": 10,
                "top1_accuracy": 0.02,
                "top5_accuracy": 0.08,
                "mean_target_probability": 0.01,
                "predicted_class_fraction": 0.5,
                "normalized_predicted_class_entropy": 0.6,
                "target_negative_log_likelihood": 5.0,
            },
        },
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_arm.validate_class_fidelity_report",
        lambda _: None,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_arm.validate_capacity_screen_arm_validation",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_confirmation_arm.validate_capacity_confirmation_launch_receipt_contract",
        lambda value, **_: value,
    )
    return {
        "arm": arm,
        "launch_receipt_path": launch_path,
        "expected_launch_receipt_sha256": file_sha256(launch_path),
        "screen_arm_validation_path": screen_path,
        "expected_screen_arm_validation_sha256": file_sha256(screen_path),
        "sampling_report_path": sampling_report,
        "metrics_report_path": metrics_report,
        "class_fidelity_report_path": class_report,
    }


def test_builds_physical_confirmation_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    report = build_capacity_confirmation_arm_validation(**kwargs)
    assert report["status"] == "pass"
    assert report["sampling"]["sample_count"] == 10_000
    assert report["frozen_training"]["training_performed"] is False
    assert report["authorization_boundary"] == ARM_VALIDATION_BOUNDARY
    assert validate_capacity_confirmation_arm_validation(report) == report


def test_builds_dense_confirmation_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch, arm="base128_dense_identity")
    report = build_capacity_confirmation_arm_validation(**kwargs)
    assert report["method"] == "dense_identity"
    assert report["sampling"]["sampling"]["prefix_budgets"] == [1]


def test_rejects_confirmation_sample_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    path = Path(kwargs["sampling_report_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    report["sampling"]["num_samples"] = 1_000
    _write(path, report)
    with pytest.raises(ValueError, match="sampling differs"):
        build_capacity_confirmation_arm_validation(**kwargs)


def test_replay_detects_confirmation_metric_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    report = build_capacity_confirmation_arm_validation(**kwargs)
    path = Path(kwargs["metrics_report_path"])
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["metrics"]["frechet_inception_distance"] = 80.0
    _write(path, metrics)
    with pytest.raises(ValueError, match="exact replay"):
        validate_capacity_confirmation_arm_validation(report)


def test_contract_rejects_training_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path, monkeypatch)
    report = build_capacity_confirmation_arm_validation(**kwargs)
    report["authorization_boundary"] = dict(report["authorization_boundary"])
    report["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_confirmation_arm_validation(report)
