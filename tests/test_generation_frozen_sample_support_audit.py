from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.image_integrity import image_tree_sha256, sample_set_sha256
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256
from scripts.build_generation_frozen_sample_support_audit import (
    FrozenSupportContract,
    _analyze_image_paths,
    build_frozen_sample_support_audit,
)
from scripts.evaluate_generation_metrics import validate_sampling_provenance


AUDIT_GIT = {
    "revision": "a" * 40,
    "branch": "scale/test-frozen-support-audit",
    "tracked_dirty": False,
    "full_clean": True,
}
AUDIT_RUNTIME = {
    "schema_version": 1,
    "purpose": "unit_test_audit",
    "device": {"type": "cpu"},
    "environment_variables": {"CUDA_VISIBLE_DEVICES": ""},
}
EVALUATION_GIT = {
    "revision": "e" * 40,
    "branch": "scale/test-evaluation",
    "tracked_dirty": False,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _pattern(*, class_id: int, sample_id: int, size: int = 8) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    array = np.empty((size, size, 3), dtype=np.uint8)
    array[..., 0] = (class_id * 73 + sample_id * 17 + xx * 11) % 256
    array[..., 1] = (class_id * 31 + sample_id * 29 + yy * 13) % 256
    array[..., 2] = (class_id * 97 + sample_id * 7 + (xx + yy) * 5) % 256
    return array


def _write_image(
    path: Path,
    array: np.ndarray,
    *,
    compress_level: int = 6,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(
        path, format="PNG", compress_level=compress_level
    )


def _sampling_protocol(
    *,
    sample_count: int,
    budget: int,
    class_schedule: str,
    image_size: int,
) -> dict:
    return {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        },
        "sampler": "ddim",
        "num_samples": sample_count,
        "num_train_timesteps": 1_000,
        "sample_steps": 100,
        "actual_timesteps": select_sampling_timesteps(1_000, 100),
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "batch_size": 32,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "precision": "bf16",
        "seed": 0,
        "start_index": 0,
        "class_schedule": class_schedule,
        "image_shape": [3, image_size, image_size],
        "prefix_budgets": [budget],
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


def _sampling_evidence(
    root: Path,
    *,
    method: str,
    image_dir: Path,
    budget: int,
    checkpoint_sha: str,
    class_schedule: str,
) -> tuple[Path, dict]:
    sampling_root = root / method / "samples"
    report_path = sampling_root / "sampling_report.json"
    manifest_path = sampling_root / "sampling_manifest.json"
    progress_path = sampling_root / "sampling_progress.json"
    images = sorted(image_dir.glob("*.png"))
    sample_sha = sample_set_sha256(images)
    environment = {"schema_version": 1, "purpose": "unit_test_sampling"}
    environment_sha = runtime_environment_sha256(environment)
    sampling = _sampling_protocol(
        sample_count=len(images),
        budget=budget,
        class_schedule=class_schedule,
        image_size=8,
    )
    checkpoint = (root / method / "checkpoint_step_00050000.pt").resolve()
    integrity = Path(f"{checkpoint}.integrity.json")
    output_dirs = {str(budget): image_dir.resolve().as_posix()}
    sample_sets = {str(budget): {"count": len(images), "sha256": sample_sha}}
    shared = {
        "git": EVALUATION_GIT,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_integrity_manifest": integrity.as_posix(),
        "checkpoint_step": 50_000,
        "weights": "ema",
        "sampling": sampling,
        "output_dirs": output_dirs,
    }
    manifest = {
        "schema_version": 3,
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        **shared,
    }
    _write_json(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)
    progress = {
        "schema_version": 3,
        "status": "completed",
        "sampling_manifest_sha256": manifest_sha,
        "completed_samples": len(images),
        "prefix_budgets": [budget],
        "sample_sets": sample_sets,
        "invocation": 1,
        "cumulative_elapsed_seconds": 1.0,
    }
    _write_json(progress_path, progress)
    report = {
        "schema_version": 6,
        "status": "completed",
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "sampling_manifest_sha256": manifest_sha,
        "sampling_progress": progress_path.resolve().as_posix(),
        "sample_sets": sample_sets,
        **shared,
    }
    _write_json(report_path, report)
    verified = validate_sampling_provenance(report_path, image_dir, images)
    return report_path, verified


def _metrics_evidence(
    path: Path,
    *,
    generated_dir: Path,
    real_dir: Path,
    real_set: dict,
    sampling: dict,
    metrics: dict,
) -> dict:
    source = {
        name: sampling[name]
        for name in (
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_integrity_manifest",
            "checkpoint_step",
            "weights",
            "git",
            "runtime_environment_sha256",
            "selected_prefix_budget",
            "image_shape",
            "sample_set_sha256",
            "sampling",
            "sampling_protocol_contract",
        )
    }
    source["sampling_progress"] = {
        name: sampling["sampling_progress"][name]
        for name in (
            "report",
            "status",
            "invocation",
            "completed_samples",
            "cumulative_elapsed_seconds",
        )
    }
    source["report"] = sampling["report"]
    source["schema_version"] = 2
    source["status"] = "completed"
    environment = {"schema_version": 1, "purpose": "unit_test_metrics"}
    report = {
        "schema_version": 2,
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "git": EVALUATION_GIT,
        "runtime_environment": environment,
        "runtime_environment_sha256": runtime_environment_sha256(environment),
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "paths": {
            "generated_dir": generated_dir.resolve().as_posix(),
            "real_dir": real_dir.resolve().as_posix(),
        },
        "counts": {
            "generated_image_count": 4,
            "real_image_count": 6,
        },
        "real_set": real_set,
        "sample_provenance": source,
        "parameters": {
            "batch_size": 64,
            "cuda": True,
            "prc_batch_size": 10_000,
            "precision_recall_enabled": True,
            "real_cache_name": "unit_test_cache",
            "requested_real_cache_name": "unit_test_cache",
            "samples_find_deep": True,
            "samples_shuffle": False,
            "seed": 2027,
        },
        "metrics": metrics,
    }
    _write_json(path, report)
    return report


def _metric_values(fid: float, precision: float, recall: float) -> dict:
    return {
        "frechet_inception_distance": fid,
        "inception_score_mean": 2.0,
        "inception_score_std": 0.1,
        "precision": precision,
        "recall": recall,
        "f_score": 2.0 * precision * recall / (precision + recall),
    }


def _fixture(tmp_path: Path, *, class_schedule: str = "balanced_modulo") -> dict:
    real_dir = tmp_path / "real"
    for class_id in range(2):
        for sample_id in range(3):
            _write_image(
                real_dir / f"class_{class_id:02d}" / f"{sample_id:03d}.png",
                _pattern(class_id=class_id, sample_id=sample_id),
            )
    cofitok_dir = tmp_path / "cofitok" / "samples" / "prefix_8"
    dense_dir = tmp_path / "dense" / "samples" / "prefix_1"
    for pass_id in range(2):
        for class_id in range(2):
            index = pass_id * 2 + class_id
            _write_image(
                cofitok_dir / f"{index:06d}.png",
                _pattern(class_id=class_id, sample_id=pass_id + 5),
            )
            _write_image(
                dense_dir / f"{index:06d}.png",
                _pattern(class_id=class_id, sample_id=pass_id + 9),
            )
    all_real = sorted(path for path in real_dir.rglob("*.png"))
    real_set = {
        "root": real_dir.resolve().as_posix(),
        "digest_schema": "cofitok_image_tree_sha256_v1",
        "image_count": 6,
        "sha256": image_tree_sha256(all_real, root=real_dir),
    }
    cofitok_sampling_path, cofitok_sampling = _sampling_evidence(
        tmp_path,
        method="cofitok",
        image_dir=cofitok_dir,
        budget=8,
        checkpoint_sha="c" * 64,
        class_schedule=class_schedule,
    )
    dense_sampling_path, dense_sampling = _sampling_evidence(
        tmp_path,
        method="dense",
        image_dir=dense_dir,
        budget=1,
        checkpoint_sha="d" * 64,
        class_schedule=class_schedule,
    )
    cofitok_metric_values = _metric_values(120.0, 0.7, 0.02)
    dense_metric_values = _metric_values(130.0, 0.72, 0.03)
    cofitok_metrics_path = tmp_path / "cofitok" / "metrics.json"
    dense_metrics_path = tmp_path / "dense" / "metrics.json"
    cofitok_metrics = _metrics_evidence(
        cofitok_metrics_path,
        generated_dir=cofitok_dir,
        real_dir=real_dir,
        real_set=real_set,
        sampling=cofitok_sampling,
        metrics=cofitok_metric_values,
    )
    dense_metrics = _metrics_evidence(
        dense_metrics_path,
        generated_dir=dense_dir,
        real_dir=real_dir,
        real_set=real_set,
        sampling=dense_sampling,
        metrics=dense_metric_values,
    )
    gate_path = tmp_path / "promotion_gate.json"
    gate = {
        "schema_version": 2,
        "status": "fail",
        "decision": "hold",
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "provenance_contract": {
            "evaluation_revision": EVALUATION_GIT["revision"],
            "evaluation_branch": EVALUATION_GIT["branch"],
            "training_revision": "t" * 40,
            "training_branch": "scale/test-training",
        },
        "thresholds": {"min_samples": 4},
        "source_reports": {
            "cofitok_generation": file_identity(cofitok_metrics_path),
            "dense_generation": file_identity(dense_metrics_path),
        },
        "gates": [
            {
                "name": "generation_metrics_complete",
                "passed": True,
                "evidence": {
                    "cofitok": {
                        name: cofitok_metric_values[name]
                        for name in (
                            "frechet_inception_distance",
                            "inception_score_mean",
                            "inception_score_std",
                            "precision",
                            "recall",
                        )
                    },
                    "dense": {
                        name: dense_metric_values[name]
                        for name in (
                            "frechet_inception_distance",
                            "inception_score_mean",
                            "inception_score_std",
                            "precision",
                            "recall",
                        )
                    },
                },
            },
            {
                "name": "matched_real_set_provenance",
                "passed": True,
                "evidence": {
                    method: {**real_set, "valid": True, "real_cache_name": "test"}
                    for method in ("cofitok", "dense_identity")
                },
            },
            {
                "name": "matched_sampling_provenance",
                "passed": True,
                "evidence": {
                    "protocols_match": True,
                    "cofitok_checkpoint_sha256": "c" * 64,
                    "cofitok_checkpoint_step": 50_000,
                    "cofitok_prefix_budget": 8,
                    "cofitok_sample_set_sha256": cofitok_sampling[
                        "sample_set_sha256"
                    ],
                    "dense_checkpoint_sha256": "d" * 64,
                    "dense_checkpoint_step": 50_000,
                    "dense_prefix_budget": 1,
                    "dense_sample_set_sha256": dense_sampling[
                        "sample_set_sha256"
                    ],
                },
            },
            {
                "name": "absolute_fid_quality",
                "passed": False,
                "evidence": {"cofitok_fid": 120.0, "max_absolute_fid": 100.0},
            },
        ],
        "summary": {
            "cofitok_fid": cofitok_metric_values["frechet_inception_distance"],
            "cofitok_inception_score": cofitok_metric_values[
                "inception_score_mean"
            ],
            "cofitok_precision": cofitok_metric_values["precision"],
            "cofitok_recall": cofitok_metric_values["recall"],
            "dense_fid": dense_metric_values["frechet_inception_distance"],
            "dense_inception_score": dense_metric_values["inception_score_mean"],
            "dense_precision": dense_metric_values["precision"],
            "dense_recall": dense_metric_values["recall"],
        },
    }
    _write_json(gate_path, gate)
    dataset_manifest = tmp_path / "image_manifest.jsonl"
    dataset_manifest.write_text('{"dataset":"unit_test"}\n', encoding="utf-8")
    contract = FrozenSupportContract(
        promotion_gate_sha256=file_sha256(gate_path),
        dataset_manifest_sha256=file_sha256(dataset_manifest),
        dataset_manifest_bytes=dataset_manifest.stat().st_size,
        real_set_sha256=real_set["sha256"],
        class_count=2,
        real_images_per_class=3,
        generated_sample_count=4,
        image_width=8,
        image_height=8,
    )
    return {
        "kwargs": {
            "cofitok_sampling_report": cofitok_sampling_path,
            "dense_sampling_report": dense_sampling_path,
            "cofitok_metrics_report": cofitok_metrics_path,
            "dense_metrics_report": dense_metrics_path,
            "promotion_gate": gate_path,
            "real_dir": real_dir,
            "dataset_manifest": dataset_manifest,
            "audit_git": AUDIT_GIT,
            "audit_runtime_environment": AUDIT_RUNTIME,
            "contract": contract,
        },
        "cofitok_dir": cofitok_dir,
        "dense_dir": dense_dir,
        "real_dir": real_dir,
        "gate": gate,
        "cofitok_metrics": cofitok_metrics,
        "dense_metrics": dense_metrics,
    }


def test_frozen_support_audit_is_deterministic_and_non_authorizing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = build_frozen_sample_support_audit(**fixture["kwargs"])
    replay = build_frozen_sample_support_audit(**fixture["kwargs"])

    assert report == replay
    assert report["status"] == "completed"
    assert report["claim_boundary"]["supplemental_non_authorizing"] is True
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["gpu_execution_authorized"] is False
    assert report["interpretation_policy"]["thresholded_gate"] is False
    assert report["sources"]["real_set"]["image_count"] == 6
    assert report["sources"]["cofitok"]["sampling"]["sample_set_sha256"]
    assert report["support_statistics"]["cofitok"]["image_count"] == 4
    assert report["contrasts_to_real"]["cofitok"][
        "mean_absolute_neighbor_gradient_ratio_to_real"
    ] > 0.0


def test_frozen_support_analysis_detects_decoded_duplicate_across_png_encoding(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"{index:06d}.png" for index in range(4)]
    duplicate = _pattern(class_id=0, sample_id=1)
    _write_image(paths[0], duplicate, compress_level=0)
    _write_image(paths[1], duplicate, compress_level=9)
    _write_image(paths[2], _pattern(class_id=1, sample_id=2))
    _write_image(paths[3], _pattern(class_id=1, sample_id=3))
    assert paths[0].read_bytes() != paths[1].read_bytes()
    contract = FrozenSupportContract(
        class_count=2,
        real_images_per_class=3,
        generated_sample_count=4,
        image_width=8,
        image_height=8,
    )

    analysis = _analyze_image_paths(
        paths, contract=contract, samples_per_class=2
    )

    assert analysis.report["decoded_pixel_duplicate_count"] == 1


def test_frozen_support_audit_rejects_unexpected_generated_png(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_image(
        fixture["cofitok_dir"] / "999999.png",
        _pattern(class_id=0, sample_id=20),
    )

    with pytest.raises(ValueError, match="generated image count"):
        build_frozen_sample_support_audit(**fixture["kwargs"])


def test_frozen_support_audit_rejects_sample_set_content_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_image(
        fixture["dense_dir"] / "000000.png",
        _pattern(class_id=1, sample_id=30),
    )

    with pytest.raises(ValueError, match="sample-set SHA256"):
        build_frozen_sample_support_audit(**fixture["kwargs"])


def test_frozen_support_audit_rejects_real_class_size_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["real_dir"] / "class_00" / "000.png").unlink()

    with pytest.raises(ValueError, match="real class image count"):
        build_frozen_sample_support_audit(**fixture["kwargs"])


def test_frozen_support_audit_rejects_class_schedule_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, class_schedule="random")

    with pytest.raises(ValueError, match="class_schedule"):
        build_frozen_sample_support_audit(**fixture["kwargs"])


def test_frozen_support_audit_rejects_gate_bound_metrics_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    metrics_path = fixture["kwargs"]["cofitok_metrics_report"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["metrics"]["recall"] = 0.5
    _write_json(metrics_path, metrics)

    with pytest.raises(ValueError, match="promotion gate cofitok_generation identity"):
        build_frozen_sample_support_audit(**fixture["kwargs"])


def test_frozen_support_runbook_is_cpu_only_and_fail_closed() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_stability_frozen_existing_sample_support_audit.sh"
    ).read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "export PYTHONPATH=.:src" in source
    assert "EXPECTED_AUDIT_REVISION=${EXPECTED_AUDIT_REVISION:?" in source
    assert "git status --porcelain" in source
    assert "git status --porcelain --untracked-files=no" not in source
    assert "nice -n 15 ionice -c3" in source
    assert "[[ ! -e \"$AUDIT_REPORT\" ]]" in source
    assert "full_training_launch_allowed" in source
    assert "gpu_execution_authorized" in source
    assert "nvidia-smi" not in source

    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_generation_frozen_sample_support_audit.py"),
            "--help",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
