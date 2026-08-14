from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_completion_training import (
    CAPACITY_COMPLETION_TRAINING_BOUNDARY,
    validate_capacity_completion_training,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256, write_json_report
from test_generation_capacity_scaling_training import _metrics, _write_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_stability_capacity_probe_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
)
REVISION = "a" * 40
BRANCH = "scale/generation-capacity-probe-v1"
PARAMETERS = 250_153_763
MICRO_BATCH = 1
ACCUMULATION = 64


def _dataset() -> dict:
    spec = FORMAL_GENERATION_DATASETS["imagenet_256"]
    report = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": "imagenet_256",
        "dataset_root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": spec.manifest_bytes,
            "sha256": spec.manifest_sha256,
        },
        "splits": {"train": spec.train_images, "val": spec.val_images},
        "issues": [],
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def _run(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    config = load_config(CONFIG)
    resolved = config_to_dict(config)
    resolved["data"]["batch_size"] = MICRO_BATCH
    resolved["optimization"]["gradient_accumulation_steps"] = ACCUMULATION
    environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "fixture GPU"},
    }
    environment_sha = runtime_environment_sha256(environment)
    dataset = _dataset()
    source = _write_checkpoint(
        run,
        step=50_000,
        environment_sha=environment_sha,
        dataset_sha=dataset["identity_sha256"],
    )
    final = _write_checkpoint(
        run,
        step=100_000,
        environment_sha=environment_sha,
        dataset_sha=dataset["identity_sha256"],
    )
    write_json_report(run / "latest.json", final)
    rows = [_metrics(1)]
    for index, step in enumerate(range(1_000, 100_001, 1_000)):
        rows.append(_metrics(step, validation_index=index))
    (run / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    reconciliation = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 50_000,
        "metrics": str((run / "train_metrics.jsonl").resolve()),
        "retained_rows": 51,
        "orphaned_rows": 0,
        "orphan_archive": None,
        "orphan_sha256": None,
    }
    manifest = {
        "config": resolved,
        "config_path": str(CONFIG.resolve()),
        "output_dir": str(run.resolve()),
        "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
        "torch_version": "fixture",
        "cuda_version": "fixture",
        "device": "cuda",
        "device_name": "fixture GPU",
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "dataset_provenance": dataset,
        "training_authorization": None,
        "parameter_count": PARAMETERS,
        "trainable_parameter_count": PARAMETERS,
        "resume": str((run / source["checkpoint"]).resolve()),
        "resume_revision_transition": None,
        "metrics_resume_reconciliation": reconciliation,
    }
    write_json_report(run / "run_manifest.json", manifest)
    write_json_report(
        run / "training_report.json",
        {
            **manifest,
            "completed_steps": 100_000,
            "target_steps": 100_000,
            "training_complete": True,
            "stop_requested": False,
            "stop_signal": None,
            "final_metrics": rows[-1],
            "segment_elapsed_seconds": 50_000.0,
            "elapsed_seconds": 100_000.0,
            "peak_vram_bytes": 80_000_000_000,
            "latest_checkpoint": final,
        },
    )
    source_path = run / source["checkpoint"]
    source_integrity = source_path.with_name(f"{source_path.name}.integrity.json")
    archive_checkpoint = tmp_path / "archive/cofitok" / source_path.name
    archive_checkpoint.parent.mkdir(parents=True)
    archive_integrity = archive_checkpoint.with_name(
        f"{archive_checkpoint.name}.integrity.json"
    )
    os.link(source_path, archive_checkpoint)
    os.link(source_integrity, archive_integrity)
    archive = {
        "schema_version": 1,
        "status": "pass",
        "role": "capacity_completion_50k_source_checkpoint_archive",
        "methods": {
            "cofitok": {
                "source_checkpoint": file_identity(source_path),
                "source_integrity_manifest": file_identity(source_integrity),
                "archive_checkpoint": file_identity(archive_checkpoint),
                "archive_integrity_manifest": file_identity(archive_integrity),
            }
        },
    }
    archive_path = tmp_path / "source_archive.json"
    write_json_report(archive_path, archive)
    return run, archive_path


def _validate(run: Path, archive: Path) -> dict:
    return validate_capacity_completion_training(
        report_path=run / "training_report.json",
        config_path=CONFIG,
        source_archive_path=archive,
        expected_source_archive_sha256=file_sha256(archive),
        method="cofitok",
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_parameter_count=PARAMETERS,
        expected_micro_batch_size=MICRO_BATCH,
        expected_gradient_accumulation_steps=ACCUMULATION,
    )


def test_exact_50k_to_100k_completion_is_valid_and_non_authorizing(
    tmp_path: Path,
) -> None:
    run, archive = _run(tmp_path)
    report = _validate(run, archive)
    assert report["completed_steps"] == 100_000
    assert report["images_seen"] == 6_400_000
    assert report["validation_event_count"] == 100
    assert report["source_checkpoint"]["step"] == 50_000
    assert report["checkpoint"]["step"] == 100_000
    assert report["authorization_boundary"] == CAPACITY_COMPLETION_TRAINING_BOUNDARY
    assert report["authorization_boundary"]["additional_training_allowed"] is False


def test_completion_training_rejects_incomplete_or_changed_archive(
    tmp_path: Path,
) -> None:
    run, archive = _run(tmp_path)
    training = json.loads((run / "training_report.json").read_text())
    training["training_complete"] = False
    write_json_report(run / "training_report.json", training)
    with pytest.raises(ValueError, match="not complete"):
        _validate(run, archive)

    run, archive = _run(tmp_path / "second")
    with archive.open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="source archive SHA256 differs"):
        validate_capacity_completion_training(
            report_path=run / "training_report.json",
            config_path=CONFIG,
            source_archive_path=archive,
            expected_source_archive_sha256="0" * 64,
            method="cofitok",
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_parameter_count=PARAMETERS,
            expected_micro_batch_size=MICRO_BATCH,
            expected_gradient_accumulation_steps=ACCUMULATION,
        )


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.validate_generation_capacity_completion_training",
        "scripts.verify_generation_capacity_completion_training",
    ],
)
def test_capacity_completion_training_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)
