from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.capacity_scaling_training import (
    CAPACITY_SCALING_TRAINING_BOUNDARY,
    validate_capacity_scaling_partial_training,
)
from cofitok.reporting import file_sha256, write_json_report


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


def _dataset() -> dict[str, object]:
    spec = FORMAL_GENERATION_DATASETS["imagenet_256"]
    report: dict[str, object] = {
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


def _metrics(step: int, *, validation_index: int | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "total_loss": 1.0,
        "epsilon_loss": 0.9,
        "grad_norm": 2.0,
        "learning_rate": 0.0001,
        "elapsed_seconds": float(max(1, step - 10_000)),
        "cumulative_elapsed_seconds": float(step),
        "samples_seen": step * 64,
    }
    if validation_index is not None:
        row.update(
            {
                "validation_epsilon_mse": 0.8,
                "validation_event_index": validation_index,
                "validation_batch_index": validation_index,
                "validation_num_images": 16,
                "validation_noise_seed": 2027 + validation_index,
            }
        )
    return row


def _write_checkpoint(
    run: Path,
    *,
    step: int,
    environment_sha: str,
    dataset_sha: str,
) -> dict[str, object]:
    checkpoint = run / f"checkpoint_step_{step:08d}.pt"
    checkpoint.write_bytes(f"capacity scaling checkpoint {step}".encode())
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_format_version": 1,
        "step": step,
        "git_revision": REVISION,
        "git_branch": BRANCH,
        "git_dirty": False,
        "runtime_environment_sha256": environment_sha,
        "dataset_identity_sha256": dataset_sha,
    }
    write_json_report(
        checkpoint.with_name(f"{checkpoint.name}.integrity.json"),
        integrity,
    )
    return {
        **integrity,
        "integrity_manifest": f"{checkpoint.name}.integrity.json",
    }


def _run(tmp_path: Path) -> Path:
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
    source_latest = _write_checkpoint(
        run,
        step=10_000,
        environment_sha=environment_sha,
        dataset_sha=str(dataset["identity_sha256"]),
    )
    latest = _write_checkpoint(
        run,
        step=50_000,
        environment_sha=environment_sha,
        dataset_sha=str(dataset["identity_sha256"]),
    )
    write_json_report(run / "latest.json", latest)

    rows = [_metrics(1)]
    for index, step in enumerate(range(1_000, 50_001, 1_000)):
        rows.append(_metrics(step, validation_index=index))
    (run / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    reconciliation = {
        "schema_version": 1,
        "status": "unchanged",
        "resume_step": 10_000,
        "metrics": str((run / "train_metrics.jsonl").resolve()),
        "retained_rows": 11,
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
        "resume": str((run / source_latest["checkpoint"]).resolve()),
        "resume_revision_transition": None,
        "metrics_resume_reconciliation": reconciliation,
    }
    write_json_report(run / "run_manifest.json", manifest)
    write_json_report(
        run / "training_report.json",
        {
            **manifest,
            "completed_steps": 50_000,
            "target_steps": 100_000,
            "training_complete": False,
            "stop_requested": False,
            "stop_signal": None,
            "final_metrics": rows[-1],
            "segment_elapsed_seconds": 40_000.0,
            "elapsed_seconds": 50_000.0,
            "peak_vram_bytes": 80_000_000_000,
            "latest_checkpoint": latest,
        },
    )
    return run


def _validate(run: Path) -> dict[str, object]:
    return validate_capacity_scaling_partial_training(
        report_path=run / "training_report.json",
        config_path=CONFIG,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_parameter_count=PARAMETERS,
        expected_micro_batch_size=MICRO_BATCH,
        expected_gradient_accumulation_steps=ACCUMULATION,
    )


def test_exact_10k_to_50k_capacity_resume_is_valid_and_non_authorizing(
    tmp_path: Path,
) -> None:
    report = _validate(_run(tmp_path))
    assert report["status"] == "pass"
    assert report["source_step"] == 10_000
    assert report["completed_steps"] == 50_000
    assert report["images_seen"] == 3_200_000
    assert report["validation_event_count"] == 50
    assert report["source_checkpoint"]["step"] == 10_000
    assert report["checkpoint"]["step"] == 50_000
    assert report["authorization_boundary"] == CAPACITY_SCALING_TRAINING_BOUNDARY
    assert report["authorization_boundary"][
        "configured_100k_training_complete"
    ] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_complete", True),
        ("completed_steps", 49_999),
        ("target_steps", 50_000),
        ("stop_requested", True),
        ("stop_signal", 15),
    ),
)
def test_capacity_scaling_rejects_wrong_stop_state(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run = _run(tmp_path)
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    write_json_report(path, report)
    with pytest.raises(ValueError, match="exact clean stop"):
        _validate(run)


def test_capacity_scaling_rejects_resume_or_metric_drift(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["resume"] = str((run / "checkpoint_step_00005000.pt").resolve())
    write_json_report(path, report)
    with pytest.raises(ValueError, match="exact-resume evidence differs"):
        _validate(run)

    run = _run(tmp_path / "second")
    metrics = run / "train_metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    rows.pop(-2)
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="validation events differ"):
        _validate(run)


def test_capacity_scaling_accepts_recovery_resume_inside_bound_segment(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    report_path = run / "training_report.json"
    manifest_path = run / "run_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report["resume"] = str((run / "checkpoint_step_00020000.pt").resolve())
    report["metrics_resume_reconciliation"]["resume_step"] = 20_000
    manifest["resume"] = report["resume"]
    manifest["metrics_resume_reconciliation"] = report[
        "metrics_resume_reconciliation"
    ]
    write_json_report(report_path, report)
    write_json_report(manifest_path, manifest)
    validated = _validate(run)
    assert validated["last_resume_step"] == 20_000
    assert validated["recovery_resume_used"] is True


def test_capacity_scaling_rejects_source_or_target_checkpoint_tamper(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)
    (run / "checkpoint_step_00010000.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        _validate(run)

    run = _run(tmp_path / "second")
    (run / "checkpoint_step_00055000.pt").write_bytes(b"later")
    with pytest.raises(ValueError, match="after the stop step"):
        _validate(run)


def test_capacity_scaling_rejects_resolved_config_drift(tmp_path: Path) -> None:
    run = _run(tmp_path)
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["config"] = copy.deepcopy(report["config"])
    report["config"]["model"]["base_channels"] = 128
    write_json_report(path, report)
    with pytest.raises(ValueError, match="resolved training config differs"):
        _validate(run)
