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
from cofitok.generation.capacity_qualification_training import (
    CAPACITY_REFERENCE_STAGE,
    validate_capacity_qualification_partial_training,
)
from cofitok.reporting import file_sha256, write_json_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_capacity_qualification_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
)
REFERENCE_CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_capacity_reference_"
    "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
)
REVISION = "a" * 40
BRANCH = "scale/generation-capacity-source-compatible-v1"
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
        "total": 1.0,
        "epsilon": 0.9,
        "grad_norm": 2.0,
        "learning_rate": 0.0001,
        "elapsed_seconds": float(step),
        "cumulative_elapsed_seconds": float(step),
        "samples_seen": step * 64,
    }
    if validation_index is not None:
        row.update(
            {
                "validation_epsilon_mse": 0.8,
                "validation_event_index": validation_index,
                "validation_batch_index": validation_index,
                "validation_num_images": 64,
                "validation_noise_seed": 2027 + validation_index,
            }
        )
    return row


def _partial_run(
    tmp_path: Path,
    *,
    config_path: Path = CONFIG,
    parameters: int = PARAMETERS,
    resume_step: int | None = None,
) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    config = load_config(config_path)
    resolved = config_to_dict(config)
    resolved["data"]["batch_size"] = MICRO_BATCH
    resolved["optimization"]["gradient_accumulation_steps"] = ACCUMULATION
    environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "fixture GPU"},
    }
    environment_sha = runtime_environment_sha256(environment)
    dataset = _dataset()

    checkpoint = run / "checkpoint_step_00010000.pt"
    checkpoint.write_bytes(b"capacity qualification checkpoint")
    integrity = {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_format_version": 1,
        "step": 10_000,
        "git_revision": REVISION,
        "git_branch": BRANCH,
        "git_dirty": False,
        "runtime_environment_sha256": environment_sha,
        "dataset_identity_sha256": dataset["identity_sha256"],
    }
    write_json_report(
        checkpoint.with_name(f"{checkpoint.name}.integrity.json"),
        integrity,
    )
    latest = {
        **integrity,
        "integrity_manifest": f"{checkpoint.name}.integrity.json",
    }
    write_json_report(run / "latest.json", latest)

    resume_path: Path | None = None
    reconciliation: dict[str, object] | None = None
    if resume_step is not None:
        resume_path = run / f"checkpoint_step_{resume_step:08d}.pt"
        resume_path.write_bytes(b"capacity qualification resume checkpoint")
        resume_integrity = {
            **integrity,
            "checkpoint": resume_path.name,
            "checkpoint_bytes": resume_path.stat().st_size,
            "checkpoint_sha256": file_sha256(resume_path),
            "step": resume_step,
        }
        write_json_report(
            resume_path.with_name(f"{resume_path.name}.integrity.json"),
            resume_integrity,
        )
        reconciliation = {
            "schema_version": 1,
            "status": "unchanged",
            "resume_step": resume_step,
            "metrics": (run / "train_metrics.jsonl").resolve().as_posix(),
            "retained_rows": 2,
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        }

    rows = [_metrics(1)]
    for index, step in enumerate(range(1_000, 10_001, 1_000)):
        rows.append(_metrics(step, validation_index=index))
    (run / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = {
        "config": resolved,
        "config_path": str(config_path.resolve()),
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
        "parameter_count": parameters,
        "trainable_parameter_count": parameters,
        "resume": resume_path.resolve().as_posix() if resume_path is not None else None,
        "resume_revision_transition": None,
        "metrics_resume_reconciliation": reconciliation,
        "horizon_extension": None,
    }
    write_json_report(run / "run_manifest.json", manifest)
    write_json_report(
        run / "training_report.json",
        {
            **manifest,
            "completed_steps": 10_000,
            "target_steps": 100_000,
            "training_complete": False,
            "stop_requested": False,
            "stop_signal": None,
            "final_metrics": rows[-1],
            "segment_elapsed_seconds": 10_000.0,
            "elapsed_seconds": 10_000.0,
            "peak_vram_bytes": 80_000_000_000,
            "latest_checkpoint": latest,
        },
    )
    return run


def _validate(
    run: Path,
    *,
    config_path: Path = CONFIG,
    parameters: int = PARAMETERS,
    stage: str = "stability_capacity_qualification",
    base_channels: int = 256,
    allow_exact_resume: bool = False,
) -> dict[str, object]:
    return validate_capacity_qualification_partial_training(
        report_path=run / "training_report.json",
        config_path=config_path,
        expected_revision=REVISION,
        expected_branch=BRANCH,
        expected_parameter_count=parameters,
        expected_micro_batch_size=MICRO_BATCH,
        expected_gradient_accumulation_steps=ACCUMULATION,
        expected_stage=stage,
        expected_base_channels=base_channels,
        allow_exact_resume=allow_exact_resume,
    )


def test_exact_10k_partial_training_is_valid_and_non_authorizing(
    tmp_path: Path,
) -> None:
    report = _validate(_partial_run(tmp_path))

    assert report["status"] == "pass"
    assert report["completed_steps"] == 10_000
    assert report["configured_steps"] == 100_000
    assert report["training_complete"] is False
    assert report["intentional_partial_stop"] is True
    assert report["fresh_initialization_verified"] is True
    assert report["images_seen"] == 640_000
    assert report["validation_event_count"] == 10
    assert report["checkpoint"]["sha256"] == file_sha256(
        Path(report["checkpoint"]["path"])
    )
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_exact_base128_reference_stop_is_valid_and_separately_labeled(
    tmp_path: Path,
) -> None:
    parameters = 62_837_576
    run = _partial_run(
        tmp_path,
        config_path=REFERENCE_CONFIG,
        parameters=parameters,
    )

    report = _validate(
        run,
        config_path=REFERENCE_CONFIG,
        parameters=parameters,
        stage=CAPACITY_REFERENCE_STAGE,
        base_channels=128,
    )

    assert report["status"] == "pass"
    assert report["stage"] == CAPACITY_REFERENCE_STAGE
    assert report["parameter_count"] == parameters


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_complete", True),
        ("completed_steps", 9_999),
        ("target_steps", 10_000),
        ("stop_requested", True),
        ("stop_signal", 15),
    ),
)
def test_partial_training_rejects_wrong_stop_state(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    run = _partial_run(tmp_path)
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    write_json_report(path, report)

    with pytest.raises(ValueError, match="exact clean stop"):
        _validate(run)


def test_partial_training_rejects_legacy_metric_field_names(tmp_path: Path) -> None:
    run = _partial_run(tmp_path)
    path = run / "train_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["total_loss"] = rows[-1].pop("total")
    rows[-1]["epsilon_loss"] = rows[-1].pop("epsilon")
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path = run / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["final_metrics"] = rows[-1]
    write_json_report(report_path, report)

    with pytest.raises(ValueError, match="final_metrics.total"):
        _validate(run)


def test_partial_training_rejects_metric_history_gap(tmp_path: Path) -> None:
    run = _partial_run(tmp_path)
    path = run / "train_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.pop(-2)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation events differ"):
        _validate(run)


def test_partial_training_rejects_checkpoint_tamper(tmp_path: Path) -> None:
    run = _partial_run(tmp_path)
    (run / "checkpoint_step_00010000.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|SHA256 mismatch"):
        _validate(run)


def test_partial_training_rejects_checkpoint_after_stop(tmp_path: Path) -> None:
    run = _partial_run(tmp_path)
    (run / "checkpoint_step_00015000.pt").write_bytes(b"later")

    with pytest.raises(ValueError, match="after the stop step"):
        _validate(run)


def test_partial_training_rejects_resolved_runtime_drift(tmp_path: Path) -> None:
    run = _partial_run(tmp_path)
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["config"] = copy.deepcopy(report["config"])
    report["config"]["optimization"]["gradient_accumulation_steps"] = 32
    write_json_report(path, report)

    with pytest.raises(ValueError, match="resolved training config differs"):
        _validate(run)


def test_partial_training_rejects_resume_state(tmp_path: Path) -> None:
    run = _partial_run(tmp_path, resume_step=5_000)

    with pytest.raises(ValueError, match="requires explicit validation"):
        _validate(run)


def test_partial_training_accepts_exact_same_run_resume(tmp_path: Path) -> None:
    run = _partial_run(tmp_path, resume_step=5_000)

    report = _validate(run, allow_exact_resume=True)

    assert report["fresh_initialization_verified"] is True
    assert report["exact_resume"]["resumed"] is True
    assert report["exact_resume"]["resume_step"] == 5_000


def test_partial_training_rejects_cross_revision_resume(tmp_path: Path) -> None:
    run = _partial_run(tmp_path, resume_step=5_000)
    path = run / "training_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["resume_revision_transition"] = {"source": "other"}
    write_json_report(path, report)
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resume_revision_transition"] = {"source": "other"}
    write_json_report(manifest_path, manifest)

    with pytest.raises(ValueError, match="cannot change revision"):
        _validate(run, allow_exact_resume=True)


def test_partial_training_rejects_symlinked_report(
    tmp_path: Path,
) -> None:
    run = _partial_run(tmp_path)
    link = tmp_path / "training_report_link.json"
    try:
        link.symlink_to(run / "training_report.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="must not contain a symlink"):
        validate_capacity_qualification_partial_training(
            report_path=link,
            config_path=CONFIG,
            expected_revision=REVISION,
            expected_branch=BRANCH,
            expected_parameter_count=PARAMETERS,
            expected_micro_batch_size=MICRO_BATCH,
            expected_gradient_accumulation_steps=ACCUMULATION,
        )


def test_capacity_qualification_training_cli_imports() -> None:
    import scripts.validate_generation_capacity_qualification_training  # noqa: F401
