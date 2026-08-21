from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.training.checkpointing import checkpoint_integrity_path
from scripts.build_generation_resume_compute_adjustment import (
    build_adjustment_report,
)
from scripts.wait_generation_quality_bridge_runtime_compute_fairness import (
    build_terminal_audit,
    file_identity,
    inspect_existing_adjustment_coverage,
    require_current_adjustment_coverage,
    resolve_training_cost,
    validate_static_contract,
    write_json_atomic,
    write_once_or_verify,
)


ROOT = Path(__file__).resolve().parents[1]
COFITOK_CONFIG = ROOT / (
    "configs/generation/"
    "imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_"
    "ema_teacher_k8_100k.json"
)
DENSE_CONFIG = ROOT / (
    "configs/generation/"
    "imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_"
    "dense_100k.json"
)
REVISION = "a" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
STEPS = 100_000


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dataset_provenance() -> dict:
    spec = FORMAL_GENERATION_DATASETS["imagenet_256"]
    report = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": spec.dataset,
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


def _resolved_config(path: Path) -> dict:
    config = config_to_dict(load_config(path))
    config["data"]["batch_size"] = 64
    config["optimization"]["gradient_accumulation_steps"] = 1
    return config


def _training_report(
    run_dir: Path,
    *,
    config_path: Path,
    parameter_count: int,
    elapsed_seconds: float,
    peak_vram_bytes: int,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint_step_00100000.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_integrity_path(checkpoint).write_text("{}", encoding="utf-8")
    provenance = _dataset_provenance()
    latest = {
        "checkpoint": checkpoint.name,
        "step": STEPS,
        "dataset_identity_sha256": provenance["identity_sha256"],
    }
    _write(run_dir / "latest.json", latest)
    report = {
        "training_complete": True,
        "completed_steps": STEPS,
        "target_steps": STEPS,
        "parameter_count": parameter_count,
        "output_dir": run_dir.resolve().as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "final_metrics": {"samples_seen": STEPS * 64},
        "git": {
            "dirty": False,
            "revision": REVISION,
            "branch": BRANCH,
        },
        "latest_checkpoint": latest,
        "config": _resolved_config(config_path),
        "dataset_provenance": provenance,
        "metrics_resume_reconciliation": {
            "schema_version": 1,
            "status": "unchanged",
            "resume_step": STEPS,
            "metrics": (run_dir / "train_metrics.jsonl").resolve().as_posix(),
            "retained_rows": 1,
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        },
    }
    _write(run_dir / "training_report.json", report)
    return report


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _add_recovery(run_dir: Path, report: dict) -> Path:
    canonical_rows = [
        {
            "step": 100,
            "samples_seen": 6400,
            "elapsed_seconds": 1000.0,
            "cumulative_elapsed_seconds": 1000.0,
        },
        {
            "step": 110,
            "samples_seen": 7040,
            "elapsed_seconds": 10.0,
            "cumulative_elapsed_seconds": 1010.0,
        },
        {
            "step": 120,
            "samples_seen": 7680,
            "elapsed_seconds": 20.0,
            "cumulative_elapsed_seconds": 1020.0,
        },
        {
            "step": 130,
            "samples_seen": 8320,
            "elapsed_seconds": 30.0,
            "cumulative_elapsed_seconds": 1030.0,
        },
        {
            "step": STEPS,
            "samples_seen": STEPS * 64,
            "elapsed_seconds": 1000.0,
            "cumulative_elapsed_seconds": 2000.0,
        },
    ]
    canonical = run_dir / "train_metrics.jsonl"
    _write_jsonl(canonical, canonical_rows)
    orphan_rows = [
        {
            "step": 110,
            "samples_seen": 7040,
            "elapsed_seconds": 1012.0,
            "cumulative_elapsed_seconds": 1012.0,
        },
        {
            "step": 120,
            "samples_seen": 7680,
            "elapsed_seconds": 1022.0,
            "cumulative_elapsed_seconds": 1022.0,
        },
    ]
    orphan_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in orphan_rows
    ).encode("utf-8")
    orphan_sha = hashlib.sha256(orphan_payload).hexdigest()
    orphan = run_dir / (
        "train_metrics_orphaned_at_resume_00000100_"
        f"{orphan_sha[:12]}.jsonl"
    )
    orphan.write_bytes(orphan_payload)
    report["metrics_resume_reconciliation"] = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": 100,
        "metrics": canonical.resolve().as_posix(),
        "retained_rows": 1,
        "orphaned_rows": 2,
        "orphan_archive": orphan.resolve().as_posix(),
        "orphan_sha256": orphan_sha,
    }
    _write(run_dir / "training_report.json", report)
    return orphan


def _static_contract_files(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    cofitok_config = tmp_path / "cofitok.json"
    dense_config = tmp_path / "dense.json"
    _write(cofitok_config, {"method": "cofitok"})
    _write(dense_config, {"method": "dense"})
    config_validation = tmp_path / "config_validation.json"
    _write(
        config_validation,
        {
            "status": "pass",
            "mismatches": [],
            "relative_parameter_gap": 0.00014924064826915946,
            "cofitok": {"parameter_count": 62_834_083},
            "dense": {"parameter_count": 62_824_707},
        },
    )
    runtime_selection = tmp_path / "runtime_selection.json"
    _write(
        runtime_selection,
        {
            "selected": {
                "micro_batch_size": 64,
                "gradient_accumulation_steps": 1,
                "effective_batch_size": 64,
            }
        },
    )
    runbook = tmp_path / "runbook.sh"
    runbook.write_text(
        "\n".join(
            (
                "read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION",
                '--micro-batch-size "$SELECTED_MICRO_BATCH"',
                '--gradient-accumulation-steps "$SELECTED_ACCUMULATION"',
                '--expected-micro-batch-size "$SELECTED_MICRO_BATCH"',
                '--expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION"',
                'train_to_milestone "$COFITOK_CONFIG" "$COFITOK_RUN"',
                'train_to_milestone "$DENSE_CONFIG" "$DENSE_RUN"',
                'require_complete "$COFITOK_RUN/training_report.json"',
                'require_complete "$DENSE_RUN/training_report.json"',
                "scripts/validate_generation_training_pair.py",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch_receipt.json"
    _write(
        launch,
        {
            "status": "pass",
            "git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "selection": {"steps": STEPS, "dataset": "imagenet_256"},
            "output_root": tmp_path.resolve().as_posix(),
            "training_run_dirs": [
                (tmp_path / "cofitok_run").resolve().as_posix(),
                (tmp_path / "dense_run").resolve().as_posix(),
            ],
            "runtime_selection": {
                "micro_batch_size": 64,
                "gradient_accumulation_steps": 1,
                "effective_batch_size": 64,
                "runtime_environment_sha256": "b" * 64,
            },
            "source_reports": {
                "config_validation": file_identity(config_validation),
                "runtime_selection": file_identity(runtime_selection),
                "cofitok_config": file_identity(cofitok_config),
                "dense_config": file_identity(dense_config),
            },
        },
    )
    paths = {
        "launch_receipt": launch,
        "config_validation": config_validation,
        "runtime_selection": runtime_selection,
        "active_runbook": runbook,
        "cofitok_config": cofitok_config,
        "dense_config": dense_config,
    }
    arguments = {
        **paths,
        "expected_launch_receipt_sha256": file_identity(launch)["sha256"],
        "expected_config_validation_sha256": file_identity(config_validation)[
            "sha256"
        ],
        "expected_runtime_selection_sha256": file_identity(runtime_selection)[
            "sha256"
        ],
        "expected_active_runbook_sha256": file_identity(runbook)["sha256"],
        "expected_training_revision": REVISION,
        "expected_training_branch": BRANCH,
        "expected_steps": STEPS,
    }
    return arguments, paths


def test_static_contract_binds_one_runtime_to_both_methods(tmp_path: Path) -> None:
    arguments, _ = _static_contract_files(tmp_path)

    contract = validate_static_contract(**arguments)

    assert contract["status"] == "verified"
    assert contract["runtime"] == {
        "micro_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 64,
        "runtime_environment_sha256": "b" * 64,
    }
    assert contract["runbook_contract"][
        "both_completion_reports_validate_selected_runtime"
    ] is True


def test_static_contract_rejects_runtime_selection_drift(tmp_path: Path) -> None:
    arguments, paths = _static_contract_files(tmp_path)
    runtime = json.loads(paths["runtime_selection"].read_text(encoding="utf-8"))
    runtime["selected"]["micro_batch_size"] = 32
    _write(paths["runtime_selection"], runtime)
    arguments["expected_runtime_selection_sha256"] = file_identity(
        paths["runtime_selection"]
    )["sha256"]

    with pytest.raises(ValueError, match="runtime selection and launch receipt"):
        validate_static_contract(**arguments)


def test_terminal_audit_includes_recovery_compute_and_observed_parity(
    tmp_path: Path,
) -> None:
    cofitok_run = tmp_path / "cofitok"
    dense_run = tmp_path / "dense"
    cofitok = _training_report(
        cofitok_run,
        config_path=COFITOK_CONFIG,
        parameter_count=62_834_083,
        elapsed_seconds=10_000.0,
        peak_vram_bytes=80_000,
    )
    _training_report(
        dense_run,
        config_path=DENSE_CONFIG,
        parameter_count=62_824_707,
        elapsed_seconds=9_000.0,
        peak_vram_bytes=70_000,
    )
    orphan = _add_recovery(cofitok_run, cofitok)
    adjustment = build_adjustment_report(
        canonical_metrics=cofitok_run / "train_metrics.jsonl",
        orphan_metrics=[orphan],
        effective_batch=64,
        continuity_end_step=130,
    )
    cofitok_adjustment = tmp_path / "cofitok_adjustment/resume_compute_adjustment.json"
    write_json_atomic(cofitok_adjustment, adjustment)
    dense_adjustment = tmp_path / "dense_adjustment/resume_compute_adjustment.json"
    static_contract = {
        "status": "verified",
        "sources": {},
        "runtime": {
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 64,
            "runtime_environment_sha256": "b" * 64,
        },
        "parameters": {
            "cofitok": 62_834_083,
            "dense_identity": 62_824_707,
            "relative_gap": (62_834_083 - 62_824_707) / 62_824_707,
        },
    }

    report = build_terminal_audit(
        static_contract=static_contract,
        auditor_checkout={"revision": "c" * 40, "tracked_dirty": False},
        waiter_source={"path": "/tmp/waiter.py", "bytes": 1, "sha256": "d" * 64},
        cofitok_config=COFITOK_CONFIG,
        dense_config=DENSE_CONFIG,
        cofitok_run_dir=cofitok_run,
        dense_run_dir=dense_run,
        cofitok_adjustment_path=cofitok_adjustment,
        dense_adjustment_path=dense_adjustment,
        expected_steps=STEPS,
        expected_training_revision=REVISION,
        expected_training_branch=BRANCH,
    )

    assert report["status"] == "pass"
    assert report["observed_runtime_parity"]["status"] == "proven"
    cofitok_cost = report["methods"]["cofitok"]["training_cost"]
    dense_cost = report["methods"]["dense_identity"]["training_cost"]
    assert cofitok_cost["resume_compute_adjustment"]["seconds"] == 22.0
    assert cofitok_cost["elapsed_seconds"] == 10_022.0
    assert dense_cost["resume_compute_adjustment"]["required"] is False
    assert dense_cost["elapsed_seconds"] == 9_000.0
    assert report["claim_boundary"]["training_speed_advantage_predeclared"] is False


def test_resolve_training_cost_builds_missing_dense_adjustment(
    tmp_path: Path,
) -> None:
    dense_run = tmp_path / "dense"
    dense = _training_report(
        dense_run,
        config_path=DENSE_CONFIG,
        parameter_count=62_824_707,
        elapsed_seconds=9_000.0,
        peak_vram_bytes=70_000,
    )
    _add_recovery(dense_run, dense)
    adjustment = tmp_path / "reports/dense/resume_compute_adjustment.json"

    evidence = resolve_training_cost(
        method="dense_identity",
        report=dense,
        run_dir=dense_run,
        adjustment_path=adjustment,
        expected_steps=STEPS,
    )

    assert adjustment.is_file()
    assert evidence["training_cost"]["resume_compute_adjustment"]["seconds"] == 22.0
    assert evidence["resume_compute_adjustment"]["sha256"] == file_identity(
        adjustment
    )["sha256"]


def test_pending_adjustment_coverage_allows_absence_until_terminal(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cofitok"
    report = _training_report(
        run_dir,
        config_path=COFITOK_CONFIG,
        parameter_count=62_834_083,
        elapsed_seconds=10_000.0,
        peak_vram_bytes=80_000,
    )
    _add_recovery(run_dir, report)

    state = inspect_existing_adjustment_coverage(
        method="cofitok",
        run_dir=run_dir,
        adjustment_path=tmp_path / "reports/resume_compute_adjustment.json",
    )

    assert state["status"] == "absent"
    assert state["valid"] is True
    assert state["terminal_adjustment_required"] is True
    assert state["discovered_orphan_archive_count"] == 1
    assert state["covered_orphan_archive_count"] == 0


def test_pending_adjustment_coverage_accepts_exact_physical_orphan_set(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cofitok"
    report = _training_report(
        run_dir,
        config_path=COFITOK_CONFIG,
        parameter_count=62_834_083,
        elapsed_seconds=10_000.0,
        peak_vram_bytes=80_000,
    )
    orphan = _add_recovery(run_dir, report)
    adjustment_path = tmp_path / "reports/resume_compute_adjustment.json"
    write_json_atomic(
        adjustment_path,
        build_adjustment_report(
            canonical_metrics=run_dir / "train_metrics.jsonl",
            orphan_metrics=[orphan],
            effective_batch=64,
            continuity_end_step=130,
        ),
    )

    state = inspect_existing_adjustment_coverage(
        method="cofitok",
        run_dir=run_dir,
        adjustment_path=adjustment_path,
    )

    assert state["status"] == "verified"
    assert state["valid"] is True
    assert state["discovered_orphan_archive_count"] == 1
    assert state["covered_orphan_archive_count"] == 1
    assert state["issues"] == []


def test_pending_adjustment_coverage_detects_new_orphan_before_terminal(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "cofitok"
    report = _training_report(
        run_dir,
        config_path=COFITOK_CONFIG,
        parameter_count=62_834_083,
        elapsed_seconds=10_000.0,
        peak_vram_bytes=80_000,
    )
    orphan = _add_recovery(run_dir, report)
    adjustment_path = tmp_path / "reports/resume_compute_adjustment.json"
    write_json_atomic(
        adjustment_path,
        build_adjustment_report(
            canonical_metrics=run_dir / "train_metrics.jsonl",
            orphan_metrics=[orphan],
            effective_batch=64,
            continuity_end_step=130,
        ),
    )
    second_payload = b'{"step": 210, "samples_seen": 13440}\n'
    second_sha = hashlib.sha256(second_payload).hexdigest()
    second_orphan = run_dir / (
        "train_metrics_orphaned_at_resume_00000200_"
        f"{second_sha[:12]}.jsonl"
    )
    second_orphan.write_bytes(second_payload)

    state = inspect_existing_adjustment_coverage(
        method="cofitok",
        run_dir=run_dir,
        adjustment_path=adjustment_path,
    )

    assert state["status"] == "stale"
    assert state["valid"] is False
    assert state["discovered_orphan_archive_count"] == 2
    assert state["covered_orphan_archive_count"] == 1
    assert state["missing_orphan_archives"] == [
        second_orphan.resolve().as_posix()
    ]
    assert state["issues"] == [
        "existing adjustment does not cover the current physical orphan archive set"
    ]
    with pytest.raises(
        ValueError,
        match="existing resume-compute adjustment is stale for: cofitok",
    ):
        require_current_adjustment_coverage({"cofitok": state})


def test_terminal_audit_rejects_observed_dense_runtime_drift(
    tmp_path: Path,
) -> None:
    cofitok_run = tmp_path / "cofitok"
    dense_run = tmp_path / "dense"
    _training_report(
        cofitok_run,
        config_path=COFITOK_CONFIG,
        parameter_count=62_834_083,
        elapsed_seconds=10_000.0,
        peak_vram_bytes=80_000,
    )
    dense = _training_report(
        dense_run,
        config_path=DENSE_CONFIG,
        parameter_count=62_824_707,
        elapsed_seconds=9_000.0,
        peak_vram_bytes=70_000,
    )
    dense["config"]["data"]["batch_size"] = 32
    dense["config"]["optimization"]["gradient_accumulation_steps"] = 2
    _write(dense_run / "training_report.json", dense)

    with pytest.raises(ValueError, match="resolved config differs"):
        build_terminal_audit(
            static_contract={
                "status": "verified",
                "sources": {},
                "runtime": {
                    "micro_batch_size": 64,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": 64,
                    "runtime_environment_sha256": "b" * 64,
                },
                "parameters": {
                    "cofitok": 62_834_083,
                    "dense_identity": 62_824_707,
                    "relative_gap": (62_834_083 - 62_824_707) / 62_824_707,
                },
            },
            auditor_checkout={"revision": "c" * 40, "tracked_dirty": False},
            waiter_source={
                "path": "/tmp/waiter.py",
                "bytes": 1,
                "sha256": "d" * 64,
            },
            cofitok_config=COFITOK_CONFIG,
            dense_config=DENSE_CONFIG,
            cofitok_run_dir=cofitok_run,
            dense_run_dir=dense_run,
            cofitok_adjustment_path=(
                tmp_path / "cofitok/resume_compute_adjustment.json"
            ),
            dense_adjustment_path=(
                tmp_path / "dense/resume_compute_adjustment.json"
            ),
            expected_steps=STEPS,
            expected_training_revision=REVISION,
            expected_training_branch=BRANCH,
        )


def test_existing_terminal_audit_must_be_equivalent(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    write_once_or_verify(output, {"status": "pass"}, label="audit")
    write_once_or_verify(output, {"status": "pass"}, label="audit")

    with pytest.raises(ValueError, match="not byte-equivalent"):
        write_once_or_verify(output, {"status": "failed"}, label="audit")
