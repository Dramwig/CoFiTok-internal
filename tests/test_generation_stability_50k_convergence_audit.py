from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256
from scripts.build_generation_stability_50k_convergence_audit import (
    ConvergenceAuditContract,
    build_convergence_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
BRANCH = "scale/generation-stability-50k-preflight"
DATASET_IDENTITY = "b" * 64
RUNTIME_IDENTITY = "c" * 64
AUDIT_GIT = {
    "revision": "f" * 40,
    "branch": "scale/generation-stability-50k-convergence-audit-v1",
    "tracked_dirty": False,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _config(*, dense: bool) -> dict:
    return {
        "data": {
            "batch_size": 4,
            "dataset": "imagenet_256_10pct",
        },
        "diffusion": {"prediction_target": "epsilon"},
        "optimization": {
            "gradient_accumulation_steps": 2,
            "log_interval": 50,
            "learning_rate": 1e-4,
            "min_learning_rate": 1e-5,
        },
        "runtime": {
            "evaluation_interval": 100,
            "steps": 500,
        },
        "model": {
            "base_channels": 32,
            "token_count": 1 if dense else 8,
            "token_channels": 3 if dense else 8,
            "predictor_use_feedback": not dense,
            "synthesis_mode": "dense_identity" if dense else "fixed_basis",
        },
        "loss": {
            "epsilon_weight": 1.0,
            "rollout_consistency_weight": 0.1,
            "rollout_consistency_start_step": 0,
            "rollout_consistency_warmup_steps": 200,
            "rollout_consistency_timestep_delta": 10,
            "rollout_consistency_unroll_steps": 2,
            "rollout_consistency_batch_fraction": 0.125,
            "rollout_consistency_clip_x0": True,
            "rollout_consistency_mode": "clipped_x0",
            "ema_teacher_consistency_weight": 0.25,
            "ema_teacher_consistency_start_step": 300,
            "ema_teacher_consistency_warmup_steps": 100,
            "ema_teacher_consistency_batch_fraction": 0.0625,
            "prefix_weight": 0.0,
        },
    }


def _manifest(*, dense: bool, run_dir: Path) -> dict:
    manifest = {
        "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
        "parameter_count": 999 if dense else 1_000,
        "runtime_environment_sha256": RUNTIME_IDENTITY,
        "dataset_provenance": {
            "status": "pass",
            "formal": True,
            "dataset": "imagenet_256_10pct",
            "identity_sha256": DATASET_IDENTITY,
        },
        "config": _config(dense=dense),
        "resume": None,
        "metrics_resume_reconciliation": None,
    }
    if not dense:
        manifest["resume"] = (run_dir / "checkpoint_step_00000365.pt").as_posix()
        manifest["metrics_resume_reconciliation"] = {
            "schema_version": 1,
            "status": "unchanged",
            "resume_step": 365,
            "retained_rows": 8,
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        }
    return manifest


def _rows(*, dense: bool) -> list[dict]:
    steps = [1, *range(50, 501, 50)]
    if not dense:
        steps.append(366)
    rows = []
    for step in sorted(steps):
        if step <= 400:
            epsilon = 0.08 / (1.0 + step / 100.0) + 0.018
        else:
            epsilon = 0.031 + (step - 450) * 0.000001
        epsilon += 0.00003 if dense else 0.0
        rollout_scale = min(step / 200.0, 1.0)
        teacher_scale = 0.0 if step < 300 else min((step - 300) / 100.0, 1.0)
        row = {
            "step": step,
            "samples_seen": step * 8,
            "epsilon": epsilon,
            "total": epsilon + (0.02 if not dense else 0.003),
            "rollout_consistency": 0.04 - min(step, 500) * 0.00002,
            "rollout_consistency_scale": rollout_scale,
            "ema_teacher_consistency": (
                0.0 if teacher_scale == 0.0 else 0.001 / (1.0 + step / 100.0)
            ),
            "ema_teacher_consistency_scale": teacher_scale,
            "grad_norm": 0.5 / (1.0 + step / 100.0),
            "learning_rate": max(1e-5, 1e-4 * (1.0 - step / 550.0)),
        }
        if step % 100 == 0:
            event = step // 100 - 1
            row.update(
                {
                    "validation_epsilon_mse": (
                        0.04 - event * 0.002 + (0.00002 if dense else 0.0)
                    ),
                    "validation_event_index": event,
                    "validation_batch_index": event,
                    "validation_num_images": 8,
                    "validation_noise_seed": 102030,
                }
            )
        rows.append(row)
    return rows


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_run(
    scaling_root: Path,
    *,
    label: str,
    dense: bool,
) -> tuple[dict, str]:
    run_name = (
        "dense_rollout_x0_u2_ema_teacher"
        if dense
        else "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
    )
    run_dir = scaling_root / run_name
    run_dir.mkdir(parents=True)
    manifest = _manifest(dense=dense, run_dir=run_dir)
    _write_json(run_dir / "run_manifest.json", manifest)
    rows = _rows(dense=dense)
    _write_metrics(run_dir / "train_metrics.jsonl", rows)
    checkpoint = run_dir / "checkpoint_step_00000500.pt"
    checkpoint.write_bytes((b"dense" if dense else b"cofitok") * 17)
    checkpoint_sha = file_sha256(checkpoint)
    sidecar = {
        "schema_version": 1,
        "step": 500,
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_format_version": 1,
        "dataset_identity_sha256": DATASET_IDENTITY,
        "runtime_environment_sha256": RUNTIME_IDENTITY,
        "git_revision": REVISION,
        "git_branch": BRANCH,
        "git_dirty": False,
    }
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".integrity.json")
    _write_json(sidecar_path, sidecar)
    latest = {**sidecar, "integrity_manifest": sidecar_path.name}
    _write_json(run_dir / "latest.json", latest)
    report = {
        "training_complete": True,
        "completed_steps": 500,
        "target_steps": 500,
        "stop_requested": False,
        "git": manifest["git"],
        "config": manifest["config"],
        "dataset_provenance": manifest["dataset_provenance"],
        "runtime_environment_sha256": RUNTIME_IDENTITY,
        "final_metrics": rows[-1],
        "latest_checkpoint": latest,
        "elapsed_seconds": 123.5 if dense else 125.0,
        "peak_vram_bytes": 1_000_000,
    }
    _write_json(run_dir / "training_report.json", report)
    return manifest, checkpoint_sha


def _fixture(tmp_path: Path) -> dict:
    scaling_root = tmp_path / "scaling"
    full_root = tmp_path / "full"
    (scaling_root / "reports/frozen_posteval_supplemental").mkdir(
        parents=True
    )
    (full_root / "reports").mkdir(parents=True)
    cofitok_manifest, cofitok_checkpoint_sha = _write_run(
        scaling_root, label="cofitok", dense=False
    )
    dense_manifest, dense_checkpoint_sha = _write_run(
        scaling_root, label="dense_identity", dense=True
    )
    _write_json(
        scaling_root / "pair_monitor.json",
        {"schema_version": 2, "status": "pass", "stage": "complete"},
    )
    pair_summary_path = scaling_root / "reports/pair_summary.json"
    _write_json(
        pair_summary_path,
        {
            "schema_version": 1,
            "status": "completed",
            "stage": "stability_matched_50k",
            "completed_steps_per_method": 500,
            "images_seen_per_method": 4_000,
            "formal_300k_authorization_allowed": False,
            "formal_ema_sampling_gate_required": True,
        },
    )
    _write_json(
        scaling_root / "reports/dense_recovery_status.json",
        {
            "schema_version": 1,
            "status": "pass",
            "full_training_launch_allowed": False,
        },
    )
    _write_json(
        scaling_root / "reports/posteval_waiter.json",
        {
            "schema_version": 1,
            "status": "pass",
            "detail": "formal_ema_postevaluation_completed",
            "child_exit_code": 0,
        },
    )
    _write_json(
        scaling_root
        / "reports/frozen_posteval_supplemental/supplemental_waiter.json",
        {
            "schema_version": 1,
            "status": "failed",
            "detail": "RuntimeError: stability 50K post-evaluation status is stale",
            "child_exit_code": None,
            "supplemental_non_authorizing": True,
            "full_training_launch_allowed": False,
        },
    )
    _write_json(
        full_root / "reports/readiness_waiter.json",
        {
            "schema_version": 1,
            "status": "failed",
            "detail": "ValueError: scaling gate did not authorize promote_to_full_imagenet256",
            "child_exit_code": None,
            "full_training_launch_allowed": False,
        },
    )
    gate_path = scaling_root / "reports/promotion_gate.json"
    _write_json(
        gate_path,
        {
            "schema_version": 2,
            "status": "fail",
            "decision": "hold",
            "gates": [
                {"name": "source_complete", "passed": True},
                {"name": "absolute_fid_quality", "passed": False},
            ],
            "summary": {
                "cofitok_fid": 138.0,
                "dense_fid": 151.0,
                "cofitok_precision": 0.75,
                "dense_precision": 0.78,
                "cofitok_recall": 0.0087,
                "dense_recall": 0.0098,
            },
        },
    )
    gate_identity = file_identity(gate_path)
    support_path = (
        scaling_root
        / "reports/frozen_existing_sample_support_audit/support_audit.json"
    )
    support_statistics = {
        method: {
            "decoded_pixel_duplicate_count": 0,
            "dhash_duplicate_count": 0,
        }
        for method in ("real_matched_subset", "cofitok", "dense_identity")
    }
    _write_json(
        support_path,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "generation_frozen_existing_sample_support_audit",
            "claim_boundary": {
                "causal_attribution_allowed": False,
                "full_training_launch_allowed": False,
                "gpu_execution_authorized": False,
                "new_sampling_performed": False,
                "new_training_performed": False,
            },
            "interpretation_policy": {
                "thresholded_gate": False,
                "formal_distribution_support_remains_authoritative": {
                    "cofitok_precision": 0.75,
                    "cofitok_recall": 0.0087,
                    "dense_precision": 0.78,
                    "dense_recall": 0.0098,
                },
            },
            "sources": {"promotion_gate": gate_identity},
            "support_statistics": support_statistics,
            "contrasts_to_real": {
                "cofitok": {"lowres_entropy_effective_rank_ratio_to_real": 0.8},
                "dense_identity": {
                    "lowres_entropy_effective_rank_ratio_to_real": 0.6
                },
            },
            "matched_method_pair": {
                "pairing": "same global index, class id, and initial-noise seed",
                "paired_lowres_rms": {"mean": 0.067},
                "paired_lowres_rms_over_cofitok_same_class_rms": 0.30,
                "paired_lowres_rms_over_dense_same_class_rms": 0.25,
            },
        },
    )
    contract = ConvergenceAuditContract(
        training_revision=REVISION,
        training_branch=BRANCH,
        dataset_identity_sha256=DATASET_IDENTITY,
        runtime_environment_sha256=RUNTIME_IDENTITY,
        target_steps=500,
        effective_batch=8,
        promotion_gate_sha256=file_sha256(gate_path),
        support_audit_sha256=file_sha256(support_path),
        pair_summary_sha256=file_sha256(pair_summary_path),
        cofitok_checkpoint_sha256=cofitok_checkpoint_sha,
        dense_checkpoint_sha256=dense_checkpoint_sha,
    )
    return {
        "kwargs": {
            "scaling_root": scaling_root,
            "full_root": full_root,
            "project_root": PROJECT_ROOT,
            "audit_git": AUDIT_GIT,
            "contract": contract,
        },
        "scaling_root": scaling_root,
        "full_root": full_root,
        "contract": contract,
        "cofitok_manifest": cofitok_manifest,
        "dense_manifest": dense_manifest,
    }


def test_convergence_audit_is_deterministic_and_non_authorizing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    report = build_convergence_audit(**fixture["kwargs"])
    replay = build_convergence_audit(**fixture["kwargs"])

    assert report == replay
    assert report["status"] == "pass"
    assert report["training"]["cofitok"]["row_integrity"]["row_count"] == 12
    assert report["training"]["cofitok"]["row_integrity"][
        "resume_extra_steps"
    ] == [366]
    assert report["training"]["dense_identity"]["row_integrity"][
        "row_count"
    ] == 11
    assert report["paired_validation"]["event_count"] == 5
    assert report["paired_validation"]["temporal_convergence_claim_allowed"] is False
    assert report["evidence_synthesis"]["decision"] == (
        "sampling_recovery_first_then_full_data_scale_if_not_recovered"
    )
    assert report["static_next_action_sources"]["sampling_recovery"][
        "sample_count_per_case"
    ] == 1_000
    assert report["claim_boundary"]["sampling_execution_allowed"] is False
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["claim_boundary"]["full_300k_launch_allowed"] is False
    assert report["training"]["cofitok"]["terminal"]["checkpoint"][
        "physical_checkpoint_rehashed_by_this_audit"
    ] is False


def test_convergence_audit_rejects_nonfinite_metric(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = (
        fixture["scaling_root"]
        / "dense_rollout_x0_u2_ema_teacher/train_metrics.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["epsilon"] = float("nan")
    _write_metrics(path, rows)

    with pytest.raises(ValueError, match="must be finite"):
        build_convergence_audit(**fixture["kwargs"])


def test_convergence_audit_rejects_unbound_resume_row(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = (
        fixture["scaling_root"]
        / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher/train_metrics.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    resume_row = next(row for row in rows if row["step"] == 366)
    resume_row["step"] = 367
    resume_row["samples_seen"] = 367 * 8
    rows.sort(key=lambda row: row["step"])
    _write_metrics(path, rows)

    with pytest.raises(ValueError, match="exact logging/resume schedule"):
        build_convergence_audit(**fixture["kwargs"])


def test_convergence_audit_rejects_frozen_gate_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["scaling_root"] / "reports/promotion_gate.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    gate["summary"]["cofitok_fid"] = 1.0
    _write_json(path, gate)

    with pytest.raises(ValueError, match="promotion gate SHA256"):
        build_convergence_audit(**fixture["kwargs"])


def test_convergence_audit_rejects_support_authorization_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    path = (
        fixture["scaling_root"]
        / "reports/frozen_existing_sample_support_audit/support_audit.json"
    )
    support = json.loads(path.read_text(encoding="utf-8"))
    support["claim_boundary"]["full_training_launch_allowed"] = True
    _write_json(path, support)
    fixture["kwargs"]["contract"] = replace(
        fixture["contract"], support_audit_sha256=file_sha256(path)
    )

    with pytest.raises(ValueError, match="claim boundary"):
        build_convergence_audit(**fixture["kwargs"])


def test_convergence_audit_runbook_is_cpu_only_and_fail_closed() -> None:
    source = (
        PROJECT_ROOT
        / "artifacts/runbooks/generation_stability_frozen_50k_convergence_audit.sh"
    ).read_text(encoding="utf-8")

    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "EXPECTED_AUDIT_REVISION=${EXPECTED_AUDIT_REVISION:?" in source
    assert "git status --porcelain" in source
    assert '[[ ! -e "$AUDIT_DIR" ]]' in source
    assert "nice -n 15 ionice -c3" in source
    assert "sampling_execution_allowed" in source
    assert "full_training_launch_allowed" in source
    assert "full_300k_launch_allowed" in source
    assert "nvidia-smi" not in source
