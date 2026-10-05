from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.reporting import file_sha256
from scripts.audit_generation_system_goal import (
    aggregate_checks,
    authorization_boundary_evidence,
    baseline_fairness_evidence,
    checkpoint_reproducibility_evidence,
    sample_quality_evidence,
    stable_inference_evidence,
    training_scale_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _training_fixture(root: Path, *, steps: int = 2) -> None:
    for name in ("cofitok", "dense_identity"):
        run = root / name
        run.mkdir(parents=True, exist_ok=True)
        _write_json(
            run / "training_report.json",
            {
                "training_complete": True,
                "completed_steps": steps,
                "git": {
                    "revision": "a" * 40,
                    "branch": "scale/generative-system",
                    "dirty": False,
                },
            },
        )
        _write_json(
            run / "latest.json",
            {"checkpoint": f"checkpoint_step_{steps:08d}.pt", "step": steps},
        )
        (run / "train_metrics.jsonl").write_text(
            "\n".join(
                json.dumps({"step": step, "samples_seen": step * 64, "loss": 1.0 / step})
                for step in range(1, steps + 1)
            )
            + "\n",
            encoding="utf-8",
        )


def _quality_result(*, failed: list[str] | None = None) -> dict:
    failed = failed or []
    check_names = ["cofitok_precision_floor", "cofitok_recall_floor"]
    return {
        "status": "completed",
        "authorization_boundary": {
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "quality_bridge_execution_allowed": False,
            "release_authorization_allowed": False,
        },
        "quality_screen": {
            "status": "pass" if not failed else "hold",
            "failed_checks": failed,
            "non_authorizing": True,
            "thresholds": {"min_precision": 0.3, "min_recall": 0.3},
            "checks": [
                {
                    "name": name,
                    "observed": 0.0 if name in failed else 0.5,
                    "passed": name not in failed,
                    "threshold": 0.3,
                    "comparison": ">=" ,
                }
                for name in check_names
            ],
        },
    }


def _comparison() -> dict:
    return {
        "status": "hold",
        "terminal_status": "hold",
        "matched_training_rows": [
            {
                "method": "CoFiTok K=8",
                "comparison_tier": "matched_training_direct",
                "directly_comparable_to_cofitok": True,
            },
            {
                "method": "Dense identity",
                "comparison_tier": "matched_training_direct",
                "directly_comparable_to_cofitok": True,
            },
        ],
        "official_context_rows": [
            {
                "method": "D-AR",
                "comparison_tier": "official_pretrained_contextual",
                "directly_comparable_to_cofitok": False,
            }
        ],
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "cross_tier_numeric_ranking_allowed": False,
        },
    }


def _checkpoint_fixture(root: Path, *, step: int = 2) -> None:
    for name in ("cofitok", "dense_identity"):
        run = root / name
        run.mkdir(parents=True, exist_ok=True)
        checkpoint = run / f"checkpoint_step_{step:08d}.pt"
        checkpoint.write_bytes(f"checkpoint-{name}".encode("ascii"))
        sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
        _write_json(
            sidecar,
            {
                "schema_version": 1,
                "checkpoint": checkpoint.name,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_format_version": 1,
                "step": step,
            },
        )
        latest = {
            "checkpoint": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_format_version": 1,
            "step": step,
            "integrity_manifest": sidecar.name,
        }
        _write_json(run / "latest.json", latest)
        _write_json(
            run / "training_report.json",
            {
                "training_complete": True,
                "completed_steps": step,
                "latest_checkpoint": latest,
            },
        )


def test_aggregate_checks_is_fail_closed() -> None:
    report = aggregate_checks(
        [
            {"name": "pass", "status": "pass"},
            {"name": "hold", "status": "hold"},
            {"name": "missing", "status": "incomplete"},
        ]
    )
    assert report["complete"] is False
    assert report["status"] == "hold"
    assert report["held_checks"] == ["hold"]
    assert report["incomplete_checks"] == ["missing"]


def test_training_scale_requires_exact_target_and_accounting(tmp_path: Path) -> None:
    _training_fixture(tmp_path, steps=2)
    evidence = training_scale_evidence(
        tmp_path,
        target_steps=2,
        effective_batch=64,
        cofitok_run="cofitok",
        dense_run="dense_identity",
    )
    assert evidence["status"] == "pass"
    assert evidence["methods"]["cofitok"]["samples_seen"] == 128

    held = training_scale_evidence(
        tmp_path,
        target_steps=3,
        effective_batch=64,
        cofitok_run="cofitok",
        dense_run="dense_identity",
    )
    assert held["status"] == "incomplete"

    (tmp_path / "cofitok" / "train_metrics.jsonl").write_text(
        '{"step": 1, "samples_seen": 64}\n{"step": 1, "samples_seen": 64}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        training_scale_evidence(
            tmp_path,
            target_steps=2,
            effective_batch=64,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )


def test_checkpoint_reproducibility_rehashes_physical_payload(tmp_path: Path) -> None:
    _checkpoint_fixture(tmp_path, step=2)
    evidence = checkpoint_reproducibility_evidence(
        tmp_path,
        target_steps=2,
        cofitok_run="cofitok",
        dense_run="dense_identity",
    )
    assert evidence["status"] == "pass"
    assert evidence["methods"]["cofitok"]["physical_sha256_verified"] is True

    (tmp_path / "cofitok" / "checkpoint_step_00000002.pt").write_bytes(b"X" * 18)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checkpoint_reproducibility_evidence(
            tmp_path,
            target_steps=2,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )


def test_checkpoint_reproducibility_requires_latest_and_report_binding(tmp_path: Path) -> None:
    _checkpoint_fixture(tmp_path, step=2)
    latest_path = tmp_path / "cofitok" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["checkpoint_sha256"] = "0" * 64
    _write_json(latest_path, latest)
    with pytest.raises(ValueError, match="latest.json checkpoint_sha256"):
        checkpoint_reproducibility_evidence(
            tmp_path,
            target_steps=2,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )

    _checkpoint_fixture(tmp_path, step=2)
    report_path = tmp_path / "cofitok" / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("latest_checkpoint")
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="training_report.latest_checkpoint binding is missing"):
        checkpoint_reproducibility_evidence(
            tmp_path,
            target_steps=2,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )


def test_quality_hold_is_not_promoted_to_pass(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    source = tmp_path / "quality-source.json"
    _write_json(source, {"source": "fixture"})
    payload = _quality_result(failed=["cofitok_recall_floor"])
    payload["source_reports"] = {
        "fixture": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
        }
    }
    _write_json(path, payload)
    evidence = sample_quality_evidence(tmp_path, quality_result_path=path)
    assert evidence["status"] == "hold"
    assert evidence["failed_checks"] == ["cofitok_recall_floor"]


def test_quality_rejects_inconsistent_failed_check_list(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result(failed=["cofitok_recall_floor"])
    payload["quality_screen"]["failed_checks"] = []
    _write_json(path, payload)
    with pytest.raises(ValueError, match="failed_checks do not match check outcomes"):
        sample_quality_evidence(tmp_path, quality_result_path=path)


def test_quality_requires_explicit_non_authorizing_screen(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result()
    payload["quality_screen"].pop("non_authorizing")
    _write_json(path, payload)
    with pytest.raises(ValueError, match="not marked non-authorizing"):
        sample_quality_evidence(tmp_path, quality_result_path=path)


def test_quality_requires_rehashed_source_reports(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    _write_json(path, _quality_result())
    with pytest.raises(ValueError, match="source_reports are missing"):
        sample_quality_evidence(tmp_path, quality_result_path=path)


def test_quality_rejects_malformed_source_descriptor(tmp_path: Path) -> None:
    source = tmp_path / "quality-source.json"
    _write_json(source, {"source": "fixture"})
    payload = _quality_result()
    payload["source_reports"] = {
        "fixture": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
            "unexpected": True,
        }
    }
    path = tmp_path / "quality.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="descriptor fields differ"):
        sample_quality_evidence(tmp_path, quality_result_path=path)


def test_baseline_fairness_is_structural_and_keeps_context_separate(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    source = tmp_path / "comparison-source.json"
    _write_json(source, {"source": "fixture"})
    payload = _comparison()
    payload["source_reports"] = {
        "fixture": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
        }
    }
    _write_json(path, payload)
    evidence = baseline_fairness_evidence(tmp_path, comparison_path=path)
    assert evidence["status"] == "pass"
    assert evidence["direct_methods"] == ["CoFiTok K=8", "Dense identity"]
    assert evidence["direct_method_roles"] == {
        "cofitok": "CoFiTok K=8",
        "dense_identity": "Dense identity",
    }
    assert evidence["cross_tier_numeric_ranking_allowed"] is False
    assert evidence["scientific_status"] == "hold"


def test_baseline_fairness_requires_rehashed_source_reports(tmp_path: Path) -> None:
    path = tmp_path / "comparison.json"
    _write_json(path, _comparison())
    with pytest.raises(ValueError, match="source_reports are missing"):
        baseline_fairness_evidence(tmp_path, comparison_path=path)


def test_baseline_fairness_rejects_missing_tier_provenance(tmp_path: Path) -> None:
    payload = _comparison()
    payload["matched_training_rows"][0].pop("comparison_tier")
    path = tmp_path / "comparison.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="lacks CoFiTok/dense identity provenance"):
        baseline_fairness_evidence(tmp_path, comparison_path=path)


def _stable_sampling_fixture(
    root: Path,
    name: str,
    prefix: str,
    *,
    clip_x0: bool = True,
    sample_digest: str | None = None,
) -> None:
    """Write a small, schema-complete projection of the formal terminal run."""
    sample_digest = sample_digest or ("a" if name == "cofitok" else "b") * 64
    sample_root = root / name / "terminal_100k" / "samples_10000_ddim100_cfg15"
    sampling = {
        "actual_timesteps": select_sampling_timesteps(1_000, 100),
        "batch_size": 32,
        "cfg_batch_mode": "batched",
        "class_schedule": "balanced_modulo",
        "clip_x0": clip_x0,
        "eta": 0.0,
        "guidance_rescale": 0.0,
        "guidance_scale": 1.5,
        "image_shape": [3, 256, 256],
        "inference_api": {"name": "cofitok.generation.GenerationSession", "version": 1},
        "num_samples": 10_000,
        "num_train_timesteps": 1_000,
        "precision": "bf16",
        "prefix_budgets": [int(prefix)],
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "random_stream": {
            "batch_size_invariant": True,
            "prefix_budgets_share_stream": True,
            "resume_index_invariant": True,
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
        },
        "sample_set_digest": {
            "algorithm": "sha256",
            "framing": "filename_utf8_nul_file_bytes_nul",
        },
        "sample_steps": 100,
        "sampler": "ddim",
        "seed": 0,
        "start_index": 0,
    }
    checkpoint_sha256 = f"{name}-checkpoint"
    manifest = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": 100_000,
        "runtime_environment_sha256": "sampling-runtime",
        "sampling": sampling,
        "weights": "ema",
    }
    manifest_path = sample_root / "sampling_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = file_sha256(manifest_path)
    _write_json(
        root / name / "terminal_100k" / "sampling_preflight.json",
        {"checkpoint_sha256": checkpoint_sha256, "runtime_environment_sha256": "sampling-runtime"},
    )
    _write_json(
        sample_root / "sampling_report.json",
        {
            "status": "completed",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": 100_000,
            "runtime_environment_sha256": "sampling-runtime",
            "sampling": sampling,
            "sampling_manifest_sha256": manifest_sha256,
            "sample_sets": {prefix: {"count": 10_000, "sha256": sample_digest}},
            "weights": "ema",
        },
    )
    _write_json(
        sample_root / "sampling_progress.json",
        {
            "status": "completed",
            "completed_samples": 10_000,
            "total_samples": 10_000,
            "sampling_manifest_sha256": manifest_sha256,
            "sample_sets": {prefix: {"count": 10_000, "sha256": sample_digest}},
        },
    )
    _write_json(
        sample_root / "metrics" / "generation_metrics_report.json",
        {
            "runtime_environment_sha256": "evaluator-runtime",
            "sample_provenance": {
                "checkpoint_sha256": checkpoint_sha256,
                "runtime_environment_sha256": "sampling-runtime",
                "manifest_identity": {"sha256": manifest_sha256},
                "sampling_protocol_contract": {"stage": "scaling", "issues": [], "valid": True},
            },
        },
    )


def test_stable_inference_uses_nested_sampling_provenance(tmp_path: Path) -> None:
    for name, prefix in (("cofitok", "8"), ("dense_identity", "1")):
        _stable_sampling_fixture(tmp_path, name, prefix)
    evidence = stable_inference_evidence(
        tmp_path,
        expected_sample_count=10_000,
        expected_sample_steps=100,
        cofitok_run="cofitok",
        dense_run="dense_identity",
    )
    assert evidence["status"] == "pass"
    assert evidence["methods"]["cofitok"]["evaluator_runtime_environment_sha256"] == "evaluator-runtime"
    held = stable_inference_evidence(
        tmp_path,
        expected_sample_count=9_999,
        expected_sample_steps=100,
        cofitok_run="cofitok",
        dense_run="dense_identity",
    )
    assert held["status"] == "hold"


def test_stable_inference_rejects_protocol_drift(tmp_path: Path) -> None:
    for name, prefix in (("cofitok", "8"), ("dense_identity", "1")):
        _stable_sampling_fixture(tmp_path, name, prefix, clip_x0=False)
    with pytest.raises(ValueError, match="formal matched contract"):
        stable_inference_evidence(
            tmp_path,
            expected_sample_count=10_000,
            expected_sample_steps=100,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )


def test_stable_inference_rejects_malformed_sample_digest(tmp_path: Path) -> None:
    for name, prefix in (("cofitok", "8"), ("dense_identity", "1")):
        _stable_sampling_fixture(tmp_path, name, prefix, sample_digest="not-a-sha256")
    with pytest.raises(ValueError, match="sample-set digest is malformed"):
        stable_inference_evidence(
            tmp_path,
            expected_sample_count=10_000,
            expected_sample_steps=100,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )


def test_stable_inference_rejects_manifest_binding_drift(tmp_path: Path) -> None:
    for name, prefix in (("cofitok", "8"), ("dense_identity", "1")):
        _stable_sampling_fixture(tmp_path, name, prefix)
    manifest_path = tmp_path / "cofitok" / "terminal_100k" / "samples_10000_ddim100_cfg15" / "sampling_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weights"] = "model"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="sampling report manifest binding differs"):
        stable_inference_evidence(
            tmp_path,
            expected_sample_count=10_000,
            expected_sample_steps=100,
            cofitok_run="cofitok",
            dense_run="dense_identity",
        )


def test_authorization_boundary_rejects_any_true_launch_permission(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result()
    payload["authorization_boundary"]["full_300k_launch_allowed"] = True
    _write_json(path, payload)
    with pytest.raises(ValueError, match="full_300k_launch_allowed"):
        authorization_boundary_evidence(tmp_path, quality_result_path=path)


def test_authorization_boundary_requires_quality_boundary(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result()
    payload.pop("authorization_boundary")
    _write_json(path, payload)
    with pytest.raises(ValueError, match="authorization boundary is missing"):
        authorization_boundary_evidence(tmp_path, quality_result_path=path)


def test_authorization_boundary_requires_explicit_false_fields(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result()
    payload["authorization_boundary"].pop("full_training_launch_allowed")
    _write_json(path, payload)
    with pytest.raises(ValueError, match="full_training_launch_allowed is missing"):
        authorization_boundary_evidence(tmp_path, quality_result_path=path)


def test_authorization_boundary_checks_nested_scope_when_present(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result()
    payload["scope"] = {
        "training_launch_allowed": True,
        "sampling_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "gpu_execution_allowed": False,
        "process_signals_allowed": False,
        "promotion_allowed": False,
        "promotion_authorization_allowed": False,
        "release_allowed": False,
        "release_authorization_allowed": False,
    }
    _write_json(path, payload)
    with pytest.raises(ValueError, match="training_launch_allowed"):
        authorization_boundary_evidence(tmp_path, quality_result_path=path)


def test_authorization_boundary_ignores_descriptive_scope_without_permissions(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    payload = _quality_result()
    payload["scope"] = {"role": "cpu_only_evidence_replay"}
    _write_json(path, payload)
    evidence = authorization_boundary_evidence(tmp_path, quality_result_path=path)
    assert evidence["status"] == "pass"


def test_cli_empty_workspace_reports_incomplete_without_launching(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_generation_system_goal.py"),
            "--system-root",
            str(tmp_path / "empty"),
            "--output",
            str(output),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["status"] in {"hold", "failed"}
    assert report["read_only"] is True
    assert report["generation_advantage_proven"] is False
