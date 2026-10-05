from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from scripts.build_generation_milestone_report import (
    build_report,
    expected_source_report_suffixes,
    source_report_identity,
    validate_milestone_report,
    verify_milestone_source_reports,
)


ROOT = Path(__file__).resolve().parents[1]


def _source_reports(*, source_profile: str = "full") -> dict:
    suffixes = expected_source_report_suffixes(
        50_000,
        source_profile=source_profile,
    )
    return {
        name: {
            "path": f"/root/outputs/{suffixes[name]}",
            "bytes": 100 + index,
            "sha256": str(index + 1) * 64,
        }
        for index, name in enumerate(
            (
                "cofitok_generation",
                "dense_generation",
                "cofitok_checkpoint_eval",
                "dense_checkpoint_eval",
            )
        )
    }


def _generation(*, sha: str, budget: int, fid: float, sample_steps: int = 50) -> dict:
    return {
        "status": "completed",
        "counts": {"generated_image_count": 2048},
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 4.0,
        },
        "sample_provenance": {
            "checkpoint": f"/checkpoints/{sha[:4]}.pt",
            "checkpoint_sha256": sha,
            "checkpoint_integrity_manifest": f"/checkpoints/{sha[:4]}.pt.integrity.json",
            "checkpoint_step": 50_000,
            "weights": "ema",
            "selected_prefix_budget": budget,
            "sample_set_sha256": ("c" if budget > 1 else "d") * 64,
            "sampling": {
                "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_samples": 2048,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": sample_steps,
                "num_train_timesteps": 1000,
                "actual_timesteps": select_sampling_timesteps(1000, sample_steps),
                "prefix_budgets": [budget],
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "seed": 0,
                "precision": "bf16",
                "image_shape": [3, 256, 256],
                "class_schedule": "balanced_modulo",
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
            },
        },
    }


def _checkpoint_eval(*, sha: str, rank: int, zero: float = 0.0) -> dict:
    return {
        "status": "completed",
        "checkpoint_sha256": sha,
        "checkpoint_integrity_manifest": f"/checkpoints/{sha[:4]}.pt.integrity.json",
        "checkpoint_step": 50_000,
        "weights": "ema",
        "metrics": {
            "orders": {
                "ordered": {
                    "endpoint_clean_mse": 0.1,
                    "prefix_path_mse_auc": 0.2,
                }
            },
            "ordered_rank_by_path_auc": rank,
            "order_count": 6 if rank > 0 else 1,
            "zero_token_max_abs": zero,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
        },
    }


def test_milestone_report_binds_matched_checkpoint_and_sampling_protocol() -> None:
    report = build_report(
        cofitok_generation=_generation(sha="a" * 64, budget=8, fid=24.0),
        dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
        cofitok_checkpoint_eval=_checkpoint_eval(sha="a" * 64, rank=1),
        dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
        source_reports=_source_reports(),
        milestone_step=50_000,
        expected_samples=2048,
    )

    assert report["status"] == "completed"
    assert report["role"] == "training_quality_trend_only"
    assert report["claim_policy"]["formal_generation_claim_allowed"] is False
    assert report["matched_comparison"]["fid_relative_change"] == pytest.approx(0.2)
    assert report["quality_alert"] is False


def test_milestone_report_binds_quality_bridge_source_profile() -> None:
    report = build_report(
        cofitok_generation=_generation(sha="a" * 64, budget=8, fid=24.0),
        dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
        cofitok_checkpoint_eval=_checkpoint_eval(sha="a" * 64, rank=1),
        dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
        source_reports=_source_reports(source_profile="quality_bridge"),
        milestone_step=50_000,
        expected_samples=2048,
        source_profile="quality_bridge",
    )

    evidence, warnings = validate_milestone_report(
        report,
        expected_step=50_000,
        expected_source_profile="quality_bridge",
    )
    assert warnings == []
    assert report["source_profile"] == "quality_bridge"
    assert evidence["source_profile"] == "quality_bridge"
    assert "stability_full_data_100k_base128_quality_bridge_v1" in report[
        "source_reports"
    ]["cofitok_generation"]["path"]


def test_milestone_report_surfaces_quality_alerts_without_becoming_formal_gate() -> None:
    report = build_report(
        cofitok_generation=_generation(sha="a" * 64, budget=8, fid=30.0),
        dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
        cofitok_checkpoint_eval=_checkpoint_eval(sha="a" * 64, rank=2, zero=0.1),
        dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
        source_reports=_source_reports(),
        milestone_step=50_000,
        expected_samples=2048,
    )

    assert report["quality_alert"] is True
    assert "cofitok_fid_more_than_25pct_above_dense" in report["quality_alerts"]
    assert "cofitok_ordered_prefix_not_rank1" in report["quality_alerts"]
    assert "cofitok_zero_token_contract_failed" in report["quality_alerts"]


def test_milestone_report_rejects_protocol_drift() -> None:
    with pytest.raises(ValueError, match="sampling protocol is invalid"):
        build_report(
            cofitok_generation=_generation(sha="a" * 64, budget=8, fid=20.0),
            dense_generation=_generation(
                sha="b" * 64,
                budget=1,
                fid=20.0,
                sample_steps=40,
            ),
            cofitok_checkpoint_eval=_checkpoint_eval(sha="a" * 64, rank=1),
            dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
            source_reports=_source_reports(),
            milestone_step=50_000,
            expected_samples=2048,
        )


def test_milestone_report_rejects_cross_checkpoint_evaluation() -> None:
    with pytest.raises(ValueError, match="different checkpoint bytes"):
        build_report(
            cofitok_generation=_generation(sha="a" * 64, budget=8, fid=20.0),
            dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
            cofitok_checkpoint_eval=_checkpoint_eval(sha="e" * 64, rank=1),
            dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
            source_reports=_source_reports(),
            milestone_step=50_000,
            expected_samples=2048,
        )


def test_milestone_report_rejects_cross_integrity_manifest_evaluation() -> None:
    checkpoint = _checkpoint_eval(sha="a" * 64, rank=1)
    checkpoint["checkpoint_integrity_manifest"] = "/checkpoints/other.pt.integrity.json"
    with pytest.raises(ValueError, match="different integrity manifests"):
        build_report(
            cofitok_generation=_generation(sha="a" * 64, budget=8, fid=20.0),
            dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
            cofitok_checkpoint_eval=checkpoint,
            dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
            source_reports=_source_reports(),
            milestone_step=50_000,
            expected_samples=2048,
        )


def test_milestone_report_revalidates_content_addressed_sources(tmp_path) -> None:
    source_reports = {}
    for name, suffix in expected_source_report_suffixes(50_000).items():
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        source_reports[name] = source_report_identity(path)
    report = build_report(
        cofitok_generation=_generation(sha="a" * 64, budget=8, fid=20.0),
        dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
        cofitok_checkpoint_eval=_checkpoint_eval(sha="a" * 64, rank=1),
        dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
        source_reports=source_reports,
        milestone_step=50_000,
        expected_samples=2048,
    )

    verification = verify_milestone_source_reports(report)
    evidence, warnings = validate_milestone_report(
        report,
        expected_step=50_000,
        source_verification=verification,
    )
    assert warnings == []
    assert evidence["source_report_sha256"] == {
        name: identity["sha256"] for name, identity in source_reports.items()
    }

    report_path = tmp_path / "milestone.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    environment = dict(os.environ)
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(ROOT / "src"), inherited_pythonpath) if path
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_generation_milestone_report.py"),
            "--report",
            str(report_path),
            "--expected-step",
            "50000",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    (tmp_path / expected_source_report_suffixes(50_000)["cofitok_generation"]).write_text(
        "changed", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after binding"):
        verify_milestone_source_reports(report)
