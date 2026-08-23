from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from cofitok.generation.cross_protocol_reconciliation import (
    CLAIM_BOUNDARY,
    build_reconciliation_comparison,
    validate_decision_route,
    validate_quality_hold,
)
from scripts import reconcile_generation_100k_cross_protocol as reconciliation


ROOT = Path(__file__).resolve().parents[1]


def _metrics(fid: float, inception: float = 5.0) -> dict[str, float]:
    return {
        "frechet_inception_distance": fid,
        "inception_score_mean": inception,
        "inception_score_std": 0.2,
    }


def test_comparison_attributes_ranking_reversal_to_sampler_steps() -> None:
    report = build_reconciliation_comparison(
        cofitok_ddim50_2048=_metrics(220.0),
        dense_ddim50_2048=_metrics(135.0),
        cofitok_ddim100_2048=_metrics(128.0),
        dense_ddim100_2048=_metrics(136.0),
        cofitok_ddim100_10000=_metrics(115.0),
        dense_ddim100_10000=_metrics(123.0),
    )

    assert report["protocol_rows"]["ddim50_2048"]["lower_fid_method"] == (
        "dense_identity"
    )
    assert report["protocol_rows"]["ddim100_2048"]["lower_fid_method"] == (
        "cofitok"
    )
    assert report["protocol_rows"]["ddim100_10000"]["lower_fid_method"] == (
        "cofitok"
    )
    assert report["ranking_reversal_explanation"] == (
        "sampler_step_effect_dominates_observed_ranking_reversal"
    )
    assert report["generation_advantage_proven"] is False


def test_comparison_attributes_ranking_reversal_to_sample_count() -> None:
    report = build_reconciliation_comparison(
        cofitok_ddim50_2048=_metrics(150.0),
        dense_ddim50_2048=_metrics(140.0),
        cofitok_ddim100_2048=_metrics(145.0),
        dense_ddim100_2048=_metrics(135.0),
        cofitok_ddim100_10000=_metrics(115.0),
        dense_ddim100_10000=_metrics(123.0),
    )

    assert report["ranking_reversal_explanation"] == (
        "sample_count_effect_dominates_observed_ranking_reversal"
    )


def test_authoritative_route_and_hold_boundaries_are_required() -> None:
    decision = {
        "status": "completed",
        "recommended_next_stage": {
            "id": "reconcile_100k_cross_protocol_evidence",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
        "authorization_boundary": {
            "recommended_stage_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
    }
    quality = {
        "status": "completed",
        "quality_screen": {"status": "hold"},
        "authorization_boundary": {
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "report_is_promotion_gate": False,
        },
    }

    validate_decision_route(decision)
    validate_quality_hold(quality)
    decision["recommended_next_stage"]["gpu_execution_allowed"] = True
    with pytest.raises(ValueError, match="decision route differs"):
        validate_decision_route(decision)
    quality["quality_screen"]["status"] = "pass"
    with pytest.raises(ValueError, match="hold boundary differs"):
        validate_quality_hold(quality)


def _write_png(path: Path, value: int) -> None:
    Image.new("RGB", (4, 4), color=(value, value, value)).save(path)


def test_hardlink_subset_uses_existing_png_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconciliation, "INDEX_COUNT", 4)
    source = tmp_path / "source"
    source.mkdir()
    images = []
    for index in range(reconciliation.INDEX_COUNT):
        path = source / f"{index:06d}.png"
        _write_png(path, index % 256)
        images.append(path)

    subset = reconciliation.prepare_hardlink_subset(
        images,
        target_dir=tmp_path / "subset",
    )

    assert subset["new_sampling_performed"] is False
    assert subset["source_count"] == reconciliation.INDEX_COUNT
    assert subset["source_sample_set_sha256"] == subset["target_sample_set_sha256"]
    assert os.path.samefile(images[2], subset["images"][2])


def test_paired_pixel_difference_reports_nonidentical_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconciliation, "INDEX_COUNT", 4)
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_images = []
    right_images = []
    for index in range(reconciliation.INDEX_COUNT):
        left_path = left / f"{index:06d}.png"
        right_path = right / f"{index:06d}.png"
        _write_png(left_path, index % 256)
        _write_png(right_path, (index + 1) % 256)
        left_images.append(left_path)
        right_images.append(right_path)

    report = reconciliation.paired_pixel_difference(left_images, right_images)

    assert report["pair_count"] == reconciliation.INDEX_COUNT
    assert report["different_pair_count"] == reconciliation.INDEX_COUNT
    assert report["mean_squared_error_unit_range"] > 0.0
    assert report["peak_signal_to_noise_ratio_db"] > 0.0


def test_subset_metric_report_is_cpu_only_and_reusable(tmp_path: Path) -> None:
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    source_file = tmp_path / "sampling_report.json"
    source_file.write_text('{"status":"completed"}\n', encoding="utf-8")
    source_identity = reconciliation.file_identity(source_file)
    evaluator_git = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "analysis/test",
        "tracked_dirty": False,
        "path": tmp_path.as_posix(),
    }
    environment = {"schema_version": 1, "device": {"type": "cpu"}}
    real_set = {
        "status": "physically_verified",
        "root": tmp_path.as_posix(),
        "image_count": 50_000,
        "digest_schema": "cofitok_image_tree_sha256_v1",
        "sha256": "c" * 64,
        "images": [],
    }
    subset = {
        "status": "verified",
        "link_mode": "hardlink_existing_png_view",
        "new_sampling_performed": False,
        "source_count": reconciliation.INDEX_COUNT,
        "target_count": reconciliation.INDEX_COUNT,
        "source_sample_set_sha256": "d" * 64,
        "target_sample_set_sha256": "d" * 64,
        "target_dir": subset_dir.as_posix(),
        "images": [],
    }
    calls = []

    def fake_calculator(**kwargs):
        calls.append(kwargs)
        return _metrics(123.0), reconciliation.TORCH_FIDELITY_VERSION

    output = tmp_path / "metric.json"
    report, identity = reconciliation.evaluate_subset_metrics(
        method="cofitok",
        subset=subset,
        source_sampling_identity=source_identity,
        real_set=real_set,
        evaluator_git=evaluator_git,
        evaluator_environment=environment,
        evaluator_environment_sha256="e" * 64,
        batch_size=8,
        cache_root=tmp_path,
        real_cache_name="real-cache",
        output=output,
        metric_calculator=fake_calculator,
    )

    assert identity == reconciliation.file_identity(output)
    assert calls[0]["cuda"] is False
    assert calls[0]["prc"] is False
    assert report["metrics"]["frechet_inception_distance"] == 123.0

    replay, replay_identity = reconciliation.evaluate_subset_metrics(
        method="cofitok",
        subset=subset,
        source_sampling_identity=source_identity,
        real_set=real_set,
        evaluator_git=evaluator_git,
        evaluator_environment=environment,
        evaluator_environment_sha256="e" * 64,
        batch_size=8,
        cache_root=tmp_path,
        real_cache_name="real-cache",
        output=output,
        metric_calculator=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("completed subset metric should be reused")
        ),
    )
    assert replay == report
    assert replay_identity == identity


def test_claim_boundary_is_permanently_non_authorizing() -> None:
    assert CLAIM_BOUNDARY["generation_advantage_proven"] is False
    assert CLAIM_BOUNDARY["training_launch_allowed"] is False
    assert CLAIM_BOUNDARY["sampling_launch_allowed"] is False
    assert CLAIM_BOUNDARY["gpu_execution_allowed"] is False
    assert CLAIM_BOUNDARY["full_300k_launch_allowed"] is False
    assert CLAIM_BOUNDARY["promotion_authorization_allowed"] is False
    assert CLAIM_BOUNDARY["release_authorization_allowed"] is False
    assert CLAIM_BOUNDARY["process_signals_allowed"] is False


def test_entrypoint_imports_under_runbook_pythonpath() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, "scripts/reconcile_generation_100k_cross_protocol.py", "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "existing PNGs" in result.stdout


def test_runbook_has_no_training_sampling_or_gpu_entrypoint() -> None:
    runbook = (
        ROOT
        / "artifacts/runbooks/generation_100k_cross_protocol_reconciliation_v1.sh"
    ).read_text(encoding="utf-8")
    assert "reconcile_generation_100k_cross_protocol.py" in runbook
    assert "CUDA_VISIBLE_DEVICES=\"\"" in runbook
    assert "train_generation.py" not in runbook
    assert "sample_generation.py" not in runbook
    assert "infer_generation.py" not in runbook
    assert "kill " not in runbook
    assert "full_300k" not in runbook
    assert "--resume" in runbook
