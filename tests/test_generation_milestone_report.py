from __future__ import annotations

import pytest

from scripts.build_generation_milestone_report import build_report


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
            "checkpoint_step": 50_000,
            "weights": "ema",
            "selected_prefix_budget": budget,
            "sample_set_sha256": ("c" if budget > 1 else "d") * 64,
            "sampling": {
                "num_samples": 2048,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": sample_steps,
                "prefix_budgets": [budget],
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "seed": 0,
                "precision": "bf16",
                "image_shape": [3, 256, 256],
            },
        },
    }


def _checkpoint_eval(*, sha: str, rank: int, zero: float = 0.0) -> dict:
    return {
        "status": "completed",
        "checkpoint_sha256": sha,
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
        milestone_step=50_000,
        expected_samples=2048,
    )

    assert report["status"] == "completed"
    assert report["role"] == "training_quality_trend_only"
    assert report["claim_policy"]["formal_generation_claim_allowed"] is False
    assert report["matched_comparison"]["fid_relative_change"] == pytest.approx(0.2)
    assert report["quality_alert"] is False


def test_milestone_report_surfaces_quality_alerts_without_becoming_formal_gate() -> None:
    report = build_report(
        cofitok_generation=_generation(sha="a" * 64, budget=8, fid=30.0),
        dense_generation=_generation(sha="b" * 64, budget=1, fid=20.0),
        cofitok_checkpoint_eval=_checkpoint_eval(sha="a" * 64, rank=2, zero=0.1),
        dense_checkpoint_eval=_checkpoint_eval(sha="b" * 64, rank=1),
        milestone_step=50_000,
        expected_samples=2048,
    )

    assert report["quality_alert"] is True
    assert "cofitok_fid_more_than_25pct_above_dense" in report["quality_alerts"]
    assert "cofitok_ordered_prefix_not_rank1" in report["quality_alerts"]
    assert "cofitok_zero_token_contract_failed" in report["quality_alerts"]


def test_milestone_report_rejects_protocol_drift() -> None:
    with pytest.raises(ValueError, match="protocols are not matched"):
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
            milestone_step=50_000,
            expected_samples=2048,
        )
