from __future__ import annotations

from scripts.build_generation_gate_report import build_report


def _training(parameters: int, token_count: int) -> dict:
    return {
        "training_complete": True,
        "completed_steps": 50_000,
        "target_steps": 50_000,
        "parameter_count": parameters,
        "git": {"dirty": False, "revision": "abc"},
        "config": {
            "data": {"dataset": "imagenet_256_10pct"},
            "diffusion": {"schedule": "cosine"},
            "runtime": {"steps": 50_000},
            "optimization": {"batch": 64},
            "model": {"token_count": token_count},
        },
    }


def _generation(fid: float, token_count: int, sha: str) -> dict:
    return {
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "counts": {"generated_image_count": 10_000},
        "parameters": {"batch_size": 64, "seed": 2027},
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 18.0,
            "inception_score_std": 0.2,
            "precision": 0.6,
            "recall": 0.4,
        },
        "sample_provenance": {
            "checkpoint": "/checkpoint.pt",
            "checkpoint_sha256": sha,
            "checkpoint_step": 50_000,
            "weights": "ema",
            "selected_prefix_budget": token_count,
            "sampling": {
                "num_samples": 10_000,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": 100,
                "prefix_budgets": [token_count],
                "guidance_scale": 1.5,
                "seed": 0,
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                },
            },
        },
    }


def _checkpoint(endpoint: float, sha: str, rank: int = 1) -> dict:
    return {
        "checkpoint_sha256": sha,
        "checkpoint_step": 50_000,
        "metrics": {
            "orders": {"ordered": {"endpoint_clean_mse": endpoint}},
            "ordered_rank_by_path_auc": rank,
            "order_count": 18,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
        }
    }


def test_generation_gate_passes_matched_quality_and_mechanism() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.102, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_to_full_imagenet256"
    assert all(gate["passed"] for gate in report["gates"])


def test_generation_gate_holds_on_fid_regression() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(25.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert not next(gate for gate in report["gates"] if gate["name"] == "fid_within_tolerance")[
        "passed"
    ]


def test_generation_gate_holds_on_unpaired_sampling_streams() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    dense["sample_provenance"]["sampling"]["random_stream"]["batch_size_invariant"] = False
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_mechanism_eval_uses_another_checkpoint() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "c" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "checkpoint_evaluation_provenance"
    )
    assert gate["passed"] is False


def test_full_generation_gate_uses_ready_decision_and_absolute_fid() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(19.0, 8, "a" * 64),
        dense_generation=_generation(18.5, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "large_scale_generation_ready"
    assert report["stage"] == "full"


def test_full_generation_gate_holds_above_absolute_fid_limit() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "absolute_fid_quality")
    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert gate["passed"] is False


def test_generation_gate_holds_when_quality_metrics_are_incomplete() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    del cofitok["metrics"]["recall"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "generation_metrics_complete")
    assert gate["passed"] is False
