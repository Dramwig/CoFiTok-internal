from __future__ import annotations

from scripts.build_generation_gate_report import build_report


def _training(parameters: int) -> dict:
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
        },
    }


def _generation(fid: float) -> dict:
    return {
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "counts": {"generated_image_count": 10_000},
        "parameters": {"batch_size": 64, "seed": 2027},
        "metrics": {"frechet_inception_distance": fid},
    }


def _checkpoint(endpoint: float, rank: int = 1) -> dict:
    return {
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
        cofitok_training=_training(100_500),
        dense_training=_training(100_000),
        cofitok_generation=_generation(20.5),
        dense_generation=_generation(20.0),
        cofitok_checkpoint=_checkpoint(0.102),
        dense_checkpoint=_checkpoint(0.1),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_to_full_imagenet256"
    assert all(gate["passed"] for gate in report["gates"])


def test_generation_gate_holds_on_fid_regression() -> None:
    report = build_report(
        cofitok_training=_training(100_500),
        dense_training=_training(100_000),
        cofitok_generation=_generation(25.0),
        dense_generation=_generation(20.0),
        cofitok_checkpoint=_checkpoint(0.1),
        dense_checkpoint=_checkpoint(0.1),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert not next(gate for gate in report["gates"] if gate["name"] == "fid_within_tolerance")[
        "passed"
    ]
