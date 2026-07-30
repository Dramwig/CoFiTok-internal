from copy import deepcopy

from cofitok.generation.stability_scaling import build_stability_scaling_decision


def _qualification(*, seed: int, images: int, status: str = "pass") -> dict:
    gates = {
        "tail_two_energy": {"passed": True},
        "single_token_energy": {"passed": True},
        "ordered_rank": {"passed": True},
        "endpoint_regression": {"passed": True},
        "validation_regression": {"passed": True},
        "predicted_x0_high_frequency": {"passed": True},
        "reconstruction_regression": {"passed": status == "pass"},
        "zero_token": {"passed": True},
        "shuffle_mismatch": {"passed": True},
    }
    return {
        "schema_version": 2,
        "status": status,
        "protocol": {
            "weights": "model",
            "checkpoint_step": 5000,
            "checkpoint_evaluated_images": 256,
            "checkpoint_timestep": 500,
            "high_frequency_timesteps": [595, 394, 192, 91],
            "rollout": {
                "num_images": images,
                "batch_size": 8 if images >= 64 else 2,
                "sample_steps": 100,
                "guidance_scale": 1.5,
                "seed": seed,
            },
        },
        "identity": {
            "git_revision": "abc123",
            "cofitok_checkpoint_sha256": "cofitok",
            "dense_checkpoint_sha256": "dense",
        },
        "metrics": {
            "tail_two_energy_ratio": 0.57,
            "max_single_token_energy_ratio": 0.29,
            "endpoint_ratio": 1.01,
            "validation_ratio": 1.01,
            "peak_predicted_x0_high_frequency_ratio": 1.1,
            "reconstruction_ratio": 1.03,
            "cofitok_reconstruction_amplification": 1.08,
            "dense_reconstruction_amplification": 1.07,
        },
        "gates": gates,
        "pair_contract": {"valid": True},
    }


def test_scaling_decision_accepts_two_robust_seeds_after_screening_failure() -> None:
    report = build_stability_scaling_decision(
        screening_report=_qualification(seed=2029, images=8, status="fail"),
        robust_reports=[
            _qualification(seed=2029, images=64),
            _qualification(seed=2039, images=64),
        ],
    )

    assert report["status"] == "pass"
    assert report["decision"] == "authorize_fresh_matched_50k_preparation"
    assert report["screening"]["failed_gates"] == ["reconstruction_regression"]


def test_scaling_decision_rejects_duplicate_robust_seeds() -> None:
    report = build_stability_scaling_decision(
        screening_report=_qualification(seed=2029, images=8, status="fail"),
        robust_reports=[
            _qualification(seed=2029, images=64),
            _qualification(seed=2029, images=64),
        ],
    )

    assert report["status"] == "fail"
    assert "robust rollout seeds must be unique" in report["issues"]


def test_scaling_decision_rejects_additional_screening_failure() -> None:
    screening = _qualification(seed=2029, images=8, status="fail")
    screening = deepcopy(screening)
    screening["gates"]["ordered_rank"]["passed"] = False

    report = build_stability_scaling_decision(
        screening_report=screening,
        robust_reports=[
            _qualification(seed=2029, images=64),
            _qualification(seed=2039, images=64),
        ],
    )

    assert report["status"] == "fail"
    assert any("beyond reconstruction" in issue for issue in report["issues"])


def test_scaling_decision_rejects_small_or_failed_robust_report() -> None:
    failed = _qualification(seed=2039, images=32, status="fail")
    report = build_stability_scaling_decision(
        screening_report=_qualification(seed=2029, images=8, status="fail"),
        robust_reports=[
            _qualification(seed=2029, images=64),
            failed,
        ],
    )

    assert report["status"] == "fail"
    assert any("did not pass" in issue for issue in report["issues"])
    assert any("requires at least 64" in issue for issue in report["issues"])
