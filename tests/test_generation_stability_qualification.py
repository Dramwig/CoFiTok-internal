from copy import deepcopy

import pytest

from cofitok.generation.stability_qualification import build_stability_qualification


def _training(validation: float) -> dict:
    return {
        "training_complete": True,
        "completed_steps": 1000,
        "target_steps": 1000,
        "git": {"revision": "abc123"},
        "final_metrics": {"validation_epsilon_mse": validation},
    }


def _checkpoint(*, cofitok: bool, endpoint: float) -> dict:
    metrics = {
        "orders": {"ordered": {"endpoint_clean_mse": endpoint}},
        "component_energy_ratio": [0.2, 0.25, 0.25, 0.3],
        "ordered_rank_by_path_auc": 1,
        "zero_token_max_abs": 0.0,
        "shuffled_to_ordered_endpoint_ratio": 10.0,
    }
    if not cofitok:
        metrics["component_energy_ratio"] = [1.0]
    return {
        "status": "completed",
        "weights": "model",
        "checkpoint_sha256": "cofitok" if cofitok else "dense",
        "checkpoint_step": 1000,
        "metrics": metrics,
    }


def _rollout(*, cofitok: bool, high_frequency: float, final_mse: float) -> dict:
    return {
        "status": "completed",
        "weights": "model",
        "checkpoint_sha256": "cofitok" if cofitok else "dense",
        "checkpoint_step": 1000,
        "free_sampling_rollout": {
            "steps": [
                {
                    "timestep": timestep,
                    "predicted_x0_high_frequency_ratio": high_frequency,
                }
                for timestep in (595, 394, 192, 91)
            ]
        },
        "reconstruction_rollout": {
            "summary": {
                "final_clipped_x0_mse": final_mse,
                "final_to_best_x0_mse_amplification": 1.1,
            }
        },
    }


def _inputs() -> dict:
    return {
        "cofitok_training": _training(1.01),
        "dense_training": _training(1.0),
        "cofitok_checkpoint": _checkpoint(cofitok=True, endpoint=1.02),
        "dense_checkpoint": _checkpoint(cofitok=False, endpoint=1.0),
        "cofitok_rollout": _rollout(
            cofitok=True,
            high_frequency=0.3,
            final_mse=1.03,
        ),
        "dense_rollout": _rollout(
            cofitok=False,
            high_frequency=0.25,
            final_mse=1.0,
        ),
    }


def test_stability_qualification_passes_bounded_candidate() -> None:
    report = build_stability_qualification(**_inputs())

    assert report["status"] == "pass"
    assert report["metrics"]["tail_two_energy_ratio"] == pytest.approx(0.55)
    assert report["metrics"]["peak_predicted_x0_high_frequency_ratio"] == pytest.approx(
        1.2
    )
    assert all(gate["passed"] for gate in report["gates"].values())


def test_stability_qualification_fails_high_frequency_gate() -> None:
    inputs = _inputs()
    inputs["cofitok_rollout"] = _rollout(
        cofitok=True,
        high_frequency=0.5,
        final_mse=1.03,
    )

    report = build_stability_qualification(**inputs)

    assert report["status"] == "fail"
    assert report["gates"]["predicted_x0_high_frequency"]["passed"] is False


def test_stability_qualification_rejects_mismatched_checkpoint() -> None:
    inputs = _inputs()
    inputs["cofitok_rollout"] = deepcopy(inputs["cofitok_rollout"])
    inputs["cofitok_rollout"]["checkpoint_sha256"] = "different"

    with pytest.raises(ValueError, match="different checkpoints"):
        build_stability_qualification(**inputs)
