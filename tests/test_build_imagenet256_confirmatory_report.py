import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_imagenet256_confirmatory_report.py"
SPEC = importlib.util.spec_from_file_location("build_imagenet256_confirmatory_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_confirmatory_gate_logic_passes_scoped_prediction() -> None:
    rows = [
        {
            "endpoint_only_path_auc": 1.0,
            "cofitok_ordered_path_auc": 0.2,
            "cofitok_random_path_auc": 0.5,
            "cofitok_reverse_path_auc": 0.8,
            "dense_monolithic_endpoint_mse": 0.10,
            "cofitok_endpoint_mse": 0.104,
            "cofitok_ordered_endpoint_mse": 0.104,
            "cofitok_random_endpoint_mse": 0.104000001,
            "cofitok_reverse_endpoint_mse": 0.104000002,
            "cofitok_zero_ratio": 0.0,
            "nonidentity_delta_ci_low": 0.1,
            "reverse_delta_ci_low": 0.2,
            "exhaustive_endpoint_epsilon_sum_mse_max": 1e-15,
        },
        {
            "endpoint_only_path_auc": 0.9,
            "cofitok_ordered_path_auc": 0.1,
            "cofitok_random_path_auc": 0.4,
            "cofitok_reverse_path_auc": 0.7,
            "dense_monolithic_endpoint_mse": 0.11,
            "cofitok_endpoint_mse": 0.113,
            "cofitok_ordered_endpoint_mse": 0.113,
            "cofitok_random_endpoint_mse": 0.113000001,
            "cofitok_reverse_endpoint_mse": 0.113000002,
            "cofitok_zero_ratio": 0.0,
            "nonidentity_delta_ci_low": 0.1,
            "reverse_delta_ci_low": 0.2,
            "exhaustive_endpoint_epsilon_sum_mse_max": 1e-15,
        },
    ]

    decision = MODULE.evaluate_gates(rows)
    assert decision["overall_pass"]
    assert all(decision["gates"].values())


def test_confirmatory_gate_logic_rejects_endpoint_tradeoff() -> None:
    rows = [
        {
            "endpoint_only_path_auc": 1.0,
            "cofitok_ordered_path_auc": 0.2,
            "cofitok_random_path_auc": 0.5,
            "cofitok_reverse_path_auc": 0.8,
            "dense_monolithic_endpoint_mse": 0.10,
            "cofitok_endpoint_mse": 0.11,
            "cofitok_ordered_endpoint_mse": 0.11,
            "cofitok_random_endpoint_mse": 0.11,
            "cofitok_reverse_endpoint_mse": 0.11,
            "cofitok_zero_ratio": 0.0,
            "nonidentity_delta_ci_low": 0.1,
            "reverse_delta_ci_low": 0.2,
            "exhaustive_endpoint_epsilon_sum_mse_max": 1e-15,
        }
    ]

    decision = MODULE.evaluate_gates(rows)
    assert not decision["gates"]["mean_endpoint_mse_within_plus_5pct_of_dense_monolithic"]
    assert not decision["overall_pass"]


def test_quality_validation_locks_random_permutation_seed() -> None:
    config_name = MODULE.expected_config_name("cofitok", 103)
    payload = {
        "config": {
            "name": config_name,
            "data": {"dataset": "imagenet_256"},
            "runtime": {"seed": 103, "steps": 20_000},
            "model": {
                "token_count": 4,
                "synthesis_mode": "restricted",
                "predictor_use_feedback": True,
            },
            "loss": {"denoise_path_prefix_weight": 1.5},
        },
        "checkpoint": str(
            Path("/tmp") / (config_name + "_" + MODULE.RUN_TAG) / "checkpoint_final.pt"
        ),
        "evaluation": {
            "image_count": 1_024,
            "fixed_timestep": 500,
            "prefix_budgets": [1, 2, 3, 4],
            "component_order": "random",
            "random_order_seed": 1,
        },
    }

    MODULE.validate_quality(
        payload,
        seed=103,
        order="random",
        expected_token_count=4,
        expected_budgets=[1, 2, 3, 4],
        expected_method="cofitok",
    )
    payload["evaluation"]["random_order_seed"] = 0
    try:
        MODULE.validate_quality(
            payload,
            seed=103,
            order="random",
            expected_token_count=4,
            expected_budgets=[1, 2, 3, 4],
            expected_method="cofitok",
        )
    except ValueError as error:
        assert "random-order seed mismatch" in str(error)
    else:
        raise AssertionError("stale random seed must be rejected")
