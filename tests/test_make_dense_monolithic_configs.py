import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_dense_monolithic_configs.py"
SPEC = importlib.util.spec_from_file_location("make_dense_monolithic_configs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dense_config_is_direct_and_parameter_matching_ready() -> None:
    source = {
        "name": "train_demo_k8_epsilononly_p150eval_5k_cuda",
        "model": {
            "token_count": 8,
            "token_channels": 16,
            "image_channels": 3,
            "base_channels": 48,
            "predictor_use_feedback": True,
            "synthesis_mode": "restricted",
            "synthesis_active_token_channels": [],
            "synthesis_token_strides": [],
            "gamma_mode": "learned_scalar",
        },
        "loss": {
            "epsilon_weight": 1.0,
            "prefix_weight": 0.0,
            "zero_token_weight": 0.01,
        },
    }

    dense = MODULE.build_dense_config(source)

    assert dense["name"] == "train_demo_k8_densehead_p150eval_5k_cuda"
    assert dense["model"]["token_count"] == 1
    assert dense["model"]["token_channels"] == 3
    assert dense["model"]["base_channels"] == 52
    assert dense["model"]["synthesis_mode"] == "dense_identity"
    assert dense["model"]["predictor_use_feedback"] is False
    assert dense["loss"]["epsilon_weight"] == 1.0
    assert dense["loss"]["zero_token_weight"] == 0.0
    source_count = MODULE.model_parameter_count(source["model"])
    dense_count = MODULE.model_parameter_count(dense["model"])
    assert abs(dense_count / source_count - 1.0) <= 0.02


def test_all_generated_dense_configs_are_within_two_percent() -> None:
    for filename in MODULE.SOURCE_CONFIGS:
        source = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        dense = MODULE.build_dense_config(source)
        source_count = MODULE.model_parameter_count(source["model"])
        dense_count = MODULE.model_parameter_count(dense["model"])
        assert abs(dense_count / source_count - 1.0) <= 0.02, filename


def test_all_factorization_controls_share_training_conditions() -> None:
    changed_model_keys = {
        "token_count",
        "token_channels",
        "base_channels",
        "predictor_use_feedback",
        "synthesis_mode",
        "synthesis_active_token_channels",
        "synthesis_token_strides",
        "gamma_mode",
    }
    for filename in MODULE.SOURCE_CONFIGS:
        source = json.loads((ROOT / "configs" / filename).read_text(encoding="utf-8"))
        cofitok_name = filename.replace(
            "epsilononly_p150eval", "denoisepath_p150_light"
        )
        cofitok = json.loads(
            (ROOT / "configs" / cofitok_name).read_text(encoding="utf-8")
        )
        dense = MODULE.build_dense_config(source)

        for section in ("data", "diffusion", "runtime", "optimization"):
            assert source.get(section) == cofitok.get(section), (filename, section)
            assert source.get(section) == dense.get(section), (filename, section)
        assert source["model"] == cofitok["model"], filename
        for key, value in source["model"].items():
            if key not in changed_model_keys:
                assert dense["model"][key] == value, (filename, key)
