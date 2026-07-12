import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_imagenet256_long_budget_configs.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "make_imagenet256_long_budget_configs",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_config_changes_only_run_identity_and_budget() -> None:
    module = load_module()
    template = {
        "name": "old",
        "data": {"dataset": "imagenet_256"},
        "runtime": {"seed": 1, "steps": 5_000},
        "optimization": {"log_interval": 1_000},
        "model": {"token_count": 4},
    }

    payload = module.build_config(template, "cofitok", 103)

    assert payload["name"] == "train_imagenet256_k4_denoisepath_p150_light_20k_seed103_cuda"
    assert payload["runtime"] == {"seed": 103, "steps": 20_000}
    assert payload["optimization"]["log_interval"] == 2_000
    assert payload["data"] == template["data"]
    assert payload["model"] == template["model"]
    assert template["runtime"] == {"seed": 1, "steps": 5_000}
