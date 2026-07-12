from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_aaai27_experiment_tables.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_aaai27_experiment_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_result_rows_exclude_legacy_objective() -> None:
    module = load_module()
    evidence = {
        "diagnostic_rows": [
            {
                "dataset": "cifar10",
                "cofitok_psnr": 1.0,
                "dense_psnr": 2.0,
                "channel_mask_psnr": 3.0,
                "cofitok_path_auc": 0.1,
                "endpoint_only_path_auc": 0.8,
                "cofitok_effective_tokens": 6.0,
                "endpoint_only_effective_tokens": 5.0,
                "cofitok_zero_ratio": 0.0,
                "deep_synthesis_zero_ratio": 0.05,
            }
        ]
    }
    line = module.build_all_dataset_rows(evidence)[0]
    assert "CIFAR-10" in line
    assert "3.000" not in line
    assert "0.0500" in line
