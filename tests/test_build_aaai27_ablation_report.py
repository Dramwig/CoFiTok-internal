from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_aaai27_ablation_report.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_aaai27_ablation_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(variant: str, seed: int, path_auc: float, endpoint_mse: float) -> dict:
    return {
        "variant": variant,
        "seed": seed,
        "path_auc": path_auc,
        "endpoint_mse": endpoint_mse,
    }


def test_aggregate_uses_sample_standard_deviation() -> None:
    module = load_module()
    result = module.aggregate([1.0, 3.0])
    assert result["mean"] == 2.0
    assert abs(result["std"] - 2**0.5) < 1e-12


def test_paired_summary_counts_same_seed_path_wins() -> None:
    module = load_module()
    rows = [
        row("full", 103, 0.1, 0.20),
        row("full", 139, 0.2, 0.22),
        row("no_path_component", 103, 0.3, 0.19),
        row("no_path_component", 139, 0.4, 0.21),
    ]
    result = module.paired_summary(rows, "no_path_component")
    assert result["full_path_better_count"] == 2
    assert result["path_pair_count"] == 2
    assert result["mean_endpoint_relative_change"] > 0.0
