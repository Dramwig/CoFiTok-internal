import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_order_permutations.py"
SPEC = importlib.util.spec_from_file_location("evaluate_order_permutations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_component_orders_exhaust_k4_and_anchor_expected_orders() -> None:
    orders = MODULE.component_orders(4)
    assert len(orders) == 24
    assert orders[0] == (0, 1, 2, 3)
    assert (3, 2, 1, 0) in orders
    assert len(set(orders)) == 24


def test_bootstrap_mean_ci_detects_positive_paired_delta() -> None:
    values = np.linspace(0.1, 0.3, 128)
    result = MODULE.bootstrap_mean_ci(values, repetitions=1000, seed=7)
    assert result["ci_low"] > 0.0
    assert result["ci_low"] < result["mean"] < result["ci_high"]
