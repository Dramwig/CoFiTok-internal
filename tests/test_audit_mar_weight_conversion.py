import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "audit_mar_weight_conversion.py"
SPEC = importlib.util.spec_from_file_location("audit_mar_weight_conversion", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_state_mapping_requires_mapping() -> None:
    state = {"weight": object()}
    assert MODULE.state_mapping({"model_ema": state}, "model_ema") is state
    with pytest.raises(TypeError, match="missing or not a mapping"):
        MODULE.state_mapping({}, "model_ema")
