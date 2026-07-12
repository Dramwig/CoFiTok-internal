import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "baselines" / "sample_mar_official_pth.py"
SPEC = importlib.util.spec_from_file_location("sample_mar_official_pth", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_checkpoint_state_requires_named_mapping() -> None:
    state = {"weight": object()}
    assert MODULE.checkpoint_state({"model_ema": state}, "model_ema") is state
    with pytest.raises(KeyError, match="model_ema"):
        MODULE.checkpoint_state({"model": state}, "model_ema")
    with pytest.raises(TypeError, match="not a mapping"):
        MODULE.checkpoint_state({"model_ema": 3}, "model_ema")


def test_official_defaults_pin_readme_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", ["sample_mar_official_pth.py"])
    args = MODULE.parse_args()

    assert args.num_images == 50_000
    assert args.num_iter == 256
    assert args.num_sampling_steps == 100
    assert args.cfg == 2.9
    assert args.cfg_schedule == "linear"
    assert args.temperature == 1.0
    assert MODULE.EXPECTED_MODEL_SHA256.startswith("7e970a33")
