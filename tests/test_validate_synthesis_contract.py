import json
from pathlib import Path

import pytest

from scripts.validate_synthesis_contract import validate_configs


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_config(name: str, synthesis_mode: str = "restricted") -> dict:
    return {
        "name": name,
        "data": {"dataset": "random", "image_size": 16, "batch_size": 2},
        "model": {
            "image_channels": 3,
            "image_size": 16,
            "token_count": 2,
            "token_channels": 4,
            "base_channels": 8,
            "predictor_depth": 1,
            "synthesis_mode": synthesis_mode,
            "synthesis_kernel_size": 3,
            "gamma_mode": "learned_scalar",
        },
        "runtime": {"device": "cpu", "seed": 1, "steps": 1},
    }


def test_validate_configs_accepts_restricted_contract(tmp_path: Path) -> None:
    path = tmp_path / "restricted.json"
    _write_config(path, _base_config("restricted_contract"))

    result = validate_configs([path], spatial_size=4, static_only=True)

    assert result["status"] == "ok"
    assert result["validation_mode"] == "static"
    assert result["restricted_ok_count"] == 1
    assert result["ablation_count"] == 0
    assert result["results"][0]["evidence"]["mode"] == "static"


def test_validate_configs_records_deep_synthesis_as_ablation(tmp_path: Path) -> None:
    path = tmp_path / "deep.json"
    payload = _base_config("deep_ablation", synthesis_mode="deep_decoder")
    payload["model"]["deep_synthesis_hidden_channels"] = 8
    payload["model"]["deep_synthesis_depth"] = 2
    _write_config(path, payload)

    restricted = tmp_path / "restricted.json"
    _write_config(restricted, _base_config("restricted_contract"))
    result = validate_configs([path, restricted], spatial_size=4, static_only=True)

    assert result["ablation_count"] == 1
    assert result["restricted_ok_count"] == 1


def test_validate_configs_strict_mode_rejects_ablation(tmp_path: Path) -> None:
    path = tmp_path / "deep.json"
    payload = _base_config("deep_ablation", synthesis_mode="deep_decoder")
    payload["model"]["deep_synthesis_hidden_channels"] = 8
    payload["model"]["deep_synthesis_depth"] = 2
    _write_config(path, payload)

    with pytest.raises(AssertionError, match="non-restricted ablation"):
        validate_configs([path], spatial_size=4, strict_ablation=True, static_only=True)
