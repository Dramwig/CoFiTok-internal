from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_aaai27_ablation_configs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("make_aaai27_ablation_configs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_bundle_has_matched_two_seed_sweeps(tmp_path: Path) -> None:
    module = load_module()
    generated, manifest = module.build_bundle(
        ROOT / "configs",
        tmp_path / "configs",
        Path("/tmp/aaai27_ablation"),
    )
    assert len(generated) == 22
    assert len(manifest["entries"]) == 34
    assert manifest["training_protocol"]["seeds"] == [103, 139]
    assert manifest["evaluation_protocol"]["evaluation_progress_power"] == 1.5
    assert manifest["suite_root"] == "/tmp/aaai27_ablation"
    assert all("\\" not in entry["checkpoint_dir"] for entry in manifest["entries"])

    by_id = {entry["id"]: entry for entry in manifest["entries"]}
    no_prefix = generated[Path(by_id["no_path_prefix_seed103"]["config"])]
    no_component = generated[Path(by_id["no_path_component_seed139"]["config"])]
    power = generated[Path(by_id["progress_power_250_seed103"]["config"])]
    dense = generated[Path(by_id["direct_dense_seed139"]["config"])]
    assert no_prefix["loss"]["denoise_path_prefix_weight"] == 0.0
    assert no_prefix["loss"]["denoise_path_component_weight"] == 0.3
    assert no_component["loss"]["denoise_path_prefix_weight"] == 0.15
    assert no_component["loss"]["denoise_path_component_weight"] == 0.0
    assert power["loss"]["denoise_path_progress_power"] == 2.5
    assert dense["model"]["synthesis_mode"] == "dense_identity"
    assert dense["model"]["token_count"] == 1


def test_reference_defaults_participate_in_every_primary_sweep(tmp_path: Path) -> None:
    module = load_module()
    _, manifest = module.build_bundle(
        ROOT / "configs",
        tmp_path / "configs",
        Path("/tmp/aaai27_ablation"),
    )
    full = [entry for entry in manifest["entries"] if entry["variant"] == "full"]
    assert {entry["seed"] for entry in full} == {103, 139}
    for entry in full:
        assert entry["sweep_values"] == {
            "lambda_prefix": 0.15,
            "lambda_component": 0.30,
            "progress_power": 1.50,
            "token_count": 8,
        }
