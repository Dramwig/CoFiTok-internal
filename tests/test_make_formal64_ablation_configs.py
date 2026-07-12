from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "make_formal64_ablation_configs.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("make_formal64_ablation_configs", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_base(config_dir: Path, name: str, dataset: str) -> None:
    payload = {
        "name": name.removesuffix(".json"),
        "data": {"dataset": dataset},
        "model": {
            "token_count": 8,
            "predictor_use_feedback": True,
            "synthesis_mode": "restricted",
        },
        "loss": {
            "denoise_path_prefix_weight": 0.15,
            "denoise_path_component_weight": 0.3,
            "monotonic_weight": 0.0,
        },
        "runtime": {"steps": 5000, "seed": 139},
    }
    (config_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_build_configs_generates_expected_formal64_ablation_variants(tmp_path: Path) -> None:
    module = load_script_module()
    write_base(tmp_path, "train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json", "tiny_imagenet_200")
    write_base(
        tmp_path,
        "train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda.json",
        "downsampled_imagenet_64",
    )
    write_base(tmp_path, "train_ffhq64_k8_denoisepath_p150_light_5k_cuda.json", "ffhq_64")
    write_base(tmp_path, "train_afhqv2_64_k8_denoisepath_p150_light_5k_cuda.json", "afhqv2_64")

    generated = module.build_configs(tmp_path)
    by_name = {config["name"]: config for config in generated.values()}

    assert len(generated) == 16
    assert by_name[
        "train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_5k_cuda"
    ]["loss"]["denoise_path_prefix_weight"] == 0.0
    assert by_name[
        "train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_5k_cuda"
    ]["loss"]["monotonic_weight"] == 0.02
    assert by_name[
        "train_downsampled_imagenet64_k8_denoisepath_p150_light_nopathprefix_5k_cuda"
    ]["loss"]["denoise_path_prefix_weight"] == 0.0
    assert by_name[
        "train_ffhq64_k8_denoisepath_p150_light_cleanmono_5k_cuda"
    ]["loss"]["monotonic_weight"] == 0.02
    assert by_name[
        "train_afhqv2_64_k8_denoisepath_p150_light_simultaneous_5k_cuda"
    ]["model"]["predictor_use_feedback"] is False
    assert by_name[
        "train_tiny_imagenet_k8_denoisepath_p150_light_deepsk_5k_cuda"
    ]["model"]["synthesis_mode"] == "deep_decoder"
