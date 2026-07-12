from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
BASELINE_DIR = CONFIG_DIR / "baselines"

DATASETS = {
    "imagenet_256_10pct": "imagenet256_10pct",
    "imagenet_256": "imagenet256",
}


def base_config(dataset: str, name: str, seed: int = 139) -> dict[str, Any]:
    return {
        "name": name,
        "data": {
            "dataset": dataset,
            "root": "/root/autodl-tmp/CoFiTok/datasets",
            "image_size": 256,
            "channels": 3,
            "batch_size": 4,
            "num_workers": 2,
            "class_conditional": False,
        },
        "model": {
            "image_channels": 3,
            "image_size": 256,
            "token_count": 4,
            "token_channels": 16,
            "base_channels": 24,
            "predictor_depth": 2,
            "synthesis_kernel_size": 3,
            "gamma_mode": "learned_scalar",
            "synthesis_active_token_channels": [4, 8, 12, 16],
        },
        "loss": {
            "epsilon_weight": 1.0,
            "prefix_weight": 0.0,
            "monotonic_weight": 0.0,
            "denoise_path_prefix_weight": 0.15,
            "denoise_path_component_weight": 0.3,
            "denoise_path_progress_power": 1.5,
        },
        "runtime": {
            "device": "cuda",
            "seed": seed,
            "steps": 5000,
        },
        "optimization": {
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "grad_clip_norm": 1.0,
            "log_interval": 1000,
            "visualization_count": 4,
        },
    }


def variants(dataset: str, stem: str) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}

    light = base_config(dataset, f"train_{stem}_k4_denoisepath_p150_light_5k_cuda", seed=139)
    configs[f"train_{stem}_k4_denoisepath_p150_light_5k_cuda.json"] = light

    dense = base_config(dataset, f"train_{stem}_k4_epsilononly_p150eval_5k_cuda", seed=139)
    dense["loss"] = {
        "epsilon_weight": 1.0,
        "prefix_weight": 0.0,
        "monotonic_weight": 0.0,
        "zero_token_weight": 0.01,
        "denoise_path_progress_power": 1.5,
    }
    configs[f"train_{stem}_k4_epsilononly_p150eval_5k_cuda.json"] = dense

    channel = base_config(dataset, f"train_{stem}_k4_channelmask_p150eval_5k_cuda", seed=139)
    channel["loss"] = {"denoise_path_progress_power": 1.5}
    configs[f"train_{stem}_k4_channelmask_p150eval_5k_cuda.json"] = channel

    nopath = base_config(dataset, f"train_{stem}_k4_denoisepath_p150_light_nopathprefix_5k_cuda", seed=157)
    nopath["loss"]["denoise_path_prefix_weight"] = 0.0
    configs[f"train_{stem}_k4_denoisepath_p150_light_nopathprefix_5k_cuda.json"] = nopath

    cleanmono = base_config(dataset, f"train_{stem}_k4_denoisepath_p150_light_cleanmono_5k_cuda", seed=163)
    cleanmono["loss"]["monotonic_weight"] = 0.02
    cleanmono["loss"]["monotonic_margin"] = 0.0
    configs[f"train_{stem}_k4_denoisepath_p150_light_cleanmono_5k_cuda.json"] = cleanmono

    simultaneous = base_config(dataset, f"train_{stem}_k4_denoisepath_p150_light_simultaneous_5k_cuda", seed=139)
    simultaneous["model"]["predictor_use_feedback"] = False
    configs[f"train_{stem}_k4_denoisepath_p150_light_simultaneous_5k_cuda.json"] = simultaneous

    deep = base_config(dataset, f"train_{stem}_k4_denoisepath_p150_light_deepsk_5k_cuda", seed=139)
    deep["model"]["synthesis_mode"] = "deep_decoder"
    deep["model"]["deep_synthesis_hidden_channels"] = 24
    deep["model"]["deep_synthesis_depth"] = 3
    configs[f"train_{stem}_k4_denoisepath_p150_light_deepsk_5k_cuda.json"] = deep

    return configs


def baseline_config(dataset: str, baseline: str) -> dict[str, Any]:
    return {
        "name": f"baseline_{baseline}_{dataset}",
        "data": {
            "dataset": dataset,
            "root": "/root/autodl-tmp/CoFiTok/datasets",
            "image_size": 256,
            "channels": 3,
            "batch_size": 4,
            "num_workers": 2,
            "class_conditional": False,
        },
        "runtime": {
            "device": "cuda",
            "seed": 139,
            "steps": 5000,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    written: list[Path] = []
    for dataset, stem in DATASETS.items():
        for filename, payload in variants(dataset, stem).items():
            path = CONFIG_DIR / filename
            write_json(path, payload)
            written.append(path)
        for baseline in ("improved_diffusion", "edm"):
            path = BASELINE_DIR / baseline / f"{dataset}.json"
            write_json(path, baseline_config(dataset, baseline),)
            written.append(path)

    manifest = {
        "protocol": "imagenet_256_p0_5k",
        "datasets": list(DATASETS),
        "image_size": 256,
        "internal_steps": 5000,
        "internal_batch_size": 4,
        "token_count": 4,
        "token_channels": 16,
        "base_channels": 24,
        "files": [path.relative_to(ROOT).as_posix() for path in written],
    }
    manifest_path = ROOT / "docs" / "experiment_conditions" / "imagenet_256_p0_config_manifest_2026-07-10.json"
    write_json(manifest_path, manifest)
    print(f"wrote {len(written)} configs")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
