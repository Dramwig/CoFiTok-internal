from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


TARGETS = [
    {
        "prefix": "tiny_imagenet",
        "base": "train_tiny_imagenet_k8_denoisepath_p150_light_5k_cuda.json",
        "variants": ["nopathprefix", "cleanmono", "simultaneous", "deepsk"],
    },
    {
        "prefix": "downsampled_imagenet64",
        "base": "train_downsampled_imagenet64_k8_denoisepath_p150_light_5k_cuda.json",
        "variants": ["nopathprefix", "cleanmono", "simultaneous", "deepsk"],
    },
    {
        "prefix": "ffhq64",
        "base": "train_ffhq64_k8_denoisepath_p150_light_5k_cuda.json",
        "variants": ["nopathprefix", "cleanmono", "simultaneous", "deepsk"],
    },
    {
        "prefix": "afhqv2_64",
        "base": "train_afhqv2_64_k8_denoisepath_p150_light_5k_cuda.json",
        "variants": ["nopathprefix", "cleanmono", "simultaneous", "deepsk"],
    },
]


VARIANT_SEEDS = {
    "nopathprefix": 157,
    "cleanmono": 163,
    "simultaneous": 139,
    "deepsk": 139,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate formal64 internal ablation configs.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_variant(base: dict[str, Any], prefix: str, variant: str) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["name"] = f"train_{prefix}_k8_denoisepath_p150_light_{variant}_5k_cuda"
    config.setdefault("runtime", {})["steps"] = 5000
    config["runtime"]["seed"] = VARIANT_SEEDS[variant]

    loss = config.setdefault("loss", {})
    model = config.setdefault("model", {})

    if variant == "nopathprefix":
        loss["denoise_path_prefix_weight"] = 0.0
    elif variant == "cleanmono":
        loss["monotonic_weight"] = 0.02
        loss["monotonic_margin"] = 0.0
    elif variant == "simultaneous":
        model["predictor_use_feedback"] = False
    elif variant == "deepsk":
        model["synthesis_mode"] = "deep_decoder"
        model["deep_synthesis_hidden_channels"] = 32
        model["deep_synthesis_depth"] = 4
    else:
        raise ValueError(f"unknown variant: {variant}")

    return config


def build_configs(config_dir: Path) -> dict[Path, dict[str, Any]]:
    generated: dict[Path, dict[str, Any]] = {}
    for target in TARGETS:
        base_path = config_dir / target["base"]
        base = _read_json(base_path)
        for variant in target["variants"]:
            config = _make_variant(base, str(target["prefix"]), variant)
            output_path = config_dir / f"{config['name']}.json"
            generated[output_path] = config
    return generated


def main() -> None:
    args = parse_args()
    generated = build_configs(args.config_dir)
    for path, config in generated.items():
        if not args.dry_run:
            _write_json(path, config)
        print(path.as_posix())


if __name__ == "__main__":
    main()
