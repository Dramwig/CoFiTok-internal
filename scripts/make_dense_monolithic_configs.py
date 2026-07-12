#!/usr/bin/env python
"""Generate parameter-matched direct dense-epsilon baseline configs."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cofitok.configs import ModelConfig
from cofitok.models import CoFiTokTiny


SOURCE_CONFIGS = (
    "train_cifar10_k8_epsilononly_p150eval_3k_cuda.json",
    "train_tiny_imagenet_k8_epsilononly_p150eval_5k_cuda.json",
    "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_cuda.json",
    "train_downsampled_imagenet64_k8_epsilononly_p150eval_5k_cuda.json",
    "train_ffhq64_k8_epsilononly_p150eval_5k_cuda.json",
    "train_afhqv2_64_k8_epsilononly_p150eval_5k_cuda.json",
    "train_imagenet256_10pct_k4_epsilononly_p150eval_5k_cuda.json",
    "train_imagenet256_k4_epsilononly_p150eval_5k_cuda.json",
    "train_imagenet256_k4_epsilononly_p150eval_20k_seed103_cuda.json",
    "train_imagenet256_k4_epsilononly_p150eval_20k_seed139_cuda.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    return parser.parse_args()


def dense_name(source_name: str) -> str:
    if "epsilononly_p150eval" not in source_name:
        raise ValueError(f"source name does not identify endpoint-only config: {source_name}")
    return source_name.replace("epsilononly_p150eval", "densehead_p150eval")


def model_parameter_count(model: dict[str, Any]) -> int:
    values = asdict(ModelConfig())
    values.update(model)
    return sum(
        parameter.numel()
        for parameter in CoFiTokTiny(ModelConfig(**values)).parameters()
    )


def parameter_matched_width(
    source_model: dict[str, Any], dense_model: dict[str, Any]
) -> int:
    target = model_parameter_count(source_model)
    upper = max(8, int(source_model["base_channels"]))
    candidate = copy.deepcopy(dense_model)
    while True:
        candidate["base_channels"] = upper
        if model_parameter_count(candidate) >= target or upper >= 4096:
            break
        upper *= 2
    if upper >= 4096 and model_parameter_count(candidate) < target:
        raise ValueError("could not bracket a parameter-matched dense width")

    scored: list[tuple[int, bool, int]] = []
    for width in range(1, upper + 1):
        candidate["base_channels"] = width
        count = model_parameter_count(candidate)
        scored.append((abs(count - target), count > target, width))
    return min(scored)[2]


def build_dense_config(source: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(source)
    source_tokens = int(output["model"]["token_count"])
    if source_tokens not in {4, 8}:
        raise ValueError(f"expected K=4 or K=8 source, got K={source_tokens}")
    output["name"] = dense_name(str(output["name"]))
    output["model"].update(
        {
            "token_count": 1,
            "token_channels": int(output["model"]["image_channels"]),
            "predictor_use_feedback": False,
            "synthesis_mode": "dense_identity",
            "synthesis_active_token_channels": [],
            "synthesis_token_strides": [],
            "gamma_mode": "fixed_one",
        }
    )
    output["model"]["base_channels"] = parameter_matched_width(
        source["model"], output["model"]
    )
    for key in list(output["loss"]):
        if key.endswith("_weight"):
            output["loss"][key] = 0.0
    output["loss"]["epsilon_weight"] = 1.0
    return output


def main() -> None:
    args = parse_args()
    for filename in SOURCE_CONFIGS:
        source_path = args.config_dir / filename
        source = json.loads(source_path.read_text(encoding="utf-8"))
        dense = build_dense_config(source)
        destination = args.config_dir / dense_name(filename)
        destination.write_text(
            json.dumps(dense, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(destination)


if __name__ == "__main__":
    main()
