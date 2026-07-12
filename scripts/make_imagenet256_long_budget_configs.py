#!/usr/bin/env python
"""Generate matched ImageNet-256 20k CoFiTok/dense repeat configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


TEMPLATES = {
    "cofitok": Path("configs/train_imagenet256_k4_denoisepath_p150_light_5k_cuda.json"),
    "dense": Path("configs/train_imagenet256_k4_epsilononly_p150eval_5k_cuda.json"),
}
SEEDS = (103, 139)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser.parse_args()


def config_stem(method: str, seed: int) -> str:
    variant = (
        "denoisepath_p150_light" if method == "cofitok" else "epsilononly_p150eval"
    )
    return f"train_imagenet256_k4_{variant}_20k_seed{seed}_cuda"


def build_config(template: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    payload = copy.deepcopy(template)
    payload["name"] = config_stem(method, seed)
    payload["runtime"]["seed"] = seed
    payload["runtime"]["steps"] = 20_000
    payload["optimization"]["log_interval"] = 2_000
    return payload


def generate(project_root: Path) -> list[Path]:
    outputs: list[Path] = []
    for method, relative_template in TEMPLATES.items():
        template_path = project_root / relative_template
        template = json.loads(template_path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            stem = config_stem(method, seed)
            output = project_root / "configs" / f"{stem}.json"
            output.write_text(
                json.dumps(build_config(template, method, seed), indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(output)
    return outputs


def main() -> None:
    args = parse_args()
    for path in generate(args.project_root):
        print(path)


if __name__ == "__main__":
    main()
