#!/usr/bin/env python
"""Generate the matched ImageNet-64 component and hyperparameter ablation suite."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath
from typing import Any


SEEDS = (103, 139)
DATASET = "imagenet_1k_64x64_hf"
BASE_CONFIG = "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_cuda.json"
DENSE_CONFIG = "train_imagenet_1k_64x64_hf_k8_densehead_p150eval_5k_cuda.json"
REMOTE_CHECKPOINT_ROOT = PurePosixPath("/root/autodl-tmp/CoFiTok/checkpoints")
DEFAULT_SUITE_ROOT = REMOTE_CHECKPOINT_ROOT / "aaai27_ablation_2026-07-11"


REFERENCE_VARIANTS = (
    {
        "variant": "full",
        "display": "CoFiTok (full)",
        "config_stem": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k",
        "checkpoint_dirs": {
            103: "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_2026-07-08",
            139: "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_2026-07-08",
        },
        "component_ablation": True,
        "sweep_values": {
            "lambda_prefix": 0.15,
            "lambda_component": 0.30,
            "progress_power": 1.50,
            "token_count": 8,
        },
    },
    {
        "variant": "endpoint_only",
        "display": "Endpoint-only factorized",
        "config_stem": "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k",
        "checkpoint_dirs": {
            103: "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_seed2_2026-07-08",
            139: "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_5k_2026-07-08",
        },
        "component_ablation": True,
    },
    {
        "variant": "simultaneous",
        "display": "Simultaneous tokens",
        "config_stem": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k",
        "checkpoint_dirs": {
            103: "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_seed2_2026-07-08",
            139: "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_2026-07-08",
        },
        "component_ablation": True,
    },
    {
        "variant": "deep_synthesis",
        "display": "Deep synthesis",
        "config_stem": "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k",
        "checkpoint_dirs": {
            103: "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_seed2_2026-07-08",
            139: "imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_cuda",
        },
        "component_ablation": True,
    },
    {
        "variant": "token_count_4",
        "display": "K=4",
        "config_stem": "train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k",
        "checkpoint_dirs": {
            103: "train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_seed2_2026-07-08",
            139: "train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_2026-07-08",
        },
        "sweep_values": {"token_count": 4},
    },
    {
        "variant": "token_count_16",
        "display": "K=16",
        "config_stem": "train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k",
        "checkpoint_dirs": {
            103: "train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_seed2_2026-07-08",
            139: "train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_2026-07-08",
        },
        "sweep_values": {"token_count": 16},
    },
)


NEW_VARIANTS = (
    ("no_path_prefix", "No path-prefix term", {"denoise_path_prefix_weight": 0.0}, {"lambda_prefix": 0.0}, True),
    ("lambda_prefix_005", r"$\lambda_{\mathrm{prefix}}=0.05$", {"denoise_path_prefix_weight": 0.05}, {"lambda_prefix": 0.05}, False),
    ("lambda_prefix_030", r"$\lambda_{\mathrm{prefix}}=0.30$", {"denoise_path_prefix_weight": 0.30}, {"lambda_prefix": 0.30}, False),
    ("no_path_component", "No path-component term", {"denoise_path_component_weight": 0.0}, {"lambda_component": 0.0}, True),
    ("lambda_component_010", r"$\lambda_{\mathrm{component}}=0.10$", {"denoise_path_component_weight": 0.10}, {"lambda_component": 0.10}, False),
    ("lambda_component_060", r"$\lambda_{\mathrm{component}}=0.60$", {"denoise_path_component_weight": 0.60}, {"lambda_component": 0.60}, False),
    ("progress_power_075", r"$p=0.75$", {"denoise_path_progress_power": 0.75}, {"progress_power": 0.75}, False),
    ("progress_power_100", r"$p=1.00$", {"denoise_path_progress_power": 1.00}, {"progress_power": 1.00}, False),
    ("progress_power_200", r"$p=2.00$", {"denoise_path_progress_power": 2.00}, {"progress_power": 2.00}, False),
    ("progress_power_250", r"$p=2.50$", {"denoise_path_progress_power": 2.50}, {"progress_power": 2.50}, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--output-config-dir",
        type=Path,
        default=Path("configs/ablations/aaai27_2026-07-11"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/experiment_conditions/aaai27_ablation_suite_2026-07-11.json"),
    )
    parser.add_argument("--suite-root", default=str(DEFAULT_SUITE_ROOT))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_filename(stem: str, seed: int) -> str:
    suffix = "_seed2_cuda.json" if seed == 103 else "_cuda.json"
    candidate = stem + suffix
    return candidate


def reference_entries(config_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in REFERENCE_VARIANTS:
        for seed in SEEDS:
            filename = config_filename(str(spec["config_stem"]), seed)
            path = config_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)
            config = read_json(path)
            entry = {
                "id": f"{spec['variant']}_seed{seed}",
                "variant": spec["variant"],
                "display": spec["display"],
                "seed": seed,
                "dataset": DATASET,
                "token_count": int(config["model"]["token_count"]),
                "config": path.as_posix(),
                "checkpoint_dir": str(REMOTE_CHECKPOINT_ROOT / spec["checkpoint_dirs"][seed]),
                "train": False,
                "component_ablation": bool(spec.get("component_ablation", False)),
                "sweep_values": dict(spec.get("sweep_values", {})),
            }
            entries.append(entry)
    return entries


def build_bundle(
    config_dir: Path,
    output_config_dir: Path,
    suite_root: str | Path,
) -> tuple[dict[Path, dict[str, Any]], dict[str, Any]]:
    remote_suite_root = PurePosixPath(str(suite_root).replace("\\", "/"))
    base = read_json(config_dir / BASE_CONFIG)
    dense_base = read_json(config_dir / DENSE_CONFIG)
    generated: dict[Path, dict[str, Any]] = {}
    entries = reference_entries(config_dir)

    for variant, display, loss_updates, sweep_values, component_ablation in NEW_VARIANTS:
        for seed in SEEDS:
            config = copy.deepcopy(base)
            name = f"train_imagenet64_hf_k8_aaai27_{variant}_5k_seed{seed}_cuda"
            config["name"] = name
            config["runtime"]["seed"] = seed
            config["runtime"]["steps"] = 5000
            config["loss"].update(loss_updates)
            output_path = output_config_dir / f"{name}.json"
            generated[output_path] = config
            run_id = f"{variant}_seed{seed}"
            entries.append(
                {
                    "id": run_id,
                    "variant": variant,
                    "display": display,
                    "seed": seed,
                    "dataset": DATASET,
                    "token_count": 8,
                    "config": output_path.as_posix(),
                    "checkpoint_dir": str(remote_suite_root / "train" / run_id),
                    "train": True,
                    "component_ablation": component_ablation,
                    "sweep_values": sweep_values,
                }
            )

    for seed in SEEDS:
        config = copy.deepcopy(dense_base)
        name = f"train_imagenet64_hf_aaai27_direct_dense_5k_seed{seed}_cuda"
        config["name"] = name
        config["runtime"]["seed"] = seed
        config["runtime"]["steps"] = 5000
        output_path = output_config_dir / f"{name}.json"
        generated[output_path] = config
        run_id = f"direct_dense_seed{seed}"
        entries.append(
            {
                "id": run_id,
                "variant": "direct_dense",
                "display": "Parameter-matched direct dense",
                "seed": seed,
                "dataset": DATASET,
                "token_count": 1,
                "config": output_path.as_posix(),
                "checkpoint_dir": str(remote_suite_root / "train" / run_id),
                "train": True,
                "component_ablation": False,
                "architecture_control": True,
                "sweep_values": {},
            }
        )

    manifest = {
        "schema_version": 1,
        "suite": "aaai27_component_and_hyperparameter_ablation_2026-07-11",
        "dataset": DATASET,
        "training_protocol": {
            "steps": 5000,
            "seeds": list(SEEDS),
            "batch_size": 32,
            "image_size": 64,
            "class_conditional": False,
        },
        "evaluation_protocol": {
            "split": "val",
            "image_count": 1024,
            "timestep": 500,
            "component_order": "ordered",
            "evaluation_progress_power": 1.5,
            "note": "All power-sweep models are scored against the same p=1.5 reference path.",
        },
        "suite_root": str(remote_suite_root),
        "entries": sorted(entries, key=lambda row: str(row["id"])),
    }
    return generated, manifest


def main() -> None:
    args = parse_args()
    generated, manifest = build_bundle(args.config_dir, args.output_config_dir, args.suite_root)
    for path, payload in generated.items():
        write_json(path, payload)
    write_json(args.manifest, manifest)
    print(f"wrote {len(generated)} configs and {len(manifest['entries'])} manifest entries")


if __name__ == "__main__":
    main()
