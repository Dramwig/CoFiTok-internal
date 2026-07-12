#!/usr/bin/env python
"""Compare the MAR community safetensors with pinned PTH model states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/CoFiTok/checkpoints/baselines/mar/official_imagenet256_eval_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=Path, default=ROOT / "official_pth/checkpoint-last.pth")
    parser.add_argument("--safetensors", type=Path, default=ROOT / "hf_repo/mar-base.safetensors")
    parser.add_argument("--vae-ckpt", type=Path, default=ROOT / "official_pth/kl16.ckpt")
    parser.add_argument("--vae-safetensors", type=Path, default=ROOT / "hf_repo/kl16.safetensors")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "official_pth/community_conversion_audit.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_mapping(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {key!r} state is missing or not a mapping")
    return state


def compare_state_to_safetensors(state: Mapping[str, Any], safe_path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    with safe_open(safe_path, framework="pt", device="cpu") as handle:
        safe_keys = set(handle.keys())
        state_keys = set(state)
        common_keys = sorted(safe_keys & state_keys)
        exact_tensors = 0
        compared_elements = 0
        absolute_error_sum = 0.0
        squared_error_sum = 0.0
        max_absolute_error = 0.0
        shape_mismatches: list[str] = []

        for key in common_keys:
            safe_tensor = handle.get_tensor(key)
            state_tensor = state[key].detach().cpu()
            if safe_tensor.shape != state_tensor.shape:
                shape_mismatches.append(key)
                continue
            if torch.equal(safe_tensor, state_tensor):
                exact_tensors += 1
                compared_elements += safe_tensor.numel()
                continue
            difference = safe_tensor.float() - state_tensor.float()
            compared_elements += difference.numel()
            absolute_error_sum += difference.abs().sum(dtype=torch.float64).item()
            squared_error_sum += difference.square().sum(dtype=torch.float64).item()
            max_absolute_error = max(max_absolute_error, difference.abs().max().item())

    return {
        "safe_key_count": len(safe_keys),
        "state_key_count": len(state_keys),
        "common_key_count": len(common_keys),
        "missing_from_safetensors": sorted(state_keys - safe_keys),
        "missing_from_state": sorted(safe_keys - state_keys),
        "shape_mismatches": shape_mismatches,
        "exact_tensor_count": exact_tensors,
        "compared_elements": compared_elements,
        "mean_absolute_error": absolute_error_sum / compared_elements if compared_elements else None,
        "root_mean_squared_error": (
            (squared_error_sum / compared_elements) ** 0.5 if compared_elements else None
        ),
        "max_absolute_error": max_absolute_error,
        "all_tensors_exact": (
            exact_tensors == len(common_keys)
            and not shape_mismatches
            and safe_keys == state_keys
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    import torch

    args = parse_args()
    for path in (args.pth, args.safetensors, args.vae_ckpt, args.vae_safetensors):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)
    model_comparisons = {
        state_name: compare_state_to_safetensors(
            state_mapping(checkpoint, state_name), args.safetensors
        )
        for state_name in ("model", "model_ema")
    }
    del checkpoint

    vae_checkpoint = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    vae_comparison = compare_state_to_safetensors(
        state_mapping(vae_checkpoint, "model"), args.vae_safetensors
    )
    del vae_checkpoint

    best_state = min(
        model_comparisons,
        key=lambda name: (
            model_comparisons[name]["root_mean_squared_error"] is None,
            model_comparisons[name]["root_mean_squared_error"] or 0.0,
        ),
    )
    payload = {
        "schema_version": 1,
        "audit": "mar_community_safetensors_conversion",
        "assets": {
            "official_pth": {"path": str(args.pth), "sha256": sha256_file(args.pth)},
            "community_safetensors": {
                "path": str(args.safetensors),
                "sha256": sha256_file(args.safetensors),
            },
            "official_vae_ckpt": {
                "path": str(args.vae_ckpt),
                "sha256": sha256_file(args.vae_ckpt),
            },
            "community_vae_safetensors": {
                "path": str(args.vae_safetensors),
                "sha256": sha256_file(args.vae_safetensors),
            },
        },
        "model_comparisons": model_comparisons,
        "best_matching_model_state": best_state,
        "vae_comparison": vae_comparison,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
