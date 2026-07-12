#!/usr/bin/env python
"""Sample MAR-B from the pinned LTH14 checkpoint's EMA state."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/CoFiTok")
DEFAULT_REPO = ROOT / "baselines/repos/mar"
DEFAULT_ASSETS = (
    ROOT
    / "checkpoints/baselines/mar/official_imagenet256_eval_only/official_pth"
)
DEFAULT_OUTPUT = (
    ROOT
    / "checkpoints/baselines/mar/official_imagenet256_eval_only/samples_50k_official_pth"
)
EXPECTED_MODEL_SHA256 = (
    "7e970a33bc90353e2fabe3498ed1f2d194dd8d17cd387665f80b2984dfca538c"
)
EXPECTED_VAE_SHA256 = (
    "34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f"
)


def load_sampling_core() -> Any:
    path = Path(__file__).with_name("sample_mar_hf_official.py")
    spec = importlib.util.spec_from_file_location("_cofitok_mar_sampling_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load MAR sampling core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_sampling_core()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ImageNet-256 samples from the pinned LTH14 MAR-B EMA checkpoint."
    )
    parser.add_argument("--official-repo-dir", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-filename", default="checkpoint-last.pth")
    parser.add_argument("--vae-filename", default="kl16.ckpt")
    parser.add_argument("--num-images", type=int, default=50_000)
    parser.add_argument("--class-num", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-batch-size", type=int, default=16)
    parser.add_argument("--num-iter", type=int, default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--cfg", type=float, default=2.9)
    parser.add_argument("--cfg-schedule", choices=["linear", "constant"], default="linear")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--pack-npz", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def checkpoint_state(checkpoint: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in checkpoint:
        raise KeyError(f"checkpoint is missing required state {key!r}")
    state = checkpoint[key]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint state {key!r} is not a mapping")
    return state


def load_official_modules(repo_dir: Path) -> tuple[Any, Any]:
    models_dir = repo_dir / "models"
    if not (models_dir / "mar.py").is_file() or not (models_dir / "vae.py").is_file():
        raise FileNotFoundError(f"pinned LTH14 MAR models are missing under {models_dir}")

    repo_text = str(repo_dir)
    sys.path.insert(0, repo_text)
    try:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                del sys.modules[name]
        mar_module = importlib.import_module("models.mar")
        vae_module = importlib.import_module("models.vae")
    finally:
        if sys.path and sys.path[0] == repo_text:
            sys.path.pop(0)
    return mar_module, vae_module


def build_models(args: argparse.Namespace, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    model_path = args.asset_dir / args.model_filename
    vae_path = args.asset_dir / args.vae_filename
    expected = {
        model_path: EXPECTED_MODEL_SHA256,
        vae_path: EXPECTED_VAE_SHA256,
    }
    for path, expected_sha in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"required official MAR asset is missing: {path}")
        actual_sha = CORE.sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"sha256 mismatch for {path}: {actual_sha} != {expected_sha}")

    mar_module, vae_module = load_official_modules(args.official_repo_dir)
    model = mar_module.mar_base(
        img_size=256,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        buffer_size=64,
        diffloss_d=6,
        diffloss_w=1024,
        num_sampling_steps=str(args.num_sampling_steps),
    )
    model_checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_state(model_checkpoint, "model_ema"), strict=True)
    del model_checkpoint
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    vae = vae_module.AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=None,
    )
    vae_checkpoint = torch.load(vae_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(checkpoint_state(vae_checkpoint, "model"), strict=True)
    del vae_checkpoint
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    assets = {
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": EXPECTED_MODEL_SHA256,
            "checkpoint_state": "model_ema",
        },
        "vae": {
            "path": str(vae_path),
            "bytes": vae_path.stat().st_size,
            "sha256": EXPECTED_VAE_SHA256,
            "checkpoint_state": "model",
        },
        "official_repo": {
            "path": str(args.official_repo_dir),
            "commit": "c6d53f7fa6427634b5850ebed771b7c2d19ea21f",
        },
    }
    return model, vae, assets


def main() -> None:
    args = parse_args()
    args.hf_repo_dir = args.asset_dir
    args.source_root = args.asset_dir
    args.protocol = "official_lth14_pth_ema_eval_only"
    args.metric_note = (
        "Pinned LTH14 MAR-B checkpoint model_ema and original KL-16 VAE with the "
        "official ImageNet-256 sampling parameters. This is eval-only and not "
        "same-budget P0 retraining."
    )
    payload = CORE.run(args, model_builder=build_models)
    print(json.dumps(payload["progress"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
