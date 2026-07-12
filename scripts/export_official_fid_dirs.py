from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.utils import save_image

from cofitok.configs import config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.utils.seed import seed_everything
from scripts.evaluate_generated_samples import _batch_images
from scripts.sample_checkpoint import denormalize, ddim_sample

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export real and generated PNG directories for an external official FID implementation. "
            "This intentionally writes image files instead of computing an in-process proxy metric."
        )
    )
    parser.add_argument("--config", required=True, help="Experiment config defining data/model shape.")
    parser.add_argument("--checkpoint", default="", help="checkpoint_final.pt. Required for generated/both modes.")
    parser.add_argument("--output-dir", required=True, help="Export root containing real/ and generated/.")
    parser.add_argument("--mode", default="both", choices=["both", "real", "generated"])
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--real-count", type=int, default=50000)
    parser.add_argument("--sample-count", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--prefix-budget", type=int, default=-1, help="Default uses the full token budget.")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--clip-x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing PNGs in the selected export directories before writing.",
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in _IMAGE_EXTENSIONS)


def prepare_image_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = _image_files(path)
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(f"{path} already contains {len(existing)} images; pass --overwrite to replace them")
    for file in existing:
        file.unlink()


def save_png_batch(images: torch.Tensor, output_dir: Path, start_index: int) -> list[str]:
    paths = []
    for offset, image in enumerate(denormalize(images).cpu()):
        path = output_dir / f"{start_index + offset:08d}.png"
        save_image(image, path)
        paths.append(path.as_posix())
    return paths


@torch.no_grad()
def export_real_images(
    config: Any,
    output_dir: Path,
    split: str,
    total_images: int,
    batch_size: int,
    device: torch.device,
) -> int:
    loader_config = replace(config.data, batch_size=batch_size)
    loader = build_dataloader(loader_config, split=split, drop_last=False)
    count = 0
    for batch in loader:
        if count >= total_images:
            break
        images = _batch_images(batch, device)
        images = images[: total_images - count]
        save_png_batch(images, output_dir, count)
        count += int(images.shape[0])
        if count >= total_images:
            break
    if count == 0:
        raise RuntimeError(f"No real images were exported from split={split!r}")
    return count


@torch.no_grad()
def export_generated_images(
    config: Any,
    checkpoint_path: Path,
    output_dir: Path,
    total_images: int,
    batch_size: int,
    sample_steps: int,
    prefix_budget: int,
    eta: float,
    clip_x0: bool,
    seed: int,
    device: torch.device,
) -> int:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = DiffusionSchedule(config.diffusion, device=device)
    budget = config.model.token_count if prefix_budget < 1 else prefix_budget
    if budget > config.model.token_count:
        raise ValueError(f"--prefix-budget must be <= token_count ({config.model.token_count})")

    generator = torch.Generator(device=device).manual_seed(seed)
    count = 0
    while count < total_images:
        current = min(batch_size, total_images - count)
        samples = ddim_sample(
            model=model,
            schedule=schedule,
            shape=(current, config.model.image_channels, config.model.image_size, config.model.image_size),
            sample_steps=sample_steps,
            prefix_budget=budget,
            eta=eta,
            clip_x0=clip_x0,
            device=device,
            generator=generator,
        )
        save_png_batch(samples, output_dir, count)
        count += current
    return count


def official_fid_commands(real_dir: Path, generated_dir: Path, report_dir: Path, batch_size: int) -> dict[str, str]:
    return {
        "cofitok_wrapper": (
            "PYTHONPATH=src python scripts/evaluate_official_fid_dirs.py "
            f"--real-dir {real_dir.as_posix()} --generated-dir {generated_dir.as_posix()} "
            f"--output-dir {report_dir.as_posix()} --batch-size {batch_size} --require-pytorch-fid"
        ),
        "pytorch_fid_cli": (
            "python -m pytorch_fid "
            f"{real_dir.as_posix()} {generated_dir.as_posix()} --batch-size {batch_size}"
        ),
    }


def main() -> None:
    args = parse_args()
    _validate_positive("--batch-size", args.batch_size)
    if args.mode in {"both", "real"}:
        _validate_positive("--real-count", args.real_count)
    if args.mode in {"both", "generated"}:
        _validate_positive("--sample-count", args.sample_count)
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --mode is generated or both")

    config = load_config(args.config)
    seed = config.runtime.seed if args.seed < 0 else args.seed
    seed_everything(seed)
    device = _resolve_device(config.runtime.device)
    output_dir = Path(args.output_dir)
    real_dir = output_dir / "real"
    generated_dir = output_dir / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"both", "real"}:
        prepare_image_dir(real_dir, overwrite=args.overwrite)
    if args.mode in {"both", "generated"}:
        prepare_image_dir(generated_dir, overwrite=args.overwrite)

    start = time.time()
    real_count = len(_image_files(real_dir))
    generated_count = len(_image_files(generated_dir))
    if args.mode in {"both", "real"}:
        real_count = export_real_images(
            config=config,
            output_dir=real_dir,
            split=args.split,
            total_images=args.real_count,
            batch_size=args.batch_size,
            device=device,
        )
    if args.mode in {"both", "generated"}:
        generated_count = export_generated_images(
            config=config,
            checkpoint_path=Path(args.checkpoint),
            output_dir=generated_dir,
            total_images=args.sample_count,
            batch_size=args.batch_size,
            sample_steps=args.sample_steps,
            prefix_budget=args.prefix_budget,
            eta=args.eta,
            clip_x0=args.clip_x0,
            seed=seed,
            device=device,
        )

    report_dir = output_dir / "official_fid_report"
    manifest = {
        "schema_version": 1,
        "status": "ready_for_external_fid",
        "config": config_to_dict(config),
        "checkpoint": args.checkpoint,
        "paths": {
            "export_root": output_dir.as_posix(),
            "real_dir": real_dir.as_posix(),
            "generated_dir": generated_dir.as_posix(),
            "suggested_report_dir": report_dir.as_posix(),
        },
        "counts": {
            "real_image_count": real_count,
            "generated_image_count": generated_count,
        },
        "sampling": {
            "sample_steps": args.sample_steps,
            "prefix_budget": config.model.token_count if args.prefix_budget < 1 else args.prefix_budget,
            "eta": args.eta,
            "clip_x0": args.clip_x0,
            "seed": seed,
        },
        "external_fid": {
            "preferred_package": "pytorch-fid",
            "install_hint": "pip install pytorch-fid",
            "commands": official_fid_commands(real_dir, generated_dir, report_dir, args.batch_size),
        },
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    manifest_path = output_dir / "official_fid_export_manifest.json"
    write_json_report(manifest_path, manifest)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
