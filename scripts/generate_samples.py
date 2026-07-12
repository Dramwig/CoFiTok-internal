from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from cofitok.configs import config_from_dict
from cofitok.diffusion import DiffusionSchedule, ddim_sample, select_sampling_timesteps
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.runtime import autocast_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible CoFiTok samples from EMA weights.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--prefix-budgets", default="", help="Comma-separated token budgets; default K.")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_budgets(raw: str, token_count: int) -> list[int]:
    if not raw.strip():
        return [token_count]
    budgets = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if any(budget < 1 or budget > token_count for budget in budgets):
        raise ValueError("prefix budget is outside the model token range")
    return budgets


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(start: int, count: int, num_classes: int, device: torch.device) -> torch.Tensor | None:
    if num_classes <= 0:
        return None
    return torch.arange(start, start + count, device=device, dtype=torch.long) % num_classes


def _save_batch(images: torch.Tensor, directory: Path, start: int, overwrite: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        path = directory / f"{start + offset:06d}.png"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing sample {path}")
        save_image((image.float().clamp(-1.0, 1.0) + 1.0) * 0.5, path)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint sampling requested but CUDA is unavailable")
    device = torch.device(config.runtime.device)
    model = CoFiTokTiny(config.model).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.weights == "ema":
        ema = ExponentialMovingAverage(
            model,
            decay=config.optimization.ema_decay,
            warmup_steps=config.optimization.ema_warmup_steps,
        )
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()
    schedule = DiffusionSchedule(config.diffusion, device=device)
    budgets = _parse_budgets(args.prefix_budgets, config.model.token_count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_index = args.start_index + args.num_samples
    start_time = time.time()

    for batch_start in range(args.start_index, stop_index, args.batch_size):
        count = min(args.batch_size, stop_index - batch_start)
        labels = _labels(batch_start, count, config.model.num_classes, device)
        for budget in budgets:
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1_000_003 * budget + batch_start
            )
            with autocast_context(device, args.precision):
                samples = ddim_sample(
                    model,
                    schedule,
                    (count, config.model.image_channels, config.model.image_size, config.model.image_size),
                    sample_steps=args.sample_steps,
                    prefix_budget=budget,
                    eta=args.eta,
                    clip_x0=True,
                    device=device,
                    generator=generator,
                    class_labels=labels,
                    guidance_scale=args.guidance_scale,
                    guidance_rescale=args.guidance_rescale,
                )
            _save_batch(samples.cpu(), output_dir / f"prefix_{budget}", batch_start, args.overwrite)
        completed = batch_start + count - args.start_index
        print(f"generated {completed}/{args.num_samples}")

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "sampling": {
            "num_samples": args.num_samples,
            "start_index": args.start_index,
            "batch_size": args.batch_size,
            "sample_steps": args.sample_steps,
            "actual_timesteps": select_sampling_timesteps(
                schedule.num_train_timesteps,
                args.sample_steps,
            ),
            "prefix_budgets": budgets,
            "guidance_scale": args.guidance_scale,
            "guidance_rescale": args.guidance_rescale,
            "eta": args.eta,
            "seed": args.seed,
            "precision": args.precision,
            "class_schedule": "balanced_modulo" if config.model.num_classes > 0 else None,
        },
        "output_dirs": {str(budget): str((output_dir / f"prefix_{budget}").resolve()) for budget in budgets},
        "elapsed_seconds": time.time() - start_time,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    write_json_report(output_dir / "sampling_report.json", report)
    print(f"wrote {output_dir / 'sampling_report.json'}")


if __name__ == "__main__":
    main()
