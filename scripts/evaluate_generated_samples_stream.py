from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from cofitok.configs import load_config
from cofitok.diffusion import DiffusionSchedule
from cofitok.metrics import frechet_distance_from_features, lowres_image_features
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.utils.seed import seed_everything
from scripts.evaluate_generated_samples import _collect_reference_features, _inception_features, _make_inception
from scripts.sample_checkpoint import ddim_sample


def _ensure_project_torch_home(checkpoint_path: Path) -> None:
    if os.environ.get("TORCH_HOME"):
        return
    for parent in checkpoint_path.resolve().parents:
        if parent.name == "checkpoints":
            os.environ["TORCH_HOME"] = str(parent / "torch_cache")
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream generated samples from a checkpoint and evaluate sample-vs-real Frechet metrics."
    )
    parser.add_argument("--config", required=True, help="Config defining the model and reference dataset.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated-quality report.")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--max-real-images", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--prefix-budget", type=int, default=-1, help="Default uses the full token budget.")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--clip-x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-size", type=int, default=8)
    parser.add_argument("--enable-inception-fid", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_generated_features(
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    image_shape: tuple[int, int, int],
    total_samples: int,
    batch_size: int,
    sample_steps: int,
    prefix_budget: int,
    eta: float,
    clip_x0: bool,
    feature_size: int,
    device: torch.device,
    generator: torch.Generator,
    inception_model: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    lowres_features = []
    inception_features = []
    remaining = total_samples
    while remaining > 0:
        current = min(batch_size, remaining)
        samples = ddim_sample(
            model=model,
            schedule=schedule,
            shape=(current, *image_shape),
            sample_steps=sample_steps,
            prefix_budget=prefix_budget,
            eta=eta,
            clip_x0=clip_x0,
            device=device,
            generator=generator,
        )
        lowres_features.append(lowres_image_features(samples, feature_size=feature_size).cpu())
        if inception_model is not None:
            inception_features.append(_inception_features(inception_model, samples))
        remaining -= current

    return (
        torch.cat(lowres_features, dim=0),
        torch.cat(inception_features, dim=0) if inception_features else None,
        total_samples,
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.sample_count < 1:
        raise ValueError("--sample-count must be >= 1")
    if args.max_real_images < 1:
        raise ValueError("--max-real-images must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    config = load_config(args.config)
    seed = config.runtime.seed if args.seed < 0 else args.seed
    seed_everything(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.runtime.device)
    _ensure_project_torch_home(Path(args.checkpoint))

    model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    schedule = DiffusionSchedule(config.diffusion, device=device)
    prefix_budget = config.model.token_count if args.prefix_budget < 1 else args.prefix_budget
    if prefix_budget > config.model.token_count:
        raise ValueError(f"--prefix-budget must be <= token_count ({config.model.token_count})")

    inception_model = None
    inception_status: dict[str, Any] = {"available": False, "reason": "not requested"}
    if args.enable_inception_fid:
        inception_model, inception_status = _make_inception(device)

    start = time.time()
    reference_lowres, reference_inception, real_count = _collect_reference_features(
        config=config,
        split=args.split,
        max_images=args.max_real_images,
        batch_size=args.batch_size,
        feature_size=args.feature_size,
        device=device,
        inception_model=inception_model,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    sample_lowres, sample_inception, sample_count = collect_generated_features(
        model=model,
        schedule=schedule,
        image_shape=(config.model.image_channels, config.model.image_size, config.model.image_size),
        total_samples=args.sample_count,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        prefix_budget=prefix_budget,
        eta=args.eta,
        clip_x0=args.clip_x0,
        feature_size=args.feature_size,
        device=device,
        generator=generator,
        inception_model=inception_model,
    )

    metrics = {
        "lowres_frechet_proxy": frechet_distance_from_features(reference_lowres, sample_lowres),
    }
    if reference_inception is not None and sample_inception is not None:
        metrics["inception_frechet"] = frechet_distance_from_features(reference_inception, sample_inception)

    report = {
        "config_name": config.name,
        "dataset": config.data.dataset,
        "checkpoint": args.checkpoint,
        "samples_dir": "streamed_from_checkpoint",
        "sampling": {
            "sample_steps": args.sample_steps,
            "prefix_budget": prefix_budget,
            "eta": args.eta,
            "clip_x0": args.clip_x0,
            "batch_size": args.batch_size,
        },
        "evaluation": {
            "split": args.split,
            "real_image_count": real_count,
            "sample_image_count": sample_count,
            "available_sample_images": sample_count,
            "feature_size": args.feature_size,
            "seed": seed,
        },
        "metrics": metrics,
        "metric_notes": {
            "lowres_frechet_proxy": "Small-scale relative metric over deterministic low-resolution RGB/color features.",
            "inception_frechet": inception_status,
        },
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "streamed_samples": True,
        },
    }
    report_path = output_dir / "generated_quality_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
