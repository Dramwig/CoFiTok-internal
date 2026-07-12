from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from cofitok.configs import load_config
from cofitok.data import build_dataloader
from cofitok.metrics import frechet_distance_from_features, lowres_image_features
from cofitok.reporting import write_json_report
from cofitok.utils.seed import seed_everything

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated sample distribution against a dataset split.")
    parser.add_argument("--config", required=True, help="Config defining the reference dataset.")
    parser.add_argument("--samples-dir", required=True, help="Directory containing generated sample images.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated-quality report.")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--max-real-images", type=int, default=256)
    parser.add_argument("--max-sample-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-size", type=int, default=8)
    parser.add_argument("--enable-inception-fid", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def find_image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Sample directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No sample images found in: {root}")
    return files


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_images(batch: object, device: torch.device) -> torch.Tensor:
    images = batch[0] if isinstance(batch, (list, tuple)) else batch
    return images.to(device=device, dtype=torch.float32)


def _sample_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )


def _load_sample_batch(paths: list[Path], image_size: int, device: torch.device) -> torch.Tensor:
    transform = _sample_transform(image_size)
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images).to(device=device, dtype=torch.float32)


def _make_inception(device: torch.device) -> tuple[nn.Module | None, dict[str, Any]]:
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except Exception as error:  # pragma: no cover - optional dependency
        return None, {"available": False, "reason": repr(error)}
    try:
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        model.fc = nn.Identity()
        model = model.to(device).eval()
    except Exception as error:  # pragma: no cover - optional weight download
        return None, {"available": False, "reason": repr(error)}
    return model, {"available": True, "model": "torchvision.models.inception_v3/IMAGENET1K_V1"}


def _inception_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    images = (images.detach().clamp(-1.0, 1.0) + 1.0) * 0.5
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    features = model((images - mean) / std)
    if isinstance(features, tuple):
        features = features[0]
    return features.detach().to(dtype=torch.float64).cpu()


@torch.no_grad()
def _collect_reference_features(
    config: Any,
    split: str,
    max_images: int,
    batch_size: int,
    feature_size: int,
    device: torch.device,
    inception_model: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    loader_config = config.data
    loader = build_dataloader(loader_config, split=split)
    lowres_features = []
    inception_features = []
    count = 0
    for batch in loader:
        if count >= max_images:
            break
        images = _batch_images(batch, device)
        images = images[: max_images - count]
        lowres_features.append(lowres_image_features(images, feature_size=feature_size).cpu())
        if inception_model is not None:
            inception_features.append(_inception_features(inception_model, images))
        count += int(images.shape[0])
        if count >= max_images:
            break
    if count == 0:
        raise RuntimeError("No reference images were evaluated")
    return (
        torch.cat(lowres_features, dim=0),
        torch.cat(inception_features, dim=0) if inception_features else None,
        count,
    )


@torch.no_grad()
def _collect_sample_features(
    sample_paths: list[Path],
    image_size: int,
    max_images: int,
    batch_size: int,
    feature_size: int,
    device: torch.device,
    inception_model: nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    selected = sample_paths[:max_images]
    lowres_features = []
    inception_features = []
    for start in range(0, len(selected), batch_size):
        images = _load_sample_batch(selected[start : start + batch_size], image_size=image_size, device=device)
        lowres_features.append(lowres_image_features(images, feature_size=feature_size).cpu())
        if inception_model is not None:
            inception_features.append(_inception_features(inception_model, images))
    return (
        torch.cat(lowres_features, dim=0),
        torch.cat(inception_features, dim=0) if inception_features else None,
        len(selected),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.max_real_images < 1 or args.max_sample_images < 1:
        raise ValueError("max image counts must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")

    config = load_config(args.config)
    seed = config.runtime.seed if args.seed < 0 else args.seed
    seed_everything(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.runtime.device)
    sample_paths = find_image_files(Path(args.samples_dir))

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
    sample_lowres, sample_inception, sample_count = _collect_sample_features(
        sample_paths=sample_paths,
        image_size=config.data.image_size,
        max_images=args.max_sample_images,
        batch_size=args.batch_size,
        feature_size=args.feature_size,
        device=device,
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
        "samples_dir": str(args.samples_dir),
        "evaluation": {
            "split": args.split,
            "real_image_count": real_count,
            "sample_image_count": sample_count,
            "available_sample_images": len(sample_paths),
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
        },
    }
    report_path = output_dir / "generated_quality_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
