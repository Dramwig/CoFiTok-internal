from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

import torch

from cofitok.reporting import write_json_report


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated images with torch-fidelity FID/IS/precision/recall."
    )
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prc-batch-size", type=int, default=10_000)
    parser.add_argument("--min-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--real-cache-name", default="imagenet256_val_50k_torch_fidelity_v04")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-prc", action="store_true")
    return parser.parse_args()


def find_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def calculate_metrics(
    *,
    real_dir: Path,
    generated_dir: Path,
    batch_size: int,
    prc_batch_size: int,
    seed: int,
    cuda: bool,
    cache_root: str,
    real_cache_name: str,
    prc: bool,
) -> tuple[dict[str, float], str]:
    module = importlib.import_module("torch_fidelity")
    kwargs: dict[str, Any] = {
        "input1": generated_dir.as_posix(),
        "input2": real_dir.as_posix(),
        "cuda": cuda,
        "batch_size": batch_size,
        "isc": True,
        "fid": True,
        "kid": False,
        "prc": prc,
        "prc_batch_size": prc_batch_size,
        "samples_find_deep": True,
        "samples_shuffle": False,
        "rng_seed": seed,
        "save_cpu_ram": True,
        "cache": True,
        "input2_cache_name": real_cache_name,
        "verbose": True,
    }
    if cache_root:
        kwargs["cache_root"] = cache_root
    metrics = module.calculate_metrics(**kwargs)
    return {str(key): float(value) for key, value in metrics.items()}, str(module.__version__)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.prc_batch_size < 1 or args.min_samples < 1:
        raise ValueError("batch-size, prc-batch-size, and min-samples must be positive")
    real_dir = Path(args.real_dir)
    generated_dir = Path(args.generated_dir)
    output_dir = Path(args.output_dir)
    real_images = find_images(real_dir)
    generated_images = find_images(generated_dir)
    if len(real_images) < args.min_samples:
        raise ValueError(f"real image count {len(real_images)} is below {args.min_samples}")
    if len(generated_images) < args.min_samples:
        raise ValueError(f"generated image count {len(generated_images)} is below {args.min_samples}")
    cuda = torch.cuda.is_available() and not args.cpu
    start = time.time()
    metrics, version = calculate_metrics(
        real_dir=real_dir,
        generated_dir=generated_dir,
        batch_size=args.batch_size,
        prc_batch_size=args.prc_batch_size,
        seed=args.seed,
        cuda=cuda,
        cache_root=args.cache_root,
        real_cache_name=args.real_cache_name,
        prc=not args.skip_prc,
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "implementation": {
            "package": "torch_fidelity",
            "version": version,
        },
        "paths": {
            "real_dir": real_dir.resolve().as_posix(),
            "generated_dir": generated_dir.resolve().as_posix(),
        },
        "counts": {
            "real_image_count": len(real_images),
            "generated_image_count": len(generated_images),
        },
        "parameters": {
            "batch_size": args.batch_size,
            "prc_batch_size": args.prc_batch_size,
            "seed": args.seed,
            "cuda": cuda,
            "samples_find_deep": True,
            "samples_shuffle": False,
            "real_cache_name": args.real_cache_name,
            "precision_recall_enabled": not args.skip_prc,
        },
        "metrics": metrics,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "generation_metrics_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

