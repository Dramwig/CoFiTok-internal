from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cofitok.reporting import write_json_report
from scripts.evaluate_generated_samples import find_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an external pytorch-fid evaluation over already-exported real/generated image directories."
    )
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect directories and package availability.")
    parser.add_argument("--require-pytorch-fid", action="store_true", help="Exit nonzero if pytorch-fid is unavailable.")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _pytorch_fid_status() -> tuple[Any | None, dict[str, Any]]:
    try:
        module = importlib.import_module("pytorch_fid.fid_score")
    except Exception as error:
        return None, {"available": False, "reason": repr(error), "package": "pytorch-fid"}
    return module, {"available": True, "package": "pytorch-fid"}


def evaluate_with_pytorch_fid(
    real_dir: Path,
    generated_dir: Path,
    batch_size: int,
    device: str,
    dims: int,
    num_workers: int,
) -> tuple[float | None, dict[str, Any]]:
    module, status = _pytorch_fid_status()
    if module is None:
        return None, status
    value = module.calculate_fid_given_paths(
        [real_dir.as_posix(), generated_dir.as_posix()],
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=num_workers,
    )
    return float(value), status


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.dims < 1:
        raise ValueError("--dims must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    real_dir = Path(args.real_dir)
    generated_dir = Path(args.generated_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    real_paths = find_image_files(real_dir)
    generated_paths = find_image_files(generated_dir)
    device = _resolve_device(args.device)

    metric_value = None
    implementation_status: dict[str, Any]
    if args.dry_run:
        _, implementation_status = _pytorch_fid_status()
        status = "dry_run"
    else:
        metric_value, implementation_status = evaluate_with_pytorch_fid(
            real_dir=real_dir,
            generated_dir=generated_dir,
            batch_size=args.batch_size,
            device=device,
            dims=args.dims,
            num_workers=args.num_workers,
        )
        status = "ok" if metric_value is not None else "unavailable"

    report = {
        "schema_version": 1,
        "status": status,
        "metric": "fid",
        "implementation": implementation_status,
        "paths": {
            "real_dir": real_dir.as_posix(),
            "generated_dir": generated_dir.as_posix(),
        },
        "counts": {
            "real_image_count": len(real_paths),
            "generated_image_count": len(generated_paths),
        },
        "parameters": {
            "batch_size": args.batch_size,
            "dims": args.dims,
            "num_workers": args.num_workers,
            "device": device,
            "dry_run": args.dry_run,
        },
        "metrics": {
            "fid": metric_value,
        },
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    report_path = output_dir / "official_fid_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")
    if args.require_pytorch_fid and metric_value is None and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
