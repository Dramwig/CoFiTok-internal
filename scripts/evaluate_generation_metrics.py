from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from cofitok.image_integrity import is_valid_png, sample_set_sha256
from cofitok.reporting import file_sha256, git_provenance, write_json_report


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated images with torch-fidelity FID/IS/precision/recall."
    )
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument(
        "--sampling-report",
        default="",
        help="Sampling report to verify; defaults to the generated directory parent.",
    )
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


def validate_sampling_provenance(
    report_path: Path,
    generated_dir: Path,
    generated_images: list[Path],
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Sampling report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "completed":
        raise ValueError(f"Sampling report is not complete: {report_path}")
    sampling = report["sampling"]
    if int(sampling["start_index"]) != 0:
        raise ValueError("Formal generation metrics require a sample set starting at index zero")
    expected_count = int(sampling["num_samples"])
    if len(generated_images) != expected_count:
        raise ValueError(
            f"generated image count {len(generated_images)} does not match sampling report "
            f"count {expected_count}"
        )
    expected_names = {f"{index:06d}.png" for index in range(expected_count)}
    actual_names = {path.name for path in generated_images}
    if actual_names != expected_names:
        raise ValueError("Generated sample filenames are not the complete zero-based numbered set")
    image_shape = sampling.get("image_shape")
    if not isinstance(image_shape, list) or len(image_shape) != 3:
        raise ValueError("Sampling report is missing a [C, H, W] image_shape")
    channels, height, width = (int(value) for value in image_shape)
    invalid_images = [
        path
        for path in generated_images
        if not is_valid_png(path, width=width, height=height, channels=channels)
    ]
    if invalid_images:
        raise ValueError(
            f"Generated sample integrity failed for {len(invalid_images)} images; "
            f"first={invalid_images[0]}"
        )
    resolved_generated = generated_dir.resolve()
    matching_budgets = [
        int(budget)
        for budget, directory in report["output_dirs"].items()
        if Path(directory).resolve() == resolved_generated
    ]
    if len(matching_budgets) != 1:
        raise ValueError("Generated directory does not map to exactly one sampling-report budget")
    selected_budget = matching_budgets[0]
    sample_set = report.get("sample_sets", {}).get(str(selected_budget))
    if not isinstance(sample_set, dict):
        raise ValueError("Sampling report is missing the selected sample-set digest")
    if int(sample_set.get("count", -1)) != expected_count:
        raise ValueError("Sampling report sample-set count does not match num_samples")
    expected_sample_sha256 = str(sample_set.get("sha256", ""))
    if len(expected_sample_sha256) != 64:
        raise ValueError("Sampling report sample-set SHA256 is malformed")
    actual_sample_sha256 = sample_set_sha256(generated_images)
    if actual_sample_sha256 != expected_sample_sha256:
        raise ValueError("Generated sample-set SHA256 does not match the sampling report")
    git = report.get("git", {})
    if len(str(git.get("revision", ""))) != 40:
        raise ValueError("Sampling report Git revision is malformed")
    if not isinstance(git.get("tracked_dirty"), bool):
        raise ValueError("Sampling report tracked-dirty state is missing")
    checkpoint_sha256 = str(report["checkpoint_sha256"])
    if len(checkpoint_sha256) != 64:
        raise ValueError("Sampling report checkpoint SHA256 is malformed")
    checkpoint_integrity_manifest = str(report.get("checkpoint_integrity_manifest", ""))
    expected_integrity_name = f"{Path(report['checkpoint']).name}.integrity.json"
    if Path(checkpoint_integrity_manifest).name != expected_integrity_name:
        raise ValueError("Sampling report checkpoint integrity manifest is malformed")
    sampling_manifest_path = report_path.parent / "sampling_manifest.json"
    if not sampling_manifest_path.is_file():
        raise FileNotFoundError("Sampling manifest is missing beside the sampling report")
    sampling_manifest_sha256 = file_sha256(sampling_manifest_path)
    if report.get("sampling_manifest_sha256") != sampling_manifest_sha256:
        raise ValueError("Sampling report does not match the immutable sampling manifest")
    progress_path = Path(str(report.get("sampling_progress", "")))
    expected_progress_path = report_path.parent / "sampling_progress.json"
    if progress_path.resolve() != expected_progress_path.resolve() or not progress_path.is_file():
        raise ValueError("Sampling progress path is missing or outside the sample run")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError("Sampling progress is not completed")
    if progress.get("sampling_manifest_sha256") != sampling_manifest_sha256:
        raise ValueError("Sampling progress belongs to another manifest")
    if int(progress.get("completed_samples", -1)) != expected_count:
        raise ValueError("Sampling progress completed count does not match the sample set")
    if progress.get("prefix_budgets") != sampling.get("prefix_budgets"):
        raise ValueError("Sampling progress prefix budgets do not match the sampling report")
    if progress.get("sample_sets") != report.get("sample_sets"):
        raise ValueError("Sampling progress digests do not match the sampling report")
    return {
        "report": report_path.resolve().as_posix(),
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_integrity_manifest": checkpoint_integrity_manifest,
        "checkpoint_step": int(report["checkpoint_step"]),
        "weights": report["weights"],
        "git": git,
        "selected_prefix_budget": selected_budget,
        "image_shape": image_shape,
        "sample_set_sha256": actual_sample_sha256,
        "sampling_progress": {
            "report": progress_path.resolve().as_posix(),
            "status": progress["status"],
            "invocation": int(progress["invocation"]),
            "completed_samples": int(progress["completed_samples"]),
            "cumulative_elapsed_seconds": float(progress["cumulative_elapsed_seconds"]),
        },
        "sampling": sampling,
    }


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
    sampling_report_path = (
        Path(args.sampling_report)
        if args.sampling_report
        else generated_dir.parent / "sampling_report.json"
    )
    sample_provenance = validate_sampling_provenance(
        sampling_report_path,
        generated_dir,
        generated_images,
    )
    evaluator_git = git_provenance(PROJECT_ROOT)
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
        "git": evaluator_git,
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
        "sample_provenance": sample_provenance,
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
