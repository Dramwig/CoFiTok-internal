from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation import (
    SAMPLING_MANIFEST_SCHEMA_VERSION,
    SAMPLING_REPORT_SCHEMA_VERSION,
    sampling_protocol_contract,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.image_integrity import (
    IMAGE_TREE_DIGEST_SCHEMA,
    image_tree_sha256,
    is_valid_png,
    sample_set_sha256,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_METRICS_REPORT_SCHEMA_VERSION = 3
GENERATION_METRICS_REPORT_ROLE = "generation_directory_metrics_report"
GENERATION_METRICS_REPORT_FILENAME = "generation_metrics_report.json"


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse an exactly bound completed metrics report after revalidating "
            "the physical real/generated sets and sampling evidence."
        ),
    )
    return parser.parse_args()


def torch_fidelity_version() -> str:
    module = importlib.import_module("torch_fidelity")
    version = str(getattr(module, "__version__", ""))
    if not version:
        raise ValueError("torch-fidelity version is unavailable")
    return version


def _prepare_output_directory(
    output_dir: str | Path,
    *,
    resume: bool,
) -> tuple[Path, Path]:
    output = reject_symlink_chain(
        output_dir,
        name="generation metrics output directory",
    )
    if output.exists() and not output.is_dir():
        raise ValueError(f"Generation metrics output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    report_path = output / GENERATION_METRICS_REPORT_FILENAME
    unexpected = sorted(
        path.name
        for path in output.iterdir()
        if path.name != GENERATION_METRICS_REPORT_FILENAME
    )
    if unexpected:
        raise ValueError(
            "Generation metrics output contains unexpected files: "
            + ", ".join(unexpected)
        )
    reject_symlink_chain(report_path, name="generation metrics report")
    if report_path.exists() and not report_path.is_file():
        raise ValueError(
            f"Generation metrics report is not a regular file: {report_path}"
        )
    if report_path.is_file() and not resume:
        raise FileExistsError(
            "Generation metrics evidence already exists; pass --resume to validate it"
        )
    return output, report_path


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Completed generation metric {name} is missing") from error
    if not math.isfinite(value):
        raise ValueError(f"Completed generation metric {name} is not finite")
    return value


def _validate_completed_report(
    report: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != GENERATION_METRICS_REPORT_SCHEMA_VERSION
        or report.get("role") != GENERATION_METRICS_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
    ):
        raise ValueError("Completed generation metrics report contract differs")
    for field in (
        "git",
        "runtime_environment",
        "runtime_environment_sha256",
        "implementation",
        "paths",
        "counts",
        "real_set",
        "sample_provenance",
        "parameters",
    ):
        if report.get(field) != expected.get(field):
            raise ValueError(f"Completed generation metrics report {field} differs")
    metrics = report.get("metrics")
    runtime = report.get("runtime")
    if not isinstance(metrics, dict) or not isinstance(runtime, dict):
        raise ValueError("Completed generation metrics report is malformed")
    fid = _metric_value(metrics, "frechet_inception_distance")
    inception_mean = _metric_value(metrics, "inception_score_mean")
    inception_std = _metric_value(metrics, "inception_score_std")
    if fid < 0.0 or inception_mean <= 0.0 or inception_std < 0.0:
        raise ValueError(
            "Completed generation distribution metrics are outside their domains"
        )
    if expected["parameters"]["precision_recall_enabled"]:
        precision = _metric_value(metrics, "precision")
        recall = _metric_value(metrics, "recall")
        if not 0.0 <= precision <= 1.0 or not 0.0 <= recall <= 1.0:
            raise ValueError("Completed generation precision/recall is outside [0, 1]")
    try:
        elapsed_seconds = float(runtime["elapsed_seconds"])
        digest_seconds = float(runtime["real_set_digest_elapsed_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Completed generation metrics runtime is malformed") from error
    if (
        not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
        or not math.isfinite(digest_seconds)
        or digest_seconds < 0.0
        or digest_seconds > elapsed_seconds
        or runtime.get("torch_version") != torch.__version__
        or runtime.get("cuda_available") != torch.cuda.is_available()
    ):
        raise ValueError("Completed generation metrics runtime evidence differs")


def find_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )


def content_addressed_real_cache_name(base_name: str, real_set_sha256: str) -> str:
    if not base_name.strip():
        raise ValueError("real cache base name must not be empty")
    if len(real_set_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in real_set_sha256
    ):
        raise ValueError("real-set SHA256 is malformed")
    return f"{base_name}__cofitok_{real_set_sha256[:16]}"


def validate_sampling_provenance(
    report_path: Path,
    generated_dir: Path,
    generated_images: list[Path],
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Sampling report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if int(report.get("schema_version", -1)) != SAMPLING_REPORT_SCHEMA_VERSION:
        raise ValueError("Sampling report schema is unsupported")
    if report.get("status") != "completed":
        raise ValueError(f"Sampling report is not complete: {report_path}")
    sampling = report["sampling"]
    protocol_contract = sampling_protocol_contract(sampling)
    if protocol_contract["valid"] is not True:
        raise ValueError(
            "Sampling protocol is invalid: " + ", ".join(protocol_contract["issues"])
        )
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
    runtime_environment = report.get("runtime_environment")
    if not isinstance(runtime_environment, dict):
        raise ValueError("Sampling report runtime environment is missing")
    sampling_environment_sha = runtime_environment_sha256(runtime_environment)
    if report.get("runtime_environment_sha256") != sampling_environment_sha:
        raise ValueError("Sampling report runtime environment SHA256 differs")
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
    sampling_manifest = json.loads(sampling_manifest_path.read_text(encoding="utf-8"))
    if int(sampling_manifest.get("schema_version", -1)) != SAMPLING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Sampling manifest schema is unsupported")
    if (
        sampling_manifest.get("runtime_environment") != runtime_environment
        or sampling_manifest.get("runtime_environment_sha256")
        != sampling_environment_sha
    ):
        raise ValueError("Sampling manifest runtime environment differs from report")
    for field in (
        "git",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_integrity_manifest",
        "checkpoint_step",
        "weights",
        "sampling",
        "output_dirs",
    ):
        if sampling_manifest.get(field) != report.get(field):
            raise ValueError(f"Sampling report {field} differs from immutable manifest")
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
        "report_identity": file_identity(report_path),
        "manifest_identity": file_identity(sampling_manifest_path),
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_integrity_manifest": checkpoint_integrity_manifest,
        "checkpoint_step": int(report["checkpoint_step"]),
        "weights": report["weights"],
        "git": git,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": sampling_environment_sha,
        "selected_prefix_budget": selected_budget,
        "image_shape": image_shape,
        "sample_set_sha256": actual_sample_sha256,
        "sampling_progress": {
            "report": progress_path.resolve().as_posix(),
            "identity": file_identity(progress_path),
            "status": progress["status"],
            "invocation": int(progress["invocation"]),
            "completed_samples": int(progress["completed_samples"]),
            "cumulative_elapsed_seconds": float(progress["cumulative_elapsed_seconds"]),
        },
        "sampling": sampling,
        "sampling_protocol_contract": protocol_contract,
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


def _run_generation_metrics(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.prc_batch_size < 1 or args.min_samples < 1:
        raise ValueError("batch-size, prc-batch-size, and min-samples must be positive")
    real_dir = reject_symlink_chain(
        args.real_dir,
        name="generation metrics real directory",
    )
    generated_dir = reject_symlink_chain(
        args.generated_dir,
        name="generation metrics generated directory",
    )
    output_dir, report_path = _prepare_output_directory(
        args.output_dir,
        resume=args.resume,
    )
    real_images = find_images(real_dir)
    generated_images = find_images(generated_dir)
    if len(real_images) < args.min_samples:
        raise ValueError(f"real image count {len(real_images)} is below {args.min_samples}")
    if len(generated_images) < args.min_samples:
        raise ValueError(
            f"generated image count {len(generated_images)} is below {args.min_samples}"
        )
    sampling_report_path = (
        reject_symlink_chain(
            args.sampling_report,
            name="generation metrics sampling report",
        )
        if args.sampling_report
        else reject_symlink_chain(
            generated_dir.parent / "sampling_report.json",
            name="generation metrics sampling report",
        )
    )
    sample_provenance = validate_sampling_provenance(
        sampling_report_path,
        generated_dir,
        generated_images,
    )
    evaluator_git = git_provenance(PROJECT_ROOT)
    cuda = torch.cuda.is_available() and not args.cpu
    evaluator_device = torch.device("cuda" if cuda else "cpu")
    evaluator_environment = capture_runtime_environment(
        evaluator_device,
        project_root=PROJECT_ROOT,
    )
    evaluator_environment_sha = runtime_environment_sha256(evaluator_environment)
    start = time.time()
    real_set_sha = image_tree_sha256(real_images, root=real_dir)
    real_digest_elapsed_seconds = time.time() - start
    effective_real_cache_name = content_addressed_real_cache_name(
        args.real_cache_name,
        real_set_sha,
    )
    cache_root = (
        reject_symlink_chain(args.cache_root, name="generation metrics cache root")
        .resolve()
        .as_posix()
        if args.cache_root
        else ""
    )
    implementation_version = torch_fidelity_version()
    parameters = {
        "batch_size": args.batch_size,
        "prc_batch_size": args.prc_batch_size,
        "min_samples": args.min_samples,
        "seed": args.seed,
        "cuda": cuda,
        "cpu_requested": bool(args.cpu),
        "samples_find_deep": True,
        "samples_shuffle": False,
        "cache_root": cache_root,
        "requested_real_cache_name": args.real_cache_name,
        "real_cache_name": effective_real_cache_name,
        "precision_recall_enabled": not args.skip_prc,
    }
    expected = {
        "git": evaluator_git,
        "runtime_environment": evaluator_environment,
        "runtime_environment_sha256": evaluator_environment_sha,
        "implementation": {
            "package": "torch_fidelity",
            "version": implementation_version,
        },
        "paths": {
            "real_dir": real_dir.resolve().as_posix(),
            "generated_dir": generated_dir.resolve().as_posix(),
            "sampling_report": sampling_report_path.resolve().as_posix(),
            "output_dir": output_dir.as_posix(),
            "report": report_path.as_posix(),
        },
        "counts": {
            "real_image_count": len(real_images),
            "generated_image_count": len(generated_images),
        },
        "real_set": {
            "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
            "sha256": real_set_sha,
            "root": real_dir.resolve().as_posix(),
            "image_count": len(real_images),
        },
        "sample_provenance": sample_provenance,
        "parameters": parameters,
    }
    if report_path.is_file():
        existing_report = read_json_object(
            report_path,
            name="generation metrics report",
        )
        _validate_completed_report(existing_report, expected=expected)
        print(f"reused completed generation metrics {report_path}")
        return
    metrics, version = calculate_metrics(
        real_dir=real_dir,
        generated_dir=generated_dir,
        batch_size=args.batch_size,
        prc_batch_size=args.prc_batch_size,
        seed=args.seed,
        cuda=cuda,
        cache_root=cache_root,
        real_cache_name=effective_real_cache_name,
        prc=not args.skip_prc,
    )
    if version != implementation_version:
        raise ValueError("torch-fidelity version changed during metric evaluation")
    report = {
        "schema_version": GENERATION_METRICS_REPORT_SCHEMA_VERSION,
        "role": GENERATION_METRICS_REPORT_ROLE,
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        **expected,
        "metrics": metrics,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "real_set_digest_elapsed_seconds": real_digest_elapsed_seconds,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(
        args.output_dir,
        role="generation_metrics_evaluation",
    ):
        _run_generation_metrics(args)


if __name__ == "__main__":
    main()
