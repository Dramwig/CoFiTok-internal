from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import ResNet50_Weights, resnet50

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
    CLASS_FIDELITY_REPORT_ROLE,
    CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
    ClassFidelityAccumulator,
    validate_class_fidelity_metrics,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
try:
    from scripts.evaluate_generation_metrics import (
        find_images,
        validate_sampling_provenance,
    )
except ModuleNotFoundError:
    from evaluate_generation_metrics import (
        find_images,
        validate_sampling_provenance,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FILENAME = "class_fidelity_report.json"
CLASSIFIER_NAME = CLASS_FIDELITY_CLASSIFIER_NAME
CLASSIFIER_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
CLASSIFIER_BYTES = CLASS_FIDELITY_CLASSIFIER_BYTES
CLASSIFIER_SHA256 = CLASS_FIDELITY_CLASSIFIER_SHA256
DEFAULT_CLASSIFIER_CHECKPOINT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/"
    "resnet50-11ad3fa6.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether balanced ImageNet class-conditional samples match "
            "their requested classes with a fixed source-bound classifier."
        )
    )
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--sampling-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--classifier-checkpoint",
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=10_000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Revalidate and reuse an exactly bound completed report.",
    )
    return parser.parse_args()


def _categories_sha256(categories: list[str]) -> str:
    encoded = json.dumps(
        categories,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classifier_identity(checkpoint: str | Path) -> dict[str, Any]:
    source = reject_symlink_chain(checkpoint, name="class-fidelity classifier")
    if not source.is_file():
        raise FileNotFoundError(f"class-fidelity classifier is missing: {source}")
    identity = file_identity(source)
    if (
        int(identity["bytes"]) != CLASSIFIER_BYTES
        or identity["sha256"] != CLASSIFIER_SHA256
    ):
        raise ValueError("class-fidelity classifier bytes or SHA256 differ")
    weights = ResNet50_Weights.IMAGENET1K_V2
    categories = list(weights.meta["categories"])
    if len(categories) != 1000 or len(set(categories)) != 1000:
        raise ValueError("class-fidelity classifier categories differ")
    transform = weights.transforms()
    return {
        "name": CLASSIFIER_NAME,
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_url": CLASSIFIER_URL,
        "weights_path": identity["path"],
        "weights_bytes": identity["bytes"],
        "weights_sha256": identity["sha256"],
        "num_classes": len(categories),
        "categories_sha256": _categories_sha256(categories),
        "preprocessing": {
            "resize_size": list(transform.resize_size),
            "crop_size": list(transform.crop_size),
            "mean": list(transform.mean),
            "std": list(transform.std),
            "interpolation": str(transform.interpolation.value),
            "antialias": bool(transform.antialias),
        },
    }


class _IndexedImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, paths: list[Path], *, num_classes: int) -> None:
        self.paths = paths
        self.num_classes = num_classes
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.paths[index]
        image = read_image(path.as_posix(), mode=ImageReadMode.RGB)
        requested_class = int(path.stem) % self.num_classes
        return self.transform(image), requested_class


def _load_classifier(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise ValueError("class-fidelity classifier checkpoint is malformed")
    model = resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def calculate_class_fidelity(
    *,
    images: list[Path],
    classifier_checkpoint: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = _IndexedImageDataset(images, num_classes=1000)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = _load_classifier(classifier_checkpoint, device)
    accumulator = ClassFidelityAccumulator(num_classes=1000)
    with torch.inference_mode():
        for batch, targets in loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            accumulator.update(model(batch), targets)
    return accumulator.finalize()


def _prepare_output_directory(
    output_dir: str | Path,
    *,
    resume: bool,
) -> tuple[Path, Path]:
    output = reject_symlink_chain(output_dir, name="class-fidelity output directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"class-fidelity output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    report_path = output / REPORT_FILENAME
    unexpected = sorted(path.name for path in output.iterdir() if path.name != REPORT_FILENAME)
    if unexpected:
        raise ValueError(
            "class-fidelity output contains unexpected files: " + ", ".join(unexpected)
        )
    reject_symlink_chain(report_path, name="class-fidelity report")
    if report_path.exists() and not report_path.is_file():
        raise ValueError("class-fidelity report is not a regular file")
    if report_path.is_file() and not resume:
        raise FileExistsError(
            "class-fidelity evidence already exists; pass --resume to validate it"
        )
    return output, report_path


def _validate_completed_report(
    report: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != CLASS_FIDELITY_REPORT_SCHEMA_VERSION
        or report.get("role") != CLASS_FIDELITY_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("protocol") != "torchvision_imagenet_class_fidelity"
    ):
        raise ValueError("completed class-fidelity report contract differs")
    for field in (
        "git",
        "runtime_environment",
        "runtime_environment_sha256",
        "paths",
        "classifier",
        "sample_provenance",
        "parameters",
    ):
        if report.get(field) != expected.get(field):
            raise ValueError(f"completed class-fidelity report {field} differs")
    metrics = report.get("metrics")
    runtime = report.get("runtime")
    if not isinstance(metrics, dict) or not isinstance(runtime, dict):
        raise ValueError("completed class-fidelity report is malformed")
    validate_class_fidelity_metrics(metrics)
    if int(metrics["sample_count"]) != int(expected["parameters"]["sample_count"]):
        raise ValueError("completed class-fidelity sample count differs")
    try:
        elapsed_seconds = float(runtime["elapsed_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("completed class-fidelity runtime is malformed") from error
    if (
        not math.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
        or runtime.get("torch_version") != torch.__version__
    ):
        raise ValueError("completed class-fidelity runtime differs")


def _run(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.num_workers < 0 or args.min_samples < 1:
        raise ValueError("class-fidelity batch/workers/sample arguments are invalid")
    generated_dir = reject_symlink_chain(
        args.generated_dir,
        name="class-fidelity generated directory",
    )
    sampling_report = reject_symlink_chain(
        args.sampling_report,
        name="class-fidelity sampling report",
    )
    output_dir, report_path = _prepare_output_directory(
        args.output_dir,
        resume=args.resume,
    )
    images = find_images(generated_dir)
    if len(images) < args.min_samples:
        raise ValueError(
            f"generated image count {len(images)} is below {args.min_samples}"
        )
    sample_provenance = validate_sampling_provenance(
        sampling_report,
        generated_dir,
        images,
    )
    sampling = sample_provenance["sampling"]
    if sampling.get("class_schedule") != "balanced_modulo":
        raise ValueError("class-fidelity evaluation requires balanced_modulo classes")
    if int(sampling["num_samples"]) != len(images):
        raise ValueError("class-fidelity sample count differs from sampling report")

    classifier = classifier_identity(args.classifier_checkpoint)
    cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if cuda else "cpu")
    environment = capture_runtime_environment(device, project_root=PROJECT_ROOT)
    environment_sha = runtime_environment_sha256(environment)
    parameters = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "min_samples": args.min_samples,
        "sample_count": len(images),
        "num_classes": 1000,
        "target_from_filename": "int(zero_based_png_stem) mod 1000",
        "device": device.type,
        "cpu_requested": bool(args.cpu),
    }
    expected = {
        "git": git_provenance(PROJECT_ROOT),
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "paths": {
            "generated_dir": generated_dir.resolve().as_posix(),
            "sampling_report": sampling_report.resolve().as_posix(),
            "output_dir": output_dir.as_posix(),
            "report": report_path.as_posix(),
        },
        "classifier": classifier,
        "sample_provenance": sample_provenance,
        "parameters": parameters,
    }
    if report_path.is_file():
        existing = read_json_object(report_path, name="class-fidelity report")
        _validate_completed_report(existing, expected=expected)
        print(f"reused completed class-fidelity report {report_path}")
        return

    start = time.time()
    metrics = calculate_class_fidelity(
        images=images,
        classifier_checkpoint=classifier["weights_path"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    if int(metrics["sample_count"]) != len(images):
        raise ValueError("class-fidelity evaluator did not consume the full sample set")
    report = {
        "schema_version": CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
        "role": CLASS_FIDELITY_REPORT_ROLE,
        "status": "completed",
        "protocol": "torchvision_imagenet_class_fidelity",
        **expected,
        "metrics": metrics,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
        },
    }
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(args.output_dir, role="generation_class_fidelity"):
        _run(args)


if __name__ == "__main__":
    main()
