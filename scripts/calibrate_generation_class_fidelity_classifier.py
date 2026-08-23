from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import ResNet50_Weights

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation_class_fidelity import (
    ClassFidelityAccumulator,
    validate_class_fidelity_metrics,
)
from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain
from cofitok.reporting import git_provenance, write_json_report
try:
    from scripts.evaluate_generation_class_fidelity import (
        DEFAULT_CLASSIFIER_CHECKPOINT,
        _load_classifier,
        classifier_identity,
    )
except ModuleNotFoundError:
    from evaluate_generation_class_fidelity import (
        DEFAULT_CLASSIFIER_CHECKPOINT,
        _load_classifier,
        classifier_identity,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_class_fidelity_classifier_real_calibration"
SELECTION_SCHEME = "lexicographic_first_n_per_wnid_v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLAIM_BOUNDARY = {
    "classifier_calibration_only": True,
    "generated_samples_evaluated": False,
    "generation_quality_evaluated": False,
    "replaces_class_fidelity_qualification": False,
    "replaces_frozen_promotion_gate": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the pinned class-fidelity classifier and ImageNet class "
            "order on a deterministic balanced real-validation subset."
        )
    )
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--label-to-wnid", required=True)
    parser.add_argument("--timm-synsets", required=True)
    parser.add_argument("--torchvision-categories", required=True)
    parser.add_argument(
        "--classifier-checkpoint",
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-top1", type=float, default=0.50)
    parser.add_argument("--min-top5", type=float, default=0.75)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-hold",
        action="store_true",
        help="Write a source-valid hold report without returning a failure status.",
    )
    return parser.parse_args()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_class_order(
    label_to_wnid_path: str | Path,
    timm_synsets_path: str | Path,
    torchvision_categories_path: str | Path,
    *,
    expected_num_classes: int = 1_000,
) -> list[str]:
    if expected_num_classes < 5:
        raise ValueError("class-order calibration requires at least five classes")
    mapping_path = Path(label_to_wnid_path)
    mapping_payload = _read_json_object(mapping_path)
    mapping = mapping_payload.get("label_to_wnid", mapping_payload)
    if not isinstance(mapping, dict):
        raise ValueError("label-to-WNID mapping is malformed")
    expected_keys = {str(index) for index in range(expected_num_classes)}
    if set(mapping) != expected_keys:
        raise ValueError("label-to-WNID mapping lacks the exact class-index domain")
    mapped_order = [str(mapping[str(index)]) for index in range(expected_num_classes)]

    timm_order = _nonempty_lines(Path(timm_synsets_path))
    category_lines = _nonempty_lines(Path(torchvision_categories_path))
    try:
        torchvision_order = [line.rsplit(",", 1)[1].strip() for line in category_lines]
    except IndexError as error:
        raise ValueError("torchvision ImageNet categories are malformed") from error
    if (
        len(mapped_order) != expected_num_classes
        or len(timm_order) != expected_num_classes
        or len(torchvision_order) != expected_num_classes
        or len(set(mapped_order)) != expected_num_classes
    ):
        raise ValueError("ImageNet class-order source count or uniqueness differs")
    if mapped_order != timm_order:
        raise ValueError("dataset label order differs from timm ImageNet synsets")
    if mapped_order != torchvision_order:
        raise ValueError("dataset label order differs from torchvision ImageNet categories")
    return mapped_order


def select_balanced_real_images(
    real_dir: str | Path,
    class_order: Sequence[str],
    *,
    samples_per_class: int,
) -> list[tuple[Path, int]]:
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    root = reject_symlink_chain(real_dir, name="class-fidelity real directory")
    if not root.is_dir():
        raise FileNotFoundError(f"real-validation directory is missing: {root}")
    actual_classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if actual_classes != list(class_order):
        raise ValueError("real-validation class directories differ from the locked class order")

    selected: list[tuple[Path, int]] = []
    for class_index, wnid in enumerate(class_order):
        class_dir = root / wnid
        if class_dir.is_symlink():
            raise ValueError(f"real-validation class directory is a symlink: {class_dir}")
        candidates = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"real-validation class {wnid} has fewer than {samples_per_class} images"
            )
        for path in candidates[:samples_per_class]:
            if path.is_symlink():
                raise ValueError(f"selected real-validation image is a symlink: {path}")
            selected.append((path, class_index))
    return selected


def selection_sha256(
    selected: Sequence[tuple[Path, int]],
    *,
    real_dir: str | Path,
) -> str:
    root = Path(real_dir).resolve()
    digest = hashlib.sha256()
    for path, class_index in selected:
        resolved = path.resolve()
        relative = resolved.relative_to(root).as_posix()
        digest.update(str(class_index).encode("ascii"))
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def class_order_sha256(class_order: Sequence[str]) -> str:
    encoded = json.dumps(
        list(class_order),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _RealCalibrationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, selected: Sequence[tuple[Path, int]]) -> None:
        self.selected = list(selected)
        self.transform = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.selected[index]
        image = read_image(path.as_posix(), mode=ImageReadMode.RGB)
        return self.transform(image), target


def calculate_real_calibration(
    *,
    selected: Sequence[tuple[Path, int]],
    classifier_checkpoint: str | Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    dataset = _RealCalibrationDataset(selected)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = _load_classifier(classifier_checkpoint, device)
    accumulator = ClassFidelityAccumulator(num_classes=num_classes)
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            accumulator.update(model(images), targets)
    return accumulator.finalize()


def build_checks(
    metrics: dict[str, Any],
    *,
    min_top1: float,
    min_top5: float,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "real_top1_accuracy",
            "status": "pass" if float(metrics["top1_accuracy"]) >= min_top1 else "hold",
            "observed": float(metrics["top1_accuracy"]),
            "threshold": min_top1,
            "comparison": ">=",
        },
        {
            "name": "real_top5_accuracy",
            "status": "pass" if float(metrics["top5_accuracy"]) >= min_top5 else "hold",
            "observed": float(metrics["top5_accuracy"]),
            "threshold": min_top5,
            "comparison": ">=",
        },
    ]


def _finite_fraction(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be a finite fraction")
    return numeric


def _validate_completed_report(
    report: dict[str, Any],
    *,
    expected: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("classifier-calibration report contract differs")
    for field in (
        "git",
        "runtime_environment",
        "runtime_environment_sha256",
        "paths",
        "sources",
        "classifier",
        "selection",
        "parameters",
        "thresholds",
    ):
        if report.get(field) != expected.get(field):
            raise ValueError(f"classifier-calibration report {field} differs")
    metrics = report.get("metrics")
    checks = report.get("checks")
    runtime = report.get("runtime")
    if not isinstance(metrics, dict) or not isinstance(checks, list) or not isinstance(runtime, dict):
        raise ValueError("classifier-calibration completed report is malformed")
    validate_class_fidelity_metrics(metrics)
    expected_sample_count = int(expected["parameters"]["sample_count"])
    expected_num_classes = int(expected["parameters"]["num_classes"])
    expected_samples_per_class = int(expected["parameters"]["samples_per_class"])
    if (
        int(metrics["sample_count"]) != expected_sample_count
        or int(metrics["num_classes"]) != expected_num_classes
        or int(metrics["requested_class_count"]) != expected_num_classes
        or int(metrics["requested_count_min"]) != expected_samples_per_class
        or int(metrics["requested_count_max"]) != expected_samples_per_class
    ):
        raise ValueError("classifier-calibration metric population differs")
    expected_checks = build_checks(
        metrics,
        min_top1=float(expected["thresholds"]["min_top1"]),
        min_top5=float(expected["thresholds"]["min_top5"]),
    )
    if checks != expected_checks:
        raise ValueError("classifier-calibration checks differ")
    expected_status = "pass" if all(row["status"] == "pass" for row in checks) else "hold"
    if report.get("calibration_status") != expected_status:
        raise ValueError("classifier-calibration status differs")
    elapsed = float(runtime.get("elapsed_seconds", math.nan))
    if (
        not math.isfinite(elapsed)
        or elapsed <= 0.0
        or runtime.get("torch_version") != torch.__version__
    ):
        raise ValueError("classifier-calibration runtime differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples_per_class < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("calibration sample/batch/worker arguments are invalid")
    min_top1 = _finite_fraction(args.min_top1, name="min_top1")
    min_top5 = _finite_fraction(args.min_top5, name="min_top5")
    if min_top5 < min_top1:
        raise ValueError("min_top5 must be at least min_top1")

    git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError(f"classifier-calibration Git identity differs: {git}")

    real_dir = reject_symlink_chain(args.real_dir, name="class-fidelity real directory")
    mapping_path = reject_symlink_chain(args.label_to_wnid, name="label-to-WNID mapping")
    timm_path = reject_symlink_chain(args.timm_synsets, name="timm ImageNet synsets")
    categories_path = reject_symlink_chain(
        args.torchvision_categories,
        name="torchvision ImageNet categories",
    )
    output = Path(args.output).resolve()
    if output.is_symlink():
        raise ValueError("classifier-calibration output must not be a symlink")
    if output.exists() and not output.is_file():
        raise ValueError("classifier-calibration output must be a file path")
    output.parent.mkdir(parents=True, exist_ok=True)

    class_order = load_class_order(mapping_path, timm_path, categories_path)
    selected = select_balanced_real_images(
        real_dir,
        class_order,
        samples_per_class=args.samples_per_class,
    )
    selected_sha = selection_sha256(selected, real_dir=real_dir)
    classifier = classifier_identity(args.classifier_checkpoint)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    environment = capture_runtime_environment(device, project_root=PROJECT_ROOT)
    environment_sha = runtime_environment_sha256(environment)
    parameters = {
        "device": device.type,
        "cpu_requested": bool(args.cpu),
        "samples_per_class": args.samples_per_class,
        "sample_count": len(selected),
        "num_classes": len(class_order),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    thresholds = {"min_top1": min_top1, "min_top5": min_top5}
    expected = {
        "git": git,
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "paths": {
            "real_dir": real_dir.resolve().as_posix(),
            "output": output.as_posix(),
        },
        "sources": {
            "label_to_wnid": file_identity(mapping_path),
            "timm_synsets": file_identity(timm_path),
            "torchvision_categories": file_identity(categories_path),
        },
        "classifier": classifier,
        "selection": {
            "scheme": SELECTION_SCHEME,
            "samples_per_class": args.samples_per_class,
            "sample_count": len(selected),
            "class_order_sha256": class_order_sha256(class_order),
            "selected_sample_set_sha256": selected_sha,
            "first_class": class_order[0],
            "last_class": class_order[-1],
        },
        "parameters": parameters,
        "thresholds": thresholds,
    }
    if output.is_file():
        if not args.resume:
            raise FileExistsError(
                "classifier-calibration report exists; pass --resume to validate it"
            )
        existing = read_json_object(output, name="classifier-calibration report")
        _validate_completed_report(existing, expected=expected)
        return existing

    started = time.time()
    metrics = calculate_real_calibration(
        selected=selected,
        classifier_checkpoint=classifier["weights_path"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        num_classes=len(class_order),
    )
    validate_class_fidelity_metrics(metrics)
    checks = build_checks(metrics, min_top1=min_top1, min_top5=min_top5)
    calibration_status = (
        "pass" if all(row["status"] == "pass" for row in checks) else "hold"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "calibration_status": calibration_status,
        "claim_boundary": CLAIM_BOUNDARY,
        **expected,
        "checks": checks,
        "metrics": metrics,
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
        },
    }
    _validate_completed_report(report, expected=expected)
    write_json_report(output, report)
    if calibration_status != "pass" and not args.allow_hold:
        raise RuntimeError("classifier real-validation calibration did not pass")
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": Path(args.output).resolve().as_posix(),
                "status": report["calibration_status"],
                "top1_accuracy": report["metrics"]["top1_accuracy"],
                "top5_accuracy": report["metrics"]["top5_accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
