from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cofitok.environment import runtime_environment_sha256
from cofitok.image_integrity import sample_set_sha256


EPSILON_STABILITY_ARTIFACT_REPORT_SCHEMA_VERSION = 1
EPSILON_STABILITY_ARTIFACT_REPORT_ROLE = (
    "generation_epsilon_stability_artifact_statistics_report"
)
EPSILON_STABILITY_ARTIFACT_METRIC_KEYS = (
    "channel_saturation_fraction",
    "total_variation",
    "median_filter_residual_fraction",
)
EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY = {
    "matched_1000_sample_sampling_launch_allowed": False,
    "associated_non_formal_evaluation_launch_allowed": False,
    "independent_matched_10000_confirmation_launch_allowed": False,
    "training_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
}
EPSILON_STABILITY_ARTIFACT_PARAMETERS = {
    "pixel_domain": "uint8_rgb_normalized_to_0_1",
    "saturation_low_inclusive": 1,
    "saturation_high_inclusive": 254,
    "total_variation": (
        "mean_abs_horizontal_difference_plus_mean_abs_vertical_difference"
    ),
    "median_filter_kernel": [3, 3],
    "median_filter_padding": "reflect",
    "median_filter_residual_threshold": 0.2,
    "median_filter_residual_test": "abs(pixel_minus_median) > 0.2",
}


def normalize_epsilon_stability_identity(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise ValueError(f"{label} identity is malformed")
    path = str(value["path"])
    byte_count = value["bytes"]
    digest = str(value["sha256"])
    if (
        not path
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": byte_count, "sha256": digest}


def normalize_epsilon_stability_boundary(
    value: Any,
    *,
    label: str,
) -> dict[str, bool]:
    if value != EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY:
        raise ValueError(f"{label} must equal the exact non-authorizing boundary")
    return dict(EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY)


def normalize_epsilon_stability_artifact_metrics(
    value: Any,
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(
        EPSILON_STABILITY_ARTIFACT_METRIC_KEYS
    ):
        raise ValueError(f"{label} artifact metric set is incomplete")
    normalized: dict[str, float] = {}
    for name in EPSILON_STABILITY_ARTIFACT_METRIC_KEYS:
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(f"{label} metric {name} is invalid")
        try:
            metric = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} metric {name} is invalid") from error
        if not math.isfinite(metric) or metric < 0.0:
            raise ValueError(f"{label} metric {name} is invalid")
        if name != "total_variation" and metric > 1.0:
            raise ValueError(f"{label} metric {name} is outside [0, 1]")
        if name == "total_variation" and metric > 2.0:
            raise ValueError(f"{label} metric total_variation exceeds 2")
        normalized[name] = metric
    return normalized


def _valid_git_identity(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    revision = str(value.get("revision", ""))
    return (
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
        and bool(str(value.get("branch", "")))
        and value.get("tracked_dirty") is False
    )


def numbered_epsilon_stability_pngs(
    image_dir: str | Path,
    *,
    sample_count: int,
    start_index: int = 0,
) -> list[Path]:
    root = Path(image_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact image directory is invalid: {root}")
    if sample_count < 1 or start_index < 0:
        raise ValueError("artifact sample range is invalid")
    expected_names = [
        f"{index:06d}.png"
        for index in range(start_index, start_index + sample_count)
    ]
    actual_names = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if actual_names != expected_names:
        raise ValueError("artifact image directory is not the exact numbered sample set")
    paths = [root / name for name in expected_names]
    if any(path.is_symlink() for path in paths):
        raise ValueError("artifact image set contains a symlink")
    return paths


def compute_epsilon_stability_artifact_metrics(
    image_paths: Sequence[str | Path],
) -> dict[str, float]:
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("artifact statistics require at least one image")

    saturation_count = 0
    channel_value_count = 0
    horizontal_difference_sum = 0
    horizontal_difference_count = 0
    vertical_difference_sum = 0
    vertical_difference_count = 0
    median_residual_count = 0
    observed_shape: tuple[int, int, int] | None = None

    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact image is invalid: {path}")
        with Image.open(path) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (256, 256)
            ):
                raise ValueError(f"artifact image contract differs: {path}")
            pixels = np.asarray(image, dtype=np.uint8)

        shape = tuple(int(value) for value in pixels.shape)
        if observed_shape is None:
            observed_shape = shape
        elif shape != observed_shape:
            raise ValueError("artifact images do not share one shape")

        channel_value_count += int(pixels.size)
        saturation_count += int(
            np.logical_or(pixels <= 1, pixels >= 254).sum(dtype=np.int64)
        )

        signed = pixels.astype(np.int16, copy=False)
        horizontal = np.abs(np.diff(signed, axis=1))
        vertical = np.abs(np.diff(signed, axis=0))
        horizontal_difference_sum += int(horizontal.sum(dtype=np.int64))
        horizontal_difference_count += int(horizontal.size)
        vertical_difference_sum += int(vertical.sum(dtype=np.int64))
        vertical_difference_count += int(vertical.size)

        padded = np.pad(
            pixels,
            ((1, 1), (1, 1), (0, 0)),
            mode="reflect",
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (3, 3),
            axis=(0, 1),
        ).reshape(256, 256, 3, 9)
        median = np.partition(windows, 4, axis=-1)[..., 4]
        residual = np.abs(signed - median.astype(np.int16, copy=False))
        median_residual_count += int((residual > 51).sum(dtype=np.int64))

    if (
        observed_shape != (256, 256, 3)
        or channel_value_count < 1
        or horizontal_difference_count < 1
        or vertical_difference_count < 1
    ):
        raise ValueError("artifact image accounting is invalid")

    metrics = {
        "channel_saturation_fraction": (
            saturation_count / channel_value_count
        ),
        "total_variation": (
            horizontal_difference_sum
            / (horizontal_difference_count * 255.0)
            + vertical_difference_sum
            / (vertical_difference_count * 255.0)
        ),
        "median_filter_residual_fraction": (
            median_residual_count / channel_value_count
        ),
    }
    return normalize_epsilon_stability_artifact_metrics(
        metrics,
        label="computed artifact statistics",
    )


def build_epsilon_stability_artifact_report(
    *,
    source_kind: str,
    git: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    runtime_environment_sha256_value: str,
    sample_count: int,
    sample_set_sha256_value: str,
    source_binding: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if source_kind not in {"generated", "real_reference"}:
        raise ValueError("artifact source kind is unsupported")
    if not _valid_git_identity(git):
        raise ValueError("artifact evaluator Git identity is malformed")
    environment = dict(runtime_environment)
    actual_runtime_sha256 = runtime_environment_sha256(environment)
    if runtime_environment_sha256_value != actual_runtime_sha256:
        raise ValueError("artifact runtime environment SHA256 differs")
    if sample_count < 1:
        raise ValueError("artifact sample count is invalid")
    if (
        len(sample_set_sha256_value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in sample_set_sha256_value
        )
    ):
        raise ValueError("artifact sample-set SHA256 is malformed")
    if not isinstance(source_binding, Mapping) or not source_binding:
        raise ValueError("artifact source binding is missing")
    return {
        "schema_version": EPSILON_STABILITY_ARTIFACT_REPORT_SCHEMA_VERSION,
        "role": EPSILON_STABILITY_ARTIFACT_REPORT_ROLE,
        "status": "completed",
        "source_kind": source_kind,
        "git": dict(git),
        "runtime_environment": environment,
        "runtime_environment_sha256": actual_runtime_sha256,
        "image_shape": [3, 256, 256],
        "sample_count": int(sample_count),
        "sample_set_sha256": sample_set_sha256_value,
        "source_binding": dict(source_binding),
        "parameters": dict(EPSILON_STABILITY_ARTIFACT_PARAMETERS),
        "metrics": normalize_epsilon_stability_artifact_metrics(
            metrics,
            label="artifact report",
        ),
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def validate_epsilon_stability_artifact_report(
    report: Mapping[str, Any],
    *,
    expected_source_kind: str,
    expected_git: Mapping[str, Any],
    expected_runtime_environment_sha256: str,
) -> dict[str, Any]:
    if (
        report.get("schema_version")
        != EPSILON_STABILITY_ARTIFACT_REPORT_SCHEMA_VERSION
        or report.get("role") != EPSILON_STABILITY_ARTIFACT_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("source_kind") != expected_source_kind
        or report.get("git") != dict(expected_git)
        or report.get("image_shape") != [3, 256, 256]
        or report.get("parameters") != EPSILON_STABILITY_ARTIFACT_PARAMETERS
    ):
        raise ValueError("epsilon-stability artifact report contract differs")
    environment = report.get("runtime_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("epsilon-stability artifact runtime is missing")
    actual_runtime_sha256 = runtime_environment_sha256(dict(environment))
    if (
        report.get("runtime_environment_sha256") != actual_runtime_sha256
        or actual_runtime_sha256 != expected_runtime_environment_sha256
    ):
        raise ValueError("epsilon-stability artifact runtime differs")
    sample_count = report.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("epsilon-stability artifact sample count is invalid")
    digest = str(report.get("sample_set_sha256", ""))
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("epsilon-stability artifact sample digest is invalid")
    source_binding = report.get("source_binding")
    if not isinstance(source_binding, Mapping) or not source_binding:
        raise ValueError("epsilon-stability artifact source binding is missing")
    return {
        "sample_count": sample_count,
        "sample_set_sha256": digest,
        "source_binding": dict(source_binding),
        "metrics": normalize_epsilon_stability_artifact_metrics(
            report.get("metrics"),
            label="epsilon-stability artifact report",
        ),
        "authorization_boundary": normalize_epsilon_stability_boundary(
            report.get("authorization_boundary"),
            label="epsilon-stability artifact report authorization boundary",
        ),
    }


def artifact_sample_set_sha256(image_paths: Sequence[str | Path]) -> str:
    return sample_set_sha256(image_paths)
