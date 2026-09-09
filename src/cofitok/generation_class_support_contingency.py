"""Fail-closed statistics and contracts for frozen class-support diagnosis.

The preparation and replay helpers in this module are deliberately independent
of the diffusion model.  They operate only on an already generated, exactly
bound pair of ImageNet sample trees and on fixed-classifier predictions.  No
function in this module launches training, sampling, or classifier inference.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import stat
import struct
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

# Parsed artifact type mismatches are schema-value failures, not API misuse.
# ruff: noqa: TRY004


PREPARATION_SCHEMA = "cofitok_generation_class_support_contingency_preparation_v1"
PREPARATION_ROLE = "source_bound_frozen_sample_class_support_contingency_preparation"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_class_support_contingency_execution_authorization_v1"
)
EXECUTION_AUTHORIZATION_ROLE = (
    "source_bound_frozen_sample_class_support_contingency_execution_authorization"
)
STAGE_APPROVAL_SCHEMA = "cofitok_generation_class_support_contingency_stage_approval_v1"
STAGE_APPROVAL_ROLE = "user_created_frozen_sample_class_support_contingency_approval"
STAGE_APPROVAL_SCOPE = "frozen_100k_sample_class_support_contingency_evaluation_only"
RESULT_SCHEMA = "cofitok_generation_class_support_contingency_result_v1"
RESULT_ROLE = "frozen_sample_class_support_contingency_result"
VALIDATION_SCHEMA = "cofitok_generation_class_support_contingency_validation_v1"
VALIDATION_ROLE = "independent_frozen_sample_class_support_contingency_validation"

METHODS = ("cofitok", "dense_identity")
NUM_CLASSES = 1_000
SAMPLE_COUNT = 10_000
IMAGE_SHAPE = [3, 256, 256]
SOURCE_CHECKPOINT_STEP = 100_000
SOURCE_SAMPLING_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
SOURCE_SAMPLING_BRANCH = "scale/generation-stability-quality-bridge-100k"
CAUSAL_DECISION = "no_defensible_shared_intervention_selected"
FROZEN_SAMPLING_FIELDS = {
    "protocol_schema",
    "inference_api",
    "sampler",
    "num_samples",
    "start_index",
    "batch_size",
    "sample_steps",
    "num_train_timesteps",
    "actual_timesteps",
    "prefix_budgets",
    "guidance_scale",
    "guidance_rescale",
    "cfg_batch_mode",
    "eta",
    "clip_x0",
    "seed",
    "precision",
    "image_shape",
    "class_schedule",
    "random_stream",
    "sample_set_digest",
}
FROZEN_ACTUAL_TIMESTEPS = sorted(
    {round(index * 999 / 99) for index in range(100)}, reverse=True
)

CLASSIFIER = {
    "name": "torchvision_resnet50_imagenet1k_v2",
    "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
    "weights_bytes": 102_540_417,
    "weights_sha256": (
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    ),
    "categories_sha256": (
        "62fff941ecff3f19de9128c6ca9c2097807c6b7bafc5552d7589a219431221ed"
    ),
    "num_classes": NUM_CLASSES,
    "preprocessing": {
        "resize_size": [232],
        "crop_size": [224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
        "antialias": True,
    },
}

FROZEN_SOURCE_SPECS = {
    "cofitok": {
        "prefix_budget": 8,
        "checkpoint_sha256": (
            "b36a92229ba2dd021db9c7585970ddda1d17d1919ca6b0eec0362b06d4bd462e"
        ),
        "sample_set_sha256": (
            "7f57ee4a874667b17085df203503443440a84124719a53da31db726a1efaff98"
        ),
        "sampling_report_sha256": (
            "79d5ab69283d171cf0390671d3204f9eeaf13a07f80354527b02335128069a44"
        ),
        "sampling_manifest_sha256": (
            "d87f96a455664454fd790d4553a0a01494ee6580217e7755b0bb98be2a80e306"
        ),
        "sampling_progress_sha256": (
            "ee9442c3d96f80ba15284b2992c6d0d36816a3bbe7485115289e55a8b1b9afa1"
        ),
        "class_fidelity_report_sha256": (
            "d84bad6dc9834f98600af44221b82b3433b0fb91063ba0ba5c27a7ba5435fd21"
        ),
        "generation_metrics_report_sha256": (
            "4dc4c6afd68f9e9f8ac335bf84108b8134d5a6404c316079c1185330f67f76dc"
        ),
    },
    "dense_identity": {
        "prefix_budget": 1,
        "checkpoint_sha256": (
            "b6586cc906a9c38bbf9d592f5f6169b7a37a8d3aa485ca4879021fa840943d0b"
        ),
        "sample_set_sha256": (
            "3fc905a55dc368ddf268278c04963c5933b3e783f4cabd872762135b6345bdaf"
        ),
        "sampling_report_sha256": (
            "5cb1e0f1f90313101d65194058eaf78fb6dcc8545ca048b2c5b48f509d79c235"
        ),
        "sampling_manifest_sha256": (
            "28d1fdd70e73a2933a2628330c5b174af7944571669a8f88b625f0ec6c6b21c9"
        ),
        "sampling_progress_sha256": (
            "929603ff47dc4a8e0a59855b506a6e5240bedfdacbe92825a5cb3a5cd98b5173"
        ),
        "class_fidelity_report_sha256": (
            "987cf89e0d886ac9fbfdcdd814beb760a2fb7fbe3621c0eea6d665f3c1e0bf4b"
        ),
        "generation_metrics_report_sha256": (
            "b59f33f98a34906a0c7d97767728595f990cc0f52838f51fa9f3cc18bf7acfba"
        ),
    },
}

CAUSAL_SOURCE_SHA256 = {
    "decision": "6b0887dec10d6908550bee8e90c3a6e242dbddb3cb62c97935eac841dd262bc0",
    "validation": "0043680c6109de3adffae743517ef87b4efd81f3e87f6d540a144aa804354292",
}

NULL_PROTOCOL = {
    "seed": 202_609_10,
    "replicates": 512,
    "same_index_permutation_scope": "within_requested_class",
    "ami_permutation_scope": "global_predicted_labels",
    "empirical_pvalue": "(1 + null_ge_observed) / (1 + replicates)",
}

INTERPRETATION_THRESHOLDS = {
    "significance_alpha": 0.01,
    "min_adjusted_mutual_information": 0.01,
    "min_direct_top1_accuracy": 0.01,
    "min_direct_top5_accuracy": 0.05,
    "min_cyclic_top1_accuracy": 0.01,
    "min_cyclic_lift_over_direct": 0.005,
    "min_cross_validated_mapping_accuracy": 0.01,
    "min_mapping_lift_over_direct": 0.005,
    "min_shared_histogram_overlap": 0.50,
    "max_shared_histogram_jsd_bits": 0.25,
    "max_existing_recall_for_support_collapse": 0.02,
}

PER_SAMPLE_FIELDS = (
    "sample_index",
    "filename",
    "requested_class",
    "requested_probability",
    "predicted_top1_class",
    "predicted_top1_probability",
    "predicted_top5_classes",
    "predicted_top5_probabilities",
    "image_sha256",
)

REQUIRED_STATISTICS = (
    "requested_by_predicted_top1_contingency",
    "adjusted_mutual_information_arithmetic",
    "exact_binomial_top1_greater_than_chance",
    "exact_binomial_top5_greater_than_chance",
    "predicted_class_concentration_entropy_and_top_mode_mass",
    "cross_method_histogram_overlap_and_jensen_shannon_divergence",
    "same_index_cross_method_agreement_with_deterministic_null",
    "within_tree_and_cross_method_exact_duplicate_counts",
    "best_cyclic_offset_alignment",
    "best_one_to_one_mapping_with_two_fold_cross_validation",
)

INTERPRETATION_LABELS = (
    "ignored_conditioning",
    "active_but_misaligned_conditioning",
    "permuted_or_offset_conditioning",
    "shared_unconditional_support_collapse",
)

RUNTIME_FIELDS = {
    "elapsed_seconds",
    "device",
    "batch_size",
    "num_workers",
    "torch_version",
    "torchvision_version",
    "scipy_version",
}

PREPARATION_BOUNDARY = {
    "preparation_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "classifier_inference_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "confirmation_preparation_allowed": False,
    "confirmation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "full_training_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "paper_integration_allowed": False,
    "process_signals_allowed": False,
}

RESULT_BOUNDARY = {
    "diagnostic_only": True,
    "generates_new_samples": False,
    "retraining_performed": False,
    "replaces_terminal_snr_screen": False,
    "generation_advantage_proven": False,
    "confirmation_preparation_allowed": False,
    "confirmation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "full_training_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "paper_integration_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _hex(value: Any, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase {length}-character hexadecimal")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _fraction(value: Any, name: str) -> float:
    numeric = _finite(value, name)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def _absolute(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is missing")
    if not (Path(value).is_absolute() or PurePosixPath(value).is_absolute()):
        raise ValueError(f"{name} must be absolute")
    return value


def validate_identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    _absolute(row.get("path"), f"{name} path")
    if (
        isinstance(row.get("bytes"), bool)
        or not isinstance(row.get("bytes"), int)
        or row["bytes"] < 1
    ):
        raise ValueError(f"{name} byte count is invalid")
    _hex(row.get("sha256"), length=64, name=f"{name} SHA256")
    return copy.deepcopy(row)


def validate_git_identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} fields differ")
    _hex(row.get("revision"), length=40, name=f"{name} revision")
    _hex(row.get("tree"), length=40, name=f"{name} tree")
    if not isinstance(row.get("branch"), str) or not row["branch"]:
        raise ValueError(f"{name} branch is invalid")
    if row.get("tracked_dirty") is not False:
        raise ValueError(f"{name} must identify a tracked-clean checkout")
    return copy.deepcopy(row)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _reject_symlink_chain(path: Path) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    parts = absolute.parts
    if not parts:
        raise ValueError(f"invalid path: {path}")
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        mode = os.lstat(current).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is forbidden: {current}")


def stable_file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"source path must be absolute: {source}")
    _reject_symlink_chain(source)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source is not a regular file: {source}")
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    after = source.stat()
    if (
        count != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise RuntimeError(f"source changed while hashing: {source}")
    return {
        "path": source.resolve().as_posix(),
        "bytes": count,
        "sha256": digest.hexdigest(),
    }


def _png_header(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"image is not a PNG: {path}")
    length = struct.unpack(">I", header[8:12])[0]
    if length != 13 or header[12:16] != b"IHDR":
        raise ValueError(f"PNG lacks a canonical IHDR: {path}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if bit_depth != 8 or channels is None:
        raise ValueError(f"PNG bit depth or color type differs: {path}")
    return channels, height, width


def physical_sample_tree_identity(
    root: str | Path,
    *,
    expected_count: int = SAMPLE_COUNT,
    expected_shape: Sequence[int] = IMAGE_SHAPE,
    expected_image_sha256: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Physically validate and hash one immutable numbered PNG tree."""

    expected_count = _integer(expected_count, "expected sample count", minimum=1)
    directory = Path(root)
    if not directory.is_absolute():
        raise ValueError("sample tree path must be absolute")
    _reject_symlink_chain(directory)
    before = directory.stat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"sample tree is not a directory: {directory}")
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    expected_names = [f"{index:06d}.png" for index in range(expected_count)]
    if [path.name for path in entries] != expected_names:
        raise ValueError("sample tree filenames are not the exact zero-based PNG set")
    shape = _sequence(expected_shape, "expected image shape")
    if len(shape) != 3:
        raise ValueError("expected image shape must have three dimensions")
    channels, height, width = [
        _integer(value, f"expected image shape dimension {index}", minimum=1)
        for index, value in enumerate(shape)
    ]
    expected_digests = None
    if expected_image_sha256 is not None:
        expected_digests = _sequence(
            expected_image_sha256, "expected per-image SHA256 values"
        )
        if len(expected_digests) != expected_count:
            raise ValueError("expected per-image SHA256 count differs")
        expected_digests = [
            _hex(value, length=64, name=f"expected image {index} SHA256")
            for index, value in enumerate(expected_digests)
        ]
    digest = hashlib.sha256()
    total_bytes = 0
    for index, path in enumerate(entries):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sample tree entry is not a regular file: {path}")
        file_before = path.stat()
        if list(_png_header(path)) != [channels, height, width]:
            raise ValueError(f"sample PNG shape differs: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, SyntaxError, ValueError) as error:
            raise ValueError(
                f"sample PNG failed physical verification: {path}"
            ) from error
        name = path.name.encode("utf-8")
        digest.update(name)
        digest.update(b"\0")
        image_digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                image_digest.update(block)
                count += len(block)
        file_after = path.stat()
        if (
            count != file_before.st_size
            or file_after.st_size != file_before.st_size
            or file_after.st_mtime_ns != file_before.st_mtime_ns
            or file_after.st_ctime_ns != file_before.st_ctime_ns
            or file_after.st_ino != file_before.st_ino
            or file_after.st_dev != file_before.st_dev
        ):
            raise RuntimeError(f"sample changed while hashing: {path}")
        if (
            expected_digests is not None
            and image_digest.hexdigest() != expected_digests[index]
        ):
            raise ValueError(f"sample image SHA256 differs: {path}")
        total_bytes += count
        digest.update(b"\0")
    after = directory.stat()
    if (
        after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise RuntimeError("sample tree directory changed while hashing")
    return {
        "path": directory.resolve().as_posix(),
        "count": expected_count,
        "total_bytes": total_bytes,
        "sample_set_sha256": digest.hexdigest(),
        "filename_contract": "zero_based_six_digit_png",
        "image_shape": [channels, height, width],
        "physical_validation_performed": True,
    }


def _validate_frozen_source(value: Any, method: str) -> dict[str, Any]:
    row = _object(value, f"{method} frozen source")
    spec = FROZEN_SOURCE_SPECS[method]
    if set(row) != {
        "identities",
        "sample_tree",
        "sampling",
        "sampling_git",
        "checkpoint_step",
        "checkpoint_sha256",
        "prefix_budget",
        "weights",
        "existing_class_fidelity",
        "existing_generation_metrics",
    }:
        raise ValueError(f"{method} frozen source fields differ")
    identities = _object(row.get("identities"), f"{method} identities")
    expected_identity_keys = {
        "sampling_report",
        "sampling_manifest",
        "sampling_progress",
        "class_fidelity_report",
        "generation_metrics_report",
    }
    if set(identities) != expected_identity_keys:
        raise ValueError(f"{method} source identity set differs")
    checked = {
        name: validate_identity(identity, f"{method} {name}")
        for name, identity in identities.items()
    }
    for name in expected_identity_keys:
        expected = spec[f"{name}_sha256"]
        if checked[name]["sha256"] != expected:
            raise ValueError(f"{method} {name} is not the locked frozen source")
    tree = _object(row.get("sample_tree"), f"{method} sample tree")
    tree_count = _integer(tree.get("count"), f"{method} sample count", minimum=1)
    tree_bytes = _integer(
        tree.get("total_bytes"), f"{method} sample byte count", minimum=1
    )
    tree_shape = [
        _integer(value, f"{method} sample shape dimension {index}", minimum=1)
        for index, value in enumerate(
            _sequence(tree.get("image_shape"), f"{method} sample shape")
        )
    ]
    if (
        set(tree)
        != {
            "path",
            "count",
            "total_bytes",
            "sample_set_sha256",
            "filename_contract",
            "image_shape",
            "physical_validation_performed",
        }
        or not _absolute(tree.get("path"), f"{method} sample tree path")
        or tree_count != SAMPLE_COUNT
        or tree_bytes < SAMPLE_COUNT
        or tree.get("sample_set_sha256") != spec["sample_set_sha256"]
        or tree.get("filename_contract") != "zero_based_six_digit_png"
        or tree_shape != IMAGE_SHAPE
        or tree.get("physical_validation_performed") is not True
    ):
        raise ValueError(f"{method} sample tree contract differs")
    sampling = _object(row.get("sampling"), f"{method} sampling")
    sampling_git = _object(row.get("sampling_git"), f"{method} sampling Git")
    class_fidelity = _object(
        row.get("existing_class_fidelity"), f"{method} class fidelity"
    )
    generation_metrics = _object(
        row.get("existing_generation_metrics"), f"{method} generation metrics"
    )
    if set(sampling) != FROZEN_SAMPLING_FIELDS:
        raise ValueError(f"{method} frozen sampling fields differ")
    if set(sampling_git) != {"revision", "branch", "tracked_dirty"}:
        raise ValueError(f"{method} sampling Git fields differ")
    _hex(
        sampling_git.get("revision"),
        length=40,
        name=f"{method} sampling Git revision",
    )
    if not isinstance(sampling_git.get("branch"), str) or not sampling_git["branch"]:
        raise ValueError(f"{method} sampling Git branch is invalid")
    if sampling_git.get("tracked_dirty") is not False:
        raise ValueError(f"{method} sampling Git must be tracked-clean")
    if set(class_fidelity) != {
        "sample_count",
        "num_classes",
        "requested_class_count",
        "requested_count_min",
        "requested_count_max",
        "top1_accuracy",
        "top5_accuracy",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    }:
        raise ValueError(f"{method} existing class-fidelity fields differ")
    if set(generation_metrics) != {
        "frechet_inception_distance",
        "precision",
        "recall",
    }:
        raise ValueError(f"{method} existing generation-metric fields differ")
    required_sampling = {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "sampler": "ddim",
        "num_samples": SAMPLE_COUNT,
        "start_index": 0,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "image_shape": IMAGE_SHAPE,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "class_schedule": "balanced_modulo",
        "prefix_budgets": [spec["prefix_budget"]],
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
    }
    if any(
        sampling.get(key) != expected for key, expected in required_sampling.items()
    ):
        raise ValueError(f"{method} frozen sampling protocol differs")
    inference_api = _object(
        sampling.get("inference_api"), f"{method} frozen inference API"
    )
    if (
        set(inference_api) != {"name", "version"}
        or inference_api.get("name") != "cofitok.generation.GenerationSession"
        or _integer(
            inference_api.get("version"),
            f"{method} frozen inference API version",
            minimum=1,
        )
        != 1
    ):
        raise ValueError(f"{method} frozen inference API differs")
    for field, minimum in (
        ("num_samples", 1),
        ("start_index", 0),
        ("batch_size", 1),
        ("sample_steps", 1),
        ("num_train_timesteps", 1),
        ("seed", 0),
    ):
        _integer(sampling.get(field), f"{method} sampling {field}", minimum=minimum)
    image_shape = [
        _integer(
            value,
            f"{method} sampling image-shape dimension {index}",
            minimum=1,
        )
        for index, value in enumerate(
            _sequence(sampling.get("image_shape"), f"{method} sampling image shape")
        )
    ]
    prefix_budgets = [
        _integer(
            value,
            f"{method} sampling prefix budget {index}",
            minimum=1,
        )
        for index, value in enumerate(
            _sequence(
                sampling.get("prefix_budgets"), f"{method} sampling prefix budgets"
            )
        )
    ]
    for field in ("guidance_scale", "guidance_rescale", "eta"):
        _finite(sampling.get(field), f"{method} sampling {field}")
    if (
        image_shape != IMAGE_SHAPE
        or prefix_budgets != [spec["prefix_budget"]]
        or sampling.get("clip_x0") is not True
    ):
        raise ValueError(f"{method} frozen sampling typed fields differ")
    actual_timesteps = [
        _integer(value, f"{method} sampling timestep {index}", minimum=0)
        for index, value in enumerate(
            _sequence(sampling.get("actual_timesteps"), f"{method} sampling timesteps")
        )
    ]
    if (
        actual_timesteps != FROZEN_ACTUAL_TIMESTEPS
        or len(actual_timesteps) != sampling["sample_steps"]
    ):
        raise ValueError(f"{method} frozen sampling timesteps differ")
    random_stream = _object(sampling.get("random_stream"), f"{method} random stream")
    if set(random_stream) != {
        "scope",
        "seed_formula",
        "prefix_budgets_share_stream",
        "batch_size_invariant",
        "resume_index_invariant",
    }:
        raise ValueError(f"{method} frozen random-stream fields differ")
    if any(
        random_stream.get(name) is not True
        for name in (
            "prefix_budgets_share_stream",
            "batch_size_invariant",
            "resume_index_invariant",
        )
    ):
        raise ValueError(f"{method} frozen random-stream contract differs")
    if (
        random_stream.get("scope") != "per_global_sample_index"
        or random_stream.get("seed_formula") != "(seed + global_index) mod 2^63"
    ):
        raise ValueError(f"{method} frozen random-stream scope differs")
    sample_set_digest = _object(
        sampling.get("sample_set_digest"), f"{method} sample-set digest"
    )
    if sample_set_digest != {
        "algorithm": "sha256",
        "framing": "filename_utf8_nul_file_bytes_nul",
    }:
        raise ValueError(f"{method} sample-set digest contract differs")
    checkpoint_step = _integer(
        row.get("checkpoint_step"), f"{method} checkpoint step", minimum=1
    )
    prefix_budget = _integer(
        row.get("prefix_budget"), f"{method} prefix budget", minimum=1
    )
    if (
        sampling_git.get("revision") != SOURCE_SAMPLING_REVISION
        or sampling_git.get("branch") != SOURCE_SAMPLING_BRANCH
        or sampling_git.get("tracked_dirty") is not False
        or checkpoint_step != SOURCE_CHECKPOINT_STEP
        or row.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or prefix_budget != spec["prefix_budget"]
        or row.get("weights") != "ema"
    ):
        raise ValueError(f"{method} checkpoint or sampler provenance differs")
    required_fidelity = {
        "sample_count": SAMPLE_COUNT,
        "num_classes": NUM_CLASSES,
        "requested_class_count": NUM_CLASSES,
        "requested_count_min": SAMPLE_COUNT // NUM_CLASSES,
        "requested_count_max": SAMPLE_COUNT // NUM_CLASSES,
    }
    if any(
        _integer(
            class_fidelity.get(key),
            f"{method} existing class-fidelity {key}",
            minimum=0,
        )
        != expected
        for key, expected in required_fidelity.items()
    ):
        raise ValueError(f"{method} existing class-fidelity accounting differs")
    for key in (
        "top1_accuracy",
        "top5_accuracy",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    ):
        _fraction(class_fidelity.get(key), f"{method} existing {key}")
    for key in ("precision", "recall"):
        _fraction(generation_metrics.get(key), f"{method} existing {key}")
    fid = _finite(
        generation_metrics.get("frechet_inception_distance"),
        f"{method} existing FID",
    )
    if fid < 0.0:
        raise ValueError(f"{method} existing FID is negative")
    return copy.deepcopy(row)


def _validate_source_evidence(value: Any) -> dict[str, Any]:
    source = _object(value, "class-support source evidence")
    if set(source) != {
        "causal_discriminator",
        "causal_discriminator_validation",
        "causal_decision",
        "causal_scientific_status",
        "fixed_classifier",
        "frozen_methods",
    }:
        raise ValueError("class-support source evidence fields differ")
    decision_id = validate_identity(source["causal_discriminator"], "causal decision")
    validation_id = validate_identity(
        source["causal_discriminator_validation"], "causal validation"
    )
    classifier_id = validate_identity(source["fixed_classifier"], "classifier")
    if (
        source["causal_decision"] != CAUSAL_DECISION
        or source["causal_scientific_status"] != "hold"
        or decision_id["sha256"] != CAUSAL_SOURCE_SHA256["decision"]
        or validation_id["sha256"] != CAUSAL_SOURCE_SHA256["validation"]
        or classifier_id["bytes"] != CLASSIFIER["weights_bytes"]
        or classifier_id["sha256"] != CLASSIFIER["weights_sha256"]
    ):
        raise ValueError("class-support source evidence identity differs")
    frozen = _object(source["frozen_methods"], "frozen methods")
    if set(frozen) != set(METHODS):
        raise ValueError("frozen method set differs")
    for method in METHODS:
        _validate_frozen_source(frozen[method], method)
    return copy.deepcopy(source)


def _validate_diagnostic_contract(value: Any) -> dict[str, Any]:
    contract = _object(value, "class-support diagnostic contract")
    if (
        set(contract)
        != {
            "methods",
            "sample_count_per_method",
            "num_classes",
            "requested_label",
            "same_index_pairing",
            "classifier",
            "per_sample_fields",
            "required_statistics",
            "null_protocol",
            "interpretation_thresholds",
            "interpretation_labels",
        }
        or contract.get("methods") != list(METHODS)
        or contract.get("sample_count_per_method") != SAMPLE_COUNT
        or contract.get("num_classes") != NUM_CLASSES
        or contract.get("requested_label") != "int(zero_based_png_stem) mod 1000"
        or contract.get("same_index_pairing") != "identical zero_based_png_stem"
        or contract.get("classifier") != CLASSIFIER
        or contract.get("per_sample_fields") != list(PER_SAMPLE_FIELDS)
        or contract.get("required_statistics") != list(REQUIRED_STATISTICS)
        or contract.get("null_protocol") != NULL_PROTOCOL
        or contract.get("interpretation_thresholds") != INTERPRETATION_THRESHOLDS
        or contract.get("interpretation_labels") != list(INTERPRETATION_LABELS)
    ):
        raise ValueError("class-support diagnostic contract differs")
    return copy.deepcopy(contract)


def _validate_runtime(value: Any) -> dict[str, Any]:
    runtime = _object(value, "diagnostic runtime")
    if set(runtime) != RUNTIME_FIELDS:
        raise ValueError("diagnostic runtime fields differ")
    if _finite(runtime["elapsed_seconds"], "runtime elapsed seconds") <= 0.0:
        raise ValueError("diagnostic runtime must be positive")
    if runtime["device"] not in {"cpu", "cuda"}:
        raise ValueError("diagnostic runtime device differs")
    _integer(runtime["batch_size"], "runtime batch size", minimum=1)
    _integer(runtime["num_workers"], "runtime worker count", minimum=0)
    for field in ("torch_version", "torchvision_version", "scipy_version"):
        if not isinstance(runtime[field], str) or not runtime[field].strip():
            raise ValueError(f"diagnostic runtime {field} is invalid")
    return copy.deepcopy(runtime)


def build_preparation(
    *,
    causal_discriminator: Mapping[str, Any],
    causal_discriminator_identity: Mapping[str, Any],
    causal_validation: Mapping[str, Any],
    causal_validation_identity: Mapping[str, Any],
    classifier_identity: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    preparation_git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    decision = _object(causal_discriminator, "causal discriminator")
    validation = _object(causal_validation, "causal discriminator validation")
    decision_id = validate_identity(
        causal_discriminator_identity, "causal discriminator"
    )
    validation_id = validate_identity(
        causal_validation_identity, "causal discriminator validation"
    )
    if decision_id["sha256"] != CAUSAL_SOURCE_SHA256["decision"]:
        raise ValueError("causal discriminator is not the locked decision")
    if validation_id["sha256"] != CAUSAL_SOURCE_SHA256["validation"]:
        raise ValueError("causal validation is not the locked receipt")
    if (
        decision.get("status") != "completed"
        or decision.get("scientific_status") != "hold"
        or decision.get("terminal_status") != "hold"
        or decision.get("selected_candidate") is not None
        or decision.get("decision") != CAUSAL_DECISION
        or decision.get("generation_advantage_proven") is not False
        or validation.get("status") != "pass"
        or validation.get("scientific_status") != "hold"
        or validation.get("decision") != decision_id
    ):
        raise ValueError(
            "causal discriminator does not permit a diagnostic preparation"
        )
    classifier_id = validate_identity(classifier_identity, "fixed classifier")
    if (
        classifier_id["bytes"] != CLASSIFIER["weights_bytes"]
        or classifier_id["sha256"] != CLASSIFIER["weights_sha256"]
    ):
        raise ValueError("fixed classifier bytes or SHA256 differ")
    if set(sources) != set(METHODS):
        raise ValueError("frozen source method set differs")
    frozen = {
        method: _validate_frozen_source(sources[method], method) for method in METHODS
    }
    if frozen["cofitok"]["sampling"] != {
        **frozen["dense_identity"]["sampling"],
        "prefix_budgets": [8],
    }:
        raise ValueError("frozen methods do not share the exact sampling protocol")
    report = {
        "schema_version": PREPARATION_SCHEMA,
        "role": PREPARATION_ROLE,
        "status": "prepared",
        "scientific_status": "diagnostic_not_executed",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation_git": validate_git_identity(preparation_git, "preparation Git"),
        "source_evidence": {
            "causal_discriminator": decision_id,
            "causal_discriminator_validation": validation_id,
            "causal_decision": CAUSAL_DECISION,
            "causal_scientific_status": "hold",
            "fixed_classifier": classifier_id,
            "frozen_methods": frozen,
        },
        "diagnostic_contract": {
            "methods": list(METHODS),
            "sample_count_per_method": SAMPLE_COUNT,
            "num_classes": NUM_CLASSES,
            "requested_label": "int(zero_based_png_stem) mod 1000",
            "same_index_pairing": "identical zero_based_png_stem",
            "classifier": copy.deepcopy(CLASSIFIER),
            "per_sample_fields": list(PER_SAMPLE_FIELDS),
            "required_statistics": list(REQUIRED_STATISTICS),
            "null_protocol": copy.deepcopy(NULL_PROTOCOL),
            "interpretation_thresholds": copy.deepcopy(INTERPRETATION_THRESHOLDS),
            "interpretation_labels": list(INTERPRETATION_LABELS),
        },
        "output_contract": {
            "output_root": _absolute(output_root, "diagnostic output root"),
            "per_sample_files": {
                "cofitok": "cofitok_predictions.jsonl",
                "dense_identity": "dense_identity_predictions.jsonl",
            },
            "result_file": "class_support_contingency_result.json",
            "validation_file": "class_support_contingency_result.validation.json",
            "exclusive_immutable_creation_required": True,
            "physical_source_rehash_before_inference_required": True,
            "physical_source_rehash_after_inference_required": True,
            "independent_result_replay_required": True,
        },
        "execution_contract": {
            "separate_exact_execution_authorization_required": True,
            "authorization_schema": EXECUTION_AUTHORIZATION_SCHEMA,
            "authorization_must_bind_preparation_sha256": True,
            "authorization_must_bind_evaluator_git": True,
            "authorization_must_bind_all_source_identities": True,
            "no_retraining": True,
            "no_resampling": True,
        },
        "authorization_boundary": copy.deepcopy(PREPARATION_BOUNDARY),
    }
    return validate_preparation(report, expected_output_root=output_root)


def validate_preparation(
    report: Mapping[str, Any], *, expected_output_root: str | None = None
) -> dict[str, Any]:
    row = _object(report, "class-support contingency preparation")
    if set(row) != {
        "schema_version",
        "role",
        "status",
        "scientific_status",
        "terminal_status",
        "generation_advantage_proven",
        "preparation_git",
        "source_evidence",
        "diagnostic_contract",
        "output_contract",
        "execution_contract",
        "authorization_boundary",
    }:
        raise ValueError("class-support contingency preparation fields differ")
    _validate_source_evidence(row.get("source_evidence"))
    _validate_diagnostic_contract(row.get("diagnostic_contract"))
    output = _object(row.get("output_contract"), "output contract")
    execution = _object(row.get("execution_contract"), "execution contract")
    if set(output) != {
        "output_root",
        "per_sample_files",
        "result_file",
        "validation_file",
        "exclusive_immutable_creation_required",
        "physical_source_rehash_before_inference_required",
        "physical_source_rehash_after_inference_required",
        "independent_result_replay_required",
    }:
        raise ValueError("class-support output contract fields differ")
    if set(execution) != {
        "separate_exact_execution_authorization_required",
        "authorization_schema",
        "authorization_must_bind_preparation_sha256",
        "authorization_must_bind_evaluator_git",
        "authorization_must_bind_all_source_identities",
        "no_retraining",
        "no_resampling",
    }:
        raise ValueError("class-support execution contract fields differ")
    per_sample_files = _object(output.get("per_sample_files"), "per-sample files")
    root = _absolute(output.get("output_root"), "diagnostic output root")
    if expected_output_root is not None and root != expected_output_root:
        raise ValueError("diagnostic output root differs")
    if (
        row.get("schema_version") != PREPARATION_SCHEMA
        or row.get("role") != PREPARATION_ROLE
        or row.get("status") != "prepared"
        or row.get("scientific_status") != "diagnostic_not_executed"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != PREPARATION_BOUNDARY
        or output.get("exclusive_immutable_creation_required") is not True
        or output.get("physical_source_rehash_before_inference_required") is not True
        or output.get("physical_source_rehash_after_inference_required") is not True
        or output.get("independent_result_replay_required") is not True
        or per_sample_files
        != {
            "cofitok": "cofitok_predictions.jsonl",
            "dense_identity": "dense_identity_predictions.jsonl",
        }
        or output.get("result_file") != "class_support_contingency_result.json"
        or output.get("validation_file")
        != "class_support_contingency_result.validation.json"
        or execution.get("separate_exact_execution_authorization_required") is not True
        or execution.get("authorization_schema") != EXECUTION_AUTHORIZATION_SCHEMA
        or execution.get("authorization_must_bind_preparation_sha256") is not True
        or execution.get("authorization_must_bind_evaluator_git") is not True
        or execution.get("authorization_must_bind_all_source_identities") is not True
        or execution.get("no_retraining") is not True
        or execution.get("no_resampling") is not True
    ):
        raise ValueError("class-support contingency preparation contract differs")
    validate_git_identity(row.get("preparation_git"), "preparation Git")
    return copy.deepcopy(row)


def validate_execution_authorization(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a separately created execution-only authorization.

    This module intentionally provides no implicit conversion from a
    preparation into an authorization.  A future authorized stage must create
    a distinct source-bound record satisfying this contract.
    """

    row = _object(report, "class-support execution authorization")
    if set(row) != {
        "schema_version",
        "role",
        "status",
        "decision",
        "preparation",
        "stage_approval",
        "stage_approval_record",
        "source_evidence",
        "diagnostic_contract",
        "evaluator_git",
        "authorization_git",
        "authorization_boundary",
    }:
        raise ValueError("class-support execution authorization fields differ")
    prepared = validate_preparation(preparation)
    prepared_id = validate_identity(preparation_identity, "preparation")
    boundary = _object(row.get("authorization_boundary"), "execution boundary")
    expected_boundary = {
        "decision_is_execution_authorization": True,
        "remote_mutation_allowed": True,
        "classifier_inference_allowed": True,
        "gpu_execution_allowed": True,
        "training_launch_allowed": False,
        "sampling_launch_allowed": False,
        "retraining_allowed": False,
        "resampling_allowed": False,
        "confirmation_preparation_allowed": False,
        "confirmation_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "export_allowed": False,
        "release_allowed": False,
        "paper_integration_allowed": False,
        "process_signals_allowed": False,
    }
    if (
        row.get("schema_version") != EXECUTION_AUTHORIZATION_SCHEMA
        or row.get("role") != EXECUTION_AUTHORIZATION_ROLE
        or row.get("status") != "authorized"
        or row.get("decision")
        != "authorize_frozen_sample_class_support_contingency_evaluation_only"
        or row.get("preparation") != prepared_id
        or row.get("source_evidence") != prepared["source_evidence"]
        or row.get("diagnostic_contract") != prepared["diagnostic_contract"]
        or boundary != expected_boundary
    ):
        raise ValueError("class-support execution authorization contract differs")
    evaluator = validate_git_identity(
        row.get("evaluator_git"), "authorized evaluator Git"
    )
    validate_git_identity(row.get("authorization_git"), "authorization Git")
    validate_identity(row.get("stage_approval"), "stage approval")
    validate_stage_approval(
        _object(row.get("stage_approval_record"), "embedded stage approval"),
        preparation_identity=prepared_id,
        evaluator_git=evaluator,
    )
    return copy.deepcopy(row)


def validate_stage_approval(
    report: Mapping[str, Any],
    *,
    preparation_identity: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(report, "class-support stage approval")
    if set(row) != {
        "schema_version",
        "role",
        "status",
        "scope",
        "decision",
        "user_authorized",
        "preparation",
        "evaluator_git",
        "training_launch_allowed",
        "sampling_launch_allowed",
        "retraining_allowed",
        "resampling_allowed",
        "confirmation_preparation_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "export_allowed",
        "release_allowed",
        "paper_integration_allowed",
        "process_signals_allowed",
    }:
        raise ValueError("class-support stage approval fields differ")
    prepared_id = validate_identity(preparation_identity, "approved preparation")
    evaluator = validate_git_identity(evaluator_git, "approved evaluator Git")
    if (
        row.get("schema_version") != STAGE_APPROVAL_SCHEMA
        or row.get("role") != STAGE_APPROVAL_ROLE
        or row.get("status") != "approved"
        or row.get("scope") != STAGE_APPROVAL_SCOPE
        or row.get("decision")
        != "authorize_frozen_sample_class_support_contingency_evaluation_only"
        or row.get("user_authorized") is not True
        or row.get("preparation") != prepared_id
        or row.get("evaluator_git") != evaluator
        or row.get("training_launch_allowed") is not False
        or row.get("sampling_launch_allowed") is not False
        or row.get("retraining_allowed") is not False
        or row.get("resampling_allowed") is not False
        or row.get("confirmation_preparation_allowed") is not False
        or row.get("full_training_launch_allowed") is not False
        or row.get("full_300k_launch_allowed") is not False
        or row.get("promotion_allowed") is not False
        or row.get("export_allowed") is not False
        or row.get("release_allowed") is not False
        or row.get("paper_integration_allowed") is not False
        or row.get("process_signals_allowed") is not False
    ):
        raise ValueError("class-support stage approval contract differs")
    return copy.deepcopy(row)


def build_execution_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_approval: Mapping[str, Any],
    stage_approval_identity: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
    authorization_git: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = validate_preparation(preparation)
    prepared_id = validate_identity(preparation_identity, "preparation")
    evaluator = validate_git_identity(evaluator_git, "evaluator Git")
    approved = validate_stage_approval(
        stage_approval,
        preparation_identity=prepared_id,
        evaluator_git=evaluator,
    )
    approval_id = validate_identity(stage_approval_identity, "stage approval")
    authorization = {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA,
        "role": EXECUTION_AUTHORIZATION_ROLE,
        "status": "authorized",
        "decision": "authorize_frozen_sample_class_support_contingency_evaluation_only",
        "preparation": prepared_id,
        "stage_approval": approval_id,
        "stage_approval_record": approved,
        "source_evidence": copy.deepcopy(prepared["source_evidence"]),
        "diagnostic_contract": copy.deepcopy(prepared["diagnostic_contract"]),
        "evaluator_git": evaluator,
        "authorization_git": validate_git_identity(
            authorization_git, "authorization Git"
        ),
        "authorization_boundary": {
            "decision_is_execution_authorization": True,
            "remote_mutation_allowed": True,
            "classifier_inference_allowed": True,
            "gpu_execution_allowed": True,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "retraining_allowed": False,
            "resampling_allowed": False,
            "confirmation_preparation_allowed": False,
            "confirmation_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "export_allowed": False,
            "release_allowed": False,
            "paper_integration_allowed": False,
            "process_signals_allowed": False,
        },
    }
    return validate_execution_authorization(
        authorization,
        preparation=prepared,
        preparation_identity=prepared_id,
    )


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_count: int = SAMPLE_COUNT,
    num_classes: int = NUM_CLASSES,
) -> list[dict[str, Any]]:
    sample_count = _integer(sample_count, "prediction sample count", minimum=1)
    num_classes = _integer(num_classes, "prediction class count", minimum=5)
    if len(rows) != sample_count:
        raise ValueError("prediction row count differs")
    normalized: list[dict[str, Any]] = []
    expected_fields = {
        "sample_index",
        "filename",
        "requested_class",
        "requested_probability",
        "predicted_top1_class",
        "predicted_top1_probability",
        "predicted_top5_classes",
        "predicted_top5_probabilities",
        "image_sha256",
    }
    for expected_index, raw in enumerate(rows):
        row = _object(raw, f"prediction row {expected_index}")
        if set(row) != expected_fields:
            raise ValueError(f"prediction row {expected_index} fields differ")
        index = _integer(row.get("sample_index"), f"row {expected_index} sample index")
        requested = _integer(row.get("requested_class"), f"row {index} requested class")
        predicted = _integer(row.get("predicted_top1_class"), f"row {index} top1 class")
        top5 = [
            _integer(value, f"row {index} top5 class {position}")
            for position, value in enumerate(
                _sequence(
                    row.get("predicted_top5_classes"), f"row {index} top5 classes"
                )
            )
        ]
        probabilities = [
            _fraction(value, f"row {index} top5 probability")
            for value in _sequence(
                row.get("predicted_top5_probabilities"),
                f"row {index} top5 probabilities",
            )
        ]
        if (
            index != expected_index
            or row.get("filename") != f"{index:06d}.png"
            or requested != index % num_classes
            or not 0 <= predicted < num_classes
            or len(top5) != 5
            or len(set(top5)) != 5
            or any(not 0 <= value < num_classes for value in top5)
            or top5[0] != predicted
            or len(probabilities) != 5
            or any(right > left + 1e-12 for left, right in pairwise(probabilities))
            or math.fsum(probabilities) > 1.0 + 1e-6
        ):
            raise ValueError(f"prediction row {index} label contract differs")
        requested_probability = _fraction(
            row.get("requested_probability"), f"row {index} requested probability"
        )
        top1_probability = _fraction(
            row.get("predicted_top1_probability"), f"row {index} top1 probability"
        )
        if not math.isclose(
            top1_probability, probabilities[0], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"prediction row {index} top1 probability differs")
        if requested in top5:
            position = top5.index(requested)
            if not math.isclose(
                requested_probability,
                probabilities[position],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"prediction row {index} requested probability differs"
                )
        elif requested_probability > probabilities[-1] + 1e-12:
            raise ValueError(
                f"prediction row {index} requested probability exceeds top5 boundary"
            )
        _hex(row.get("image_sha256"), length=64, name=f"row {index} image SHA256")
        normalized.append(
            {
                "sample_index": index,
                "filename": row["filename"],
                "requested_class": requested,
                "requested_probability": requested_probability,
                "predicted_top1_class": predicted,
                "predicted_top1_probability": top1_probability,
                "predicted_top5_classes": top5,
                "predicted_top5_probabilities": probabilities,
                "image_sha256": row["image_sha256"],
            }
        )
    return normalized


def exact_binomial_survival(successes: int, trials: int, probability: float) -> float:
    """Return the non-asymptotic one-sided P[X >= successes]."""

    trials = _integer(trials, "binomial trials", minimum=1)
    successes = _integer(successes, "binomial successes", minimum=0)
    probability = _finite(probability, "binomial probability")
    if successes > trials or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid exact-binomial arguments")
    if successes == 0:
        return 1.0
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    logs = []
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    for value in range(successes, trials + 1):
        logs.append(
            math.lgamma(trials + 1)
            - math.lgamma(value + 1)
            - math.lgamma(trials - value + 1)
            + value * log_p
            + (trials - value) * log_q
        )
    largest = max(logs)
    result = math.exp(largest) * math.fsum(math.exp(value - largest) for value in logs)
    return min(1.0, max(0.0, result))


def _entropy(counts: Iterable[int]) -> float:
    values = []
    for index, raw in enumerate(counts):
        value = _integer(raw, f"histogram count {index}", minimum=0)
        if value > 0:
            values.append(value)
    total = sum(values)
    if total < 1:
        raise ValueError("entropy requires a nonempty histogram")
    return -math.fsum((value / total) * math.log(value / total) for value in values)


def _normalize_count_matrix(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    raw_rows = _sequence(matrix, "contingency matrix")
    if not raw_rows:
        raise ValueError("contingency matrix is malformed")
    normalized: list[list[int]] = []
    column_count: int | None = None
    for row_index, raw_row in enumerate(raw_rows):
        row = _sequence(raw_row, f"contingency row {row_index}")
        if column_count is None:
            column_count = len(row)
            if column_count < 1:
                raise ValueError("contingency matrix is malformed")
        elif len(row) != column_count:
            raise ValueError("contingency matrix is malformed")
        normalized.append(
            [
                _integer(
                    value,
                    f"contingency cell {row_index},{column_index}",
                    minimum=0,
                )
                for column_index, value in enumerate(row)
            ]
        )
    return normalized


def _mutual_information(matrix: Sequence[Sequence[int]]) -> float:
    rows = _normalize_count_matrix(matrix)
    row_totals = [sum(row) for row in rows]
    columns = len(rows[0])
    column_totals = [
        sum(rows[row][column] for row in range(len(rows))) for column in range(columns)
    ]
    total = sum(row_totals)
    if total < 1:
        raise ValueError("contingency matrix is malformed")
    terms = []
    for row_index, row in enumerate(rows):
        for column_index, raw in enumerate(row):
            value = raw
            if value > 0:
                terms.append(
                    (value / total)
                    * math.log(
                        (total * value)
                        / (row_totals[row_index] * column_totals[column_index])
                    )
                )
    return math.fsum(terms)


def _log_comb(total: int, selected: int) -> float:
    if selected < 0 or selected > total:
        return float("-inf")
    return (
        math.lgamma(total + 1)
        - math.lgamma(selected + 1)
        - math.lgamma(total - selected + 1)
    )


def _expected_mutual_information(
    row_totals: Sequence[int], column_totals: Sequence[int]
) -> float:
    rows = [
        _integer(value, f"AMI row total {index}", minimum=0)
        for index, value in enumerate(row_totals)
    ]
    columns = [
        _integer(value, f"AMI column total {index}", minimum=0)
        for index, value in enumerate(column_totals)
    ]
    total = sum(rows)
    if total != sum(columns) or total < 1:
        raise ValueError("AMI marginals differ")
    row_multiplicity = Counter(value for value in rows if value > 0)
    column_multiplicity = Counter(value for value in columns if value > 0)
    result = 0.0
    for left, left_count in row_multiplicity.items():
        for right, right_count in column_multiplicity.items():
            minimum = max(1, left + right - total)
            maximum = min(left, right)
            pair = 0.0
            denominator = _log_comb(total, right)
            for overlap in range(minimum, maximum + 1):
                log_probability = (
                    _log_comb(left, overlap)
                    + _log_comb(total - left, right - overlap)
                    - denominator
                )
                probability = math.exp(log_probability)
                pair += (
                    overlap
                    / total
                    * math.log((total * overlap) / (left * right))
                    * probability
                )
            result += left_count * right_count * pair
    return result


def adjusted_mutual_information(matrix: Sequence[Sequence[int]]) -> dict[str, float]:
    rows = _normalize_count_matrix(matrix)
    row_totals = [sum(row) for row in rows]
    column_totals = [
        sum(rows[row][column] for row in range(len(rows)))
        for column in range(len(rows[0]))
    ]
    mutual_information = _mutual_information(rows)
    expected = _expected_mutual_information(row_totals, column_totals)
    left_entropy = _entropy(row_totals)
    right_entropy = _entropy(column_totals)
    denominator = (left_entropy + right_entropy) / 2.0 - expected
    numerator = mutual_information - expected
    if abs(denominator) <= 1e-15:
        adjusted = 1.0 if abs(numerator) <= 1e-15 else 0.0
    else:
        adjusted = numerator / denominator
    return {
        "mutual_information_nats": mutual_information,
        "expected_mutual_information_nats": expected,
        "requested_entropy_nats": left_entropy,
        "predicted_entropy_nats": right_entropy,
        "adjusted_mutual_information": adjusted,
        "normalization": "arithmetic_mean_entropy",
    }


def _matrix_from_rows(
    rows: Sequence[Mapping[str, Any]], num_classes: int
) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for row in rows:
        matrix[int(row["requested_class"])][int(row["predicted_top1_class"])] += 1
    return matrix


def _sparse_contingency(matrix: Sequence[Sequence[int]]) -> dict[str, Any]:
    sparse = []
    nonzero = 0
    for requested, row in enumerate(matrix):
        entries = [
            {"predicted_class": predicted, "count": int(count)}
            for predicted, count in enumerate(row)
            if int(count) > 0
        ]
        nonzero += len(entries)
        sparse.append(
            {
                "requested_class": requested,
                "total": sum(int(value) for value in row),
                "predicted_counts": entries,
            }
        )
    return {
        "shape": [len(matrix), len(matrix[0]) if matrix else 0],
        "storage": "row_sparse",
        "nonzero_cells": nonzero,
        "rows": sparse,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _empirical_null(observed: float, values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("empirical null requires finite replicates")
    observed = _finite(observed, "empirical-null observation")
    mean = math.fsum(numbers) / len(numbers)
    variance = math.fsum((value - mean) ** 2 for value in numbers) / len(numbers)
    standard_deviation = math.sqrt(variance)
    return {
        "replicates": len(numbers),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "q025": _quantile(numbers, 0.025),
        "q500": _quantile(numbers, 0.5),
        "q975": _quantile(numbers, 0.975),
        "z_score": (
            (observed - mean) / standard_deviation
            if standard_deviation > 0.0
            else (float("inf") if observed > mean else 0.0)
        ),
        "pvalue_greater": (1 + sum(value >= observed for value in numbers))
        / (1 + len(numbers)),
    }


def _ami_permutation_null(
    rows: Sequence[Mapping[str, Any]],
    *,
    observed: float,
    num_classes: int,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    seed = _integer(seed, "AMI null seed")
    replicates = _integer(replicates, "AMI null replicate count", minimum=1)
    requested = [int(row["requested_class"]) for row in rows]
    predicted = [int(row["predicted_top1_class"]) for row in rows]
    base_matrix = _matrix_from_rows(rows, num_classes)
    row_totals = [sum(row) for row in base_matrix]
    column_totals = [
        sum(base_matrix[row][column] for row in range(num_classes))
        for column in range(num_classes)
    ]
    expected = _expected_mutual_information(row_totals, column_totals)
    denominator = (_entropy(row_totals) + _entropy(column_totals)) / 2.0 - expected
    rng = random.Random(seed)
    null = []
    indices = list(range(len(rows)))
    for _ in range(replicates):
        rng.shuffle(indices)
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for index, permuted in enumerate(indices):
            counts[(requested[index], predicted[permuted])] += 1
        terms = []
        for (left, right), count in counts.items():
            terms.append(
                (count / len(rows))
                * math.log(
                    (len(rows) * count) / (row_totals[left] * column_totals[right])
                )
            )
        mi = math.fsum(terms)
        null.append((mi - expected) / denominator if abs(denominator) > 1e-15 else 0.0)
    result = _empirical_null(observed, null)
    result.update({"seed": seed, "scope": "global_predicted_labels"})
    return result


def _cyclic_alignment(
    rows: Sequence[Mapping[str, Any]], num_classes: int
) -> dict[str, Any]:
    num_classes = _integer(num_classes, "cyclic alignment class count", minimum=1)
    if not rows:
        raise ValueError("cyclic alignment requires rows")
    counts = [0] * num_classes
    for row in rows:
        offset = (
            int(row["predicted_top1_class"]) - int(row["requested_class"])
        ) % num_classes
        counts[offset] += 1
    best_count = max(counts)
    best_offset = counts.index(best_count)
    direct = counts[0]
    chance = 1.0 / num_classes
    pvalue = exact_binomial_survival(best_count, len(rows), chance)
    return {
        "definition": "predicted_class = (requested_class + offset) mod num_classes",
        "best_offset": best_offset,
        "best_count": best_count,
        "best_accuracy": best_count / len(rows),
        "direct_offset_zero_count": direct,
        "direct_offset_zero_accuracy": direct / len(rows),
        "lift_over_direct": (best_count - direct) / len(rows),
        "bonferroni_pvalue_greater": min(1.0, pvalue * num_classes),
        "offset_counts": counts,
    }


def _best_mapping(matrix: Sequence[Sequence[int]]) -> tuple[list[int], int]:
    normalized = _normalize_count_matrix(matrix)
    if len(normalized) != len(normalized[0]):
        raise ValueError("one-to-one mapping requires a square matrix")
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - exercised by deployment preflight
        raise RuntimeError(
            "best one-to-one mapping requires numpy and scipy"
        ) from error
    values = np.asarray(normalized, dtype=np.int64)
    rows, columns = linear_sum_assignment(values, maximize=True)
    if len(rows) != values.shape[0]:
        raise RuntimeError("one-to-one mapping solver did not return a full assignment")
    mapping = [-1] * values.shape[0]
    score = 0
    for row, column in zip(rows.tolist(), columns.tolist()):
        mapping[int(row)] = int(column)
        score += int(values[row, column])
    if any(value < 0 for value in mapping) or len(set(mapping)) != len(mapping):
        raise RuntimeError("one-to-one mapping solver returned an invalid assignment")
    return mapping, score


def _mapping_accuracy(
    rows: Sequence[Mapping[str, Any]], mapping: Sequence[int]
) -> tuple[int, float]:
    if (
        not mapping
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in mapping
        )
        or len(set(mapping)) != len(mapping)
        or any(value >= len(mapping) for value in mapping)
    ):
        raise ValueError("one-to-one mapping is malformed")
    count = sum(
        int(row["predicted_top1_class"]) == int(mapping[int(row["requested_class"])])
        for row in rows
    )
    return count, count / len(rows) if rows else 0.0


def _one_to_one_alignment(
    rows: Sequence[Mapping[str, Any]], num_classes: int
) -> dict[str, Any]:
    num_classes = _integer(num_classes, "mapping class count", minimum=1)
    if not rows:
        raise ValueError("one-to-one mapping requires rows")
    full_matrix = _matrix_from_rows(rows, num_classes)
    full_mapping, full_score = _best_mapping(full_matrix)
    folds = []
    heldout_total = 0
    heldout_correct = 0
    heldout_direct_correct = 0
    fold_mappings = []
    for training_fold in (0, 1):
        training = [
            row
            for row in rows
            if (int(row["sample_index"]) // num_classes) % 2 == training_fold
        ]
        heldout = [
            row
            for row in rows
            if (int(row["sample_index"]) // num_classes) % 2 != training_fold
        ]
        if not training or not heldout:
            raise ValueError(
                "two-fold mapping requires nonempty training and held-out folds"
            )
        mapping, training_score = _best_mapping(
            _matrix_from_rows(training, num_classes)
        )
        heldout_score, heldout_accuracy = _mapping_accuracy(heldout, mapping)
        heldout_direct_score = sum(
            int(row["requested_class"]) == int(row["predicted_top1_class"])
            for row in heldout
        )
        fold_mappings.append(mapping)
        heldout_total += len(heldout)
        heldout_correct += heldout_score
        heldout_direct_correct += heldout_direct_score
        folds.append(
            {
                "training_occurrence_parity": training_fold,
                "training_count": len(training),
                "training_matched_count": training_score,
                "training_accuracy": training_score / len(training),
                "heldout_count": len(heldout),
                "heldout_matched_count": heldout_score,
                "heldout_accuracy": heldout_accuracy,
                "heldout_direct_count": heldout_direct_score,
                "heldout_direct_accuracy": heldout_direct_score / len(heldout),
                "mapping": mapping,
            }
        )
    return {
        "solver": "scipy.optimize.linear_sum_assignment_maximize",
        "full_data_mapping": full_mapping,
        "full_data_matched_count": full_score,
        "full_data_accuracy": full_score / len(rows),
        "two_fold_occurrence_parity": folds,
        "cross_validated_matched_count": heldout_correct,
        "cross_validated_count": heldout_total,
        "cross_validated_accuracy": heldout_correct / heldout_total,
        "cross_validated_direct_count": heldout_direct_correct,
        "cross_validated_direct_accuracy": heldout_direct_correct / heldout_total,
        "cross_validated_lift_over_direct": (heldout_correct - heldout_direct_correct)
        / heldout_total,
        "fold_mapping_agreement_fraction": sum(
            left == right for left, right in zip(fold_mappings[0], fold_mappings[1])
        )
        / num_classes,
    }


def _duplicate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["image_sha256"]) for row in rows)
    repeated = {digest: count for digest, count in sorted(counts.items()) if count > 1}
    return {
        "unique_image_count": len(counts),
        "duplicate_excess_image_count": sum(count - 1 for count in repeated.values()),
        "duplicate_pair_count": sum(
            count * (count - 1) // 2 for count in repeated.values()
        ),
        "repeated_content_hash_count": len(repeated),
        "repeated_content": [
            {"image_sha256": digest, "count": count}
            for digest, count in repeated.items()
        ],
    }


def summarize_method(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_classes: int = NUM_CLASSES,
    null_seed: int = int(NULL_PROTOCOL["seed"]),
    null_replicates: int = int(NULL_PROTOCOL["replicates"]),
) -> dict[str, Any]:
    num_classes = _integer(num_classes, "summary class count", minimum=5)
    null_seed = _integer(null_seed, "summary null seed")
    null_replicates = _integer(
        null_replicates, "summary null replicate count", minimum=1
    )
    normalized = validate_prediction_rows(
        rows, sample_count=len(rows), num_classes=num_classes
    )
    matrix = _matrix_from_rows(normalized, num_classes)
    sample_count = len(normalized)
    predicted_histogram = [
        sum(matrix[row][column] for row in range(num_classes))
        for column in range(num_classes)
    ]
    requested_histogram = [sum(row) for row in matrix]
    top1_correct = sum(
        row["requested_class"] == row["predicted_top1_class"] for row in normalized
    )
    top5_correct = sum(
        row["requested_class"] in row["predicted_top5_classes"] for row in normalized
    )
    entropy = _entropy(predicted_histogram)
    probabilities = [value / sample_count for value in predicted_histogram]
    ami = adjusted_mutual_information(matrix)
    ami["deterministic_permutation_null"] = _ami_permutation_null(
        normalized,
        observed=ami["adjusted_mutual_information"],
        num_classes=num_classes,
        seed=null_seed,
        replicates=null_replicates,
    )
    return {
        "sample_count": sample_count,
        "num_classes": num_classes,
        "requested_histogram": requested_histogram,
        "predicted_histogram": predicted_histogram,
        "requested_by_predicted_top1_contingency": _sparse_contingency(matrix),
        "direct_class_fidelity": {
            "top1_correct": top1_correct,
            "top1_accuracy": top1_correct / sample_count,
            "top1_chance_probability": 1.0 / num_classes,
            "top1_exact_binomial_pvalue_greater": exact_binomial_survival(
                top1_correct, sample_count, 1.0 / num_classes
            ),
            "top5_correct": top5_correct,
            "top5_accuracy": top5_correct / sample_count,
            "top5_chance_probability": min(5, num_classes) / num_classes,
            "top5_exact_binomial_pvalue_greater": exact_binomial_survival(
                top5_correct, sample_count, min(5, num_classes) / num_classes
            ),
            "mean_requested_probability": math.fsum(
                row["requested_probability"] for row in normalized
            )
            / sample_count,
        },
        "adjusted_mutual_information": ami,
        "predicted_class_support": {
            "predicted_class_count": sum(value > 0 for value in predicted_histogram),
            "predicted_class_fraction": sum(value > 0 for value in predicted_histogram)
            / num_classes,
            "entropy_nats": entropy,
            "normalized_entropy": entropy / math.log(num_classes),
            "herfindahl_concentration": math.fsum(
                value * value for value in probabilities
            ),
            "effective_class_count": math.exp(entropy),
            "top_mode_class": predicted_histogram.index(max(predicted_histogram)),
            "top_mode_count": max(predicted_histogram),
            "top_mode_mass": max(predicted_histogram) / sample_count,
        },
        "best_cyclic_offset_alignment": _cyclic_alignment(normalized, num_classes),
        "best_one_to_one_mapping": _one_to_one_alignment(normalized, num_classes),
        "exact_duplicates": _duplicate_summary(normalized),
    }


def _histogram_comparison(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    left_values = [
        _integer(value, f"left histogram count {index}", minimum=0)
        for index, value in enumerate(left)
    ]
    right_values = [
        _integer(value, f"right histogram count {index}", minimum=0)
        for index, value in enumerate(right)
    ]
    if (
        len(left_values) != len(right_values)
        or not left_values
        or sum(left_values) < 1
        or sum(right_values) < 1
    ):
        raise ValueError("histogram comparison inputs differ")
    left_total = sum(left_values)
    right_total = sum(right_values)
    p = [value / left_total for value in left_values]
    q = [value / right_total for value in right_values]
    middle = [(a + b) / 2.0 for a, b in zip(p, q)]

    def divergence(values: Sequence[float]) -> float:
        return math.fsum(
            value * math.log(value / reference)
            for value, reference in zip(values, middle)
            if value > 0.0
        )

    js_nats = (divergence(p) + divergence(q)) / 2.0
    return {
        "histogram_overlap": math.fsum(min(a, b) for a, b in zip(p, q)),
        "jensen_shannon_divergence_nats": js_nats,
        "jensen_shannon_divergence_bits": js_nats / math.log(2.0),
    }


def _same_index_agreement(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    num_classes: int,
) -> dict[str, Any]:
    seed = _integer(seed, "same-index null seed")
    replicates = _integer(replicates, "same-index null replicate count", minimum=1)
    num_classes = _integer(num_classes, "same-index class count", minimum=1)
    if len(left) != len(right):
        raise ValueError("same-index methods have different sample counts")
    if not left:
        raise ValueError("same-index agreement requires rows")
    if any(
        int(left_row["sample_index"]) != int(right_row["sample_index"])
        or int(left_row["requested_class"]) != int(right_row["requested_class"])
        for left_row, right_row in zip(left, right)
    ):
        raise ValueError("same-index methods are not aligned")
    observed_count = sum(
        int(a["predicted_top1_class"]) == int(b["predicted_top1_class"])
        for a, b in zip(left, right)
    )
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(left):
        by_class[int(row["requested_class"])].append(index)
    if set(by_class) != set(range(num_classes)):
        raise ValueError("same-index null lacks a requested class")
    rng = random.Random(seed)
    null = []
    for _ in range(replicates):
        matches = 0
        for requested in range(num_classes):
            positions = by_class[requested]
            permuted = positions.copy()
            rng.shuffle(permuted)
            matches += sum(
                int(left[source]["predicted_top1_class"])
                == int(right[target]["predicted_top1_class"])
                for source, target in zip(positions, permuted)
            )
        null.append(matches / len(left))
    observed = observed_count / len(left)
    return {
        "same_index_count": observed_count,
        "same_index_fraction": observed,
        "null_scope": "dense predictions permuted within requested class",
        "null_seed": seed,
        **_empirical_null(observed, null),
    }


def _cross_method_duplicates(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left_counts = Counter(str(row["image_sha256"]) for row in left)
    right_counts = Counter(str(row["image_sha256"]) for row in right)
    shared = sorted(set(left_counts) & set(right_counts))
    same_index = sum(
        left_row["image_sha256"] == right_row["image_sha256"]
        for left_row, right_row in zip(left, right)
    )
    return {
        "shared_content_hash_count": len(shared),
        "cofitok_images_on_shared_hashes": sum(
            left_counts[digest] for digest in shared
        ),
        "dense_images_on_shared_hashes": sum(right_counts[digest] for digest in shared),
        "cross_method_duplicate_pair_count": sum(
            left_counts[digest] * right_counts[digest] for digest in shared
        ),
        "same_index_exact_duplicate_count": same_index,
        "shared_content": [
            {
                "image_sha256": digest,
                "cofitok_count": left_counts[digest],
                "dense_identity_count": right_counts[digest],
            }
            for digest in shared
        ],
    }


def _interpret(
    methods: Mapping[str, Mapping[str, Any]],
    cross: Mapping[str, Any],
    *,
    existing_recall: Mapping[str, float],
    num_classes: int,
) -> dict[str, Any]:
    threshold = INTERPRETATION_THRESHOLDS
    method_flags: dict[str, dict[str, bool]] = {}
    for method in METHODS:
        summary = methods[method]
        direct = summary["direct_class_fidelity"]
        ami = summary["adjusted_mutual_information"]
        cyclic = summary["best_cyclic_offset_alignment"]
        mapping = summary["best_one_to_one_mapping"]
        method_flags[method] = {
            "direct_practical": (
                direct["top1_accuracy"] >= threshold["min_direct_top1_accuracy"]
                and direct["top5_accuracy"] >= threshold["min_direct_top5_accuracy"]
            ),
            "dependence_detected": (
                ami["adjusted_mutual_information"]
                >= threshold["min_adjusted_mutual_information"]
                and ami["deterministic_permutation_null"]["pvalue_greater"]
                <= threshold["significance_alpha"]
            ),
            "cyclic_relabeling_detected": (
                cyclic["best_accuracy"] >= threshold["min_cyclic_top1_accuracy"]
                and cyclic["lift_over_direct"]
                >= threshold["min_cyclic_lift_over_direct"]
                and cyclic["bonferroni_pvalue_greater"]
                <= threshold["significance_alpha"]
            ),
            "one_to_one_relabeling_detected": (
                mapping["cross_validated_accuracy"]
                >= threshold["min_cross_validated_mapping_accuracy"]
                and mapping["cross_validated_lift_over_direct"]
                >= threshold["min_mapping_lift_over_direct"]
            ),
        }
    common_cyclic = methods["cofitok"]["best_cyclic_offset_alignment"][
        "best_offset"
    ] == methods["dense_identity"]["best_cyclic_offset_alignment"][
        "best_offset"
    ] and all(method_flags[method]["cyclic_relabeling_detected"] for method in METHODS)
    common_mapping_fraction = (
        sum(
            left == right
            for left, right in zip(
                methods["cofitok"]["best_one_to_one_mapping"]["full_data_mapping"],
                methods["dense_identity"]["best_one_to_one_mapping"][
                    "full_data_mapping"
                ],
            )
        )
        / num_classes
    )
    shared_distribution = (
        cross["predicted_histograms"]["histogram_overlap"]
        >= threshold["min_shared_histogram_overlap"]
        and cross["predicted_histograms"]["jensen_shannon_divergence_bits"]
        <= threshold["max_shared_histogram_jsd_bits"]
    )
    low_recall = all(
        _fraction(existing_recall[method], f"{method} existing recall")
        <= threshold["max_existing_recall_for_support_collapse"]
        for method in METHODS
    )
    if common_cyclic or all(
        method_flags[method]["one_to_one_relabeling_detected"] for method in METHODS
    ):
        label = "permuted_or_offset_conditioning"
    elif any(method_flags[method]["dependence_detected"] for method in METHODS):
        label = "active_but_misaligned_conditioning"
    elif shared_distribution and low_recall:
        label = "shared_unconditional_support_collapse"
    else:
        label = "ignored_conditioning"
    return {
        "primary_interpretation": label,
        "method_flags": method_flags,
        "common_cyclic_offset_detected": common_cyclic,
        "full_mapping_agreement_fraction_between_methods": common_mapping_fraction,
        "shared_predicted_distribution": shared_distribution,
        "existing_low_recall_both_methods": low_recall,
        "thresholds": copy.deepcopy(INTERPRETATION_THRESHOLDS),
        "claim_boundary": (
            "This deterministic diagnostic distinguishes label dependence and relabeling "
            "patterns in frozen samples. It does not establish a training-time cause, "
            "validate a repair, or authorize any later stage."
        ),
    }


def build_analysis(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    existing_recall: Mapping[str, float],
    num_classes: int = NUM_CLASSES,
    null_seed: int = int(NULL_PROTOCOL["seed"]),
    null_replicates: int = int(NULL_PROTOCOL["replicates"]),
) -> dict[str, Any]:
    num_classes = _integer(num_classes, "analysis class count", minimum=5)
    null_seed = _integer(null_seed, "analysis null seed")
    null_replicates = _integer(
        null_replicates, "analysis null replicate count", minimum=1
    )
    if set(rows_by_method) != set(METHODS):
        raise ValueError("prediction method set differs")
    if set(existing_recall) != set(METHODS):
        raise ValueError("existing recall method set differs")
    normalized = {
        method: validate_prediction_rows(
            rows_by_method[method],
            sample_count=len(rows_by_method[method]),
            num_classes=num_classes,
        )
        for method in METHODS
    }
    if len(normalized["cofitok"]) != len(normalized["dense_identity"]):
        raise ValueError("prediction methods have different sample counts")
    sample_count = len(normalized["cofitok"])
    if sample_count % num_classes != 0 or sample_count < 2 * num_classes:
        raise ValueError(
            "analysis requires at least two complete balanced class occurrences"
        )
    methods = {
        method: summarize_method(
            normalized[method],
            num_classes=num_classes,
            null_seed=null_seed + index,
            null_replicates=null_replicates,
        )
        for index, method in enumerate(METHODS)
    }
    cross = {
        "predicted_histograms": _histogram_comparison(
            methods["cofitok"]["predicted_histogram"],
            methods["dense_identity"]["predicted_histogram"],
        ),
        "same_index_top1_agreement": _same_index_agreement(
            normalized["cofitok"],
            normalized["dense_identity"],
            seed=null_seed + 10_000,
            replicates=null_replicates,
            num_classes=num_classes,
        ),
        "exact_duplicates": _cross_method_duplicates(
            normalized["cofitok"], normalized["dense_identity"]
        ),
    }
    interpretation = _interpret(
        methods,
        cross,
        existing_recall=existing_recall,
        num_classes=num_classes,
    )
    return {
        "methods": methods,
        "cross_method": cross,
        "interpretation": interpretation,
        "analysis_canonical_sha256": canonical_sha256(
            {
                "methods": methods,
                "cross_method": cross,
                "interpretation": interpretation,
            }
        ),
    }


def _validate_analysis_contract(value: Any) -> dict[str, Any]:
    analysis = _object(value, "class-support analysis")
    if set(analysis) != {
        "methods",
        "cross_method",
        "interpretation",
        "analysis_canonical_sha256",
    }:
        raise ValueError("class-support analysis fields differ")
    if analysis["analysis_canonical_sha256"] != canonical_sha256(
        {
            "methods": analysis["methods"],
            "cross_method": analysis["cross_method"],
            "interpretation": analysis["interpretation"],
        }
    ):
        raise ValueError("class-support analysis canonical SHA256 differs")
    methods = _object(analysis["methods"], "class-support method analysis")
    if set(methods) != set(METHODS):
        raise ValueError("class-support method analysis differs")
    expected_requested = [SAMPLE_COUNT // NUM_CLASSES] * NUM_CLASSES
    for method in METHODS:
        summary = _object(methods[method], f"{method} analysis")
        if set(summary) != {
            "sample_count",
            "num_classes",
            "requested_histogram",
            "predicted_histogram",
            "requested_by_predicted_top1_contingency",
            "direct_class_fidelity",
            "adjusted_mutual_information",
            "predicted_class_support",
            "best_cyclic_offset_alignment",
            "best_one_to_one_mapping",
            "exact_duplicates",
        }:
            raise ValueError(f"{method} analysis fields differ")
        if (
            summary.get("sample_count") != SAMPLE_COUNT
            or summary.get("num_classes") != NUM_CLASSES
            or summary.get("requested_histogram") != expected_requested
        ):
            raise ValueError(f"{method} analysis accounting differs")
        predicted = _sequence(
            summary.get("predicted_histogram"), f"{method} predicted histogram"
        )
        if (
            len(predicted) != NUM_CLASSES
            or sum(
                _integer(value, f"{method} predicted count {index}", minimum=0)
                for index, value in enumerate(predicted)
            )
            != SAMPLE_COUNT
        ):
            raise ValueError(f"{method} predicted histogram differs")
        direct = _object(summary["direct_class_fidelity"], f"{method} direct")
        if (
            _integer(direct.get("top1_correct"), f"{method} top1 count", minimum=0)
            > SAMPLE_COUNT
            or _integer(direct.get("top5_correct"), f"{method} top5 count", minimum=0)
            > SAMPLE_COUNT
        ):
            raise ValueError(f"{method} direct class-fidelity count differs")
        for field in ("top1_accuracy", "top5_accuracy", "mean_requested_probability"):
            _fraction(direct.get(field), f"{method} {field}")
        ami = _object(summary["adjusted_mutual_information"], f"{method} AMI")
        null = _object(ami.get("deterministic_permutation_null"), f"{method} AMI null")
        if null.get("replicates") != int(NULL_PROTOCOL["replicates"]):
            raise ValueError(f"{method} AMI null replicate count differs")
        mapping = _object(
            summary["best_one_to_one_mapping"], f"{method} one-to-one mapping"
        )
        if (
            mapping.get("cross_validated_count") != SAMPLE_COUNT
            or mapping.get("cross_validated_direct_count") != direct["top1_correct"]
        ):
            raise ValueError(f"{method} mapping accounting differs")
    cross = _object(analysis["cross_method"], "cross-method analysis")
    if set(cross) != {
        "predicted_histograms",
        "same_index_top1_agreement",
        "exact_duplicates",
    }:
        raise ValueError("cross-method analysis fields differ")
    agreement = _object(cross["same_index_top1_agreement"], "same-index agreement")
    if _integer(
        agreement.get("same_index_count"), "same-index count", minimum=0
    ) > SAMPLE_COUNT or agreement.get("replicates") != int(NULL_PROTOCOL["replicates"]):
        raise ValueError("same-index agreement contract differs")
    duplicates = _object(cross["exact_duplicates"], "cross-method duplicates")
    if (
        _integer(
            duplicates.get("same_index_exact_duplicate_count"),
            "same-index exact duplicate count",
            minimum=0,
        )
        > SAMPLE_COUNT
    ):
        raise ValueError("cross-method duplicate accounting differs")
    interpretation = _object(analysis["interpretation"], "analysis interpretation")
    if (
        interpretation.get("primary_interpretation") not in set(INTERPRETATION_LABELS)
        or interpretation.get("thresholds") != INTERPRETATION_THRESHOLDS
    ):
        raise ValueError("analysis interpretation differs")
    return copy.deepcopy(analysis)


def build_result(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    execution_authorization_identity: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
    prediction_identities: Mapping[str, Mapping[str, Any]],
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    existing_recall: Mapping[str, float],
    runtime: Mapping[str, Any],
    elapsed_started_at: float | None = None,
) -> dict[str, Any]:
    """Build the immutable result and optionally close its elapsed timer.

    When ``elapsed_started_at`` is provided, ``runtime`` must omit
    ``elapsed_seconds``.  The timer is stamped only after the deterministic
    statistical analysis completes, so the reported duration covers both
    classifier work performed by the caller and this module's analysis.
    """

    prepared = validate_preparation(preparation)
    prepared_id = validate_identity(preparation_identity, "preparation")
    authorization = validate_execution_authorization(
        execution_authorization,
        preparation=prepared,
        preparation_identity=prepared_id,
    )
    authorization_id = validate_identity(
        execution_authorization_identity, "execution authorization"
    )
    evaluator = validate_git_identity(evaluator_git, "evaluator Git")
    if authorization["evaluator_git"] != evaluator:
        raise ValueError("evaluator Git differs from the execution authorization")
    if set(prediction_identities) != set(METHODS):
        raise ValueError("prediction identity method set differs")
    if set(rows_by_method) != set(METHODS):
        raise ValueError("prediction method set differs")
    if any(len(rows_by_method[method]) != SAMPLE_COUNT for method in METHODS):
        raise ValueError("prediction rows must contain the exact frozen sample count")
    prediction_ids = {
        method: validate_identity(
            prediction_identities[method], f"{method} predictions"
        )
        for method in METHODS
    }
    expected_output_root = Path(prepared["output_contract"]["output_root"])
    for method in METHODS:
        expected_prediction_path = (
            expected_output_root
            / prepared["output_contract"]["per_sample_files"][method]
        ).as_posix()
        if prediction_ids[method]["path"] != expected_prediction_path:
            raise ValueError(f"{method} prediction path differs from preparation")
    if set(existing_recall) != set(METHODS):
        raise ValueError("existing recall method set differs")
    recall = {
        method: _fraction(existing_recall[method], f"{method} existing recall")
        for method in METHODS
    }
    expected_recall = {
        method: float(
            prepared["source_evidence"]["frozen_methods"][method][
                "existing_generation_metrics"
            ]["recall"]
        )
        for method in METHODS
    }
    if recall != expected_recall:
        raise ValueError("existing recall differs from the frozen metric sources")
    if elapsed_started_at is None:
        runtime_row = _validate_runtime(runtime)
        runtime_template = None
    else:
        started_at = _finite(elapsed_started_at, "diagnostic elapsed start")
        if started_at <= 0.0:
            raise ValueError("diagnostic elapsed start must be positive")
        runtime_template = _object(runtime, "diagnostic runtime template")
        if set(runtime_template) != RUNTIME_FIELDS - {"elapsed_seconds"}:
            raise ValueError("diagnostic runtime template fields differ")
        _validate_runtime({**runtime_template, "elapsed_seconds": 1.0})
        runtime_row = None
    analysis = build_analysis(
        rows_by_method,
        existing_recall=recall,
        num_classes=NUM_CLASSES,
        null_seed=int(NULL_PROTOCOL["seed"]),
        null_replicates=int(NULL_PROTOCOL["replicates"]),
    )
    if runtime_template is not None:
        runtime_row = _validate_runtime(
            {
                **runtime_template,
                "elapsed_seconds": time.perf_counter() - started_at,
            }
        )
    if runtime_row is None:  # pragma: no cover - guarded by the branches above
        raise RuntimeError("diagnostic runtime row was not constructed")
    result = {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "scientific_status": "diagnostic_completed_non_authorizing",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "preparation": prepared_id,
        "execution_authorization": authorization_id,
        "evaluator_git": evaluator,
        "source_evidence": copy.deepcopy(prepared["source_evidence"]),
        "diagnostic_contract": copy.deepcopy(prepared["diagnostic_contract"]),
        "prediction_files": prediction_ids,
        "existing_generation_recall": recall,
        "analysis": analysis,
        "runtime": copy.deepcopy(runtime_row),
        "claim_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }
    return validate_result(
        result,
        preparation=prepared,
        preparation_identity=prepared_id,
    )


def validate_result(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any] | None = None,
    preparation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = _object(report, "class-support contingency result")
    if set(row) != {
        "schema_version",
        "role",
        "status",
        "scientific_status",
        "terminal_status",
        "generation_advantage_proven",
        "preparation",
        "execution_authorization",
        "evaluator_git",
        "source_evidence",
        "diagnostic_contract",
        "prediction_files",
        "existing_generation_recall",
        "analysis",
        "runtime",
        "claim_boundary",
    }:
        raise ValueError("class-support contingency result fields differ")
    source = _validate_source_evidence(row.get("source_evidence"))
    _validate_diagnostic_contract(row.get("diagnostic_contract"))
    if (
        row.get("schema_version") != RESULT_SCHEMA
        or row.get("role") != RESULT_ROLE
        or row.get("status") != "completed"
        or row.get("scientific_status") != "diagnostic_completed_non_authorizing"
        or row.get("terminal_status") != "hold"
        or row.get("generation_advantage_proven") is not False
        or row.get("claim_boundary") != RESULT_BOUNDARY
    ):
        raise ValueError("class-support contingency result contract differs")
    validate_identity(row.get("preparation"), "result preparation")
    validate_identity(row.get("execution_authorization"), "result authorization")
    validate_git_identity(row.get("evaluator_git"), "result evaluator Git")
    prediction_files = _object(row.get("prediction_files"), "result predictions")
    if set(prediction_files) != set(METHODS):
        raise ValueError("result prediction files differ")
    for method in METHODS:
        validate_identity(prediction_files[method], f"result {method} predictions")
    recall = _object(row.get("existing_generation_recall"), "result existing recall")
    if set(recall) != set(METHODS):
        raise ValueError("result existing recall method set differs")
    normalized_recall = {
        method: _fraction(recall[method], f"result {method} existing recall")
        for method in METHODS
    }
    expected_recall = {
        method: float(
            source["frozen_methods"][method]["existing_generation_metrics"]["recall"]
        )
        for method in METHODS
    }
    if normalized_recall != expected_recall:
        raise ValueError("result existing recall differs from frozen evidence")
    _validate_analysis_contract(row.get("analysis"))
    _validate_runtime(row.get("runtime"))
    if preparation is not None or preparation_identity is not None:
        if preparation is None or preparation_identity is None:
            raise ValueError("preparation and identity must be paired")
        prepared = validate_preparation(preparation)
        prepared_id = validate_identity(preparation_identity, "expected preparation")
        if row["preparation"] != prepared_id:
            raise ValueError("result preparation binding differs")
        if row["source_evidence"] != prepared["source_evidence"]:
            raise ValueError("result source evidence differs from preparation")
        if row["diagnostic_contract"] != prepared["diagnostic_contract"]:
            raise ValueError("result diagnostic contract differs from preparation")
        expected_root = Path(prepared["output_contract"]["output_root"])
        for method in METHODS:
            expected_path = (
                expected_root / prepared["output_contract"]["per_sample_files"][method]
            ).as_posix()
            if prediction_files[method]["path"] != expected_path:
                raise ValueError(f"result {method} prediction path differs")
    return copy.deepcopy(row)


def build_validation_receipt(
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
    replayed_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_result(result)
    identity = validate_identity(result_identity, "result")
    validator = validate_git_identity(validator_git, "validator Git")
    if replayed_analysis != validated["analysis"]:
        raise ValueError(
            "independent prediction replay does not reproduce the analysis"
        )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": validated["scientific_status"],
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "result": identity,
        "validator_git": validator,
        "prediction_files": copy.deepcopy(validated["prediction_files"]),
        "physical_prediction_replay": {
            "performed": True,
            "method_count": len(METHODS),
            "sample_count_per_method": SAMPLE_COUNT,
            "recomputed_analysis_canonical_sha256": replayed_analysis[
                "analysis_canonical_sha256"
            ],
        },
        "claim_boundary": copy.deepcopy(RESULT_BOUNDARY),
    }


def validate_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "class-support contingency validation")
    if set(row) != {
        "schema_version",
        "role",
        "status",
        "scientific_status",
        "terminal_status",
        "generation_advantage_proven",
        "result",
        "validator_git",
        "prediction_files",
        "physical_prediction_replay",
        "claim_boundary",
    }:
        raise ValueError("class-support contingency validation fields differ")
    expected = build_validation_receipt(
        result=result,
        result_identity=result_identity,
        validator_git=validate_git_identity(row.get("validator_git"), "validator Git"),
        replayed_analysis=_object(result, "result")["analysis"],
    )
    if row != expected:
        raise ValueError("class-support contingency validation receipt differs")
    return expected


__all__ = [
    "CAUSAL_SOURCE_SHA256",
    "CLASSIFIER",
    "EXECUTION_AUTHORIZATION_ROLE",
    "EXECUTION_AUTHORIZATION_SCHEMA",
    "FROZEN_SOURCE_SPECS",
    "IMAGE_SHAPE",
    "INTERPRETATION_LABELS",
    "INTERPRETATION_THRESHOLDS",
    "METHODS",
    "NULL_PROTOCOL",
    "NUM_CLASSES",
    "PER_SAMPLE_FIELDS",
    "PREPARATION_BOUNDARY",
    "PREPARATION_ROLE",
    "PREPARATION_SCHEMA",
    "REQUIRED_STATISTICS",
    "RESULT_BOUNDARY",
    "RESULT_ROLE",
    "RESULT_SCHEMA",
    "RUNTIME_FIELDS",
    "SAMPLE_COUNT",
    "STAGE_APPROVAL_ROLE",
    "STAGE_APPROVAL_SCHEMA",
    "STAGE_APPROVAL_SCOPE",
    "VALIDATION_ROLE",
    "VALIDATION_SCHEMA",
    "adjusted_mutual_information",
    "build_analysis",
    "build_execution_authorization",
    "build_preparation",
    "build_result",
    "build_validation_receipt",
    "canonical_sha256",
    "exact_binomial_survival",
    "physical_sample_tree_identity",
    "stable_file_identity",
    "summarize_method",
    "validate_execution_authorization",
    "validate_git_identity",
    "validate_identity",
    "validate_prediction_rows",
    "validate_preparation",
    "validate_result",
    "validate_stage_approval",
    "validate_validation_receipt",
]
