from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cofitok.environment import (
    capture_runtime_environment,
    runtime_environment_sha256,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA, image_tree_sha256
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_metrics import (
    find_images,
    validate_sampling_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_frozen_existing_sample_support_audit"
DESCRIPTOR_SCHEMA = "rgb_8x8_bilinear_support_descriptor_v1"
PAIRING_SCHEMA = "balanced_modulo_pass_major_v1"

FROZEN_PROMOTION_GATE_SHA256 = (
    "2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90"
)
FROZEN_DATASET_MANIFEST_SHA256 = (
    "9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0"
)
FROZEN_REAL_SET_SHA256 = (
    "19ace4e37bee2fcaeac2b7cbd26ed785aa0e6e01015c90b4955027d163f6e44f"
)


@dataclass(frozen=True)
class FrozenSupportContract:
    promotion_gate_sha256: str = FROZEN_PROMOTION_GATE_SHA256
    dataset_manifest_sha256: str = FROZEN_DATASET_MANIFEST_SHA256
    dataset_manifest_bytes: int = 405_484_553
    real_set_sha256: str = FROZEN_REAL_SET_SHA256
    class_count: int = 1_000
    real_images_per_class: int = 50
    generated_sample_count: int = 10_000
    image_width: int = 256
    image_height: int = 256
    checkpoint_step: int = 50_000
    cofitok_prefix_budget: int = 8
    dense_prefix_budget: int = 1
    num_train_timesteps: int = 1_000
    sample_steps: int = 100
    guidance_scale: float = 1.5
    guidance_rescale: float = 0.0
    batch_size: int = 32
    metrics_seed: int = 2027


@dataclass
class ImageSetAnalysis:
    report: dict[str, Any]
    descriptors: np.ndarray
    dhashes: np.ndarray


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_contract(contract: FrozenSupportContract) -> int:
    for name in (
        "promotion_gate_sha256",
        "dataset_manifest_sha256",
        "real_set_sha256",
    ):
        if not _is_sha256(getattr(contract, name)):
            raise ValueError(f"frozen support contract {name} is malformed")
    positive = (
        "dataset_manifest_bytes",
        "class_count",
        "real_images_per_class",
        "generated_sample_count",
        "image_width",
        "image_height",
        "checkpoint_step",
        "cofitok_prefix_budget",
        "dense_prefix_budget",
        "num_train_timesteps",
        "sample_steps",
        "batch_size",
    )
    if any(int(getattr(contract, name)) < 1 for name in positive):
        raise ValueError("frozen support contract integer fields must be positive")
    if contract.class_count < 2:
        raise ValueError("frozen support audit requires at least two classes")
    if contract.generated_sample_count % contract.class_count:
        raise ValueError("generated sample count must divide evenly across classes")
    samples_per_class = contract.generated_sample_count // contract.class_count
    if samples_per_class < 2:
        raise ValueError("frozen support audit requires at least two samples per class")
    if samples_per_class > contract.real_images_per_class:
        raise ValueError("real set has too few images for matched class support")
    for name in ("guidance_scale", "guidance_rescale"):
        value = float(getattr(contract, name))
        if not math.isfinite(value):
            raise ValueError(f"frozen support contract {name} must be finite")
    if not 0.0 <= float(contract.guidance_rescale) <= 1.0:
        raise ValueError("frozen support contract guidance_rescale is outside [0, 1]")
    return samples_per_class


def _identity_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(name) == expected.get(name) for name in ("path", "bytes", "sha256"))


def _gate_row(gate: dict[str, Any], name: str) -> dict[str, Any]:
    rows = gate.get("gates")
    if not isinstance(rows, list):
        raise ValueError("frozen promotion gate rows are missing")
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"frozen promotion gate row {name} is missing or duplicated")
    return matches[0]


def _validate_frozen_gate(
    path: Path,
    *,
    contract: FrozenSupportContract,
    cofitok_metrics_identity: dict[str, Any],
    dense_metrics_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != contract.promotion_gate_sha256:
        raise ValueError("frozen promotion gate SHA256 differs from the audit contract")
    gate = read_json_object(path, name="frozen promotion gate")
    if (
        gate.get("schema_version") != 2
        or gate.get("status") != "fail"
        or gate.get("decision") != "hold"
        or gate.get("stage") != "scaling"
        or gate.get("source_profile") != "stability_scaling"
    ):
        raise ValueError("frozen promotion gate terminal contract differs")
    failed = sorted(
        row.get("name")
        for row in gate.get("gates", [])
        if isinstance(row, dict) and row.get("passed") is False
    )
    if failed != ["absolute_fid_quality"]:
        raise ValueError("frozen promotion gate failure set differs")
    absolute_fid = _gate_row(gate, "absolute_fid_quality")
    if absolute_fid.get("passed") is not False:
        raise ValueError("frozen absolute-FID gate is not failed")
    sources = gate.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError("frozen promotion gate source reports are missing")
    for name, actual in (
        ("cofitok_generation", cofitok_metrics_identity),
        ("dense_generation", dense_metrics_identity),
    ):
        expected = sources.get(name)
        if not isinstance(expected, dict) or not _identity_matches(actual, expected):
            raise ValueError(f"frozen promotion gate {name} identity differs")
    provenance = gate.get("provenance_contract")
    if (
        not isinstance(provenance, dict)
        or len(str(provenance.get("evaluation_revision", ""))) != 40
        or not str(provenance.get("evaluation_branch", ""))
        or len(str(provenance.get("training_revision", ""))) != 40
        or not str(provenance.get("training_branch", ""))
    ):
        raise ValueError("frozen promotion gate provenance contract is malformed")
    thresholds = gate.get("thresholds")
    if (
        not isinstance(thresholds, dict)
        or int(thresholds.get("min_samples", -1)) != contract.generated_sample_count
    ):
        raise ValueError("frozen promotion gate sample threshold differs")
    return gate, identity


def _sampling_contract_fields(
    provenance: dict[str, Any],
    *,
    contract: FrozenSupportContract,
    expected_budget: int,
) -> dict[str, Any]:
    sampling = provenance.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("sampling provenance is missing its protocol")
    random_stream = sampling.get("random_stream")
    sample_digest = sampling.get("sample_set_digest")
    expected = {
        "num_samples": contract.generated_sample_count,
        "num_train_timesteps": contract.num_train_timesteps,
        "sample_steps": contract.sample_steps,
        "guidance_scale": contract.guidance_scale,
        "guidance_rescale": contract.guidance_rescale,
        "batch_size": contract.batch_size,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "precision": "bf16",
        "seed": 0,
        "start_index": 0,
        "class_schedule": "balanced_modulo",
        "image_shape": [3, contract.image_height, contract.image_width],
        "prefix_budgets": [expected_budget],
    }
    differing = [name for name, value in expected.items() if sampling.get(name) != value]
    if differing:
        raise ValueError(
            "frozen sampling protocol differs: " + ", ".join(sorted(differing))
        )
    if random_stream != {
        "scope": "per_global_sample_index",
        "seed_formula": "(seed + global_index) mod 2^63",
        "prefix_budgets_share_stream": True,
        "batch_size_invariant": True,
        "resume_index_invariant": True,
    }:
        raise ValueError("frozen sampling random-stream contract differs")
    if sample_digest != {
        "algorithm": "sha256",
        "framing": "filename_utf8_nul_file_bytes_nul",
    }:
        raise ValueError("frozen sample-set digest contract differs")
    if (
        provenance.get("weights") != "ema"
        or int(provenance.get("checkpoint_step", -1)) != contract.checkpoint_step
        or int(provenance.get("selected_prefix_budget", -1)) != expected_budget
        or provenance.get("image_shape")
        != [3, contract.image_height, contract.image_width]
    ):
        raise ValueError("frozen sampling checkpoint or prefix contract differs")
    return sampling


def _validate_matched_sampling(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
    *,
    contract: FrozenSupportContract,
) -> None:
    cofitok_sampling = _sampling_contract_fields(
        cofitok,
        contract=contract,
        expected_budget=contract.cofitok_prefix_budget,
    )
    dense_sampling = _sampling_contract_fields(
        dense,
        contract=contract,
        expected_budget=contract.dense_prefix_budget,
    )
    ignored = {"prefix_budgets"}
    if {
        key: value for key, value in cofitok_sampling.items() if key not in ignored
    } != {key: value for key, value in dense_sampling.items() if key not in ignored}:
        raise ValueError("CoFiTok and dense frozen sampling protocols are not matched")
    for name in ("git", "runtime_environment_sha256"):
        if cofitok.get(name) != dense.get(name):
            raise ValueError(f"CoFiTok and dense frozen sampling {name} differs")
    if cofitok.get("sample_set_sha256") == dense.get("sample_set_sha256"):
        raise ValueError("CoFiTok and dense frozen sample sets unexpectedly share a digest")


def _finite_metric(metrics: dict[str, Any], name: str) -> float:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"frozen metric {name} is missing") from error
    if not math.isfinite(value):
        raise ValueError(f"frozen metric {name} is not finite")
    return value


def _validate_metrics_report(
    path: Path,
    *,
    method: str,
    generated_dir: Path,
    real_dir: Path,
    real_set: dict[str, Any],
    sampling: dict[str, Any],
    contract: FrozenSupportContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    report = read_json_object(path, name=f"{method} frozen metrics report")
    if (
        report.get("schema_version") != 2
        or report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("implementation")
        != {"package": "torch_fidelity", "version": "0.4.0"}
    ):
        raise ValueError(f"{method} frozen metrics report contract differs")
    paths = report.get("paths")
    counts = report.get("counts")
    if (
        not isinstance(paths, dict)
        or Path(str(paths.get("generated_dir", ""))).resolve()
        != generated_dir.resolve()
        or Path(str(paths.get("real_dir", ""))).resolve() != real_dir.resolve()
        or counts
        != {
            "generated_image_count": contract.generated_sample_count,
            "real_image_count": contract.class_count * contract.real_images_per_class,
        }
    ):
        raise ValueError(f"{method} frozen metrics paths or counts differ")
    if report.get("real_set") != real_set:
        raise ValueError(f"{method} frozen metrics real-set provenance differs")
    environment = report.get("runtime_environment")
    if (
        not isinstance(environment, dict)
        or runtime_environment_sha256(environment)
        != report.get("runtime_environment_sha256")
    ):
        raise ValueError(f"{method} frozen evaluator runtime environment differs")
    parameters = report.get("parameters")
    if (
        not isinstance(parameters, dict)
        or parameters.get("precision_recall_enabled") is not True
        or int(parameters.get("seed", -1)) != contract.metrics_seed
        or parameters.get("samples_find_deep") is not True
        or parameters.get("samples_shuffle") is not False
    ):
        raise ValueError(f"{method} frozen metrics parameters differ")
    source = report.get("sample_provenance")
    if not isinstance(source, dict):
        raise ValueError(f"{method} frozen sample provenance is missing")
    for name in (
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_integrity_manifest",
        "checkpoint_step",
        "weights",
        "git",
        "runtime_environment_sha256",
        "selected_prefix_budget",
        "image_shape",
        "sample_set_sha256",
        "sampling",
        "sampling_protocol_contract",
    ):
        if source.get(name) != sampling.get(name):
            raise ValueError(f"{method} frozen metrics sample provenance {name} differs")
    progress = source.get("sampling_progress")
    verified_progress = sampling.get("sampling_progress")
    if not isinstance(progress, dict) or not isinstance(verified_progress, dict):
        raise ValueError(f"{method} frozen metrics sampling progress is missing")
    for name in (
        "report",
        "status",
        "invocation",
        "completed_samples",
        "cumulative_elapsed_seconds",
    ):
        if progress.get(name) != verified_progress.get(name):
            raise ValueError(
                f"{method} frozen metrics sample provenance sampling_progress.{name} differs"
            )
    if Path(str(source.get("report", ""))).resolve() != Path(sampling["report"]).resolve():
        raise ValueError(f"{method} frozen metrics sampling report path differs")
    git = report.get("git")
    if git != sampling.get("git"):
        raise ValueError(f"{method} frozen evaluator Git differs from sampling Git")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{method} frozen metrics values are missing")
    values = {
        name: _finite_metric(metrics, name)
        for name in (
            "frechet_inception_distance",
            "inception_score_mean",
            "inception_score_std",
            "precision",
            "recall",
            "f_score",
        )
    }
    if (
        values["frechet_inception_distance"] < 0.0
        or values["inception_score_mean"] <= 0.0
        or values["inception_score_std"] < 0.0
        or not 0.0 <= values["precision"] <= 1.0
        or not 0.0 <= values["recall"] <= 1.0
        or not 0.0 <= values["f_score"] <= 1.0
    ):
        raise ValueError(f"{method} frozen metrics values are outside their domains")
    expected_f = (
        2.0
        * values["precision"]
        * values["recall"]
        / (values["precision"] + values["recall"])
    )
    if not math.isclose(values["f_score"], expected_f, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{method} frozen precision/recall F-score differs")
    return report, identity


def _validate_gate_bindings(
    gate: dict[str, Any],
    *,
    cofitok_metrics: dict[str, Any],
    dense_metrics: dict[str, Any],
    cofitok_sampling: dict[str, Any],
    dense_sampling: dict[str, Any],
    real_set: dict[str, Any],
) -> None:
    rows = {
        name: _gate_row(gate, name)
        for name in (
            "generation_metrics_complete",
            "matched_real_set_provenance",
            "matched_sampling_provenance",
        )
    }
    if any(row.get("passed") is not True for row in rows.values()):
        raise ValueError("frozen gate provenance rows are not passed")
    generation = rows["generation_metrics_complete"].get("evidence")
    metric_names = (
        "frechet_inception_distance",
        "inception_score_mean",
        "inception_score_std",
        "precision",
        "recall",
    )
    expected_generation = {
        "cofitok": {
            name: cofitok_metrics["metrics"][name] for name in metric_names
        },
        "dense": {name: dense_metrics["metrics"][name] for name in metric_names},
    }
    if generation != expected_generation:
        raise ValueError("frozen gate generation metrics evidence differs")
    real_evidence = rows["matched_real_set_provenance"].get("evidence")
    if not isinstance(real_evidence, dict):
        raise ValueError("frozen gate real-set evidence is missing")
    for method in ("cofitok", "dense_identity"):
        row = real_evidence.get(method)
        if (
            not isinstance(row, dict)
            or row.get("digest_schema") != real_set["digest_schema"]
            or int(row.get("image_count", -1)) != real_set["image_count"]
            or Path(str(row.get("root", ""))).resolve()
            != Path(real_set["root"]).resolve()
            or row.get("sha256") != real_set["sha256"]
            or row.get("valid") is not True
        ):
            raise ValueError(f"frozen gate {method} real-set evidence differs")
    sampling = rows["matched_sampling_provenance"].get("evidence")
    if not isinstance(sampling, dict) or sampling.get("protocols_match") is not True:
        raise ValueError("frozen gate matched sampling evidence is invalid")
    expected_fields = {
        "cofitok_checkpoint_sha256": cofitok_sampling["checkpoint_sha256"],
        "cofitok_checkpoint_step": cofitok_sampling["checkpoint_step"],
        "cofitok_prefix_budget": cofitok_sampling["selected_prefix_budget"],
        "cofitok_sample_set_sha256": cofitok_sampling["sample_set_sha256"],
        "dense_checkpoint_sha256": dense_sampling["checkpoint_sha256"],
        "dense_checkpoint_step": dense_sampling["checkpoint_step"],
        "dense_prefix_budget": dense_sampling["selected_prefix_budget"],
        "dense_sample_set_sha256": dense_sampling["sample_set_sha256"],
    }
    differing = [
        name for name, value in expected_fields.items() if sampling.get(name) != value
    ]
    if differing:
        raise ValueError(
            "frozen gate sampling provenance differs: " + ", ".join(differing)
        )
    summary = gate.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("frozen gate summary is missing")
    summary_fields = {
        "cofitok_fid": cofitok_metrics["metrics"]["frechet_inception_distance"],
        "cofitok_inception_score": cofitok_metrics["metrics"]["inception_score_mean"],
        "cofitok_precision": cofitok_metrics["metrics"]["precision"],
        "cofitok_recall": cofitok_metrics["metrics"]["recall"],
        "dense_fid": dense_metrics["metrics"]["frechet_inception_distance"],
        "dense_inception_score": dense_metrics["metrics"]["inception_score_mean"],
        "dense_precision": dense_metrics["metrics"]["precision"],
        "dense_recall": dense_metrics["metrics"]["recall"],
    }
    if any(summary.get(name) != value for name, value in summary_fields.items()):
        raise ValueError("frozen gate metric summary differs")


def _ordered_real_set(
    root: Path,
    *,
    contract: FrozenSupportContract,
    samples_per_class: int,
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    declared = reject_symlink_chain(root, name="frozen support real set")
    if not declared.is_dir():
        raise FileNotFoundError(f"frozen support real set is missing: {declared}")
    entries = sorted(declared.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() for path in entries):
        raise ValueError("frozen support real-set root contains a symlink")
    if any(not path.is_dir() for path in entries) or len(entries) != contract.class_count:
        raise ValueError("frozen support real-set class layout differs")
    class_files: list[list[Path]] = []
    all_images: list[Path] = []
    for directory in entries:
        children = sorted(directory.iterdir(), key=lambda path: path.name)
        if any(path.is_symlink() or not path.is_file() for path in children):
            raise ValueError(f"frozen support real class layout differs: {directory}")
        images = [
            path for path in children if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if len(images) != len(children) or len(images) != contract.real_images_per_class:
            raise ValueError(f"frozen support real class image count differs: {directory}")
        class_files.append(images)
        all_images.extend(images)
    positions = np.rint(
        np.linspace(0, contract.real_images_per_class - 1, samples_per_class)
    ).astype(np.int64)
    if len(set(int(value) for value in positions)) != samples_per_class:
        raise ValueError("frozen support real selection positions are not unique")
    selected = [
        class_files[class_index][int(positions[pass_index])]
        for pass_index in range(samples_per_class)
        for class_index in range(contract.class_count)
    ]
    relative = [path.relative_to(declared).as_posix() for path in selected]
    relative_digest = hashlib.sha256()
    for value in relative:
        relative_digest.update(value.encode("utf-8"))
        relative_digest.update(b"\0")
    class_digest = hashlib.sha256()
    for directory in entries:
        class_digest.update(directory.name.encode("utf-8"))
        class_digest.update(b"\0")
    return all_images, selected, {
        "schema": "sorted_class_evenly_spaced_real_support_subset_v1",
        "class_order": "lexicographic_directory_name",
        "within_class_order": "lexicographic_filename",
        "output_order": "selection_pass_major_then_class",
        "zero_based_positions": [int(value) for value in positions],
        "selected_image_count": len(selected),
        "selected_relative_paths_sha256": relative_digest.hexdigest(),
        "class_names_sha256": class_digest.hexdigest(),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("frozen support statistic is empty or non-finite")
    quantiles = np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(array.max()),
    }


def _effective_rank(descriptors: np.ndarray) -> dict[str, Any]:
    centered = descriptors - descriptors.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = np.square(singular)
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0.0:
        return {
            "entropy_effective_rank": 0.0,
            "participation_ratio": 0.0,
            "components_for_50_percent_variance": 0,
            "components_for_90_percent_variance": 0,
        }
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0.0]
    cumulative = np.cumsum(probabilities)
    return {
        "entropy_effective_rank": float(
            np.exp(-(nonzero * np.log(nonzero)).sum())
        ),
        "participation_ratio": float(1.0 / np.square(probabilities).sum()),
        "components_for_50_percent_variance": int(
            np.searchsorted(cumulative, 0.50) + 1
        ),
        "components_for_90_percent_variance": int(
            np.searchsorted(cumulative, 0.90) + 1
        ),
    }


def _dhash(image: Image.Image) -> np.ndarray:
    gray = np.asarray(
        image.convert("L").resize((9, 8), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    return (gray[:, 1:] > gray[:, :-1]).reshape(-1)


def _decoded_pixel_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"RGB_uint8_v1\0")
    digest.update(int(array.shape[0]).to_bytes(4, "big"))
    digest.update(int(array.shape[1]).to_bytes(4, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _analyze_image_paths(
    paths: list[Path],
    *,
    contract: FrozenSupportContract,
    samples_per_class: int,
) -> ImageSetAnalysis:
    expected_count = contract.class_count * samples_per_class
    if len(paths) != expected_count:
        raise ValueError("frozen support analysis image count differs")
    descriptors = np.empty((expected_count, 8 * 8 * 3), dtype=np.float64)
    dhashes = np.empty((expected_count, 64), dtype=np.bool_)
    decoded_hashes: list[str] = []
    dhash_bytes: list[bytes] = []
    pixel_mean = np.empty(expected_count, dtype=np.float64)
    pixel_std = np.empty(expected_count, dtype=np.float64)
    gradient = np.empty(expected_count, dtype=np.float64)
    exact_endpoint = np.empty(expected_count, dtype=np.float64)
    near_endpoint = np.empty(expected_count, dtype=np.float64)
    for index, path in enumerate(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"frozen support image is not a regular file: {path}")
        try:
            with Image.open(path) as source:
                if source.size != (contract.image_width, contract.image_height):
                    raise ValueError(f"frozen support image shape differs: {path}")
                image = source.convert("RGB")
                array_u8 = np.asarray(image, dtype=np.uint8)
                decoded_hashes.append(_decoded_pixel_digest(array_u8))
                array = array_u8.astype(np.float32) / 255.0
                pixel_mean[index] = float(array.mean())
                pixel_std[index] = float(array.std())
                horizontal = np.abs(array[:, 1:, :] - array[:, :-1, :]).mean()
                vertical = np.abs(array[1:, :, :] - array[:-1, :, :]).mean()
                gradient[index] = float(0.5 * (horizontal + vertical))
                exact_endpoint[index] = float(
                    np.logical_or(array_u8 == 0, array_u8 == 255).mean()
                )
                near_endpoint[index] = float(
                    np.logical_or(array_u8 <= 1, array_u8 >= 254).mean()
                )
                lowres = np.asarray(
                    image.resize((8, 8), Image.Resampling.BILINEAR),
                    dtype=np.float64,
                ) / 255.0
                descriptors[index] = lowres.reshape(-1)
                bits = _dhash(image)
                dhashes[index] = bits
                dhash_bytes.append(np.packbits(bits).tobytes())
        except (OSError, SyntaxError) as error:
            raise ValueError(f"frozen support image is unreadable: {path}") from error
    shaped = descriptors.reshape(samples_per_class, contract.class_count, -1)
    shaped_hash = dhashes.reshape(samples_per_class, contract.class_count, -1)
    same_rms = np.sqrt(np.square(shaped[1:] - shaped[:-1]).mean(axis=2)).reshape(-1)
    different_rms = np.sqrt(
        np.square(shaped[:, 1:] - shaped[:, :-1]).mean(axis=2)
    ).reshape(-1)
    same_hamming = np.not_equal(shaped_hash[1:], shaped_hash[:-1]).sum(axis=2).reshape(-1)
    different_hamming = (
        np.not_equal(shaped_hash[:, 1:], shaped_hash[:, :-1]).sum(axis=2).reshape(-1)
    )
    class_means = shaped.mean(axis=0)
    grand_mean = descriptors.mean(axis=0)
    within_variance = float(np.square(shaped - class_means[None, :, :]).mean())
    between_variance = float(np.square(class_means - grand_mean[None, :]).mean())
    total_variance = float(np.square(descriptors - grand_mean[None, :]).mean())
    if not math.isclose(
        within_variance + between_variance,
        total_variance,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise RuntimeError("frozen support class variance decomposition failed")
    report = {
        "image_count": expected_count,
        "decoded_pixel_duplicate_count": len(decoded_hashes)
        - len(set(decoded_hashes)),
        "dhash_duplicate_count": len(dhash_bytes) - len(set(dhash_bytes)),
        "per_image_pixel_mean": _distribution(pixel_mean),
        "per_image_pixel_std": _distribution(pixel_std),
        "per_image_mean_absolute_neighbor_gradient": _distribution(gradient),
        "per_image_exact_endpoint_fraction": _distribution(exact_endpoint),
        "per_image_near_endpoint_fraction": _distribution(near_endpoint),
        "lowres_support": _effective_rank(descriptors),
        "class_support": {
            "same_class_consecutive_pass_lowres_rms": _distribution(same_rms),
            "adjacent_class_same_pass_lowres_rms": _distribution(different_rms),
            "same_over_adjacent_class_rms_ratio": float(
                same_rms.mean() / different_rms.mean()
            ),
            "same_class_consecutive_pass_dhash_hamming": _distribution(
                same_hamming
            ),
            "adjacent_class_same_pass_dhash_hamming": _distribution(
                different_hamming
            ),
            "lowres_within_class_variance": within_variance,
            "lowres_between_class_variance": between_variance,
            "lowres_total_variance": total_variance,
            "lowres_between_class_variance_fraction": float(
                between_variance / total_variance
            ),
        },
    }
    return ImageSetAnalysis(report=report, descriptors=descriptors, dhashes=dhashes)


def _positive_ratio(numerator: float, denominator: float, *, name: str) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"frozen support ratio {name} is undefined")
    return float(numerator / denominator)


def _comparison(
    generated: ImageSetAnalysis,
    real: ImageSetAnalysis,
) -> dict[str, Any]:
    generated_class = generated.report["class_support"]
    real_class = real.report["class_support"]
    generated_rank = generated.report["lowres_support"]
    real_rank = real.report["lowres_support"]
    return {
        "mean_absolute_neighbor_gradient_ratio_to_real": _positive_ratio(
            generated.report["per_image_mean_absolute_neighbor_gradient"]["mean"],
            real.report["per_image_mean_absolute_neighbor_gradient"]["mean"],
            name="mean absolute neighbor gradient",
        ),
        "pixel_std_ratio_to_real": _positive_ratio(
            generated.report["per_image_pixel_std"]["mean"],
            real.report["per_image_pixel_std"]["mean"],
            name="pixel standard deviation",
        ),
        "lowres_entropy_effective_rank_ratio_to_real": _positive_ratio(
            generated_rank["entropy_effective_rank"],
            real_rank["entropy_effective_rank"],
            name="lowres entropy effective rank",
        ),
        "lowres_participation_ratio_to_real": _positive_ratio(
            generated_rank["participation_ratio"],
            real_rank["participation_ratio"],
            name="lowres participation ratio",
        ),
        "same_over_adjacent_class_rms_ratio_delta_from_real": float(
            generated_class["same_over_adjacent_class_rms_ratio"]
            - real_class["same_over_adjacent_class_rms_ratio"]
        ),
        "between_class_variance_fraction_ratio_to_real": _positive_ratio(
            generated_class["lowres_between_class_variance_fraction"],
            real_class["lowres_between_class_variance_fraction"],
            name="between-class variance fraction",
        ),
    }


def _paired_method_comparison(
    cofitok: ImageSetAnalysis,
    dense: ImageSetAnalysis,
) -> dict[str, Any]:
    if cofitok.descriptors.shape != dense.descriptors.shape:
        raise ValueError("frozen matched method descriptor shapes differ")
    rms = np.sqrt(
        np.square(cofitok.descriptors - dense.descriptors).mean(axis=1)
    )
    hamming = np.not_equal(cofitok.dhashes, dense.dhashes).sum(axis=1)
    cofitok_same = cofitok.report["class_support"][
        "same_class_consecutive_pass_lowres_rms"
    ]["mean"]
    dense_same = dense.report["class_support"][
        "same_class_consecutive_pass_lowres_rms"
    ]["mean"]
    return {
        "pairing": "same global index, class id, and initial-noise seed",
        "paired_lowres_rms": _distribution(rms),
        "paired_dhash_hamming": _distribution(hamming),
        "paired_lowres_rms_over_cofitok_same_class_rms": _positive_ratio(
            float(rms.mean()), cofitok_same, name="paired versus CoFiTok same class"
        ),
        "paired_lowres_rms_over_dense_same_class_rms": _positive_ratio(
            float(rms.mean()), dense_same, name="paired versus dense same class"
        ),
    }


def build_frozen_sample_support_audit(
    *,
    cofitok_sampling_report: Path,
    dense_sampling_report: Path,
    cofitok_metrics_report: Path,
    dense_metrics_report: Path,
    promotion_gate: Path,
    real_dir: Path,
    dataset_manifest: Path,
    audit_git: dict[str, Any],
    audit_runtime_environment: dict[str, Any],
    contract: FrozenSupportContract = FrozenSupportContract(),
) -> dict[str, Any]:
    samples_per_class = _require_contract(contract)
    if (
        len(str(audit_git.get("revision", ""))) != 40
        or not str(audit_git.get("branch", ""))
        or audit_git.get("tracked_dirty") is not False
        or audit_git.get("full_clean") is not True
    ):
        raise ValueError("frozen support audit Git identity must be exact and clean")
    if (
        audit_runtime_environment.get("device") != {"type": "cpu"}
        or audit_runtime_environment.get("environment_variables", {}).get(
            "CUDA_VISIBLE_DEVICES"
        )
        != ""
    ):
        raise ValueError("frozen support audit runtime must hide CUDA and use CPU")
    audit_runtime_sha = runtime_environment_sha256(audit_runtime_environment)

    dataset_manifest_path = reject_symlink_chain(
        dataset_manifest, name="frozen support dataset manifest"
    )
    dataset_identity = file_identity(dataset_manifest_path)
    if (
        dataset_identity["bytes"] != contract.dataset_manifest_bytes
        or dataset_identity["sha256"] != contract.dataset_manifest_sha256
    ):
        raise ValueError("frozen support dataset manifest identity differs")

    all_real_images, selected_real_images, real_selection = _ordered_real_set(
        real_dir,
        contract=contract,
        samples_per_class=samples_per_class,
    )
    real_root = reject_symlink_chain(real_dir, name="frozen support real set").resolve()
    actual_real_sha = image_tree_sha256(all_real_images, root=real_root)
    real_set = {
        "root": real_root.as_posix(),
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "image_count": len(all_real_images),
        "sha256": actual_real_sha,
    }
    if (
        len(all_real_images)
        != contract.class_count * contract.real_images_per_class
        or actual_real_sha != contract.real_set_sha256
    ):
        raise ValueError("frozen support physical real-set identity differs")

    cofitok_metrics_identity = file_identity(cofitok_metrics_report)
    dense_metrics_identity = file_identity(dense_metrics_report)
    gate, gate_identity = _validate_frozen_gate(
        promotion_gate,
        contract=contract,
        cofitok_metrics_identity=cofitok_metrics_identity,
        dense_metrics_identity=dense_metrics_identity,
    )

    raw_cofitok_metrics = read_json_object(
        cofitok_metrics_report, name="CoFiTok frozen metrics report"
    )
    raw_dense_metrics = read_json_object(
        dense_metrics_report, name="dense frozen metrics report"
    )
    cofitok_dir = Path(raw_cofitok_metrics.get("paths", {}).get("generated_dir", ""))
    dense_dir = Path(raw_dense_metrics.get("paths", {}).get("generated_dir", ""))
    cofitok_images = find_images(cofitok_dir)
    dense_images = find_images(dense_dir)
    cofitok_sampling = validate_sampling_provenance(
        Path(cofitok_sampling_report), cofitok_dir, cofitok_images
    )
    dense_sampling = validate_sampling_provenance(
        Path(dense_sampling_report), dense_dir, dense_images
    )
    _validate_matched_sampling(cofitok_sampling, dense_sampling, contract=contract)
    cofitok_metrics, _ = _validate_metrics_report(
        Path(cofitok_metrics_report),
        method="cofitok",
        generated_dir=cofitok_dir,
        real_dir=real_root,
        real_set=real_set,
        sampling=cofitok_sampling,
        contract=contract,
    )
    dense_metrics, _ = _validate_metrics_report(
        Path(dense_metrics_report),
        method="dense_identity",
        generated_dir=dense_dir,
        real_dir=real_root,
        real_set=real_set,
        sampling=dense_sampling,
        contract=contract,
    )
    if (
        cofitok_metrics.get("parameters") != dense_metrics.get("parameters")
        or cofitok_metrics.get("git") != dense_metrics.get("git")
        or cofitok_metrics.get("runtime_environment_sha256")
        != dense_metrics.get("runtime_environment_sha256")
    ):
        raise ValueError("CoFiTok and dense frozen evaluator contracts differ")
    if cofitok_metrics.get("git") != {
        "revision": gate["provenance_contract"]["evaluation_revision"],
        "branch": gate["provenance_contract"]["evaluation_branch"],
        "tracked_dirty": False,
    }:
        raise ValueError("frozen evaluator Git differs from promotion gate")
    _validate_gate_bindings(
        gate,
        cofitok_metrics=cofitok_metrics,
        dense_metrics=dense_metrics,
        cofitok_sampling=cofitok_sampling,
        dense_sampling=dense_sampling,
        real_set=real_set,
    )

    real_analysis = _analyze_image_paths(
        selected_real_images,
        contract=contract,
        samples_per_class=samples_per_class,
    )
    cofitok_analysis = _analyze_image_paths(
        cofitok_images,
        contract=contract,
        samples_per_class=samples_per_class,
    )
    dense_analysis = _analyze_image_paths(
        dense_images,
        contract=contract,
        samples_per_class=samples_per_class,
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "source_profile": "stability_scaling_frozen_50k_existing_samples",
        "audit_git": audit_git,
        "audit_runtime_environment": audit_runtime_environment,
        "audit_runtime_environment_sha256": audit_runtime_sha,
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "full_training_launch_allowed": False,
            "gpu_execution_authorized": False,
            "new_sampling_performed": False,
            "new_training_performed": False,
            "causal_attribution_allowed": False,
            "replaces_formal_fid_is_precision_recall": False,
            "purpose": (
                "Describe duplicate, low-resolution support, class-separation, "
                "and high-frequency signatures already present in the immutable "
                "frozen sample sets."
            ),
        },
        "contract": {
            **contract.__dict__,
            "generated_samples_per_class": samples_per_class,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "pairing_schema": PAIRING_SCHEMA,
            "real_selection_schema": real_selection["schema"],
        },
        "sources": {
            "promotion_gate": gate_identity,
            "dataset_manifest": dataset_identity,
            "real_set": real_set,
            "real_selection": real_selection,
            "cofitok": {
                "sampling": cofitok_sampling,
                "metrics_report": cofitok_metrics_identity,
                "formal_metrics": cofitok_metrics["metrics"],
            },
            "dense_identity": {
                "sampling": dense_sampling,
                "metrics_report": dense_metrics_identity,
                "formal_metrics": dense_metrics["metrics"],
            },
        },
        "support_statistics": {
            "real_matched_subset": real_analysis.report,
            "cofitok": cofitok_analysis.report,
            "dense_identity": dense_analysis.report,
        },
        "contrasts_to_real": {
            "cofitok": _comparison(cofitok_analysis, real_analysis),
            "dense_identity": _comparison(dense_analysis, real_analysis),
        },
        "matched_method_pair": _paired_method_comparison(
            cofitok_analysis, dense_analysis
        ),
        "interpretation_policy": {
            "thresholded_gate": False,
            "reason": (
                "Pixel-gradient, low-resolution rank, and class-separation proxies "
                "are descriptive diagnostics without pre-registered promotion "
                "thresholds. They cannot independently distinguish sampling-policy "
                "failure from checkpoint undertraining."
            ),
            "formal_distribution_support_remains_authoritative": {
                "cofitok_precision": cofitok_metrics["metrics"]["precision"],
                "cofitok_recall": cofitok_metrics["metrics"]["recall"],
                "dense_precision": dense_metrics["metrics"]["precision"],
                "dense_recall": dense_metrics["metrics"]["recall"],
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CPU-only, non-authorizing support audit from the immutable "
            "frozen stability 10K sample sets."
        )
    )
    parser.add_argument("--cofitok-sampling-report", required=True)
    parser.add_argument("--dense-sampling-report", required=True)
    parser.add_argument("--cofitok-metrics-report", required=True)
    parser.add_argument("--dense-metrics-report", required=True)
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-audit-revision", required=True)
    parser.add_argument("--expected-audit-branch", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit_git = git_provenance(PROJECT_ROOT)
    full_status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    audit_git["full_clean"] = not bool(full_status)
    if audit_git != {
        "revision": args.expected_audit_revision,
        "branch": args.expected_audit_branch,
        "tracked_dirty": False,
        "full_clean": True,
    }:
        raise ValueError("frozen support audit execution Git identity differs")
    output = reject_symlink_chain(args.output, name="frozen support audit output")
    if output.exists():
        raise FileExistsError(f"frozen support audit output already exists: {output}")
    runtime = capture_runtime_environment(torch.device("cpu"), project_root=PROJECT_ROOT)
    report = build_frozen_sample_support_audit(
        cofitok_sampling_report=Path(args.cofitok_sampling_report),
        dense_sampling_report=Path(args.dense_sampling_report),
        cofitok_metrics_report=Path(args.cofitok_metrics_report),
        dense_metrics_report=Path(args.dense_metrics_report),
        promotion_gate=Path(args.promotion_gate),
        real_dir=Path(args.real_dir),
        dataset_manifest=Path(args.dataset_manifest),
        audit_git=audit_git,
        audit_runtime_environment=runtime,
    )
    write_json_report(output, report)
    print(output.resolve())


if __name__ == "__main__":
    main()
