from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA, image_tree_sha256
from cofitok.inference_replay import reject_symlink_chain
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report

try:
    from evaluate_generation_metrics import (
        content_addressed_real_cache_name,
        find_images,
        validate_sampling_provenance,
    )
except (
    ModuleNotFoundError
):  # Imported as scripts.<module> by tests and library callers.
    from scripts.evaluate_generation_metrics import (
        content_addressed_real_cache_name,
        find_images,
        validate_sampling_provenance,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "non_authorizing_matched_generation_uncertainty_audit"
EXECUTION_MANIFEST_SCHEMA_VERSION = 1
EXECUTION_MANIFEST_ROLE = (
    "non_authorizing_matched_generation_uncertainty_execution_manifest"
)
FEATURE_EXTRACTOR = "inception-v3-compat"
FEATURE_LAYER = "2048"
KID_KERNEL = {
    "name": "polynomial",
    "degree": 3,
    "gamma": "1/feature_dimension",
    "coef0": 1.0,
    "estimator": "paired_unbiased_mmd2_difference_shared_reference",
}
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "replaces_fid_point_estimates": False,
    "replaces_promotion_gate": False,
    "broad_generation_superiority_claim_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
}
DEFAULT_EXECUTION_ARGUMENTS = {
    "real_cache_name": "imagenet256_val_50k_torch_fidelity_v04",
    "batch_size": 64,
    "min_samples": 10_000,
    "block_size": 500,
    "real_fold_count": 5,
    "bootstrap_repetitions": 10_000,
    "seed": 2027,
    "cpu": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paired block-KID uncertainty audit for aligned CoFiTok and "
            "dense sample indices. The audit is permanently non-authorizing."
        )
    )
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--cofitok-generated-dir", type=Path)
    parser.add_argument("--dense-generated-dir", type=Path)
    parser.add_argument("--cofitok-sampling-report", type=Path)
    parser.add_argument("--dense-sampling-report", type=Path)
    parser.add_argument("--cofitok-metrics-report", type=Path)
    parser.add_argument("--dense-metrics-report", type=Path)
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        help=(
            "Optional immutable source-bound execution manifest. When provided, "
            "all scientific inputs, outputs, parameters, report bytes/SHA256 "
            "values, sample identities, and the matched global-index window must "
            "match before feature extraction starts."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--real-cache-name",
        default=None,
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument(
        "--real-fold-count",
        type=int,
        default=None,
        help=(
            "Number of disjoint real blocks paired with each generated block; "
            "the ImageNet-256 10K audit uses all 50K validation images with 5."
        ),
    )
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-advantage", action="store_true")
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON report is not an object: {path}")
    return payload


def hydrate_execution_arguments(args: argparse.Namespace) -> argparse.Namespace:
    manifest_argument = getattr(args, "execution_manifest", None)
    if manifest_argument is not None:
        manifest_path = reject_symlink_chain(
            manifest_argument,
            name="matched uncertainty execution manifest",
        )
        manifest = _read_object(manifest_path)
        manifest_arguments = manifest.get("arguments")
        if not isinstance(manifest_arguments, dict):
            raise ValueError("Matched uncertainty execution manifest lacks arguments")
        for key, value in manifest_arguments.items():
            if not hasattr(args, key):
                continue
            if getattr(args, key) is None:
                setattr(
                    args,
                    key,
                    Path(value)
                    if key.endswith(("_dir", "_report"))
                    or key in {"output", "cache_root"}
                    else value,
                )
    for key, value in DEFAULT_EXECUTION_ARGUMENTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    required_paths = (
        "real_dir",
        "cofitok_generated_dir",
        "dense_generated_dir",
        "cofitok_sampling_report",
        "dense_sampling_report",
        "cofitok_metrics_report",
        "dense_metrics_report",
        "output",
    )
    missing = [key for key in required_paths if getattr(args, key, None) is None]
    if missing:
        raise ValueError(
            "Matched uncertainty inputs are missing: " + ", ".join(sorted(missing))
        )
    return args


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_path_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} is not absolute")
    return path.resolve().as_posix()


def _verify_bound_source(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"{label} source record is not an object")
    path_text = _resolved_path_text(record.get("path"), label=f"{label} path")
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not _is_sha256(expected_sha256)
    ):
        raise ValueError(f"{label} source identity is invalid")
    actual = _source(Path(path_text))
    if actual != {
        "path": path_text,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
    }:
        raise ValueError(f"{label} source identity mismatch")
    return actual


def validate_execution_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    manifest_argument = getattr(args, "execution_manifest", None)
    if manifest_argument is None:
        return None
    manifest_path = reject_symlink_chain(
        manifest_argument,
        name="matched uncertainty execution manifest",
    )
    manifest = _read_object(manifest_path)
    arguments = manifest.get("arguments")
    source_files = manifest.get("source_files")
    expected = manifest.get("expected")
    if (
        manifest.get("schema_version") != EXECUTION_MANIFEST_SCHEMA_VERSION
        or manifest.get("role") != EXECUTION_MANIFEST_ROLE
        or not isinstance(manifest.get("stream_id"), str)
        or not manifest["stream_id"]
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
        or not isinstance(arguments, dict)
        or not isinstance(source_files, dict)
        or not isinstance(expected, dict)
    ):
        raise ValueError("Matched uncertainty execution manifest contract mismatch")

    path_arguments = (
        "real_dir",
        "cofitok_generated_dir",
        "dense_generated_dir",
        "cofitok_sampling_report",
        "dense_sampling_report",
        "cofitok_metrics_report",
        "dense_metrics_report",
        "output",
        "cache_root",
    )
    scalar_arguments = (
        "real_cache_name",
        "batch_size",
        "min_samples",
        "block_size",
        "real_fold_count",
        "bootstrap_repetitions",
        "seed",
        "cpu",
    )
    expected_argument_keys = set(path_arguments + scalar_arguments)
    if set(arguments) != expected_argument_keys:
        raise ValueError("Matched uncertainty manifest argument set mismatch")
    for key in path_arguments:
        manifest_value = _resolved_path_text(
            arguments.get(key),
            label=f"manifest arguments.{key}",
        )
        argument_value = getattr(args, key, None)
        if (
            argument_value is None
            or Path(argument_value).resolve().as_posix() != manifest_value
        ):
            raise ValueError(f"Matched uncertainty manifest argument mismatch: {key}")
    for key in scalar_arguments:
        if getattr(args, key, None) != arguments.get(key):
            raise ValueError(f"Matched uncertainty manifest argument mismatch: {key}")

    source_argument_keys = (
        "cofitok_sampling_report",
        "dense_sampling_report",
        "cofitok_metrics_report",
        "dense_metrics_report",
    )
    if set(source_files) != set(source_argument_keys):
        raise ValueError("Matched uncertainty manifest source-file set mismatch")
    verified_sources = {
        key: _verify_bound_source(source_files[key], label=f"manifest {key}")
        for key in source_argument_keys
    }
    for key in source_argument_keys:
        if verified_sources[key]["path"] != _resolved_path_text(
            arguments[key],
            label=f"manifest arguments.{key}",
        ):
            raise ValueError(
                f"Matched uncertainty manifest source path mismatch: {key}"
            )

    matched = expected.get("matched_sampling")
    real_set = expected.get("real_set")
    sample_sets = expected.get("sample_sets")
    fid = expected.get("fid_point_estimates")
    if (
        not isinstance(matched, dict)
        or not isinstance(real_set, dict)
        or not isinstance(sample_sets, dict)
        or not isinstance(fid, dict)
        or not _is_sha256(matched.get("signature_sha256"))
        or int(matched.get("start_index", -1)) < 0
        or int(matched.get("end_index_exclusive", -1))
        <= int(matched.get("start_index", -1))
        or int(matched.get("sample_count", -1))
        != int(matched["end_index_exclusive"]) - int(matched["start_index"])
        or real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or not _is_sha256(real_set.get("sha256"))
        or int(real_set.get("image_count", -1)) < 1
        or set(sample_sets) != {"cofitok", "dense_identity"}
        or set(fid) != {"cofitok", "dense_identity"}
        or not _is_sha256(expected.get("metrics_runtime_environment_sha256"))
    ):
        raise ValueError("Matched uncertainty manifest expected identity is invalid")
    for method in ("cofitok", "dense_identity"):
        row = sample_sets[method]
        if (
            not isinstance(row, dict)
            or not _is_sha256(row.get("sha256"))
            or not _is_sha256(row.get("checkpoint_sha256"))
            or int(row.get("checkpoint_step", -1)) < 1
        ):
            raise ValueError(
                f"Matched uncertainty manifest {method} sample identity is invalid"
            )
        _finite_float(fid[method], label=f"manifest {method} FID")
    return {
        "source": _source(manifest_path),
        "source_files": source_files,
        "stream_id": manifest["stream_id"],
        "expected": expected,
    }


def validate_execution_manifest_observations(
    binding: dict[str, Any] | None,
    *,
    matched_sampling: dict[str, Any],
    real_set: dict[str, Any],
    sample_provenance: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if binding is None:
        return None
    _verify_bound_source(binding["source"], label="execution manifest replay")
    for key, record in binding["source_files"].items():
        _verify_bound_source(record, label=f"execution manifest replay {key}")
    expected = binding["expected"]
    observed = {
        "matched_sampling": {
            "start_index": int(matched_sampling["start_index"]),
            "end_index_exclusive": int(matched_sampling["end_index_exclusive"]),
            "sample_count": int(matched_sampling["sample_count"]),
            "cofitok_prefix_budgets": matched_sampling["cofitok_prefix_budgets"],
            "dense_prefix_budgets": matched_sampling["dense_prefix_budgets"],
            "signature_sha256": _canonical_sha256(matched_sampling["signature"]),
        },
        "real_set": real_set,
        "sample_sets": {
            method: {
                "sha256": sample_provenance[method]["sample_set_sha256"],
                "checkpoint_sha256": sample_provenance[method]["checkpoint_sha256"],
                "checkpoint_step": int(sample_provenance[method]["checkpoint_step"]),
            }
            for method in ("cofitok", "dense_identity")
        },
        "fid_point_estimates": {
            method: float(metrics[method]["fid"])
            for method in ("cofitok", "dense_identity")
        },
        "metrics_runtime_environment_sha256": metrics["cofitok"][
            "runtime_environment_sha256"
        ],
    }
    if observed != expected:
        raise ValueError("Matched uncertainty execution manifest observation mismatch")
    return {
        "source": binding["source"],
        "stream_id": binding["stream_id"],
        "status": "verified",
    }


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def matched_sampling_signature(sampling: dict[str, Any]) -> dict[str, Any]:
    signature = dict(sampling)
    signature.pop("prefix_budgets", None)
    return signature


def validate_matched_sampling(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
) -> dict[str, Any]:
    cofitok_sampling = cofitok.get("sampling")
    dense_sampling = dense.get("sampling")
    if not isinstance(cofitok_sampling, dict) or not isinstance(dense_sampling, dict):
        raise TypeError("Matched sampling provenance is missing sampling objects")
    cofitok_signature = matched_sampling_signature(cofitok_sampling)
    dense_signature = matched_sampling_signature(dense_sampling)
    if cofitok_signature != dense_signature:
        raise ValueError("CoFiTok and dense sampling protocols are not matched")
    random_stream = cofitok_sampling.get("random_stream")
    expected_random_stream = {
        "batch_size_invariant": True,
        "prefix_budgets_share_stream": True,
        "resume_index_invariant": True,
        "scope": "per_global_sample_index",
        "seed_formula": "(seed + global_index) mod 2^63",
    }
    if random_stream != expected_random_stream:
        raise ValueError("Matched uncertainty requires per-global-index random streams")
    count = int(cofitok_sampling.get("num_samples", -1))
    start_index = int(cofitok_sampling.get("start_index", -1))
    if count < 1 or start_index < 0:
        raise ValueError("Matched sampling window is invalid")
    return {
        "signature": cofitok_signature,
        "start_index": start_index,
        "end_index_exclusive": start_index + count,
        "sample_count": count,
        "cofitok_prefix_budgets": cofitok_sampling.get("prefix_budgets"),
        "dense_prefix_budgets": dense_sampling.get("prefix_budgets"),
    }


def validate_metrics_report(
    path: Path,
    *,
    generated_dir: Path,
    real_dir: Path,
    sample_provenance: dict[str, Any],
    generated_count: int,
    real_count: int,
    real_set_sha256: str,
) -> dict[str, Any]:
    report = _read_object(path)
    paths = report.get("paths")
    counts = report.get("counts")
    real_set = report.get("real_set")
    parameters = report.get("parameters")
    provenance = report.get("sample_provenance")
    metrics = report.get("metrics")
    if (
        report.get("schema_version") != 3
        or report.get("role") != "generation_directory_metrics_report"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("status") != "completed"
        or not isinstance(paths, dict)
        or Path(str(paths.get("generated_dir", ""))).resolve()
        != generated_dir.resolve()
        or Path(str(paths.get("real_dir", ""))).resolve() != real_dir.resolve()
        or not isinstance(counts, dict)
        or int(counts.get("generated_image_count", -1)) != generated_count
        or int(counts.get("real_image_count", -1)) != real_count
        or not isinstance(real_set, dict)
        or real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or int(real_set.get("image_count", -1)) != real_count
        or real_set.get("sha256") != real_set_sha256
        or not isinstance(parameters, dict)
        or parameters.get("samples_find_deep") is not True
        or parameters.get("samples_shuffle") is not False
        or not isinstance(provenance, dict)
        or provenance.get("sample_set_sha256")
        != sample_provenance.get("sample_set_sha256")
        or provenance.get("checkpoint_sha256")
        != sample_provenance.get("checkpoint_sha256")
        or provenance.get("checkpoint_step") != sample_provenance.get("checkpoint_step")
        or provenance.get("sampling") != sample_provenance.get("sampling")
        or not isinstance(metrics, dict)
    ):
        raise ValueError(f"Generation metrics report contract mismatch: {path}")
    fid = _finite_float(
        metrics.get("frechet_inception_distance"),
        label=f"{path} FID",
    )
    if fid < 0.0:
        raise ValueError(f"Generation metrics report FID is negative: {path}")
    return {
        "source": _source(path),
        "fid": fid,
        "implementation": report.get("implementation"),
        "runtime_environment_sha256": report.get("runtime_environment_sha256"),
        "real_set": real_set,
        "sample_set_sha256": provenance["sample_set_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_step": int(provenance["checkpoint_step"]),
    }


def polynomial_mmd2_unbiased(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> float:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.ndim != 2 or reference.ndim != 2:
        raise ValueError("KID features must be two-dimensional")
    if candidate.shape != reference.shape:
        raise ValueError("Paired block KID requires equal candidate/reference shapes")
    sample_count, feature_dimension = candidate.shape
    if sample_count < 2 or feature_dimension < 1:
        raise ValueError("KID blocks require at least two samples and one feature")
    gamma = 1.0 / feature_dimension
    candidate_kernel = (gamma * candidate.dot(candidate.T) + 1.0) ** 3
    reference_kernel = (gamma * reference.dot(reference.T) + 1.0) ** 3
    cross_kernel = (gamma * candidate.dot(reference.T) + 1.0) ** 3
    denominator = sample_count * (sample_count - 1)
    candidate_off_diagonal = (
        candidate_kernel.sum(dtype=np.float64)
        - np.trace(candidate_kernel, dtype=np.float64)
    ) / denominator
    reference_off_diagonal = (
        reference_kernel.sum(dtype=np.float64)
        - np.trace(reference_kernel, dtype=np.float64)
    ) / denominator
    cross_mean = cross_kernel.mean(dtype=np.float64)
    return float(candidate_off_diagonal + reference_off_diagonal - 2.0 * cross_mean)


def polynomial_mmd2_difference_shared_reference(
    cofitok: np.ndarray,
    dense: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    cofitok = np.asarray(cofitok)
    dense = np.asarray(dense)
    reference = np.asarray(reference)
    if cofitok.ndim != 2 or dense.ndim != 2 or reference.ndim != 2:
        raise ValueError("Paired KID contrast features must be two-dimensional")
    if cofitok.shape != dense.shape or cofitok.shape[1] != reference.shape[1]:
        raise ValueError("Paired KID contrast feature matrices are incompatible")
    sample_count, feature_dimension = cofitok.shape
    if sample_count < 2 or reference.shape[0] < 2 or feature_dimension < 1:
        raise ValueError("Paired KID contrast requires at least two samples per side")
    gamma = 1.0 / feature_dimension
    cofitok_self_kernel = (gamma * cofitok.dot(cofitok.T) + 1.0) ** 3
    dense_self_kernel = (gamma * dense.dot(dense.T) + 1.0) ** 3
    cofitok_cross_kernel = (gamma * cofitok.dot(reference.T) + 1.0) ** 3
    dense_cross_kernel = (gamma * dense.dot(reference.T) + 1.0) ** 3
    denominator = sample_count * (sample_count - 1)
    cofitok_self = (
        cofitok_self_kernel.sum(dtype=np.float64)
        - np.trace(cofitok_self_kernel, dtype=np.float64)
    ) / denominator
    dense_self = (
        dense_self_kernel.sum(dtype=np.float64)
        - np.trace(dense_self_kernel, dtype=np.float64)
    ) / denominator
    cofitok_cross = cofitok_cross_kernel.mean(dtype=np.float64)
    dense_cross = dense_cross_kernel.mean(dtype=np.float64)
    difference = cofitok_self - dense_self - 2.0 * (cofitok_cross - dense_cross)
    return {
        "cofitok_candidate_self_term": float(cofitok_self),
        "dense_candidate_self_term": float(dense_self),
        "cofitok_cross_term": float(cofitok_cross),
        "dense_cross_term": float(dense_cross),
        "cofitok_minus_dense": float(difference),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "Bootstrap values must be one-dimensional with at least two rows"
        )
    if repetitions < 100:
        raise ValueError("Bootstrap repetitions must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    return {
        "repetitions": repetitions,
        "seed": seed,
        "mean": float(values.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_probability_cofitok_better": float(np.mean(means < 0.0)),
    }


def exact_one_sided_sign_test(*, wins: int, losses: int) -> float:
    if wins < 0 or losses < 0:
        raise ValueError("Sign test counts must be nonnegative")
    if wins + losses == 0:
        return 1.0
    non_ties = wins + losses
    numerator = sum(math.comb(non_ties, value) for value in range(wins, non_ties + 1))
    return float(numerator / (2**non_ties))


def _index_sha256(indices: np.ndarray) -> str:
    array = np.ascontiguousarray(indices, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    view = memoryview(contiguous).cast("B")
    chunk_size = 8 * 1024 * 1024
    for offset in range(0, len(view), chunk_size):
        digest.update(view[offset : offset + chunk_size])
    return digest.hexdigest()


def paired_block_kid(
    *,
    cofitok_features: np.ndarray,
    dense_features: np.ndarray,
    real_features: np.ndarray,
    block_size: int,
    real_fold_count: int,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    cofitok_features = np.asarray(cofitok_features)
    dense_features = np.asarray(dense_features)
    real_features = np.asarray(real_features)
    if (
        cofitok_features.ndim != 2
        or dense_features.ndim != 2
        or real_features.ndim != 2
        or cofitok_features.shape != dense_features.shape
        or cofitok_features.shape[1] != real_features.shape[1]
    ):
        raise ValueError("Matched uncertainty feature matrices are incompatible")
    sample_count = cofitok_features.shape[0]
    if block_size < 2 or sample_count % block_size != 0:
        raise ValueError(
            "Generated sample count must be exactly divisible by block size"
        )
    block_count = sample_count // block_size
    if block_count < 8:
        raise ValueError("Matched uncertainty requires at least eight disjoint blocks")
    if real_fold_count < 1:
        raise ValueError("Matched uncertainty requires at least one real fold")
    required_real_count = sample_count * real_fold_count
    if real_features.shape[0] < required_real_count:
        raise ValueError(
            "Real feature count is below the requested disjoint real folds"
        )

    rng = np.random.default_rng(seed)
    generated_permutation = rng.permutation(sample_count)
    real_selection = rng.permutation(real_features.shape[0])[:required_real_count]
    rows: list[dict[str, Any]] = []
    differences: list[float] = []
    for block_index in range(block_count):
        start = block_index * block_size
        end = start + block_size
        generated_indices = generated_permutation[start:end]
        real_indices = np.concatenate(
            [
                real_selection[
                    fold_index * sample_count + start : fold_index * sample_count + end
                ]
                for fold_index in range(real_fold_count)
            ]
        )
        contrast = polynomial_mmd2_difference_shared_reference(
            cofitok_features[generated_indices],
            dense_features[generated_indices],
            real_features[real_indices],
        )
        difference = contrast["cofitok_minus_dense"]
        differences.append(difference)
        rows.append(
            {
                "block_index": block_index,
                "generated_index_sha256": _index_sha256(generated_indices),
                "real_index_sha256": _index_sha256(real_indices),
                "real_sample_count": int(real_indices.size),
                **contrast,
            }
        )
    difference_array = np.asarray(differences, dtype=np.float64)
    wins = int(np.sum(difference_array < 0.0))
    losses = int(np.sum(difference_array > 0.0))
    ties = int(np.sum(difference_array == 0.0))
    bootstrap = bootstrap_mean_ci(
        difference_array,
        repetitions=bootstrap_repetitions,
        seed=seed + 1,
    )
    sign_test_p = exact_one_sided_sign_test(wins=wins, losses=losses)
    uncertainty_supports_advantage = (
        float(bootstrap["ci_high"]) < 0.0 and sign_test_p <= 0.05
    )
    return {
        "block_size": block_size,
        "block_count": block_count,
        "real_fold_count": real_fold_count,
        "real_samples_per_block": block_size * real_fold_count,
        "real_samples_used": required_real_count,
        "generated_permutation_sha256": _index_sha256(generated_permutation),
        "real_selection_sha256": _index_sha256(real_selection),
        "cofitok_minus_dense_block_mean": float(difference_array.mean()),
        "cofitok_better_blocks": wins,
        "dense_better_blocks": losses,
        "tied_blocks": ties,
        "cofitok_better_block_fraction": float(wins / block_count),
        "one_sided_exact_sign_test_p": sign_test_p,
        "paired_block_bootstrap": bootstrap,
        "uncertainty_supports_cofitok_advantage": uncertainty_supports_advantage,
        "rows": rows,
    }


def extract_feature_triplet(
    *,
    real_dir: Path,
    cofitok_dir: Path,
    dense_dir: Path,
    real_cache_name: str,
    cofitok_cache_name: str,
    dense_cache_name: str,
    cache_root: Path | None,
    batch_size: int,
    cuda: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    torch_fidelity = importlib.import_module("torch_fidelity")
    utils = importlib.import_module("torch_fidelity.utils")
    feature_extractor = utils.create_feature_extractor(
        FEATURE_EXTRACTOR,
        [FEATURE_LAYER],
        cuda=cuda,
        verbose=True,
    )

    def extract(path: Path, cache_name: str) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "input1": path.resolve().as_posix(),
            "input1_cache_name": cache_name,
            "cuda": cuda,
            "batch_size": batch_size,
            "samples_find_deep": True,
            "samples_shuffle": False,
            "rng_seed": 0,
            "save_cpu_ram": True,
            "cache": True,
            "verbose": True,
        }
        if cache_root is not None:
            kwargs["cache_root"] = cache_root.resolve().as_posix()
        features = utils.extract_featuresdict_from_input_id_cached(
            1,
            feature_extractor,
            **kwargs,
        )[FEATURE_LAYER]
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError(f"torch-fidelity returned malformed features for {path}")
        return features.detach().cpu().numpy()

    real = extract(real_dir, real_cache_name)
    cofitok = extract(cofitok_dir, cofitok_cache_name)
    dense = extract(dense_dir, dense_cache_name)
    return cofitok, dense, real, str(torch_fidelity.__version__)


def build_report(
    *,
    cofitok_features: np.ndarray,
    dense_features: np.ndarray,
    real_features: np.ndarray,
    cofitok_metrics: dict[str, Any],
    dense_metrics: dict[str, Any],
    matched_sampling: dict[str, Any],
    sources: dict[str, Any],
    parameters: dict[str, Any],
    implementation: dict[str, Any],
    evaluator_git: dict[str, Any],
    runtime_environment: dict[str, Any],
) -> dict[str, Any]:
    expected_generated = int(matched_sampling["sample_count"])
    expected_real = int(sources["real_set"]["image_count"])
    if (
        cofitok_features.shape[0] != expected_generated
        or dense_features.shape[0] != expected_generated
        or real_features.shape[0] != expected_real
    ):
        raise ValueError("Feature counts do not match bound sample counts")
    block_audit = paired_block_kid(
        cofitok_features=cofitok_features,
        dense_features=dense_features,
        real_features=real_features,
        block_size=int(parameters["block_size"]),
        real_fold_count=int(parameters["real_fold_count"]),
        bootstrap_repetitions=int(parameters["bootstrap_repetitions"]),
        seed=int(parameters["seed"]),
    )
    cofitok_fid = float(cofitok_metrics["fid"])
    dense_fid = float(dense_metrics["fid"])
    fid_direction_supports_advantage = cofitok_fid < dense_fid
    advantage_supported = (
        fid_direction_supports_advantage
        and block_audit["uncertainty_supports_cofitok_advantage"] is True
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass" if advantage_supported else "hold",
        "role": REPORT_ROLE,
        "decision": (
            "matched_relative_generation_advantage_supported"
            if advantage_supported
            else "matched_relative_generation_advantage_not_confirmed"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "git": evaluator_git,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256(runtime_environment),
        "implementation": implementation,
        "parameters": parameters,
        "sources": sources,
        "matched_sampling": matched_sampling,
        "feature_matrices": {
            "cofitok": {
                "shape": list(cofitok_features.shape),
                "dtype": str(cofitok_features.dtype),
                "sha256": _array_sha256(cofitok_features),
            },
            "dense_identity": {
                "shape": list(dense_features.shape),
                "dtype": str(dense_features.dtype),
                "sha256": _array_sha256(dense_features),
            },
            "real": {
                "shape": list(real_features.shape),
                "dtype": str(real_features.dtype),
                "sha256": _array_sha256(real_features),
            },
        },
        "fid_point_estimates": {
            "cofitok": cofitok_fid,
            "dense_identity": dense_fid,
            "cofitok_relative_to_dense": cofitok_fid / dense_fid - 1.0,
            "direction_supports_cofitok_advantage": fid_direction_supports_advantage,
        },
        "paired_block_kid": block_audit,
        "advantage_supported": advantage_supported,
        "limitations": [
            "This report supports only a relative matched-sample claim for the bound checkpoint, protocol, and global-index window.",
            "The block bootstrap treats disjoint randomized sample blocks as the uncertainty units.",
            "KID uncertainty does not repair poor absolute FID, recall, class fidelity, or visual quality.",
            "This report does not replace the frozen promotion gate or authorize further training.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.batch_size < 1
        or args.min_samples < 1
        or args.block_size < 2
        or args.real_fold_count < 1
        or args.bootstrap_repetitions < 100
    ):
        raise ValueError("Invalid matched uncertainty audit numeric parameters")
    execution_binding = validate_execution_manifest(args)
    real_dir = reject_symlink_chain(args.real_dir, name="uncertainty real directory")
    cofitok_dir = reject_symlink_chain(
        args.cofitok_generated_dir,
        name="uncertainty CoFiTok generated directory",
    )
    dense_dir = reject_symlink_chain(
        args.dense_generated_dir,
        name="uncertainty dense generated directory",
    )
    sampling_paths = {
        "cofitok": reject_symlink_chain(
            args.cofitok_sampling_report,
            name="uncertainty CoFiTok sampling report",
        ),
        "dense_identity": reject_symlink_chain(
            args.dense_sampling_report,
            name="uncertainty dense sampling report",
        ),
    }
    metrics_paths = {
        "cofitok": reject_symlink_chain(
            args.cofitok_metrics_report,
            name="uncertainty CoFiTok metrics report",
        ),
        "dense_identity": reject_symlink_chain(
            args.dense_metrics_report,
            name="uncertainty dense metrics report",
        ),
    }
    real_images = find_images(real_dir)
    cofitok_images = find_images(cofitok_dir)
    dense_images = find_images(dense_dir)
    if (
        len(cofitok_images) != len(dense_images)
        or len(cofitok_images) < args.min_samples
        or [path.name for path in cofitok_images]
        != [path.name for path in dense_images]
    ):
        raise ValueError("Generated sample sets are not aligned matched windows")
    if len(real_images) < len(cofitok_images):
        raise ValueError("Real set is smaller than the matched generated window")
    sample_provenance = {
        "cofitok": validate_sampling_provenance(
            sampling_paths["cofitok"],
            cofitok_dir,
            cofitok_images,
        ),
        "dense_identity": validate_sampling_provenance(
            sampling_paths["dense_identity"],
            dense_dir,
            dense_images,
        ),
    }
    matched_sampling = validate_matched_sampling(
        sample_provenance["cofitok"],
        sample_provenance["dense_identity"],
    )
    if matched_sampling["sample_count"] != len(cofitok_images):
        raise ValueError("Matched sampling count differs from physical generated sets")
    real_set_sha = image_tree_sha256(real_images, root=real_dir)
    real_set = {
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "root": real_dir.resolve().as_posix(),
        "image_count": len(real_images),
        "sha256": real_set_sha,
    }
    metrics = {
        method: validate_metrics_report(
            metrics_paths[method],
            generated_dir=cofitok_dir if method == "cofitok" else dense_dir,
            real_dir=real_dir,
            sample_provenance=sample_provenance[method],
            generated_count=len(cofitok_images),
            real_count=len(real_images),
            real_set_sha256=real_set_sha,
        )
        for method in ("cofitok", "dense_identity")
    }
    if (
        metrics["cofitok"]["implementation"]
        != metrics["dense_identity"]["implementation"]
        or metrics["cofitok"]["runtime_environment_sha256"]
        != metrics["dense_identity"]["runtime_environment_sha256"]
        or metrics["cofitok"]["real_set"] != metrics["dense_identity"]["real_set"]
    ):
        raise ValueError(
            "Matched FID source reports do not share one evaluator and real set"
        )
    execution_verification = validate_execution_manifest_observations(
        execution_binding,
        matched_sampling=matched_sampling,
        real_set=real_set,
        sample_provenance=sample_provenance,
        metrics=metrics,
    )

    cache_root = (
        reject_symlink_chain(args.cache_root, name="uncertainty feature cache root")
        if args.cache_root is not None
        else None
    )
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if cuda else "cpu")
    runtime_environment = capture_runtime_environment(device, project_root=PROJECT_ROOT)
    effective_real_cache_name = content_addressed_real_cache_name(
        args.real_cache_name,
        real_set_sha,
    )
    cache_names = {
        "real": effective_real_cache_name,
        "cofitok": "matched_uncertainty_cofitok_"
        + sample_provenance["cofitok"]["sample_set_sha256"][:16],
        "dense_identity": "matched_uncertainty_dense_"
        + sample_provenance["dense_identity"]["sample_set_sha256"][:16],
    }
    cofitok_features, dense_features, real_features, torch_fidelity_version = (
        extract_feature_triplet(
            real_dir=real_dir,
            cofitok_dir=cofitok_dir,
            dense_dir=dense_dir,
            real_cache_name=cache_names["real"],
            cofitok_cache_name=cache_names["cofitok"],
            dense_cache_name=cache_names["dense_identity"],
            cache_root=cache_root,
            batch_size=args.batch_size,
            cuda=cuda,
        )
    )
    sources = {
        "real_set": real_set,
        "sampling_reports": {
            method: _source(path) for method, path in sampling_paths.items()
        },
        "metrics_reports": {
            method: metrics[method]["source"]
            for method in ("cofitok", "dense_identity")
        },
        "sample_sets": {
            method: {
                "sha256": sample_provenance[method]["sample_set_sha256"],
                "checkpoint_sha256": sample_provenance[method]["checkpoint_sha256"],
                "checkpoint_step": sample_provenance[method]["checkpoint_step"],
            }
            for method in ("cofitok", "dense_identity")
        },
    }
    if execution_verification is not None:
        sources["execution_manifest"] = execution_verification
    return build_report(
        cofitok_features=cofitok_features,
        dense_features=dense_features,
        real_features=real_features,
        cofitok_metrics=metrics["cofitok"],
        dense_metrics=metrics["dense_identity"],
        matched_sampling=matched_sampling,
        sources=sources,
        parameters={
            "batch_size": args.batch_size,
            "min_samples": args.min_samples,
            "block_size": args.block_size,
            "real_fold_count": args.real_fold_count,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "cuda": cuda,
            "cpu_requested": bool(args.cpu),
            "cache_root": cache_root.resolve().as_posix() if cache_root else "",
            "cache_names": cache_names,
        },
        implementation={
            "torch_fidelity_version": torch_fidelity_version,
            "feature_extractor": FEATURE_EXTRACTOR,
            "feature_layer": FEATURE_LAYER,
            "kid_kernel": KID_KERNEL,
            "image_order": "torch_fidelity_recursive_lexicographic_path_order",
            "pairing": "same_zero_padded_global_index_filename",
        },
        evaluator_git=git_provenance(PROJECT_ROOT),
        runtime_environment=runtime_environment,
    )


def main() -> int:
    args = hydrate_execution_arguments(parse_args())
    with exclusive_output_lock(args.output, role=REPORT_ROLE):
        report = run(args)
        if args.output.exists():
            if not args.resume:
                raise FileExistsError(
                    "matched uncertainty report already exists; pass --resume to verify it"
                )
            existing = _read_object(args.output)
            if existing != report:
                raise ValueError(
                    "existing matched uncertainty report differs from exact replay"
                )
            print(f"reused {args.output}")
        else:
            write_json_report(args.output, report)
            print(f"wrote {args.output}")
    return (
        2 if args.require_advantage and report["advantage_supported"] is not True else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
