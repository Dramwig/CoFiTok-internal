from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .image_integrity import IMAGE_TREE_DIGEST_SCHEMA, image_tree_sha256
from .inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)

METRICS_TRUST_RECEIPT_SCHEMA_VERSION = 1
METRICS_TRUST_RECEIPT_ROLE = "generation_metrics_trust_boundary_receipt"
FILE_TREE_DIGEST_SCHEMA = "cofitok_file_tree_sha256_v1"
EXPECTED_TORCH_FIDELITY_VERSION = "0.4.0"

AUTHORIZATION_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "cpu_only_physical_integrity": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
}

CACHE_SPECS = {
    "inception_features_2048": {
        "suffix": "-inception-v3-compat-features-2048.pt",
        "kind": "tensor",
        "shape_tail": [2048],
        "dtype": "torch.float32",
    },
    "inception_logits_unbiased": {
        "suffix": "-inception-v3-compat-features-logits_unbiased.pt",
        "kind": "tensor",
        "shape_tail": [1008],
        "dtype": "torch.float32",
    },
    "inception_fid_statistics": {
        "suffix": "-inception-v3-compat-stat-fid-2048.pt",
        "kind": "fid_statistics",
        "mu_shape": [2048],
        "mu_dtype": "float32",
        "sigma_shape": [2048, 2048],
        "sigma_dtype": "float64",
    },
    "vgg_features_fc2_relu": {
        "suffix": "-vgg16-features-fc2_relu.pt",
        "kind": "tensor",
        "shape_tail": [4096],
        "dtype": "torch.float32",
    },
}

COVARIANCE_PROBE_INDICES = (0, 1, 7, 31, 127, 511, 1023, 1535, 2047)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _regular_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"package tree must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def file_tree_identity(root: str | Path) -> dict[str, Any]:
    tree_root = reject_symlink_chain(root, name="metrics dependency tree").resolve()
    if not tree_root.is_dir():
        raise FileNotFoundError(f"metrics dependency tree is missing: {tree_root}")
    entries = []
    total_bytes = 0
    for path in _regular_files(tree_root):
        identity = file_identity(path)
        relative = path.relative_to(tree_root).as_posix()
        total_bytes += int(identity["bytes"])
        entries.append(
            {
                "path": relative,
                "bytes": int(identity["bytes"]),
                "sha256": identity["sha256"],
            }
        )
    if not entries:
        raise ValueError(f"metrics dependency tree is empty: {tree_root}")
    digest_rows = [
        [entry["path"], entry["bytes"], entry["sha256"]] for entry in entries
    ]
    return {
        "digest_schema": FILE_TREE_DIGEST_SCHEMA,
        "root": tree_root.as_posix(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "sha256": _stable_sha256(digest_rows),
        "files": entries,
    }


def torch_fidelity_distribution_identity(
    package_root: str | Path,
    dist_info_root: str | Path,
) -> dict[str, Any]:
    try:
        version = importlib.metadata.version("torch-fidelity")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("torch-fidelity distribution is unavailable") from error
    if version != EXPECTED_TORCH_FIDELITY_VERSION:
        raise ValueError(f"torch-fidelity version differs: {version}")
    return {
        "package": "torch_fidelity",
        "version": version,
        "package_tree": file_tree_identity(package_root),
        "dist_info_tree": file_tree_identity(dist_info_root),
    }


def expected_cache_paths(
    cache_root: str | Path,
    *,
    real_cache_name: str,
) -> dict[str, Path]:
    root = reject_symlink_chain(cache_root, name="torch-fidelity cache root").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"torch-fidelity cache root is missing: {root}")
    if not real_cache_name or Path(real_cache_name).name != real_cache_name:
        raise ValueError("torch-fidelity real cache name is malformed")
    paths = {
        role: reject_symlink_chain(
            root / f"{real_cache_name}{spec['suffix']}",
            name=f"torch-fidelity cache {role}",
        ).resolve()
        for role, spec in CACHE_SPECS.items()
    }
    expected_names = {path.name for path in paths.values()}
    actual_names = {
        path.name for path in root.glob(f"{real_cache_name}*.pt") if path.is_file()
    }
    if actual_names != expected_names:
        raise ValueError(
            "torch-fidelity real cache file set differs: "
            f"expected={sorted(expected_names)} actual={sorted(actual_names)}"
        )
    return paths


def cache_file_identities(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    if set(paths) != set(CACHE_SPECS):
        raise ValueError("torch-fidelity cache role set differs")
    return {role: file_identity(paths[role]) for role in sorted(paths)}


def _torch_load(path: Path) -> Any:
    import torch

    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _tensor_finite(tensor: Any, *, chunk_rows: int = 1024) -> bool:
    import torch

    if not torch.is_tensor(tensor):
        return False
    if tensor.ndim == 0:
        return bool(torch.isfinite(tensor))
    return all(bool(torch.isfinite(chunk).all()) for chunk in tensor.split(chunk_rows))


def validate_cache_payloads(
    paths: Mapping[str, Path],
    *,
    image_count: int,
    specs: Mapping[str, Mapping[str, Any]] = CACHE_SPECS,
    covariance_probe_indices: Sequence[int] = COVARIANCE_PROBE_INDICES,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if image_count < 1:
        raise ValueError("metrics cache image count must be positive")
    if set(paths) != set(specs):
        raise ValueError("metrics cache payload role set differs")
    loaded = {role: _torch_load(paths[role]) for role in sorted(paths)}
    metadata: dict[str, Any] = {}
    for role, spec in specs.items():
        payload = loaded[role]
        if spec["kind"] == "tensor":
            expected_shape = [image_count, *spec["shape_tail"]]
            if (
                not torch.is_tensor(payload)
                or list(payload.shape) != expected_shape
                or str(payload.dtype) != spec["dtype"]
                or str(payload.device) != "cpu"
                or not _tensor_finite(payload)
            ):
                raise ValueError(f"metrics cache tensor payload differs: {role}")
            metadata[role] = {
                "kind": "tensor",
                "shape": expected_shape,
                "dtype": str(payload.dtype),
                "finite": True,
            }
            continue
        if spec["kind"] != "fid_statistics" or not isinstance(payload, dict):
            raise ValueError(f"metrics cache statistics payload differs: {role}")
        if set(payload) != {"mu", "sigma"}:
            raise ValueError("metrics cache FID statistic keys differ")
        mu = payload["mu"]
        sigma = payload["sigma"]
        if (
            not isinstance(mu, np.ndarray)
            or not isinstance(sigma, np.ndarray)
            or list(mu.shape) != spec["mu_shape"]
            or str(mu.dtype) != spec["mu_dtype"]
            or list(sigma.shape) != spec["sigma_shape"]
            or str(sigma.dtype) != spec["sigma_dtype"]
            or not bool(np.isfinite(mu).all())
            or not bool(np.isfinite(sigma).all())
        ):
            raise ValueError("metrics cache FID statistics are malformed")
        metadata[role] = {
            "kind": "fid_statistics",
            "mu_shape": list(mu.shape),
            "mu_dtype": str(mu.dtype),
            "sigma_shape": list(sigma.shape),
            "sigma_dtype": str(sigma.dtype),
            "finite": True,
        }

    features = loaded["inception_features_2048"].numpy()
    statistics = loaded["inception_fid_statistics"]
    mu = np.mean(features, axis=0)
    if not np.array_equal(mu, statistics["mu"]):
        raise ValueError("cached FID mean does not exactly match cached features")
    diagonal = np.var(features, axis=0, ddof=1, dtype=np.float64)
    expected_diagonal = np.diag(statistics["sigma"])
    diagonal_error = float(np.max(np.abs(diagonal - expected_diagonal)))
    if not np.allclose(diagonal, expected_diagonal, rtol=1e-12, atol=1e-12):
        raise ValueError("cached FID covariance diagonal differs from cached features")

    indices = [int(index) for index in covariance_probe_indices]
    feature_width = int(features.shape[1])
    if (
        not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= feature_width
    ):
        raise ValueError("FID covariance probe indices are invalid")
    probed = np.cov(features[:, indices], rowvar=False)
    expected_probed = statistics["sigma"][np.ix_(indices, indices)]
    probe_error = float(np.max(np.abs(probed - expected_probed)))
    if not np.allclose(probed, expected_probed, rtol=1e-12, atol=1e-12):
        raise ValueError("cached FID covariance probes differ from cached features")
    if not math.isfinite(diagonal_error) or not math.isfinite(probe_error):
        raise ValueError("metrics cache consistency error is not finite")
    return {
        "payloads": metadata,
        "fid_statistics_consistency": {
            "mean_exact": True,
            "covariance_diagonal_allclose": True,
            "covariance_diagonal_max_abs_error": diagonal_error,
            "covariance_probe_indices": indices,
            "covariance_probe_allclose": True,
            "covariance_probe_max_abs_error": probe_error,
            "rtol": 1e-12,
            "atol": 1e-12,
        },
    }


def _weight_identity(path: str | Path, *, label: str) -> dict[str, Any]:
    declared = Path(os.path.abspath(Path(path).expanduser()))
    if not declared.is_file():
        raise FileNotFoundError(f"{label} is missing: {declared}")
    resolved = declared.resolve()
    reject_symlink_chain(resolved, name=f"{label} physical target")
    if declared.name != resolved.name:
        raise ValueError(f"{label} alias changes the checkpoint filename")
    physical = file_identity(resolved)
    identity = {
        "path": declared.as_posix(),
        "resolved_path": physical["path"],
        "bytes": physical["bytes"],
        "sha256": physical["sha256"],
    }
    match = re.search(r"-([0-9a-f]{8})\.pth$", declared.name)
    if match is None or not str(identity["sha256"]).startswith(match.group(1)):
        raise ValueError(f"{label} filename digest does not match physical SHA256")
    return identity


def verify_real_set_identity(real_set: Mapping[str, Any]) -> dict[str, Any]:
    if (
        real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or not _is_sha256(real_set.get("sha256"))
        or int(real_set.get("image_count", -1)) < 1
    ):
        raise ValueError("metrics trust receipt real-set identity is malformed")
    root = reject_symlink_chain(
        Path(str(real_set.get("root", ""))),
        name="metrics trust real set",
    ).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"metrics trust real set is missing: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    observed = {
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "root": root.as_posix(),
        "image_count": len(images),
        "sha256": image_tree_sha256(images, root=root),
    }
    if observed != dict(real_set):
        raise ValueError("metrics trust real-set physical identity differs")
    return observed


def validate_dependency_repair_report(
    report: Mapping[str, Any],
    *,
    inception_weight: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 1
        or report.get("dependency") != "torch-fidelity inception-v3-compat"
        or report.get("status") != "pass"
        or report.get("cache_path") != inception_weight["path"]
        or int(report.get("cache_bytes", -1)) != int(inception_weight["bytes"])
        or report.get("cache_sha256") != inception_weight["sha256"]
        or report.get("expected_sha256_prefix") != str(inception_weight["sha256"])[:8]
    ):
        raise ValueError("torch-fidelity Inception repair evidence differs")
    return {
        "dependency": report["dependency"],
        "status": report["status"],
        "source_url": report.get("source_url"),
        "cache_path": report["cache_path"],
        "cache_bytes": int(report["cache_bytes"]),
        "cache_sha256": report["cache_sha256"],
        "state_dict_entries": int(report.get("state_dict_entries", -1)),
    }


def validate_cache_creation_evidence(
    *,
    creation_report: Mapping[str, Any],
    creation_log_text: str,
    cache_root: Path,
    cache_paths: Mapping[str, Path],
    real_cache_name: str,
    real_set: Mapping[str, Any],
    implementation: Mapping[str, Any],
    vgg_weight: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        creation_report.get("status") != "completed"
        or creation_report.get("protocol") != "torch_fidelity_directory_metrics"
        or creation_report.get("implementation") != dict(implementation)
        or creation_report.get("real_set") != dict(real_set)
        or creation_report.get("parameters", {}).get("real_cache_name")
        != real_cache_name
    ):
        raise ValueError("torch-fidelity cache creation report differs")
    paths = creation_report.get("paths")
    if not isinstance(paths, Mapping):
        raise TypeError("torch-fidelity cache creation report paths are missing")
    real_dir = Path(str(paths.get("real_dir", ""))).as_posix()
    generated_dir = Path(str(paths.get("generated_dir", "")))
    original_report = (
        generated_dir.parent / "metrics" / "generation_metrics_report.json"
    )
    expected_real_count_line = f"Found {int(real_set['image_count'])} samples"
    real_path_line = f'Looking for samples recursively in "{real_dir}"'
    vgg_download = (
        f'Downloading: "https://download.pytorch.org/models/{Path(vgg_weight["path"]).name}" '
        f"to {vgg_weight['path']}"
    )
    cache_lines = {
        role: f"Caching {path.as_posix()}" for role, path in cache_paths.items()
    }
    required = {
        "inception_real_extraction": "Extracting features from input2",
        "inception_real_path": real_path_line,
        "inception_real_count": expected_real_count_line,
        "inception_features_cache": cache_lines["inception_features_2048"],
        "inception_logits_cache": cache_lines["inception_logits_unbiased"],
        "inception_statistics_cache": cache_lines["inception_fid_statistics"],
        "vgg_create": "Creating feature extractor \"vgg16\" with features ['fc2_relu']",
        "vgg_weight_download": vgg_download,
        "vgg_real_extraction": "Extracting features from input2",
        "vgg_real_path": real_path_line,
        "vgg_real_count": expected_real_count_line,
        "vgg_features_cache": cache_lines["vgg_features_fc2_relu"],
        "report_write": f"wrote {original_report.as_posix()}",
    }

    positions: dict[str, int] = {}
    cursor = 0
    for name in (
        "inception_real_extraction",
        "inception_real_path",
        "inception_real_count",
        "inception_features_cache",
        "inception_logits_cache",
        "inception_statistics_cache",
        "vgg_create",
        "vgg_weight_download",
        "vgg_real_extraction",
        "vgg_real_path",
        "vgg_real_count",
        "vgg_features_cache",
        "report_write",
    ):
        position = creation_log_text.find(required[name], cursor)
        if position < 0:
            raise ValueError(f"cache creation log lacks ordered evidence: {name}")
        positions[name] = position
        cursor = position + len(required[name])
    return {
        "ordered_event_sequence_verified": True,
        "event_offsets": positions,
        "real_dir": real_dir,
        "real_image_count": int(real_set["image_count"]),
        "original_metrics_report": original_report.as_posix(),
        "cache_root": cache_root.as_posix(),
    }


def build_metrics_trust_receipt(
    *,
    git: Mapping[str, Any],
    cache_root: str | Path,
    real_cache_name: str,
    real_set: Mapping[str, Any],
    creation_log: str | Path,
    creation_report: str | Path,
    inception_repair_report: str | Path,
    inception_weight: str | Path,
    vgg_weight: str | Path,
    torch_fidelity_package_root: str | Path,
    torch_fidelity_dist_info_root: str | Path,
) -> dict[str, Any]:
    observed_real_set = verify_real_set_identity(real_set)
    expected_cache_name = f"imagenet256_val_50k_torch_fidelity_v04__cofitok_{str(real_set['sha256'])[:16]}"
    if real_cache_name != expected_cache_name:
        raise ValueError("metrics trust receipt cache name is not content-addressed")
    root = reject_symlink_chain(cache_root, name="torch-fidelity cache root").resolve()
    paths = expected_cache_paths(root, real_cache_name=real_cache_name)
    cache_identities = cache_file_identities(paths)
    semantic = validate_cache_payloads(paths, image_count=int(real_set["image_count"]))
    implementation = torch_fidelity_distribution_identity(
        torch_fidelity_package_root,
        torch_fidelity_dist_info_root,
    )
    inception_identity = _weight_identity(
        inception_weight,
        label="torch-fidelity Inception weight",
    )
    vgg_identity = _weight_identity(vgg_weight, label="torch-fidelity VGG weight")
    creation_log_path = reject_symlink_chain(
        creation_log,
        name="torch-fidelity cache creation log",
    ).resolve()
    creation_report_path = reject_symlink_chain(
        creation_report,
        name="torch-fidelity cache creation report",
    ).resolve()
    repair_path = reject_symlink_chain(
        inception_repair_report,
        name="torch-fidelity Inception repair report",
    ).resolve()
    creation_report_payload = read_json_object(
        creation_report_path,
        name="torch-fidelity cache creation report",
    )
    repair_payload = read_json_object(
        repair_path,
        name="torch-fidelity Inception repair report",
    )
    repair = validate_dependency_repair_report(
        repair_payload,
        inception_weight=inception_identity,
    )
    creation = validate_cache_creation_evidence(
        creation_report=creation_report_payload,
        creation_log_text=creation_log_path.read_text(
            encoding="utf-8", errors="replace"
        ),
        cache_root=root,
        cache_paths=paths,
        real_cache_name=real_cache_name,
        real_set=observed_real_set,
        implementation={
            "package": implementation["package"],
            "version": implementation["version"],
        },
        vgg_weight=vgg_identity,
    )
    return {
        "schema_version": METRICS_TRUST_RECEIPT_SCHEMA_VERSION,
        "role": METRICS_TRUST_RECEIPT_ROLE,
        "status": "pass",
        "detail": "torch_fidelity_real_cache_and_dependencies_physically_attested",
        "git": copy.deepcopy(dict(git)),
        "real_set": observed_real_set,
        "cache": {
            "root": root.as_posix(),
            "real_cache_name": real_cache_name,
            "files": cache_identities,
            "semantic_validation": semantic,
        },
        "feature_extractor_weights": {
            "inception_v3_compat": inception_identity,
            "vgg16": vgg_identity,
        },
        "torch_fidelity": implementation,
        "origin_evidence": {
            "creation_log": file_identity(creation_log_path),
            "creation_report": file_identity(creation_report_path),
            "inception_repair_report": file_identity(repair_path),
            "creation_validation": creation,
            "inception_repair_validation": repair,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def _replay_identity(descriptor: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{label} identity is missing")
    expected = dict(descriptor)
    if int(expected.get("bytes", 0)) < 1 or not _is_sha256(expected.get("sha256")):
        raise ValueError(f"{label} identity is malformed")
    actual = file_identity(Path(str(expected.get("path", ""))))
    if actual != expected:
        raise ValueError(f"{label} changed after attestation")
    return actual


def verify_metrics_trust_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if (
        receipt.get("schema_version") != METRICS_TRUST_RECEIPT_SCHEMA_VERSION
        or receipt.get("role") != METRICS_TRUST_RECEIPT_ROLE
        or receipt.get("status") != "pass"
        or receipt.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("metrics trust receipt contract differs")
    cache = receipt.get("cache")
    real_set = receipt.get("real_set")
    weights = receipt.get("feature_extractor_weights")
    implementation = receipt.get("torch_fidelity")
    origin = receipt.get("origin_evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (cache, real_set, weights, implementation, origin)
    ):
        raise ValueError("metrics trust receipt evidence is incomplete")
    observed_real_set = verify_real_set_identity(real_set)
    root = reject_symlink_chain(
        Path(str(cache.get("root", ""))),
        name="attested torch-fidelity cache root",
    ).resolve()
    real_cache_name = str(cache.get("real_cache_name", ""))
    paths = expected_cache_paths(root, real_cache_name=real_cache_name)
    observed_cache = cache_file_identities(paths)
    if observed_cache != cache.get("files"):
        raise ValueError("attested torch-fidelity cache files changed")
    semantic = validate_cache_payloads(paths, image_count=int(real_set["image_count"]))
    if semantic != cache.get("semantic_validation"):
        raise ValueError("attested torch-fidelity cache semantics changed")
    observed_weights = {
        "inception_v3_compat": _weight_identity(
            weights.get("inception_v3_compat", {}).get("path", ""),
            label="attested torch-fidelity Inception weight",
        ),
        "vgg16": _weight_identity(
            weights.get("vgg16", {}).get("path", ""),
            label="attested torch-fidelity VGG weight",
        ),
    }
    if observed_weights != dict(weights):
        raise ValueError("attested torch-fidelity feature extractor weights changed")
    observed_implementation = torch_fidelity_distribution_identity(
        implementation.get("package_tree", {}).get("root", ""),
        implementation.get("dist_info_tree", {}).get("root", ""),
    )
    if observed_implementation != dict(implementation):
        raise ValueError("attested torch-fidelity installation changed")
    origin_identities = {
        name: _replay_identity(origin.get(name), label=f"metrics trust {name}")
        for name in (
            "creation_log",
            "creation_report",
            "inception_repair_report",
        )
    }
    creation_report = read_json_object(
        Path(origin_identities["creation_report"]["path"]),
        name="attested cache creation report",
    )
    repair_report = read_json_object(
        Path(origin_identities["inception_repair_report"]["path"]),
        name="attested Inception repair report",
    )
    creation = validate_cache_creation_evidence(
        creation_report=creation_report,
        creation_log_text=Path(origin_identities["creation_log"]["path"]).read_text(
            encoding="utf-8",
            errors="replace",
        ),
        cache_root=root,
        cache_paths=paths,
        real_cache_name=real_cache_name,
        real_set=observed_real_set,
        implementation={
            "package": implementation["package"],
            "version": implementation["version"],
        },
        vgg_weight=observed_weights["vgg16"],
    )
    repair = validate_dependency_repair_report(
        repair_report,
        inception_weight=observed_weights["inception_v3_compat"],
    )
    if creation != origin.get("creation_validation") or repair != origin.get(
        "inception_repair_validation"
    ):
        raise ValueError("metrics trust origin validation changed")
    return {
        "status": "verified",
        "real_set": observed_real_set,
        "cache_root": root.as_posix(),
        "real_cache_name": real_cache_name,
        "cache_files": observed_cache,
        "semantic_validation": semantic,
        "feature_extractor_weights": observed_weights,
        "torch_fidelity": observed_implementation,
        "origin_evidence": origin_identities,
    }


def validate_generation_reports_against_metrics_trust(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    verified_trust: Mapping[str, Any],
) -> dict[str, Any]:
    expected_methods = {"cofitok", "dense_identity"}
    if set(reports) != expected_methods:
        raise ValueError("terminal generation report set differs")
    expected_implementation = {
        "package": verified_trust["torch_fidelity"]["package"],
        "version": verified_trust["torch_fidelity"]["version"],
    }
    rows: dict[str, Any] = {}
    for method in sorted(reports):
        report = reports[method]
        parameters = report.get("parameters")
        if (
            report.get("schema_version") != 3
            or report.get("role") != "generation_directory_metrics_report"
            or report.get("status") != "completed"
            or report.get("protocol") != "torch_fidelity_directory_metrics"
            or report.get("implementation") != expected_implementation
            or report.get("real_set") != verified_trust["real_set"]
            or not isinstance(parameters, Mapping)
            or parameters.get("cache_root") != verified_trust["cache_root"]
            or parameters.get("real_cache_name") != verified_trust["real_cache_name"]
            or parameters.get("precision_recall_enabled") is not True
        ):
            raise ValueError(
                f"{method} terminal metrics report is not bound to attested cache inputs"
            )
        rows[method] = {
            "implementation": copy.deepcopy(dict(report["implementation"])),
            "runtime_environment_sha256": report.get("runtime_environment_sha256"),
            "real_set": copy.deepcopy(dict(report["real_set"])),
            "cache_root": parameters["cache_root"],
            "real_cache_name": parameters["real_cache_name"],
            "precision_recall_enabled": True,
        }
    if rows["cofitok"] != rows["dense_identity"]:
        raise ValueError("terminal generation pair uses different metrics trust inputs")
    return {
        "status": "verified",
        "methods": rows,
        "matched_attested_metrics_inputs": True,
    }
