from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cofitok.reporting import file_sha256


DATASET_PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FormalDatasetProvenanceSpec:
    dataset: str
    manifest_sha256: str
    manifest_bytes: int
    train_images: int
    val_images: int


FORMAL_GENERATION_DATASETS = {
    "imagenet_256_10pct": FormalDatasetProvenanceSpec(
        dataset="imagenet_256_10pct",
        manifest_sha256="dcdd622564941ad418fffa960051f14f0ad41852c681ed9a8329c5e6f561cca7",
        manifest_bytes=54_885_982,
        train_images=128_161,
        val_images=50_000,
    ),
    "imagenet_256": FormalDatasetProvenanceSpec(
        dataset="imagenet_256",
        manifest_sha256="9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0",
        manifest_bytes=405_484_553,
        train_images=1_281_167,
        val_images=50_000,
    ),
}


def _identity_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    manifest = report.get("manifest")
    splits = report.get("splits")
    if not isinstance(manifest, Mapping) or not isinstance(splits, Mapping):
        raise ValueError("dataset provenance lacks manifest or split identity")
    return {
        "schema_version": int(report.get("schema_version", -1)),
        "dataset": str(report.get("dataset", "")),
        "dataset_root": str(report.get("dataset_root", "")),
        "manifest": {
            "relative_path": str(manifest.get("relative_path", "")),
            "bytes": int(manifest.get("bytes", -1)),
            "sha256": str(manifest.get("sha256", "")),
        },
        "splits": {
            "train": int(splits.get("train", -1)),
            "val": int(splits.get("val", -1)),
        },
    }


def dataset_provenance_identity_sha256(report: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _identity_payload(report),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def capture_dataset_provenance(
    data_config: Any,
    *,
    train_images: int,
    val_images: int,
) -> dict[str, Any]:
    dataset = str(data_config.dataset).lower()
    configured_root = Path(data_config.root) / dataset
    dataset_root = configured_root.resolve()
    spec = FORMAL_GENERATION_DATASETS.get(dataset)
    if spec is None:
        return {
            "schema_version": DATASET_PROVENANCE_SCHEMA_VERSION,
            "status": "not_formal",
            "formal": False,
            "dataset": dataset,
            "dataset_root": str(dataset_root),
            "splits": {"train": int(train_images), "val": int(val_images)},
        }

    manifest_path = configured_root / "metadata" / "image_manifest.jsonl"
    issues = []
    if configured_root.is_symlink():
        issues.append("dataset root is a symlink")
    if manifest_path.is_symlink():
        issues.append("dataset manifest is a symlink")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"formal dataset manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.stat().st_size
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_bytes != spec.manifest_bytes:
        issues.append(
            f"manifest bytes expected {spec.manifest_bytes}, got {manifest_bytes}"
        )
    if manifest_sha256 != spec.manifest_sha256:
        issues.append("manifest SHA256 differs from the formal dataset record")
    if int(train_images) != spec.train_images:
        issues.append(f"train images expected {spec.train_images}, got {train_images}")
    if int(val_images) != spec.val_images:
        issues.append(f"val images expected {spec.val_images}, got {val_images}")

    report: dict[str, Any] = {
        "schema_version": DATASET_PROVENANCE_SCHEMA_VERSION,
        "status": "pass" if not issues else "fail",
        "formal": True,
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": manifest_bytes,
            "sha256": manifest_sha256,
        },
        "splits": {"train": int(train_images), "val": int(val_images)},
        "issues": issues,
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def validate_dataset_provenance(
    report: Mapping[str, Any],
    *,
    expected_dataset: str,
) -> dict[str, Any]:
    if report.get("schema_version") != DATASET_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("dataset provenance schema is unsupported")
    if report.get("formal") is not True or report.get("status") != "pass":
        raise ValueError("formal dataset provenance did not pass")
    if report.get("issues") != []:
        raise ValueError("formal dataset provenance contains issues")
    dataset = str(report.get("dataset", ""))
    if dataset != expected_dataset:
        raise ValueError(f"dataset provenance does not describe {expected_dataset}")
    spec = FORMAL_GENERATION_DATASETS.get(expected_dataset)
    if spec is None:
        raise ValueError(f"no formal dataset provenance spec for {expected_dataset}")
    identity = _identity_payload(report)
    expected_manifest = {
        "relative_path": "metadata/image_manifest.jsonl",
        "bytes": spec.manifest_bytes,
        "sha256": spec.manifest_sha256,
    }
    expected_splits = {"train": spec.train_images, "val": spec.val_images}
    if identity["manifest"] != expected_manifest:
        raise ValueError("dataset manifest identity differs from the formal record")
    if identity["splits"] != expected_splits:
        raise ValueError("dataset split identity differs from the formal record")
    if not identity["dataset_root"]:
        raise ValueError("dataset provenance root is missing")
    actual_sha256 = dataset_provenance_identity_sha256(report)
    if report.get("identity_sha256") != actual_sha256:
        raise ValueError("dataset provenance identity SHA256 is inconsistent")
    return {
        "dataset": expected_dataset,
        "dataset_root": identity["dataset_root"],
        "manifest_sha256": spec.manifest_sha256,
        "manifest_bytes": spec.manifest_bytes,
        "splits": expected_splits,
        "identity_sha256": actual_sha256,
    }
