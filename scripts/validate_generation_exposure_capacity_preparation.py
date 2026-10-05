from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity import (
    validate_preparation,
    validate_source_checkpoint_bindings,
)
from cofitok.path_security import reject_symlink_chain
from cofitok.reporting import file_sha256


def _verify_sources(report: dict[str, Any]) -> int:
    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("preparation source descriptors are missing")
    for name, descriptor in sources.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"preparation source {name} is malformed")
        source = reject_symlink_chain(
            Path(str(descriptor.get("path", ""))),
            name=f"preparation source {name}",
        )
        if not source.is_file():
            raise FileNotFoundError(f"preparation source {name} is missing: {source}")
        actual_bytes = source.stat().st_size
        actual_sha256 = file_sha256(source)
        if descriptor.get("bytes") != actual_bytes:
            raise ValueError(f"preparation source {name} byte count changed")
        if descriptor.get("sha256") != actual_sha256:
            raise ValueError(f"preparation source {name} SHA256 changed")
    return len(sources)


def _verify_training_sources(report: dict[str, Any]) -> None:
    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("preparation source descriptors are missing")
    reports: dict[str, dict[str, Any]] = {}
    for method, source_name in (
        ("cofitok", "cofitok_training_report"),
        ("dense_identity", "dense_training_report"),
    ):
        descriptor = sources.get(source_name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"preparation source {source_name} is malformed")
        source = reject_symlink_chain(
            Path(str(descriptor.get("path", ""))),
            name=f"preparation source {source_name}",
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"preparation source {source_name} is missing: {source}"
            )
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"preparation source {source_name} is not an object")
        reports[method] = value
    validate_source_checkpoint_bindings(
        report,
        cofitok_training_report=reports["cofitok"],
        dense_training_report=reports["dense_identity"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a non-authorizing exposure/capacity preparation report."
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    args = parser.parse_args()
    preparation = reject_symlink_chain(args.preparation, name="preparation report")
    if not preparation.is_file():
        raise FileNotFoundError(f"preparation report is missing: {preparation}")
    if file_sha256(preparation) != args.expected_preparation_sha256:
        raise ValueError("preparation SHA256 differs")
    report = json.loads(preparation.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("preparation report must be a JSON object")
    validated = validate_preparation(report)
    validated["source_count"] = _verify_sources(report)
    _verify_training_sources(report)
    print(json.dumps(validated, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
