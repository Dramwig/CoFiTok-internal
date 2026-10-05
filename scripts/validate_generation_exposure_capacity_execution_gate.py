"""Validate a non-authorizing exposure/capacity gate and its sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity import (
    validate_preparation,
    validate_source_checkpoint_bindings,
)
from cofitok.generation.exposure_capacity_gate import validate_execution_gate
from cofitok.path_security import reject_symlink_chain
from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import resolve_latest_checkpoint, verify_training_checkpoint


def _read(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="gate source")
    if not source.is_file():
        raise FileNotFoundError(f"gate source is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {source}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="gate source")
    if not source.is_file():
        raise FileNotFoundError(f"source is missing: {source}")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _verify_checkpoint_binding(binding: dict[str, Any] | None) -> None:
    if binding is None:
        return
    run_dir = reject_symlink_chain(
        Path(str(binding["run_dir"])),
        name="source checkpoint run directory",
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"source checkpoint run directory is missing: {run_dir}")
    for name in ("checkpoint", "integrity_manifest", "latest"):
        expected = dict(binding[name])
        actual = _identity(Path(str(expected["path"])))
        if actual != expected:
            raise ValueError(f"source checkpoint binding {name} changed")
        if Path(actual["path"]).parent != run_dir:
            raise ValueError(f"source checkpoint binding {name} is outside run directory")
    checkpoint = resolve_latest_checkpoint(run_dir)
    expected_checkpoint = reject_symlink_chain(
        Path(str(binding["checkpoint"]["path"])),
        name="source checkpoint",
    )
    if checkpoint != expected_checkpoint:
        raise ValueError("source latest pointer resolves to another checkpoint")
    expected_integrity = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    supplied_integrity = reject_symlink_chain(
        Path(str(binding["integrity_manifest"]["path"])),
        name="source checkpoint integrity manifest",
    )
    if expected_integrity != supplied_integrity:
        raise ValueError("source checkpoint sidecar path differs")
    supplied_latest = reject_symlink_chain(
        Path(str(binding["latest"]["path"])),
        name="source latest pointer",
    )
    if run_dir / "latest.json" != supplied_latest:
        raise ValueError("source latest pointer path differs")
    integrity = verify_training_checkpoint(checkpoint)
    if int(integrity.get("step", -1)) != int(binding["step"]):
        raise ValueError("source checkpoint step differs")


def _verify_preparation_sources(preparation: dict[str, Any]) -> int:
    sources = preparation.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("preparation source descriptors are missing")
    for name, descriptor in sources.items():
        expected = dict(descriptor)
        actual = _identity(Path(str(expected.get("path", ""))))
        if actual != expected:
            raise ValueError(f"preparation source {name} changed")
    return len(sources)


def _verify_preparation_training_sources(preparation: dict[str, Any]) -> None:
    sources = preparation.get("sources")
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
        reports[method] = _read(Path(str(descriptor.get("path", ""))))
    validate_source_checkpoint_bindings(
        preparation,
        cofitok_training_report=reports["cofitok"],
        dense_training_report=reports["dense_identity"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an exposure/capacity gate.")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    args = parser.parse_args()
    gate_path = reject_symlink_chain(args.gate, name="execution gate")
    preparation_path = reject_symlink_chain(args.preparation, name="preparation report")
    if not gate_path.is_file():
        raise FileNotFoundError(f"execution gate is missing: {gate_path}")
    if not preparation_path.is_file():
        raise FileNotFoundError(f"preparation report is missing: {preparation_path}")
    if file_sha256(gate_path) != args.expected_gate_sha256:
        raise ValueError("gate SHA256 differs")
    if file_sha256(preparation_path) != args.expected_preparation_sha256:
        raise ValueError("preparation SHA256 differs")
    preparation = _read(preparation_path)
    validate_preparation(preparation)
    source_count = _verify_preparation_sources(preparation)
    _verify_preparation_training_sources(preparation)
    gate = _read(gate_path)
    validated = validate_execution_gate(
        gate,
        preparation=preparation,
        preparation_identity=_identity(preparation_path),
    )
    source_checkpoint = validated.get("source_checkpoint")
    _verify_checkpoint_binding(source_checkpoint)
    print(
        json.dumps(
            {
                "schema_version": validated["schema_version"],
                "status": "pass",
                "selected_arm": validated["selected_arm"],
                "execution_ready": validated["execution_ready"],
                "source_checkpoint_verified": source_checkpoint is None
                or isinstance(source_checkpoint, dict),
                "source_count": source_count,
                "authorization_boundary": validated["authorization_boundary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
