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
from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import resolve_latest_checkpoint, verify_training_checkpoint


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"source is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _verify_checkpoint_binding(binding: dict[str, Any] | None) -> None:
    if binding is None:
        return
    run_dir = Path(str(binding["run_dir"])).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"source checkpoint run directory is missing: {run_dir}")
    for name in ("checkpoint", "integrity_manifest", "latest"):
        expected = dict(binding[name])
        actual = _identity(Path(str(expected["path"])))
        if actual != expected:
            raise ValueError(f"source checkpoint binding {name} changed")
        if Path(actual["path"]).parent != run_dir:
            raise ValueError(f"source checkpoint binding {name} is outside run directory")
    checkpoint = resolve_latest_checkpoint(run_dir).resolve()
    if checkpoint != Path(str(binding["checkpoint"]["path"])).resolve():
        raise ValueError("source latest pointer resolves to another checkpoint")
    expected_integrity = checkpoint.with_name(f"{checkpoint.name}.integrity.json").resolve()
    if expected_integrity != Path(str(binding["integrity_manifest"]["path"])).resolve():
        raise ValueError("source checkpoint sidecar path differs")
    if (run_dir / "latest.json").resolve() != Path(str(binding["latest"]["path"])).resolve():
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
    if file_sha256(args.gate) != args.expected_gate_sha256:
        raise ValueError("gate SHA256 differs")
    if file_sha256(args.preparation) != args.expected_preparation_sha256:
        raise ValueError("preparation SHA256 differs")
    preparation = _read(args.preparation)
    validate_preparation(preparation)
    source_count = _verify_preparation_sources(preparation)
    _verify_preparation_training_sources(preparation)
    gate = _read(args.gate)
    validated = validate_execution_gate(
        gate,
        preparation=preparation,
        preparation_identity=_identity(args.preparation),
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
