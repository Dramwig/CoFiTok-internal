from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_paper_integration import (
        PAPER_INTEGRATION_ROLE,
        PAPER_INTEGRATION_SCHEMA_VERSION,
        PAPER_INTEGRATION_STATUS,
        build_paper_integration_manifest,
        checkout_identity,
        file_identity,
    )
except ModuleNotFoundError:  # Direct execution with scripts/ on sys.path.
    from build_generation_paper_integration import (  # type: ignore[no-redef]
        PAPER_INTEGRATION_ROLE,
        PAPER_INTEGRATION_SCHEMA_VERSION,
        PAPER_INTEGRATION_STATUS,
        build_paper_integration_manifest,
        checkout_identity,
        file_identity,
    )


VALIDATION_SCHEMA_VERSION = 1
VALIDATION_ROLE = "terminal_generation_paper_integration_validation"


def _read_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"{name} is missing or is a symlink: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def validate_paper_integration_manifest(
    manifest_path: str | Path,
    *,
    implementation_git: dict[str, Any],
    release_verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = _read_object(manifest_path, name="paper-integration manifest")
    if (
        manifest.get("schema_version") != PAPER_INTEGRATION_SCHEMA_VERSION
        or manifest.get("role") != PAPER_INTEGRATION_ROLE
        or manifest.get("status") != PAPER_INTEGRATION_STATUS
        or manifest.get("implementation_git") != implementation_git
    ):
        raise ValueError("paper-integration manifest header or Git identity differs")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("paper-integration manifest lacks source identities")
    for name in ("completion_audit", "comparison", "final_gate", "release_receipt"):
        descriptor = sources.get(name)
        if not isinstance(descriptor, dict) or file_identity(descriptor.get("path", "")) != descriptor:
            raise ValueError(f"paper-integration source changed: {name}")
    rebuilt, contents = build_paper_integration_manifest(
        completion_audit_path=sources["completion_audit"]["path"],
        comparison_path=sources["comparison"]["path"],
        final_gate_path=sources["final_gate"]["path"],
        release_receipt_path=sources["release_receipt"]["path"],
        output_dir=manifest.get("output_root", ""),
        implementation_git=implementation_git,
        release_verifier=release_verifier,
    )
    if rebuilt != manifest:
        raise ValueError("paper-integration manifest differs from rebuilt evidence")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(contents):
        raise ValueError("paper-integration output set differs")
    for name, descriptor in outputs.items():
        if file_identity(descriptor["path"]) != descriptor:
            raise ValueError(f"paper-integration output changed: {name}")
        if Path(descriptor["path"]).read_text(encoding="utf-8") != contents[name]:
            raise ValueError(f"paper-integration output content differs: {name}")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "manifest": file_identity(manifest_path),
        "implementation_git": dict(implementation_git),
        "completion_profile": manifest["completion_profile"],
        "sources": manifest["sources"],
        "outputs": manifest["outputs"],
        "paper_application_ready": True,
        "post_application_lock_required": True,
        "authorization_boundary": {
            "gpu_execution_authorized": False,
            "training_authorized": False,
            "evaluation_authorized": False,
        },
    }


def write_validation_receipt(
    manifest_path: str | Path,
    output: str | Path,
    *,
    implementation_git: dict[str, Any],
    release_verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt = validate_paper_integration_manifest(
        manifest_path,
        implementation_git=implementation_git,
        release_verifier=release_verifier,
    )
    target = Path(output)
    if target.exists():
        existing = _read_object(target, name="paper-integration validation receipt")
        if existing != receipt:
            raise ValueError("existing paper-integration validation receipt differs")
        return existing
    write_json_report(target, receipt)
    written = _read_object(target, name="paper-integration validation receipt")
    if written != receipt:
        raise ValueError("written paper-integration validation receipt differs")
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a terminal generation paper-integration bundle and emit an "
            "immutable validation receipt."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = write_validation_receipt(
        args.manifest,
        args.output,
        implementation_git=checkout_identity(args.project_root),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
