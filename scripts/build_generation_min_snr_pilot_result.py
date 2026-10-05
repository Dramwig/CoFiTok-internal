from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.min_snr_pilot import METHODS, build_result
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPES = ("legacy_gamma0", "pilot_gamma5")
KINDS = (
    "generation",
    "class_fidelity",
    "checkpoint_eval",
    "sampling_preflight",
)


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-gate", required=True)
    parser.add_argument("--expected-execution-gate-sha256", required=True)
    parser.add_argument("--pilot-cofitok-training-audit", required=True)
    parser.add_argument("--pilot-dense-training-audit", required=True)
    for recipe in RECIPES:
        for method in METHODS:
            stem = f"{recipe.replace('_', '-')}-{method.replace('_', '-')}"
            for kind in KINDS:
                parser.add_argument(
                    f"--{stem}-{kind.replace('_', '-')}",
                    dest=f"{recipe}_{method}_{kind}",
                    required=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the bounded matched Min-SNR 50K pilot result."
    )
    add_source_arguments(parser)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _builder_git() -> dict[str, Any]:
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {**git_provenance(PROJECT_ROOT), "tree": tree}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    preparation_identity = _identity(args.preparation)
    gate_identity = _identity(args.execution_gate)
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("Min-SNR pilot preparation SHA256 differs")
    if gate_identity["sha256"] != args.expected_execution_gate_sha256:
        raise ValueError("Min-SNR pilot execution gate SHA256 differs")
    audit_paths = {
        "cofitok": args.pilot_cofitok_training_audit,
        "dense_identity": args.pilot_dense_training_audit,
    }
    arms: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, str] = {
        "preparation": args.preparation,
        "execution_gate": args.execution_gate,
        "pilot_cofitok_training_audit": args.pilot_cofitok_training_audit,
        "pilot_dense_training_audit": args.pilot_dense_training_audit,
    }
    for recipe in RECIPES:
        for method in METHODS:
            arm = f"{recipe}_{method}"
            arms[arm] = {}
            for kind in KINDS:
                path = getattr(args, f"{arm}_{kind}")
                arms[arm][kind] = _read(path)
                source_paths[f"{arm}_{kind}"] = path
    return build_result(
        preparation=_read(args.preparation),
        preparation_identity=preparation_identity,
        execution_gate=_read(args.execution_gate),
        execution_gate_identity=gate_identity,
        pilot_training_audits={
            method: _read(path) for method, path in audit_paths.items()
        },
        arms=arms,
        source_identities={
            name: _identity(path) for name, path in source_paths.items()
        },
        builder_git=_builder_git(),
    )


def main() -> None:
    args = parse_args()
    write_json_report(args.output, build_report(args))
    print(file_sha256(args.output))


if __name__ == "__main__":
    main()

