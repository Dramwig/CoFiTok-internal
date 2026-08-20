from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.factorization_quality_regression import (
    build_execution_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bound(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def build_from_paths(
    *,
    preparation_path: Path,
    expected_preparation_sha256: str,
    source_binding_path: Path,
    expected_source_binding_sha256: str,
    standing_authorization_path: Path,
    expected_standing_authorization_sha256: str,
    expected_revision: str,
    expected_branch: str,
    project: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    preparation, preparation_identity = _bound(
        preparation_path,
        expected_sha256=expected_preparation_sha256,
        label="factorization-regression preparation",
    )
    binding, binding_identity = _bound(
        source_binding_path,
        expected_sha256=expected_source_binding_sha256,
        label="factorization-regression source binding",
    )
    standing, standing_identity = _bound(
        standing_authorization_path,
        expected_sha256=expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    return build_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_identity,
        source_binding=binding,
        source_binding_identity=binding_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        execution_git=git_provenance(project),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standing-authorization-backed, source-bound execution receipt for "
            "the non-authorizing factorization quality-regression diagnostic."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--expected-source-binding-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        preparation_path=args.preparation,
        expected_preparation_sha256=args.expected_preparation_sha256,
        source_binding_path=args.source_binding,
        expected_source_binding_sha256=args.expected_source_binding_sha256,
        standing_authorization_path=args.standing_authorization,
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
