from __future__ import annotations

import argparse
from pathlib import Path

try:
    from deploy_generation_postbridge_two_stage import build_manifest, write_json_atomic
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts.deploy_generation_postbridge_two_stage import (
        build_manifest,
        write_json_atomic,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-bound two-stage post-bridge deployment manifest."
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--formal-repository-path", required=True)
    parser.add_argument("--formal-revision", required=True)
    parser.add_argument("--formal-branch", default="scale/generative-system")
    parser.add_argument("--bootstrap-bundle", type=Path, required=True)
    parser.add_argument("--bootstrap-revision", required=True)
    parser.add_argument("--bootstrap-ref", required=True)
    parser.add_argument(
        "--bootstrap-prerequisite",
        action="append",
        required=True,
    )
    parser.add_argument("--integration-bundle", type=Path, required=True)
    parser.add_argument("--integration-revision", required=True)
    parser.add_argument("--integration-ref", required=True)
    parser.add_argument("--quality-bridge-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_manifest(
        repository=args.repository,
        formal_repository_path=args.formal_repository_path,
        formal_revision=args.formal_revision,
        formal_branch=args.formal_branch,
        bootstrap_bundle=args.bootstrap_bundle,
        bootstrap_revision=args.bootstrap_revision,
        bootstrap_ref=args.bootstrap_ref,
        bootstrap_prerequisites=args.bootstrap_prerequisite,
        integration_bundle=args.integration_bundle,
        integration_revision=args.integration_revision,
        integration_ref=args.integration_ref,
        quality_bridge_root=args.quality_bridge_root,
    )
    write_json_atomic(args.output, report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
