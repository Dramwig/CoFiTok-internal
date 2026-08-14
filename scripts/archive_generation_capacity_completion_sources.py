from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_completion_decision import (
    validate_capacity_completion_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from cofitok.training.checkpointing import verify_training_checkpoint


ARCHIVE_BOUNDARY = {
    "source_checkpoint_mutation_allowed": False,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}


def _assert_identity(path: Path, expected: dict[str, Any], *, label: str) -> None:
    if file_identity(path) != expected:
        raise ValueError(f"{label} physical identity differs")


def build_source_archive(
    *,
    decision_path: Path,
    expected_decision_sha256: str,
    expected_decision_revision: str,
    expected_decision_tree: str,
    expected_decision_branch: str,
    output_root: Path,
    create_missing: bool,
) -> dict[str, Any]:
    decision_identity = file_identity(decision_path)
    if decision_identity["sha256"] != expected_decision_sha256:
        raise ValueError("capacity completion decision SHA256 differs")
    decision = read_json_object(decision_path, name="capacity completion decision")
    evidence = validate_capacity_completion_decision(
        decision,
        expected_decision_revision=expected_decision_revision,
        expected_decision_tree=expected_decision_tree,
        expected_decision_branch=expected_decision_branch,
    )
    if evidence["execution_authorized"] is not True:
        raise ValueError("capacity completion decision does not authorize execution")
    root = output_root.resolve()
    if evidence["output_root"] != root.as_posix():
        raise ValueError("capacity completion archive output root differs")
    archive_root = root / "reports/capacity_completion_100k/source_checkpoints"
    methods = {}
    for method in ("cofitok", "dense_identity"):
        expected = evidence["resume_sources"][method]
        source_checkpoint = Path(expected["checkpoint"]["path"]).resolve()
        source_integrity = Path(
            expected["checkpoint_integrity_manifest"]["path"]
        ).resolve()
        try:
            source_checkpoint.relative_to(root)
            source_integrity.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"capacity completion {method} source escapes output root"
            ) from exc
        _assert_identity(
            source_checkpoint,
            expected["checkpoint"],
            label=f"capacity completion {method} checkpoint",
        )
        _assert_identity(
            source_integrity,
            expected["checkpoint_integrity_manifest"],
            label=f"capacity completion {method} integrity",
        )
        verified = verify_training_checkpoint(source_checkpoint)
        if int(verified.get("step", -1)) != 50_000:
            raise ValueError(f"capacity completion {method} source step differs")
        destination = archive_root / method / source_checkpoint.name
        destination_integrity = destination.with_name(
            f"{destination.name}.integrity.json"
        )
        if create_missing:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(source_checkpoint, destination)
            if not destination_integrity.exists():
                os.link(source_integrity, destination_integrity)
        if not destination.is_file() or not destination_integrity.is_file():
            raise ValueError(f"capacity completion {method} archive is missing")
        if not destination.samefile(source_checkpoint):
            raise ValueError(
                f"capacity completion {method} checkpoint archive is not a hard link"
            )
        if not destination_integrity.samefile(source_integrity):
            raise ValueError(
                f"capacity completion {method} integrity archive is not a hard link"
            )
        archive_checkpoint = file_identity(destination)
        archive_integrity = file_identity(destination_integrity)
        if (
            archive_checkpoint["bytes"] != expected["checkpoint"]["bytes"]
            or archive_checkpoint["sha256"] != expected["checkpoint"]["sha256"]
            or archive_integrity["bytes"]
            != expected["checkpoint_integrity_manifest"]["bytes"]
            or archive_integrity["sha256"]
            != expected["checkpoint_integrity_manifest"]["sha256"]
        ):
            raise ValueError(f"capacity completion {method} archive bytes differ")
        verify_training_checkpoint(destination)
        methods[method] = {
            "source_checkpoint": copy.deepcopy(expected["checkpoint"]),
            "source_integrity_manifest": copy.deepcopy(
                expected["checkpoint_integrity_manifest"]
            ),
            "archive_checkpoint": archive_checkpoint,
            "archive_integrity_manifest": archive_integrity,
            "checkpoint_same_file": True,
            "integrity_same_file": True,
            "checkpoint_link_count": destination.stat().st_nlink,
            "integrity_link_count": destination_integrity.stat().st_nlink,
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "role": "capacity_completion_50k_source_checkpoint_archive",
        "capacity_completion_decision": decision_identity,
        "output_root": root.as_posix(),
        "methods": methods,
        "authorization_boundary": copy.deepcopy(ARCHIVE_BOUNDARY),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create same-filesystem hard-link archives for the exact matched "
            "step-50K capacity-completion resume checkpoints."
        )
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-tree", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity completion source archive exists: {args.output}")
    report = build_source_archive(
        decision_path=args.decision.resolve(),
        expected_decision_sha256=args.expected_decision_sha256,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_tree=args.expected_decision_tree,
        expected_decision_branch=args.expected_decision_branch,
        output_root=args.output_root.resolve(),
        create_missing=True,
    )
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
