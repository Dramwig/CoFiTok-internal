from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_semantic_trajectory import (
    CHECKPOINT_STEPS,
    METHOD_RUN_DIRS,
    validate_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import to_jsonable, write_json_report
from scripts import build_generation_exposure_semantic_trajectory_report as builder


SCHEMA_VERSION = 1
ROLE = "generation_exposure_semantic_trajectory_replay_audit"
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "non_authorizing": True,
    "generates_new_samples": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_followup_training": False,
    "authorizes_full_training": False,
    "authorizes_300k_training": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_export": False,
    "authorizes_release": False,
    "authorizes_process_signals": False,
    "generation_advantage_proven": False,
}


def _git(project: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_identity(project: Path) -> dict[str, Any]:
    return {
        "revision": _git(project, "rev-parse", "HEAD"),
        "tree": _git(project, "rev-parse", "HEAD^{tree}"),
        "branch": _git(project, "branch", "--show-current"),
        "tracked_dirty": bool(
            _git(project, "status", "--porcelain", "--untracked-files=no")
        ),
    }


def _stat_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def _verify_bound_file(
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if set(expected) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} identity fields differ")
    path = reject_symlink_chain(Path(str(expected["path"])), name=label).resolve()
    before = _stat_identity(path)
    actual = file_identity(path)
    after = _stat_identity(path)
    if before != after:
        raise ValueError(f"{label} changed during physical SHA256 replay")
    if actual != dict(expected):
        raise ValueError(f"{label} physical identity differs")
    return actual


def _verify_expected_sha256(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    path = reject_symlink_chain(path, name=label).resolve()
    identity = _verify_bound_file(file_identity(path), label=label)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the expected source")
    return identity


def _render_json_bytes(value: Mapping[str, Any]) -> bytes:
    normalized = to_jsonable(dict(value))
    return (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _physical_training_inventory(
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    methods = preparation.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHOD_RUN_DIRS):
        raise ValueError("trajectory preparation method inventory differs")
    result: dict[str, Any] = {}
    total_checkpoint_bytes = 0
    for method in METHOD_RUN_DIRS:
        row = methods[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"{method} preparation row is malformed")
        source_files = row.get("source_files")
        checkpoints = row.get("checkpoints")
        if not isinstance(source_files, Mapping) or not isinstance(checkpoints, list):
            raise ValueError(f"{method} preparation inventory is malformed")
        verified_sources = {
            name: _verify_bound_file(identity, label=f"{method} {name}")
            for name, identity in sorted(source_files.items())
        }
        if [int(item.get("step", -1)) for item in checkpoints] != list(
            CHECKPOINT_STEPS
        ):
            raise ValueError(f"{method} checkpoint steps differ")
        verified_checkpoints = []
        for item in checkpoints:
            step = int(item["step"])
            if (
                item.get("physical_sha256_verified") is not True
                or item.get("sidecar_verified") is not True
                or int(item.get("samples_seen", -1)) != step * 64
            ):
                raise ValueError(f"{method} step {step} preparation binding differs")
            checkpoint = _verify_bound_file(
                item["checkpoint"],
                label=f"{method} step {step} checkpoint",
            )
            sidecar = _verify_bound_file(
                item["sidecar"],
                label=f"{method} step {step} sidecar",
            )
            total_checkpoint_bytes += int(checkpoint["bytes"])
            verified_checkpoints.append(
                {
                    "step": step,
                    "samples_seen": step * 64,
                    "checkpoint": checkpoint,
                    "sidecar": sidecar,
                    "physical_sha256_verified": True,
                }
            )
        result[method] = {
            "source_files": verified_sources,
            "checkpoints": verified_checkpoints,
        }
    return {
        "methods": result,
        "physical_checkpoint_count": sum(
            len(item["checkpoints"]) for item in result.values()
        ),
        "physical_checkpoint_bytes": total_checkpoint_bytes,
    }


def _load_sensitivity_sources(
    canonical: Mapping[str, Any],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    canonical_sources = canonical.get("sources")
    if not isinstance(canonical_sources, Mapping):
        raise ValueError("canonical trajectory source inventory is malformed")
    expected_sensitivity = canonical_sources.get("sensitivity")
    if not isinstance(expected_sensitivity, Mapping) or set(
        expected_sensitivity
    ) != set(METHOD_RUN_DIRS):
        raise ValueError("canonical sensitivity method inventory differs")
    reports: dict[str, dict[int, dict[str, Any]]] = {}
    verified: dict[str, Any] = {}
    for method in METHOD_RUN_DIRS:
        method_sources = expected_sensitivity[method]
        if not isinstance(method_sources, Mapping) or set(method_sources) != {
            str(step) for step in CHECKPOINT_STEPS
        }:
            raise ValueError(f"{method} sensitivity step inventory differs")
        reports[method] = {}
        verified[method] = {}
        for step in CHECKPOINT_STEPS:
            expected = method_sources[str(step)]
            if not isinstance(expected, Mapping) or set(expected) != {
                "manifest",
                "report",
            }:
                raise ValueError(f"{method} step {step} sensitivity identity differs")
            report_identity = _verify_bound_file(
                expected["report"],
                label=f"{method} step {step} sensitivity report",
            )
            _verify_bound_file(
                expected["manifest"],
                label=f"{method} step {step} sensitivity manifest",
            )
            report, combined_identity = builder._load_sensitivity(
                Path(report_identity["path"]),
                label=f"{method} step {step}",
            )
            if combined_identity != expected:
                raise ValueError(f"{method} step {step} sensitivity sources differ")
            reports[method][step] = report
            verified[method][str(step)] = combined_identity
    return reports, verified


def build_replay_audit(
    *,
    project: Path,
    verifier_git: Mapping[str, Any],
    subject_git: Mapping[str, Any],
    preparation_path: Path,
    authorization_path: Path,
    canonical_path: Path,
    expected_preparation_sha256: str,
    expected_authorization_sha256: str,
    expected_canonical_sha256: str,
) -> dict[str, Any]:
    verifier_path = Path(__file__).resolve()
    builder_path = Path(builder.__file__).resolve()
    if (
        verifier_path.parents[1] != project.resolve()
        or builder_path.parents[1] != project.resolve()
    ):
        raise ValueError("trajectory replay code paths differ from verifier project")
    preparation_identity = _verify_expected_sha256(
        preparation_path,
        expected_sha256=expected_preparation_sha256,
        label="trajectory preparation",
    )
    authorization_identity = _verify_expected_sha256(
        authorization_path,
        expected_sha256=expected_authorization_sha256,
        label="trajectory execution authorization",
    )
    canonical_identity = _verify_expected_sha256(
        canonical_path,
        expected_sha256=expected_canonical_sha256,
        label="canonical trajectory report",
    )
    preparation = read_json_object(preparation_path, name="trajectory preparation")
    authorization = read_json_object(
        authorization_path,
        name="trajectory execution authorization",
    )
    canonical = read_json_object(canonical_path, name="canonical trajectory report")
    if preparation.get("git") != dict(subject_git):
        raise ValueError("trajectory preparation subject Git differs")
    validate_preparation(
        preparation,
        expected_revision=str(subject_git["revision"]),
        expected_tree=str(subject_git["tree"]),
        expected_branch=str(subject_git["branch"]),
    )
    training_inventory = _physical_training_inventory(preparation)
    sensitivity_reports, sensitivity_sources = _load_sensitivity_sources(canonical)
    sources = {
        "preparation": preparation_identity,
        "execution_authorization": authorization_identity,
        "sensitivity": sensitivity_sources,
    }
    expected = builder.build_report(
        preparation=preparation,
        authorization=authorization,
        sensitivity_reports=sensitivity_reports,
        sources=sources,
        git={
            "revision": subject_git["revision"],
            "branch": subject_git["branch"],
            "tracked_dirty": False,
        },
    )
    if canonical != expected:
        raise ValueError("canonical trajectory report differs from replayed sources")
    expected_bytes = _render_json_bytes(expected)
    if canonical_path.read_bytes() != expected_bytes:
        raise ValueError("canonical trajectory report is not byte-exact on replay")
    if _verify_expected_sha256(
        canonical_path,
        expected_sha256=expected_canonical_sha256,
        label="canonical trajectory report after replay",
    ) != canonical_identity:
        raise ValueError("canonical trajectory report changed during replay")
    return to_jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "role": ROLE,
            "status": "pass",
            "scope": "matched_5k_existing_checkpoint_exposure_trajectory_replay_only",
            "verifier_git": dict(verifier_git),
            "subject_git": dict(subject_git),
            "code": {
                "verifier": file_identity(verifier_path),
                "trajectory_builder": file_identity(builder_path),
            },
            "sources": {
                "preparation": preparation_identity,
                "execution_authorization": authorization_identity,
                "canonical_trajectory_report": canonical_identity,
                "sensitivity": sensitivity_sources,
            },
            "physical_training_replay": training_inventory,
            "canonical_json_native_reconstruction": True,
            "canonical_byte_exact_replay": True,
            "canonical_replay_bytes": len(expected_bytes),
            "scientific_decision_preserved": canonical["decision"],
            "claim_boundary": CLAIM_BOUNDARY,
            "generation_advantage_proven": False,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the completed exposure trajectory across verifier revisions."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-verifier-revision", required=True)
    parser.add_argument("--expected-verifier-tree", required=True)
    parser.add_argument("--expected-verifier-branch", required=True)
    parser.add_argument("--expected-subject-revision", required=True)
    parser.add_argument("--expected-subject-tree", required=True)
    parser.add_argument("--expected-subject-branch", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--trajectory-report", type=Path, required=True)
    parser.add_argument("--expected-trajectory-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("trajectory replay must keep CUDA hidden")
    project = args.project.resolve()
    verifier_git = _code_identity(project)
    expected_verifier_git = {
        "revision": args.expected_verifier_revision,
        "tree": args.expected_verifier_tree,
        "branch": args.expected_verifier_branch,
        "tracked_dirty": False,
    }
    if verifier_git != expected_verifier_git:
        raise ValueError("trajectory replay verifier Git differs")
    subject_git = {
        "revision": args.expected_subject_revision,
        "tree": args.expected_subject_tree,
        "branch": args.expected_subject_branch,
        "tracked_dirty": False,
    }
    output = reject_symlink_chain(args.output, name="trajectory replay audit output")
    with exclusive_output_lock(output, role=ROLE):
        audit = build_replay_audit(
            project=project,
            verifier_git=verifier_git,
            subject_git=subject_git,
            preparation_path=args.preparation.resolve(),
            authorization_path=args.execution_authorization.resolve(),
            canonical_path=args.trajectory_report.resolve(),
            expected_preparation_sha256=args.expected_preparation_sha256,
            expected_authorization_sha256=args.expected_authorization_sha256,
            expected_canonical_sha256=args.expected_trajectory_report_sha256,
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "trajectory replay audit exists; pass --resume to validate it"
                )
            if read_json_object(output, name="trajectory replay audit") != audit:
                raise ValueError("completed trajectory replay audit differs")
        else:
            write_json_report(output, audit)
            output.chmod(0o444)
    print(output.resolve().as_posix())


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
