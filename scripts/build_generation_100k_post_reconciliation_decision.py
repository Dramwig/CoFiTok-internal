from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.post_reconciliation_decision import (
    POST_RECONCILIATION_DECISION_ROLE,
    TRAINING_BRANCH,
    TRAINING_REVISION,
    build_post_reconciliation_decision,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_git_identity(project: Path = PROJECT_ROOT) -> dict[str, Any]:
    identity = git_provenance(project)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        **identity,
        "tree": tree,
        "path": project.resolve().as_posix(),
    }


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(path, name=label), identity


def _same_content(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> None:
    if {
        "bytes": left.get("bytes"),
        "sha256": left.get("sha256"),
    } != {
        "bytes": right.get("bytes"),
        "sha256": right.get("sha256"),
    }:
        raise ValueError(f"{label} content identity differs")


def _terminal_method_evidence(
    quality: dict[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    source_reports = quality.get("source_reports")
    terminal = quality.get("terminal", {}).get("methods", {}).get(method)
    if not isinstance(source_reports, dict) or not isinstance(terminal, dict):
        raise ValueError(f"quality result has no terminal {method} evidence")
    generation_key = (
        "cofitok_generation" if method == "cofitok" else "dense_generation"
    )
    class_key = (
        "cofitok_class_fidelity"
        if method == "cofitok"
        else "dense_class_fidelity"
    )
    generation_identity = source_reports.get(generation_key)
    class_identity = source_reports.get(class_key)
    if not isinstance(generation_identity, dict) or not isinstance(
        class_identity, dict
    ):
        raise ValueError(f"quality result terminal {method} source binding is missing")
    generation_path = Path(str(generation_identity.get("path", "")))
    class_path = Path(str(class_identity.get("path", "")))
    actual_generation_identity = file_identity(generation_path)
    actual_class_identity = file_identity(class_path)
    if actual_generation_identity != generation_identity:
        raise ValueError(f"terminal {method} generation report changed")
    if actual_class_identity != class_identity:
        raise ValueError(f"terminal {method} class-fidelity report changed")
    generation = read_json_object(
        generation_path,
        name=f"terminal {method} generation report",
    )
    class_fidelity = read_json_object(
        class_path,
        name=f"terminal {method} class-fidelity report",
    )
    if (
        generation.get("status") != "completed"
        or class_fidelity.get("status") != "completed"
    ):
        raise ValueError(f"terminal {method} reports are incomplete")
    generation_provenance = generation.get("sample_provenance")
    class_provenance = class_fidelity.get("sample_provenance")
    if not isinstance(generation_provenance, dict) or not isinstance(
        class_provenance, dict
    ):
        raise ValueError(f"terminal {method} sample provenance is missing")
    for key in (
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "checkpoint_integrity_manifest",
        "sample_set_sha256",
        "selected_prefix_budget",
        "sampling",
        "weights",
    ):
        if generation_provenance.get(key) != class_provenance.get(key):
            raise ValueError(f"terminal {method} report provenance differs: {key}")

    checkpoint = Path(str(generation_provenance["checkpoint"]))
    sidecar = Path(str(generation_provenance["checkpoint_integrity_manifest"]))
    latest = checkpoint.parent / "latest.json"
    checkpoint_identity = file_identity(checkpoint)
    sidecar_identity = file_identity(sidecar)
    latest_identity = file_identity(latest)
    sidecar_payload = read_json_object(
        sidecar,
        name=f"terminal {method} checkpoint sidecar",
    )
    latest_payload = read_json_object(latest, name=f"terminal {method} latest")
    expected_checkpoint = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint_identity["bytes"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_format_version": 1,
        "step": 100_000,
        "git_revision": TRAINING_REVISION,
        "git_branch": TRAINING_BRANCH,
        "git_dirty": False,
    }
    for key, value in expected_checkpoint.items():
        if sidecar_payload.get(key) != value:
            raise ValueError(f"terminal {method} sidecar differs: {key}")
        if latest_payload.get(key) != value:
            raise ValueError(f"terminal {method} latest differs: {key}")
    if latest_payload.get("integrity_manifest") != sidecar.name:
        raise ValueError(f"terminal {method} latest sidecar binding differs")
    if (
        generation_provenance.get("checkpoint_sha256")
        != checkpoint_identity["sha256"]
        or int(generation_provenance.get("checkpoint_step", -1)) != 100_000
        or terminal.get("checkpoint_sha256") != checkpoint_identity["sha256"]
        or terminal.get("sample_set_sha256")
        != generation_provenance.get("sample_set_sha256")
    ):
        raise ValueError(f"terminal {method} checkpoint/sample binding differs")

    metrics = generation.get("metrics")
    class_metrics = class_fidelity.get("metrics")
    sampling = generation_provenance.get("sampling")
    if not isinstance(metrics, dict) or not isinstance(class_metrics, dict) or not isinstance(
        sampling, dict
    ):
        raise ValueError(f"terminal {method} metric payload is malformed")
    return {
        "generation_report": actual_generation_identity,
        "class_fidelity_report": actual_class_identity,
        "checkpoint_payload": checkpoint_identity,
        "checkpoint_sidecar": sidecar_identity,
        "latest": latest_identity,
        "physical_sha256_verified": True,
        "sidecar_latest_reconciled": True,
        "checkpoint_step": 100_000,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "sample_set_sha256": generation_provenance["sample_set_sha256"],
        "prefix_budget": int(generation_provenance["selected_prefix_budget"]),
        "generation_metrics": {
            "fid": metrics.get("frechet_inception_distance"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "inception_score_mean": metrics.get("inception_score_mean"),
        },
        "class_fidelity_metrics": class_metrics,
        "sampling": sampling,
    }


def build_from_paths(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise ValueError("post-reconciliation decision replay requires CUDA hidden")
    builder_git = project_git_identity()
    expected_git = {
        "revision": args.expected_builder_revision,
        "tree": args.expected_builder_tree,
        "branch": args.expected_builder_branch,
        "tracked_dirty": False,
        "path": PROJECT_ROOT.resolve().as_posix(),
    }
    if builder_git != expected_git:
        raise ValueError("post-reconciliation builder Git identity differs")
    authoritative_decision, authoritative_decision_identity = _bound_json(
        args.authoritative_decision,
        expected_sha256=args.expected_authoritative_decision_sha256,
        label="authoritative pre-reconciliation decision",
    )
    reconciliation, reconciliation_identity = _bound_json(
        args.reconciliation,
        expected_sha256=args.expected_reconciliation_sha256,
        label="cross-protocol reconciliation",
    )
    quality, quality_identity = _bound_json(
        args.quality_result,
        expected_sha256=args.expected_quality_result_sha256,
        label="quality bridge result",
    )
    exposure, exposure_identity = _bound_json(
        args.training_exposure,
        expected_sha256=args.expected_training_exposure_sha256,
        label="terminal training exposure",
    )
    terminal_evidence = {
        method: _terminal_method_evidence(quality, method=method)
        for method in ("cofitok", "dense_identity")
    }
    report = build_post_reconciliation_decision(
        authoritative_decision=authoritative_decision,
        authoritative_decision_identity=authoritative_decision_identity,
        reconciliation=reconciliation,
        reconciliation_identity=reconciliation_identity,
        quality_result=quality,
        quality_result_identity=quality_identity,
        training_exposure=exposure,
        training_exposure_identity=exposure_identity,
        terminal_evidence=terminal_evidence,
        builder_git=builder_git,
    )
    for label, path in (
        ("authoritative decision", args.authoritative_decision),
        ("reconciliation", args.reconciliation),
        ("quality result", args.quality_result),
        ("training exposure", args.training_exposure),
    ):
        expected = report["source_evidence"][
            {
                "authoritative decision": "authoritative_pre_reconciliation_decision",
                "reconciliation": "cross_protocol_reconciliation",
                "quality result": "quality_bridge_result",
                "training exposure": "terminal_training_exposure",
            }[label]
        ]
        _same_content(file_identity(path), expected, label=label)
    return report


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--authoritative-decision", type=Path, required=True)
    parser.add_argument(
        "--expected-authoritative-decision-sha256",
        required=True,
    )
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--expected-reconciliation-sha256", required=True)
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--expected-quality-result-sha256", required=True)
    parser.add_argument("--training-exposure", type=Path, required=True)
    parser.add_argument("--expected-training-exposure-sha256", required=True)
    parser.add_argument("--expected-builder-revision", required=True)
    parser.add_argument("--expected-builder-tree", required=True)
    parser.add_argument("--expected-builder-branch", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a permanently non-authorizing post-reconciliation decision "
            "for the matched 100K quality bridge."
        )
    )
    add_source_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(args.output, role=POST_RECONCILIATION_DECISION_ROLE):
        report = build_from_paths(args)
        if args.output.is_file():
            if not args.resume:
                raise FileExistsError(
                    "post-reconciliation decision exists; pass --resume to verify it"
                )
            existing = read_json_object(
                args.output,
                name="post-reconciliation decision",
            )
            if existing != report:
                raise ValueError(
                    "existing post-reconciliation decision differs from exact replay"
                )
            print(f"reused {args.output}")
            return
        write_json_report(args.output, report)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
