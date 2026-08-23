from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any

from cofitok.generation.quality_bridge import QUALITY_BRIDGE_STEPS
from cofitok.reporting import (
    file_sha256,
    git_provenance,
    write_json_report,
)
from scripts.build_generation_training_exposure_audit import build_report


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_training_exposure_audit_waiter"
AUDIT_FILENAME = "training_exposure_report.json"
MANIFEST_FILENAME = "snapshot_manifest.json"
AUTHORITATIVE_TERMINAL_VERIFICATION_FILENAME = (
    "quality_bridge_terminal_verification.json"
)
AUTHORITATIVE_TERMINAL_VERIFICATION_SCHEMA_VERSION = 1
AUTHORITATIVE_TERMINAL_VERIFICATION_ROLE = (
    "generation_quality_bridge_authoritative_terminal_verification"
)
QUALITY_BRIDGE_VERIFIER_RELATIVE_PATH = Path(
    "scripts/verify_generation_quality_bridge_result.py"
)
QUALITY_BRIDGE_BUILDER_RELATIVE_PATH = Path(
    "scripts/build_generation_quality_bridge_result.py"
)
QUALITY_BRIDGE_SOURCE_ARGUMENTS = {
    "preparation": "--preparation",
    "launch_receipt": "--launch-receipt",
    "cofitok_training": "--cofitok-training",
    "dense_training": "--dense-training",
    "training_pair_validation": "--training-pair-validation",
    "cofitok_training_audit": "--cofitok-training-audit",
    "dense_training_audit": "--dense-training-audit",
    "milestone_50000": "--milestone-50000",
    "milestone_100000": "--milestone-100000",
    "cofitok_sampling_preflight": "--cofitok-sampling-preflight",
    "dense_sampling_preflight": "--dense-sampling-preflight",
    "cofitok_generation": "--cofitok-generation",
    "dense_generation": "--dense-generation",
    "cofitok_checkpoint_eval": "--cofitok-checkpoint-eval",
    "dense_checkpoint_eval": "--dense-checkpoint-eval",
    "class_fidelity_qualification": "--class-fidelity-qualification",
    "cofitok_class_fidelity": "--cofitok-class-fidelity",
    "dense_class_fidelity": "--dense-class-fidelity",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _identity(path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": (reported_path or resolved).resolve().as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _git_tree(project: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_self_identity(args: argparse.Namespace) -> dict[str, Any]:
    identity = git_provenance(args.project)
    identity["tree"] = _git_tree(args.project)
    expected = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    if identity != expected:
        raise ValueError("training exposure waiter Git identity differs")
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a matched generation milestone, then freeze and audit the "
            "mutable training reports before later resume stages replace them."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--cofitok-training", type=Path, required=True)
    parser.add_argument("--dense-training", type=Path, required=True)
    parser.add_argument("--quality-bridge-preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--milestone-report", type=Path, required=True)
    parser.add_argument("--expected-milestone-step", type=int, required=True)
    parser.add_argument(
        "--milestone-source-profile",
        choices=("full", "stability_full", "quality_bridge"),
        required=True,
    )
    parser.add_argument("--quality-bridge-terminal-result", type=Path)
    parser.add_argument("--quality-bridge-execution-status", type=Path)
    parser.add_argument("--quality-bridge-verifier-project", type=Path)
    parser.add_argument("--quality-bridge-verifier-python", type=Path)
    parser.add_argument("--expected-quality-bridge-revision")
    parser.add_argument("--expected-quality-bridge-tree")
    parser.add_argument("--expected-quality-bridge-branch")
    parser.add_argument("--expected-quality-bridge-verifier-sha256")
    parser.add_argument("--expected-quality-bridge-builder-sha256")
    parser.add_argument("--expected-quality-bridge-result-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.expected_milestone_step < 1:
        parser.error("--expected-milestone-step must be positive")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if len(args.expected_preparation_sha256) != 64:
        parser.error("--expected-preparation-sha256 must be a SHA256 digest")
    if (args.quality_bridge_terminal_result is None) != (
        args.quality_bridge_execution_status is None
    ):
        parser.error(
            "--quality-bridge-terminal-result and "
            "--quality-bridge-execution-status must be paired"
        )
    if (
        args.quality_bridge_terminal_result is not None
        and args.expected_milestone_step != QUALITY_BRIDGE_STEPS
    ):
        parser.error("quality bridge terminal binding requires the exact 100K step")
    quality_verifier_fields = (
        "quality_bridge_verifier_project",
        "quality_bridge_verifier_python",
        "expected_quality_bridge_revision",
        "expected_quality_bridge_tree",
        "expected_quality_bridge_branch",
        "expected_quality_bridge_verifier_sha256",
        "expected_quality_bridge_builder_sha256",
        "expected_quality_bridge_result_sha256",
    )
    if _terminal_binding_requested(args):
        missing = [name for name in quality_verifier_fields if getattr(args, name) is None]
        if missing:
            parser.error(
                "quality bridge terminal binding requires authoritative verifier "
                f"arguments: {', '.join(missing)}"
            )
        for name in (
            "expected_quality_bridge_verifier_sha256",
            "expected_quality_bridge_builder_sha256",
            "expected_quality_bridge_result_sha256",
        ):
            if len(getattr(args, name)) != 64:
                parser.error(f"--{name.replace('_', '-')} must be a SHA256 digest")
    elif any(getattr(args, name) is not None for name in quality_verifier_fields):
        parser.error(
            "authoritative quality bridge verifier arguments require terminal binding"
        )
    return args


def _terminal_binding_requested(args: argparse.Namespace) -> bool:
    return getattr(args, "quality_bridge_terminal_result", None) is not None


def _authoritative_quality_project_identity(
    args: argparse.Namespace,
) -> dict[str, Any]:
    project = args.quality_bridge_verifier_project.resolve()
    identity = git_provenance(project)
    identity["tree"] = _git_tree(project)
    identity["path"] = project.as_posix()
    expected = {
        "revision": args.expected_quality_bridge_revision,
        "branch": args.expected_quality_bridge_branch,
        "tracked_dirty": False,
        "tree": args.expected_quality_bridge_tree,
        "path": project.as_posix(),
    }
    if identity != expected:
        raise ValueError("authoritative quality bridge project identity differs")
    return identity


def _authoritative_quality_sources(args: argparse.Namespace) -> dict[str, Any]:
    project = args.quality_bridge_verifier_project.resolve()
    verifier = project / QUALITY_BRIDGE_VERIFIER_RELATIVE_PATH
    builder = project / QUALITY_BRIDGE_BUILDER_RELATIVE_PATH
    python = args.quality_bridge_verifier_python.resolve()
    for path, label in (
        (verifier, "authoritative quality bridge verifier"),
        (builder, "authoritative quality bridge builder"),
        (python, "authoritative quality bridge Python"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    verifier_identity = _identity(verifier)
    builder_identity = _identity(builder)
    if (
        verifier_identity["sha256"]
        != args.expected_quality_bridge_verifier_sha256
    ):
        raise ValueError("authoritative quality bridge verifier SHA256 differs")
    if builder_identity["sha256"] != args.expected_quality_bridge_builder_sha256:
        raise ValueError("authoritative quality bridge builder SHA256 differs")
    return {
        "quality_project": _authoritative_quality_project_identity(args),
        "verifier_source": verifier_identity,
        "builder_source": builder_identity,
        "python": _identity(python),
    }


def _run_authoritative_quality_bridge_verifier(
    args: argparse.Namespace,
    *,
    reported_terminal_result: Path,
) -> dict[str, Any]:
    binding = _authoritative_quality_sources(args)
    terminal_path = args.quality_bridge_terminal_result.resolve()
    terminal_identity = _identity(
        terminal_path,
        reported_path=reported_terminal_result,
    )
    if terminal_identity["sha256"] != args.expected_quality_bridge_result_sha256:
        raise ValueError("authoritative quality bridge result SHA256 differs")
    terminal = _read_json(terminal_path, label="quality bridge terminal result")
    terminal_git = terminal.get("git")
    if terminal_git != {
        "revision": args.expected_quality_bridge_revision,
        "branch": args.expected_quality_bridge_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("quality bridge terminal result Git identity differs")
    sources = terminal.get("source_reports")
    if not isinstance(sources, dict) or set(sources) != set(
        QUALITY_BRIDGE_SOURCE_ARGUMENTS
    ):
        raise ValueError("quality bridge terminal result source set differs")

    command = [
        str(args.quality_bridge_verifier_python.resolve()),
        str(
            args.quality_bridge_verifier_project.resolve()
            / QUALITY_BRIDGE_VERIFIER_RELATIVE_PATH
        ),
    ]
    for name, flag in QUALITY_BRIDGE_SOURCE_ARGUMENTS.items():
        source = sources[name]
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise ValueError(f"quality bridge terminal source is malformed: {name}")
        command.extend([flag, source["path"]])
    command.extend(
        [
            "--expected-preparation-sha256",
            sources["preparation"]["sha256"],
            "--expected-launch-receipt-sha256",
            sources["launch_receipt"]["sha256"],
            "--expected-revision",
            args.expected_quality_bridge_revision,
            "--expected-branch",
            args.expected_quality_bridge_branch,
            "--result",
            terminal_path.as_posix(),
            "--expected-result-sha256",
            args.expected_quality_bridge_result_sha256,
        ]
    )
    environment = os.environ.copy()
    project = args.quality_bridge_verifier_project.resolve()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join((str(project / "src"), str(project))),
        }
    )
    completed = subprocess.run(
        command,
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(
            "authoritative quality bridge verifier failed: "
            f"exit={completed.returncode}: {detail}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("authoritative quality bridge verifier produced no output")
    try:
        verifier_output = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError(
            "authoritative quality bridge verifier output is not JSON"
        ) from error
    actual_terminal_identity = _identity(terminal_path)
    if (
        not isinstance(verifier_output, dict)
        or verifier_output.get("status") != "verified"
        or verifier_output.get("result") != actual_terminal_identity
        or verifier_output.get("quality_screen") != terminal.get("quality_screen")
        or verifier_output.get("authorization_boundary")
        != terminal.get("authorization_boundary")
    ):
        raise ValueError("authoritative quality bridge verifier output differs")
    verifier_output = dict(verifier_output)
    verifier_output["result"] = terminal_identity
    return {
        "schema_version": AUTHORITATIVE_TERMINAL_VERIFICATION_SCHEMA_VERSION,
        "role": AUTHORITATIVE_TERMINAL_VERIFICATION_ROLE,
        "status": "verified",
        **binding,
        "terminal_result": terminal_identity,
        "verifier_output": verifier_output,
        "execution_policy": {
            "cuda_visible_devices": "-1",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
        },
    }


def _status(
    args: argparse.Namespace,
    *,
    self_git: dict[str, Any],
    status: str,
    detail: str,
    polls: int,
    audit_identity: dict[str, Any] | None = None,
    manifest_identity: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    sources = {
        "cofitok_training": args.cofitok_training.resolve().as_posix(),
        "dense_training": args.dense_training.resolve().as_posix(),
        "quality_bridge_preparation": (
            args.quality_bridge_preparation.resolve().as_posix()
        ),
        "milestone_report": args.milestone_report.resolve().as_posix(),
    }
    if _terminal_binding_requested(args):
        sources.update(
            {
                "quality_bridge_terminal_result": (
                    args.quality_bridge_terminal_result.resolve().as_posix()
                ),
                "quality_bridge_execution_status": (
                    args.quality_bridge_execution_status.resolve().as_posix()
                ),
                "quality_bridge_verifier_project": (
                    args.quality_bridge_verifier_project.resolve().as_posix()
                ),
            }
        )
    write_json_report(
        args.status,
        {
            "schema_version": WAITER_SCHEMA_VERSION,
            "role": WAITER_ROLE,
            "status": status,
            "detail": detail,
            "error": error,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "polls": polls,
            "poll_seconds": args.poll_seconds,
            "updated_at": _utc_now(),
            "self_git": self_git,
            "expected_milestone_step": args.expected_milestone_step,
            "milestone_source_profile": args.milestone_source_profile,
            "sources": sources,
            "output_root": args.output_root.resolve().as_posix(),
            "audit_report": audit_identity,
            "snapshot_manifest": manifest_identity,
            "binding_mode": (
                "quality_bridge_terminal"
                if _terminal_binding_requested(args)
                else "matched_milestone"
            ),
            "authorization_boundary": {
                "read_only_source_observation": True,
                "gpu_use_allowed": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "evaluation_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "release_allowed": False,
            },
        },
    )


def _training_ready(path: Path, *, expected_step: int, label: str) -> bool:
    if not path.is_file():
        return False
    report = _read_json(path, label=label)
    completed_steps = int(report.get("completed_steps", -1))
    if completed_steps < expected_step:
        return False
    if completed_steps > expected_step:
        raise ValueError(f"{label} advanced past the unsnapshotted milestone")
    return True


def _snapshot_paths(
    output_root: Path,
    *,
    include_terminal_binding: bool = False,
) -> dict[str, Path]:
    source = output_root / "source"
    paths = {
        "cofitok": source / "cofitok_training_report.json",
        "dense_identity": source / "dense_training_report.json",
        "quality_bridge_preparation": source / "quality_bridge_preparation.json",
        "milestone_report": source / "milestone_report.json",
    }
    if include_terminal_binding:
        paths.update(
            {
                "quality_bridge_terminal_result": (
                    source / "quality_bridge_terminal_result.json"
                ),
                "quality_bridge_execution_status": (
                    source / "quality_bridge_execution_status.json"
                ),
                "quality_bridge_terminal_verification": (
                    source / AUTHORITATIVE_TERMINAL_VERIFICATION_FILENAME
                ),
            }
        )
    return paths


def _build_from_snapshots(
    args: argparse.Namespace,
    *,
    snapshot_root: Path,
    reported_root: Path,
) -> dict[str, Any]:
    include_terminal = _terminal_binding_requested(args)
    observed = _snapshot_paths(
        snapshot_root,
        include_terminal_binding=include_terminal,
    )
    reported = _snapshot_paths(
        reported_root,
        include_terminal_binding=include_terminal,
    )
    training_reports = {
        "cofitok": (
            _read_json(observed["cofitok"], label="snapshotted CoFiTok training"),
            _identity(observed["cofitok"], reported_path=reported["cofitok"]),
        ),
        "dense_identity": (
            _read_json(
                observed["dense_identity"],
                label="snapshotted dense training",
            ),
            _identity(
                observed["dense_identity"],
                reported_path=reported["dense_identity"],
            ),
        ),
    }
    preparation = (
        _read_json(
            observed["quality_bridge_preparation"],
            label="snapshotted quality bridge preparation",
        ),
        _identity(
            observed["quality_bridge_preparation"],
            reported_path=reported["quality_bridge_preparation"],
        ),
    )
    milestone = (
        _read_json(
            observed["milestone_report"],
            label="snapshotted milestone report",
        ),
        _identity(
            observed["milestone_report"],
            reported_path=reported["milestone_report"],
        ),
    )
    terminal_result = None
    execution_status = None
    terminal_verification = None
    if include_terminal:
        terminal_result = (
            _read_json(
                observed["quality_bridge_terminal_result"],
                label="snapshotted quality bridge terminal result",
            ),
            _identity(
                observed["quality_bridge_terminal_result"],
                reported_path=reported["quality_bridge_terminal_result"],
            ),
        )
        execution_status = (
            _read_json(
                observed["quality_bridge_execution_status"],
                label="snapshotted quality bridge execution status",
            ),
            _identity(
                observed["quality_bridge_execution_status"],
                reported_path=reported["quality_bridge_execution_status"],
            ),
        )
        terminal_verification = _read_json(
            observed["quality_bridge_terminal_verification"],
            label="authoritative quality bridge terminal verification",
        )
    build_kwargs = {
        "quality_bridge_preparation": preparation,
        "milestone_report": milestone,
        "expected_milestone_step": args.expected_milestone_step,
        "milestone_source_profile": args.milestone_source_profile,
    }
    if include_terminal:
        build_kwargs.update(
            {
                "quality_bridge_terminal_result": terminal_result,
                "quality_bridge_execution_status": execution_status,
                "quality_bridge_terminal_verification": terminal_verification,
            }
        )
    return build_report(training_reports, **build_kwargs)


def _verify_existing(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_path = args.output_root / AUDIT_FILENAME
    manifest_path = args.output_root / MANIFEST_FILENAME
    if not audit_path.is_file() or not manifest_path.is_file():
        raise ValueError("training exposure snapshot output is partial")
    manifest = _read_json(manifest_path, label="training exposure snapshot manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("role") != "generation_training_exposure_snapshot_manifest"
        or manifest.get("status") != "completed"
        or int(manifest.get("expected_milestone_step", -1))
        != args.expected_milestone_step
        or manifest.get("milestone_source_profile") != args.milestone_source_profile
        or manifest.get("binding_mode", "matched_milestone")
        != (
            "quality_bridge_terminal"
            if _terminal_binding_requested(args)
            else "matched_milestone"
        )
    ):
        raise ValueError("training exposure snapshot manifest contract differs")
    expected_self_git = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    if manifest.get("self_git") != expected_self_git:
        raise ValueError("training exposure snapshot code identity differs")
    if manifest.get("claim_boundary") != {
        "source_snapshot_only": True,
        "formal_generation_claim_allowed": False,
        "training_launch_allowed": False,
        "gpu_use_allowed": False,
    }:
        raise ValueError("training exposure snapshot claim boundary differs")
    paths = _snapshot_paths(
        args.output_root,
        include_terminal_binding=_terminal_binding_requested(args),
    )
    actual_sources = {name: _identity(path) for name, path in paths.items()}
    if manifest.get("sources") != actual_sources:
        raise ValueError("training exposure snapshot source identity differs")
    if _terminal_binding_requested(args):
        live_binding = _authoritative_quality_sources(args)
        verification = _read_json(
            paths["quality_bridge_terminal_verification"],
            label="authoritative quality bridge terminal verification",
        )
        for name in (
            "quality_project",
            "verifier_source",
            "builder_source",
            "python",
        ):
            if verification.get(name) != live_binding[name]:
                raise ValueError(
                    "authoritative quality bridge verifier binding differs"
                )
    expected = _build_from_snapshots(
        args,
        snapshot_root=args.output_root,
        reported_root=args.output_root,
    )
    actual = _read_json(audit_path, label="training exposure audit report")
    if actual != expected:
        raise ValueError("training exposure audit report is not reproducible")
    audit_identity = _identity(audit_path)
    if manifest.get("audit_report") != audit_identity:
        raise ValueError("training exposure audit report identity differs")
    return audit_identity, _identity(manifest_path)


def _create_snapshot(args: argparse.Namespace, self_git: dict[str, Any]) -> None:
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=args.output_root.parent,
            prefix=f".{args.output_root.name}.tmp-",
        )
    )
    try:
        include_terminal = _terminal_binding_requested(args)
        temporary_paths = _snapshot_paths(
            temporary,
            include_terminal_binding=include_terminal,
        )
        source_paths = {
            "cofitok": args.cofitok_training,
            "dense_identity": args.dense_training,
            "quality_bridge_preparation": args.quality_bridge_preparation,
            "milestone_report": args.milestone_report,
        }
        if include_terminal:
            source_paths.update(
                {
                    "quality_bridge_terminal_result": (
                        args.quality_bridge_terminal_result
                    ),
                    "quality_bridge_execution_status": (
                        args.quality_bridge_execution_status
                    ),
                }
            )
            verification = _run_authoritative_quality_bridge_verifier(
                args,
                reported_terminal_result=_snapshot_paths(
                    args.output_root,
                    include_terminal_binding=True,
                )["quality_bridge_terminal_result"],
            )
            write_json_report(
                temporary_paths["quality_bridge_terminal_verification"],
                verification,
            )
        for name, source in source_paths.items():
            target = temporary_paths[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _identity(source)["sha256"] != _identity(target)["sha256"]:
                raise ValueError(f"training exposure snapshot copy differs: {name}")
        report = _build_from_snapshots(
            args,
            snapshot_root=temporary,
            reported_root=args.output_root,
        )
        audit_path = temporary / AUDIT_FILENAME
        write_json_report(audit_path, report)
        final_audit_path = args.output_root / AUDIT_FILENAME
        source_identities = {
            name: _identity(
                path,
                reported_path=_snapshot_paths(
                    args.output_root,
                    include_terminal_binding=include_terminal,
                )[name],
            )
            for name, path in temporary_paths.items()
        }
        write_json_report(
            temporary / MANIFEST_FILENAME,
            {
                "schema_version": 1,
                "role": "generation_training_exposure_snapshot_manifest",
                "status": "completed",
                "created_at": _utc_now(),
                "self_git": self_git,
                "expected_milestone_step": args.expected_milestone_step,
                "milestone_source_profile": args.milestone_source_profile,
                "binding_mode": (
                    "quality_bridge_terminal"
                    if include_terminal
                    else "matched_milestone"
                ),
                "sources": source_identities,
                "audit_report": _identity(
                    audit_path,
                    reported_path=final_audit_path,
                ),
                "claim_boundary": {
                    "source_snapshot_only": True,
                    "formal_generation_claim_allowed": False,
                    "training_launch_allowed": False,
                    "gpu_use_allowed": False,
                },
            },
        )
        if args.output_root.exists():
            raise ValueError("training exposure snapshot output appeared concurrently")
        os.replace(temporary, args.output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def poll_once(
    args: argparse.Namespace,
    *,
    self_git: dict[str, Any],
) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
    if args.output_root.exists():
        audit, manifest = _verify_existing(args)
        return (
            "completed",
            (
                "matched_terminal_training_exposure_frozen"
                if _terminal_binding_requested(args)
                else "matched_milestone_training_exposure_frozen"
            ),
            audit,
            manifest,
        )
    if not args.quality_bridge_preparation.is_file():
        return "waiting", "waiting_for_quality_bridge_preparation", None, None
    preparation_sha256 = file_sha256(args.quality_bridge_preparation)
    if preparation_sha256 != args.expected_preparation_sha256:
        raise ValueError("quality bridge preparation SHA256 differs")
    if not _training_ready(
        args.cofitok_training,
        expected_step=args.expected_milestone_step,
        label="CoFiTok training report",
    ):
        return "waiting", "waiting_for_cofitok_training_milestone", None, None
    if not _training_ready(
        args.dense_training,
        expected_step=args.expected_milestone_step,
        label="dense training report",
    ):
        return "waiting", "waiting_for_dense_training_milestone", None, None
    if not args.milestone_report.is_file():
        return "waiting", "waiting_for_matched_milestone_report", None, None
    if _terminal_binding_requested(args):
        if not args.quality_bridge_terminal_result.is_file():
            return "waiting", "waiting_for_quality_bridge_terminal_result", None, None
        if not args.quality_bridge_execution_status.is_file():
            return "waiting", "waiting_for_quality_bridge_execution_status", None, None
        execution = _read_json(
            args.quality_bridge_execution_status,
            label="quality bridge execution status",
        )
        execution_state = execution.get("status")
        if execution_state == "failed":
            raise ValueError("quality bridge execution failed before terminal binding")
        if execution_state != "completed":
            return (
                "waiting",
                "waiting_for_quality_bridge_verified_completion",
                None,
                None,
            )
    _create_snapshot(args, self_git)
    audit, manifest = _verify_existing(args)
    return (
        "completed",
        (
            "matched_terminal_training_exposure_frozen"
            if _terminal_binding_requested(args)
            else "matched_milestone_training_exposure_frozen"
        ),
        audit,
        manifest,
    )


def main() -> None:
    args = parse_args()
    self_git = verify_self_identity(args)
    polls = 0
    try:
        while True:
            polls += 1
            status, detail, audit, manifest = poll_once(args, self_git=self_git)
            _status(
                args,
                self_git=self_git,
                status=status,
                detail=detail,
                polls=polls,
                audit_identity=audit,
                manifest_identity=manifest,
            )
            if status == "completed":
                return
            time.sleep(args.poll_seconds)
    except Exception as error:
        _status(
            args,
            self_git=self_git,
            status="failed",
            detail=f"{type(error).__name__}:{error}",
            polls=polls,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    main()
