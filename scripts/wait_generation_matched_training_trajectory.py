from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows test import path
    fcntl = None

from cofitok.reporting import file_sha256, write_json_report


WAITER_ROLE = "generation_matched_training_trajectory_waiter"
RECEIPT_ROLE = "generation_matched_training_trajectory_waiter_receipt"
TRAJECTORY_ROLE = "matched_training_trajectory_diagnostic"
SCHEDULE_ROLE = "generation_consistency_schedule_transition_audit"
SCOPE = {
    "read_only_training_sources": True,
    "diagnostic_artifact_writes_only": True,
    "gpu_required": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON source is not an object: {path}")
    return payload


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _git_value(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    expected_branch: str,
    expected_tree: str,
) -> dict[str, Any]:
    resolved = checkout.resolve()
    if not resolved.is_dir():
        raise ValueError(f"checkout does not exist: {resolved}")
    observed = {
        "path": resolved.as_posix(),
        "revision": _git_value(resolved, "rev-parse", "HEAD"),
        "branch": _git_value(resolved, "branch", "--show-current"),
        "tree": _git_value(resolved, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(
            _git_value(resolved, "status", "--porcelain", "--untracked-files=no")
        ),
    }
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tree": expected_tree,
        "tracked_dirty": False,
    }
    for field, value in expected.items():
        if observed[field] != value:
            raise ValueError(f"checkout {field} differs")
    return observed


def verify_control_identity(args: argparse.Namespace) -> dict[str, Any]:
    checkout = verify_checkout(
        args.control_project,
        expected_revision=args.expected_control_revision,
        expected_branch=args.expected_control_branch,
        expected_tree=args.expected_control_tree,
    )
    source = Path(__file__).resolve()
    expected_source = (
        args.control_project
        / "scripts/wait_generation_matched_training_trajectory.py"
    ).resolve()
    if source != expected_source:
        raise ValueError("control source is outside the bound checkout")
    if file_sha256(source) != args.expected_control_source_sha256:
        raise ValueError("control source SHA256 differs")
    return {"checkout": checkout, "source": file_identity(source)}


def verify_builder_identity(args: argparse.Namespace) -> dict[str, Any]:
    checkout = verify_checkout(
        args.builder_project,
        expected_revision=args.expected_builder_revision,
        expected_branch=args.expected_builder_branch,
        expected_tree=args.expected_builder_tree,
    )
    source = (
        args.builder_project / "scripts/build_generation_matched_training_trajectory.py"
    ).resolve()
    if not source.is_file():
        raise ValueError("matched trajectory builder is missing")
    if file_sha256(source) != args.expected_builder_source_sha256:
        raise ValueError("matched trajectory builder SHA256 differs")
    return {"checkout": checkout, "source": file_identity(source)}


def _scope_is_non_authorizing(scope: Any) -> bool:
    if not isinstance(scope, dict):
        return False
    return (
        scope.get("read_only_metrics_verification") is True
        and scope.get("gpu_required") is False
        and scope.get("training_process_signals_allowed") is False
        and scope.get("unrelated_process_signals_allowed") is False
        and scope.get("promotion_authorization_allowed") is False
        and scope.get("release_authorization_allowed") is False
    )


def validate_schedule_report(
    path: Path,
    *,
    dense_run_dir: Path,
    training_checkout: Path,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_training_tree: str,
    expected_waiter_source_sha256: str,
    cutoff_step: int,
    expected_effective_batch: int,
    expected_training_steps: int,
    expected_start_step: int,
    expected_warmup_steps: int,
    expected_weight: float,
) -> dict[str, Any]:
    report = read_object(path)
    if report.get("schema_version") != 1:
        raise ValueError("schedule report schema differs")
    if report.get("role") != SCHEDULE_ROLE or report.get("status") != "pass":
        raise ValueError("schedule report is not a passing transition audit")
    if not _scope_is_non_authorizing(report.get("scope")):
        raise ValueError("schedule report scope is not read-only and non-authorizing")
    if not _same_path(report.get("run_dir", ""), dense_run_dir):
        raise ValueError("schedule report run directory differs")

    checkout = report.get("training_checkout")
    if not isinstance(checkout, dict):
        raise ValueError("schedule report lacks training checkout identity")
    expected_checkout = {
        "path": training_checkout.resolve().as_posix(),
        "revision": expected_training_revision,
        "branch": expected_training_branch,
        "tree": expected_training_tree,
        "tracked_dirty": False,
    }
    for field, expected in expected_checkout.items():
        observed = checkout.get(field)
        if field == "path":
            if not _same_path(str(observed), expected):
                raise ValueError("schedule report training checkout path differs")
        elif observed != expected:
            raise ValueError(f"schedule report training checkout {field} differs")

    waiter_source = report.get("waiter_source")
    if (
        not isinstance(waiter_source, dict)
        or waiter_source.get("sha256") != expected_waiter_source_sha256
    ):
        raise ValueError("schedule waiter source SHA256 differs")

    run_manifest = report.get("run_manifest")
    if not isinstance(run_manifest, dict):
        raise ValueError("schedule report lacks run manifest evidence")
    manifest_git = run_manifest.get("git")
    if not isinstance(manifest_git, dict):
        raise ValueError("schedule report lacks run manifest Git identity")
    expected_git = {
        "revision": expected_training_revision,
        "branch": expected_training_branch,
        "dirty": False,
    }
    for field, expected in expected_git.items():
        if manifest_git.get(field) != expected:
            raise ValueError(f"schedule run manifest Git {field} differs")
    manifest_source = run_manifest.get("source")
    if not isinstance(manifest_source, dict) or not _same_path(
        manifest_source.get("path", ""), dense_run_dir / "run_manifest.json"
    ):
        raise ValueError("schedule run manifest source differs")

    schedule = run_manifest.get("schedule")
    expected_schedule = {
        "start_step": expected_start_step,
        "warmup_steps": expected_warmup_steps,
        "weight": expected_weight,
    }
    if not isinstance(schedule, dict):
        raise ValueError("schedule report lacks the run schedule")
    for field, expected in expected_schedule.items():
        observed = schedule.get(field)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"schedule {field} differs")
        elif observed != expected:
            raise ValueError(f"schedule {field} differs")
    if run_manifest.get("effective_batch") != expected_effective_batch:
        raise ValueError("schedule effective batch differs")
    if run_manifest.get("target_steps") != expected_training_steps:
        raise ValueError("schedule training target differs")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("schedule report lacks metrics evidence")
    metrics_source = metrics.get("source")
    if not isinstance(metrics_source, dict) or not _same_path(
        metrics_source.get("path", ""), dense_run_dir / "train_metrics.jsonl"
    ):
        raise ValueError("schedule metrics source differs")
    if int(metrics.get("last_step_at_read", -1)) < cutoff_step:
        raise ValueError("schedule metrics had not reached the cutoff")

    transition = report.get("transition")
    if not isinstance(transition, dict) or transition.get("status") != "verified":
        raise ValueError("schedule transition was not verified")
    expected_transition = {
        "start_step": expected_start_step,
        "warmup_steps": expected_warmup_steps,
        "target_step": cutoff_step,
        "samples_seen_binding_verified": True,
        "strictly_increasing_metrics": True,
    }
    for field, expected in expected_transition.items():
        if transition.get(field) != expected:
            raise ValueError(f"schedule transition {field} differs")
    if not math.isclose(
        float(transition.get("expected_target_scale", -1.0)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("schedule target scale is not full strength")
    target_row = transition.get("target_row")
    if not isinstance(target_row, dict):
        raise ValueError("schedule transition lacks the target row")
    if int(target_row.get("step", -1)) != cutoff_step:
        raise ValueError("schedule target row step differs")
    if int(target_row.get("samples_seen", -1)) != cutoff_step * expected_effective_batch:
        raise ValueError("schedule target row samples_seen differs")
    if not math.isclose(
        float(target_row.get("ema_teacher_consistency_scale", -1.0)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("schedule target row is not at full strength")
    if int(target_row.get("validation_event_index", -1)) != cutoff_step // 1000 - 1:
        raise ValueError("schedule target validation event differs")

    return {"report": report, "identity": file_identity(path)}


def schedule_is_ready(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "dense_schedule_report_missing"
    try:
        payload = read_object(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False, "dense_schedule_report_unreadable"
    if payload.get("status") != "pass":
        return False, "dense_schedule_report_not_pass"
    return True, "ready"


def _invoke_builder(
    args: argparse.Namespace,
    *,
    cofitok_metrics: Path,
    dense_metrics: Path,
    output: Path,
    snapshot_dir: Path | None,
) -> subprocess.CompletedProcess[str]:
    builder = (
        args.builder_project / "scripts/build_generation_matched_training_trajectory.py"
    ).resolve()
    command = [
        str(args.python),
        str(builder),
        "--cofitok-metrics",
        str(cofitok_metrics),
        "--dense-metrics",
        str(dense_metrics),
        "--cofitok-manifest",
        str(args.cofitok_run_dir / "run_manifest.json"),
        "--dense-manifest",
        str(args.dense_run_dir / "run_manifest.json"),
        "--cutoff-step",
        str(args.cutoff_step),
        "--expected-revision",
        args.expected_training_revision,
        "--expected-branch",
        args.expected_training_branch,
        "--cofitok-origin",
        str(args.cofitok_run_dir / "train_metrics.jsonl"),
        "--dense-origin",
        str(args.dense_run_dir / "train_metrics.jsonl"),
        "--output",
        str(output),
    ]
    if snapshot_dir is not None:
        command.extend(["--snapshot-dir", str(snapshot_dir)])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    builder_src = str((args.builder_project / "src").resolve())
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        builder_src
        if not prior_pythonpath
        else builder_src + os.pathsep + prior_pythonpath
    )
    return subprocess.run(
        command,
        cwd=args.builder_project,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=args.builder_timeout_seconds,
    )


def validate_trajectory_report(
    path: Path,
    *,
    args: argparse.Namespace,
    cofitok_snapshot: Path,
    dense_snapshot: Path,
) -> dict[str, Any]:
    report = read_object(path)
    if report.get("schema_version") != 3:
        raise ValueError("trajectory report is not resume-aware schema 3")
    if report.get("role") != TRAJECTORY_ROLE or report.get("status") != "pass":
        raise ValueError("trajectory report is not a passing matched diagnostic")
    if report.get("cutoff_step") != args.cutoff_step:
        raise ValueError("trajectory cutoff differs")
    if report.get("images_seen_per_method") != (
        args.cutoff_step * args.expected_effective_batch
    ):
        raise ValueError("trajectory image exposure differs")

    contract = report.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("trajectory report lacks its matched contract")
    git = contract.get("git")
    expected_git = {
        "revision": args.expected_training_revision,
        "branch": args.expected_training_branch,
        "dirty": False,
    }
    if not isinstance(git, dict) or any(
        git.get(field) != expected for field, expected in expected_git.items()
    ):
        raise ValueError("trajectory training Git identity differs")
    if contract.get("effective_batch_size") != args.expected_effective_batch:
        raise ValueError("trajectory effective batch differs")
    if contract.get("generation_pair_contract", {}).get("valid") is not True:
        raise ValueError("trajectory generation pair contract failed")

    trajectories = report.get("trajectories")
    if not isinstance(trajectories, dict):
        raise ValueError("trajectory report lacks method trajectories")
    for label in ("cofitok", "dense_identity"):
        method = trajectories.get(label)
        if not isinstance(method, dict):
            raise ValueError(f"trajectory report lacks {label}")
        if method.get("last_step") != args.cutoff_step:
            raise ValueError(f"{label} trajectory cutoff differs")
        if method.get("samples_seen") != (
            args.cutoff_step * args.expected_effective_batch
        ):
            raise ValueError(f"{label} trajectory samples_seen differs")
        if method.get("all_numeric_metrics_finite") is not True:
            raise ValueError(f"{label} trajectory contains nonfinite metrics")
        if method.get("validation_provenance_complete") is not True:
            raise ValueError(f"{label} validation provenance is incomplete")

    paired = report.get("paired_fixed_validation")
    if not isinstance(paired, dict):
        raise ValueError("trajectory report lacks paired validation")
    events = paired.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("trajectory report lacks paired validation events")
    if events[-1].get("step") != args.cutoff_step:
        raise ValueError("trajectory paired endpoint differs")
    observed_scales = events[-1].get("observed_schedule_scales")
    if (
        not isinstance(observed_scales, dict)
        or not math.isclose(
            float(observed_scales.get("ema_teacher_consistency", -1.0)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("trajectory endpoint is not at full EMA-teacher strength")

    boundary = report.get("claim_boundary")
    required_false = (
        "schedule_regime_quality_claim_allowed",
        "quality_claim_allowed",
        "formal_50k_gate_substitute",
        "promotion_authorization_allowed",
        "full_training_launch_allowed",
        "sample_quality_metrics_present",
    )
    if not isinstance(boundary, dict) or any(
        boundary.get(field) is not False for field in required_false
    ):
        raise ValueError("trajectory claim boundary is not non-authorizing")

    builder = report.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("trajectory report lacks builder identity")
    if builder.get("sha256") != args.expected_builder_source_sha256:
        raise ValueError("trajectory report builder SHA256 differs")
    builder_git = builder.get("git")
    expected_builder_git = {
        "revision": args.expected_builder_revision,
        "branch": args.expected_builder_branch,
        "tracked_dirty": False,
    }
    if not isinstance(builder_git, dict) or any(
        builder_git.get(field) != expected
        for field, expected in expected_builder_git.items()
    ):
        raise ValueError("trajectory report builder Git identity differs")

    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("trajectory report lacks source identities")
    expected_sources = {
        "cofitok_metrics": (
            args.cofitok_run_dir / "train_metrics.jsonl",
            cofitok_snapshot,
        ),
        "dense_metrics": (
            args.dense_run_dir / "train_metrics.jsonl",
            dense_snapshot,
        ),
    }
    for label, (origin, snapshot) in expected_sources.items():
        source = sources.get(label)
        if not isinstance(source, dict):
            raise ValueError(f"trajectory report lacks {label} identity")
        if not _same_path(source.get("origin", ""), origin):
            raise ValueError(f"trajectory {label} origin differs")
        observed_file = source.get("observed_file")
        if not isinstance(observed_file, dict) or not _same_path(
            observed_file.get("path", ""), snapshot
        ):
            raise ValueError(f"trajectory {label} snapshot path differs")
        identity = file_identity(snapshot)
        if observed_file.get("bytes") != identity["bytes"]:
            raise ValueError(f"trajectory {label} snapshot bytes differ")
        if observed_file.get("sha256") != identity["sha256"]:
            raise ValueError(f"trajectory {label} snapshot SHA256 differs")

    return report


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            for block in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                destination_handle.write(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_trajectory_artifact(
    args: argparse.Namespace,
    *,
    control_identity: dict[str, Any],
    builder_identity: dict[str, Any],
    schedule_evidence: dict[str, Any],
) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "trajectory_report.json"
    receipt_path = output_dir / "waiter_receipt.json"
    snapshot_dir = output_dir / "source_snapshots"
    cofitok_snapshot = (
        snapshot_dir
        / f"cofitok_train_metrics_through_step_{args.cutoff_step:08d}.jsonl"
    )
    dense_snapshot = (
        snapshot_dir
        / f"dense_train_metrics_through_step_{args.cutoff_step:08d}.jsonl"
    )
    if output_dir.exists():
        if not report_path.is_file() or not receipt_path.is_file():
            raise FileExistsError("trajectory output directory is incomplete")
        validate_trajectory_report(
            report_path,
            args=args,
            cofitok_snapshot=cofitok_snapshot,
            dense_snapshot=dense_snapshot,
        )
        receipt = read_object(receipt_path)
        if receipt.get("status") != "pass" or receipt.get("scope") != SCOPE:
            raise ValueError("existing trajectory waiter receipt is invalid")
        return {
            "reused": True,
            "report": file_identity(report_path),
            "receipt": file_identity(receipt_path),
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.staging.",
    ) as temporary_name:
        staging = Path(temporary_name)
        staging_snapshots = staging / "source_snapshots"
        staging_report = staging / "trajectory_report.json"
        _invoke_builder(
            args,
            cofitok_metrics=args.cofitok_run_dir / "train_metrics.jsonl",
            dense_metrics=args.dense_run_dir / "train_metrics.jsonl",
            output=staging_report,
            snapshot_dir=staging_snapshots,
        )
        staged_cofitok = (
            staging_snapshots
            / f"cofitok_train_metrics_through_step_{args.cutoff_step:08d}.jsonl"
        )
        staged_dense = (
            staging_snapshots
            / f"dense_train_metrics_through_step_{args.cutoff_step:08d}.jsonl"
        )
        validate_trajectory_report(
            staging_report,
            args=args,
            cofitok_snapshot=staged_cofitok,
            dense_snapshot=staged_dense,
        )

        output_dir.mkdir(parents=False, exist_ok=False)
        snapshot_dir.mkdir(parents=False, exist_ok=False)
        _atomic_copy(staged_cofitok, cofitok_snapshot)
        _atomic_copy(staged_dense, dense_snapshot)

        replay_one = output_dir / ".trajectory_report.replay_one.json"
        replay_two = output_dir / ".trajectory_report.replay_two.json"
        _invoke_builder(
            args,
            cofitok_metrics=cofitok_snapshot,
            dense_metrics=dense_snapshot,
            output=replay_one,
            snapshot_dir=None,
        )
        _invoke_builder(
            args,
            cofitok_metrics=cofitok_snapshot,
            dense_metrics=dense_snapshot,
            output=replay_two,
            snapshot_dir=None,
        )
        if replay_one.read_bytes() != replay_two.read_bytes():
            raise ValueError("trajectory builder replay is not byte reproducible")
        validate_trajectory_report(
            replay_one,
            args=args,
            cofitok_snapshot=cofitok_snapshot,
            dense_snapshot=dense_snapshot,
        )
        replay_sha256 = file_sha256(replay_one)
        os.replace(replay_one, report_path)
        replay_two.unlink()

    if verify_control_identity(args) != control_identity:
        raise ValueError("control identity changed while building the trajectory")
    if verify_builder_identity(args) != builder_identity:
        raise ValueError("builder identity changed while building the trajectory")
    if file_identity(args.dense_schedule_report) != schedule_evidence["identity"]:
        raise ValueError("dense schedule report changed while building the trajectory")

    receipt = {
        "schema_version": 1,
        "role": RECEIPT_ROLE,
        "status": "pass",
        "created_at": utc_now(),
        "scope": SCOPE,
        "cutoff_step": args.cutoff_step,
        "control": control_identity,
        "builder": builder_identity,
        "dense_schedule_report": schedule_evidence["identity"],
        "outputs": {
            "trajectory_report": file_identity(report_path),
            "cofitok_metrics_snapshot": file_identity(cofitok_snapshot),
            "dense_metrics_snapshot": file_identity(dense_snapshot),
            "byte_replay_verified": True,
            "replay_sha256": replay_sha256,
        },
    }
    write_json_report(receipt_path, receipt)
    return {
        "reused": False,
        "report": file_identity(report_path),
        "receipt": file_identity(receipt_path),
    }


def status_payload(
    args: argparse.Namespace,
    *,
    status: str,
    detail: str,
    started_at: str,
    control_identity: dict[str, Any],
    builder_identity: dict[str, Any],
    artifact: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "started_at": started_at,
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "scope": SCOPE,
        "cutoff_step": args.cutoff_step,
        "dense_schedule_report": args.dense_schedule_report.resolve().as_posix(),
        "output_dir": args.output_dir.resolve().as_posix(),
        "control": control_identity,
        "builder": builder_identity,
    }
    if artifact is not None:
        payload["artifact"] = artifact
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def run_waiter(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CUDA_VISIBLE_DEVICES must be exactly -1")
    if (
        args.cutoff_step < 1
        or args.poll_seconds < 1
        or args.timeout_seconds < 1
        or args.builder_timeout_seconds < 1
        or args.expected_effective_batch < 1
    ):
        raise ValueError("invalid waiter timing or trajectory parameters")
    control_identity = verify_control_identity(args)
    builder_identity = verify_builder_identity(args)
    started_at = utc_now()
    deadline = time.monotonic() + args.timeout_seconds

    while True:
        ready, detail = schedule_is_ready(args.dense_schedule_report)
        if ready:
            try:
                control_identity = verify_control_identity(args)
                builder_identity = verify_builder_identity(args)
                schedule_evidence = validate_schedule_report(
                    args.dense_schedule_report,
                    dense_run_dir=args.dense_run_dir,
                    training_checkout=args.training_checkout,
                    expected_training_revision=args.expected_training_revision,
                    expected_training_branch=args.expected_training_branch,
                    expected_training_tree=args.expected_training_tree,
                    expected_waiter_source_sha256=(
                        args.expected_schedule_waiter_source_sha256
                    ),
                    cutoff_step=args.cutoff_step,
                    expected_effective_batch=args.expected_effective_batch,
                    expected_training_steps=args.expected_training_steps,
                    expected_start_step=args.expected_schedule_start_step,
                    expected_warmup_steps=args.expected_schedule_warmup_steps,
                    expected_weight=args.expected_schedule_weight,
                )
                write_json_report(
                    args.status_output,
                    status_payload(
                        args,
                        status="building",
                        detail="dense_schedule_verified_freezing_matched_prefixes",
                        started_at=started_at,
                        control_identity=control_identity,
                        builder_identity=builder_identity,
                    ),
                )
                artifact = build_trajectory_artifact(
                    args,
                    control_identity=control_identity,
                    builder_identity=builder_identity,
                    schedule_evidence=schedule_evidence,
                )
                write_json_report(
                    args.status_output,
                    status_payload(
                        args,
                        status="pass",
                        detail="matched_trajectory_frozen_and_replayed",
                        started_at=started_at,
                        control_identity=control_identity,
                        builder_identity=builder_identity,
                        artifact=artifact,
                    ),
                )
                return 0
            except BaseException as error:
                write_json_report(
                    args.status_output,
                    status_payload(
                        args,
                        status="failed",
                        detail="matched_trajectory_invalid",
                        started_at=started_at,
                        control_identity=control_identity,
                        builder_identity=builder_identity,
                        error=error,
                    ),
                )
                raise

        write_json_report(
            args.status_output,
            status_payload(
                args,
                status="waiting",
                detail=detail,
                started_at=started_at,
                control_identity=control_identity,
                builder_identity=builder_identity,
            ),
        )
        if args.once:
            return 0
        if time.monotonic() >= deadline:
            error = TimeoutError("matched trajectory waiter timed out")
            write_json_report(
                args.status_output,
                status_payload(
                    args,
                    status="failed",
                    detail="timeout",
                    started_at=started_at,
                    control_identity=control_identity,
                    builder_identity=builder_identity,
                    error=error,
                ),
            )
            raise error
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for an exact dense schedule audit, then freeze and byte-replay "
            "a non-authorizing matched training trajectory without GPU use."
        )
    )
    parser.add_argument("--control-project", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-source-sha256", required=True)
    parser.add_argument("--builder-project", type=Path, required=True)
    parser.add_argument("--expected-builder-revision", required=True)
    parser.add_argument("--expected-builder-branch", required=True)
    parser.add_argument("--expected-builder-tree", required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--training-checkout", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--dense-schedule-report", type=Path, required=True)
    parser.add_argument("--expected-schedule-waiter-source-sha256", required=True)
    parser.add_argument("--cutoff-step", type=int, required=True)
    parser.add_argument("--expected-effective-batch", type=int, required=True)
    parser.add_argument("--expected-training-steps", type=int, required=True)
    parser.add_argument("--expected-schedule-start-step", type=int, required=True)
    parser.add_argument("--expected-schedule-warmup-steps", type=int, required=True)
    parser.add_argument("--expected-schedule-weight", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=86_400)
    parser.add_argument("--builder-timeout-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_path = args.status_output.with_suffix(args.status_output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("matched trajectory waiter lock is held") from error
        return run_waiter(args)


if __name__ == "__main__":
    raise SystemExit(main())
