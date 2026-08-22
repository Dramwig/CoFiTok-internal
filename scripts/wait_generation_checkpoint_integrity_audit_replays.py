from __future__ import annotations

import argparse
import importlib.util
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCE = (
    PROJECT_ROOT / "scripts" / "verify_generation_checkpoint_integrity_audit_replay.py"
)
ROLE = "generation_checkpoint_physical_integrity_audit_replay_waiter"
REPLAY_ROLE = "generation_checkpoint_physical_integrity_audit_replay"
SOURCE_WAITER_ROLE = "generation_checkpoint_physical_integrity_milestone_waiter"
READ_ONLY_SCOPE = {
    "read_only_checkpoint_verification": True,
    "gpu_required": False,
    "training_process_signals_allowed": False,
    "sampling_authorization_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_verifier(path: Path = VERIFIER_SOURCE) -> ModuleType:
    name = "cofitok_generation_checkpoint_audit_replay_waiter_verifier"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checkpoint replay verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_checkpoint_steps(values: list[int]) -> tuple[int, ...]:
    if not values or any(step < 1 for step in values):
        raise ValueError("at least one positive checkpoint step is required")
    if len(values) != len(set(values)):
        raise ValueError("checkpoint steps must be unique")
    return tuple(sorted(values))


def milestone_paths(
    audit_dir: Path,
    *,
    alias: str,
    step: int,
) -> dict[str, Path]:
    stem = f"{alias}_checkpoint_step_{step:08d}"
    return {
        "audit_report": audit_dir / f"{stem}_physical_integrity_audit.json",
        "waiter_status": audit_dir / f"{stem}_waiter_status.json",
        "replay_output": audit_dir / f"{stem}_physical_integrity_audit_replay.json",
    }


def _timestamp_age_seconds(value: Any) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError("source waiter timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source waiter timestamp is not timezone-aware")
    return max(
        0.0,
        (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds(),
    )


def observe_source_milestone(
    verifier: ModuleType,
    *,
    paths: dict[str, Path],
    run_dir: Path,
    step: int,
    auditor_source: dict[str, Any],
    silence_seconds: float,
) -> dict[str, Any]:
    status_path = paths["waiter_status"]
    audit_path = paths["audit_report"]
    if not status_path.is_file():
        return {
            "state": "waiting",
            "detail": "source_waiter_status_missing",
            "checkpoint_step": step,
        }
    status, status_identity = verifier.stable_json(
        status_path,
        name=f"checkpoint {step} source waiter status",
    )
    source_status = str(status.get("status", ""))
    if (
        status.get("schema_version") != 1
        or status.get("role") != SOURCE_WAITER_ROLE
        or source_status not in {"waiting", "pass", "failed"}
        or int(status.get("checkpoint_step", -1)) != step
        or status.get("run_dir") != run_dir.resolve().as_posix()
        or status.get("audit_output") != audit_path.resolve().as_posix()
        or status.get("waiter_source") != auditor_source
    ):
        raise ValueError(f"checkpoint {step} source waiter contract differs")
    source_scope = status.get("scope")
    expected_source_scope = {
        "read_only_checkpoint_verification": True,
        "gpu_required": False,
        "training_process_signals_allowed": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
    }
    if source_scope != expected_source_scope:
        raise ValueError(f"checkpoint {step} source waiter scope differs")
    if source_status == "failed":
        raise RuntimeError(
            f"checkpoint {step} source waiter failed: {status.get('detail')}"
        )
    if source_status == "waiting":
        age_seconds = _timestamp_age_seconds(status.get("updated_at"))
        if age_seconds > silence_seconds:
            raise RuntimeError(
                f"checkpoint {step} source waiter is stale by {age_seconds:.1f}s"
            )
        return {
            "state": "waiting",
            "detail": str(status.get("detail", "waiting_for_source_audit")),
            "checkpoint_step": step,
            "source_status": status_identity,
            "source_status_age_seconds": age_seconds,
        }
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"checkpoint {step} source waiter passed without its audit report"
        )
    audit, audit_identity = verifier.stable_json(
        audit_path,
        name=f"checkpoint {step} source audit report",
    )
    if status.get("audit_output_identity") != audit_identity:
        raise ValueError(f"checkpoint {step} source audit identity differs")
    checkpoint = audit.get("checkpoint")
    payload = checkpoint.get("payload") if isinstance(checkpoint, dict) else None
    checkpoint_sha256 = (
        str(payload.get("sha256", "")) if isinstance(payload, dict) else ""
    )
    verifier._require_sha256(checkpoint_sha256, name=f"checkpoint {step}")
    if int(checkpoint.get("step", -1)) != step:
        raise ValueError(f"checkpoint {step} source audit target differs")
    return {
        "state": "ready",
        "detail": "source_audit_passed",
        "checkpoint_step": step,
        "checkpoint_sha256": checkpoint_sha256,
        "source_status": status_identity,
        "source_audit": audit_identity,
    }


def build_verifier_command(
    *,
    python: str,
    verifier_source: Path,
    source: dict[str, Any],
    replay_output: Path,
    auditor_source_path: Path,
    auditor_source_sha256: str,
    auditor_checkout: Path,
    auditor_git: dict[str, Any],
    training_checkout: Path,
    training_git: dict[str, Any],
    dataset_sha256: str,
    runtime_sha256: str,
    effective_batch: int,
    project_git: dict[str, Any],
    verifier_source_sha256: str,
) -> list[str]:
    return [
        python,
        str(verifier_source),
        "--audit-report",
        source["source_audit"]["path"],
        "--expected-audit-report-sha256",
        source["source_audit"]["sha256"],
        "--waiter-status",
        source["source_status"]["path"],
        "--expected-waiter-status-sha256",
        source["source_status"]["sha256"],
        "--auditor-source",
        str(auditor_source_path),
        "--expected-auditor-source-sha256",
        auditor_source_sha256,
        "--auditor-checkout",
        str(auditor_checkout),
        "--expected-auditor-revision",
        auditor_git["revision"],
        "--expected-auditor-tree",
        auditor_git["tree"],
        "--expected-auditor-branch",
        auditor_git["branch"],
        "--training-checkout",
        str(training_checkout),
        "--expected-training-revision",
        training_git["revision"],
        "--expected-training-tree",
        training_git["tree"],
        "--expected-training-branch",
        training_git["branch"],
        "--expected-checkpoint-step",
        str(source["checkpoint_step"]),
        "--expected-checkpoint-sha256",
        source["checkpoint_sha256"],
        "--expected-dataset-sha256",
        dataset_sha256,
        "--expected-runtime-sha256",
        runtime_sha256,
        "--effective-batch",
        str(effective_batch),
        "--expected-verifier-revision",
        project_git["revision"],
        "--expected-verifier-tree",
        project_git["tree"],
        "--expected-verifier-branch",
        project_git["branch"],
        "--expected-verifier-source-sha256",
        verifier_source_sha256,
        "--output",
        str(replay_output),
    ]


def immutable_receipt_view(receipt: dict[str, Any]) -> dict[str, Any]:
    latest = receipt.get("latest_pointer_replay")
    metrics = receipt.get("metrics_replay")
    if not isinstance(latest, dict) or not isinstance(metrics, dict):
        raise ValueError("checkpoint replay receipt historical evidence is malformed")
    return {
        "schema_version": receipt.get("schema_version"),
        "role": receipt.get("role"),
        "status": receipt.get("status"),
        "verifier_git": receipt.get("verifier_git"),
        "verifier_source": receipt.get("verifier_source"),
        "audit_report": receipt.get("audit_report"),
        "waiter_status": receipt.get("waiter_status"),
        "auditor_source": receipt.get("auditor_source"),
        "auditor_checkout": receipt.get("auditor_checkout"),
        "training_checkout": receipt.get("training_checkout"),
        "checkpoint": receipt.get("checkpoint"),
        "historical_latest": {
            "historical_identity": latest.get("historical_identity"),
            "historical_content": latest.get("historical_content"),
            "historical_byte_identity_reconstructed": latest.get(
                "historical_byte_identity_reconstructed"
            ),
        },
        "historical_metrics": {
            "audit_time_prefix": metrics.get("audit_time_prefix"),
            "audit_time_prefix_byte_exact": metrics.get(
                "audit_time_prefix_byte_exact"
            ),
            "target_row": metrics.get("target_row"),
        },
        "checks": receipt.get("checks"),
        "scope": receipt.get("scope"),
    }


def _basic_receipt_contract(
    receipt: dict[str, Any],
    *,
    step: int,
    checkpoint_sha256: str,
    project_git: dict[str, Any],
) -> None:
    checkpoint = receipt.get("checkpoint")
    payload = checkpoint.get("payload") if isinstance(checkpoint, dict) else None
    if (
        receipt.get("schema_version") != 1
        or receipt.get("role") != REPLAY_ROLE
        or receipt.get("status") != "pass"
        or receipt.get("verifier_git") != project_git
        or not isinstance(checkpoint, dict)
        or int(checkpoint.get("step", -1)) != step
        or not isinstance(payload, dict)
        or payload.get("sha256") != checkpoint_sha256
        or receipt.get("scope") != {
            "read_only_checkpoint_verification": True,
            "gpu_required": False,
            "training_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "report_is_checkpoint_promotion_gate": False,
            "report_is_generation_quality_evidence": False,
            "sampling_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        }
    ):
        raise ValueError(f"checkpoint {step} replay receipt contract differs")


def validate_existing_replay(
    verifier: ModuleType,
    *,
    output: Path,
    source: dict[str, Any],
    auditor_source_path: Path,
    auditor_source_sha256: str,
    auditor_checkout: Path,
    auditor_git: dict[str, Any],
    training_checkout: Path,
    training_git: dict[str, Any],
    dataset_sha256: str,
    runtime_sha256: str,
    effective_batch: int,
    project_git: dict[str, Any],
) -> dict[str, Any]:
    existing, existing_identity = verifier.stable_json(
        output,
        name=f"checkpoint {source['checkpoint_step']} replay receipt",
    )
    fresh = verifier.verify_replay(
        audit_report_path=Path(source["source_audit"]["path"]),
        expected_audit_report_sha256=source["source_audit"]["sha256"],
        waiter_status_path=Path(source["source_status"]["path"]),
        expected_waiter_status_sha256=source["source_status"]["sha256"],
        auditor_source_path=auditor_source_path,
        expected_auditor_source_sha256=auditor_source_sha256,
        auditor_checkout=auditor_checkout,
        expected_auditor_revision=auditor_git["revision"],
        expected_auditor_tree=auditor_git["tree"],
        expected_auditor_branch=auditor_git["branch"],
        training_checkout=training_checkout,
        expected_training_revision=training_git["revision"],
        expected_training_tree=training_git["tree"],
        expected_training_branch=training_git["branch"],
        expected_checkpoint_step=source["checkpoint_step"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_dataset_sha256=dataset_sha256,
        expected_runtime_sha256=runtime_sha256,
        effective_batch=effective_batch,
        verifier_git=project_git,
    )
    if immutable_receipt_view(existing) != immutable_receipt_view(fresh):
        raise ValueError(
            f"checkpoint {source['checkpoint_step']} existing replay receipt differs"
        )
    return existing_identity


def execute_replay(
    verifier: ModuleType,
    *,
    command: list[str],
    project: Path,
    output: Path,
    step: int,
    checkpoint_sha256: str,
    project_git: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"checkpoint {step} replay output already exists")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join(
        [str((project / "src").resolve()), str(project.resolve())]
    )
    result = subprocess.run(
        command,
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"checkpoint {step} replay verifier failed"
            + (f": {diagnostic[-4000:]}" if diagnostic else "")
        )
    if not output.is_file():
        raise RuntimeError(f"checkpoint {step} replay produced no receipt")
    receipt, identity = verifier.stable_json(
        output,
        name=f"checkpoint {step} replay receipt",
    )
    _basic_receipt_contract(
        receipt,
        step=step,
        checkpoint_sha256=checkpoint_sha256,
        project_git=project_git,
    )
    return identity


def static_context(args: argparse.Namespace, verifier: ModuleType) -> dict[str, Any]:
    project = verifier.reject_symlink_chain(
        args.project,
        name="checkpoint replay waiter project",
    )
    run_dir = verifier.reject_symlink_chain(
        args.run_dir,
        name="checkpoint replay waiter run directory",
    )
    audit_dir = verifier.reject_symlink_chain(
        args.audit_dir,
        name="checkpoint replay waiter audit directory",
    )
    auditor_checkout = verifier.reject_symlink_chain(
        args.auditor_checkout,
        name="checkpoint replay waiter auditor checkout",
    )
    training_checkout = verifier.reject_symlink_chain(
        args.training_checkout,
        name="checkpoint replay waiter training checkout",
    )
    if project != PROJECT_ROOT.resolve() or not run_dir.is_dir() or not audit_dir.is_dir():
        raise ValueError("checkpoint replay waiter canonical paths differ")
    project_git = verifier.require_git_identity(
        project,
        revision=args.expected_project_revision,
        tree=args.expected_project_tree,
        branch=args.expected_project_branch,
        label="checkpoint replay waiter project",
    )
    auditor_git = verifier.require_git_identity(
        auditor_checkout,
        revision=args.expected_auditor_revision,
        tree=args.expected_auditor_tree,
        branch=args.expected_auditor_branch,
        label="checkpoint integrity auditor checkout",
    )
    training_git = verifier.require_git_identity(
        training_checkout,
        revision=args.expected_training_revision,
        tree=args.expected_training_tree,
        branch=args.expected_training_branch,
        label="training checkout",
    )
    waiter_source = verifier.stable_file_identity(
        Path(__file__),
        name="checkpoint replay waiter source",
    )
    verifier_source = verifier.stable_file_identity(
        VERIFIER_SOURCE,
        name="checkpoint replay verifier source",
    )
    auditor_source_path = (
        auditor_checkout / "scripts" / "wait_generation_checkpoint_integrity_audit.py"
    )
    auditor_source = verifier.stable_file_identity(
        auditor_source_path,
        name="checkpoint integrity auditor source",
    )
    expected_hashes = (
        ("waiter", waiter_source["sha256"], args.expected_waiter_source_sha256),
        ("verifier", verifier_source["sha256"], args.expected_verifier_source_sha256),
        ("auditor", auditor_source["sha256"], args.expected_auditor_source_sha256),
    )
    for label, actual, expected in expected_hashes:
        if actual != expected:
            raise ValueError(f"checkpoint replay {label} source SHA256 differs")
    return {
        "project": project,
        "run_dir": run_dir,
        "audit_dir": audit_dir,
        "auditor_checkout": auditor_checkout,
        "training_checkout": training_checkout,
        "project_git": project_git,
        "auditor_git": auditor_git,
        "training_git": training_git,
        "waiter_source": waiter_source,
        "verifier_source": verifier_source,
        "auditor_source": auditor_source,
        "auditor_source_path": auditor_source_path,
    }


def _expected(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_steps": list(args.checkpoint_steps),
        "alias": args.alias,
        "project_revision": args.expected_project_revision,
        "project_tree": args.expected_project_tree,
        "project_branch": args.expected_project_branch,
        "training_revision": args.expected_training_revision,
        "training_tree": args.expected_training_tree,
        "training_branch": args.expected_training_branch,
        "auditor_revision": args.expected_auditor_revision,
        "auditor_tree": args.expected_auditor_tree,
        "auditor_branch": args.expected_auditor_branch,
        "dataset_sha256": args.expected_dataset_sha256,
        "runtime_sha256": args.expected_runtime_sha256,
        "effective_batch": args.effective_batch,
        **READ_ONLY_SCOPE,
    }


def status_payload(
    *,
    status: str,
    detail: str,
    args: argparse.Namespace,
    started_at: str,
    context: dict[str, Any] | None = None,
    milestones: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": utc_now(),
        "expected": _expected(args),
        "waiter_git": context.get("project_git") if context else None,
        "waiter_source": context.get("waiter_source") if context else None,
        "verifier_source": context.get("verifier_source") if context else None,
        "milestones": milestones or {},
        "completed_count": sum(
            row.get("state") == "pass" for row in (milestones or {}).values()
        ),
        "expected_count": len(args.checkpoint_steps),
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "scope": READ_ONLY_SCOPE,
    }


def run_locked(args: argparse.Namespace, verifier: ModuleType) -> int:
    started_at = utc_now()
    deadline = time.monotonic() + args.timeout_seconds
    completed: dict[int, dict[str, Any]] = {}
    milestone_status: dict[str, Any] = {}
    while True:
        context = static_context(args, verifier)
        for step in args.checkpoint_steps:
            key = str(step)
            paths = milestone_paths(
                context["audit_dir"],
                alias=args.alias,
                step=step,
            )
            source = observe_source_milestone(
                verifier,
                paths=paths,
                run_dir=context["run_dir"],
                step=step,
                auditor_source=context["auditor_source"],
                silence_seconds=args.source_status_silence_seconds,
            )
            if step in completed:
                milestone_status[key] = {
                    "state": "pass",
                    "detail": "replay_completed",
                    "checkpoint_step": step,
                    "replay_output": completed[step],
                }
                continue
            if paths["replay_output"].exists():
                if source["state"] != "ready":
                    raise ValueError(
                        f"checkpoint {step} replay exists before its source audit passes"
                    )
                write_json_report(
                    args.status_output,
                    status_payload(
                        status="running",
                        detail=f"validating_existing_replay_step_{step:08d}",
                        args=args,
                        started_at=started_at,
                        context=context,
                        milestones=milestone_status,
                    ),
                )
                completed[step] = validate_existing_replay(
                    verifier,
                    output=paths["replay_output"],
                    source=source,
                    auditor_source_path=context["auditor_source_path"],
                    auditor_source_sha256=args.expected_auditor_source_sha256,
                    auditor_checkout=context["auditor_checkout"],
                    auditor_git=context["auditor_git"],
                    training_checkout=context["training_checkout"],
                    training_git=context["training_git"],
                    dataset_sha256=args.expected_dataset_sha256,
                    runtime_sha256=args.expected_runtime_sha256,
                    effective_batch=args.effective_batch,
                    project_git=context["project_git"],
                )
            elif source["state"] == "ready":
                write_json_report(
                    args.status_output,
                    status_payload(
                        status="running",
                        detail=f"replaying_checkpoint_step_{step:08d}",
                        args=args,
                        started_at=started_at,
                        context=context,
                        milestones=milestone_status,
                    ),
                )
                command = build_verifier_command(
                    python=sys.executable,
                    verifier_source=VERIFIER_SOURCE,
                    source=source,
                    replay_output=paths["replay_output"],
                    auditor_source_path=context["auditor_source_path"],
                    auditor_source_sha256=args.expected_auditor_source_sha256,
                    auditor_checkout=context["auditor_checkout"],
                    auditor_git=context["auditor_git"],
                    training_checkout=context["training_checkout"],
                    training_git=context["training_git"],
                    dataset_sha256=args.expected_dataset_sha256,
                    runtime_sha256=args.expected_runtime_sha256,
                    effective_batch=args.effective_batch,
                    project_git=context["project_git"],
                    verifier_source_sha256=args.expected_verifier_source_sha256,
                )
                completed[step] = execute_replay(
                    verifier,
                    command=command,
                    project=context["project"],
                    output=paths["replay_output"],
                    step=step,
                    checkpoint_sha256=source["checkpoint_sha256"],
                    project_git=context["project_git"],
                )
            if step in completed:
                milestone_status[key] = {
                    "state": "pass",
                    "detail": "replay_completed",
                    "checkpoint_step": step,
                    "replay_output": completed[step],
                }
            else:
                milestone_status[key] = source

        if len(completed) == len(args.checkpoint_steps):
            write_json_report(
                args.status_output,
                status_payload(
                    status="pass",
                    detail="all_checkpoint_integrity_replays_completed",
                    args=args,
                    started_at=started_at,
                    context=context,
                    milestones=milestone_status,
                ),
            )
            return 0
        pending = next(step for step in args.checkpoint_steps if step not in completed)
        write_json_report(
            args.status_output,
            status_payload(
                status="waiting",
                detail=f"waiting_for_source_audit_step_{pending:08d}",
                args=args,
                started_at=started_at,
                context=context,
                milestones=milestone_status,
            ),
        )
        if args.once:
            return 0
        if time.monotonic() >= deadline:
            raise TimeoutError("checkpoint integrity replay waiter timed out")
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for source-bound checkpoint physical-integrity audits and "
            "publish independent, read-only, non-authorizing replay receipts."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        action="append",
        dest="checkpoint_step_values",
        required=True,
    )
    parser.add_argument("--auditor-checkout", type=Path, required=True)
    parser.add_argument("--training-checkout", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--expected-project-revision", required=True)
    parser.add_argument("--expected-project-tree", required=True)
    parser.add_argument("--expected-project-branch", required=True)
    parser.add_argument("--expected-waiter-source-sha256", required=True)
    parser.add_argument("--expected-verifier-source-sha256", required=True)
    parser.add_argument("--expected-auditor-source-sha256", required=True)
    parser.add_argument("--expected-auditor-revision", required=True)
    parser.add_argument("--expected-auditor-tree", required=True)
    parser.add_argument("--expected-auditor-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=604_800.0)
    parser.add_argument(
        "--source-status-silence-seconds",
        type=float,
        default=1_800.0,
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.checkpoint_steps = parse_checkpoint_steps(args.checkpoint_step_values)
    if not re.fullmatch(r"[a-z0-9_]+", args.alias):
        raise ValueError("checkpoint audit alias is invalid")
    if (
        args.effective_batch < 1
        or args.poll_seconds <= 0.0
        or args.timeout_seconds <= 0.0
        or args.source_status_silence_seconds <= 0.0
    ):
        raise ValueError("checkpoint replay waiter numeric arguments are invalid")
    verifier = load_verifier()
    for name, value in (
        ("waiter source", args.expected_waiter_source_sha256),
        ("verifier source", args.expected_verifier_source_sha256),
        ("auditor source", args.expected_auditor_source_sha256),
        ("dataset", args.expected_dataset_sha256),
        ("runtime", args.expected_runtime_sha256),
    ):
        verifier._require_sha256(value, name=name)
    return args


def run_waiter(args: argparse.Namespace, verifier: ModuleType | None = None) -> int:
    loaded = verifier or load_verifier()
    started_at = utc_now()
    try:
        with exclusive_output_lock(args.status_output, role=ROLE):
            try:
                return run_locked(args, loaded)
            except Exception as error:
                context = None
                try:
                    context = static_context(args, loaded)
                except Exception:
                    pass
                write_json_report(
                    args.status_output,
                    status_payload(
                        status="failed",
                        detail=f"{type(error).__name__}: {error}",
                        args=args,
                        started_at=started_at,
                        context=context,
                        error=error,
                    ),
                )
                print(f"{type(error).__name__}: {error}", file=sys.stderr)
                return 1
    except OutputLockError as error:
        print(str(error), file=sys.stderr)
        return 75


def main() -> int:
    return run_waiter(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
