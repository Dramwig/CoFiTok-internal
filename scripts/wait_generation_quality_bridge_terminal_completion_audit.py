from __future__ import annotations

import argparse
import copy
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ModuleNotFoundError:  # Windows-only unit tests.
    fcntl = None  # type: ignore[assignment]

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report

try:
    import build_generation_quality_bridge_terminal_completion_audit as builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_quality_bridge_terminal_completion_audit as builder,
    )


SCHEMA_VERSION = 1
ROLE = "generation_quality_bridge_terminal_completion_audit_waiter"
DEPLOYMENT_ROLE = (
    "generation_quality_bridge_terminal_completion_audit_waiter_deployment"
)
TERMINAL_WAITER_ROLE = "generation_terminal_system_claim_guard_waiter"
COMPARISON_WAITER_ROLE = "generation_quality_bridge_comparison_waiter"
CHECKPOINT_WAITER_ROLE = "generation_checkpoint_physical_integrity_milestone_waiter"
REPLAY_WAITER_ROLE = "generation_checkpoint_physical_integrity_audit_replay_waiter"
CHECKPOINT_STEPS = (90_000, 95_000, 100_000)
ALIASES = ("cofitok", "dense")
SCOPE = {
    "diagnostic_non_authorizing": True,
    "cpu_only_evidence_replay": True,
    "gpu_required": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
}


class LockBusy(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def _read_json_if_present(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json_object(path, name=label)


def validate_runtime(*, require_detached: bool = True) -> dict[str, Any]:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    omp = os.environ.get("OMP_NUM_THREADS")
    mkl = os.environ.get("MKL_NUM_THREADS")
    if cuda not in {"", "-1"}:
        raise ValueError("terminal completion waiter must run with CUDA hidden")
    if omp != "1" or mkl != "1":
        raise ValueError(
            "terminal completion waiter must use OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1"
        )
    ppid = os.getppid()
    nice = os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None
    ionice = subprocess.run(
        ["ionice", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if require_detached:
        if ppid != 1:
            raise ValueError(
                "terminal completion waiter must be detached with parent PID 1"
            )
        if nice is None or nice < 10:
            raise ValueError(
                "terminal completion waiter nice priority must be at least 10"
            )
        if ionice != "idle":
            raise ValueError("terminal completion waiter must use idle I/O priority")
    return {
        "pid": os.getpid(),
        "ppid": ppid,
        "cuda_visible_devices": cuda,
        "omp_num_threads": omp,
        "mkl_num_threads": mkl,
        "nice": nice,
        "ionice": ionice,
        "python_executable": Path(sys.executable).resolve().as_posix(),
    }


def _canonical_context(args: argparse.Namespace) -> dict[str, Any]:
    project = reject_symlink_chain(
        args.project,
        name="terminal completion waiter project",
    ).resolve()
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality bridge output root",
    ).resolve()
    output_dir_name = builder.validated_report_dir_name(
        args.expected_output_dir_name,
        prefix="terminal_completion_audit_v",
        label="terminal completion output directory name",
    )
    terminal_dir_name = builder.validated_report_dir_name(
        args.expected_terminal_dir_name,
        prefix="terminal_system_claim_guard_v",
        label="terminal guard directory name",
    )
    comparison_dir_name = builder.validated_report_dir_name(
        args.expected_comparison_dir_name,
        prefix="quality_bridge_comparison_v",
        label="comparison directory name",
    )
    output_dir = quality_root / "reports" / output_dir_name
    expected = {
        "terminal_guard": quality_root
        / "reports"
        / terminal_dir_name
        / "terminal_system_claim_guard.json",
        "terminal_status": quality_root
        / "reports"
        / terminal_dir_name
        / "waiter_status.json",
        "comparison": quality_root
        / "reports"
        / comparison_dir_name
        / "quality_bridge_comparison.json",
        "comparison_status": quality_root
        / "reports"
        / comparison_dir_name
        / "waiter_status.json",
        "audit_dir": quality_root / "reports" / "checkpoint_audits",
        "replay_status": quality_root
        / "reports"
        / "checkpoint_audits"
        / "dense_checkpoint_integrity_replay_waiter_status.json",
        "metrics_trust_receipt": quality_root
        / "reports"
        / "metrics_trust_boundary_v1"
        / "metrics_trust_receipt.json",
        "output": output_dir / "terminal_completion_audit.json",
        "status": output_dir / "waiter_status.json",
        "deployment": output_dir / "deployment_receipt.json",
        "lock": output_dir / "waiter.lock",
    }
    actual = {
        "terminal_guard": args.terminal_system_guard.resolve(),
        "terminal_status": args.terminal_waiter_status.resolve(),
        "comparison": args.comparison.resolve(),
        "comparison_status": args.comparison_waiter_status.resolve(),
        "audit_dir": args.checkpoint_audit_dir.resolve(),
        "replay_status": args.checkpoint_replay_waiter_status.resolve(),
        "metrics_trust_receipt": args.metrics_trust_receipt.resolve(),
        "output": args.output.resolve(),
        "status": args.status_output.resolve(),
        "deployment": args.deployment_receipt_output.resolve(),
        "lock": args.lock.resolve(),
    }
    for name, expected_path in expected.items():
        if actual[name] != expected_path.resolve():
            raise ValueError(
                f"terminal completion waiter canonical path differs: {name}"
            )

    git = {**git_provenance(project), "tree": _tree(project)}
    expected_git = {
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError(f"terminal completion waiter Git identity differs: {git}")
    cwd = Path.cwd().resolve()
    if cwd != project:
        raise ValueError("terminal completion waiter cwd is not the project checkout")
    waiter_source = file_identity(Path(__file__))
    builder_source = file_identity(
        project
        / "scripts"
        / "build_generation_quality_bridge_terminal_completion_audit.py"
    )
    if waiter_source["sha256"] != args.expected_waiter_source_sha256:
        raise ValueError("terminal completion waiter source SHA256 differs")
    if builder_source["sha256"] != args.expected_builder_source_sha256:
        raise ValueError("terminal completion builder source SHA256 differs")
    training_git = builder.require_git_identity(
        args.training_checkout,
        revision=args.expected_training_revision,
        tree=args.expected_training_tree,
        branch=args.expected_training_branch,
        label="quality bridge training checkout",
    )
    auditor_git = builder.require_git_identity(
        args.physical_auditor_checkout,
        revision=args.expected_physical_auditor_revision,
        tree=args.expected_physical_auditor_tree,
        branch=args.expected_physical_auditor_branch,
        label="physical checkpoint auditor checkout",
    )
    replay_git = builder.require_git_identity(
        args.checkpoint_replay_checkout,
        revision=args.expected_checkpoint_replay_revision,
        tree=args.expected_checkpoint_replay_tree,
        branch=args.expected_checkpoint_replay_branch,
        label="checkpoint replay checkout",
    )
    auditor_source = file_identity(
        args.physical_auditor_checkout.resolve()
        / "scripts"
        / "wait_generation_checkpoint_integrity_audit.py"
    )
    if auditor_source["sha256"] != args.expected_physical_auditor_source_sha256:
        raise ValueError("physical checkpoint auditor source SHA256 differs")
    replay_verifier_source = file_identity(
        args.checkpoint_replay_checkout.resolve()
        / "scripts"
        / "verify_generation_checkpoint_integrity_audit_replay.py"
    )
    if (
        replay_verifier_source["sha256"]
        != args.expected_checkpoint_replay_verifier_source_sha256
    ):
        raise ValueError("checkpoint replay verifier source SHA256 differs")
    metrics_trust_identity = file_identity(actual["metrics_trust_receipt"])
    if metrics_trust_identity["sha256"] != args.expected_metrics_trust_receipt_sha256:
        raise ValueError("terminal metrics trust receipt SHA256 differs")
    return {
        "project": project,
        "cwd": cwd,
        "quality_root": quality_root,
        "git": git,
        "waiter_source": waiter_source,
        "builder_source": builder_source,
        "training_git": training_git,
        "auditor_git": auditor_git,
        "replay_git": replay_git,
        "auditor_source": auditor_source,
        "replay_verifier_source": replay_verifier_source,
        "metrics_trust_identity": metrics_trust_identity,
        **actual,
    }


def deployment_payload(
    args: argparse.Namespace,
    *,
    context: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": DEPLOYMENT_ROLE,
        "created_at": utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": runtime,
        "git": context["git"],
        "waiter_source": context["waiter_source"],
        "builder_source": context["builder_source"],
        "quality_output_root": context["quality_root"].as_posix(),
        "control": {
            "cwd": context["cwd"].as_posix(),
            "python_executable": runtime["python_executable"],
            "poll_seconds": args.poll_seconds,
            "timeout_seconds": args.timeout_seconds,
        },
        "sources": {
            "terminal_system_guard": context["terminal_guard"].as_posix(),
            "terminal_waiter_status": context["terminal_status"].as_posix(),
            "comparison": context["comparison"].as_posix(),
            "comparison_waiter_status": context["comparison_status"].as_posix(),
            "checkpoint_audit_dir": context["audit_dir"].as_posix(),
            "checkpoint_replay_waiter_status": context["replay_status"].as_posix(),
            "metrics_trust_receipt": context["metrics_trust_receipt"].as_posix(),
            "training_checkout": args.training_checkout.resolve().as_posix(),
            "physical_auditor_checkout": (
                args.physical_auditor_checkout.resolve().as_posix()
            ),
            "checkpoint_replay_checkout": (
                args.checkpoint_replay_checkout.resolve().as_posix()
            ),
        },
        "expected": {
            "training_git": {
                **context["training_git"],
            },
            "physical_auditor_git": {
                **context["auditor_git"],
            },
            "checkpoint_replay_git": {
                **context["replay_git"],
            },
            "physical_auditor_source": context["auditor_source"],
            "checkpoint_replay_verifier_source": context["replay_verifier_source"],
            "metrics_trust_receipt": context["metrics_trust_identity"],
            "dataset_sha256": args.expected_dataset_sha256,
            "runtime_sha256": args.expected_runtime_sha256,
            "effective_batch": args.effective_batch,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
        },
        "output": context["output"].as_posix(),
        "status_output": context["status"].as_posix(),
        "lock": context["lock"].as_posix(),
        "scope": copy.deepcopy(SCOPE),
    }


def prepare_deployment_receipt(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        existing = read_json_object(path, name="terminal completion deployment receipt")
        normalized = copy.deepcopy(payload)
        normalized["created_at"] = existing.get("created_at")
        normalized["pid"] = existing.get("pid")
        normalized["runtime"] = existing.get("runtime")
        if existing != normalized:
            raise ValueError("existing terminal completion deployment receipt differs")
    elif path.exists():
        raise ValueError("terminal completion deployment receipt is not a file")
    else:
        write_json_report(path, payload)
    return file_identity(path)


def _failure(path: Path, *, label: str) -> str | None:
    report = _read_json_if_present(path, label=label)
    if report is None:
        return None
    if report.get("status") in {"failed", "error", "invalid"}:
        return f"{label}:{report.get('status')}:{report.get('detail', '')}"
    return None


def observe_upstreams(context: dict[str, Any]) -> dict[str, Any]:
    failures = [
        failure
        for failure in (
            _failure(context["terminal_status"], label="terminal system guard waiter"),
            _failure(context["comparison_status"], label="comparison waiter"),
            _failure(context["replay_status"], label="checkpoint replay waiter"),
            _failure(context["metrics_trust_receipt"], label="metrics trust receipt"),
        )
        if failure is not None
    ]
    for alias in ALIASES:
        for step in CHECKPOINT_STEPS:
            status_path = (
                context["audit_dir"]
                / f"{alias}_checkpoint_step_{step:08d}_waiter_status.json"
            )
            failure = _failure(status_path, label=f"{alias} checkpoint {step} waiter")
            if failure is not None:
                failures.append(failure)
    if failures:
        return {
            "state": "failed",
            "detail": "upstream_terminal_evidence_failed",
            "failures": failures,
        }

    missing: list[str] = []
    terminal_status = _read_json_if_present(
        context["terminal_status"],
        label="terminal system guard waiter status",
    )
    comparison_status = _read_json_if_present(
        context["comparison_status"],
        label="comparison waiter status",
    )
    replay_status = _read_json_if_present(
        context["replay_status"],
        label="checkpoint replay waiter status",
    )
    metrics_trust_receipt = _read_json_if_present(
        context["metrics_trust_receipt"],
        label="terminal metrics trust receipt",
    )
    if terminal_status is None or terminal_status.get("status") != "completed":
        missing.append("terminal_system_guard_waiter")
    if not context["terminal_guard"].is_file():
        missing.append("terminal_system_guard")
    if metrics_trust_receipt is None or metrics_trust_receipt.get("status") != "pass":
        missing.append("metrics_trust_receipt")
    if comparison_status is None or comparison_status.get("status") != "pass":
        missing.append("quality_bridge_comparison_waiter")
    for path_name in ("comparison",):
        if not context[path_name].is_file():
            missing.append(path_name)
    comparison_dir = context["comparison"].parent
    for name in ("quality_bridge_comparison.md", "quality_bridge_comparison.csv"):
        if not (comparison_dir / name).is_file():
            missing.append(name)
    for alias in ALIASES:
        for step in CHECKPOINT_STEPS:
            stem = f"{alias}_checkpoint_step_{step:08d}"
            status_path = context["audit_dir"] / f"{stem}_waiter_status.json"
            audit_path = context["audit_dir"] / f"{stem}_physical_integrity_audit.json"
            status = _read_json_if_present(
                status_path,
                label=f"{alias} checkpoint {step} waiter status",
            )
            if status is None or status.get("status") != "pass":
                missing.append(f"{alias}_checkpoint_{step}_waiter")
            if not audit_path.is_file():
                missing.append(f"{alias}_checkpoint_{step}_audit")
    if replay_status is None or replay_status.get("status") != "pass":
        missing.append("dense_checkpoint_replay_waiter")
    for step in CHECKPOINT_STEPS:
        replay_path = (
            context["audit_dir"]
            / f"dense_checkpoint_step_{step:08d}_physical_integrity_audit_replay.json"
        )
        if not replay_path.is_file():
            missing.append(f"dense_checkpoint_{step}_replay")
    if missing:
        return {
            "state": "waiting",
            "detail": "waiting_for_exact_terminal_completion_sources",
            "missing": sorted(set(missing)),
        }
    return {
        "state": "ready",
        "detail": "exact_terminal_completion_sources_ready",
        "terminal_status": terminal_status.get("guard_status"),
    }


def upstream_metadata_paths(context: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "terminal_guard": context["terminal_guard"],
        "terminal_status": context["terminal_status"],
        "comparison": context["comparison"],
        "comparison_markdown": context["comparison"].parent
        / "quality_bridge_comparison.md",
        "comparison_csv": context["comparison"].parent
        / "quality_bridge_comparison.csv",
        "comparison_status": context["comparison_status"],
        "replay_status": context["replay_status"],
        "metrics_trust_receipt": context["metrics_trust_receipt"],
    }
    for alias in ALIASES:
        for step in CHECKPOINT_STEPS:
            stem = f"{alias}_checkpoint_step_{step:08d}"
            paths[f"{stem}_status"] = (
                context["audit_dir"] / f"{stem}_waiter_status.json"
            )
            paths[f"{stem}_audit"] = (
                context["audit_dir"] / f"{stem}_physical_integrity_audit.json"
            )
    for step in CHECKPOINT_STEPS:
        paths[f"dense_checkpoint_step_{step:08d}_replay"] = (
            context["audit_dir"]
            / f"dense_checkpoint_step_{step:08d}_physical_integrity_audit_replay.json"
        )
    return paths


def snapshot_upstream_metadata(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: file_identity(path)
        for name, path in sorted(upstream_metadata_paths(context).items())
    }


def _builder_args(
    args: argparse.Namespace,
    *,
    context: dict[str, Any],
    terminal_guard_sha256: str,
    comparison_sha256: str,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "expected_terminal_system_guard_sha256": terminal_guard_sha256,
            "expected_comparison_sha256": comparison_sha256,
        }
    )
    return argparse.Namespace(**values)


def status_payload(
    *,
    status: str,
    detail: str,
    args: argparse.Namespace,
    context: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
    deployment_receipt: dict[str, Any] | None,
    observation: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "updated_at": utc_now(),
        "poll_seconds": args.poll_seconds,
        "timeout_seconds": args.timeout_seconds,
        "runtime": runtime,
        "git": context.get("git") if context else None,
        "waiter_source": context.get("waiter_source") if context else None,
        "builder_source": context.get("builder_source") if context else None,
        "deployment_receipt": deployment_receipt,
        "observation": observation,
        "audit": audit,
        "terminal_status": audit.get("terminal_status") if audit else None,
        "generation_advantage_proven": (
            audit.get("generation_advantage_proven") if audit else False
        ),
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "scope": copy.deepcopy(SCOPE),
    }


def publish(
    path: Path,
    **kwargs: Any,
) -> None:
    write_json_report(path, status_payload(**kwargs))


def run_locked(
    args: argparse.Namespace,
    *,
    require_detached: bool = True,
) -> int:
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("terminal completion waiter timing values must be positive")
    runtime = validate_runtime(require_detached=require_detached)
    context = _canonical_context(args)
    context["status"].parent.mkdir(parents=True, exist_ok=True)
    deployment = prepare_deployment_receipt(
        context["deployment"],
        deployment_payload(args, context=context, runtime=runtime),
    )
    started = time.monotonic()
    while True:
        current = _canonical_context(args)
        if any(
            current[name] != context[name]
            for name in (
                "git",
                "waiter_source",
                "builder_source",
                "training_git",
                "auditor_git",
                "replay_git",
                "auditor_source",
                "replay_verifier_source",
                "metrics_trust_identity",
            )
        ):
            raise ValueError("terminal completion waiter static context changed")
        observation = observe_upstreams(context)
        if observation["state"] == "failed":
            publish(
                context["status"],
                status="failed",
                detail=observation["detail"],
                args=args,
                context=context,
                runtime=runtime,
                deployment_receipt=deployment,
                observation=observation,
            )
            return 1
        if observation["state"] != "ready":
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("terminal completion waiter timed out")
            publish(
                context["status"],
                status="waiting",
                detail=observation["detail"],
                args=args,
                context=context,
                runtime=runtime,
                deployment_receipt=deployment,
                observation=observation,
            )
            time.sleep(args.poll_seconds)
            continue

        terminal_guard_identity = file_identity(context["terminal_guard"])
        comparison_identity = file_identity(context["comparison"])
        metadata_sources_before = snapshot_upstream_metadata(context)
        publish(
            context["status"],
            status="running",
            detail="physically_replaying_terminal_completion_evidence",
            args=args,
            context=context,
            runtime=runtime,
            deployment_receipt=deployment,
            observation=observation,
        )
        report = builder.build_audit(
            _builder_args(
                args,
                context=context,
                terminal_guard_sha256=terminal_guard_identity["sha256"],
                comparison_sha256=comparison_identity["sha256"],
            )
        )
        metadata_sources_after = snapshot_upstream_metadata(context)
        if metadata_sources_after != metadata_sources_before:
            raise ValueError(
                "terminal completion metadata sources changed during replay"
            )
        audit_identity = prepare_manifest(
            context["output"],
            report,
            resume=context["output"].is_file(),
            overwrite=False,
        )
        publish(
            context["status"],
            status="pass",
            detail="terminal_completion_audit_revalidated",
            args=args,
            context=context,
            runtime=runtime,
            deployment_receipt=deployment,
            observation=observation,
            audit={
                "identity": audit_identity,
                "terminal_status": report["terminal_status"],
                "terminal_decision": report["terminal_decision"],
                "generation_advantage_proven": report["generation_advantage_proven"],
            },
        )
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for all exact terminal quality-bridge sources, then perform a "
            "CPU-only, permanently non-authorizing physical completion replay."
        )
    )
    builder.add_common_arguments(parser, include_dynamic_hashes=False)
    parser.add_argument(
        "--expected-output-dir-name",
        default="terminal_completion_audit_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--expected-waiter-source-sha256", required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = reject_symlink_chain(args.lock, name="terminal completion waiter lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    status_path = args.status_output.resolve()
    try:
        with lock.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise LockBusy(
                        "another terminal completion waiter owns the lock"
                    ) from error
            raise SystemExit(run_locked(args))
    except LockBusy:
        raise
    except Exception as error:
        try:
            context = _canonical_context(args)
            runtime = validate_runtime(require_detached=False)
            deployment = (
                file_identity(context["deployment"])
                if context["deployment"].is_file()
                else None
            )
            publish(
                status_path,
                status="failed",
                detail=f"{type(error).__name__}:{error}",
                args=args,
                context=context,
                runtime=runtime,
                deployment_receipt=deployment,
                error=error,
            )
        except Exception as status_error:  # noqa: BLE001 - fail-closed boundary
            print(f"unable to publish terminal completion failure: {status_error}")
        raise


if __name__ == "__main__":
    main()
