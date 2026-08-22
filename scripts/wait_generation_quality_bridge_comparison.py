from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ROLE = "generation_quality_bridge_comparison_waiter"
DEPLOYMENT_SCHEMA_VERSION = 1
DEPLOYMENT_ROLE = "generation_quality_bridge_comparison_waiter_deployment"
TERMINAL_WAITER_ROLE = "generation_terminal_system_claim_guard_waiter"
TERMINAL_GUARD_ROLE = "generation_terminal_system_claim_guard"
COMPARISON_ROLE = "stability_full_data_quality_bridge_comparison"
OFFICIAL_RELATED_PATH = Path(
    "artifacts/reports/baselines/official_related_methods_2026-07-11_final/"
    "official_related_methods_table.json"
)

TERMINAL_WAITER_BOUNDARY = {
    "cpu_only_evidence_binding_allowed": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
}

COMPARISON_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
}

SCOPE = {
    "diagnostic_non_authorizing": True,
    "cpu_only": True,
    "gpu_required": False,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "checkpoint_physical_hash_replay_allowed": True,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
}


class WaiterLockError(RuntimeError):
    """Another process already owns the canonical comparison waiter lock."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def reject_symlink_chain(path: str | Path, *, name: str) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = absolute
    while True:
        if current.is_symlink():
            raise ValueError(f"{name} path must not contain a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    return absolute


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path, *, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    resolved = source.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = reject_symlink_chain(path, name="comparison waiter JSON output")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_exact_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(payload)
    if path.is_file():
        if read_json(path, name="comparison waiter deployment receipt") != expected:
            raise ValueError("existing comparison waiter deployment receipt differs")
    elif path.exists():
        raise ValueError("comparison waiter deployment receipt is not a file")
    else:
        write_json(path, expected)
    return file_identity(path, name="comparison waiter deployment receipt")


def git_identity(project: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def process_start_ticks(pid: int) -> int | None:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return None
    try:
        return int(stat.read_text(encoding="utf-8").split()[21])
    except (OSError, ValueError, IndexError):
        return None


def validate_low_priority_cpu_runtime(
    *, reparent_timeout_seconds: float = 30.0
) -> dict[str, Any]:
    environment = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }
    if environment["CUDA_VISIBLE_DEVICES"] not in {"", "-1"}:
        raise ValueError("comparison waiter must run with CUDA hidden")
    if environment["OMP_NUM_THREADS"] != "1" or environment["MKL_NUM_THREADS"] != "1":
        raise ValueError(
            "comparison waiter must use OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1"
        )

    parent_pid = os.getppid()
    nice_value: int | None = None
    ionice = None
    if os.name != "nt":
        deadline = time.monotonic() + reparent_timeout_seconds
        while parent_pid != 1 and time.monotonic() < deadline:
            time.sleep(0.1)
            parent_pid = os.getppid()
        if parent_pid != 1:
            raise ValueError("comparison waiter must be detached with parent PID 1")
        nice_value = os.getpriority(os.PRIO_PROCESS, 0)
        if nice_value < 10:
            raise ValueError("comparison waiter nice priority must be at least 10")
        ionice = subprocess.run(
            ["ionice", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not ionice.startswith("idle"):
            raise ValueError("comparison waiter must use idle I/O priority")
    return {
        "pid": os.getpid(),
        "parent_pid": parent_pid,
        "start_ticks": process_start_ticks(os.getpid()),
        "cwd": Path.cwd().resolve().as_posix(),
        "argv": list(sys.argv),
        "environment": environment,
        "nice": nice_value,
        "ionice": ionice,
    }


def _lock_owner(handle, *, lock_path: Path) -> dict[str, Any]:
    owner = {
        "schema_version": 1,
        "role": ROLE,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "lock": lock_path.as_posix(),
        "acquired_at": utc_now(),
    }
    encoded = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
    handle.seek(0)
    handle.truncate(0)
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
    return owner


@contextmanager
def exclusive_waiter_lock(path: Path) -> Iterator[dict[str, Any]]:
    lock_path = reject_symlink_chain(path, name="comparison waiter lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(handle.fileno()).st_size < 1:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise WaiterLockError(
                f"another comparison waiter owns the lock: {lock_path}"
            ) from error
        acquired = True
        yield _lock_owner(handle, lock_path=lock_path)
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the canonical terminal system claim guard, then build one "
            "CPU-only, permanently non-authorizing quality-bridge comparison."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--terminal-waiter-status", type=Path, required=True)
    parser.add_argument("--official-related", type=Path, required=True)
    parser.add_argument("--expected-official-related-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--expected-project-revision", required=True)
    parser.add_argument("--expected-project-tree", required=True)
    parser.add_argument("--expected-project-branch", required=True)
    parser.add_argument("--expected-waiter-source-sha256", required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for name, value in (
        ("official related table", args.expected_official_related_sha256),
        ("waiter source", args.expected_waiter_source_sha256),
        ("builder source", args.expected_builder_source_sha256),
    ):
        if not is_sha256(value):
            parser.error(f"{name} SHA256 is invalid")
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        parser.error("comparison waiter timing values must be positive")
    return args


def static_context(args: argparse.Namespace) -> dict[str, Any]:
    project = reject_symlink_chain(
        args.project, name="comparison waiter project"
    ).resolve()
    quality_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality bridge output root",
    ).resolve()
    terminal_guard = reject_symlink_chain(
        args.terminal_system_guard,
        name="terminal system claim guard",
    ).resolve()
    terminal_status = reject_symlink_chain(
        args.terminal_waiter_status,
        name="terminal system guard waiter status",
    ).resolve()
    official = reject_symlink_chain(
        args.official_related,
        name="official related-method table",
    ).resolve()
    output_dir = reject_symlink_chain(
        args.output_dir,
        name="quality bridge comparison output directory",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="quality bridge comparison waiter status",
    ).resolve()
    deployment_output = reject_symlink_chain(
        args.deployment_receipt_output,
        name="quality bridge comparison deployment receipt",
    ).resolve()
    lock = reject_symlink_chain(
        args.lock, name="quality bridge comparison lock"
    ).resolve()
    builder = (
        project / "scripts" / "build_generation_quality_bridge_comparison.py"
    ).resolve()
    waiter_source = Path(__file__).resolve()

    expected_output_dir = quality_root / "reports" / "quality_bridge_comparison_v1"
    expected_terminal_root = quality_root / "reports" / "terminal_system_claim_guard_v1"
    if (
        output_dir != expected_output_dir
        or status_output != output_dir / "waiter_status.json"
        or deployment_output != output_dir / "deployment_receipt.json"
        or lock != output_dir / "waiter.lock"
        or terminal_guard != expected_terminal_root / "terminal_system_claim_guard.json"
        or terminal_status != expected_terminal_root / "waiter_status.json"
        or official != (project / OFFICIAL_RELATED_PATH).resolve()
        or waiter_source != project / "scripts" / waiter_source.name
        or not project.is_dir()
        or not quality_root.is_dir()
    ):
        raise ValueError("comparison waiter canonical path contract differs")

    expected_git = {
        "revision": args.expected_project_revision,
        "tree": args.expected_project_tree,
        "branch": args.expected_project_branch,
        "tracked_dirty": False,
    }
    observed_git = git_identity(project)
    if observed_git != expected_git:
        raise ValueError(f"comparison waiter Git identity differs: {observed_git}")

    waiter_identity = file_identity(waiter_source, name="comparison waiter source")
    builder_identity = file_identity(builder, name="quality bridge comparison builder")
    official_identity = file_identity(official, name="official related-method table")
    for label, actual, expected in (
        (
            "waiter source",
            waiter_identity["sha256"],
            args.expected_waiter_source_sha256,
        ),
        (
            "builder source",
            builder_identity["sha256"],
            args.expected_builder_source_sha256,
        ),
        (
            "official related-method table",
            official_identity["sha256"],
            args.expected_official_related_sha256,
        ),
    ):
        if actual != expected:
            raise ValueError(f"comparison waiter {label} SHA256 differs")

    outputs = {
        "json": output_dir / "quality_bridge_comparison.json",
        "markdown": output_dir / "quality_bridge_comparison.md",
        "csv": output_dir / "quality_bridge_comparison.csv",
    }
    return {
        "project": project,
        "quality_root": quality_root,
        "terminal_guard": terminal_guard,
        "terminal_status": terminal_status,
        "official": official,
        "output_dir": output_dir,
        "status_output": status_output,
        "deployment_output": deployment_output,
        "lock": lock,
        "builder": builder,
        "git": observed_git,
        "waiter_source": waiter_identity,
        "builder_source": builder_identity,
        "official_source": official_identity,
        "outputs": outputs,
    }


def deployment_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "project_git": context["git"],
        "sources": {
            "waiter": context["waiter_source"],
            "builder": context["builder_source"],
            "official_related_methods": context["official_source"],
        },
        "targets": {
            "quality_output_root": context["quality_root"].as_posix(),
            "terminal_system_guard": context["terminal_guard"].as_posix(),
            "terminal_waiter_status": context["terminal_status"].as_posix(),
            "comparison_output_dir": context["output_dir"].as_posix(),
            "waiter_status": context["status_output"].as_posix(),
            "waiter_lock": context["lock"].as_posix(),
        },
        "terminal_guard_statuses_accepted": ["pass", "hold"],
        "operational_completion_status": "pass",
        "scope": SCOPE,
    }


def observe_terminal(context: Mapping[str, Any]) -> dict[str, Any]:
    status_path: Path = context["terminal_status"]
    guard_path: Path = context["terminal_guard"]
    if not status_path.is_file():
        if guard_path.exists():
            raise ValueError(
                "terminal guard exists without its canonical waiter status"
            )
        return {"state": "waiting", "detail": "waiting_for_terminal_guard_waiter"}
    status_identity = file_identity(status_path, name="terminal guard waiter status")
    report = read_json(status_path, name="terminal guard waiter status")
    status = report.get("status")
    expected = report.get("expected")
    if (
        report.get("schema_version") != 1
        or report.get("role") != TERMINAL_WAITER_ROLE
        or report.get("authorization_boundary") != TERMINAL_WAITER_BOUNDARY
        or status not in {"waiting", "completed", "failed"}
        or not isinstance(expected, Mapping)
        or expected.get("output") != guard_path.as_posix()
    ):
        raise ValueError("terminal guard waiter status contract differs")
    if status == "failed":
        return {
            "state": "failed",
            "detail": str(report.get("detail", "terminal_guard_waiter_failed")),
            "status": status_identity,
        }
    if status == "waiting":
        if guard_path.exists():
            raise ValueError("terminal guard exists while its waiter is still waiting")
        return {
            "state": "waiting",
            "detail": str(report.get("detail", "waiting_for_terminal_guard")),
            "status": status_identity,
        }

    if not guard_path.is_file():
        raise ValueError("terminal guard waiter completed without its guard output")
    guard_identity = file_identity(guard_path, name="terminal system claim guard")
    guard = read_json(guard_path, name="terminal system claim guard")
    terminal_status = guard.get("status")
    if (
        report.get("guard") != guard_identity
        or report.get("guard_status") != terminal_status
        or guard.get("schema_version") != 1
        or guard.get("role") != TERMINAL_GUARD_ROLE
        or terminal_status not in {"pass", "hold"}
    ):
        raise ValueError("terminal system claim guard binding differs")
    return {
        "state": "ready",
        "detail": "terminal_system_claim_guard_ready",
        "status": status_identity,
        "guard": guard_identity,
        "terminal_status": terminal_status,
        "terminal_decision": guard.get("decision"),
    }


def build_command(
    context: Mapping[str, Any],
    *,
    guard_identity: Mapping[str, Any],
) -> list[str]:
    return [
        sys.executable,
        str(context["builder"]),
        "--terminal-system-guard",
        str(context["terminal_guard"]),
        "--expected-terminal-system-guard-sha256",
        str(guard_identity["sha256"]),
        "--official-related",
        str(context["official"]),
        "--expected-official-related-sha256",
        str(context["official_source"]["sha256"]),
        "--output-dir",
        str(context["output_dir"]),
    ]


def run_builder(
    context: Mapping[str, Any],
    *,
    guard_identity: Mapping[str, Any],
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    command = build_command(context, guard_identity=guard_identity)
    completed = subprocess.run(
        command,
        cwd=context["project"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"quality bridge comparison builder failed: {detail}")


def validate_comparison_outputs(
    context: Mapping[str, Any],
    *,
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    identities = {
        name: file_identity(path, name=f"quality bridge comparison {name}")
        for name, path in context["outputs"].items()
    }
    report = read_json(
        context["outputs"]["json"],
        name="quality bridge comparison report",
    )
    sources = report.get("source_reports")
    policy = report.get("comparison_policy")
    if (
        report.get("schema_version") != 1
        or report.get("role") != COMPARISON_ROLE
        or report.get("status") != terminal["terminal_status"]
        or report.get("authorization_boundary") != COMPARISON_BOUNDARY
        or not isinstance(sources, Mapping)
        or sources.get("terminal_system_claim_guard") != terminal["guard"]
        or sources.get("official_related_methods") != context["official_source"]
        or not isinstance(policy, Mapping)
        or policy.get("primary_direct_tier") != "matched_training_direct"
        or policy.get("external_context_tier") != "official_pretrained_contextual"
        or policy.get("cross_tier_numeric_ranking_allowed") is not False
        or policy.get("compute_matched_claim_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or policy.get("sota_claim_allowed") is not False
    ):
        raise ValueError("quality bridge comparison output contract differs")
    direct = report.get("matched_training_rows")
    contextual = report.get("official_context_rows")
    if (
        not isinstance(direct, list)
        or len(direct) != 2
        or {row.get("comparison_tier") for row in direct if isinstance(row, Mapping)}
        != {"matched_training_direct"}
        or not isinstance(contextual, list)
        or len(contextual) != 3
        or {
            row.get("comparison_tier") for row in contextual if isinstance(row, Mapping)
        }
        != {"official_pretrained_contextual"}
    ):
        raise ValueError("quality bridge comparison row tiers differ")
    return identities


def status_payload(
    *,
    status: str,
    detail: str,
    args: argparse.Namespace,
    started_at: str,
    runtime: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    deployment_receipt: Mapping[str, Any] | None,
    terminal: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    expected = None
    if context is not None:
        expected = {
            "project_git": context["git"],
            "waiter_source": context["waiter_source"],
            "builder_source": context["builder_source"],
            "official_related_methods": context["official_source"],
            "terminal_system_guard": context["terminal_guard"].as_posix(),
            "terminal_waiter_status": context["terminal_status"].as_posix(),
            "output_dir": context["output_dir"].as_posix(),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": utc_now(),
        "poll_seconds": args.poll_seconds,
        "timeout_seconds": args.timeout_seconds,
        "runtime": dict(runtime) if runtime is not None else None,
        "expected": expected,
        "deployment_receipt": (
            dict(deployment_receipt) if deployment_receipt is not None else None
        ),
        "terminal": dict(terminal) if terminal is not None else None,
        "terminal_status": (
            terminal.get("terminal_status") if terminal is not None else None
        ),
        "comparison": dict(comparison) if comparison is not None else None,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "scope": SCOPE,
    }


def publish(
    path: Path,
    *,
    status: str,
    detail: str,
    args: argparse.Namespace,
    started_at: str,
    runtime: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    deployment_receipt: Mapping[str, Any] | None,
    terminal: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    write_json(
        path,
        status_payload(
            status=status,
            detail=detail,
            args=args,
            started_at=started_at,
            runtime=runtime,
            context=context,
            deployment_receipt=deployment_receipt,
            terminal=terminal,
            comparison=comparison,
            error=error,
        ),
    )


def run_locked(
    args: argparse.Namespace,
    *,
    runtime: Mapping[str, Any] | None,
) -> int:
    started_at = utc_now()
    context = static_context(args)
    receipt = prepare_exact_json(
        context["deployment_output"],
        deployment_payload(context),
    )
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if static_context(args) != context:
            raise ValueError("comparison waiter static context changed")
        terminal = observe_terminal(context)
        if terminal["state"] == "failed":
            publish(
                context["status_output"],
                status="failed",
                detail="terminal_system_claim_guard_waiter_failed",
                args=args,
                started_at=started_at,
                runtime=runtime,
                context=context,
                deployment_receipt=receipt,
                terminal=terminal,
            )
            return 1
        if terminal["state"] == "waiting":
            if any(path.exists() for path in context["outputs"].values()):
                raise ValueError(
                    "comparison output exists before terminal guard completion"
                )
            publish(
                context["status_output"],
                status="waiting",
                detail=terminal["detail"],
                args=args,
                started_at=started_at,
                runtime=runtime,
                context=context,
                deployment_receipt=receipt,
                terminal=terminal,
            )
            if args.once:
                return 0
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                publish(
                    context["status_output"],
                    status="failed",
                    detail="comparison_waiter_timeout",
                    args=args,
                    started_at=started_at,
                    runtime=runtime,
                    context=context,
                    deployment_receipt=receipt,
                    terminal=terminal,
                )
                return 1
            time.sleep(min(args.poll_seconds, remaining))
            continue

        publish(
            context["status_output"],
            status="running",
            detail="building_source_bound_quality_bridge_comparison",
            args=args,
            started_at=started_at,
            runtime=runtime,
            context=context,
            deployment_receipt=receipt,
            terminal=terminal,
        )
        run_builder(context, guard_identity=terminal["guard"])
        if static_context(args) != context:
            raise ValueError("comparison waiter static context changed during build")
        terminal_after = observe_terminal(context)
        if terminal_after != terminal:
            raise ValueError("terminal guard evidence changed during comparison build")
        comparison = validate_comparison_outputs(
            context,
            terminal=terminal,
        )
        publish(
            context["status_output"],
            status="pass",
            detail="terminal_quality_bridge_comparison_revalidated",
            args=args,
            started_at=started_at,
            runtime=runtime,
            context=context,
            deployment_receipt=receipt,
            terminal=terminal,
            comparison=comparison,
        )
        return 0


def safe_status_path(args: argparse.Namespace) -> Path | None:
    try:
        quality_root = reject_symlink_chain(
            args.quality_output_root,
            name="quality bridge output root",
        ).resolve()
        output_dir = reject_symlink_chain(
            args.output_dir,
            name="quality bridge comparison output directory",
        ).resolve()
        status = reject_symlink_chain(
            args.status_output,
            name="quality bridge comparison waiter status",
        ).resolve()
        if (
            output_dir != quality_root / "reports" / "quality_bridge_comparison_v1"
            or status != output_dir / "waiter_status.json"
        ):
            return None
        return status
    except (OSError, ValueError):
        return None


def run_waiter(
    args: argparse.Namespace,
    *,
    enforce_runtime: bool = True,
) -> int:
    started_at = utc_now()
    try:
        with exclusive_waiter_lock(args.lock):
            runtime = None
            try:
                if enforce_runtime:
                    runtime = validate_low_priority_cpu_runtime()
                return run_locked(args, runtime=runtime)
            except Exception as error:  # noqa: BLE001 - terminal fail-closed boundary
                path = safe_status_path(args)
                context = None
                receipt = None
                if path is not None:
                    try:
                        context = static_context(args)
                        if context["deployment_output"].is_file():
                            receipt = file_identity(
                                context["deployment_output"],
                                name="comparison waiter deployment receipt",
                            )
                    except (
                        OSError,
                        TypeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as context_error:
                        print(
                            f"failure-status context unavailable: {context_error}",
                            file=sys.stderr,
                        )
                    publish(
                        path,
                        status="failed",
                        detail=f"{type(error).__name__}: {error}",
                        args=args,
                        started_at=started_at,
                        runtime=runtime,
                        context=context,
                        deployment_receipt=receipt,
                        error=error,
                    )
                print(f"{type(error).__name__}: {error}", file=sys.stderr)
                return 1
    except WaiterLockError as error:
        print(str(error), file=sys.stderr)
        return 75


def main() -> int:
    return run_waiter(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
