from __future__ import annotations

import argparse
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    import build_generation_terminal_runtime_strict_conjunct as builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import build_generation_terminal_runtime_strict_conjunct as builder


SCHEMA_VERSION = 1
ROLE = "generation_terminal_runtime_strict_conjunct_waiter"
DEPLOYMENT_ROLE = "generation_terminal_runtime_strict_conjunct_deployment"
SCOPE = {
    "cpu_only": True,
    "diagnostic_non_authorizing": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "source_waiter_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "upstream_decisions_modified": False,
}
RUNTIME_POLICY = {
    "detached_parent_pid": 1,
    "cuda_visible_devices": "",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "minimum_nice": 10,
    "ionice": "idle",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(project: Path) -> dict[str, Any]:
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


def _runtime_identity(*, require_detached: bool) -> dict[str, Any]:
    pid = os.getpid()
    parent_pid = os.getppid()
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ionice = subprocess.run(
        ["ionice", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    proc = Path("/proc") / str(pid)
    stat = proc.joinpath("stat").read_text(encoding="utf-8").split()
    runtime = {
        "pid": pid,
        "parent_pid": parent_pid,
        "start_ticks": int(stat[21]),
        "hostname": socket.gethostname(),
        "cwd": os.readlink(proc / "cwd"),
        "cmdline": (
            proc.joinpath("cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
            .strip()
        ),
        "python_executable": os.path.realpath(os.sys.executable),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "nice": nice,
        "ionice": ionice,
    }
    if (
        (require_detached and parent_pid != RUNTIME_POLICY["detached_parent_pid"])
        or runtime["cuda_visible_devices"] != RUNTIME_POLICY["cuda_visible_devices"]
        or runtime["omp_num_threads"] != RUNTIME_POLICY["omp_num_threads"]
        or runtime["mkl_num_threads"] != RUNTIME_POLICY["mkl_num_threads"]
        or nice < RUNTIME_POLICY["minimum_nice"]
        or ionice != RUNTIME_POLICY["ionice"]
    ):
        raise ValueError("terminal runtime conjunct waiter runtime policy differs")
    return runtime


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _terminal_state(status_path: Path, output_path: Path) -> dict[str, Any]:
    if not status_path.is_file():
        return {"state": "waiting", "detail": "waiting_for_terminal_completion_status"}
    report = read_json_object(status_path, name="terminal completion waiter status")
    status = report.get("status")
    if status == "failed":
        return {
            "state": "failed",
            "detail": "terminal_completion_waiter_failed",
            "source_detail": report.get("detail"),
        }
    if status == "pass" and output_path.is_file():
        return {"state": "ready", "detail": "terminal_completion_audit_ready"}
    if status not in {"waiting", "running", "pass"}:
        raise ValueError("terminal completion waiter state differs")
    return {"state": "waiting", "detail": "waiting_for_terminal_completion_audit"}


def _runtime_state(status_path: Path, output_path: Path) -> dict[str, Any]:
    if not status_path.is_file():
        return {"state": "waiting", "detail": "waiting_for_runtime_strict_status"}
    report = read_json_object(status_path, name="runtime strict comparison waiter status")
    status = report.get("status")
    if status == "failed":
        return {
            "state": "failed",
            "detail": "runtime_strict_comparison_waiter_failed",
            "source_detail": report.get("detail"),
        }
    if status == "pass" and output_path.is_file():
        return {"state": "ready", "detail": "runtime_strict_comparison_ready"}
    if status not in {"waiting", "pass"}:
        raise ValueError("runtime strict comparison waiter state differs")
    return {"state": "waiting", "detail": "waiting_for_runtime_strict_comparison"}


def _publish(
    path: Path,
    *,
    status: str,
    detail: str,
    context: Mapping[str, Any],
    deployment: Mapping[str, Any],
    sources: Mapping[str, Any] | None = None,
    conjunct: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    write_json_report(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "role": ROLE,
            "status": status,
            "detail": detail,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "updated_at": _utc_now(),
            "git": context["git"],
            "runtime": context["runtime"],
            "waiter_source": context["waiter_source"],
            "builder_source": context["builder_source"],
            "deployment_receipt": dict(deployment),
            "sources": dict(sources) if sources is not None else None,
            "conjunct": dict(conjunct) if conjunct is not None else None,
            "generation_advantage_proven": (
                bool(conjunct.get("generation_advantage_proven"))
                if conjunct is not None
                else False
            ),
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error) if error is not None else None,
            "scope": SCOPE,
        },
    )


def _context(args: argparse.Namespace, *, require_detached: bool) -> dict[str, Any]:
    project = reject_symlink_chain(args.project, name="conjunct project").resolve()
    git = _git_identity(project)
    expected = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    if git != expected:
        raise ValueError("terminal runtime conjunct control Git identity differs")
    waiter_source = file_identity(
        project / "scripts/wait_generation_terminal_runtime_strict_conjunct.py"
    )
    builder_source = file_identity(
        project / "scripts/build_generation_terminal_runtime_strict_conjunct.py"
    )
    if waiter_source["sha256"] != args.expected_waiter_source_sha256:
        raise ValueError("terminal runtime conjunct waiter source SHA256 differs")
    if builder_source["sha256"] != args.expected_builder_source_sha256:
        raise ValueError("terminal runtime conjunct builder source SHA256 differs")
    root = reject_symlink_chain(
        args.quality_output_root,
        name="quality output root",
    ).resolve()
    output_root = reject_symlink_chain(args.output_root, name="conjunct output root").resolve()
    paths = {
        "terminal_status": reject_symlink_chain(
            args.terminal_status, name="terminal completion status"
        ).resolve(),
        "terminal_audit": reject_symlink_chain(
            args.terminal_audit, name="terminal completion audit"
        ).resolve(),
        "runtime_status": reject_symlink_chain(
            args.runtime_status, name="runtime strict comparison status"
        ).resolve(),
        "runtime_comparison": reject_symlink_chain(
            args.runtime_comparison, name="runtime strict comparison"
        ).resolve(),
        "output": reject_symlink_chain(args.output, name="conjunct output").resolve(),
        "status": reject_symlink_chain(args.status_output, name="conjunct status").resolve(),
        "pid": reject_symlink_chain(args.pid_file, name="conjunct PID").resolve(),
        "deployment": reject_symlink_chain(
            args.deployment_receipt_output, name="conjunct deployment receipt"
        ).resolve(),
    }
    if output_root == root or not _within(output_root, root):
        raise ValueError("terminal runtime conjunct output root is outside quality root")
    for name in (
        "terminal_status",
        "terminal_audit",
        "runtime_status",
        "runtime_comparison",
    ):
        if not _within(paths[name], root):
            raise ValueError(f"terminal runtime conjunct {name} is outside quality root")
        if _within(paths[name], output_root):
            raise ValueError(f"terminal runtime conjunct {name} overlaps output root")
    for name in ("output", "status", "pid", "deployment"):
        if not _within(paths[name], output_root):
            raise ValueError(f"terminal runtime conjunct {name} is outside output root")
    return {
        "project": project,
        "quality_root": root,
        "output_root": output_root,
        "paths": paths,
        "git": git,
        "runtime": _runtime_identity(require_detached=require_detached),
        "waiter_source": waiter_source,
        "builder_source": builder_source,
    }


def _deployment_payload(args: argparse.Namespace, context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "control_git": context["git"],
        "runtime_policy": dict(RUNTIME_POLICY),
        "sources": {
            "waiter": context["waiter_source"],
            "builder": context["builder_source"],
        },
        "expected_training": {
            "revision": args.expected_training_revision,
            "branch": args.expected_training_branch,
        },
        "targets": {
            name: path.as_posix() for name, path in context["paths"].items()
        },
        "scope": SCOPE,
    }


def _prepare_deployment(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_file():
        existing = read_json_object(path, name="conjunct deployment receipt")
        if existing != dict(payload):
            raise ValueError("terminal runtime conjunct deployment receipt differs")
    else:
        write_json_report(path, dict(payload))
    return file_identity(path)


def _write_pid(path: Path, context: Mapping[str, Any]) -> None:
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": ROLE,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "control_revision": context["git"]["revision"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        report = read_json_object(path, name="conjunct PID")
    except ValueError:
        return
    if report.get("role") == ROLE and int(report.get("pid", -1)) == os.getpid():
        path.unlink()


def run_waiter(args: argparse.Namespace, *, require_detached: bool = True) -> int:
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("terminal runtime conjunct timing values must be positive")
    context = _context(args, require_detached=require_detached)
    output_root = context["output_root"]
    paths = context["paths"]
    output_root.mkdir(parents=True, exist_ok=True)
    deployment = _prepare_deployment(
        paths["deployment"],
        _deployment_payload(args, context),
    )
    _write_pid(paths["pid"], context)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while True:
            terminal = _terminal_state(paths["terminal_status"], paths["terminal_audit"])
            runtime = _runtime_state(paths["runtime_status"], paths["runtime_comparison"])
            sources = {"terminal": terminal, "runtime": runtime}
            if "failed" in {terminal["state"], runtime["state"]}:
                detail = terminal["detail"] if terminal["state"] == "failed" else runtime["detail"]
                _publish(
                    paths["status"],
                    status="failed",
                    detail=detail,
                    context=context,
                    deployment=deployment,
                    sources=sources,
                )
                return 1
            if terminal["state"] == runtime["state"] == "ready":
                break
            _publish(
                paths["status"],
                status="waiting",
                detail=(runtime["detail"] if terminal["state"] == "ready" else terminal["detail"]),
                context=context,
                deployment=deployment,
                sources=sources,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("terminal runtime conjunct waiter timed out")
            time.sleep(min(args.poll_seconds, remaining))

        identities = {
            name: file_identity(paths[name])
            for name in (
                "terminal_status",
                "terminal_audit",
                "runtime_status",
                "runtime_comparison",
            )
        }
        report = builder.build_conjunct(
            terminal_status_path=paths["terminal_status"],
            expected_terminal_status_sha256=identities["terminal_status"]["sha256"],
            terminal_audit_path=paths["terminal_audit"],
            expected_terminal_audit_sha256=identities["terminal_audit"]["sha256"],
            runtime_status_path=paths["runtime_status"],
            expected_runtime_status_sha256=identities["runtime_status"]["sha256"],
            runtime_comparison_path=paths["runtime_comparison"],
            expected_runtime_comparison_sha256=identities["runtime_comparison"]["sha256"],
            expected_training_revision=args.expected_training_revision,
            expected_training_branch=args.expected_training_branch,
        )
        if any(file_identity(paths[name]) != identity for name, identity in identities.items()):
            raise ValueError("terminal runtime conjunct sources changed during replay")
        output_identity = prepare_manifest(
            paths["output"],
            report,
            resume=paths["output"].is_file(),
            overwrite=False,
        )
        _publish(
            paths["status"],
            status="pass",
            detail=report["detail"],
            context=context,
            deployment=deployment,
            sources=identities,
            conjunct={
                "identity": output_identity,
                "terminal_status": report["terminal_status"],
                "terminal_decision": report["terminal_decision"],
                "generation_advantage_proven": report["generation_advantage_proven"],
                "runtime_strict_comparator_passed": True,
            },
        )
        return 0
    except Exception as error:
        _publish(
            paths["status"],
            status="failed",
            detail=f"{type(error).__name__}: {error}",
            context=context,
            deployment=deployment,
            error=error,
        )
        raise
    finally:
        _remove_pid(paths["pid"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for both terminal completion and the strict runtime comparator, "
            "then publish a CPU-only non-authorizing conjunct."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--terminal-status", type=Path, required=True)
    parser.add_argument("--terminal-audit", type=Path, required=True)
    parser.add_argument("--runtime-status", type=Path, required=True)
    parser.add_argument("--runtime-comparison", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-waiter-source-sha256", required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with exclusive_output_lock(args.output_root, role=ROLE):
            return run_waiter(args)
    except OutputLockError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
