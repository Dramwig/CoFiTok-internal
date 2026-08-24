from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX-only supervisor
    fcntl = None

from cofitok.generation.exposure_semantic_trajectory import (
    CLAIM_BOUNDARY,
    METHOD_RUN_DIRS,
    OUTPUT_ROOT,
    REPORT_ROLE,
    SUPERVISOR_ROLE,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import file_sha256, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unique non-authorizing CPU exposure-trajectory replay."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--runbook", required=True)
    parser.add_argument("--expected-runbook-sha256", required=True)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--python", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def _full_git(project: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=project, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=project, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=project, text=True
    ).strip()
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": bool(status),
    }


def _verify_checkout(
    project: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
) -> dict[str, Any]:
    actual = _full_git(project)
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if actual != expected:
        raise ValueError("exposure-trajectory checkout identity differs")
    return actual


def _runtime_contract() -> dict[str, Any]:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    omp = os.environ.get("OMP_NUM_THREADS")
    mkl = os.environ.get("MKL_NUM_THREADS")
    if cuda not in (None, "") or omp != "1" or mkl != "1":
        raise ValueError("supervisor must be CUDA-hidden with OMP/MKL=1")
    return {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "cuda_visible_devices": "" if cuda is None else cuda,
        "omp_num_threads": omp,
        "mkl_num_threads": mkl,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }


def _completed_count(output_root: Path) -> int:
    root = output_root / "reports" / "exposure_semantic_trajectory_v1"
    count = 0
    for method in METHOD_RUN_DIRS:
        method_root = root / method
        if method_root.is_dir():
            count += sum(
                (path / "conditioning_sensitivity_report.json").is_file()
                for path in method_root.iterdir()
                if path.is_dir()
            )
    return count


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: Path,
    runtime: Mapping[str, Any],
    sources: Mapping[str, Any],
    child_pid: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": SUPERVISOR_ROLE,
        "status": status,
        "detail": detail,
        "project": project.as_posix(),
        "output_root": output_root.as_posix(),
        "runtime": dict(runtime),
        "child_pid": child_pid,
        "completed_sensitivity_reports": _completed_count(output_root),
        "expected_sensitivity_reports": 6,
        "sources": dict(sources),
        "error": error,
        "claim_boundary": CLAIM_BOUNDARY,
        "generation_advantage_proven": False,
    }


def _verify_authorization(
    *,
    project: Path,
    python: Path,
    authorization: Path,
    expected_sha256: str,
    revision: str,
    tree: str,
    branch: str,
    output_root: Path,
) -> dict[str, Any]:
    if file_sha256(authorization) != expected_sha256:
        raise ValueError("trajectory authorization SHA256 differs")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": f"{project.as_posix()}:{(project / 'src').as_posix()}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run(
        [
            python.as_posix(),
            "scripts/verify_generation_exposure_semantic_trajectory_execution_authorization.py",
            "--authorization",
            authorization.as_posix(),
            "--expected-authorization-sha256",
            expected_sha256,
            "--expected-revision",
            revision,
            "--expected-tree",
            tree,
            "--expected-branch",
            branch,
            "--expected-output-root",
            output_root.as_posix(),
        ],
        cwd=project,
        env=environment,
        check=True,
    )
    return read_json_object(authorization, name="trajectory authorization")


def main() -> int:
    args = parse_args()
    if fcntl is None:
        raise RuntimeError("trajectory supervisor requires POSIX locking")
    if args.poll_seconds < 30.0:
        raise ValueError("trajectory supervisor poll interval is too short")
    runtime = _runtime_contract()
    project = reject_symlink_chain(args.project, name="trajectory project").resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="trajectory output root",
    ).resolve()
    if output_root.as_posix() != OUTPUT_ROOT:
        raise ValueError("trajectory output root differs")
    authorization = reject_symlink_chain(
        args.authorization,
        name="trajectory authorization",
    ).resolve()
    runbook = reject_symlink_chain(args.runbook, name="trajectory runbook").resolve()
    python = reject_symlink_chain(args.python, name="trajectory Python").resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="trajectory supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(args.pid_file, name="trajectory PID file").resolve()
    lock_path = reject_symlink_chain(args.lock, name="trajectory supervisor lock").resolve()
    for directory in (status_output.parent, pid_file.parent, lock_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another trajectory supervisor owns the lock") from error
    _verify_checkout(
        project,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
    )
    if not python.is_file() or not runbook.is_file():
        raise FileNotFoundError("trajectory Python or runbook is missing")
    if file_sha256(runbook) != args.expected_runbook_sha256:
        raise ValueError("trajectory runbook SHA256 differs")
    authorization_payload = _verify_authorization(
        project=project,
        python=python,
        authorization=authorization,
        expected_sha256=args.expected_authorization_sha256,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
        output_root=output_root,
    )
    source_reports = authorization_payload.get("source_reports")
    if not isinstance(source_reports, Mapping):
        raise ValueError("trajectory authorization sources are missing")
    sources = {
        "authorization": file_identity(authorization),
        "preparation": source_reports["preparation"],
        "runbook": file_identity(runbook),
    }
    write_json_report(
        pid_file,
        {
            "schema_version": 1,
            "role": f"{SUPERVISOR_ROLE}_pid",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "runtime": runtime,
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_tree": args.expected_tree,
            "expected_branch": args.expected_branch,
        },
    )
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("trajectory output or execution lock exists")
    write_json_report(
        status_output,
        _status(
            status="launching",
            detail="authorization_replayed_cpu_runbook_launching",
            project=project,
            output_root=output_root,
            runtime=runtime,
            sources=sources,
        ),
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PROJECT": project.as_posix(),
            "PYTHON": python.as_posix(),
            "EXPECTED_REVISION": args.expected_revision,
            "EXPECTED_TREE": args.expected_tree,
            "EXPECTED_BRANCH": args.expected_branch,
            "PREPARATION_REPORT": source_reports["preparation"]["path"],
            "EXPECTED_PREPARATION_SHA256": source_reports["preparation"]["sha256"],
            "EXECUTION_AUTHORIZATION": authorization.as_posix(),
            "EXPECTED_EXECUTION_AUTHORIZATION_SHA256": args.expected_authorization_sha256,
            "EXPECTED_RUNBOOK_SHA256": args.expected_runbook_sha256,
            "OUTPUT_ROOT": output_root.as_posix(),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    child = subprocess.Popen(["bash", runbook.as_posix()], cwd=project, env=environment)
    while child.poll() is None:
        write_json_report(
            status_output,
            _status(
                status="running",
                detail="six_checkpoint_cpu_sensitivity_replay_running",
                project=project,
                output_root=output_root,
                runtime=runtime,
                sources=sources,
                child_pid=child.pid,
            ),
        )
        time.sleep(args.poll_seconds)
    if child.returncode != 0:
        write_json_report(
            status_output,
            _status(
                status="failed",
                detail="trajectory_runbook_failed",
                project=project,
                output_root=output_root,
                runtime=runtime,
                sources=sources,
                child_pid=child.pid,
                error=f"runbook exit code {child.returncode}",
            ),
        )
        return int(child.returncode or 1)
    report_path = (
        output_root
        / "reports"
        / "exposure_semantic_trajectory_v1"
        / "trajectory_report.json"
    )
    report = read_json_object(report_path, name="trajectory report")
    if (
        report.get("schema_version") != 1
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("claim_boundary") != CLAIM_BOUNDARY
        or report.get("generation_advantage_proven") is not False
        or _completed_count(output_root) != 6
        or (report_path.stat().st_mode & 0o777) != 0o444
    ):
        raise ValueError("completed trajectory report boundary differs")
    sources["trajectory_report"] = file_identity(report_path)
    write_json_report(
        status_output,
        _status(
            status="completed",
            detail="six_checkpoint_cpu_exposure_trajectory_completed",
            project=project,
            output_root=output_root,
            runtime=runtime,
            sources=sources,
            child_pid=child.pid,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
