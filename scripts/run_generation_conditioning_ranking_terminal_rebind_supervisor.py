from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_terminal_rebind import (
    AUTHORIZATION_ROLE,
    CLAIM_BOUNDARY,
    EXECUTION_BOUNDARY,
    EXPECTED_SOURCE_SHA256,
    OUTPUT_ROOT,
    STAGE,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import file_sha256, git_provenance, write_json_report

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - Linux deployment only.
    fcntl = None


ROLE = "generation_conditioning_ranking_terminal_rebind_supervisor"
AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "mixed_terminal_failure_required": True,
    "completed_sampling_recovery_with_no_candidate_required": True,
    "gain_only_semantic_non_recovery_required": True,
    "legacy_v1_supersession_interlock_required": True,
    "official_epsilon_result_replay_required": True,
    "exact_revision_tree_branch_output_and_runbook_required": True,
    "consecutive_idle_gpu_polls_required": 5,
    "unrelated_process_signaling_allowed": False,
    "sampling_launch_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_allowed": False,
    "release_allowed": False,
}


def parse_gpu_process_pids(output: str) -> list[int]:
    pids: list[int] = []
    for line in output.splitlines():
        value = line.strip()
        if not value or value.lower().startswith("no running"):
            continue
        try:
            pids.append(int(value.split(",", 1)[0].strip()))
        except ValueError as error:
            raise ValueError(f"unparseable nvidia-smi compute row: {value}") from error
    return sorted(set(pids))


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_process_pids(result.stdout)


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project,
        text=True,
    ).strip()


def _full_status(project: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=project,
        text=True,
    ).strip()


def _verify_checkout(
    project: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
) -> None:
    if (
        git_provenance(project)
        != {"revision": revision, "branch": branch, "tracked_dirty": False}
        or _tree(project) != tree
        or _full_status(project)
    ):
        raise ValueError("terminal-rebind supervisor checkout identity differs")


def _verify_runtime_contract() -> None:
    if os.getppid() != 1:
        raise ValueError("terminal-rebind supervisor must be detached with parent PID 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("terminal-rebind supervisor must start CUDA-hidden")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise ValueError("terminal-rebind supervisor must use one CPU thread")
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("terminal-rebind supervisor must run with nice=10")
    ionice = subprocess.check_output(
        ["ionice", "-p", str(os.getpid())],
        text=True,
    ).strip()
    if ionice != "idle":
        raise ValueError("terminal-rebind supervisor must use idle I/O priority")


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
    identity = file_identity(authorization)
    if identity["sha256"] != expected_sha256:
        raise ValueError("terminal-rebind authorization SHA256 differs")
    payload = read_json_object(authorization, name="terminal-rebind authorization")
    if (
        payload.get("schema_version") != 1
        or payload.get("role") != AUTHORIZATION_ROLE
        or payload.get("status") != "authorized"
        or payload.get("stage") != STAGE
        or payload.get("output_root") != output_root.as_posix()
        or payload.get("execution_boundary") != EXECUTION_BOUNDARY
        or payload.get("claim_boundary") != CLAIM_BOUNDARY
        or payload.get("generation_advantage_proven") is not False
    ):
        raise ValueError("terminal-rebind authorization boundary differs")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": f"{project / 'src'}:{project}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [
            python.resolve().as_posix(),
            (project / "scripts" / "verify_generation_conditioning_ranking_terminal_rebind_authorization.py").as_posix(),
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
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "terminal-rebind authorization replay failed: " + completed.stderr[-2000:]
        )
    return {"payload": payload, "identity": identity}


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: Path,
    idle_polls: int,
    sources: Mapping[str, Any],
    child_pid: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "child_pid": child_pid,
        "project": project.as_posix(),
        "output_root": output_root.as_posix(),
        "idle_gpu_polls": idle_polls,
        "sources": dict(sources),
        "authorization_boundary": AUTHORIZATION_BOUNDARY,
        "generation_advantage_proven": False,
    }
    if error is not None:
        payload["error"] = error
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one source-bound four-arm 1K semantic-alignment probe "
            "after five idle-GPU polls."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--expected-runbook-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if fcntl is None:
        raise RuntimeError("terminal-rebind supervisor requires POSIX advisory locking")
    if args.required_idle_polls != 5 or args.poll_seconds < 30.0:
        raise ValueError("terminal-rebind supervisor requires five spaced idle-GPU polls")
    _verify_runtime_contract()
    project = reject_symlink_chain(args.project, name="terminal-rebind project").resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="terminal-rebind output root",
    ).resolve()
    if output_root.as_posix() != OUTPUT_ROOT:
        raise ValueError("terminal-rebind output root differs")
    status_output = reject_symlink_chain(
        args.status_output,
        name="terminal-rebind supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="terminal-rebind supervisor pid",
    ).resolve()
    lock_path = reject_symlink_chain(
        args.lock,
        name="terminal-rebind supervisor lock",
    ).resolve()
    authorization = reject_symlink_chain(
        args.authorization,
        name="terminal-rebind authorization",
    ).resolve()
    runbook = reject_symlink_chain(args.runbook, name="terminal-rebind runbook").resolve()
    python = reject_symlink_chain(args.python, name="terminal-rebind Python").resolve()
    status_output.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another terminal-rebind supervisor owns the lock") from error
    _verify_checkout(
        project,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
    )
    if not python.is_file() or not runbook.is_file():
        raise FileNotFoundError("terminal-rebind runtime or runbook is missing")
    if file_sha256(runbook) != args.expected_runbook_sha256:
        raise ValueError("terminal-rebind runbook SHA256 differs")
    replay = _verify_authorization(
        project=project,
        python=python,
        authorization=authorization,
        expected_sha256=args.expected_authorization_sha256,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
        output_root=output_root,
    )
    source_reports = replay["payload"].get("source_reports")
    if not isinstance(source_reports, Mapping):
        raise ValueError("terminal-rebind authorization sources are missing")
    sources = {
        "authorization": replay["identity"],
        "runbook": file_identity(runbook),
        "preparation": source_reports["preparation"],
        "legacy_preparation": source_reports["legacy_preparation"],
        "supersession_marker": source_reports["supersession_marker"],
    }
    write_json_report(
        pid_file,
        {
            "schema_version": 1,
            "role": f"{ROLE}_pid",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_tree": args.expected_tree,
            "expected_branch": args.expected_branch,
        },
    )
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("terminal-rebind output or execution lock already exists")

    started = time.monotonic()
    idle_polls = 0
    while idle_polls < 5:
        if time.monotonic() - started > args.timeout_seconds:
            write_json_report(
                status_output,
                _status(
                    status="failed",
                    detail="timeout_before_terminal_rebind_launch",
                    project=project,
                    output_root=output_root,
                    idle_polls=idle_polls,
                    sources=sources,
                ),
            )
            return 4
        _verify_checkout(
            project,
            revision=args.expected_revision,
            tree=args.expected_tree,
            branch=args.expected_branch,
        )
        if file_sha256(authorization) != args.expected_authorization_sha256:
            raise ValueError("terminal-rebind authorization changed while waiting")
        if file_sha256(runbook) != args.expected_runbook_sha256:
            raise ValueError("terminal-rebind runbook changed while waiting")
        pids = gpu_compute_pids()
        idle_polls = idle_polls + 1 if not pids else 0
        write_json_report(
            status_output,
            _status(
                status="waiting",
                detail=("waiting_for_five_idle_gpu_polls" if not pids else "gpu_busy"),
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                sources={**sources, "gpu_compute_pids": pids},
            ),
        )
        if idle_polls < 5:
            time.sleep(args.poll_seconds)

    replay = _verify_authorization(
        project=project,
        python=python,
        authorization=authorization,
        expected_sha256=args.expected_authorization_sha256,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
        output_root=output_root,
    )
    if gpu_compute_pids():
        raise RuntimeError("GPU became busy after the terminal-rebind authorization replay")
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("terminal-rebind output appeared before launch")
    _verify_checkout(
        project,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
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
            "LEGACY_PREPARATION": source_reports["legacy_preparation"]["path"],
            "EXPECTED_LEGACY_PREPARATION_SHA256": EXPECTED_SOURCE_SHA256[
                "legacy_preparation"
            ],
            "EXECUTION_AUTHORIZATION": authorization.as_posix(),
            "EXPECTED_EXECUTION_AUTHORIZATION_SHA256": replay["identity"]["sha256"],
            "EXPECTED_RUNBOOK_SHA256": args.expected_runbook_sha256,
            "OUTPUT_ROOT": output_root.as_posix(),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    child = subprocess.Popen(
        ["bash", runbook.as_posix()],
        cwd=project,
        env=environment,
    )
    while child.poll() is None:
        write_json_report(
            status_output,
            _status(
                status="running",
                detail="terminal_rebind_four_arm_probe_and_posteval_running",
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                child_pid=child.pid,
                sources=sources,
            ),
        )
        time.sleep(min(args.poll_seconds, 60.0))
    if child.returncode != 0:
        write_json_report(
            status_output,
            _status(
                status="failed",
                detail="terminal_rebind_runbook_failed",
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                child_pid=child.pid,
                sources=sources,
                error=f"runbook exit code {child.returncode}",
            ),
        )
        return int(child.returncode or 1)
    postevaluation = (
        output_root
        / "reports"
        / "conditioning_ranking_posteval_v1"
        / "postevaluation.json"
    )
    if not postevaluation.is_file():
        raise FileNotFoundError("terminal-rebind postevaluation is missing")
    sources["postevaluation"] = file_identity(postevaluation)
    write_json_report(
        status_output,
        _status(
            status="completed",
            detail="terminal_rebind_four_arm_probe_and_posteval_completed",
            project=project,
            output_root=output_root,
            idle_polls=idle_polls,
            child_pid=child.pid,
            sources=sources,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
