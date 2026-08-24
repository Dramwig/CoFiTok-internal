from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.semantic_residual_alignment_probe import (
    AUTHORIZATION_ROLE,
    CLAIM_BOUNDARY,
    EXECUTION_BOUNDARY,
    OUTPUT_ROOT,
    POSTEVALUATION_ROLE,
    STAGE,
    SUPERVISOR_ROLE,
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


AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "failed_prior_four_arm_postevaluation_required": True,
    "exact_revision_tree_branch_output_and_runbook_required": True,
    "consecutive_idle_gpu_polls_required": 5,
    "unrelated_process_signaling_allowed": False,
    "generation_sampling_allowed": False,
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
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_process_pids(completed.stdout)


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
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
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=project,
        text=True,
    ).strip()
    if (
        git_provenance(project)
        != {"revision": revision, "branch": branch, "tracked_dirty": False}
        or _tree(project) != tree
        or status
    ):
        raise ValueError("residual-alignment supervisor checkout identity differs")


def _verify_runtime_contract() -> None:
    if os.getppid() != 1:
        raise ValueError("residual-alignment supervisor must have parent PID 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("residual-alignment supervisor must start CUDA-hidden")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise ValueError("residual-alignment supervisor must use one CPU thread")
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("residual-alignment supervisor must run with nice=10")
    if subprocess.check_output(["ionice", "-p", str(os.getpid())], text=True).strip() != "idle":
        raise ValueError("residual-alignment supervisor must use idle I/O priority")


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(authorization)
    if identity["sha256"] != expected_sha256:
        raise ValueError("residual-alignment authorization SHA256 differs")
    payload = read_json_object(authorization, name="residual-alignment authorization")
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
        raise ValueError("residual-alignment authorization boundary differs")
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
            (
                project
                / "scripts"
                / "verify_generation_semantic_residual_alignment_execution_authorization.py"
            ).as_posix(),
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
            "residual-alignment authorization replay failed: "
            + completed.stderr[-2_000:]
        )
    return payload, identity


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
        "role": SUPERVISOR_ROLE,
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
            "Run exactly one source-bound semantic residual-alignment four-arm "
            "probe after five idle-GPU polls."
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
        raise RuntimeError("residual-alignment supervisor requires POSIX locking")
    if args.required_idle_polls != 5 or args.poll_seconds < 30.0:
        raise ValueError("supervisor requires five spaced idle-GPU polls")
    _verify_runtime_contract()
    project = reject_symlink_chain(args.project, name="residual-alignment project").resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="residual-alignment output root",
    ).resolve()
    if output_root.as_posix() != OUTPUT_ROOT:
        raise ValueError("residual-alignment output root differs")
    status_output = reject_symlink_chain(args.status_output, name="supervisor status").resolve()
    pid_file = reject_symlink_chain(args.pid_file, name="supervisor PID file").resolve()
    lock_path = reject_symlink_chain(args.lock, name="supervisor lock").resolve()
    authorization = reject_symlink_chain(args.authorization, name="authorization").resolve()
    runbook = reject_symlink_chain(args.runbook, name="runbook").resolve()
    python = reject_symlink_chain(args.python, name="Python runtime").resolve()
    for path in (status_output.parent, pid_file.parent, lock_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another residual-alignment supervisor owns the lock") from error
    _verify_checkout(
        project,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
    )
    if not python.is_file() or not runbook.is_file():
        raise FileNotFoundError("residual-alignment runtime or runbook is missing")
    if file_sha256(runbook) != args.expected_runbook_sha256:
        raise ValueError("residual-alignment runbook SHA256 differs")
    authorization_payload, authorization_identity = _verify_authorization(
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
        raise ValueError("residual-alignment authorization sources are missing")
    sources: dict[str, Any] = {
        "authorization": authorization_identity,
        "runbook": file_identity(runbook),
        "preparation": source_reports["preparation"],
        "quality_bridge_result": source_reports["quality_bridge_result"],
        "source_postevaluation": source_reports["source_postevaluation"],
    }
    write_json_report(
        pid_file,
        {
            "schema_version": 1,
            "role": f"{SUPERVISOR_ROLE}_pid",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_tree": args.expected_tree,
            "expected_branch": args.expected_branch,
        },
    )
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("residual-alignment output or execution lock exists")

    started = time.monotonic()
    idle_polls = 0
    while idle_polls < 5:
        if time.monotonic() - started > args.timeout_seconds:
            write_json_report(
                status_output,
                _status(
                    status="failed",
                    detail="timeout_before_residual_alignment_launch",
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
            raise ValueError("authorization changed while waiting")
        if file_sha256(runbook) != args.expected_runbook_sha256:
            raise ValueError("runbook changed while waiting")
        pids = gpu_compute_pids()
        idle_polls = idle_polls + 1 if not pids else 0
        write_json_report(
            status_output,
            _status(
                status="waiting",
                detail="waiting_for_five_idle_gpu_polls" if not pids else "gpu_busy",
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                sources={**sources, "gpu_compute_pids": pids},
            ),
        )
        if idle_polls < 5:
            time.sleep(args.poll_seconds)

    authorization_payload, authorization_identity = _verify_authorization(
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
        raise RuntimeError("GPU became busy after authorization replay")
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("residual-alignment output appeared before launch")
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
            "EXECUTION_AUTHORIZATION": authorization.as_posix(),
            "EXPECTED_EXECUTION_AUTHORIZATION_SHA256": authorization_identity["sha256"],
            "EXPECTED_RUNBOOK_SHA256": args.expected_runbook_sha256,
            "OUTPUT_ROOT": output_root.as_posix(),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    child = subprocess.Popen(["bash", runbook.as_posix()], cwd=project, env=environment)
    while child.poll() is None:
        write_json_report(
            status_output,
            _status(
                status="running",
                detail="four_arm_training_and_cpu_postevaluation_running",
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
                detail="residual_alignment_runbook_failed",
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
        / "semantic_residual_alignment_posteval_v1"
        / "postevaluation.json"
    )
    if not postevaluation.is_file():
        raise FileNotFoundError("residual-alignment postevaluation is missing")
    report = read_json_object(postevaluation, name="residual-alignment postevaluation")
    claim_boundary = report.get("claim_boundary")
    if (
        report.get("role") != POSTEVALUATION_ROLE
        or report.get("status") != "completed"
        or report.get("generation_advantage_proven") is not False
        or not isinstance(claim_boundary, Mapping)
        or any(
            claim_boundary.get(field) is not False
            for field in (
                "authorizes_training",
                "authorizes_sampling",
                "authorizes_checkpoint_promotion",
                "authorizes_followup_training",
                "authorizes_full_training",
                "authorizes_300k_training",
                "authorizes_export",
                "authorizes_release",
                "authorizes_process_signals",
            )
        )
    ):
        raise ValueError("residual-alignment postevaluation boundary differs")
    sources["postevaluation"] = file_identity(postevaluation)
    write_json_report(
        status_output,
        _status(
            status="completed",
            detail="four_arm_training_and_cpu_postevaluation_completed",
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
