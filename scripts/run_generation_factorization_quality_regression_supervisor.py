from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Mapping

try:
    import fcntl
except ModuleNotFoundError:  # Windows-only local CPU tests.
    fcntl = None  # type: ignore[assignment]

from cofitok.generation.capacity_probe_execution import (
    validate_standing_experiment_authorization,
)
from cofitok.generation.factorization_quality_regression import (
    DIAGNOSTIC_REPORT_ROLE,
    EXECUTION_BOUNDARY,
    FOLLOWUP_DECISION_PATH,
    OUTPUT_ROOT,
    PREPARATION_ROLE,
    SCOPE,
    classify_followup_decision,
    validate_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report

try:
    from build_generation_factorization_quality_regression_execution_authorization import (
        build_from_paths as build_authorization_from_paths,
    )
    from build_generation_factorization_quality_regression_source_binding import (
        build_from_paths as build_source_binding_from_paths,
    )
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts.build_generation_factorization_quality_regression_execution_authorization import (
        build_from_paths as build_authorization_from_paths,
    )
    from scripts.build_generation_factorization_quality_regression_source_binding import (
        build_from_paths as build_source_binding_from_paths,
    )


ROLE = "generation_factorization_quality_regression_supervisor"
DEPLOYMENT_ROLE = "generation_factorization_quality_regression_supervisor_deployment"
AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "exact_matched_quality_regression_route_required": True,
    "terminal_training_exposure_required": True,
    "terminal_system_guard_required": True,
    "exact_100k_ema_checkpoints_required": True,
    "five_consecutive_idle_gpu_polls_required": True,
    "unrelated_process_signaling_allowed": False,
    "training_launch_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_experiment_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def parse_gpu_process_pids(output: str) -> list[int]:
    pids: list[int] = []
    for line in output.splitlines():
        value = line.strip()
        if not value or value.lower().startswith("no running"):
            continue
        first = value.split(",", 1)[0].strip()
        if not first.isdigit():
            raise ValueError(f"unparseable nvidia-smi compute process row: {value}")
        pids.append(int(first))
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
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def _full_status(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"],
        text=True,
    ).strip()


def validate_self_git(
    project: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    observed = {**git_provenance(project), "tree": _tree(project)}
    expected = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if observed != expected or _full_status(project):
        raise ValueError("factorization-regression supervisor requires an exact clean checkout")
    return observed


def build_deployment_receipt(
    args: argparse.Namespace,
    *,
    control_git: Mapping[str, Any],
    supervisor_source: Mapping[str, Any],
    runbook: Mapping[str, Any],
    python_runtime: Mapping[str, Any],
    preparation: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "hostname": socket.gethostname(),
        "control": {
            "checkout": {
                "path": args.project.resolve().as_posix(),
                **dict(control_git),
            },
            "supervisor_source": dict(supervisor_source),
            "runbook": dict(runbook),
            "python_runtime": dict(python_runtime),
        },
        "authorization_inputs": {
            "preparation": dict(preparation),
            "standing_authorization": dict(standing_authorization),
        },
        "waiting_sources": {
            "quality_bridge_followup_decision": (
                args.followup_decision.resolve().as_posix()
            ),
            "terminal_system_guard": args.terminal_system_guard.resolve().as_posix(),
        },
        "targets": {
            "control_root": args.status_output.resolve().parent.as_posix(),
            "source_binding": args.source_binding.resolve().as_posix(),
            "execution_authorization": (
                args.execution_authorization.resolve().as_posix()
            ),
            "diagnostic_output_root": args.output_root.resolve().as_posix(),
            "status_output": args.status_output.resolve().as_posix(),
            "pid_file": args.pid_file.resolve().as_posix(),
            "deployment_receipt": (
                args.deployment_receipt_output.resolve().as_posix()
            ),
        },
        "timing": {
            "poll_seconds": args.poll_seconds,
            "required_idle_gpu_polls": args.required_idle_polls,
            "timeout_seconds": args.timeout_seconds,
        },
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
    }


def _status_payload(
    *,
    status: str,
    detail: str,
    args: argparse.Namespace,
    idle_polls: int,
    sources: Mapping[str, Any] | None = None,
    child_pid: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "updated_at_unix": time.time(),
        "project": args.project.resolve().as_posix(),
        "output_root": args.output_root.resolve().as_posix(),
        "idle_gpu_polls": idle_polls,
        "required_idle_gpu_polls": args.required_idle_polls,
        "sources": dict(sources or {}),
        "authorization_boundary": AUTHORIZATION_BOUNDARY,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_status(
    args: argparse.Namespace,
    *,
    status: str,
    detail: str,
    idle_polls: int,
    sources: Mapping[str, Any] | None = None,
    child_pid: int | None = None,
    error: str | None = None,
) -> None:
    write_json_report(
        args.status_output,
        _status_payload(
            status=status,
            detail=detail,
            args=args,
            idle_polls=idle_polls,
            sources=sources,
            child_pid=child_pid,
            error=error,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact terminal matched-quality-regression route, bind its "
            "100K sources, and launch one non-authorizing matched rollout diagnostic "
            "only after five consecutive idle-GPU polls."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if (
        args.poll_seconds <= 0.0
        or args.timeout_seconds <= 0.0
        or args.required_idle_polls != 5
    ):
        raise ValueError("factorization-regression supervisor timing contract differs")
    project = reject_symlink_chain(args.project, name="diagnostic project").resolve()
    args.project = project
    followup_decision = reject_symlink_chain(
        args.followup_decision,
        name="exposure-aware quality-bridge follow-up decision",
    ).resolve()
    args.followup_decision = followup_decision
    if followup_decision.as_posix() != FOLLOWUP_DECISION_PATH:
        raise ValueError(
            "factorization-regression requires the exposure-aware v2 follow-up decision"
        )
    output_root = reject_symlink_chain(
        args.output_root,
        name="factorization-regression output root",
    ).resolve()
    args.output_root = output_root
    if output_root.as_posix() != OUTPUT_ROOT:
        raise ValueError("factorization-regression output root differs")
    control_git = validate_self_git(
        project,
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
    )
    supervisor_source = reject_symlink_chain(
        project / "scripts/run_generation_factorization_quality_regression_supervisor.py",
        name="factorization-regression supervisor source",
    ).resolve()
    runbook = reject_symlink_chain(
        args.runbook,
        name="factorization-regression runbook",
    ).resolve()
    python_runtime = reject_symlink_chain(
        args.python,
        name="factorization-regression Python runtime",
    ).resolve()
    args.runbook = runbook
    args.python = python_runtime
    if (
        not supervisor_source.is_file()
        or not python_runtime.is_file()
        or not runbook.is_file()
        or runbook
        != project
        / "artifacts/runbooks/generation_factorization_quality_regression_probe_v1.sh"
    ):
        raise FileNotFoundError("factorization-regression runtime or runbook is missing")
    control_paths = {
        "preparation": args.preparation.resolve(),
        "source_binding": args.source_binding.resolve(),
        "execution_authorization": args.execution_authorization.resolve(),
        "status_output": args.status_output.resolve(),
        "pid_file": args.pid_file.resolve(),
        "deployment_receipt": args.deployment_receipt_output.resolve(),
    }
    control_root = control_paths["status_output"].parent
    if (
        any(path.parent != control_root for path in control_paths.values())
        or len(set(control_paths.values())) != len(control_paths)
        or _is_within(control_root, output_root)
        or _is_within(output_root, control_root)
    ):
        raise ValueError("factorization-regression control path scope differs")
    for name, path in control_paths.items():
        setattr(args, name if name != "deployment_receipt" else "deployment_receipt_output", path)
    preparation = read_json_object(
        reject_symlink_chain(args.preparation, name="diagnostic preparation"),
        name="factorization-regression preparation",
    )
    preparation_identity = file_identity(args.preparation)
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("factorization-regression preparation SHA256 differs")
    validate_preparation(
        preparation,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    standing = read_json_object(
        reject_symlink_chain(
            args.standing_authorization,
            name="standing experiment authorization",
        ),
        name="standing experiment authorization",
    )
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    validate_standing_experiment_authorization(standing)
    control_root.mkdir(parents=True, exist_ok=True)
    deployment = build_deployment_receipt(
        args,
        control_git=control_git,
        supervisor_source=file_identity(supervisor_source),
        runbook=file_identity(runbook),
        python_runtime=file_identity(python_runtime),
        preparation=preparation_identity,
        standing_authorization=standing_identity,
    )
    deployment_identity = prepare_manifest(
        args.deployment_receipt_output,
        deployment,
        resume=args.deployment_receipt_output.is_file(),
        overwrite=False,
    )
    supervisor_lock = args.pid_file.with_suffix(".lock")
    supervisor_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = supervisor_lock.open("a+", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another factorization-regression supervisor owns the lock"
            ) from error
    write_json_report(
        args.pid_file,
        {
            "schema_version": 1,
            "role": f"{ROLE}_pid",
            "pid": os.getpid(),
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_tree": args.expected_tree,
            "expected_branch": args.expected_branch,
        },
    )
    started = time.monotonic()
    idle_polls = 0
    sources: dict[str, Any] = {
        "preparation": preparation_identity,
        "standing_authorization": standing_identity,
        "deployment_receipt": deployment_identity,
    }
    while True:
        if time.monotonic() - started > args.timeout_seconds:
            _write_status(
                args,
                status="failed",
                detail="timeout_before_diagnostic_launch",
                idle_polls=idle_polls,
                sources=sources,
            )
            return 4
        validate_self_git(
            project,
            expected_revision=args.expected_revision,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
        )
        if not args.followup_decision.is_file():
            idle_polls = 0
            _write_status(
                args,
                status="waiting",
                detail="waiting_for_quality_bridge_followup_decision",
                idle_polls=idle_polls,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        decision = read_json_object(
            args.followup_decision,
            name="quality-bridge follow-up decision",
        )
        decision_identity = file_identity(args.followup_decision)
        sources["quality_bridge_followup_decision"] = decision_identity
        route = classify_followup_decision(decision)
        if route == "not_selected":
            _write_status(
                args,
                status="completed",
                detail="factorization_quality_regression_diagnostic_not_selected",
                idle_polls=0,
                sources=sources,
            )
            return 0
        if not args.terminal_system_guard.is_file():
            idle_polls = 0
            _write_status(
                args,
                status="waiting",
                detail="waiting_for_terminal_system_guard",
                idle_polls=idle_polls,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        source_binding = build_source_binding_from_paths(
            preparation_path=args.preparation,
            expected_preparation_sha256=args.expected_preparation_sha256,
            followup_decision_path=args.followup_decision,
            terminal_system_guard_path=args.terminal_system_guard,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
        )
        binding_identity = prepare_manifest(
            args.source_binding,
            source_binding,
            resume=args.source_binding.is_file(),
            overwrite=False,
        )
        sources["source_binding"] = binding_identity
        authorization = build_authorization_from_paths(
            preparation_path=args.preparation,
            expected_preparation_sha256=args.expected_preparation_sha256,
            source_binding_path=args.source_binding,
            expected_source_binding_sha256=binding_identity["sha256"],
            standing_authorization_path=args.standing_authorization,
            expected_standing_authorization_sha256=(
                args.expected_standing_authorization_sha256
            ),
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            project=project,
        )
        authorization_identity = prepare_manifest(
            args.execution_authorization,
            authorization,
            resume=args.execution_authorization.is_file(),
            overwrite=False,
        )
        sources["execution_authorization"] = authorization_identity
        if output_root.exists() or Path(f"{output_root}.lock").exists():
            raise FileExistsError("factorization-regression output or lock already exists")
        pids = gpu_compute_pids()
        idle_polls = idle_polls + 1 if not pids else 0
        if idle_polls < args.required_idle_polls:
            _write_status(
                args,
                status="waiting",
                detail=("waiting_for_consecutive_idle_gpu_polls" if not pids else "gpu_busy"),
                idle_polls=idle_polls,
                sources={**sources, "gpu_compute_pids": pids},
            )
            time.sleep(args.poll_seconds)
            continue
        replayed_binding = build_source_binding_from_paths(
            preparation_path=args.preparation,
            expected_preparation_sha256=args.expected_preparation_sha256,
            followup_decision_path=args.followup_decision,
            terminal_system_guard_path=args.terminal_system_guard,
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
        )
        if replayed_binding != source_binding:
            raise ValueError("factorization-regression sources changed before launch")
        validate_self_git(
            project,
            expected_revision=args.expected_revision,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PROJECT": project.as_posix(),
                "PYTHON": args.python.resolve().as_posix(),
                "EXPECTED_REVISION": args.expected_revision,
                "EXPECTED_TREE": args.expected_tree,
                "EXPECTED_BRANCH": args.expected_branch,
                "PREPARATION": args.preparation.resolve().as_posix(),
                "EXPECTED_PREPARATION_SHA256": args.expected_preparation_sha256,
                "SOURCE_BINDING": args.source_binding.resolve().as_posix(),
                "EXPECTED_SOURCE_BINDING_SHA256": binding_identity["sha256"],
                "EXECUTION_AUTHORIZATION": (
                    args.execution_authorization.resolve().as_posix()
                ),
                "EXPECTED_EXECUTION_AUTHORIZATION_SHA256": (
                    authorization_identity["sha256"]
                ),
                "STANDING_AUTHORIZATION": (
                    args.standing_authorization.resolve().as_posix()
                ),
                "EXPECTED_STANDING_AUTHORIZATION_SHA256": (
                    args.expected_standing_authorization_sha256
                ),
                "FOLLOWUP_DECISION": args.followup_decision.resolve().as_posix(),
                "TERMINAL_SYSTEM_GUARD": (
                    args.terminal_system_guard.resolve().as_posix()
                ),
                "OUTPUT_ROOT": output_root.as_posix(),
            }
        )
        child = subprocess.Popen(
            ["bash", args.runbook.resolve().as_posix()],
            cwd=project,
            env=environment,
        )
        while child.poll() is None:
            _write_status(
                args,
                status="running",
                detail="matched_factorization_quality_regression_diagnostic_running",
                idle_polls=idle_polls,
                sources=sources,
                child_pid=child.pid,
            )
            time.sleep(min(args.poll_seconds, 60.0))
        if child.returncode != 0:
            _write_status(
                args,
                status="failed",
                detail="factorization_quality_regression_runbook_failed",
                idle_polls=idle_polls,
                sources=sources,
                child_pid=child.pid,
                error=f"runbook exit code {child.returncode}",
            )
            return int(child.returncode or 1)
        diagnostic_path = (
            output_root / "reports" / "factorization_quality_regression" / "diagnostic_report.json"
        )
        diagnostic = read_json_object(
            diagnostic_path,
            name="factorization quality-regression diagnostic",
        )
        if (
            diagnostic.get("schema_version") != 1
            or diagnostic.get("role") != DIAGNOSTIC_REPORT_ROLE
            or diagnostic.get("status") != "completed"
            or diagnostic.get("scope") != SCOPE
            or diagnostic.get("claim_boundary", {}).get("training_launch_allowed")
            is not False
            or diagnostic.get("sources", {}).get("execution_authorization")
            != authorization_identity
        ):
            raise ValueError("completed factorization-regression diagnostic differs")
        sources["diagnostic_report"] = file_identity(diagnostic_path)
        _write_status(
            args,
            status="completed",
            detail="matched_factorization_quality_regression_diagnostic_completed",
            idle_polls=idle_polls,
            sources=sources,
            child_pid=child.pid,
        )
        return 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
