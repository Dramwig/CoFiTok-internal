from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ID,
    FOLLOWUP_DECISION_BUILDER_GIT,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
    QUALITY_BRIDGE_EXECUTION_GIT,
    build_conditioning_ranking_probe_execution_authorization,
    validate_class_conditioning_followup_decision,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised on Linux deployment.
    fcntl = None


ROLE = "generation_conditioning_ranking_probe_supervisor"
AUTHORIZATION_BOUNDARY = {
    "standing_authorization_required": True,
    "exact_terminal_class_fidelity_route_required": True,
    "requested_class_visual_evidence_required": True,
    "exact_revision_stage_and_output_binding_required": True,
    "consecutive_idle_gpu_polls_required": True,
    "unrelated_process_signaling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}


def _bound_json(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _full_git_status(project: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project,
        text=True,
    ).strip()


def _embedded_json(
    descriptor: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} identity is missing")
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=label,
    ).resolve()
    identity = file_identity(source)
    if identity != dict(descriptor):
        raise ValueError(f"{label} identity differs")
    return read_json_object(source, name=label), identity


def parse_gpu_process_pids(output: str) -> list[int]:
    pids: list[int] = []
    for line in output.splitlines():
        value = line.strip()
        if not value or value.lower().startswith("no running"):
            continue
        try:
            pids.append(int(value.split(",", 1)[0].strip()))
        except ValueError as error:
            raise ValueError(f"unparseable nvidia-smi compute process row: {value}") from error
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


def followup_route(
    report: Mapping[str, Any],
    *,
    expected_quality_bridge_result: Mapping[str, Any] | None = None,
) -> str:
    recommendation = report.get("recommended_next_stage")
    if (
        report.get("schema_version") != FOLLOWUP_DECISION_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("role") != FOLLOWUP_DECISION_ROLE
        or report.get("decision_builder_git") != FOLLOWUP_DECISION_BUILDER_GIT
        or report.get("quality_bridge_execution_git")
        != QUALITY_BRIDGE_EXECUTION_GIT
        or report.get("authorization_boundary") != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or not isinstance(recommendation.get("id"), str)
    ):
        return "invalid"
    if recommendation.get("id") != FOLLOWUP_DECISION_ID:
        return "not_selected"
    try:
        validate_class_conditioning_followup_decision(
            report,
            expected_quality_bridge_result=expected_quality_bridge_result,
        )
    except (TypeError, ValueError):
        return "invalid"
    return "selected"


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: Path,
    idle_polls: int,
    child_pid: int | None = None,
    sources: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "project": project.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
        "idle_gpu_polls": idle_polls,
        "sources": dict(sources or {}),
        "authorization_boundary": AUTHORIZATION_BOUNDARY,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_status(path: Path, **kwargs: Any) -> None:
    write_json_report(path, _status(**kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact terminal class-fidelity route and then run the "
            "source-bound four-arm 1K conditioning-ranking probe on an idle GPU."
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
    parser.add_argument("--quality-bridge-result", type=Path, required=True)
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--terminal-system-guard-status", type=Path, required=True)
    parser.add_argument(
        "--requested-class-visual-audit-status",
        type=Path,
        required=True,
    )
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = reject_symlink_chain(args.project, name="ranking supervisor project").resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="ranking probe output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="ranking supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(args.pid_file, name="ranking supervisor pid").resolve()
    lock_path = reject_symlink_chain(
        args.lock,
        name="ranking supervisor lock",
    ).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        raise RuntimeError("ranking supervisor requires POSIX advisory locking")
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another conditioning-ranking supervisor owns the lock") from error
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if (
        git_provenance(project) != expected_git
        or _git_tree(project) != args.expected_tree
        or _full_git_status(project)
    ):
        raise ValueError("ranking supervisor requires the exact fully clean checkout")
    if not args.python.is_file() or not args.runbook.is_file():
        raise FileNotFoundError("ranking supervisor runtime or runbook is missing")
    preparation, preparation_identity = _bound_json(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        label="conditioning-ranking preparation",
    )
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    write_json_report(
        pid_file,
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
    }
    while True:
        if time.monotonic() - started > args.timeout_seconds:
            _write_status(
                status_output,
                status="failed",
                detail="timeout_before_probe_launch",
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                sources=sources,
            )
            return 4
        if not args.followup_decision.is_file():
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_quality_bridge_followup_decision",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        if not args.quality_bridge_result.is_file():
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_bound_quality_bridge_result",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        quality, quality_identity = _bound_json(
            args.quality_bridge_result,
            expected_sha256=None,
            label="quality-bridge terminal result",
        )
        sources["quality_bridge_result"] = quality_identity
        followup, followup_identity = _bound_json(
            args.followup_decision,
            expected_sha256=None,
            label="quality-bridge follow-up decision",
        )
        sources["quality_bridge_followup_decision"] = followup_identity
        route = followup_route(
            followup,
            expected_quality_bridge_result=quality_identity,
        )
        if route == "not_selected":
            _write_status(
                status_output,
                status="completed",
                detail="conditioning_ranking_probe_not_selected",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources=sources,
            )
            return 0
        if route != "selected":
            raise ValueError("quality-bridge follow-up decision is malformed")
        if not all(
            path.is_file()
            for path in (
                args.terminal_system_guard,
                args.terminal_system_guard_status,
                args.requested_class_visual_audit_status,
            )
        ):
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_terminal_system_and_visual_evidence",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        terminal, terminal_identity = _bound_json(
            args.terminal_system_guard,
            expected_sha256=None,
            label="terminal-system claim guard",
        )
        sources["terminal_system_claim_guard"] = terminal_identity
        terminal_status, terminal_status_identity = _bound_json(
            args.terminal_system_guard_status,
            expected_sha256=None,
            label="terminal-system claim guard waiter status",
        )
        sources["terminal_system_claim_guard_waiter_status"] = (
            terminal_status_identity
        )
        visual_status, visual_status_identity = _bound_json(
            args.requested_class_visual_audit_status,
            expected_sha256=None,
            label="requested-class visual-audit waiter status",
        )
        sources["requested_class_visual_audit_waiter_status"] = (
            visual_status_identity
        )
        visual_report, visual_report_identity = _embedded_json(
            visual_status.get("visual_audit"),
            label="requested-class visual-audit report",
        )
        sources["requested_class_visual_audit_report"] = visual_report_identity
        report = build_conditioning_ranking_probe_execution_authorization(
            preparation=preparation,
            preparation_identity=preparation_identity,
            standing_authorization=standing,
            standing_authorization_identity=standing_identity,
            followup_decision=followup,
            followup_decision_identity=followup_identity,
            quality_bridge_result=quality,
            quality_bridge_result_identity=quality_identity,
            terminal_system_guard=terminal,
            terminal_system_guard_identity=terminal_identity,
            terminal_system_guard_status=terminal_status,
            terminal_system_guard_status_identity=terminal_status_identity,
            requested_class_visual_audit_status=visual_status,
            requested_class_visual_audit_status_identity=visual_status_identity,
            requested_class_visual_audit_report=visual_report,
            requested_class_visual_audit_report_identity=visual_report_identity,
            authorization_git=git_provenance(project),
            authorization_tree=_git_tree(project),
            expected_revision=args.expected_revision,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
            expected_output_root=output_root.as_posix(),
        )
        authorization_identity = prepare_manifest(
            args.execution_authorization,
            report,
            resume=args.execution_authorization.is_file(),
            overwrite=False,
        )
        sources["execution_authorization"] = authorization_identity
        if output_root.exists() or Path(f"{output_root}.lock").exists():
            raise FileExistsError("conditioning-ranking output or lock already exists")
        pids = gpu_compute_pids()
        idle_polls = idle_polls + 1 if not pids else 0
        if idle_polls < args.required_idle_polls:
            _write_status(
                status_output,
                status="waiting",
                detail=("waiting_for_consecutive_idle_gpu_polls" if not pids else "gpu_busy"),
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                sources={**sources, "gpu_compute_pids": pids},
            )
            time.sleep(args.poll_seconds)
            continue
        if (
            git_provenance(project) != expected_git
            or _git_tree(project) != args.expected_tree
            or _full_git_status(project)
        ):
            raise ValueError("ranking supervisor checkout changed before launch")
        environment = os.environ.copy()
        environment.update(
            {
                "PROJECT": project.as_posix(),
                "PYTHON": args.python.resolve().as_posix(),
                "EXPECTED_REVISION": args.expected_revision,
                "EXPECTED_TREE": args.expected_tree,
                "EXPECTED_BRANCH": args.expected_branch,
                "PREPARATION_REPORT": args.preparation.resolve().as_posix(),
                "EXPECTED_PREPARATION_SHA256": args.expected_preparation_sha256,
                "EXECUTION_AUTHORIZATION": args.execution_authorization.resolve().as_posix(),
                "EXPECTED_EXECUTION_AUTHORIZATION_SHA256": authorization_identity[
                    "sha256"
                ],
                "EXPECTED_STANDING_AUTHORIZATION_SHA256": (
                    args.expected_standing_authorization_sha256
                ),
                "OUTPUT_ROOT": output_root.as_posix(),
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
        child = subprocess.Popen(
            ["bash", args.runbook.resolve().as_posix()],
            cwd=project,
            env=environment,
        )
        while child.poll() is None:
            _write_status(
                status_output,
                status="running",
                detail="conditioning_ranking_four_arm_probe_and_posteval_running",
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                child_pid=child.pid,
                sources=sources,
            )
            time.sleep(min(args.poll_seconds, 60.0))
        if child.returncode != 0:
            _write_status(
                status_output,
                status="failed",
                detail="conditioning_ranking_runbook_failed",
                project=project,
                output_root=output_root,
                idle_polls=idle_polls,
                child_pid=child.pid,
                sources=sources,
                error=f"runbook exit code {child.returncode}",
            )
            return int(child.returncode or 1)
        postevaluation = (
            output_root
            / "reports"
            / "conditioning_ranking_posteval_v1"
            / "postevaluation.json"
        )
        if not postevaluation.is_file():
            raise FileNotFoundError("conditioning-ranking postevaluation is missing")
        sources["postevaluation"] = file_identity(postevaluation)
        _write_status(
            status_output,
            status="completed",
            detail="conditioning_ranking_four_arm_probe_and_posteval_completed",
            project=project,
            output_root=output_root,
            idle_polls=idle_polls,
            child_pid=child.pid,
            sources=sources,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
