from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RESULT_ROLE,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.build_generation_requested_class_visual_audit import (
        CLAIM_BOUNDARY as VISUAL_AUDIT_CLAIM_BOUNDARY,
        REPORT_FILENAME as VISUAL_AUDIT_REPORT_FILENAME,
        REPORT_ROLE as VISUAL_AUDIT_REPORT_ROLE,
    )
except ModuleNotFoundError:
    from build_generation_requested_class_visual_audit import (
        CLAIM_BOUNDARY as VISUAL_AUDIT_CLAIM_BOUNDARY,
        REPORT_FILENAME as VISUAL_AUDIT_REPORT_FILENAME,
        REPORT_ROLE as VISUAL_AUDIT_REPORT_ROLE,
    )


ROLE = "quality_bridge_terminal_requested_class_visual_audit_waiter"
SCHEMA_VERSION = 1
FIXED_INDICES = tuple(range(16))
AUTHORIZATION_BOUNDARY = {
    "cpu_only_visual_diagnostic_allowed": True,
    "gpu_use_allowed": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "quality_bridge_or_followup_decision_modified": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact 100K quality-bridge terminal result, then build "
            "a CPU-only source-bound requested-class visual audit."
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--quality-result", required=True)
    parser.add_argument("--cofitok-sampling-report", required=True)
    parser.add_argument("--dense-sampling-report", required=True)
    parser.add_argument("--cofitok-dir", required=True)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--classifier-calibration-report", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def validate_self_git(
    project: str | Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    root = reject_symlink_chain(project, name="visual-audit waiter project")
    if not root.is_dir():
        raise FileNotFoundError(f"visual-audit waiter project is missing: {root}")
    observed = {**git_provenance(root), "tree": _tree(root)}
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
        "tree": expected_tree,
    }
    if observed != expected:
        raise ValueError(f"visual-audit waiter Git identity differs: {observed}")
    return observed


def validate_quality_bridge_result(path: str | Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="quality-bridge terminal result")
    report = read_json_object(source, name="quality-bridge terminal result")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "completed"
        or report.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or report.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("quality-bridge terminal result contract differs")
    return {"identity": file_identity(source), "quality_screen": report.get("quality_screen")}


def build_visual_audit_command(args: argparse.Namespace) -> list[str]:
    return [
        str(Path(args.python).resolve()),
        str(
            Path(args.project).resolve()
            / "scripts"
            / "build_generation_requested_class_visual_audit.py"
        ),
        "--cofitok-sampling-report",
        str(Path(args.cofitok_sampling_report).resolve()),
        "--dense-sampling-report",
        str(Path(args.dense_sampling_report).resolve()),
        "--cofitok-dir",
        str(Path(args.cofitok_dir).resolve()),
        "--dense-dir",
        str(Path(args.dense_dir).resolve()),
        "--quality-result",
        str(Path(args.quality_result).resolve()),
        "--classifier-calibration-report",
        str(Path(args.classifier_calibration_report).resolve()),
        "--real-dir",
        str(Path(args.real_dir).resolve()),
        "--indices",
        ",".join(str(index) for index in FIXED_INDICES),
        "--panel-columns",
        "8",
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--expected-revision",
        args.expected_revision,
        "--expected-branch",
        args.expected_branch,
        "--resume",
    ]


def validate_visual_audit_report(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir).resolve() / VISUAL_AUDIT_REPORT_FILENAME
    report = read_json_object(path, name="terminal requested-class visual-audit report")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "completed"
        or report.get("role") != VISUAL_AUDIT_REPORT_ROLE
        or report.get("claim_boundary") != VISUAL_AUDIT_CLAIM_BOUNDARY
        or report.get("indices") != list(FIXED_INDICES)
    ):
        raise ValueError("terminal requested-class visual-audit report differs")
    panels = report.get("panels")
    if not isinstance(panels, list) or len(panels) != 2:
        raise ValueError("terminal requested-class visual-audit panels differ")
    for panel in panels:
        identity = file_identity(
            reject_symlink_chain(panel.get("path", ""), name="terminal visual panel")
        )
        if any(panel.get(field) != identity.get(field) for field in ("path", "bytes", "sha256")):
            raise ValueError("terminal requested-class visual panel identity differs")
    return file_identity(path)


def _base_status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "updated_at": _utc_now(),
        "poll_seconds": args.poll_seconds,
        "authorization_boundary": AUTHORIZATION_BOUNDARY,
        "expected": {
            "git": {
                "revision": args.expected_revision,
                "tree": args.expected_tree,
                "branch": args.expected_branch,
                "tracked_dirty": False,
            },
            "quality_result": str(Path(args.quality_result).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "indices": list(FIXED_INDICES),
        },
    }


def _write_status(
    path: Path,
    base: dict[str, Any],
    *,
    status: str,
    detail: str,
    polls: int,
    **extra: Any,
) -> None:
    write_json_report(
        path,
        {
            **base,
            "updated_at": _utc_now(),
            "status": status,
            "detail": detail,
            "polls": polls,
            **extra,
        },
    )


def run(args: argparse.Namespace) -> int:
    if args.poll_seconds < 1:
        raise ValueError("poll_seconds must be positive")
    project = Path(args.project).resolve()
    status_path = Path(args.status).resolve()
    lock_path = Path(args.lock).resolve()
    if status_path.is_symlink() or lock_path.is_symlink():
        raise ValueError("visual-audit waiter status and lock must not be symlinks")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    base = _base_status(args)
    polls = 0
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another terminal visual-audit waiter owns the lock") from error
        try:
            while True:
                git = validate_self_git(
                    project,
                    expected_revision=args.expected_revision,
                    expected_tree=args.expected_tree,
                    expected_branch=args.expected_branch,
                )
                quality_path = Path(args.quality_result).resolve()
                if not quality_path.is_file():
                    _write_status(
                        status_path,
                        base,
                        status="waiting",
                        detail="waiting_for_exact_quality_bridge_terminal_result",
                        polls=polls,
                        git=git,
                        quality_result=None,
                        visual_audit=None,
                    )
                    polls += 1
                    time.sleep(args.poll_seconds)
                    continue

                quality_result = validate_quality_bridge_result(quality_path)
                _write_status(
                    status_path,
                    base,
                    status="running",
                    detail="quality_bridge_complete:building_cpu_only_visual_audit",
                    polls=polls,
                    git=git,
                    quality_result=quality_result,
                    visual_audit=None,
                )
                command = build_visual_audit_command(args)
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "4",
                        "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4",
                        "PYTHONPATH": f"{project}:{project / 'src'}",
                    }
                )
                completed = subprocess.run(
                    command,
                    cwd=project,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.stdout:
                    print(completed.stdout, end="", flush=True)
                if completed.stderr:
                    print(completed.stderr, end="", flush=True)
                if completed.returncode != 0:
                    _write_status(
                        status_path,
                        base,
                        status="failed",
                        detail="terminal_visual_audit_child_failed",
                        polls=polls,
                        git=git,
                        quality_result=quality_result,
                        visual_audit=None,
                        child_exit_code=completed.returncode,
                    )
                    return completed.returncode
                visual_audit = validate_visual_audit_report(args.output_dir)
                _write_status(
                    status_path,
                    base,
                    status="completed",
                    detail="terminal_visual_audit_source_revalidated",
                    polls=polls,
                    git=git,
                    quality_result=quality_result,
                    visual_audit=visual_audit,
                    child_exit_code=0,
                )
                return 0
        except Exception as error:
            _write_status(
                status_path,
                base,
                status="failed",
                detail=f"{type(error).__name__}:{error}",
                polls=polls,
                quality_result=None,
                visual_audit=None,
            )
            raise


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
