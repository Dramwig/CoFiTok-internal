from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.generation_pipeline_lineage import (
    build_lineage_report,
    inspect_lineage_stage,
    load_lineage_plan,
)
from cofitok.generation_recovery_supersession import (
    inspect_recovery_supersession,
    load_recovery_supersession_contract,
)
from cofitok.reporting import file_sha256, write_json_report


DEFAULT_PLAN = Path(
    "configs/generation/diagnostics/capacity_generation_pipeline_lineage_v1.json"
)


def _git(project: Path, *, include_porcelain_identity: bool = False) -> dict[str, Any]:
    def run(*arguments: str, binary: bool = False) -> str | bytes:
        result = subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=not binary,
        )
        return result.stdout

    tracked = str(run("status", "--porcelain", "--untracked-files=no")).strip()
    state: dict[str, Any] = {
        "path": project.resolve().as_posix(),
        "revision": str(run("rev-parse", "HEAD")).strip(),
        "tree": str(run("rev-parse", "HEAD^{tree}")).strip(),
        "branch": str(run("branch", "--show-current")).strip(),
        "tracked_dirty": bool(tracked),
    }
    if include_porcelain_identity:
        porcelain = bytes(run("status", "--porcelain=v1", binary=True))
        state.update(
            {
                "porcelain_count": len(porcelain.splitlines()),
                "porcelain_sha256": hashlib.sha256(porcelain).hexdigest(),
            }
        )
    return state


def _process_exists(pid: int) -> bool:
    return (Path("/proc") / str(pid)).is_dir()


def _gpu_compute_processes() -> tuple[list[dict[str, Any]], bool]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [], False
    rows = []
    complete = True
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) != 3:
            complete = False
            continue
        try:
            pid = int(values[0])
            memory = int(values[2])
            process = Path("/proc") / str(pid)
            command = (
                (process / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
            stat = (process / "stat").read_text(encoding="utf-8").rsplit(")", 1)[
                1
            ].split()
            rows.append(
                {
                    "pid": pid,
                    "process_name": values[1],
                    "used_memory_mib": memory,
                    "start_ticks": int(stat[19]),
                    "cwd": (process / "cwd").resolve().as_posix(),
                    "command": command,
                }
            )
        except (FileNotFoundError, IndexError, OSError, PermissionError, ValueError):
            complete = False
    return rows, complete


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Observe the source-bound quality-bridge through capacity-full "
            "generation lineage without launching or signaling any process."
        )
    )
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation"),
    )
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--quality-bridge-root", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-formal-revision", required=True)
    parser.add_argument("--expected-formal-branch", required=True)
    parser.add_argument("--expected-formal-porcelain-count", type=int, required=True)
    parser.add_argument("--expected-formal-porcelain-sha256", required=True)
    parser.add_argument("--recovery-supersession-contract", type=Path, required=True)
    parser.add_argument(
        "--expected-recovery-supersession-contract-sha256", required=True
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--timeout-seconds", type=float, default=31_536_000.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.stale_seconds <= args.poll_seconds
        or args.timeout_seconds <= 0
        or args.expected_formal_porcelain_count < 0
        or len(args.expected_plan_sha256) != 64
        or len(args.expected_formal_porcelain_sha256) != 64
        or len(args.expected_recovery_supersession_contract_sha256) != 64
    ):
        parser.error("capacity lineage observer thresholds or hashes are invalid")
    return args


def main() -> int:
    args = _parse_args()
    project = args.project.resolve()
    plan_path = (
        args.plan.resolve()
        if args.plan.is_absolute()
        else (project / args.plan).resolve()
    )
    checkpoint_root = args.checkpoint_root.resolve()
    source_root = args.source_output_root.resolve()
    quality_root = args.quality_bridge_root.resolve()
    formal_project = args.formal_project.resolve()
    output = args.output.resolve()
    supersession_contract_path = args.recovery_supersession_contract.resolve()
    expected_git = {
        "path": project.as_posix(),
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if _git(project) != expected_git:
        raise ValueError("capacity lineage observer checkout differs")
    if file_sha256(plan_path) != args.expected_plan_sha256:
        raise ValueError("capacity lineage plan SHA256 differs")
    if (
        file_sha256(supersession_contract_path)
        != args.expected_recovery_supersession_contract_sha256
    ):
        raise ValueError("recovery supersession contract SHA256 differs")
    variables = {
        "checkpoint_root": checkpoint_root.as_posix(),
        "source_output_root": source_root.as_posix(),
        "quality_bridge_root": quality_root.as_posix(),
    }
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        observed_at = datetime.now(timezone.utc)
        observer_git = _git(project)
        formal_git = _git(formal_project, include_porcelain_identity=True)
        external_issues = []
        if observer_git != expected_git:
            external_issues.append("observer_checkout_changed")
        if file_sha256(plan_path) != args.expected_plan_sha256:
            external_issues.append("lineage_plan_changed")
        if (
            file_sha256(supersession_contract_path)
            != args.expected_recovery_supersession_contract_sha256
        ):
            external_issues.append("recovery_supersession_contract_changed")
        expected_formal = {
            "revision": args.expected_formal_revision,
            "branch": args.expected_formal_branch,
            "porcelain_count": args.expected_formal_porcelain_count,
            "porcelain_sha256": args.expected_formal_porcelain_sha256,
        }
        if any(formal_git.get(key) != value for key, value in expected_formal.items()):
            external_issues.append("formal_checkout_snapshot_changed")
        plan = load_lineage_plan(plan_path, variables=variables)
        stages = [
            inspect_lineage_stage(
                spec,
                now=observed_at,
                stale_seconds=args.stale_seconds,
                process_exists=_process_exists,
            )
            for spec in plan["stages"]
        ]
        recovery_index = next(
            (
                index
                for index, stage in enumerate(stages)
                if stage.get("name") == "quality_bridge_100k_recovery"
            ),
            None,
        )
        if recovery_index is None:
            external_issues.append("quality_bridge_recovery_stage_missing")
        elif stages[recovery_index].get("classification") == "failure":
            try:
                supersession_contract = load_recovery_supersession_contract(
                    supersession_contract_path
                )
                stages[recovery_index] = inspect_recovery_supersession(
                    stages[recovery_index],
                    contract=supersession_contract,
                    now=observed_at,
                    stale_seconds=args.stale_seconds,
                    process_exists=_process_exists,
                )
            except (OSError, ValueError) as error:
                stages[recovery_index]["issues"] = [
                    *stages[recovery_index].get("issues", []),
                    f"recovery_supersession_invalid:{error}",
                ]
        gpu, gpu_complete = _gpu_compute_processes()
        usage = shutil.disk_usage(checkpoint_root)
        report = build_lineage_report(
            stages=stages,
            plan_identity=plan["identity"],
            observed_at=observed_at,
            stale_seconds=args.stale_seconds,
            observer_git=observer_git,
            formal_checkout=formal_git,
            gpu_compute_processes=gpu,
            gpu_query_complete=gpu_complete,
            disk={
                "path": checkpoint_root.as_posix(),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            hostname=socket.gethostname(),
            external_issues=external_issues,
        )
        report["observer"] = {
            "pid": os.getpid(),
            "poll_seconds": args.poll_seconds,
            "timeout_seconds": args.timeout_seconds,
            "output": output.as_posix(),
            "recovery_supersession_contract": supersession_contract_path.as_posix(),
        }
        write_json_report(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "current_stage": report["progress"]["current_stage"],
                    "detail": report["detail"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.once or report["status"] == "pass":
            return 1 if report["status"] in {"failed", "stalled"} else 0
        if time.monotonic() >= deadline:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
