from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from cofitok.generation.capacity_scaling_execution import (
    build_capacity_scaling_launch_receipt,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.build_generation_capacity_scaling_decision import (
        build_from_sources as rebuild_decision,
    )
except ModuleNotFoundError:
    from build_generation_capacity_scaling_decision import (
        build_from_sources as rebuild_decision,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(status),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--capacity-probe-result", type=Path, required=True)
    parser.add_argument("--expected-capacity-probe-result-sha256", required=True)
    parser.add_argument("--capacity-probe-launch-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-capacity-probe-launch-receipt-sha256",
        required=True,
    )
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--expected-storage-capacity-sha256", required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--expected-capacity-revision", required=True)
    parser.add_argument("--expected-capacity-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)


def _expected_git(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = _git_identity(PROJECT_ROOT)
    expected_execution = {
        "revision": args.expected_execution_revision,
        "tree": args.expected_execution_tree,
        "branch": args.expected_execution_branch,
        "tracked_dirty": False,
    }
    if execution != expected_execution:
        raise ValueError("capacity scaling execution checkout identity differs")
    training = _git_identity(args.training_project.resolve())
    expected_training = {
        "revision": args.expected_training_revision,
        "tree": args.expected_training_tree,
        "branch": args.expected_training_branch,
        "tracked_dirty": False,
    }
    if training != expected_training:
        raise ValueError("capacity scaling training checkout identity differs")
    return execution, training


def _replay_decision(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(args.decision)
    if identity["sha256"] != args.expected_decision_sha256:
        raise ValueError("capacity scaling decision SHA256 differs")
    actual = read_json_object(args.decision, name="capacity scaling decision")
    expected = rebuild_decision(
        capacity_probe_result_path=args.capacity_probe_result.resolve(),
        expected_capacity_probe_result_sha256=(
            args.expected_capacity_probe_result_sha256
        ),
        standing_authorization_path=args.standing_authorization.resolve(),
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
        decision_git={
            "revision": args.expected_decision_revision,
            "branch": args.expected_decision_branch,
            "tracked_dirty": False,
        },
        expected_capacity_revision=args.expected_capacity_revision,
        expected_capacity_branch=args.expected_capacity_branch,
    )
    if actual != expected:
        raise ValueError("capacity scaling decision is not reproducible")
    return actual, identity


def _read_source(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    return read_json_object(path, name=name), identity


def _resume_checkpoints(
    decision: dict[str, Any],
    *,
    require_initial_state: bool,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    sources = decision["selection"]["resume_sources"]
    result: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        expected = sources[method]
        checkpoint_path = Path(expected["checkpoint"]["path"])
        integrity_path = Path(expected["checkpoint_integrity_manifest"]["path"])
        checkpoint = file_identity(checkpoint_path)
        integrity = file_identity(integrity_path)
        if (
            checkpoint != expected["checkpoint"]
            or integrity != expected["checkpoint_integrity_manifest"]
        ):
            raise ValueError(f"capacity scaling {method} physical checkpoint changed")
        integrity_payload = read_json_object(
            integrity_path,
            name=f"capacity scaling {method} checkpoint integrity",
        )
        if (
            int(integrity_payload.get("step", -1)) != 10_000
            or integrity_payload.get("checkpoint_sha256") != checkpoint["sha256"]
            or int(integrity_payload.get("checkpoint_bytes", -1))
            != checkpoint["bytes"]
            or integrity_payload.get("git_revision") != expected_revision
            or integrity_payload.get("git_branch") != expected_branch
            or integrity_payload.get("git_dirty") is not False
        ):
            raise ValueError(f"capacity scaling {method} checkpoint integrity differs")
        run_dir = checkpoint_path.parent
        if require_initial_state:
            latest = read_json_object(
                run_dir / "latest.json",
                name=f"capacity scaling {method} latest checkpoint",
            )
            later = [
                path
                for path in run_dir.glob("checkpoint_step_*.pt")
                if int(path.stem.rsplit("_", 1)[-1]) > 10_000
            ]
            if (
                int(latest.get("step", -1)) != 10_000
                or latest.get("checkpoint") != checkpoint_path.name
                or latest.get("checkpoint_sha256") != checkpoint["sha256"]
                or latest.get("integrity_manifest") != integrity_path.name
                or later
            ):
                raise ValueError(
                    f"capacity scaling {method} is not at the exact initial step"
                )
        result[method] = {
            "checkpoint": checkpoint,
            "checkpoint_integrity_manifest": integrity,
        }
    return result


def _gpu_idle() -> bool:
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
    return not any(line.strip().isdigit() for line in result.stdout.splitlines())


def _relevant_processes_absent(output_root: str) -> bool:
    result = subprocess.run(
        ["ps", "-eo", "args="],
        check=True,
        capture_output=True,
        text=True,
    )
    return not any(
        output_root in line
        and (
            "scripts/train_generation.py" in line
            or "scripts/run_generation_training_watchdog.py" in line
        )
        for line in result.stdout.splitlines()
    )


def build_from_sources(
    args: argparse.Namespace,
    *,
    require_initial_state: bool,
    require_idle_launch: bool,
) -> dict[str, Any]:
    execution_git, training_git = _expected_git(args)
    decision, decision_identity = _replay_decision(args)
    result, result_identity = _read_source(
        args.capacity_probe_result,
        expected_sha256=args.expected_capacity_probe_result_sha256,
        name="capacity probe result",
    )
    launch, launch_identity = _read_source(
        args.capacity_probe_launch_receipt,
        expected_sha256=args.expected_capacity_probe_launch_receipt_sha256,
        name="capacity probe launch receipt",
    )
    storage, storage_identity = _read_source(
        args.storage_capacity,
        expected_sha256=args.expected_storage_capacity_sha256,
        name="capacity scaling storage capacity",
    )
    cofitok_config = file_identity(args.cofitok_config)
    dense_config = file_identity(args.dense_config)
    resume = _resume_checkpoints(
        decision,
        require_initial_state=require_initial_state,
        expected_revision=args.expected_training_revision,
        expected_branch=args.expected_training_branch,
    )
    idle = _gpu_idle() if require_idle_launch else True
    processes_absent = (
        _relevant_processes_absent(args.output_root)
        if require_idle_launch
        else True
    )
    return build_capacity_scaling_launch_receipt(
        decision=decision,
        capacity_probe_result=result,
        capacity_probe_launch_receipt=launch,
        storage_capacity=storage,
        source_identities={
            "capacity_scaling_decision": decision_identity,
            "capacity_probe_result": result_identity,
            "capacity_probe_launch_receipt": launch_identity,
            "cofitok_config": cofitok_config,
            "dense_config": dense_config,
            "storage_capacity": storage_identity,
        },
        resume_checkpoints=resume,
        execution_git=execution_git,
        training_git=training_git,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_branch=args.expected_decision_branch,
        output_root=args.output_root,
        storage_path=args.storage_path,
        training_project=args.training_project.resolve().as_posix(),
        gpu_idle_at_launch=idle,
        relevant_processes_absent_at_launch=processes_absent,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the immutable launch receipt for only the exact matched "
            "250M step-10K to step-50K resume segment."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity scaling launch receipt exists: {args.output}")
    report = build_from_sources(
        args,
        require_initial_state=True,
        require_idle_launch=True,
    )
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
