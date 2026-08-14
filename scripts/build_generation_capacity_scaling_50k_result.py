from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from cofitok.generation.capacity_probe import CAPACITY_PROBE_PARAMETER_COUNTS
from cofitok.generation.capacity_scaling_execution import (
    build_capacity_scaling_launch_receipt,
)
from cofitok.generation.capacity_scaling_result import (
    build_capacity_scaling_50k_result,
)
from cofitok.generation.capacity_scaling_training import (
    validate_capacity_scaling_partial_training,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_capacity_scaling_decision import (
        build_from_sources as rebuild_capacity_decision,
        replay_capacity_probe_result,
    )
    from scripts.build_generation_capacity_scaling_launch_receipt import (
        _resume_checkpoints,
    )
    from scripts.build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
except ModuleNotFoundError:
    from build_generation_capacity_scaling_decision import (
        build_from_sources as rebuild_capacity_decision,
        replay_capacity_probe_result,
    )
    from build_generation_capacity_scaling_launch_receipt import (
        _resume_checkpoints,
    )
    from build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
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
    parser.add_argument("--capacity-probe-result", type=Path, required=True)
    parser.add_argument("--expected-capacity-probe-result-sha256", required=True)
    parser.add_argument("--capacity-scaling-decision", type=Path, required=True)
    parser.add_argument("--expected-capacity-scaling-decision-sha256", required=True)
    parser.add_argument("--capacity-scaling-launch-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-capacity-scaling-launch-receipt-sha256",
        required=True,
    )
    parser.add_argument("--execution-status", type=Path, required=True)
    parser.add_argument("--expected-execution-status-sha256", required=True)
    parser.add_argument("--cofitok-training-validation", type=Path, required=True)
    parser.add_argument(
        "--expected-cofitok-training-validation-sha256",
        required=True,
    )
    parser.add_argument("--dense-training-validation", type=Path, required=True)
    parser.add_argument(
        "--expected-dense-training-validation-sha256",
        required=True,
    )
    parser.add_argument("--milestone-50000", type=Path, required=True)
    parser.add_argument("--expected-milestone-50000-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--expected-capacity-revision", required=True)
    parser.add_argument("--expected-capacity-tree", required=True)
    parser.add_argument("--expected-capacity-branch", required=True)
    parser.add_argument("--expected-decision-revision", required=True)
    parser.add_argument("--expected-decision-branch", required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)


def _read_exact(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    return read_json_object(path, name=name), identity


def _replay_decision(
    args: argparse.Namespace,
    *,
    capacity_result: dict[str, Any],
    capacity_result_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision, identity = _read_exact(
        args.capacity_scaling_decision,
        expected_sha256=args.expected_capacity_scaling_decision_sha256,
        name="capacity scaling decision",
    )
    recomputed = rebuild_capacity_decision(
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
    if decision != recomputed:
        raise ValueError("capacity scaling decision is not reproducible")
    if decision.get("source_evidence", {}).get("capacity_probe_result") != (
        capacity_result_identity
    ):
        raise ValueError("capacity scaling decision binds another capacity result")
    if capacity_result.get("decision", {}).get("capacity_supported") is not True:
        raise ValueError("capacity scaling decision source is not supported")
    return decision, identity


def _replay_launch_receipt(
    args: argparse.Namespace,
    *,
    decision: dict[str, Any],
    decision_identity: dict[str, Any],
    capacity_result: dict[str, Any],
    capacity_result_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    launch, identity = _read_exact(
        args.capacity_scaling_launch_receipt,
        expected_sha256=args.expected_capacity_scaling_launch_receipt_sha256,
        name="capacity scaling launch receipt",
    )
    source_reports = launch.get("source_reports")
    if not isinstance(source_reports, dict):
        raise ValueError("capacity scaling launch sources are missing")
    training_git = _git_identity(args.training_project.resolve())
    expected_training = {
        "revision": args.expected_capacity_revision,
        "tree": args.expected_capacity_tree,
        "branch": args.expected_capacity_branch,
        "tracked_dirty": False,
    }
    if training_git != expected_training:
        raise ValueError("capacity scaling training checkout identity differs")
    capacity_probe_launch, capacity_probe_launch_identity = _read_exact(
        Path(str(source_reports["capacity_probe_launch_receipt"]["path"])),
        expected_sha256=source_reports["capacity_probe_launch_receipt"]["sha256"],
        name="capacity probe launch receipt",
    )
    storage, storage_identity = _read_exact(
        Path(str(source_reports["storage_capacity"]["path"])),
        expected_sha256=source_reports["storage_capacity"]["sha256"],
        name="capacity scaling storage capacity",
    )
    cofitok_config = file_identity(source_reports["cofitok_config"]["path"])
    dense_config = file_identity(source_reports["dense_config"]["path"])
    if (
        cofitok_config != source_reports["cofitok_config"]
        or dense_config != source_reports["dense_config"]
        or decision_identity != source_reports["capacity_scaling_decision"]
        or capacity_result_identity != source_reports["capacity_probe_result"]
    ):
        raise ValueError("capacity scaling launch physical source differs")
    resume = _resume_checkpoints(
        decision,
        require_initial_state=False,
        expected_revision=args.expected_capacity_revision,
        expected_branch=args.expected_capacity_branch,
    )
    recomputed = build_capacity_scaling_launch_receipt(
        decision=decision,
        capacity_probe_result=capacity_result,
        capacity_probe_launch_receipt=capacity_probe_launch,
        storage_capacity=storage,
        source_identities={
            "capacity_scaling_decision": decision_identity,
            "capacity_probe_result": capacity_result_identity,
            "capacity_probe_launch_receipt": capacity_probe_launch_identity,
            "cofitok_config": cofitok_config,
            "dense_config": dense_config,
            "storage_capacity": storage_identity,
        },
        resume_checkpoints=resume,
        execution_git={
            "revision": args.expected_execution_revision,
            "tree": args.expected_execution_tree,
            "branch": args.expected_execution_branch,
            "tracked_dirty": False,
        },
        training_git=training_git,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_branch=args.expected_decision_branch,
        output_root=str(launch.get("selection", {}).get("output_root", "")),
        storage_path=str(launch.get("storage_capacity", {}).get("storage_path", "")),
        training_project=args.training_project.resolve().as_posix(),
        gpu_idle_at_launch=True,
        relevant_processes_absent_at_launch=True,
    )
    if launch != recomputed:
        raise ValueError("capacity scaling launch receipt is not reproducible")
    return launch, identity, training_git


def _replay_training_validation(
    path: Path,
    *,
    expected_sha256: str,
    method: str,
    args: argparse.Namespace,
    launch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual, identity = _read_exact(
        path,
        expected_sha256=expected_sha256,
        name=f"capacity scaling {method} training validation",
    )
    runtime = launch["selection"]["runtime_selection"]
    expected = validate_capacity_scaling_partial_training(
        report_path=actual["training_report"],
        config_path=actual["config"],
        expected_revision=args.expected_capacity_revision,
        expected_branch=args.expected_capacity_branch,
        expected_parameter_count=CAPACITY_PROBE_PARAMETER_COUNTS["base256"][method],
        expected_micro_batch_size=int(runtime["micro_batch_size"]),
        expected_gradient_accumulation_steps=int(
            runtime["gradient_accumulation_steps"]
        ),
    )
    if actual != expected:
        raise ValueError(f"capacity scaling {method} validation is not reproducible")
    return actual, identity


def build_from_sources(args: argparse.Namespace) -> dict[str, Any]:
    builder_git = _git_identity(PROJECT_ROOT)
    expected_builder = {
        "revision": args.expected_result_revision,
        "tree": args.expected_result_tree,
        "branch": args.expected_result_branch,
        "tracked_dirty": False,
    }
    if builder_git != expected_builder:
        raise ValueError("capacity scaling result checkout identity differs")
    capacity_result, capacity_result_identity = replay_capacity_probe_result(
        args.capacity_probe_result.resolve(),
        expected_sha256=args.expected_capacity_probe_result_sha256,
        expected_revision=args.expected_capacity_revision,
        expected_branch=args.expected_capacity_branch,
    )
    decision, decision_identity = _replay_decision(
        args,
        capacity_result=capacity_result,
        capacity_result_identity=capacity_result_identity,
    )
    launch, launch_identity, _ = _replay_launch_receipt(
        args,
        decision=decision,
        decision_identity=decision_identity,
        capacity_result=capacity_result,
        capacity_result_identity=capacity_result_identity,
    )
    cofitok_training, cofitok_training_identity = _replay_training_validation(
        args.cofitok_training_validation.resolve(),
        expected_sha256=args.expected_cofitok_training_validation_sha256,
        method="cofitok",
        args=args,
        launch=launch,
    )
    dense_training, dense_training_identity = _replay_training_validation(
        args.dense_training_validation.resolve(),
        expected_sha256=args.expected_dense_training_validation_sha256,
        method="dense_identity",
        args=args,
        launch=launch,
    )
    execution, execution_identity = _read_exact(
        args.execution_status.resolve(),
        expected_sha256=args.expected_execution_status_sha256,
        name="capacity scaling execution status",
    )
    milestone, milestone_identity = _read_exact(
        args.milestone_50000.resolve(),
        expected_sha256=args.expected_milestone_50000_sha256,
        name="capacity scaling step-50K milestone",
    )
    milestone_verification = verify_milestone_source_reports(
        milestone,
        source_profile="capacity_scaling",
    )
    milestone_evidence, _ = validate_milestone_report(
        milestone,
        expected_step=50_000,
        source_verification=milestone_verification,
        expected_source_profile="capacity_scaling",
    )
    cofitok_eval_path = Path(
        str(milestone["source_reports"]["cofitok_checkpoint_eval"]["path"])
    )
    cofitok_eval = read_json_object(
        cofitok_eval_path,
        name="capacity scaling CoFiTok checkpoint evaluation",
    )
    cofitok_eval_identity = file_identity(cofitok_eval_path)
    if cofitok_eval_identity != milestone["source_reports"]["cofitok_checkpoint_eval"]:
        raise ValueError("capacity scaling CoFiTok mechanism source changed")
    return build_capacity_scaling_50k_result(
        capacity_probe_result=capacity_result,
        capacity_scaling_decision=decision,
        launch_receipt=launch,
        execution_status=execution,
        training_validations={
            "cofitok": cofitok_training,
            "dense_identity": dense_training,
        },
        milestone_report=milestone,
        milestone_source_verification=milestone_verification,
        milestone_evidence=milestone_evidence,
        cofitok_checkpoint_evaluation=cofitok_eval,
        source_identities={
            "capacity_probe_result": capacity_result_identity,
            "capacity_scaling_decision": decision_identity,
            "capacity_scaling_launch_receipt": launch_identity,
            "capacity_scaling_execution_status": execution_identity,
            "cofitok_training_validation": cofitok_training_identity,
            "dense_identity_training_validation": dense_training_identity,
            "milestone_50000": milestone_identity,
            "cofitok_checkpoint_eval": cofitok_eval_identity,
        },
        result_builder_git=builder_git,
        expected_capacity_revision=args.expected_capacity_revision,
        expected_capacity_branch=args.expected_capacity_branch,
        expected_decision_revision=args.expected_decision_revision,
        expected_decision_branch=args.expected_decision_branch,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_training_tree=args.expected_capacity_tree,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a physical-source-replayed, non-authorizing result for the "
            "matched 250M step-10K to step-50K capacity-scaling segment."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity scaling 50K result exists: {args.output}")
    report = build_from_sources(args)
    write_json_report(args.output, report)
    print(report["decision"]["recommendation"]["id"])


if __name__ == "__main__":
    main()
