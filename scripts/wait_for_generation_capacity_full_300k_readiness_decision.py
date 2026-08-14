from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from cofitok.generation.capacity_completion_result import (
    validate_capacity_completion_100k_result,
)
from cofitok.generation.capacity_full_readiness_decision import (
    readiness_branch_selected,
    validate_capacity_full_readiness_decision,
)
from cofitok.generation.capacity_probe_execution import (
    validate_standing_experiment_authorization,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_capacity_full_300k_readiness_decision import (
        _git_identity,
        build_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_capacity_full_300k_readiness_decision import (
        _git_identity,
        build_from_sources,
    )


WAIT_BOUNDARY = {
    "cpu_only_waiter": True,
    "gpu_execution_allowed": False,
    "readiness_gpu_benchmark_launch_allowed": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "process_signaling_allowed": False,
    "decision_build_allowed_only_for_selected_result_branch": True,
    "existing_decision_must_be_byte_equivalent": True,
    "promotion_or_release_allowed": False,
}


def _publish_immutable(
    path: Path,
    payload: dict[str, Any],
    *,
    name: str,
) -> None:
    if path.exists():
        actual = read_json_object(path, name=name)
        expected_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if os.linesep != "\n":
            expected_text = expected_text.replace("\n", os.linesep)
        if actual != payload or path.read_bytes() != expected_text.encode("utf-8"):
            raise ValueError(f"existing {name} is not byte-equivalent to replay")
        return
    write_json_report(path, payload)


def _result_ready(
    status_path: Path,
    result_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not status_path.is_file():
        return None
    status = read_json_object(
        status_path,
        name="capacity completion result waiter status",
    )
    state = status.get("status")
    if state in {"failed", "error", "cancelled"}:
        raise RuntimeError(
            "capacity completion result waiter is terminal: "
            f"{state}: {status.get('detail')}"
        )
    if state != "completed":
        return None
    if not result_path.is_file():
        raise RuntimeError("completed capacity result waiter has no result")
    return status, read_json_object(
        result_path,
        name="capacity completion 100K result",
    )


def _status(
    *,
    state: str,
    detail: str,
    self_git: dict[str, Any],
    source_output_root: Path,
    full_output_root: Path,
    result_identity: dict[str, Any] | None = None,
    recommendation: dict[str, Any] | None = None,
    decision_identity: dict[str, Any] | None = None,
    readiness_execution_allowed: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "capacity_full_300k_readiness_decision_waiter",
        "status": state,
        "detail": detail,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git": copy.deepcopy(self_git),
        "source_output_root": source_output_root.resolve().as_posix(),
        "full_output_root": full_output_root.resolve().as_posix(),
        "capacity_completion_100k_result": copy.deepcopy(result_identity),
        "observed_recommendation": copy.deepcopy(recommendation),
        "readiness_decision": copy.deepcopy(decision_identity),
        "readiness_execution_allowed": readiness_execution_allowed,
        "error": error,
        "authorization_boundary": copy.deepcopy(WAIT_BOUNDARY),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait on CPU for the exact 100K capacity result and build only the "
            "selected fresh-300K readiness-execution decision."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--capacity-completion-result", type=Path, required=True)
    parser.add_argument("--result-waiter-status", type=Path, required=True)
    parser.add_argument(
        "--result-waiter-deployment-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-result-waiter-deployment-receipt-sha256",
        required=True,
    )
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--target-cofitok-config", type=Path, required=True)
    parser.add_argument("--target-dense-config", type=Path, required=True)
    parser.add_argument("--full-output-root", type=Path, required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--expected-result-execution-revision", required=True)
    parser.add_argument("--expected-result-execution-tree", required=True)
    parser.add_argument("--expected-result-execution-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-result-revision", required=True)
    parser.add_argument("--expected-result-tree", required=True)
    parser.add_argument("--expected-result-branch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    source_root = args.source_output_root.resolve()
    full_root = args.full_output_root.resolve()
    self_git = _git_identity(project)
    expected_self = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    if self_git != expected_self:
        raise ValueError("capacity-full decision waiter checkout identity differs")
    if project != Path(__file__).resolve().parents[1]:
        raise ValueError("capacity-full decision waiter project path differs")
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    validate_standing_experiment_authorization(
        read_json_object(
            args.standing_authorization,
            name="standing experiment authorization",
        )
    )
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            ready = _result_ready(
                args.result_waiter_status,
                args.capacity_completion_result,
            )
            if ready is None:
                write_json_report(
                    args.status,
                    _status(
                        state="waiting",
                        detail="waiting_for_source_replayed_capacity_completion_result",
                        self_git=self_git,
                        source_output_root=source_root,
                        full_output_root=full_root,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            _, result = ready
            result_identity = file_identity(args.capacity_completion_result)
            evidence = validate_capacity_completion_100k_result(
                result,
                expected_execution_revision=args.expected_result_execution_revision,
                expected_execution_tree=args.expected_result_execution_tree,
                expected_execution_branch=args.expected_result_execution_branch,
                expected_training_revision=args.expected_training_revision,
                expected_training_tree=args.expected_training_tree,
                expected_training_branch=args.expected_training_branch,
                expected_result_revision=args.expected_result_revision,
                expected_result_tree=args.expected_result_tree,
                expected_result_branch=args.expected_result_branch,
            )
            recommendation = evidence["recommended_next_stage"]
            if not readiness_branch_selected(result):
                write_json_report(
                    args.status,
                    _status(
                        state="not_selected",
                        detail=(
                            "capacity_result_selected_other_fail_closed_branch:"
                            + str(recommendation.get("id"))
                        ),
                        self_git=self_git,
                        source_output_root=source_root,
                        full_output_root=full_root,
                        result_identity=result_identity,
                        recommendation=recommendation,
                    ),
                )
                return
            builder_args = argparse.Namespace(
                capacity_completion_result=args.capacity_completion_result,
                result_waiter_status=args.result_waiter_status,
                result_waiter_deployment_receipt=(
                    args.result_waiter_deployment_receipt
                ),
                expected_result_waiter_deployment_receipt_sha256=(
                    args.expected_result_waiter_deployment_receipt_sha256
                ),
                standing_authorization=args.standing_authorization,
                expected_standing_authorization_sha256=(
                    args.expected_standing_authorization_sha256
                ),
                target_cofitok_config=args.target_cofitok_config,
                target_dense_config=args.target_dense_config,
                full_output_root=args.full_output_root,
                cofitok_run_dir=args.cofitok_run_dir,
                dense_run_dir=args.dense_run_dir,
                benchmark_root=args.benchmark_root,
                storage_path=args.storage_path,
                expected_decision_revision=args.expected_self_revision,
                expected_decision_tree=args.expected_self_tree,
                expected_decision_branch=args.expected_self_branch,
                expected_result_execution_revision=(
                    args.expected_result_execution_revision
                ),
                expected_result_execution_tree=args.expected_result_execution_tree,
                expected_result_execution_branch=(
                    args.expected_result_execution_branch
                ),
                expected_training_revision=args.expected_training_revision,
                expected_training_tree=args.expected_training_tree,
                expected_training_branch=args.expected_training_branch,
                expected_result_revision=args.expected_result_revision,
                expected_result_tree=args.expected_result_tree,
                expected_result_branch=args.expected_result_branch,
            )
            decision = build_from_sources(builder_args)
            validation = validate_capacity_full_readiness_decision(
                decision,
                expected_decision_revision=args.expected_self_revision,
                expected_decision_tree=args.expected_self_tree,
                expected_decision_branch=args.expected_self_branch,
                expected_result_revision=args.expected_result_revision,
                expected_result_tree=args.expected_result_tree,
                expected_result_branch=args.expected_result_branch,
            )
            _publish_immutable(
                args.decision,
                decision,
                name="capacity-full 300K readiness decision",
            )
            decision_identity = file_identity(args.decision)
            readiness_allowed = validation["execution_authorization"][
                "readiness_gpu_runtime_benchmark_allowed"
            ]
            write_json_report(
                args.status,
                _status(
                    state="completed",
                    detail="capacity_full_300k_readiness_decision_emitted",
                    self_git=self_git,
                    source_output_root=source_root,
                    full_output_root=full_root,
                    result_identity=result_identity,
                    recommendation=recommendation,
                    decision_identity=decision_identity,
                    readiness_execution_allowed=readiness_allowed,
                ),
            )
            return
        except Exception as exc:
            write_json_report(
                args.status,
                _status(
                    state="failed",
                    detail="capacity_full_300k_readiness_decision_replay_failed",
                    self_git=self_git,
                    source_output_root=source_root,
                    full_output_root=full_root,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise


if __name__ == "__main__":
    main()
