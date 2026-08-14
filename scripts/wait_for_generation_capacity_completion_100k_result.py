from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from typing import Any

from cofitok.generation.capacity_completion_result import (
    validate_capacity_completion_100k_result,
)
from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    validate_standing_experiment_authorization,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_capacity_completion_100k_result import (
        _git_identity,
        build_from_sources,
        source_paths,
    )
except ModuleNotFoundError:
    from build_generation_capacity_completion_100k_result import (
        _git_identity,
        build_from_sources,
        source_paths,
    )


WAIT_BOUNDARY = {
    "cpu_only_waiter": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "additional_training_allowed": False,
    "process_signaling_allowed": False,
    "source_reports_must_be_physically_replayed": True,
    "existing_result_must_be_byte_equivalent": True,
    "result_is_promotion_gate": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "new_source_compatible_gate_or_decision_required": True,
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


def _execution_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    report = read_json_object(
        path,
        name="capacity completion 100K execution status",
    )
    status = report.get("status")
    if status in {"failed", "error", "cancelled"}:
        raise RuntimeError(
            "capacity completion 100K execution is terminal: "
            f"{status}: {report.get('detail')}"
        )
    return status == "completed" and report.get("stage") == "complete"


def _standing_authorization_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    standing = validate_standing_experiment_authorization(
        read_json_object(path, name="standing experiment authorization")
    )
    if (
        standing["preserved_safety_boundaries"]
        != STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
    ):
        raise ValueError("standing experiment authorization safety boundary differs")
    return {"source": identity, "validated_record": standing}


def _status(
    *,
    state: str,
    detail: str,
    self_git: dict[str, Any],
    output_root: Path,
    standing_authorization: dict[str, Any],
    result_identity: dict[str, Any] | None = None,
    recommendation: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "capacity_completion_100k_result_source_replay_waiter",
        "status": state,
        "detail": detail,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git": copy.deepcopy(self_git),
        "output_root": output_root.resolve().as_posix(),
        "standing_authorization": copy.deepcopy(standing_authorization),
        "capacity_completion_100k_result": copy.deepcopy(result_identity),
        "recommended_next_stage": copy.deepcopy(recommendation),
        "error": error,
        "authorization_boundary": copy.deepcopy(WAIT_BOUNDARY),
    }


def _builder_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=args.output_root,
        expected_completion_execution_revision=(
            args.expected_completion_execution_revision
        ),
        expected_completion_execution_tree=args.expected_completion_execution_tree,
        expected_completion_execution_branch=(
            args.expected_completion_execution_branch
        ),
        expected_completion_decision_revision=(
            args.expected_completion_decision_revision
        ),
        expected_completion_decision_tree=args.expected_completion_decision_tree,
        expected_completion_decision_branch=args.expected_completion_decision_branch,
        expected_scaling_execution_revision=args.expected_scaling_execution_revision,
        expected_scaling_execution_tree=args.expected_scaling_execution_tree,
        expected_scaling_execution_branch=args.expected_scaling_execution_branch,
        expected_scaling_result_revision=args.expected_scaling_result_revision,
        expected_scaling_result_tree=args.expected_scaling_result_tree,
        expected_scaling_result_branch=args.expected_scaling_result_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_result_revision=args.expected_self_revision,
        expected_result_tree=args.expected_self_tree,
        expected_result_branch=args.expected_self_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait without using a GPU, physically replay the matched 250M "
            "step-100K evidence, and emit a non-authorizing terminal result."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--expected-completion-execution-revision", required=True)
    parser.add_argument("--expected-completion-execution-tree", required=True)
    parser.add_argument("--expected-completion-execution-branch", required=True)
    parser.add_argument("--expected-completion-decision-revision", required=True)
    parser.add_argument("--expected-completion-decision-tree", required=True)
    parser.add_argument("--expected-completion-decision-branch", required=True)
    parser.add_argument("--expected-scaling-execution-revision", required=True)
    parser.add_argument("--expected-scaling-execution-tree", required=True)
    parser.add_argument("--expected-scaling-execution-branch", required=True)
    parser.add_argument("--expected-scaling-result-revision", required=True)
    parser.add_argument("--expected-scaling-result-tree", required=True)
    parser.add_argument("--expected-scaling-result-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 1:
        raise ValueError("poll seconds must be positive")
    project = args.project.resolve()
    output_root = args.output_root.resolve()
    args.output_root = output_root
    self_git = _git_identity(project)
    expected_self = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    if self_git != expected_self:
        raise ValueError("capacity completion result waiter checkout identity differs")
    if project != Path(__file__).resolve().parents[1]:
        raise ValueError("capacity completion result waiter project path differs")
    standing_status = _standing_authorization_evidence(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
    )
    paths = source_paths(output_root)
    execution_status = paths["capacity_completion_execution_status"]
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            if not _execution_completed(execution_status):
                write_json_report(
                    args.status,
                    _status(
                        state="waiting",
                        detail="waiting_for_completed_capacity_completion_100k_execution",
                        self_git=self_git,
                        output_root=output_root,
                        standing_authorization=standing_status,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            result = build_from_sources(_builder_args(args))
            evidence = validate_capacity_completion_100k_result(
                result,
                expected_execution_revision=(
                    args.expected_completion_execution_revision
                ),
                expected_execution_tree=args.expected_completion_execution_tree,
                expected_execution_branch=args.expected_completion_execution_branch,
                expected_training_revision=args.expected_training_revision,
                expected_training_tree=args.expected_training_tree,
                expected_training_branch=args.expected_training_branch,
                expected_result_revision=args.expected_self_revision,
                expected_result_tree=args.expected_self_tree,
                expected_result_branch=args.expected_self_branch,
            )
            _publish_immutable(
                args.result,
                result,
                name="capacity completion 100K result",
            )
            result_identity = file_identity(args.result)
            recommendation = evidence["recommended_next_stage"]
            write_json_report(
                args.status,
                _status(
                    state="completed",
                    detail="source_replayed_capacity_completion_100k_result_emitted",
                    self_git=self_git,
                    output_root=output_root,
                    standing_authorization=standing_status,
                    result_identity=result_identity,
                    recommendation=recommendation,
                ),
            )
            return
        except Exception as exc:
            write_json_report(
                args.status,
                _status(
                    state="failed",
                    detail="capacity_completion_100k_result_source_replay_failed",
                    self_git=self_git,
                    output_root=output_root,
                    standing_authorization=standing_status,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise


if __name__ == "__main__":
    main()
