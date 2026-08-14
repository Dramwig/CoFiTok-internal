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

from cofitok.generation.capacity_completion_decision import (
    validate_capacity_completion_decision,
)
from cofitok.generation.capacity_scaling_result import (
    validate_capacity_scaling_50k_result,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report

try:
    from scripts.build_generation_capacity_completion_decision import (
        build_from_sources as build_completion_decision,
    )
    from scripts.build_generation_capacity_scaling_50k_result import (
        _git_identity,
        build_from_sources as build_scaling_result,
    )
except ModuleNotFoundError:
    from build_generation_capacity_completion_decision import (
        build_from_sources as build_completion_decision,
    )
    from build_generation_capacity_scaling_50k_result import (
        _git_identity,
        build_from_sources as build_scaling_result,
    )


WAIT_BOUNDARY = {
    "cpu_only_waiter": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "process_signaling_allowed": False,
    "source_reports_must_be_physically_replayed": True,
    "existing_result_or_decision_must_be_byte_equivalent": True,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}


def source_paths(output_root: Path) -> dict[str, Path]:
    reports = output_root / "reports"
    return {
        "capacity_probe_result": reports / "capacity_probe_result.json",
        "capacity_scaling_decision": reports / "capacity_scaling_decision.json",
        "capacity_scaling_launch_receipt": (
            reports / "capacity_scaling_50k_launch_receipt.json"
        ),
        "capacity_scaling_execution_status": (
            reports / "capacity_scaling_50k_execution_status.json"
        ),
        "cofitok_training_validation": (
            reports / "capacity_scaling_50k/training/cofitok.json"
        ),
        "dense_training_validation": (
            reports / "capacity_scaling_50k/training/dense_identity.json"
        ),
        "milestone_50000": (
            reports / "capacity_scaling_50k/milestone_step_00050000.json"
        ),
        "capacity_scaling_result": reports / "capacity_scaling_50k_result.json",
        "capacity_completion_decision": (
            reports / "capacity_completion_100k_decision.json"
        ),
    }


def _publish_immutable(path: Path, payload: dict[str, Any], *, name: str) -> None:
    if path.exists():
        actual = read_json_object(path, name=name)
        expected_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if os.linesep != "\n":
            expected_text = expected_text.replace("\n", os.linesep)
        expected_bytes = expected_text.encode("utf-8")
        if actual != payload or path.read_bytes() != expected_bytes:
            raise ValueError(f"existing {name} is not byte-equivalent to replay")
        return
    write_json_report(path, payload)


def _status(
    *,
    state: str,
    detail: str,
    self_git: dict[str, Any],
    output_root: Path,
    source_identities: dict[str, dict[str, Any]] | None = None,
    result_identity: dict[str, Any] | None = None,
    decision_identity: dict[str, Any] | None = None,
    execution_authorized: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "capacity_completion_100k_source_replay_waiter",
        "status": state,
        "detail": detail,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git": copy.deepcopy(self_git),
        "output_root": output_root.resolve().as_posix(),
        "source_identities": copy.deepcopy(source_identities),
        "capacity_scaling_result": copy.deepcopy(result_identity),
        "capacity_completion_decision": copy.deepcopy(decision_identity),
        "execution_authorized": execution_authorized,
        "error": error,
        "authorization_boundary": copy.deepcopy(WAIT_BOUNDARY),
    }


def _builder_args(
    args: argparse.Namespace,
    paths: dict[str, Path],
    identities: dict[str, dict[str, Any]],
) -> SimpleNamespace:
    result_path = paths["capacity_scaling_result"]
    result_sha = (
        file_identity(result_path)["sha256"] if result_path.is_file() else None
    )
    return SimpleNamespace(
        capacity_probe_result=paths["capacity_probe_result"],
        expected_capacity_probe_result_sha256=identities[
            "capacity_probe_result"
        ]["sha256"],
        capacity_scaling_decision=paths["capacity_scaling_decision"],
        expected_capacity_scaling_decision_sha256=identities[
            "capacity_scaling_decision"
        ]["sha256"],
        capacity_scaling_launch_receipt=paths[
            "capacity_scaling_launch_receipt"
        ],
        expected_capacity_scaling_launch_receipt_sha256=identities[
            "capacity_scaling_launch_receipt"
        ]["sha256"],
        execution_status=paths["capacity_scaling_execution_status"],
        expected_execution_status_sha256=identities[
            "capacity_scaling_execution_status"
        ]["sha256"],
        cofitok_training_validation=paths["cofitok_training_validation"],
        expected_cofitok_training_validation_sha256=identities[
            "cofitok_training_validation"
        ]["sha256"],
        dense_training_validation=paths["dense_training_validation"],
        expected_dense_training_validation_sha256=identities[
            "dense_training_validation"
        ]["sha256"],
        milestone_50000=paths["milestone_50000"],
        expected_milestone_50000_sha256=identities["milestone_50000"][
            "sha256"
        ],
        standing_authorization=args.standing_authorization,
        expected_standing_authorization_sha256=(
            args.expected_standing_authorization_sha256
        ),
        training_project=args.training_project,
        expected_capacity_revision=args.expected_training_revision,
        expected_capacity_tree=args.expected_training_tree,
        expected_capacity_branch=args.expected_training_branch,
        expected_decision_revision=args.expected_scaling_decision_revision,
        expected_decision_branch=args.expected_scaling_decision_branch,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_result_revision=args.expected_self_revision,
        expected_result_tree=args.expected_self_tree,
        expected_result_branch=args.expected_self_branch,
        capacity_scaling_50k_result=result_path,
        expected_capacity_scaling_50k_result_sha256=result_sha,
        expected_completion_decision_revision=args.expected_self_revision,
        expected_completion_decision_tree=args.expected_self_tree,
        expected_completion_decision_branch=args.expected_self_branch,
    )


def _execution_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    report = read_json_object(path, name="capacity scaling 50K execution status")
    status = report.get("status")
    if status in {"failed", "error", "cancelled"}:
        raise RuntimeError(
            "capacity scaling 50K execution is terminal: "
            f"{status}: {report.get('detail')}"
        )
    return status == "completed" and report.get("stage") == "complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait without using a GPU, physically replay the completed 250M "
            "step-50K evidence, and emit the bounded step-100K decision."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--training-project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-self-revision", required=True)
    parser.add_argument("--expected-self-tree", required=True)
    parser.add_argument("--expected-self-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-scaling-decision-revision", required=True)
    parser.add_argument("--expected-scaling-decision-branch", required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--expected-execution-tree", required=True)
    parser.add_argument("--expected-execution-branch", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 1:
        raise ValueError("poll seconds must be positive")
    project = args.project.resolve()
    output_root = args.output_root.resolve()
    paths = source_paths(output_root)
    self_git = _git_identity(project)
    expected_self = {
        "revision": args.expected_self_revision,
        "tree": args.expected_self_tree,
        "branch": args.expected_self_branch,
        "tracked_dirty": False,
    }
    if self_git != expected_self:
        raise ValueError("capacity completion waiter checkout identity differs")
    if project != Path(__file__).resolve().parents[1]:
        raise ValueError("capacity completion waiter project path differs")
    standing_identity = file_identity(args.standing_authorization)
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    args.status.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            if not _execution_completed(paths["capacity_scaling_execution_status"]):
                write_json_report(
                    args.status,
                    _status(
                        state="waiting",
                        detail="waiting_for_completed_capacity_scaling_50k_execution",
                        self_git=self_git,
                        output_root=output_root,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            required = {
                name: path
                for name, path in paths.items()
                if name
                not in {
                    "capacity_scaling_result",
                    "capacity_completion_decision",
                }
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                write_json_report(
                    args.status,
                    _status(
                        state="waiting",
                        detail="waiting_for_capacity_scaling_sources:" + ",".join(missing),
                        self_git=self_git,
                        output_root=output_root,
                    ),
                )
                time.sleep(args.poll_seconds)
                continue
            identities = {
                name: file_identity(path) for name, path in required.items()
            }
            builder_args = _builder_args(args, paths, identities)
            scaling_result = build_scaling_result(builder_args)
            validate_capacity_scaling_50k_result(
                scaling_result,
                expected_execution_revision=args.expected_execution_revision,
                expected_execution_tree=args.expected_execution_tree,
                expected_execution_branch=args.expected_execution_branch,
                expected_result_revision=args.expected_self_revision,
                expected_result_tree=args.expected_self_tree,
                expected_result_branch=args.expected_self_branch,
            )
            _publish_immutable(
                paths["capacity_scaling_result"],
                scaling_result,
                name="capacity scaling 50K result",
            )
            result_identity = file_identity(paths["capacity_scaling_result"])
            builder_args.expected_capacity_scaling_50k_result_sha256 = (
                result_identity["sha256"]
            )
            completion_decision = build_completion_decision(builder_args)
            evidence = validate_capacity_completion_decision(
                completion_decision,
                expected_decision_revision=args.expected_self_revision,
                expected_decision_tree=args.expected_self_tree,
                expected_decision_branch=args.expected_self_branch,
            )
            _publish_immutable(
                paths["capacity_completion_decision"],
                completion_decision,
                name="capacity completion 100K decision",
            )
            decision_identity = file_identity(
                paths["capacity_completion_decision"]
            )
            write_json_report(
                args.status,
                _status(
                    state="completed",
                    detail=(
                        "source_replayed_capacity_completion_decision_emitted"
                    ),
                    self_git=self_git,
                    output_root=output_root,
                    source_identities=identities,
                    result_identity=result_identity,
                    decision_identity=decision_identity,
                    execution_authorized=evidence["execution_authorized"],
                ),
            )
            return
        except Exception as exc:
            write_json_report(
                args.status,
                _status(
                    state="failed",
                    detail="capacity_completion_source_replay_failed",
                    self_git=self_git,
                    output_root=output_root,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise


if __name__ == "__main__":
    main()
