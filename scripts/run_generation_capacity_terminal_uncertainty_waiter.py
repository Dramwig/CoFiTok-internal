from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    import build_generation_capacity_matched_uncertainty_manifest as manifest_builder
    import build_generation_capacity_statistical_claim_qualification as qualification_builder
    import run_generation_matched_uncertainty_waiter as base_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_capacity_matched_uncertainty_manifest as manifest_builder,
    )
    from scripts import (
        build_generation_capacity_statistical_claim_qualification as qualification_builder,
    )
    from scripts import run_generation_matched_uncertainty_waiter as base_waiter


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_capacity_terminal_uncertainty_waiter"
CLAIM_BOUNDARY = {
    **base_waiter.CLAIM_BOUNDARY,
    "training_launch_allowed": False,
    "release_allowed": False,
    "formal_large_scale_generation_advantage_claim_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _sleep_until_next_poll(*, deadline: float, poll_seconds: float) -> bool:
    if time.monotonic() >= deadline:
        return False
    time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return time.monotonic() < deadline


def global_uncertainty_processes(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    markers = (
        "audit_generation_matched_uncertainty.py",
        "build_generation_matched_uncertainty_summary.py",
        "run_generation_stage_once.py",
    )
    return [
        dict(row)
        for row in rows
        if int(row.get("pid", -1)) != os.getpid()
        and any(marker in str(row.get("command", "")) for marker in markers)
    ]


def _manifest_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    report = manifest["report"]
    source = report.get("capacity_source", {})
    matched = report["expected"]["matched_sampling"]
    return {
        "identity": dict(manifest["identity"]),
        "stream_id": manifest["stream_id"],
        "source_kind": source.get("kind"),
        "source_anchor": source.get("anchor"),
        "output": Path(manifest["output"]).as_posix(),
        "cache_root": Path(manifest["cache_root"]).as_posix(),
        "start_index": int(manifest["start_index"]),
        "end_index_exclusive": int(manifest["end_index_exclusive"]),
        "sample_count": int(matched["sample_count"]),
        "checkpoint_step": int(manifest["checkpoint_step"]),
        "cofitok_checkpoint_sha256": manifest["cofitok_checkpoint_sha256"],
        "dense_checkpoint_sha256": manifest["dense_checkpoint_sha256"],
        "real_set_sha256": manifest["real_set_sha256"],
    }


def _status(
    *,
    status: str,
    detail: str,
    phase: str,
    expected: Mapping[str, Any],
    source_anchor: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    cache: Mapping[str, Any] | None = None,
    stage: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
    qualification: Mapping[str, Any] | None = None,
    gpu_rows: Sequence[Mapping[str, Any]] | None = None,
    competing_rows: Sequence[Mapping[str, Any]] | None = None,
    idle_polls: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": WAITER_SCHEMA_VERSION,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "phase": phase,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "expected": dict(expected),
        "source_anchor": dict(source_anchor) if source_anchor is not None else None,
        "execution_manifest": dict(manifest) if manifest is not None else None,
        "real_feature_cache": dict(cache) if cache is not None else None,
        "stage": dict(stage) if stage is not None else None,
        "audit": dict(audit) if audit is not None else None,
        "statistical_claim_qualification": (
            dict(qualification) if qualification is not None else None
        ),
        "gpu_compute_rows": [dict(row) for row in gpu_rows or []],
        "competing_uncertainty_processes": [
            dict(row) for row in competing_rows or []
        ],
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "updated_at": _utc_now(),
    }


def _write_pid(path: Path, *, expected: Mapping[str, Any]) -> None:
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": WAITER_ROLE,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "source_kind": expected["source_kind"],
            "expected_control_revision": expected["control_git"]["revision"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        payload = read_json_object(path, name="capacity uncertainty waiter PID")
    except ValueError:
        return
    if payload.get("role") == WAITER_ROLE and int(payload.get("pid", -1)) == os.getpid():
        path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a source-bound capacity 100K result or capacity-full "
            "300K final gate, then run one non-authorizing matched uncertainty "
            "audit and statistical claim qualification."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--evaluator-project", type=Path, required=True)
    parser.add_argument(
        "--source-kind",
        choices=sorted(manifest_builder.SOURCE_KINDS),
        required=True,
    )
    parser.add_argument("--source-anchor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--qualification-output", type=Path, required=True)
    parser.add_argument("--gpu-slot-lock-target", type=Path, required=True)
    parser.add_argument("--real-feature-cache-source", type=Path, required=True)
    parser.add_argument(
        "--expected-real-feature-cache-bytes",
        type=int,
        default=base_waiter.REAL_FEATURE_CACHE_BYTES,
    )
    parser.add_argument(
        "--expected-real-feature-cache-sha256",
        default=base_waiter.REAL_FEATURE_CACHE_SHA256,
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-tree", required=True)
    parser.add_argument("--expected-evaluator-branch", default="")
    parser.add_argument("--expected-execution-revision", default="")
    parser.add_argument("--expected-execution-tree", default="")
    parser.add_argument("--expected-execution-branch", default="")
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-result-revision", default="")
    parser.add_argument("--expected-result-tree", default="")
    parser.add_argument("--expected-result-branch", default="")
    parser.add_argument("--expected-source-evaluation-revision", default="")
    parser.add_argument("--expected-source-evaluation-tree", default="")
    parser.add_argument("--expected-source-evaluation-branch", default="")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=31_536_000.0)
    args = parser.parse_args()
    if (
        args.expected_real_feature_cache_bytes < 1
        or args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.timeout_seconds <= 0
    ):
        parser.error("capacity uncertainty waiter arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(args.project, name="control checkout").resolve()
    evaluator_project = reject_symlink_chain(
        args.evaluator_project,
        name="uncertainty evaluator checkout",
    ).resolve()
    source_anchor_path = reject_symlink_chain(
        args.source_anchor,
        name="capacity uncertainty source anchor",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity uncertainty output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="capacity uncertainty waiter status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="capacity uncertainty waiter PID",
    ).resolve()
    manifest_output = reject_symlink_chain(
        args.manifest_output,
        name="capacity uncertainty execution manifest",
    ).resolve()
    qualification_output = reject_symlink_chain(
        args.qualification_output,
        name="capacity statistical claim qualification",
    ).resolve()
    gpu_slot_lock_target = reject_symlink_chain(
        args.gpu_slot_lock_target,
        name="matched uncertainty GPU slot",
    ).resolve()
    python_executable = reject_symlink_chain(
        args.python_executable,
        name="capacity uncertainty Python executable",
    ).resolve()
    real_feature_cache_source = reject_symlink_chain(
        args.real_feature_cache_source,
        name="capacity uncertainty real feature cache source",
    ).resolve()
    if (
        not _is_within(status_output, output_root)
        or not _is_within(pid_file, output_root)
        or not _is_within(manifest_output, output_root)
        or not _is_within(qualification_output, output_root)
        or gpu_slot_lock_target == output_root
        or not python_executable.is_file()
    ):
        raise ValueError("capacity uncertainty waiter path scope differs")

    def verify_checkouts() -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            base_waiter.verify_checkout(
                project,
                expected_revision=args.expected_control_revision,
                expected_tree=args.expected_control_tree,
                expected_branch=args.expected_control_branch,
                label="capacity uncertainty control",
            ),
            base_waiter.verify_checkout(
                evaluator_project,
                expected_revision=args.expected_evaluator_revision,
                expected_tree=args.expected_evaluator_tree,
                expected_branch=args.expected_evaluator_branch,
                label="capacity uncertainty evaluator",
            ),
        )

    control_git, evaluator_git = verify_checkouts()
    expected = {
        "control_git": control_git,
        "evaluator_git": evaluator_git,
        "source_kind": args.source_kind,
        "source_anchor_path": source_anchor_path.as_posix(),
        "source_git": {
            "execution": {
                "revision": args.expected_execution_revision,
                "tree": args.expected_execution_tree,
                "branch": args.expected_execution_branch,
            },
            "training": {
                "revision": args.expected_training_revision,
                "tree": args.expected_training_tree,
                "branch": args.expected_training_branch,
            },
            "result": {
                "revision": args.expected_result_revision,
                "tree": args.expected_result_tree,
                "branch": args.expected_result_branch,
            },
            "evaluation": {
                "revision": args.expected_source_evaluation_revision,
                "tree": args.expected_source_evaluation_tree,
                "branch": args.expected_source_evaluation_branch,
            },
        },
        "output_root": output_root.as_posix(),
        "manifest_output": manifest_output.as_posix(),
        "qualification_output": qualification_output.as_posix(),
        "gpu_slot_lock_target": gpu_slot_lock_target.as_posix(),
        "real_feature_cache_source": {
            "path": real_feature_cache_source.as_posix(),
            "bytes": args.expected_real_feature_cache_bytes,
            "sha256": args.expected_real_feature_cache_sha256,
        },
        "required_idle_polls": args.required_idle_polls,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_pid(pid_file, expected=expected)
    deadline = time.monotonic() + args.timeout_seconds
    anchor_identity: dict[str, Any] | None = None
    manifest_record: dict[str, Any] | None = None
    manifest_status: dict[str, Any] | None = None
    cache_receipt: dict[str, Any] | None = None
    stage_result: dict[str, Any] | None = None
    audit_result: dict[str, Any] | None = None
    qualification_record: dict[str, Any] | None = None

    def publish(
        *,
        status: str,
        detail: str,
        phase: str,
        gpu_rows: Sequence[Mapping[str, Any]] | None = None,
        competing_rows: Sequence[Mapping[str, Any]] | None = None,
        idle_polls: int = 0,
    ) -> None:
        write_json_report(
            status_output,
            _status(
                status=status,
                detail=detail,
                phase=phase,
                expected=expected,
                source_anchor=anchor_identity,
                manifest=manifest_status,
                cache=cache_receipt,
                stage=stage_result,
                audit=audit_result,
                qualification=qualification_record,
                gpu_rows=gpu_rows,
                competing_rows=competing_rows,
                idle_polls=idle_polls,
            ),
        )

    while not source_anchor_path.is_file():
        verify_checkouts()
        publish(
            status="waiting",
            detail="waiting_for_capacity_terminal_source_anchor",
            phase="source",
        )
        if not _sleep_until_next_poll(
            deadline=deadline,
            poll_seconds=args.poll_seconds,
        ):
            publish(
                status="failed",
                detail="capacity_uncertainty_waiter_timeout",
                phase="source",
            )
            return 1

    verify_checkouts()
    anchor_identity = file_identity(source_anchor_path)
    manifest_payload = manifest_builder.build_manifest(
        source_kind=args.source_kind,
        source_anchor_path=source_anchor_path,
        expected_source_anchor_sha256=anchor_identity["sha256"],
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
        expected_evaluation_revision=args.expected_source_evaluation_revision,
        expected_evaluation_tree=args.expected_source_evaluation_tree,
        expected_evaluation_branch=args.expected_source_evaluation_branch,
        output_root=output_root,
    )
    manifest_identity = prepare_manifest(
        manifest_output,
        manifest_payload,
        resume=manifest_output.is_file(),
        overwrite=False,
    )
    manifest_record = base_waiter.validate_execution_manifest(
        manifest_output,
        expected_sha256=manifest_identity["sha256"],
        output_root=output_root,
    )
    if manifest_record["report"].get("capacity_source", {}).get(
        "anchor"
    ) != anchor_identity:
        raise ValueError("capacity uncertainty manifest anchor differs")
    manifest_status = _manifest_status(manifest_record)
    publish(
        status="running",
        detail="capacity_uncertainty_manifest_verified",
        phase="cache_preparation",
    )

    real_cache = base_waiter._feature_cache_path(
        manifest_record["cache_root"],
        manifest_record["cache_names"]["real"],
    )
    cache_receipt = base_waiter.prepare_real_feature_cache(
        source=real_feature_cache_source,
        destination=real_cache,
        receipt_path=output_root / "reports" / "real_feature_cache_receipt.json",
        expected_bytes=args.expected_real_feature_cache_bytes,
        expected_sha256=args.expected_real_feature_cache_sha256,
    )

    def assert_bound_sources_unchanged() -> None:
        if file_identity(source_anchor_path) != anchor_identity:
            raise ValueError("capacity uncertainty source anchor changed")
        if file_identity(manifest_output) != manifest_record["identity"]:
            raise ValueError("capacity uncertainty execution manifest changed")
        for expected_source in manifest_record["bound_sources"]:
            if file_identity(Path(expected_source["path"])) != expected_source:
                raise ValueError("capacity uncertainty bound source changed")

    idle_polls = 0
    while True:
        verify_checkouts()
        assert_bound_sources_unchanged()
        rows = base_waiter._process_rows()
        competing = global_uncertainty_processes(rows)
        gpu_rows = base_waiter._gpu_compute_rows()
        idle_polls = base_waiter.next_idle_count(
            idle_polls,
            gpu_rows=gpu_rows,
            competing_processes=competing,
        )
        if idle_polls < args.required_idle_polls:
            publish(
                status="waiting",
                detail=(
                    "waiting_for_gpu_idle"
                    if gpu_rows or competing
                    else "confirming_gpu_idle"
                ),
                phase="gpu_slot",
                gpu_rows=gpu_rows,
                competing_rows=competing,
                idle_polls=idle_polls,
            )
        else:
            final_gpu_rows = base_waiter._gpu_compute_rows()
            final_competing = global_uncertainty_processes(
                base_waiter._process_rows()
            )
            if final_gpu_rows or final_competing:
                idle_polls = 0
                publish(
                    status="waiting",
                    detail="gpu_slot_changed_during_final_recheck",
                    phase="gpu_slot",
                    gpu_rows=final_gpu_rows,
                    competing_rows=final_competing,
                )
            else:
                try:
                    with exclusive_output_lock(
                        gpu_slot_lock_target,
                        role=f"{WAITER_ROLE}:{args.source_kind}",
                    ):
                        assert_bound_sources_unchanged()
                        final_gpu_rows = base_waiter._gpu_compute_rows()
                        final_competing = global_uncertainty_processes(
                            base_waiter._process_rows()
                        )
                        if final_gpu_rows or final_competing:
                            idle_polls = 0
                            publish(
                                status="waiting",
                                detail="gpu_slot_changed_after_lock",
                                phase="gpu_slot",
                                gpu_rows=final_gpu_rows,
                                competing_rows=final_competing,
                            )
                        else:
                            stage_spec = base_waiter.audit_stage_spec(
                                name=args.source_kind,
                                evaluator_project=evaluator_project,
                                python_executable=python_executable,
                                manifest=manifest_record,
                                real_cache=real_cache,
                                stage_root=output_root / "stage_receipts",
                            )
                            publish(
                                status="running",
                                detail=(
                                    "running_capacity_terminal_matched_"
                                    "uncertainty_audit"
                                ),
                                phase="audit",
                                idle_polls=idle_polls,
                            )
                            stage_result = base_waiter.run_stage(
                                stage_spec,
                                evaluator_project=evaluator_project,
                                python_executable=python_executable,
                            )
                            break
                except OutputLockError:
                    idle_polls = 0
                    publish(
                        status="waiting",
                        detail="waiting_for_shared_uncertainty_gpu_slot_lock",
                        phase="gpu_slot",
                    )
        if stage_result is not None:
            break
        if not _sleep_until_next_poll(
            deadline=deadline,
            poll_seconds=args.poll_seconds,
        ):
            publish(
                status="failed",
                detail="capacity_uncertainty_waiter_timeout",
                phase="gpu_slot",
                idle_polls=idle_polls,
            )
            return 1

    audit_result = base_waiter.validate_audit_output(
        Path(manifest_record["output"]),
        manifest_identity=manifest_record["identity"],
        evaluator_git=evaluator_git,
    )
    if audit_result["status"] not in {"pass", "hold"}:
        raise ValueError("capacity uncertainty scientific status differs")
    verify_checkouts()
    assert_bound_sources_unchanged()
    qualification_payload = qualification_builder.build_qualification(
        source_kind=args.source_kind,
        source_anchor_path=source_anchor_path,
        expected_source_anchor_sha256=anchor_identity["sha256"],
        execution_manifest_path=manifest_output,
        expected_execution_manifest_sha256=manifest_record["identity"]["sha256"],
        uncertainty_report_path=Path(audit_result["source"]["path"]),
        expected_uncertainty_report_sha256=audit_result["source"]["sha256"],
        expected_evaluator_revision=evaluator_git["revision"],
        expected_evaluator_branch=evaluator_git["branch"],
        output_root=output_root,
    )
    qualification_identity = prepare_manifest(
        qualification_output,
        qualification_payload,
        resume=qualification_output.is_file(),
        overwrite=False,
    )
    qualification_record = {
        "identity": qualification_identity,
        "status": qualification_payload["status"],
        "decision": qualification_payload["decision"],
        "claim_policy": qualification_payload["claim_policy"],
        "claim_boundary": qualification_payload["claim_boundary"],
    }
    publish(
        status=str(qualification_payload["status"]),
        detail=str(qualification_payload["decision"]),
        phase="completed",
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity uncertainty output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="capacity uncertainty waiter status",
            ).resolve()
            if _is_within(status_output, output_root):
                write_json_report(
                    status_output,
                    {
                        "schema_version": WAITER_SCHEMA_VERSION,
                        "role": WAITER_ROLE,
                        "status": "failed",
                        "detail": f"{type(error).__name__}: {error}",
                        "phase": "failed",
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "claim_boundary": dict(CLAIM_BOUNDARY),
                        "updated_at": _utc_now(),
                    },
                )
            raise
        finally:
            _remove_pid(args.pid_file.resolve())


def main() -> int:
    args = _parse_args()
    try:
        return run_waiter(args)
    except OutputLockError as error:
        print(str(error), file=sys.stderr)
        return base_waiter.LOCK_CONTENDED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
