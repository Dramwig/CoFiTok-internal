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
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    import build_generation_quality_bridge_matched_uncertainty_manifest as manifest_builder
    import run_generation_matched_uncertainty_waiter as base_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_quality_bridge_matched_uncertainty_manifest as manifest_builder,
    )
    from scripts import run_generation_matched_uncertainty_waiter as base_waiter


WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_quality_bridge_terminal_uncertainty_waiter"
EXPECTED_STREAM_ID = (
    "full_data_quality_bridge_terminal_100k_00000000_00010000"
)
PRECEDING_TERMINAL_STATUSES = {"pass", "hold", "failed"}
CLAIM_BOUNDARY = dict(base_waiter.CLAIM_BOUNDARY)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source(path: Path) -> dict[str, Any]:
    return file_identity(path)


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _sleep_until_next_poll(*, deadline: float, poll_seconds: float) -> bool:
    if time.monotonic() >= deadline:
        return False
    time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    return time.monotonic() < deadline


def validate_preceding_waiter_status(
    report: Mapping[str, Any],
    *,
    expected_output_root: Path,
    expected_pid: int,
    expected_control_git: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status", ""))
    phase = str(report.get("phase", ""))
    expected = report.get("expected")
    if (
        int(report.get("schema_version", -1))
        != base_waiter.WAITER_SCHEMA_VERSION
        or report.get("role") != base_waiter.WAITER_ROLE
        or int(report.get("pid", -1)) != expected_pid
        or status
        not in {"waiting", "running", *PRECEDING_TERMINAL_STATUSES}
        or report.get("claim_boundary") != base_waiter.CLAIM_BOUNDARY
        or not isinstance(expected, Mapping)
        or expected.get("output_root") != expected_output_root.as_posix()
        or expected.get("control_git") != dict(expected_control_git)
    ):
        raise ValueError("preceding matched-uncertainty waiter contract differs")
    if status in {"pass", "hold"} and phase != "completed":
        raise ValueError("preceding matched-uncertainty terminal phase differs")
    if status == "failed" and phase != "failed":
        raise ValueError("preceding matched-uncertainty failure phase differs")
    return {
        "status": status,
        "detail": report.get("detail"),
        "phase": phase,
        "pid": expected_pid,
        "terminal_status_published": status in PRECEDING_TERMINAL_STATUSES,
        "updated_at": report.get("updated_at"),
        "summary": report.get("summary"),
        "claim_boundary": report.get("claim_boundary"),
    }


def _validate_preceding_pid_file(
    report: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_control_revision: str,
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("role") != base_waiter.WAITER_ROLE
        or int(report.get("pid", -1)) != expected_pid
        or report.get("expected_control_revision")
        != expected_control_revision
    ):
        raise ValueError("preceding matched-uncertainty PID receipt differs")
    return dict(report)


def preceding_waiter_state(
    *,
    output_root: Path,
    expected_pid: int,
    expected_control_git: Mapping[str, Any],
    process_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    reports = output_root / "reports"
    status_path = reports / "waiter_status.json"
    pid_path = reports / "waiter.pid"
    if not status_path.is_file():
        return {
            "terminal": False,
            "status": "missing",
            "detail": "preceding_waiter_status_absent",
            "status_source": None,
            "pid_receipt": None,
            "active_processes": [],
        }

    validated = validate_preceding_waiter_status(
        read_json_object(
            status_path,
            name="preceding matched-uncertainty waiter status",
        ),
        expected_output_root=output_root,
        expected_pid=expected_pid,
        expected_control_git=expected_control_git,
    )
    pid_receipt: dict[str, Any] | None = None
    if pid_path.is_file():
        pid_receipt = _validate_preceding_pid_file(
            read_json_object(
                pid_path,
                name="preceding matched-uncertainty waiter PID",
            ),
            expected_pid=expected_pid,
            expected_control_revision=str(expected_control_git["revision"]),
        )

    rows = list(process_rows) if process_rows is not None else base_waiter._process_rows()
    related = base_waiter.uncertainty_processes(rows, output_root=output_root)
    by_pid = [dict(row) for row in rows if int(row.get("pid", -1)) == expected_pid]
    seen: set[tuple[int, str]] = set()
    active: list[dict[str, Any]] = []
    for row in [*by_pid, *related]:
        key = (int(row.get("pid", -1)), str(row.get("command", "")))
        if key not in seen:
            seen.add(key)
            active.append(dict(row))
    terminal = (
        validated["terminal_status_published"] is True
        and pid_receipt is None
        and not active
    )
    return {
        **validated,
        "terminal": terminal,
        "status_source": _source(status_path),
        "pid_receipt": pid_receipt,
        "active_processes": active,
    }


def _manifest_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity": dict(manifest["identity"]),
        "stream_id": manifest["stream_id"],
        "output": Path(manifest["output"]).as_posix(),
        "cache_root": Path(manifest["cache_root"]).as_posix(),
        "start_index": int(manifest["start_index"]),
        "end_index_exclusive": int(manifest["end_index_exclusive"]),
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
    quality: Mapping[str, Any] | None = None,
    preceding: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    cache: Mapping[str, Any] | None = None,
    stage: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
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
        "quality_bridge": dict(quality) if quality is not None else None,
        "preceding_uncertainty": (
            dict(preceding) if preceding is not None else None
        ),
        "execution_manifest": (
            dict(manifest) if manifest is not None else None
        ),
        "real_feature_cache": dict(cache) if cache is not None else None,
        "stage": dict(stage) if stage is not None else None,
        "audit": dict(audit) if audit is not None else None,
        "gpu_compute_rows": [dict(row) for row in gpu_rows or []],
        "competing_uncertainty_processes": [
            dict(row) for row in competing_rows or []
        ],
        "idle_polls": idle_polls,
        "required_idle_polls": int(expected["required_idle_polls"]),
        "claim_boundary": CLAIM_BOUNDARY,
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
            "expected_control_revision": expected["control_git"]["revision"],
            "expected_preceding_pid": expected["preceding_waiter"]["pid"],
            "started_at": _utc_now(),
        },
    )


def _remove_pid(path: Path) -> None:
    if not path.is_file():
        return
    try:
        payload = read_json_object(path, name="terminal uncertainty waiter PID")
    except ValueError:
        return
    if payload.get("role") == WAITER_ROLE and int(payload.get("pid", -1)) == os.getpid():
        path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact full-data 100K quality-bridge result and the "
            "preceding frozen-pair uncertainty chain, then run one source-bound "
            "non-authorizing terminal matched-uncertainty audit."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--evaluator-project", type=Path, required=True)
    parser.add_argument("--quality-project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--preceding-output-root", type=Path, required=True)
    parser.add_argument("--expected-preceding-pid", type=int, required=True)
    parser.add_argument("--expected-preceding-control-revision", required=True)
    parser.add_argument("--expected-preceding-control-tree", required=True)
    parser.add_argument("--expected-preceding-control-branch", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
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
    parser.add_argument("--expected-quality-revision", required=True)
    parser.add_argument("--expected-quality-tree", required=True)
    parser.add_argument("--expected-quality-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    args = parser.parse_args()
    if (
        args.expected_preceding_pid < 1
        or args.expected_real_feature_cache_bytes < 1
        or args.poll_seconds <= 0
        or args.required_idle_polls < 1
        or args.timeout_seconds <= 0
    ):
        parser.error("terminal uncertainty waiter arguments are invalid")
    return args


def _run_locked(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(args.project, name="control checkout").resolve()
    evaluator_project = reject_symlink_chain(
        args.evaluator_project,
        name="uncertainty evaluator checkout",
    ).resolve()
    quality_project = reject_symlink_chain(
        args.quality_project,
        name="quality-bridge checkout",
    ).resolve()
    quality_output_root = reject_symlink_chain(
        args.quality_output_root,
        name="quality-bridge output root",
    ).resolve()
    preceding_output_root = reject_symlink_chain(
        args.preceding_output_root,
        name="preceding uncertainty output root",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="terminal uncertainty output root",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="terminal uncertainty waiter status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="terminal uncertainty waiter PID",
    ).resolve()
    manifest_output = reject_symlink_chain(
        args.manifest_output,
        name="terminal uncertainty execution manifest",
    ).resolve()
    python_executable = reject_symlink_chain(
        args.python_executable,
        name="terminal uncertainty Python executable",
    ).resolve()
    real_feature_cache_source = reject_symlink_chain(
        args.real_feature_cache_source,
        name="terminal uncertainty real feature cache source",
    ).resolve()
    if (
        not _is_within(status_output, output_root)
        or not _is_within(pid_file, output_root)
        or not _is_within(manifest_output, output_root)
        or output_root == preceding_output_root
        or output_root == quality_output_root
        or not python_executable.is_file()
    ):
        raise ValueError("terminal uncertainty waiter path scope differs")

    def verify_checkouts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            base_waiter.verify_checkout(
                project,
                expected_revision=args.expected_control_revision,
                expected_tree=args.expected_control_tree,
                expected_branch=args.expected_control_branch,
                label="terminal uncertainty control",
            ),
            base_waiter.verify_checkout(
                evaluator_project,
                expected_revision=args.expected_evaluator_revision,
                expected_tree=args.expected_evaluator_tree,
                expected_branch=args.expected_evaluator_branch,
                label="terminal uncertainty evaluator",
            ),
            base_waiter.verify_checkout(
                quality_project,
                expected_revision=args.expected_quality_revision,
                expected_tree=args.expected_quality_tree,
                expected_branch=args.expected_quality_branch,
                label="quality bridge",
            ),
        )

    control_git, evaluator_git, quality_git = verify_checkouts()
    preceding_control_git = {
        "revision": args.expected_preceding_control_revision,
        "tree": args.expected_preceding_control_tree,
        "branch": args.expected_preceding_control_branch,
        "tracked_dirty": False,
    }
    expected = {
        "control_git": control_git,
        "evaluator_git": evaluator_git,
        "quality_git": quality_git,
        "quality_output_root": quality_output_root.as_posix(),
        "preceding_waiter": {
            "output_root": preceding_output_root.as_posix(),
            "pid": args.expected_preceding_pid,
            "control_git": preceding_control_git,
        },
        "output_root": output_root.as_posix(),
        "manifest_output": manifest_output.as_posix(),
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
    quality_verification: dict[str, Any] | None = None
    preceding_state: dict[str, Any] | None = None
    manifest_record: dict[str, Any] | None = None
    manifest_status: dict[str, Any] | None = None
    cache_receipt: dict[str, Any] | None = None
    stage_result: dict[str, Any] | None = None
    audit_result: dict[str, Any] | None = None

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
                quality=quality_verification,
                preceding=preceding_state,
                manifest=manifest_status,
                cache=cache_receipt,
                stage=stage_result,
                audit=audit_result,
                gpu_rows=gpu_rows,
                competing_rows=competing_rows,
                idle_polls=idle_polls,
            ),
        )

    quality_paths = base_waiter._quality_paths(quality_output_root)
    while quality_verification is None:
        verify_checkouts()
        if not quality_paths["execution_status"].is_file():
            publish(
                status="waiting",
                detail="waiting_for_quality_bridge_execution_status",
                phase="quality_bridge",
            )
        else:
            execution = base_waiter.validate_quality_execution_status(
                read_json_object(
                    quality_paths["execution_status"],
                    name="quality-bridge execution status",
                ),
                expected_revision=args.expected_quality_revision,
                expected_branch=args.expected_quality_branch,
            )
            if (
                execution["status"] != "completed"
                or not quality_paths["result"].is_file()
            ):
                publish(
                    status="waiting",
                    detail="waiting_for_quality_bridge_terminal_result",
                    phase="quality_bridge",
                )
            else:
                active = base_waiter.quality_processes(
                    base_waiter._process_rows(),
                    quality_project=quality_project,
                    quality_output_root=quality_output_root,
                )
                if active:
                    publish(
                        status="waiting",
                        detail="waiting_for_quality_bridge_process_exit",
                        phase="quality_bridge",
                        competing_rows=active,
                    )
                else:
                    quality_verification = base_waiter.verify_quality_bridge_result(
                        python_executable=python_executable,
                        quality_project=quality_project,
                        quality_output_root=quality_output_root,
                        expected_revision=args.expected_quality_revision,
                        expected_branch=args.expected_quality_branch,
                    )
                    publish(
                        status="running",
                        detail="quality_bridge_terminal_result_verified",
                        phase="manifest",
                    )
                    break
        if not _sleep_until_next_poll(
            deadline=deadline,
            poll_seconds=args.poll_seconds,
        ):
            publish(
                status="failed",
                detail="terminal_uncertainty_waiter_timeout",
                phase="quality_bridge",
            )
            return 1

    if quality_verification is None:
        raise RuntimeError("quality-bridge verification is absent")
    manifest_payload = manifest_builder.build_manifest(
        quality_result_path=Path(quality_verification["result"]["path"]),
        expected_quality_result_sha256=str(
            quality_verification["result"]["sha256"]
        ),
        expected_revision=args.expected_quality_revision,
        expected_branch=args.expected_quality_branch,
        output_root=output_root,
    )
    manifest_identity = manifest_builder.prepare_manifest(
        manifest_output,
        manifest_payload,
        resume=manifest_output.is_file(),
        overwrite=False,
    )
    manifest_record = base_waiter.validate_execution_manifest(
        manifest_output,
        expected_sha256=str(manifest_identity["sha256"]),
        output_root=output_root,
    )
    if manifest_record["stream_id"] != EXPECTED_STREAM_ID:
        raise ValueError("terminal uncertainty stream identity differs")
    manifest_status = _manifest_status(manifest_record)
    publish(
        status="running",
        detail="terminal_uncertainty_manifest_verified",
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

    def assert_quality_sources_unchanged() -> None:
        if (
            _source(quality_paths["execution_status"])
            != quality_verification["execution_status"]
            or _source(quality_paths["result"]) != quality_verification["result"]
            or _source(manifest_output) != manifest_record["identity"]
        ):
            raise ValueError("terminal uncertainty bound sources changed")

    while True:
        verify_checkouts()
        assert_quality_sources_unchanged()
        preceding_state = preceding_waiter_state(
            output_root=preceding_output_root,
            expected_pid=args.expected_preceding_pid,
            expected_control_git=preceding_control_git,
        )
        if preceding_state["terminal"] is True:
            publish(
                status="waiting",
                detail="preceding_uncertainty_chain_terminal",
                phase="gpu_slot",
            )
            break
        publish(
            status="waiting",
            detail="waiting_for_preceding_uncertainty_chain_terminal",
            phase="preceding_uncertainty",
            competing_rows=preceding_state.get("active_processes", []),
        )
        if not _sleep_until_next_poll(
            deadline=deadline,
            poll_seconds=args.poll_seconds,
        ):
            publish(
                status="failed",
                detail="terminal_uncertainty_waiter_timeout",
                phase="preceding_uncertainty",
            )
            return 1

    preceding_terminal_source = preceding_state["status_source"]
    idle_polls = 0
    while True:
        verify_checkouts()
        assert_quality_sources_unchanged()
        current_preceding = preceding_waiter_state(
            output_root=preceding_output_root,
            expected_pid=args.expected_preceding_pid,
            expected_control_git=preceding_control_git,
        )
        if (
            current_preceding["terminal"] is not True
            or current_preceding["status_source"] != preceding_terminal_source
        ):
            raise ValueError("preceding uncertainty terminal source changed")
        preceding_state = current_preceding
        processes = base_waiter._process_rows()
        competing = base_waiter.uncertainty_processes(
            processes,
            output_root=output_root,
        )
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
            final_preceding = preceding_waiter_state(
                output_root=preceding_output_root,
                expected_pid=args.expected_preceding_pid,
                expected_control_git=preceding_control_git,
            )
            final_gpu_rows = base_waiter._gpu_compute_rows()
            final_competing = base_waiter.uncertainty_processes(
                base_waiter._process_rows(),
                output_root=output_root,
            )
            if (
                final_preceding["terminal"] is not True
                or final_preceding["status_source"] != preceding_terminal_source
                or final_gpu_rows
                or final_competing
            ):
                idle_polls = 0
                preceding_state = final_preceding
                publish(
                    status="waiting",
                    detail="gpu_slot_changed_during_final_recheck",
                    phase="gpu_slot",
                    gpu_rows=final_gpu_rows,
                    competing_rows=final_competing,
                )
            else:
                preceding_state = final_preceding
                break
        if not _sleep_until_next_poll(
            deadline=deadline,
            poll_seconds=args.poll_seconds,
        ):
            publish(
                status="failed",
                detail="terminal_uncertainty_waiter_timeout",
                phase="gpu_slot",
                idle_polls=idle_polls,
            )
            return 1

    stage_spec = base_waiter.audit_stage_spec(
        name="terminal_100k",
        evaluator_project=evaluator_project,
        python_executable=python_executable,
        manifest=manifest_record,
        real_cache=real_cache,
        stage_root=output_root / "stage_receipts",
    )
    publish(
        status="running",
        detail="running_terminal_100k_matched_uncertainty_audit",
        phase="audit",
        idle_polls=idle_polls,
    )
    stage_result = base_waiter.run_stage(
        stage_spec,
        evaluator_project=evaluator_project,
        python_executable=python_executable,
    )
    audit_result = base_waiter.validate_audit_output(
        Path(manifest_record["output"]),
        manifest_identity=manifest_record["identity"],
        evaluator_git=evaluator_git,
    )
    if audit_result["status"] not in {"pass", "hold"}:
        raise ValueError("terminal uncertainty scientific status differs")
    verify_checkouts()
    assert_quality_sources_unchanged()
    final_preceding = preceding_waiter_state(
        output_root=preceding_output_root,
        expected_pid=args.expected_preceding_pid,
        expected_control_git=preceding_control_git,
    )
    if (
        final_preceding["terminal"] is not True
        or final_preceding["status_source"] != preceding_terminal_source
    ):
        raise ValueError("preceding uncertainty terminal source changed after audit")
    preceding_state = final_preceding
    publish(
        status=str(audit_result["status"]),
        detail=str(audit_result["decision"]),
        phase="completed",
    )
    return 0


def run_waiter(args: argparse.Namespace) -> int:
    output_root = reject_symlink_chain(
        args.output_root,
        name="terminal uncertainty output root",
    ).resolve()
    with exclusive_output_lock(output_root, role=WAITER_ROLE):
        try:
            return _run_locked(args)
        except Exception as error:
            status_output = reject_symlink_chain(
                args.status_output,
                name="terminal uncertainty waiter status",
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
                        "claim_boundary": CLAIM_BOUNDARY,
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
