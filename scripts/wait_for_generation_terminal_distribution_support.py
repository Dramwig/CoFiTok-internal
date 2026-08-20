from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
)
from cofitok.generation.terminal_distribution_support import (
    TERMINAL_DISTRIBUTION_SUPPORT_ROUTE,
    validate_terminal_distribution_support_report,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report


ROLE = "generation_quality_bridge_terminal_distribution_support_waiter"
SCHEMA_VERSION = 1
RUNBOOK_NAME = "generation_quality_bridge_terminal_distribution_support_waiter.sh"
DIAGNOSTIC_SCRIPT_NAME = "diagnose_generation_terminal_distribution_support.py"
DECISION_NAME = "followup_experiment_decision_exposure_aware_v2.json"
RESULT_NAME = "quality_bridge_result.json"
OUTPUT_RELATIVE_PATH = Path("reports/terminal_distribution_support_v1/report.json")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the canonical quality-bridge v2 follow-up decision and run "
            "the CPU-only terminal distribution-support diagnostic only when its "
            "exact route is selected."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-bridge-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-self-sha256", required=True)
    parser.add_argument("--expected-followup-revision", required=True)
    parser.add_argument("--expected-followup-branch", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-polls", type=int, default=0)
    parser.add_argument("--nearest-chunk-size", type=int, default=128)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    if args.max_polls < 0:
        parser.error("--max-polls must be nonnegative")
    if args.nearest_chunk_size < 1:
        parser.error("--nearest-chunk-size must be positive")
    return args


def _base_status(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "project": str(args.project),
        "quality_bridge_root": str(args.quality_bridge_root),
        "poll_seconds": args.poll_seconds,
        "scope": {
            "source_observation_allowed": True,
            "diagnostic_execution_allowed_only_if_selected": True,
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "evaluation_launch_allowed": False,
            "recipe_probe_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
            "full_300k_launch_allowed": False,
            "process_signals_allowed": False,
        },
    }


def _write_status(
    args: argparse.Namespace,
    base: dict[str, Any],
    *,
    status: str,
    detail: str,
    **extra: Any,
) -> None:
    write_json_report(
        args.status,
        {
            **base,
            "status": status,
            "detail": detail,
            "updated_at": _utc_now(),
            **extra,
        },
    )


def _validate_checkout(args: argparse.Namespace) -> dict[str, Any]:
    if not args.project.is_dir():
        raise ValueError(f"project is missing: {args.project}")
    observed = {
        "revision": _run(["git", "rev-parse", "HEAD"], cwd=args.project),
        "tree": _run(["git", "rev-parse", "HEAD^{tree}"], cwd=args.project),
        "branch": _run(["git", "branch", "--show-current"], cwd=args.project),
        "tracked_dirty": bool(
            _run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=args.project,
            )
        ),
    }
    expected = {
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if observed != expected:
        raise ValueError(
            "terminal distribution-support checkout differs: "
            + json.dumps(observed, sort_keys=True)
        )
    runbook = args.project / "artifacts" / "runbooks" / RUNBOOK_NAME
    diagnostic = args.project / "scripts" / DIAGNOSTIC_SCRIPT_NAME
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    if not diagnostic.is_file():
        raise FileNotFoundError(diagnostic)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    return {
        **observed,
        "runbook_sha256": _sha256(runbook),
        "diagnostic_script_sha256": _sha256(diagnostic),
    }


def _decision_route(decision: dict[str, Any]) -> str:
    recommendation = decision.get("recommended_next_stage")
    if (
        int(decision.get("schema_version", -1)) != FOLLOWUP_DECISION_SCHEMA_VERSION
        or decision.get("status") != "completed"
        or decision.get("role") != FOLLOWUP_DECISION_ROLE
        or decision.get("authorization_boundary")
        != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, dict)
        or not isinstance(recommendation.get("id"), str)
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("quality bridge follow-up decision boundary differs")
    return str(recommendation["id"])


def _verify_report_sources(report: dict[str, Any]) -> None:
    from cofitok.inference_replay import reject_symlink_chain
    from scripts.evaluate_generation_metrics import (
        find_images,
        validate_sampling_provenance,
    )

    sources = report.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError("terminal distribution-support sources are missing")
    identities = [
        sources.get("quality_bridge_followup_decision"),
        sources.get("quality_bridge_result"),
    ]
    methods = sources.get("methods")
    if not isinstance(methods, dict) or set(methods) != {"cofitok", "dense_identity"}:
        raise ValueError("terminal distribution-support method sources differ")
    for method in ("cofitok", "dense_identity"):
        row = methods[method]
        if not isinstance(row, dict):
            raise ValueError(f"terminal distribution-support {method} sources differ")
        identities.extend(
            row.get(name)
            for name in (
                "metrics_report",
                "sampling_report",
                "sampling_manifest",
                "sampling_progress",
            )
        )
        generated_dir = reject_symlink_chain(
            str(row.get("generated_dir", "")),
            name=f"terminal distribution-support {method} generated samples",
        ).resolve()
        sampling_report = row.get("sampling_report")
        if not isinstance(sampling_report, dict):
            raise ValueError(
                f"terminal distribution-support {method} sampling report differs"
            )
        physical = validate_sampling_provenance(
            Path(str(sampling_report.get("path", ""))),
            generated_dir,
            find_images(generated_dir),
        )
        if (
            physical.get("sample_set_sha256") != row.get("sample_set_sha256")
            or int(physical.get("checkpoint_step", -1))
            != int(row.get("checkpoint_step", -1))
            or physical.get("checkpoint_sha256") != row.get("checkpoint_sha256")
        ):
            raise ValueError(
                f"terminal distribution-support {method} physical samples changed"
            )
    for declared in identities:
        if not isinstance(declared, dict):
            raise ValueError("terminal distribution-support source identity is missing")
        if file_identity(str(declared.get("path", ""))) != declared:
            raise ValueError("terminal distribution-support source identity changed")


def _postcondition(
    output: Path,
    *,
    decision_identity: dict[str, Any],
    result_identity: dict[str, Any],
) -> dict[str, Any]:
    report = read_json_object(output, name="terminal distribution-support report")
    summary = validate_terminal_distribution_support_report(
        report,
        expected_followup_identity=decision_identity,
        expected_quality_bridge_identity=result_identity,
    )
    _verify_report_sources(report)
    return summary


def main() -> int:
    args = parse_args()
    base = _base_status(args)
    observed_self_sha256 = _sha256(Path(__file__).resolve())
    if observed_self_sha256 != args.expected_self_sha256:
        _write_status(
            args,
            base,
            status="failed",
            detail="waiter_self_sha256_mismatch",
            expected_self_sha256=args.expected_self_sha256,
            observed_self_sha256=observed_self_sha256,
        )
        return 81
    try:
        checkout = _validate_checkout(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _write_status(
            args,
            base,
            status="failed",
            detail="diagnostic_checkout_validation_failed",
            error=str(error),
            waiter_sha256=observed_self_sha256,
        )
        return 82
    base.update({"waiter_sha256": observed_self_sha256, "git": checkout})

    reports = args.quality_bridge_root / "reports"
    decision_path = reports / DECISION_NAME
    result_path = reports / RESULT_NAME
    output = args.quality_bridge_root / OUTPUT_RELATIVE_PATH
    polls = 0
    while not decision_path.is_file():
        polls += 1
        _write_status(
            args,
            base,
            status="waiting",
            detail="waiting_for_quality_bridge_followup_decision",
            polls=polls,
            decision_exists=False,
            output_exists=output.is_file(),
        )
        if args.max_polls and polls >= args.max_polls:
            _write_status(
                args,
                base,
                status="stopped",
                detail="bounded_wait_completed_without_followup_decision",
                polls=polls,
                decision_exists=False,
                output_exists=output.is_file(),
            )
            return 78
        time.sleep(args.poll_seconds)

    try:
        decision = read_json_object(
            decision_path, name="quality bridge follow-up decision"
        )
        route = _decision_route(decision)
        decision_identity = file_identity(decision_path)
    except (OSError, TypeError, ValueError) as error:
        _write_status(
            args,
            base,
            status="failed",
            detail="followup_decision_validation_failed",
            polls=polls,
            error=str(error),
        )
        return 83
    if route != TERMINAL_DISTRIBUTION_SUPPORT_ROUTE:
        _write_status(
            args,
            base,
            status="completed",
            detail="terminal_distribution_support_route_not_selected",
            polls=polls,
            selected_route=route,
            decision=decision_identity,
            diagnostic_executed=False,
            output_exists=output.is_file(),
        )
        return 0

    declared_result = decision.get("source_reports", {}).get(
        "quality_bridge_result"
    )
    try:
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result_identity = file_identity(result_path)
        if result_identity != declared_result:
            raise ValueError("selected decision binds another quality bridge result")
    except (OSError, TypeError, ValueError) as error:
        _write_status(
            args,
            base,
            status="failed",
            detail="selected_route_quality_bridge_binding_failed",
            polls=polls,
            decision=decision_identity,
            error=str(error),
        )
        return 84

    if output.is_file():
        try:
            postcondition = _postcondition(
                output,
                decision_identity=decision_identity,
                result_identity=result_identity,
            )
        except (OSError, TypeError, ValueError) as error:
            _write_status(
                args,
                base,
                status="failed",
                detail="existing_terminal_distribution_support_report_invalid",
                polls=polls,
                error=str(error),
            )
            return 85
        _write_status(
            args,
            base,
            status="completed",
            detail="existing_terminal_distribution_support_report_verified",
            polls=polls,
            diagnostic_executed=False,
            report=file_identity(output),
            verification=postcondition,
        )
        return 0

    diagnostic = args.project / "scripts" / DIAGNOSTIC_SCRIPT_NAME
    command = [
        str(args.python),
        str(diagnostic),
        "--project",
        str(args.project),
        "--followup-decision",
        str(decision_path),
        "--expected-followup-decision-sha256",
        decision_identity["sha256"],
        "--expected-followup-revision",
        args.expected_followup_revision,
        "--expected-followup-branch",
        args.expected_followup_branch,
        "--quality-bridge-result",
        str(result_path),
        "--expected-quality-bridge-result-sha256",
        result_identity["sha256"],
        "--expected-diagnostic-revision",
        args.expected_revision,
        "--expected-diagnostic-branch",
        args.expected_branch,
        "--nearest-chunk-size",
        str(args.nearest_chunk_size),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    _write_status(
        args,
        base,
        status="running",
        detail="running_selected_terminal_distribution_support_diagnostic",
        polls=polls,
        selected_route=route,
        decision=decision_identity,
        quality_bridge_result=result_identity,
        output=str(output),
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{_utc_now()}] decision_sha256={decision_identity['sha256']} "
            f"quality_bridge_result_sha256={result_identity['sha256']}\n"
        )
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(args.project),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"[{_utc_now()}] exit_code={completed.returncode}\n")
    if completed.returncode != 0:
        _write_status(
            args,
            base,
            status="failed",
            detail="terminal_distribution_support_diagnostic_failed",
            polls=polls,
            exit_code=completed.returncode,
            log=str(args.log),
        )
        return completed.returncode
    try:
        postcondition = _postcondition(
            output,
            decision_identity=decision_identity,
            result_identity=result_identity,
        )
    except (OSError, TypeError, ValueError) as error:
        _write_status(
            args,
            base,
            status="failed",
            detail="terminal_distribution_support_postcondition_failed",
            polls=polls,
            error=str(error),
            log=str(args.log),
        )
        return 86
    _write_status(
        args,
        base,
        status="completed",
        detail="terminal_distribution_support_diagnostic_verified",
        polls=polls,
        diagnostic_executed=True,
        report=file_identity(output),
        verification=postcondition,
        log=str(args.log),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
