from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows test import path
    fcntl = None

from cofitok.reporting import file_sha256, write_json_report
from scripts.build_generation_milestone_report import (
    build_report,
    source_report_identity,
    validate_milestone_report,
    verify_milestone_source_reports,
)


WAITER_ROLE = "generation_quality_bridge_50k_milestone_claim_guard_waiter"
SUMMARY_ROLE = "generation_quality_bridge_50k_milestone_claim_guard"
RECEIPT_ROLE = "generation_quality_bridge_50k_milestone_claim_guard_receipt"
MILESTONE_STEP = 50_000
EXPECTED_SAMPLES = 2_048
SOURCE_PROFILE = "quality_bridge"
OPERATION_SCOPE = {
    "read_only_training_sources": True,
    "diagnostic_artifact_writes_only": True,
    "gpu_required": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "training_launch_allowed": False,
    "process_signals_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
}
CLAIM_BOUNDARY = {
    "early_warning_only": True,
    "formal_generation_quality_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "sampling_launch_allowed": False,
    "training_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "process_signals_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "cross_tier_numeric_ranking_allowed": False,
    "replaces_terminal_100k_result": False,
    "requires_terminal_100k_10000_sample_uncertainty": True,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON source is not an object: {path}")
    return payload


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"file does not exist: {resolved}")
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _git_value(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    expected_branch: str,
    expected_tree: str,
) -> dict[str, Any]:
    resolved = checkout.resolve()
    if not resolved.is_dir():
        raise ValueError(f"control checkout does not exist: {resolved}")
    observed = {
        "path": resolved.as_posix(),
        "revision": _git_value(resolved, "rev-parse", "HEAD"),
        "branch": _git_value(resolved, "branch", "--show-current"),
        "tree": _git_value(resolved, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": bool(
            _git_value(resolved, "status", "--porcelain", "--untracked-files=no")
        ),
    }
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tree": expected_tree,
        "tracked_dirty": False,
    }
    for field, value in expected.items():
        if observed[field] != value:
            raise ValueError(f"control checkout {field} differs")
    return observed


def verify_control_identity(args: argparse.Namespace) -> dict[str, Any]:
    checkout = verify_checkout(
        args.project,
        expected_revision=args.expected_control_revision,
        expected_branch=args.expected_control_branch,
        expected_tree=args.expected_control_tree,
    )
    source = Path(__file__).resolve()
    expected_source = (
        args.project
        / "scripts/wait_for_generation_quality_bridge_50k_milestone_claim_guard.py"
    ).resolve()
    if source != expected_source:
        raise ValueError("control source is outside the bound checkout")
    if file_sha256(source) != args.expected_control_source_sha256:
        raise ValueError("control source SHA256 differs")
    return {"checkout": checkout, "source": file_identity(source)}


def _require_training_git(
    payload: dict[str, Any],
    *,
    label: str,
    expected_revision: str,
    expected_branch: str,
) -> None:
    git = payload.get("git")
    if not isinstance(git, dict):
        raise ValueError(f"{label} lacks Git provenance")
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    for field, value in expected.items():
        if git.get(field) != value:
            raise ValueError(f"{label} Git {field} differs")


def rebuild_and_validate_milestone(
    milestone_path: Path,
    *,
    expected_training_revision: str,
    expected_training_branch: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = read_object(milestone_path)
    source_verification = verify_milestone_source_reports(
        report,
        source_profile=SOURCE_PROFILE,
    )
    evidence, warnings = validate_milestone_report(
        report,
        expected_step=MILESTONE_STEP,
        source_verification=source_verification,
        expected_source_profile=SOURCE_PROFILE,
    )

    source_reports = source_verification["source_reports"]
    payloads = {
        name: read_object(Path(identity["path"]))
        for name, identity in source_reports.items()
    }
    for name, payload in payloads.items():
        _require_training_git(
            payload,
            label=name,
            expected_revision=expected_training_revision,
            expected_branch=expected_training_branch,
        )

    rebuilt = build_report(
        cofitok_generation=payloads["cofitok_generation"],
        dense_generation=payloads["dense_generation"],
        cofitok_checkpoint_eval=payloads["cofitok_checkpoint_eval"],
        dense_checkpoint_eval=payloads["dense_checkpoint_eval"],
        source_reports={
            name: source_report_identity(identity["path"])
            for name, identity in source_reports.items()
        },
        milestone_step=MILESTONE_STEP,
        expected_samples=EXPECTED_SAMPLES,
        source_profile=SOURCE_PROFILE,
    )
    if rebuilt != report:
        raise ValueError("milestone report is not an exact replay of its four sources")
    replay = {
        "status": "byte_equivalent_payload",
        "canonical_sha256": canonical_sha256(rebuilt),
        "source_report_sha256": evidence["source_report_sha256"],
        "warnings": warnings,
    }
    return report, source_verification, replay


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite milestone metric: {label}")
    return result


def _direction(lower_is_better: bool, left: float, right: float) -> str:
    if left == right:
        return "tie"
    left_better = left < right if lower_is_better else left > right
    return "cofitok" if left_better else "dense_identity"


def build_claim_guard_summary(
    milestone_path: Path,
    *,
    control: dict[str, Any],
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    report, source_verification, replay = rebuild_and_validate_milestone(
        milestone_path,
        expected_training_revision=expected_training_revision,
        expected_training_branch=expected_training_branch,
    )
    methods = report["methods"]
    cofitok = methods["cofitok"]
    dense = methods["dense_identity"]

    cofitok_fid = _finite(cofitok["fid"], label="cofitok_fid")
    dense_fid = _finite(dense["fid"], label="dense_fid")
    cofitok_is = _finite(cofitok["inception_score"], label="cofitok_is")
    dense_is = _finite(dense["inception_score"], label="dense_is")
    cofitok_endpoint = _finite(
        cofitok["endpoint_clean_mse"], label="cofitok_endpoint_clean_mse"
    )
    dense_endpoint = _finite(
        dense["endpoint_clean_mse"], label="dense_endpoint_clean_mse"
    )

    return {
        "schema_version": 1,
        "role": SUMMARY_ROLE,
        "status": "pass",
        "operational_scope": dict(OPERATION_SCOPE),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "milestone": {
            "step": MILESTONE_STEP,
            "sample_count_per_method": EXPECTED_SAMPLES,
            "sampling_protocol": "DDIM-50 CFG=1.5 EMA bf16",
            "report": file_identity(milestone_path),
            "source_profile": SOURCE_PROFILE,
            "source_verification": source_verification,
            "exact_replay": replay,
        },
        "training_identity": {
            "revision": expected_training_revision,
            "branch": expected_training_branch,
        },
        "control": control,
        "descriptive_quality": {
            "fid": {
                "cofitok": cofitok_fid,
                "dense_identity": dense_fid,
                "cofitok_minus_dense": cofitok_fid - dense_fid,
                "relative_change": (cofitok_fid - dense_fid)
                / max(dense_fid, 1e-12),
                "lower_value": _direction(True, cofitok_fid, dense_fid),
            },
            "inception_score": {
                "cofitok": cofitok_is,
                "dense_identity": dense_is,
                "cofitok_minus_dense": cofitok_is - dense_is,
                "relative_change": (cofitok_is - dense_is)
                / max(abs(dense_is), 1e-12),
                "higher_value": _direction(False, cofitok_is, dense_is),
            },
        },
        "descriptive_mechanism": {
            "endpoint_clean_mse": {
                "cofitok": cofitok_endpoint,
                "dense_identity": dense_endpoint,
                "cofitok_minus_dense": cofitok_endpoint - dense_endpoint,
                "relative_change": (cofitok_endpoint - dense_endpoint)
                / max(dense_endpoint, 1e-12),
                "lower_value": _direction(True, cofitok_endpoint, dense_endpoint),
            },
            "cofitok_prefix_path_mse_auc": _finite(
                cofitok["prefix_path_mse_auc"],
                label="cofitok_prefix_path_mse_auc",
            ),
            "cofitok_ordered_rank_by_path_auc": int(
                cofitok["ordered_rank_by_path_auc"]
            ),
            "cofitok_order_count": int(cofitok["order_count"]),
            "cofitok_zero_token_max_abs": _finite(
                cofitok["zero_token_max_abs"], label="cofitok_zero_token_max_abs"
            ),
            "cofitok_shuffled_to_ordered_endpoint_ratio": _finite(
                cofitok["shuffled_to_ordered_endpoint_ratio"],
                label="cofitok_shuffled_to_ordered_endpoint_ratio",
            ),
        },
        "quality_alerts": list(report["quality_alerts"]),
        "scientific_interpretation": {
            "generation_advantage_proven": False,
            "formal_quality_conclusion": "pending_terminal_100k_matched_uncertainty",
            "allowed_use": "descriptive_matched_50k_early_warning_only",
            "required_next_evidence": [
                "matched 100K training completion",
                "matched 10,000-sample DDIM-100 terminal metrics",
                "source-bound matched uncertainty analysis",
                "class-fidelity qualification",
            ],
        },
    }


def _write_or_verify(path: Path, payload: dict[str, Any], *, label: str) -> None:
    if path.exists():
        if read_object(path) != payload:
            raise ValueError(f"existing {label} differs")
        return
    write_json_report(path, payload)


def _status_payload(
    args: argparse.Namespace,
    *,
    status: str,
    detail: str,
    control: dict[str, Any] | None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "milestone_report": args.milestone_report.resolve().as_posix(),
        "summary_output": args.summary_output.resolve().as_posix(),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "operational_scope": dict(OPERATION_SCOPE),
        "control": control,
    }
    if summary is not None:
        payload["summary"] = summary
    if error is not None:
        payload["error"] = error
    return payload


def write_deployment_receipt(
    args: argparse.Namespace,
    *,
    control: dict[str, Any],
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "role": RECEIPT_ROLE,
        "status": "deployed",
        "control": control,
        "milestone_report": args.milestone_report.resolve().as_posix(),
        "summary_output": args.summary_output.resolve().as_posix(),
        "status_output": args.status_output.resolve().as_posix(),
        "expected_training": {
            "revision": args.expected_training_revision,
            "branch": args.expected_training_branch,
            "milestone_step": MILESTONE_STEP,
            "sample_count_per_method": EXPECTED_SAMPLES,
            "source_profile": SOURCE_PROFILE,
        },
        "operational_scope": dict(OPERATION_SCOPE),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    _write_or_verify(
        args.deployment_receipt_output,
        receipt,
        label="deployment receipt",
    )
    return receipt


def run_once(
    args: argparse.Namespace,
    *,
    control: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if not args.milestone_report.is_file():
        write_json_report(
            args.status_output,
            _status_payload(
                args,
                status="waiting",
                detail="matched_50k_milestone_report_missing",
                control=control,
            ),
        )
        return "waiting", None

    first = build_claim_guard_summary(
        args.milestone_report,
        control=control,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
    )
    second = build_claim_guard_summary(
        args.milestone_report,
        control=control,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
    )
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise ValueError("claim guard summary replay is not byte-equivalent")
    _write_or_verify(args.summary_output, first, label="claim guard summary")
    if read_object(args.summary_output) != first:
        raise ValueError("published claim guard summary does not replay")
    summary_identity = file_identity(args.summary_output)
    write_json_report(
        args.status_output,
        _status_payload(
            args,
            status="completed",
            detail="matched_50k_milestone_claim_guard_published",
            control=control,
            summary={
                **summary_identity,
                "canonical_sha256": canonical_sha256(first),
                "byte_equivalent_replay": True,
            },
        ),
    )
    return "completed", first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact full-data matched 50K milestone and publish a "
            "source-replayed, permanently non-authorizing claim guard."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--milestone-report", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-control-source-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=1_209_600.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _open_lifetime_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise RuntimeError("another claim guard waiter holds the lifetime lock") from error
    return handle


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("poll-seconds and timeout-seconds must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("claim guard requires CUDA_VISIBLE_DEVICES=-1")
    for path in (
        args.summary_output,
        args.status_output,
        args.deployment_receipt_output,
        args.lock,
        args.pid_file,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    lock_handle = _open_lifetime_lock(args.lock)
    try:
        control = verify_control_identity(args)
        write_deployment_receipt(args, control=control)
        write_json_report(
            args.pid_file,
            {
                "schema_version": 1,
                "role": WAITER_ROLE,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": utc_now(),
                "control": control,
            },
        )
        started = time.monotonic()
        while True:
            try:
                status, _ = run_once(args, control=control)
            except Exception as error:
                write_json_report(
                    args.status_output,
                    _status_payload(
                        args,
                        status="failed",
                        detail="claim_guard_validation_failed",
                        control=control,
                        error=f"{type(error).__name__}: {error}",
                    ),
                )
                raise
            if status == "completed" or args.once:
                return 0
            if time.monotonic() - started >= args.timeout_seconds:
                write_json_report(
                    args.status_output,
                    _status_payload(
                        args,
                        status="failed",
                        detail="claim_guard_wait_timeout",
                        control=control,
                    ),
                )
                return 2
            time.sleep(args.poll_seconds)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
