from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import read_json_object, reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCHEMA_VERSION = 1
REPLAY_ROLE = "generation_checkpoint_physical_integrity_audit_replay"
AUDIT_ROLE = "generation_checkpoint_physical_integrity_milestone_audit"
WAITER_ROLE = "generation_checkpoint_physical_integrity_milestone_waiter"
READ_ONLY_SCOPE = {
    "read_only_checkpoint_verification": True,
    "gpu_required": False,
    "training_process_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a source-bound checkpoint physical-integrity audit after "
            "train_metrics.jsonl and latest.json may have advanced."
        )
    )
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--expected-audit-report-sha256", required=True)
    parser.add_argument("--waiter-status", required=True)
    parser.add_argument("--expected-waiter-status-sha256", required=True)
    parser.add_argument("--auditor-source", required=True)
    parser.add_argument("--expected-auditor-source-sha256", required=True)
    parser.add_argument("--auditor-checkout", required=True)
    parser.add_argument("--expected-auditor-revision", required=True)
    parser.add_argument("--expected-auditor-tree", required=True)
    parser.add_argument("--expected-auditor-branch", required=True)
    parser.add_argument("--training-checkout", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--expected-verifier-revision", required=True)
    parser.add_argument("--expected-verifier-tree", required=True)
    parser.add_argument("--expected-verifier-branch", required=True)
    parser.add_argument("--expected-verifier-source-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} SHA256 must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} SHA256 is not hexadecimal") from error


def _git(*arguments: str, root: Path) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_identity(root: Path) -> dict[str, Any]:
    return {
        "path": root.resolve().as_posix(),
        "revision": _git("rev-parse", "HEAD", root=root),
        "tree": _git("rev-parse", "HEAD^{tree}", root=root),
        "branch": _git("branch", "--show-current", root=root),
        "tracked_dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no", root=root)
        ),
    }


def require_git_identity(
    root: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    actual = git_identity(root)
    expected = {
        "path": root.resolve().as_posix(),
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if actual != expected:
        raise ValueError(f"{label} Git identity differs: {actual!r}")
    return actual


def stable_file_identity(path: str | Path, *, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    before = source.stat()
    digest = file_sha256(source)
    after = source.stat()
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise ValueError(f"{name} changed while it was being hashed")
    return {
        "path": source.resolve().as_posix(),
        "bytes": after.st_size,
        "sha256": digest,
    }


def stable_json(
    path: str | Path,
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=name)
    before = stable_file_identity(source, name=name)
    payload = read_json_object(source, name=name)
    after = stable_file_identity(source, name=name)
    if before != after:
        raise ValueError(f"{name} changed while it was being parsed")
    return payload, after


def canonical_json_identity(
    payload: dict[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return {
        "path": path,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_checkpoint_binding(
    report: dict[str, Any],
    *,
    run_dir: Path,
    expected_step: int,
    expected_sha256: str,
    expected_revision: str,
    expected_branch: str,
    expected_dataset_sha256: str,
    expected_runtime_sha256: str,
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("audit checkpoint evidence is malformed")
    payload_claim = checkpoint.get("payload")
    integrity_claim = checkpoint.get("integrity_manifest")
    integrity = checkpoint.get("integrity")
    if not all(isinstance(value, dict) for value in (payload_claim, integrity_claim, integrity)):
        raise ValueError("audit checkpoint identities are malformed")
    payload = stable_file_identity(
        str(payload_claim.get("path", "")),
        name="audited checkpoint payload",
    )
    current_integrity, integrity_identity = stable_json(
        str(integrity_claim.get("path", "")),
        name="audited checkpoint integrity manifest",
    )
    if payload != payload_claim or integrity_identity != integrity_claim:
        raise ValueError("audited checkpoint physical identity differs")
    if current_integrity != integrity:
        raise ValueError("audited checkpoint integrity content differs")
    expected_checkpoint_name = f"checkpoint_step_{expected_step:08d}.pt"
    expected_payload = run_dir / expected_checkpoint_name
    expected_integrity = run_dir / f"{expected_checkpoint_name}.integrity.json"
    if (
        Path(payload["path"]) != expected_payload
        or Path(integrity_identity["path"]) != expected_integrity
    ):
        raise ValueError("audited checkpoint paths are not canonical")
    expected = {
        "schema_version": 1,
        "checkpoint": Path(payload["path"]).name,
        "checkpoint_bytes": payload["bytes"],
        "checkpoint_sha256": expected_sha256,
        "checkpoint_format_version": 1,
        "step": expected_step,
        "git_revision": expected_revision,
        "git_branch": expected_branch,
        "git_dirty": False,
        "dataset_identity_sha256": expected_dataset_sha256,
        "runtime_environment_sha256": expected_runtime_sha256,
    }
    for key, value in expected.items():
        if integrity.get(key) != value:
            raise ValueError(f"audited checkpoint integrity {key} differs")
    if (
        payload["sha256"] != expected_sha256
        or int(checkpoint.get("step", -1)) != expected_step
        or checkpoint.get("physical_sha256_verified") is not True
    ):
        raise ValueError("audited checkpoint exact target binding differs")
    return {
        "step": expected_step,
        "payload": payload,
        "integrity_manifest": integrity_identity,
        "integrity": integrity,
        "physical_sha256_reverified": True,
    }


def validate_historical_latest(
    report: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    claimed = report.get("latest_pointer")
    if not isinstance(claimed, dict):
        raise ValueError("audit latest-pointer evidence is malformed")
    historical_identity = claimed.get("identity")
    historical = claimed.get("content")
    if not isinstance(historical_identity, dict) or not isinstance(historical, dict):
        raise ValueError("audit historical latest snapshot is malformed")
    expected_latest_path = (run_dir / "latest.json").as_posix()
    if historical_identity.get("path") != expected_latest_path:
        raise ValueError("audit historical latest path is not canonical")
    reconstructed = canonical_json_identity(
        historical,
        path=str(historical_identity.get("path", "")),
    )
    if reconstructed != historical_identity:
        raise ValueError("audit historical latest byte identity cannot be reconstructed")
    integrity = checkpoint["integrity"]
    expected = {
        **integrity,
        "integrity_manifest": Path(checkpoint["integrity_manifest"]["path"]).name,
    }
    if historical != expected or claimed.get("exact_target_binding") is not True:
        raise ValueError("audit historical latest checkpoint binding differs")

    current, current_identity = stable_json(
        historical_identity["path"],
        name="current latest pointer",
    )
    historical_step = int(historical["step"])
    current_step = int(current.get("step", -1))
    if current_step < historical_step:
        raise ValueError("current latest pointer regressed behind the audit")
    if current_step == historical_step:
        if current != historical or current_identity != historical_identity:
            raise ValueError("current latest pointer changed at the audited step")
        relation = "exact_historical_pointer_still_current"
    else:
        if (
            current.get("schema_version") != 1
            or current.get("checkpoint_format_version") != 1
        ):
            raise ValueError("advanced latest pointer schema differs")
        for key in (
            "git_revision",
            "git_branch",
            "git_dirty",
            "dataset_identity_sha256",
            "runtime_environment_sha256",
        ):
            if current.get(key) != historical.get(key):
                raise ValueError(f"advanced latest pointer {key} differs")
        current_checkpoint = run_dir / str(current.get("checkpoint", ""))
        current_integrity_path = run_dir / str(current.get("integrity_manifest", ""))
        if (
            current_checkpoint.name != current.get("checkpoint")
            or current_integrity_path.name != current.get("integrity_manifest")
            or not current_checkpoint.is_file()
            or not current_integrity_path.is_file()
            or current_checkpoint.stat().st_size != int(current.get("checkpoint_bytes", -1))
        ):
            raise ValueError("advanced latest pointer files are inconsistent")
        current_integrity, current_integrity_identity = stable_json(
            current_integrity_path,
            name="advanced latest integrity manifest",
        )
        for key in (
            "checkpoint",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "step",
            "git_revision",
            "git_branch",
            "git_dirty",
            "dataset_identity_sha256",
            "runtime_environment_sha256",
        ):
            if current_integrity.get(key) != current.get(key):
                raise ValueError(f"advanced latest integrity {key} differs")
        relation = "advanced_with_consistent_metadata"
    return {
        "historical_identity": historical_identity,
        "historical_content": historical,
        "historical_byte_identity_reconstructed": True,
        "current_identity": current_identity,
        "current_content": current,
        "current_integrity_manifest": (
            current_integrity_identity if current_step > historical_step else None
        ),
        "relation": relation,
        "current_payload_hash_recomputed": current_step == historical_step,
    }


def _parse_metrics_rows(
    payload: bytes,
    *,
    effective_batch: int,
) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("metrics byte prefix is not atomically terminated")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(keepends=True), start=1):
        if not raw.strip():
            continue
        if not raw.endswith(b"\n"):
            raise ValueError(f"metrics line {line_number} is not atomically terminated")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"metrics line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_number} is not an object")
        step = int(row.get("step", -1))
        samples_seen = int(row.get("samples_seen", -1))
        if step < 1 or samples_seen != step * effective_batch:
            raise ValueError("metrics samples_seen binding differs")
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("metrics contain a non-finite floating value")
        rows.append(row)
    if not rows:
        raise ValueError("metrics are empty during replay")
    steps = [int(row["step"]) for row in rows]
    if not all(right > left for left, right in zip(steps, steps[1:])):
        raise ValueError("metrics are not strictly increasing")
    return rows


def _metrics_snapshot(
    path: Path,
    *,
    historical_identity: dict[str, Any],
    effective_batch: int,
) -> dict[str, Any]:
    try:
        historical_bytes = int(historical_identity.get("bytes", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("audit metrics byte identity is malformed") from error
    historical_sha256 = str(historical_identity.get("sha256", ""))
    if historical_bytes < 1:
        raise ValueError("audit metrics byte identity is malformed")
    _require_sha256(historical_sha256, name="audit metrics")

    before = path.stat()
    if before.st_size < historical_bytes:
        raise ValueError("current metrics are shorter than the audit-time prefix")
    with path.open("rb") as handle:
        observed = handle.read(before.st_size)
    after = path.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ValueError("metrics file changed identity during replay")
    if after.st_size < before.st_size or len(observed) != before.st_size:
        raise ValueError("metrics file shrank during replay")

    historical_payload = observed[:historical_bytes]
    if (
        len(historical_payload) != historical_bytes
        or hashlib.sha256(historical_payload).hexdigest() != historical_sha256
    ):
        raise ValueError("audit-time metrics byte prefix differs")
    last_newline = observed.rfind(b"\n")
    if last_newline < historical_bytes - 1:
        raise ValueError("current metrics no longer contain a complete audit prefix")
    observed_prefix = observed[: last_newline + 1]
    historical_rows = _parse_metrics_rows(
        historical_payload,
        effective_batch=effective_batch,
    )
    observed_rows = _parse_metrics_rows(
        observed_prefix,
        effective_batch=effective_batch,
    )
    historical_steps = [int(row["step"]) for row in historical_rows]
    observed_steps = [int(row["step"]) for row in observed_rows]
    return {
        "historical": {
            "path": path.resolve().as_posix(),
            "bytes": historical_bytes,
            "sha256": historical_sha256,
            "row_count": len(historical_rows),
            "first_step": historical_steps[0],
            "last_step": historical_steps[-1],
            "rows": historical_rows,
        },
        "current_observed_prefix": {
            "path": path.resolve().as_posix(),
            "bytes": len(observed_prefix),
            "sha256": hashlib.sha256(observed_prefix).hexdigest(),
            "row_count": len(observed_rows),
            "first_step": observed_steps[0],
            "last_step": observed_steps[-1],
            "complete_line_prefix": True,
            "sampled_file_bytes": before.st_size,
            "file_bytes_after_read": after.st_size,
        },
    }


def validate_metrics_replay(
    report: dict[str, Any],
    *,
    run_dir: Path,
    effective_batch: int,
    expected_checkpoint_step: int,
) -> dict[str, Any]:
    claimed = report.get("metrics")
    if not isinstance(claimed, dict) or not isinstance(claimed.get("identity"), dict):
        raise ValueError("audit metrics evidence is malformed")
    historical_last_step = int(claimed.get("last_step", -1))
    if historical_last_step < expected_checkpoint_step:
        raise ValueError("audit metrics stop before the checkpoint")
    path = reject_symlink_chain(
        str(claimed["identity"].get("path", "")),
        name="checkpoint audit metrics",
    )
    if path != run_dir / "train_metrics.jsonl":
        raise ValueError("audit metrics path is not canonical")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint audit metrics are missing: {path}")
    snapshot = _metrics_snapshot(
        path,
        historical_identity=claimed["identity"],
        effective_batch=effective_batch,
    )
    historical = snapshot["historical"]
    historical_identity = {
        key: historical[key] for key in ("path", "bytes", "sha256")
    }
    if historical_identity != claimed["identity"]:
        raise ValueError("audit-time metrics byte prefix differs")
    if (
        historical["row_count"] != int(claimed.get("row_count", -1))
        or historical["first_step"] != int(claimed.get("first_step", -1))
        or historical["last_step"] != historical_last_step
        or claimed.get("strictly_increasing") is not True
        or claimed.get("samples_seen_binding_verified") is not True
        or claimed.get("samples_seen_binding")
        != f"samples_seen == step * {effective_batch}"
    ):
        raise ValueError("audit metrics summary differs")
    target = next(
        (
            row
            for row in historical["rows"]
            if int(row["step"]) == expected_checkpoint_step
        ),
        None,
    )
    if target is None or target != claimed.get("target_row"):
        raise ValueError("audit checkpoint metrics row differs")
    del historical["rows"]
    return {
        "audit_time_prefix": historical,
        "current_observed_prefix": snapshot["current_observed_prefix"],
        "audit_time_prefix_byte_exact": True,
        "current_metrics_remain_strict_and_exposure_bound": True,
        "target_row": target,
    }


def verify_replay(
    *,
    audit_report_path: Path,
    expected_audit_report_sha256: str,
    waiter_status_path: Path,
    expected_waiter_status_sha256: str,
    auditor_source_path: Path,
    expected_auditor_source_sha256: str,
    auditor_checkout: Path,
    expected_auditor_revision: str,
    expected_auditor_tree: str,
    expected_auditor_branch: str,
    training_checkout: Path,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_checkpoint_step: int,
    expected_checkpoint_sha256: str,
    expected_dataset_sha256: str,
    expected_runtime_sha256: str,
    effective_batch: int,
    verifier_git: dict[str, Any],
) -> dict[str, Any]:
    report, report_identity = stable_json(
        audit_report_path,
        name="checkpoint physical-integrity audit report",
    )
    status, status_identity = stable_json(
        waiter_status_path,
        name="checkpoint physical-integrity waiter status",
    )
    auditor_source = stable_file_identity(
        auditor_source_path,
        name="checkpoint integrity auditor source",
    )
    auditor_git = require_git_identity(
        auditor_checkout,
        revision=expected_auditor_revision,
        tree=expected_auditor_tree,
        branch=expected_auditor_branch,
        label="checkpoint integrity auditor checkout",
    )
    expected_auditor_source = (
        auditor_checkout.resolve()
        / "scripts"
        / "wait_generation_checkpoint_integrity_audit.py"
    ).as_posix()
    if auditor_source["path"] != expected_auditor_source:
        raise ValueError("checkpoint auditor source path is not canonical")
    if report_identity["sha256"] != expected_audit_report_sha256:
        raise ValueError("checkpoint audit report SHA256 differs")
    if status_identity["sha256"] != expected_waiter_status_sha256:
        raise ValueError("checkpoint waiter status SHA256 differs")
    if auditor_source["sha256"] != expected_auditor_source_sha256:
        raise ValueError("checkpoint auditor source SHA256 differs")
    if (
        report.get("schema_version") != 1
        or report.get("role") != AUDIT_ROLE
        or report.get("status") != "pass"
        or report.get("scope") != READ_ONLY_SCOPE
    ):
        raise ValueError("checkpoint audit report contract differs")
    if (
        status.get("schema_version") != 1
        or status.get("role") != WAITER_ROLE
        or status.get("status") != "pass"
        or status.get("detail") != "checkpoint_physical_integrity_verified"
        or status.get("scope") != READ_ONLY_SCOPE
        or status.get("audit_output_identity") != report_identity
        or status.get("waiter_source") != auditor_source
        or int(status.get("checkpoint_step", -1)) != expected_checkpoint_step
        or status.get("audit_output") != report_identity["path"]
        or status.get("run_dir") != report.get("run_dir")
    ):
        raise ValueError("checkpoint waiter status contract differs")

    run_dir = reject_symlink_chain(
        str(report.get("run_dir", "")),
        name="checkpoint audit run directory",
    )
    if not run_dir.is_dir() or report.get("run_dir") != str(run_dir.resolve()):
        raise ValueError("checkpoint audit run directory is not canonical")

    training_git = require_git_identity(
        training_checkout,
        revision=expected_training_revision,
        tree=expected_training_tree,
        branch=expected_training_branch,
        label="training checkout",
    )
    if report.get("training_checkout") != training_git:
        raise ValueError("checkpoint audit training checkout binding differs")
    checkpoint = validate_checkpoint_binding(
        report,
        run_dir=run_dir,
        expected_step=expected_checkpoint_step,
        expected_sha256=expected_checkpoint_sha256,
        expected_revision=expected_training_revision,
        expected_branch=expected_training_branch,
        expected_dataset_sha256=expected_dataset_sha256,
        expected_runtime_sha256=expected_runtime_sha256,
    )
    historical_latest = validate_historical_latest(
        report,
        checkpoint=checkpoint,
        run_dir=run_dir,
    )
    metrics = validate_metrics_replay(
        report,
        run_dir=run_dir,
        effective_batch=effective_batch,
        expected_checkpoint_step=expected_checkpoint_step,
    )
    final_report = stable_file_identity(
        audit_report_path,
        name="checkpoint physical-integrity audit report",
    )
    final_status = stable_file_identity(
        waiter_status_path,
        name="checkpoint physical-integrity waiter status",
    )
    final_auditor_source = stable_file_identity(
        auditor_source_path,
        name="checkpoint integrity auditor source",
    )
    final_checkpoint_payload = stable_file_identity(
        checkpoint["payload"]["path"],
        name="audited checkpoint payload",
    )
    final_checkpoint_integrity, final_checkpoint_integrity_identity = stable_json(
        checkpoint["integrity_manifest"]["path"],
        name="audited checkpoint integrity manifest",
    )
    final_training_git = require_git_identity(
        training_checkout,
        revision=expected_training_revision,
        tree=expected_training_tree,
        branch=expected_training_branch,
        label="training checkout",
    )
    final_auditor_git = require_git_identity(
        auditor_checkout,
        revision=expected_auditor_revision,
        tree=expected_auditor_tree,
        branch=expected_auditor_branch,
        label="checkpoint integrity auditor checkout",
    )
    if (
        final_report != report_identity
        or final_status != status_identity
        or final_auditor_source != auditor_source
        or final_checkpoint_payload != checkpoint["payload"]
        or final_checkpoint_integrity_identity != checkpoint["integrity_manifest"]
        or final_checkpoint_integrity != checkpoint["integrity"]
        or final_training_git != training_git
        or final_auditor_git != auditor_git
    ):
        raise ValueError("checkpoint audit replay sources changed during verification")
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "role": REPLAY_ROLE,
        "status": "pass",
        "verified_at": utc_now(),
        "verifier_git": verifier_git,
        "verifier_source": stable_file_identity(
            Path(__file__),
            name="checkpoint audit replay verifier source",
        ),
        "audit_report": report_identity,
        "waiter_status": status_identity,
        "auditor_source": auditor_source,
        "auditor_checkout": auditor_git,
        "training_checkout": training_git,
        "checkpoint": checkpoint,
        "latest_pointer_replay": historical_latest,
        "metrics_replay": metrics,
        "checks": {
            "checkpoint_payload_physically_rehashed": True,
            "integrity_manifest_rehashed_and_reparsed": True,
            "historical_latest_byte_identity_reconstructed": True,
            "historical_metrics_prefix_reconstructed_byte_exact": True,
            "training_checkout_identity_reverified": True,
            "auditor_checkout_identity_reverified": True,
            "audit_and_waiter_sources_rehashed": True,
            "bound_sources_rehashed_after_checkpoint_verification": True,
            "checkpoint_payload_and_sidecar_rehashed_twice": True,
        },
        "scope": {
            **READ_ONLY_SCOPE,
            "report_is_checkpoint_promotion_gate": False,
            "report_is_generation_quality_evidence": False,
            "sampling_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def main() -> None:
    args = parse_args()
    for name, value in (
        ("audit report", args.expected_audit_report_sha256),
        ("waiter status", args.expected_waiter_status_sha256),
        ("auditor source", args.expected_auditor_source_sha256),
        ("checkpoint", args.expected_checkpoint_sha256),
        ("dataset", args.expected_dataset_sha256),
        ("runtime", args.expected_runtime_sha256),
        ("verifier source", args.expected_verifier_source_sha256),
    ):
        _require_sha256(value, name=name)
    if args.expected_checkpoint_step < 1 or args.effective_batch < 1:
        raise ValueError("checkpoint step and effective batch must be positive")
    verifier_git = require_git_identity(
        PROJECT_ROOT,
        revision=args.expected_verifier_revision,
        tree=args.expected_verifier_tree,
        branch=args.expected_verifier_branch,
        label="checkpoint audit replay verifier",
    )
    verifier_source = stable_file_identity(
        Path(__file__),
        name="checkpoint audit replay verifier source",
    )
    if verifier_source["sha256"] != args.expected_verifier_source_sha256:
        raise ValueError("checkpoint audit replay verifier source SHA256 differs")
    output = reject_symlink_chain(args.output, name="checkpoint audit replay output")
    if output.exists():
        raise FileExistsError(f"checkpoint audit replay output already exists: {output}")
    replay = verify_replay(
        audit_report_path=Path(args.audit_report),
        expected_audit_report_sha256=args.expected_audit_report_sha256,
        waiter_status_path=Path(args.waiter_status),
        expected_waiter_status_sha256=args.expected_waiter_status_sha256,
        auditor_source_path=Path(args.auditor_source),
        expected_auditor_source_sha256=args.expected_auditor_source_sha256,
        auditor_checkout=Path(args.auditor_checkout),
        expected_auditor_revision=args.expected_auditor_revision,
        expected_auditor_tree=args.expected_auditor_tree,
        expected_auditor_branch=args.expected_auditor_branch,
        training_checkout=Path(args.training_checkout),
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_checkpoint_step=args.expected_checkpoint_step,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_runtime_sha256=args.expected_runtime_sha256,
        effective_batch=args.effective_batch,
        verifier_git=verifier_git,
    )
    if replay["verifier_source"] != verifier_source:
        raise ValueError("checkpoint audit verifier changed during replay")
    write_json_report(output, replay)
    print(output)


if __name__ == "__main__":
    main()
