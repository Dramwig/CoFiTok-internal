from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import write_json_report

try:
    import run_generation_capacity_terminal_uncertainty_waiter as capacity_waiter
    import run_generation_matched_uncertainty_waiter as base_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import run_generation_capacity_terminal_uncertainty_waiter as capacity_waiter
    from scripts import run_generation_matched_uncertainty_waiter as base_waiter


SCHEMA_VERSION = 1
ROLE = "generation_capacity_terminal_uncertainty_waiter_deployment"
SOURCE_KINDS = {"capacity_completion_100k", "capacity_full_300k"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_identity(identity: Mapping[str, Any], *, label: str) -> None:
    if (
        not isinstance(identity.get("path"), str)
        or int(identity.get("bytes", -1)) < 0
        or not _is_hex(identity.get("sha256"), 64)
    ):
        raise ValueError(f"{label} identity is invalid")


def _require_git(identity: Mapping[str, Any], *, label: str) -> None:
    if (
        not _is_hex(identity.get("revision"), 40)
        or not _is_hex(identity.get("tree"), 40)
        or not isinstance(identity.get("branch"), str)
        or identity.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity is invalid")


def _git_contract(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key)
        for key in ("revision", "tree", "branch", "tracked_dirty")
    }


def _bundle_prerequisites(path: Path) -> list[str]:
    prerequisites: list[str] = []
    with path.open("rb") as handle:
        header = handle.readline()
        if not header.startswith(b"# v") or b"git bundle" not in header:
            raise ValueError("capacity uncertainty bundle header is invalid")
        for raw_line in handle:
            line = raw_line.rstrip(b"\r\n")
            if not line:
                break
            if not line.startswith(b"-"):
                continue
            token = line[1:].split(b" ", 1)[0].decode("ascii").lower()
            if not _is_hex(token, 40):
                raise ValueError("capacity uncertainty bundle prerequisite is invalid")
            prerequisites.append(token)
    return sorted(set(prerequisites))


def _bundle_heads(path: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "bundle", "list-heads", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    heads: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _is_hex(fields[0], 40):
            raise ValueError("capacity uncertainty bundle head is invalid")
        heads.append({"revision": fields[0].lower(), "ref": fields[1]})
    if not heads:
        raise ValueError("capacity uncertainty bundle has no advertised head")
    return heads


def _stable_json_snapshot(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload, {
        "path": path.resolve().as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_receipt(
    *,
    source_kind: str,
    output_root: str,
    source_anchor_present: bool,
    control_checkout: Mapping[str, Any],
    evaluator_checkout: Mapping[str, Any],
    receipt_builder_checkout: Mapping[str, Any],
    waiter_bundle: Mapping[str, Any],
    receipt_builder_bundle: Mapping[str, Any],
    waiter_status: Mapping[str, Any],
    waiter_status_identity: Mapping[str, Any],
    pid_payload: Mapping[str, Any],
    pid_identity: Mapping[str, Any],
    log_identity: Mapping[str, Any],
    process: Mapping[str, Any],
    gpu_compute_rows: Sequence[Mapping[str, Any]],
    hostname: str,
    created_at: str,
) -> dict[str, Any]:
    if source_kind not in SOURCE_KINDS:
        raise ValueError("capacity uncertainty source kind is invalid")
    _require_git(control_checkout, label="capacity uncertainty control checkout")
    _require_git(evaluator_checkout, label="capacity uncertainty evaluator checkout")
    _require_git(
        receipt_builder_checkout,
        label="capacity uncertainty receipt-builder checkout",
    )
    for label, identity in (
        ("waiter status", waiter_status_identity),
        ("waiter PID", pid_identity),
        ("waiter log", log_identity),
    ):
        _require_identity(identity, label=label)

    expected = waiter_status.get("expected")
    claim_boundary = waiter_status.get("claim_boundary")
    if not isinstance(expected, Mapping):
        raise ValueError("capacity uncertainty waiter expected contract is missing")
    if claim_boundary != capacity_waiter.CLAIM_BOUNDARY:
        raise ValueError("capacity uncertainty waiter claim boundary differs")
    if (
        waiter_status.get("schema_version") != capacity_waiter.WAITER_SCHEMA_VERSION
        or waiter_status.get("role") != capacity_waiter.WAITER_ROLE
        or waiter_status.get("status") != "waiting"
        or waiter_status.get("phase") != "source"
        or waiter_status.get("detail")
        != "waiting_for_capacity_terminal_source_anchor"
        or waiter_status.get("source_anchor") is not None
        or waiter_status.get("execution_manifest") is not None
        or waiter_status.get("audit") is not None
        or waiter_status.get("statistical_claim_qualification") is not None
        or expected.get("source_kind") != source_kind
        or expected.get("output_root") != output_root
        or expected.get("control_git") != _git_contract(control_checkout)
        or expected.get("evaluator_git") != _git_contract(evaluator_checkout)
    ):
        raise ValueError("capacity uncertainty waiter initial status differs")
    if source_anchor_present:
        raise ValueError("capacity uncertainty source anchor already existed at deployment")

    pid = int(waiter_status.get("pid", -1))
    if (
        pid < 1
        or pid_payload.get("schema_version") != 1
        or pid_payload.get("role") != capacity_waiter.WAITER_ROLE
        or int(pid_payload.get("pid", -1)) != pid
        or pid_payload.get("source_kind") != source_kind
        or pid_payload.get("expected_control_revision")
        != control_checkout.get("revision")
        or int(process.get("pid", -1)) != pid
    ):
        raise ValueError("capacity uncertainty waiter PID binding differs")
    command = str(process.get("command", ""))
    if (
        "scripts/run_generation_capacity_terminal_uncertainty_waiter.py" not in command
        or f"--source-kind {source_kind}" not in command
        or f"--output-root {output_root}" not in command
        or str(expected.get("source_anchor_path", "")) not in command
        or str(control_checkout.get("revision", "")) not in command
        or process.get("cwd") != control_checkout.get("path")
    ):
        raise ValueError("capacity uncertainty waiter process command differs")
    if any(int(row.get("pid", -1)) == pid for row in gpu_compute_rows):
        raise ValueError("capacity uncertainty waiter unexpectedly allocated GPU compute")

    waiter_bundle_identity = waiter_bundle.get("identity")
    waiter_bundle_heads = waiter_bundle.get("heads")
    waiter_prerequisites = waiter_bundle.get("prerequisites")
    if not isinstance(waiter_bundle_identity, Mapping):
        raise ValueError("capacity uncertainty bundle identity is missing")
    _require_identity(
        waiter_bundle_identity,
        label="capacity uncertainty waiter bundle",
    )
    if (
        waiter_bundle_heads
        != [
            {
                "revision": control_checkout["revision"],
                "ref": f"refs/heads/{control_checkout['branch']}",
            }
        ]
        or not isinstance(waiter_prerequisites, list)
        or len(waiter_prerequisites) != 1
        or not _is_hex(waiter_prerequisites[0], 40)
    ):
        raise ValueError("capacity uncertainty bundle contract differs")

    builder_bundle_identity = receipt_builder_bundle.get("identity")
    builder_bundle_heads = receipt_builder_bundle.get("heads")
    builder_prerequisites = receipt_builder_bundle.get("prerequisites")
    if not isinstance(builder_bundle_identity, Mapping):
        raise ValueError("capacity uncertainty receipt-builder bundle is missing")
    _require_identity(
        builder_bundle_identity,
        label="capacity uncertainty receipt-builder bundle",
    )
    if (
        not isinstance(builder_bundle_heads, list)
        or len(builder_bundle_heads) != 1
        or builder_bundle_heads[0].get("revision")
        != receipt_builder_checkout.get("revision")
        or not isinstance(builder_bundle_heads[0].get("ref"), str)
        or not builder_bundle_heads[0]["ref"].startswith("refs/heads/")
        or builder_prerequisites != [control_checkout.get("revision")]
    ):
        raise ValueError("capacity uncertainty receipt-builder bundle differs")

    source_git = expected.get("source_git")
    real_cache = expected.get("real_feature_cache_source")
    if not isinstance(source_git, Mapping) or not isinstance(real_cache, Mapping):
        raise ValueError("capacity uncertainty source contract is incomplete")
    for role in ("execution", "training", "result", "evaluation"):
        identity = source_git.get(role)
        if not isinstance(identity, Mapping):
            raise ValueError("capacity uncertainty source Git contract is incomplete")
        populated = any(identity.get(field) for field in ("revision", "tree", "branch"))
        if populated:
            _require_git(
                {**identity, "tracked_dirty": False},
                label=f"capacity uncertainty source {role}",
            )
    if (
        not isinstance(real_cache.get("path"), str)
        or int(real_cache.get("bytes", 0)) < 1
        or not _is_hex(real_cache.get("sha256"), 64)
    ):
        raise ValueError("capacity uncertainty real-cache contract is invalid")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "active",
        "source_kind": source_kind,
        "authorization_boundary": dict(claim_boundary),
        "bundles": {
            "waiter_control": dict(waiter_bundle),
            "receipt_builder": dict(receipt_builder_bundle),
        },
        "control_checkout": dict(control_checkout),
        "evaluator_checkout": dict(evaluator_checkout),
        "receipt_builder_checkout": dict(receipt_builder_checkout),
        "source_contract": {
            "anchor_path": expected["source_anchor_path"],
            "anchor_present_at_deployment": source_anchor_present,
            "git": dict(source_git),
        },
        "waiter": {
            "pid": pid,
            "process": dict(process),
            "output_root": output_root,
            "status_snapshot": {
                "identity": dict(waiter_status_identity),
                "payload": dict(waiter_status),
            },
            "pid_file": {
                "identity": dict(pid_identity),
                "payload": dict(pid_payload),
            },
            "log_identity": dict(log_identity),
        },
        "runtime": {
            "hostname": hostname,
            "gpu_compute_rows": [dict(row) for row in gpu_compute_rows],
            "waiter_allocated_gpu": False,
        },
        "validation": {
            "bundle_head_matches_control_revision": True,
            "checkouts_tracked_clean": True,
            "exact_source_contract_bound": True,
            "initial_waiting_state_verified": True,
            "source_anchor_absent_at_deployment": True,
            "waiter_process_alive": True,
            "waiter_pid_not_in_gpu_compute": True,
        },
        "created_at": created_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write an immutable deployment receipt for a source-bound, "
            "non-authorizing capacity terminal uncertainty waiter."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waiter-bundle", type=Path, required=True)
    parser.add_argument("--expected-waiter-bundle-prerequisite", required=True)
    parser.add_argument("--receipt-builder-bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-receipt-builder-bundle-prerequisite",
        required=True,
    )
    parser.add_argument("--control-project", type=Path, required=True)
    parser.add_argument("--evaluator-project", type=Path, required=True)
    parser.add_argument("--waiter-status", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--waiter-log", type=Path, required=True)
    parser.add_argument("--expected-pid", type=int, required=True)
    args = parser.parse_args()

    output = reject_symlink_chain(args.output, name="capacity deployment receipt")
    if output.exists():
        raise FileExistsError(f"capacity deployment receipt already exists: {output}")
    waiter_bundle_path = reject_symlink_chain(
        args.waiter_bundle,
        name="capacity uncertainty waiter bundle",
    )
    builder_bundle_path = reject_symlink_chain(
        args.receipt_builder_bundle,
        name="capacity uncertainty receipt-builder bundle",
    )
    status_path = reject_symlink_chain(
        args.waiter_status,
        name="capacity uncertainty waiter status",
    )
    pid_path = reject_symlink_chain(args.pid_file, name="capacity uncertainty waiter PID")
    log_path = reject_symlink_chain(args.waiter_log, name="capacity uncertainty waiter log")
    status, status_identity = _stable_json_snapshot(
        status_path,
        name="capacity uncertainty waiter status",
    )
    pid_payload, pid_identity = _stable_json_snapshot(
        pid_path,
        name="capacity uncertainty waiter PID",
    )
    if int(status.get("pid", -1)) != args.expected_pid:
        raise ValueError("capacity uncertainty expected PID differs")
    expected = status.get("expected", {})
    if not isinstance(expected, Mapping):
        raise ValueError("capacity uncertainty expected contract is missing")
    source_kind = str(expected.get("source_kind", ""))
    output_root = str(expected.get("output_root", ""))
    source_anchor = Path(str(expected.get("source_anchor_path", "")))

    control_project = reject_symlink_chain(
        args.control_project,
        name="capacity uncertainty control checkout",
    ).resolve()
    evaluator_project = reject_symlink_chain(
        args.evaluator_project,
        name="capacity uncertainty evaluator checkout",
    ).resolve()
    control_checkout = {
        "path": control_project.as_posix(),
        **base_waiter.verify_checkout(
            control_project,
            expected_revision=str(expected.get("control_git", {}).get("revision", "")),
            expected_tree=str(expected.get("control_git", {}).get("tree", "")),
            expected_branch=str(expected.get("control_git", {}).get("branch", "")),
            label="capacity uncertainty control",
        ),
    }
    evaluator_checkout = {
        "path": evaluator_project.as_posix(),
        **base_waiter.verify_checkout(
            evaluator_project,
            expected_revision=str(
                expected.get("evaluator_git", {}).get("revision", "")
            ),
            expected_tree=str(expected.get("evaluator_git", {}).get("tree", "")),
            expected_branch=str(expected.get("evaluator_git", {}).get("branch", "")),
            label="capacity uncertainty evaluator",
        ),
    }
    receipt_builder_project = Path(__file__).resolve().parents[1]
    receipt_builder_checkout = {
        "path": receipt_builder_project.as_posix(),
        **base_waiter._git_identity(receipt_builder_project),
    }
    _require_git(
        receipt_builder_checkout,
        label="capacity uncertainty receipt-builder checkout",
    )

    processes = [
        row
        for row in base_waiter._process_rows()
        if int(row.get("pid", -1)) == args.expected_pid
    ]
    if len(processes) != 1:
        raise ValueError("capacity uncertainty waiter process is not uniquely alive")
    waiter_prerequisites = _bundle_prerequisites(waiter_bundle_path)
    if waiter_prerequisites != [args.expected_waiter_bundle_prerequisite.lower()]:
        raise ValueError("capacity uncertainty bundle prerequisite differs")
    builder_prerequisites = _bundle_prerequisites(builder_bundle_path)
    if builder_prerequisites != [
        args.expected_receipt_builder_bundle_prerequisite.lower()
    ]:
        raise ValueError("capacity uncertainty receipt-builder prerequisite differs")
    receipt = build_receipt(
        source_kind=source_kind,
        output_root=output_root,
        source_anchor_present=source_anchor.is_file(),
        control_checkout=control_checkout,
        evaluator_checkout=evaluator_checkout,
        receipt_builder_checkout=receipt_builder_checkout,
        waiter_bundle={
            "identity": file_identity(waiter_bundle_path),
            "heads": _bundle_heads(waiter_bundle_path),
            "prerequisites": waiter_prerequisites,
        },
        receipt_builder_bundle={
            "identity": file_identity(builder_bundle_path),
            "heads": _bundle_heads(builder_bundle_path),
            "prerequisites": builder_prerequisites,
        },
        waiter_status=status,
        waiter_status_identity=status_identity,
        pid_payload=pid_payload,
        pid_identity=pid_identity,
        log_identity=file_identity(log_path),
        process=processes[0],
        gpu_compute_rows=base_waiter._gpu_compute_rows(),
        hostname=socket.gethostname(),
        created_at=_utc_now(),
    )
    write_json_report(output, receipt)
    read_json_object(output, name="capacity uncertainty deployment receipt")
    print(output)


if __name__ == "__main__":
    main()
