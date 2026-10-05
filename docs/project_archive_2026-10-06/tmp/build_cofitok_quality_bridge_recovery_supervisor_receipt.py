from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE = Path("/tmp/cofitok-quality-bridge-execution-cf0e5fa")
PROJECT = BASE / "CoFiTok-internal"
FORMAL = Path("/root/autodl-tmp/CoFiTok/CoFiTok-internal")
OUTPUT_ROOT = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1"
)
REPORT_ROOT = OUTPUT_ROOT / "reports"
RECEIPT = REPORT_ROOT / "recovery_supervisor_deployment_receipt.json"
SUPERVISOR = BASE / "recovery_supervisor.py"
SUPERVISOR_SHA256 = "008a9d83e59fa433d2d4542dbb3b3e29ec8e5bee8eadf5190f5b3b1c99b40cbf"
PRELAUNCH_AUDIT = REPORT_ROOT / "prelaunch_contract_audit.json"
PRELAUNCH_AUDIT_SHA256 = "00dd284187a060be945e47d8ade4d1b3e1d51cbbfdd030413cf66232ffd979d4"
REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
BRANCH = "scale/generation-stability-quality-bridge-100k"
FORMAL_REVISION = "1ebcc15210e63a776a2ba448481cbd8bb94a4066"
FORMAL_BRANCH = "scale/generative-system"
FORMAL_PORCELAIN_COUNT = 87
FORMAL_PORCELAIN_SHA256 = (
    "a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path, expected: str) -> dict[str, Any]:
    observed = sha(path)
    if observed != expected:
        raise RuntimeError(f"identity differs for {path}: {observed}")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": observed}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def git(path: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


supervisor_identity = identity(SUPERVISOR, SUPERVISOR_SHA256)
audit_identity = identity(PRELAUNCH_AUDIT, PRELAUNCH_AUDIT_SHA256)
status_path = BASE / "recovery_supervisor_status.json"
duplicate_probe_path = BASE / "recovery_supervisor_duplicate_probe_v2.log"
waiter_status_path = BASE / "idle_waiter_status.json"
status = load(status_path)
waiter = load(waiter_status_path)

if (
    status.get("status") != "observing"
    or status.get("detail") != "existing_quality_bridge_execution_is_active"
    or status.get("attempt") != 0
    or status.get("retryable_exit_codes") != [-15, -9, 9, 12, 15, 87, 88, 89, 137, 143]
    or status.get("scope")
    != {
        "quality_bridge_recovery_allowed": True,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "release_authorization_allowed": False,
        "report_is_promotion_gate": False,
        "unrelated_gpu_processes_must_not_be_signaled": True,
    }
    or status.get("identities", {}).get("recovery_supervisor", {}).get("sha256")
    != SUPERVISOR_SHA256
):
    raise RuntimeError("recovery supervisor status does not satisfy deployment contract")
if status.get("supervisor_pid") != int((BASE / "recovery_supervisor.pid").read_text().strip()):
    raise RuntimeError("recovery supervisor PID file differs")
pid = int(status["supervisor_pid"])
proc = Path(f"/proc/{pid}")
if not proc.is_dir():
    raise RuntimeError("recovery supervisor process is absent")
command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
cwd = os.readlink(proc / "cwd")
if command != f"/root/autodl-tmp/conda/envs/pf-vlm/bin/python {SUPERVISOR}":
    raise RuntimeError(f"recovery supervisor command differs: {command}")
if cwd != PROJECT.as_posix():
    raise RuntimeError(f"recovery supervisor cwd differs: {cwd}")
if duplicate_probe_path.read_text(encoding="utf-8").strip() != (
    "quality bridge recovery supervisor already owns its lock"
):
    raise RuntimeError("duplicate supervisor probe did not prove lock exclusion")
if (
    waiter.get("status") != "waiting"
    or waiter.get("detail") != "waiting_for_gpu_idle"
    or waiter.get("revision") != REVISION
    or waiter.get("tree") != TREE
    or waiter.get("full_300k_launch_allowed") is not False
):
    raise RuntimeError("idle waiter status differs")

if str(git(PROJECT, "rev-parse", "HEAD")).strip() != REVISION:
    raise RuntimeError("execution revision changed")
if str(git(PROJECT, "rev-parse", "HEAD^{tree}")).strip() != TREE:
    raise RuntimeError("execution tree changed")
if str(git(PROJECT, "branch", "--show-current")).strip() != BRANCH:
    raise RuntimeError("execution branch changed")
if bytes(git(PROJECT, "status", "--porcelain=v1", binary=True)):
    raise RuntimeError("execution checkout became dirty")
formal_porcelain = bytes(git(FORMAL, "status", "--porcelain=v1", binary=True))
if str(git(FORMAL, "rev-parse", "HEAD")).strip() != FORMAL_REVISION:
    raise RuntimeError("formal revision changed")
if str(git(FORMAL, "branch", "--show-current")).strip() != FORMAL_BRANCH:
    raise RuntimeError("formal branch changed")
if len(formal_porcelain.splitlines()) != FORMAL_PORCELAIN_COUNT:
    raise RuntimeError("formal porcelain count changed")
if hashlib.sha256(formal_porcelain).hexdigest() != FORMAL_PORCELAIN_SHA256:
    raise RuntimeError("formal porcelain hash changed")

gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
gpu = [row.strip() for row in gpu if row.strip()]

report = {
    "schema_version": 1,
    "role": "cofitok_quality_bridge_100k_recovery_supervisor_deployment_receipt",
    "status": "deployed_observing_existing_waiter",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "supervisor": {
        **supervisor_identity,
        "pid": pid,
        "command": command,
        "cwd": cwd,
        "lock_exclusion_verified": True,
        "poll_seconds": status["poll_seconds"],
        "max_recovery_attempts": status["max_recovery_attempts"],
        "retryable_exit_codes": status["retryable_exit_codes"],
        "monitor_health_exit_86_retryable": False,
        "watchdog_transient_evidence_required_for_87_88_89": True,
        "script_self_hash_revalidated_each_heartbeat": True,
    },
    "runtime": {
        "supervisor_status": {
            "path": status_path.as_posix(),
            "bytes": status_path.stat().st_size,
            "sha256": sha(status_path),
            "updated_at": status["updated_at"],
        },
        "idle_waiter_status": {
            "path": waiter_status_path.as_posix(),
            "bytes": waiter_status_path.stat().st_size,
            "sha256": sha(waiter_status_path),
            "pid": int((BASE / "idle_waiter.pid").read_text().strip()),
            "updated_at": waiter["updated_at"],
        },
        "gpu_compute_rows": gpu,
        "launch_receipt_present": (REPORT_ROOT / "launch_receipt.json").exists(),
        "quality_bridge_result_present": (REPORT_ROOT / "quality_bridge_result.json").exists(),
    },
    "git": {
        "execution": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "formal": {
            "revision": FORMAL_REVISION,
            "branch": FORMAL_BRANCH,
            "porcelain_count": FORMAL_PORCELAIN_COUNT,
            "porcelain_sha256": FORMAL_PORCELAIN_SHA256,
        },
    },
    "authorization_boundary": {
        "quality_bridge_recovery_allowed": True,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "release_authorization_allowed": False,
        "report_is_promotion_gate": False,
        "unrelated_gpu_processes_must_not_be_signaled": True,
    },
    "prelaunch_contract_audit": audit_identity,
    "issues": [],
}

temporary = RECEIPT.with_name(f".{RECEIPT.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(RECEIPT)
print(RECEIPT)
