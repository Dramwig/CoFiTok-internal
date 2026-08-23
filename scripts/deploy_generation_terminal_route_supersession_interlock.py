from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


QUALITY_ROOT = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1"
)
TRAINING_CHECKOUT = Path(
    "/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal"
)
TRAINING_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TRAINING_TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
TRAINING_BRANCH = "scale/generation-stability-quality-bridge-100k"
STANDING_AUTHORIZATION = Path(
    "/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json"
)
STANDING_AUTHORIZATION_SHA256 = (
    "5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df"
)

PAIR_MONITOR = QUALITY_ROOT / "pair_monitor.json"
PAIR_MONITOR_SHA256 = (
    "78306a02a47d96be78f325ec2cd708ff4741f248c544fbc861dc9acfa829a2eb"
)
LEGACY_RUNTIME_GUARD = (
    QUALITY_ROOT
    / "reports/runtime_compute_claim_guard_v1/runtime_compute_claim_guard.json"
)
LEGACY_RUNTIME_PAIR_MONITOR_SHA256 = (
    "501e24ddf834c0bd8b429e47acdfaba7104badf3f3a3a0688009cbc9a37fe192"
)
STRICT_RUNTIME_GUARD = (
    QUALITY_ROOT
    / "reports/runtime_compute_claim_guard_strict_replay_v3/"
    "runtime_compute_claim_guard.json"
)
STRICT_RUNTIME_GUARD_SHA256 = (
    "f39e3e4238aba2ba098f807c331b4dadc6fb1690064bb376bdfba307b88fa3cf"
)
STRICT_RUNTIME_STATUS = (
    QUALITY_ROOT
    / "reports/runtime_compute_claim_guard_strict_replay_v3/waiter_status.json"
)

LEGACY_TERMINAL_GUARD = (
    QUALITY_ROOT
    / "reports/terminal_system_claim_guard_v1/terminal_system_claim_guard.json"
)
LEGACY_TERMINAL_STATUS = (
    QUALITY_ROOT / "reports/terminal_system_claim_guard_v1/waiter_status.json"
)
AUTHORITATIVE_TERMINAL_GUARD = (
    QUALITY_ROOT
    / "reports/terminal_system_claim_guard_v2_runtime_strict/"
    "terminal_system_claim_guard.json"
)
AUTHORITATIVE_TERMINAL_STATUS = (
    QUALITY_ROOT
    / "reports/terminal_system_claim_guard_v2_runtime_strict/waiter_status.json"
)
QUALITY_RESULT = QUALITY_ROOT / "reports/quality_bridge_result.json"
FOLLOWUP_DECISION = (
    QUALITY_ROOT / "reports/followup_experiment_decision_exposure_aware_v2.json"
)
DENSE_TERMINAL_PROGRESS = (
    QUALITY_ROOT
    / "dense_rollout_x0_u2_ema_teacher/terminal_100k/"
    "samples_10000_ddim100_cfg15/sampling_progress.json"
)

CONTROL_DIR = QUALITY_ROOT / "reports/terminal_route_supersession_interlock_v1"
CANONICAL_RECEIPT = CONTROL_DIR / "supersession_receipt.json"

ROLE = "generation_terminal_route_supersession_interlock"
MARKER_ROLE = "generation_legacy_gpu_route_supersession_marker"
SCHEMA_VERSION = 1
SCOPE = {
    "cpu_only": True,
    "diagnostic_non_authorizing": True,
    "static_filesystem_interlock_only": True,
    "checkpoint_payload_loading_allowed": False,
    "gpu_execution_allowed": False,
    "new_gpu_supervisor_launch_allowed": False,
    "process_signals_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "upstream_evidence_modified": False,
}


@dataclass(frozen=True)
class LegacyRoute:
    name: str
    output_root: Path
    interlock_path: Path
    interlock_kind: str
    checkout: Path
    revision: str
    tree: str
    branch: str
    supervisor_source: Path | None
    supervisor_sha256: str | None
    runbook_source: Path
    runbook_sha256: str
    status_path: Path | None
    expected_waiting_detail: str | None
    process_needles: tuple[str, ...]
    cuda_hidden_value: str | None
    supervisor_required_snippets: tuple[str, ...]
    runbook_required_snippets: tuple[str, ...]


FACTORIZATION_OUTPUT = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_factorization_quality_regression_v1"
)
CONDITIONING_OUTPUT = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_probe1k_v1"
)
RANDOM_TOKEN_OUTPUT = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_random_token_semantic_visual_v1"
)

LEGACY_ROUTES = (
    LegacyRoute(
        name="factorization_quality_regression_v1",
        output_root=FACTORIZATION_OUTPUT,
        interlock_path=Path(f"{FACTORIZATION_OUTPUT}.lock"),
        interlock_kind="directory_with_marker",
        checkout=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "factorization-classifier-integrity-14e82e4"
        ),
        revision="14e82e43f128e92e70d25c020548e7192a561df9",
        tree="6fdde0b748687f0fcf631b6f3c589ef82436e253",
        branch="analysis/generation-factorization-classifier-integrity-v1-20260822",
        supervisor_source=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "factorization-classifier-integrity-14e82e4/scripts/"
            "run_generation_factorization_quality_regression_supervisor.py"
        ),
        supervisor_sha256=(
            "3f253b257685f958e04c469259d249d91b9cee8032f0fc0d1474b2c8656bcbbf"
        ),
        runbook_source=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "factorization-classifier-integrity-14e82e4/artifacts/runbooks/"
            "generation_factorization_quality_regression_probe_v1.sh"
        ),
        runbook_sha256=(
            "8ddfbb55b3f3bd05f224e667c127f99679f4df37eb2306b41d424c794f3c8322"
        ),
        status_path=(
            QUALITY_ROOT
            / "reports/factorization_quality_regression_supervisor_v1/"
            "supervisor_status.json"
        ),
        expected_waiting_detail="waiting_for_quality_bridge_followup_decision",
        process_needles=(
            "scripts/run_generation_factorization_quality_regression_supervisor.py",
            "factorization-classifier-integrity-14e82e4",
        ),
        cuda_hidden_value="-1",
        supervisor_required_snippets=(
            'if output_root.exists() or Path(f"{output_root}.lock").exists():',
            'raise FileExistsError("factorization-regression output or lock already exists")',
            "child = subprocess.Popen(",
        ),
        runbook_required_snippets=(
            'LOCK="${OUTPUT_ROOT}.lock"',
            'if ! mkdir "$LOCK"; then',
            str(FACTORIZATION_OUTPUT),
        ),
    ),
    LegacyRoute(
        name="conditioning_ranking_v1",
        output_root=CONDITIONING_OUTPUT,
        interlock_path=Path(f"{CONDITIONING_OUTPUT}.lock"),
        interlock_kind="directory_with_marker",
        checkout=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "conditioning-classifier-integrity-fc78b85/CoFiTok-internal"
        ),
        revision="fc78b85ac78faa1498d749c6a24d78289cde8438",
        tree="587145f7962332ae74b89e177e29dbcf38b3139e",
        branch="analysis/generation-conditioning-classifier-integrity-v1-20260822",
        supervisor_source=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "conditioning-classifier-integrity-fc78b85/CoFiTok-internal/scripts/"
            "run_generation_conditioning_ranking_probe_supervisor.py"
        ),
        supervisor_sha256=(
            "c90b3bee114ff7d69d11c32ede4969074e9811aa1abd83ac604480c79564492f"
        ),
        runbook_source=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "conditioning-classifier-integrity-fc78b85/CoFiTok-internal/"
            "artifacts/runbooks/generation_conditioning_ranking_four_arm_probe1k_v1.sh"
        ),
        runbook_sha256=(
            "fc1b2c08d5a64b2776c9b720ddc528ba77559a12fac31d7a6e40dfe1ad268686"
        ),
        status_path=Path(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/"
            "conditioning_ranking_four_arm_probe1k_standing_auth_v1/"
            "fc78b85ac78faa1498d749c6a24d78289cde8438/supervisor_status.json"
        ),
        expected_waiting_detail="waiting_for_quality_bridge_followup_decision",
        process_needles=(
            "scripts/run_generation_conditioning_ranking_probe_supervisor.py",
            "conditioning-classifier-integrity-fc78b85",
        ),
        cuda_hidden_value="-1",
        supervisor_required_snippets=(
            'if output_root.exists() or Path(f"{output_root}.lock").exists():',
            'raise FileExistsError("conditioning-ranking output or lock already exists")',
            "child = subprocess.Popen(",
        ),
        runbook_required_snippets=(
            "LOCK_DIR=${OUTPUT_ROOT}.lock",
            'if ! mkdir "$LOCK_DIR" 2>/dev/null; then',
            str(CONDITIONING_OUTPUT),
        ),
    ),
    LegacyRoute(
        name="random_token_semantic_visual_v1",
        output_root=RANDOM_TOKEN_OUTPUT,
        interlock_path=RANDOM_TOKEN_OUTPUT,
        interlock_kind="regular_file_marker",
        checkout=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "random-token-classifier-integrity-4ca66c8/CoFiTok-internal"
        ),
        revision="4ca66c83a573a04345def51d3501919b6b427a4c",
        tree="09bc1f2dcb186ccc0aabe0789c00daf29cdf116b",
        branch="analysis/generation-random-token-classifier-integrity-v1-20260823",
        supervisor_source=None,
        supervisor_sha256=None,
        runbook_source=Path(
            "/root/autodl-tmp/CoFiTok/checkouts/"
            "random-token-classifier-integrity-4ca66c8/CoFiTok-internal/"
            "artifacts/runbooks/generation_random_token_semantic_visual_100k_v1.sh"
        ),
        runbook_sha256=(
            "8e760b6ff7c498ed461b131b39a564a5ad18de8b81ce0b6081477638d29c32d1"
        ),
        status_path=None,
        expected_waiting_detail=None,
        process_needles=(
            "/tmp/cofitok-random-token-waiter-eae02a6.sh",
            "generation_random_token_semantic_visual_100k_v1.sh",
        ),
        cuda_hidden_value=None,
        supervisor_required_snippets=(),
        runbook_required_snippets=(
            f'TERMINAL_SYSTEM_GUARD="{LEGACY_TERMINAL_GUARD}"',
            "FACTORIZATION_SUPERVISOR_STATUS=",
            'if supervisor.get("status") != "completed":',
            '"matched_factorization_quality_regression_diagnostic_completed"',
            'if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then',
            str(RANDOM_TOKEN_OUTPUT),
        ),
    ),
)

RANDOM_TOKEN_WRAPPER = Path("/tmp/cofitok-random-token-waiter-eae02a6.sh")
RANDOM_TOKEN_WRAPPER_SHA256 = (
    "d569a3019dfd4709de9b86650f11a590596e53c4b362b08e0c934d174c5a9b66"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_symlink_chain(path: str | Path, *, label: str) -> Path:
    target = Path(os.path.abspath(Path(path).expanduser()))
    current = target
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    return target


def file_identity(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    source = reject_symlink_chain(path, label="evidence file")
    if not source.is_file():
        raise FileNotFoundError(f"evidence file is missing: {source}")
    identity = {
        "path": source.resolve().as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if expected_sha256 is not None:
        require(
            identity["sha256"] == expected_sha256,
            f"evidence SHA256 differs: {source}",
        )
    return identity


def read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, label=label)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def git_identity(path: str | Path) -> dict[str, Any]:
    checkout = reject_symlink_chain(path, label="Git checkout")

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args], text=True
        ).strip()

    return {
        "path": checkout.resolve().as_posix(),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(
            git("status", "--porcelain=v1", "--untracked-files=no")
        ),
    }


def validate_git_identity(
    path: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    observed = git_identity(path)
    require(observed["revision"] == revision, f"{label} revision differs")
    require(observed["tree"] == tree, f"{label} tree differs")
    require(observed["branch"] == branch, f"{label} branch differs")
    require(observed["tracked_dirty"] is False, f"{label} tracked checkout is dirty")
    return observed


def _proc_fields(pid: int) -> tuple[list[str], bytes]:
    root = Path("/proc") / str(pid)
    raw_cmdline = (root / "cmdline").read_bytes()
    raw_stat = (root / "stat").read_text(encoding="utf-8")
    fields = raw_stat.rsplit(")", 1)[1].strip().split()
    return fields, raw_cmdline


def process_children(pid: int) -> list[int]:
    children: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields, _ = _proc_fields(int(entry.name))
            if int(fields[1]) == pid:
                children.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return sorted(children)


def process_identity(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    fields, raw_cmdline = _proc_fields(pid)
    environment: dict[str, str] = {}
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode("utf-8", "replace")
        if name in {
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONPATH",
        }:
            environment[name] = value.decode("utf-8", "replace")
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "state": fields[0],
        "nice": int(fields[16]),
        "start_ticks": int(fields[19]),
        "cwd": os.readlink(root / "cwd"),
        "cmdline": raw_cmdline.replace(b"\0", b" ")
        .decode("utf-8", "replace")
        .strip(),
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
        "environment": environment,
        "ionice": subprocess.check_output(
            ["ionice", "-p", str(pid)], text=True
        ).strip(),
        "children": process_children(pid),
    }


def discover_unique_process(needles: Iterable[str]) -> dict[str, Any]:
    required = tuple(needles)
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command.strip() and all(needle in command for needle in required):
            matches.append(int(entry.name))
    require(
        len(matches) == 1,
        f"expected exactly one process for {required}, observed {matches}",
    )
    return process_identity(matches[0])


def validate_process_runtime(
    process: Mapping[str, Any],
    *,
    expected_cwd: Path | None,
    expected_cuda: str | None,
    allow_sleep_child: bool,
) -> None:
    require(process.get("ppid") == 1, "legacy consumer is not detached under PID 1")
    require(process.get("nice") == 10, "legacy consumer nice value differs")
    require(process.get("ionice") == "idle", "legacy consumer ionice differs")
    environment = process.get("environment", {})
    require(environment.get("OMP_NUM_THREADS") == "1", "legacy consumer OMP differs")
    require(environment.get("MKL_NUM_THREADS") == "1", "legacy consumer MKL differs")
    if expected_cuda is not None:
        require(
            environment.get("CUDA_VISIBLE_DEVICES") == expected_cuda,
            "legacy consumer CUDA visibility differs",
        )
    if expected_cwd is not None:
        require(
            Path(str(process.get("cwd"))).resolve() == expected_cwd.resolve(),
            "legacy consumer cwd differs",
        )
    children = process.get("children", [])
    if not allow_sleep_child:
        require(children == [], "legacy GPU supervisor already owns a child process")
    else:
        for child_pid in children:
            child = process_identity(int(child_pid))
            require(
                child["cmdline"].startswith("sleep 300"),
                "random-token wrapper owns an unexpected child",
            )


def validate_source_contract(route: LegacyRoute) -> dict[str, Any]:
    checkout = validate_git_identity(
        route.checkout,
        revision=route.revision,
        tree=route.tree,
        branch=route.branch,
        label=f"{route.name} checkout",
    )
    sources: dict[str, Any] = {"checkout": checkout}
    if route.supervisor_source is not None:
        supervisor = file_identity(
            route.supervisor_source,
            expected_sha256=route.supervisor_sha256,
        )
        text = route.supervisor_source.read_text(encoding="utf-8")
        for snippet in route.supervisor_required_snippets:
            require(snippet in text, f"{route.name} supervisor contract snippet is missing")
        require(
            text.index(route.supervisor_required_snippets[0])
            < text.index("child = subprocess.Popen("),
            f"{route.name} lock check does not precede child launch",
        )
        sources["supervisor"] = {
            **supervisor,
            "required_snippets_present": True,
            "lock_check_precedes_child_launch": True,
        }
    runbook = file_identity(
        route.runbook_source, expected_sha256=route.runbook_sha256
    )
    runbook_text = route.runbook_source.read_text(encoding="utf-8")
    for snippet in route.runbook_required_snippets:
        require(snippet in runbook_text, f"{route.name} runbook contract snippet is missing")
    sources["runbook"] = {**runbook, "required_snippets_present": True}
    return sources


def validate_waiting_status(route: LegacyRoute, process: Mapping[str, Any]) -> dict[str, Any] | None:
    if route.status_path is None:
        return None
    status_payload = read_json_object(
        route.status_path, label=f"{route.name} supervisor status"
    )
    require(status_payload.get("status") == "waiting", f"{route.name} is not waiting")
    require(
        status_payload.get("detail") == route.expected_waiting_detail,
        f"{route.name} waiting detail differs",
    )
    require(status_payload.get("pid") == process.get("pid"), f"{route.name} PID differs")
    require(
        status_payload.get("child_pid") in (None, 0),
        f"{route.name} status already records a child",
    )
    return {
        "identity": file_identity(route.status_path),
        "status": status_payload.get("status"),
        "detail": status_payload.get("detail"),
        "pid": status_payload.get("pid"),
        "child_pid": status_payload.get("child_pid"),
    }


def gpu_processes() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows: list[dict[str, Any]] = []
    for line in output.splitlines() if output else []:
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 3, f"malformed nvidia-smi row: {line}")
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_memory_mib": int(fields[2]),
            }
        )
    return rows


def validate_live_gpu_chain() -> dict[str, Any]:
    progress = read_json_object(DENSE_TERMINAL_PROGRESS, label="dense terminal progress")
    require(progress.get("status") == "running", "dense terminal sampling is not running")
    completed = progress.get("completed_samples")
    total = progress.get("total_samples")
    require(
        isinstance(completed, int)
        and isinstance(total, int)
        and 0 <= completed < total == 10_000,
        "dense terminal sample accounting differs",
    )
    rows = gpu_processes()
    require(len(rows) == 1, f"expected sole dense terminal GPU owner, observed {rows}")
    require(rows[0]["pid"] == progress.get("pid"), "GPU owner differs from sampling progress")
    sampler = process_identity(rows[0]["pid"])
    expected_output = str(DENSE_TERMINAL_PROGRESS.parent)
    require("scripts/generate_samples.py" in sampler["cmdline"], "GPU owner is not sampler")
    require(expected_output in sampler["cmdline"], "GPU owner samples another output root")
    require("--num-samples 10000" in sampler["cmdline"], "dense sample count differs")
    require("--sample-steps 100" in sampler["cmdline"], "dense DDIM step count differs")
    return {
        "progress": file_identity(DENSE_TERMINAL_PROGRESS),
        "progress_state": {
            "status": progress["status"],
            "completed_samples": completed,
            "total_samples": total,
            "pid": progress["pid"],
        },
        "gpu_processes": rows,
        "sampler": sampler,
        "sole_gpu_owner_is_exact_dense_terminal_sampler": True,
    }


def validate_runtime_guard_supersession() -> dict[str, Any]:
    pair_identity = file_identity(PAIR_MONITOR, expected_sha256=PAIR_MONITOR_SHA256)
    pair = read_json_object(PAIR_MONITOR, label="terminal pair monitor")
    require(pair.get("status") == "pass", "pair monitor is not pass")
    require(pair.get("stage") == "complete", "pair monitor is not complete")
    require(pair.get("issues") == [], "pair monitor issues are non-empty")

    legacy_identity = file_identity(LEGACY_RUNTIME_GUARD)
    legacy = read_json_object(LEGACY_RUNTIME_GUARD, label="legacy runtime guard")
    legacy_pair = legacy.get("sources", {}).get("terminal_pair_monitor", {})
    require(legacy.get("status") == "pass", "legacy runtime guard is not pass")
    require(
        legacy_pair.get("sha256") == LEGACY_RUNTIME_PAIR_MONITOR_SHA256,
        "legacy runtime guard pair-monitor binding differs",
    )
    require(
        legacy_pair.get("sha256") != pair_identity["sha256"],
        "legacy runtime guard unexpectedly binds the authoritative pair monitor",
    )

    strict_identity = file_identity(
        STRICT_RUNTIME_GUARD, expected_sha256=STRICT_RUNTIME_GUARD_SHA256
    )
    strict = read_json_object(STRICT_RUNTIME_GUARD, label="strict runtime guard")
    strict_pair = strict.get("sources", {}).get("terminal_pair_monitor", {})
    require(strict.get("status") == "pass", "strict runtime guard is not pass")
    require(
        strict_pair == pair_identity,
        "strict runtime guard does not bind the authoritative pair monitor",
    )
    strict_status = read_json_object(STRICT_RUNTIME_STATUS, label="strict runtime status")
    require(strict_status.get("status") == "pass", "strict runtime waiter is not pass")

    legacy_terminal_status = read_json_object(
        LEGACY_TERMINAL_STATUS, label="legacy terminal guard status"
    )
    authoritative_terminal_status = read_json_object(
        AUTHORITATIVE_TERMINAL_STATUS,
        label="authoritative terminal guard status",
    )
    require(
        legacy_terminal_status.get("status") == "waiting",
        "legacy terminal guard is no longer waiting",
    )
    require(
        authoritative_terminal_status.get("status") == "waiting",
        "authoritative terminal guard is no longer waiting",
    )
    return {
        "authoritative_pair_monitor": pair_identity,
        "legacy_runtime_guard": legacy_identity,
        "legacy_runtime_pair_monitor": dict(legacy_pair),
        "strict_runtime_guard": strict_identity,
        "strict_runtime_pair_monitor": dict(strict_pair),
        "strict_runtime_status": file_identity(STRICT_RUNTIME_STATUS),
        "legacy_terminal_status": file_identity(LEGACY_TERMINAL_STATUS),
        "authoritative_terminal_status": file_identity(AUTHORITATIVE_TERMINAL_STATUS),
        "legacy_binding_is_stale": True,
        "strict_binding_matches_authoritative_pair_monitor": True,
    }


def marker_payload(
    route: LegacyRoute,
    *,
    control_git: Mapping[str, Any],
    control_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": MARKER_ROLE,
        "status": "active",
        "route": route.name,
        "legacy_output_root": route.output_root.as_posix(),
        "interlock_path": route.interlock_path.as_posix(),
        "interlock_kind": route.interlock_kind,
        "reason": "legacy_consumer_binds_stale_runtime_v1_terminal_guard",
        "legacy_terminal_guard": LEGACY_TERMINAL_GUARD.as_posix(),
        "authoritative_terminal_guard": AUTHORITATIVE_TERMINAL_GUARD.as_posix(),
        "canonical_receipt": CANONICAL_RECEIPT.as_posix(),
        "control_git": dict(control_git),
        "control_source": dict(control_source),
        "scope": dict(SCOPE),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = reject_symlink_chain(path, label="exclusive JSON output")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    fsync_directory(target.parent)
    return file_identity(target)


def _marker_file(route: LegacyRoute) -> Path:
    if route.interlock_kind == "directory_with_marker":
        return route.interlock_path / "supersession_marker.json"
    if route.interlock_kind == "regular_file_marker":
        return route.interlock_path
    raise ValueError(f"unknown interlock kind: {route.interlock_kind}")


def install_or_verify_interlock(
    route: LegacyRoute,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    require(not route.output_root.is_symlink(), f"{route.name} output root is a symlink")
    if route.interlock_kind == "directory_with_marker":
        require(
            not route.output_root.exists(),
            f"{route.name} legacy output root already exists",
        )
        lock = reject_symlink_chain(route.interlock_path, label=f"{route.name} lock")
        created_lock = False
        if not lock.exists():
            os.mkdir(lock, 0o555)
            created_lock = True
            fsync_directory(lock.parent)
        require(lock.is_dir() and not lock.is_symlink(), f"{route.name} lock is not a directory")
        marker = _marker_file(route)
        if marker.exists():
            require(
                read_json_object(marker, label=f"{route.name} marker") == dict(payload),
                f"{route.name} existing marker differs",
            )
        else:
            require(
                created_lock,
                f"{route.name} existing lock has no recognized marker",
            )
            # Temporarily grant owner write permission only while publishing the marker.
            os.chmod(lock, 0o755)
            try:
                write_json_exclusive(marker, payload)
            finally:
                os.chmod(lock, 0o555)
        fsync_directory(lock)
    elif route.interlock_kind == "regular_file_marker":
        marker = reject_symlink_chain(
            route.interlock_path, label=f"{route.name} regular-file interlock"
        )
        if marker.exists():
            require(marker.is_file() and not marker.is_symlink(), f"{route.name} marker differs")
            require(
                read_json_object(marker, label=f"{route.name} marker") == dict(payload),
                f"{route.name} existing marker differs",
            )
        else:
            write_json_exclusive(marker, payload)
    else:
        raise ValueError(f"unknown interlock kind: {route.interlock_kind}")
    marker = _marker_file(route)
    require(read_json_object(marker, label=f"{route.name} marker") == dict(payload), f"{route.name} marker replay differs")
    mode = stat.S_IMODE(marker.stat().st_mode)
    return {
        "route": route.name,
        "legacy_output_root": route.output_root.as_posix(),
        "interlock_path": route.interlock_path.as_posix(),
        "interlock_kind": route.interlock_kind,
        "marker": file_identity(marker),
        "marker_mode_octal": oct(mode),
        "active": True,
    }


def validate_preterminal_absence(routes: Iterable[LegacyRoute]) -> None:
    for path, label in (
        (QUALITY_RESULT, "quality bridge result"),
        (FOLLOWUP_DECISION, "follow-up decision"),
        (LEGACY_TERMINAL_GUARD, "legacy terminal guard"),
        (AUTHORITATIVE_TERMINAL_GUARD, "authoritative terminal guard"),
    ):
        require(not path.exists(), f"{label} appeared before interlock deployment")
    for route in routes:
        require(
            not route.output_root.exists(),
            f"{route.name} legacy output root already exists",
        )
        require(
            not route.interlock_path.exists(),
            f"{route.name} interlock path already exists",
        )


def build_preflight(
    *,
    project: Path,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    control_git = validate_git_identity(
        project,
        revision=expected_revision,
        tree=expected_tree,
        branch=expected_branch,
        label="supersession interlock checkout",
    )
    source_path = project / "scripts/deploy_generation_terminal_route_supersession_interlock.py"
    control_source = file_identity(source_path, expected_sha256=expected_source_sha256)
    training_git = validate_git_identity(
        TRAINING_CHECKOUT,
        revision=TRAINING_REVISION,
        tree=TRAINING_TREE,
        branch=TRAINING_BRANCH,
        label="quality-bridge training checkout",
    )
    standing = file_identity(
        STANDING_AUTHORIZATION, expected_sha256=STANDING_AUTHORIZATION_SHA256
    )
    runtime_supersession = validate_runtime_guard_supersession()
    live_gpu_chain = validate_live_gpu_chain()
    wrapper = file_identity(
        RANDOM_TOKEN_WRAPPER, expected_sha256=RANDOM_TOKEN_WRAPPER_SHA256
    )

    routes: dict[str, Any] = {}
    for route in LEGACY_ROUTES:
        sources = validate_source_contract(route)
        process = discover_unique_process(route.process_needles)
        validate_process_runtime(
            process,
            expected_cwd=(route.checkout if route.name != "random_token_semantic_visual_v1" else Path("/root")),
            expected_cuda=route.cuda_hidden_value,
            allow_sleep_child=route.name == "random_token_semantic_visual_v1",
        )
        status_evidence = validate_waiting_status(route, process)
        routes[route.name] = {
            "legacy_output_root": route.output_root.as_posix(),
            "interlock_path": route.interlock_path.as_posix(),
            "interlock_kind": route.interlock_kind,
            "sources": sources,
            "process": process,
            "status": status_evidence,
            "legacy_terminal_guard_bound_in_cmdline": (
                str(LEGACY_TERMINAL_GUARD) in process["cmdline"]
                if route.name != "random_token_semantic_visual_v1"
                else True
            ),
        }
        if route.name != "random_token_semantic_visual_v1":
            require(
                routes[route.name]["legacy_terminal_guard_bound_in_cmdline"] is True,
                f"{route.name} no longer binds the legacy terminal guard",
            )

    validate_preterminal_absence(LEGACY_ROUTES)
    return {
        "control_git": control_git,
        "control_source": control_source,
        "training_git": training_git,
        "standing_authorization": standing,
        "runtime_guard_supersession": runtime_supersession,
        "live_gpu_chain": live_gpu_chain,
        "random_token_wrapper": wrapper,
        "legacy_routes": routes,
    }


def build_receipt(
    *,
    preflight: Mapping[str, Any],
    interlocks: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "generation_advantage_proven": False,
        "decision": "legacy_v1_gpu_route_consumers_superseded_fail_closed",
        "risk": {
            "legacy_runtime_guard_pair_monitor_sha256": (
                LEGACY_RUNTIME_PAIR_MONITOR_SHA256
            ),
            "authoritative_pair_monitor_sha256": PAIR_MONITOR_SHA256,
            "binding_mismatch_verified": True,
            "legacy_terminal_guard": LEGACY_TERMINAL_GUARD.as_posix(),
            "authoritative_terminal_guard": AUTHORITATIVE_TERMINAL_GUARD.as_posix(),
            "legacy_gpu_consumers": [route.name for route in LEGACY_ROUTES],
        },
        "scope": dict(SCOPE),
        "preflight": dict(preflight),
        "interlocks": dict(interlocks),
        "guarantees": {
            "legacy_factorization_child_launch_blocked_by_preexisting_lock_directory": True,
            "legacy_conditioning_child_launch_blocked_by_preexisting_lock_directory": True,
            "legacy_random_token_evaluator_blocked_by_non_directory_output_marker": True,
            "legacy_random_token_also_requires_completed_factorization_route": True,
            "no_existing_process_was_signaled": True,
            "no_gpu_process_was_started": True,
            "no_authoritative_terminal_evidence_was_modified": True,
            "corrected_route_must_use_new_versioned_output_roots": True,
        },
        "limitations": [
            "This receipt blocks only the three enumerated stale v1 GPU-capable consumers.",
            "It does not select a scientific route or authorize a corrected GPU experiment.",
            "It does not establish generation quality or a CoFiTok advantage.",
            "The legacy CPU-only terminal/comparison/completion artifacts remain historical and may still finish independently.",
        ],
    }


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = reject_symlink_chain(args.receipt_output, label="canonical receipt")
    require(receipt_path == CANONICAL_RECEIPT, "canonical receipt path differs")
    require(not receipt_path.exists(), "canonical supersession receipt already exists")
    require(not CONTROL_DIR.exists(), "supersession control directory already exists")

    preflight = build_preflight(
        project=args.project,
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
        expected_source_sha256=args.expected_source_sha256,
    )
    control_git = preflight["control_git"]
    control_source = preflight["control_source"]
    interlocks: dict[str, Any] = {}
    for route in LEGACY_ROUTES:
        payload = marker_payload(
            route,
            control_git=control_git,
            control_source=control_source,
        )
        interlocks[route.name] = install_or_verify_interlock(route, payload)

    os.mkdir(CONTROL_DIR, 0o555)
    os.chmod(CONTROL_DIR, 0o755)
    try:
        receipt = build_receipt(preflight=preflight, interlocks=interlocks)
        receipt_identity = write_json_exclusive(receipt_path, receipt)
    finally:
        os.chmod(CONTROL_DIR, 0o555)
    fsync_directory(CONTROL_DIR)
    verified = verify_receipt(receipt_path, expected_sha256=receipt_identity["sha256"])
    return {"receipt": receipt_identity, "verified": verified}


def _detect_legacy_gpu_children() -> list[dict[str, Any]]:
    runbook_prefixes = [f"bash {route.runbook_source}" for route in LEGACY_ROUTES]
    evaluator_needles = (
        "scripts/evaluate_generation_random_token_semantics.py",
        str(RANDOM_TOKEN_OUTPUT),
    )
    matches: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", "replace")
                .strip()
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(command.startswith(prefix) for prefix in runbook_prefixes) or all(
            needle in command for needle in evaluator_needles
        ):
            matches.append(process_identity(int(entry.name)))
    return matches


def verify_receipt(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    identity = file_identity(path, expected_sha256=expected_sha256)
    receipt = read_json_object(path, label="supersession receipt")
    require(receipt.get("schema_version") == SCHEMA_VERSION, "receipt schema differs")
    require(receipt.get("role") == ROLE, "receipt role differs")
    require(receipt.get("status") == "pass", "receipt status differs")
    require(receipt.get("scope") == SCOPE, "receipt scope differs")
    require(receipt.get("generation_advantage_proven") is False, "receipt scientific state differs")
    control_git = receipt.get("preflight", {}).get("control_git")
    control_source = receipt.get("preflight", {}).get("control_source")
    require(isinstance(control_git, dict), "receipt control Git is missing")
    require(isinstance(control_source, dict), "receipt control source is missing")
    for route in LEGACY_ROUTES:
        require(
            not route.output_root.is_dir(),
            f"{route.name} legacy output directory exists",
        )
        expected_marker = marker_payload(
            route, control_git=control_git, control_source=control_source
        )
        marker = _marker_file(route)
        require(marker.is_file() and not marker.is_symlink(), f"{route.name} marker is missing")
        require(
            read_json_object(marker, label=f"{route.name} marker") == expected_marker,
            f"{route.name} marker replay differs",
        )
        if route.interlock_kind == "directory_with_marker":
            require(
                route.interlock_path.is_dir() and not route.interlock_path.is_symlink(),
                f"{route.name} lock directory is missing",
            )
        else:
            require(
                route.interlock_path.is_file() and not route.interlock_path.is_symlink(),
                f"{route.name} regular-file interlock is missing",
            )
    legacy_children = _detect_legacy_gpu_children()
    require(legacy_children == [], f"legacy GPU child process is active: {legacy_children}")
    return {
        "status": "pass",
        "receipt": identity,
        "interlock_count": len(LEGACY_ROUTES),
        "legacy_gpu_children": legacy_children,
        "generation_advantage_proven": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install or verify static fail-closed interlocks for stale terminal-v1 "
            "GPU route consumers without signaling processes or launching GPU work."
        )
    )
    parser.add_argument(
        "--mode", choices=("preflight", "deploy", "verify"), required=True
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--receipt-output", type=Path, default=CANONICAL_RECEIPT)
    parser.add_argument("--expected-receipt-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        preflight = build_preflight(
            project=args.project,
            expected_revision=args.expected_revision,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
            expected_source_sha256=args.expected_source_sha256,
        )
        result = {
            "status": "pass",
            "control_git": preflight["control_git"],
            "control_source": preflight["control_source"],
            "legacy_binding_is_stale": preflight["runtime_guard_supersession"][
                "legacy_binding_is_stale"
            ],
            "sole_gpu_owner_is_exact_dense_terminal_sampler": preflight[
                "live_gpu_chain"
            ]["sole_gpu_owner_is_exact_dense_terminal_sampler"],
            "dense_terminal_progress": preflight["live_gpu_chain"][
                "progress_state"
            ],
            "legacy_routes": {
                name: {
                    "pid": evidence["process"]["pid"],
                    "start_ticks": evidence["process"]["start_ticks"],
                    "status": evidence["status"],
                }
                for name, evidence in preflight["legacy_routes"].items()
            },
            "generation_advantage_proven": False,
        }
    elif args.mode == "deploy":
        result = deploy(args)
    else:
        require(
            isinstance(args.expected_receipt_sha256, str)
            and len(args.expected_receipt_sha256) == 64,
            "verify mode requires --expected-receipt-sha256",
        )
        result = verify_receipt(
            reject_symlink_chain(args.receipt_output, label="canonical receipt"),
            expected_sha256=args.expected_receipt_sha256,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
