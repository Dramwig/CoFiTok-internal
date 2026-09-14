from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_base128_quality_bridge_v1"
)
TRAINING = Path("/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal")
REPORT_DIR = BASE / "reports/terminal_guard_consumer_integrity_audit_v1"
REPORT_PATH = REPORT_DIR / "consumer_coverage.json"
TRAINING_REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TRAINING_TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
TRAINING_BRANCH = "scale/generation-stability-quality-bridge-100k"
DATASET_SHA256 = "6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659"
RUNTIME_SHA256 = "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def file_identity(path: str | Path) -> dict:
    source = Path(path)
    require(source.is_file(), f"missing file: {source}")
    require(not source.is_symlink(), f"symlink file rejected: {source}")
    payload = source.read_bytes()
    return {
        "path": source.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def git_identity(path: str | Path) -> dict:
    source = Path(path)
    return {
        "path": source.resolve().as_posix(),
        "revision": run("git", "-C", str(source), "rev-parse", "HEAD"),
        "tree": run("git", "-C", str(source), "rev-parse", "HEAD^{tree}"),
        "branch": run("git", "-C", str(source), "branch", "--show-current"),
        "tracked_dirty": bool(
            run(
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            )
        ),
    }


def discover_one(*needles: str) -> int:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if command.strip() and all(needle in command for needle in needles):
            matches.append(int(entry.name))
    require(len(matches) == 1, f"expected one process for {needles}, got {matches}")
    return matches[0]


def process_identity(pid: int) -> dict:
    root = Path("/proc") / str(pid)
    require(root.is_dir(), f"process missing: {pid}")
    raw_cmdline = (root / "cmdline").read_bytes()
    stat = (root / "stat").read_text(encoding="utf-8")
    fields = stat.rsplit(")", 1)[1].strip().split()
    environment: dict[str, str] = {}
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode("utf-8", "replace")
        if name in {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH"}:
            environment[name] = value.decode("utf-8", "replace")
    children: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            child_fields = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
            if int(child_fields[1]) == pid:
                children.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            pass
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "state": fields[0],
        "nice": int(fields[16]),
        "start_ticks": int(fields[19]),
        "started_at_local": run("ps", "-p", str(pid), "-o", "lstart="),
        "cwd": os.readlink(root / "cwd"),
        "cmdline": raw_cmdline.replace(b"\0", b" ").decode("utf-8", "replace").strip(),
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
        "environment": environment,
        "ionice": run("ionice", "-p", str(pid)),
        "children": sorted(children),
    }


def last_jsonl(path: Path) -> dict:
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    require(isinstance(last, dict), f"no metric row: {path}")
    return last


def finite_numbers(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    return True


checkout_specs = {
    "terminal_guard_producer": (
        "/root/autodl-tmp/CoFiTok/checkouts/terminal-system-classifier-integrity-4087f42/CoFiTok-internal",
        "4087f4293e3fd197c81cbfa9029f3d6083654415",
        "7d5acbe2542f5c0584a7fd847fdb806e6757ba86",
        "analysis/generation-terminal-classifier-integrity-v1-20260822",
    ),
    "terminal_completion_consumer": (
        "/root/autodl-tmp/CoFiTok/checkouts/class-fidelity-integrity-a253d56/CoFiTok-internal",
        "a253d56e49b78e1c8ddb10c3bd07af5aa22e5919",
        "5871db9dc625fbd9809f9b6ffc6fd4114754c021",
        "analysis/generation-class-fidelity-integrity-v1-20260822",
    ),
    "factorization_consumer": (
        "/root/autodl-tmp/CoFiTok/checkouts/factorization-classifier-integrity-14e82e4",
        "14e82e43f128e92e70d25c020548e7192a561df9",
        "6fdde0b748687f0fcf631b6f3c589ef82436e253",
        "analysis/generation-factorization-classifier-integrity-v1-20260822",
    ),
    "conditioning_consumer": (
        "/root/autodl-tmp/CoFiTok/checkouts/conditioning-classifier-integrity-fc78b85/CoFiTok-internal",
        "fc78b85ac78faa1498d749c6a24d78289cde8438",
        "587145f7962332ae74b89e177e29dbcf38b3139e",
        "analysis/generation-conditioning-classifier-integrity-v1-20260822",
    ),
    "comparison_consumer": (
        "/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-comparison-classifier-integrity-8d11f0a/CoFiTok-internal",
        "8d11f0aad652197c922454104353f88b4f09d46a",
        "9259e1db0471ae3929c9e9657019e95dcc0022bc",
        "analysis/generation-quality-bridge-comparison-classifier-integrity-v1-20260823",
    ),
    "random_token_consumer": (
        "/root/autodl-tmp/CoFiTok/checkouts/random-token-classifier-integrity-4ca66c8/CoFiTok-internal",
        "4ca66c83a573a04345def51d3501919b6b427a4c",
        "09bc1f2dcb186ccc0aabe0789c00daf29cdf116b",
        "analysis/generation-random-token-classifier-integrity-v1-20260823",
    ),
}
checkouts: dict[str, dict] = {}
for role, (path, revision, tree, branch) in checkout_specs.items():
    observed = git_identity(path)
    require(observed["revision"] == revision, f"{role} revision differs")
    require(observed["tree"] == tree, f"{role} tree differs")
    require(observed["branch"] == branch, f"{role} branch differs")
    require(observed["tracked_dirty"] is False, f"{role} tracked checkout dirty")
    checkouts[role] = observed

training_git = git_identity(TRAINING)
require(training_git["revision"] == TRAINING_REVISION, "training revision differs")
require(training_git["tree"] == TRAINING_TREE, "training tree differs")
require(training_git["branch"] == TRAINING_BRANCH, "training branch differs")
require(training_git["tracked_dirty"] is False, "training tracked checkout dirty")

fixed_sources = {
    "standing_authorization": (
        "/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json",
        "5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df",
    ),
    "controller": (
        "/tmp/cofitok-quality-bridge-execution-cf0e5fa/dense_only_resume_20260822.py",
        "f29f8cda690febcdc4ddbd565c3a28f56549c59878e18bf5e688d4c532b8731d",
    ),
    "terminal_runbook": (
        str(TRAINING / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh"),
        "f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73",
    ),
}
fixed_source_evidence: dict[str, dict] = {}
for name, (path, expected_sha) in fixed_sources.items():
    identity = file_identity(path)
    require(identity["sha256"] == expected_sha, f"{name} SHA differs")
    fixed_source_evidence[name] = identity

source_specs = {
    "terminal_guard_builder": (
        checkout_specs["terminal_guard_producer"][0] + "/scripts/build_generation_terminal_system_claim_guard.py",
        "8aef79d8efa8c311b025cf2d3933e48a88f2db2367e158e8a5cabceadaf9ff76",
        [
            "def _verify_class_fidelity_classifier_sources",
            "physical_weight_sha256_replayed",
            "final_classifier_integrity",
            '"class_fidelity_classifier_physical_integrity_verified": True',
        ],
        "physically rehashes the fixed classifier, replays it again before return, and emits the policy bit only after verification",
    ),
    "terminal_completion_builder": (
        checkout_specs["terminal_completion_consumer"][0] + "/scripts/build_generation_quality_bridge_terminal_completion_audit.py",
        "27b9b3826fe416d8c3d4e17d34bf428ac89e2a9537261c93658146ac7057136a",
        [
            "def verify_class_fidelity_classifier_sources",
            "physical_weight_sha256_replayed",
            "final_class_fidelity_integrity",
            '"physical_classifier_weight_revalidated_before_and_after": True',
        ],
        "independently rehashes the classifier and rejects any before/after evidence change",
    ),
    "factorization_route_validator": (
        checkout_specs["factorization_consumer"][0] + "/src/cofitok/generation/factorization_quality_regression.py",
        "24256e926f6bc7d1dec6a12235484fd291bee9d566dd6d6305b2eb2fc0b9dfec",
        ["class_fidelity_classifier_physical_integrity_verified", 'classifier_integrity.get("status") != "verified"'],
        "requires the terminal policy bit and verified evidence before factorization authorization",
    ),
    "conditioning_route_validator": (
        checkout_specs["conditioning_consumer"][0] + "/src/cofitok/generation/conditioning_ranking_probe.py",
        "2187ad0dbd515101f76d9be4c384a4e5c5f63e0181e8c8dfac5c16b72fdff9dc",
        ["class_fidelity_classifier_physical_integrity_verified", 'classifier_integrity.get("status") != "verified"'],
        "requires the terminal policy bit and verified evidence before conditioning authorization",
    ),
    "comparison_builder": (
        checkout_specs["comparison_consumer"][0] + "/scripts/build_generation_quality_bridge_comparison.py",
        "3f3e7f7d99ba25ad41e93db3c357ca50d1ce41164aa5afc65b657d4595ad4a61",
        ["class_fidelity_classifier_physical_integrity_verified", 'classifier_integrity.get("status") != "verified"'],
        "refuses to build the comparison without both classifier-integrity conditions",
    ),
    "random_token_runbook": (
        checkout_specs["random_token_consumer"][0] + "/artifacts/runbooks/generation_random_token_semantic_visual_100k_v1.sh",
        "8e760b6ff7c498ed461b131b39a564a5ad18de8b81ce0b6081477638d29c32d1",
        ["class_fidelity_classifier_physical_integrity_verified", 'classifier_integrity.get("status") != "verified"'],
        "preflight refuses the route without both classifier-integrity conditions",
    ),
    "random_token_evaluator": (
        checkout_specs["random_token_consumer"][0] + "/scripts/evaluate_generation_random_token_semantics.py",
        "55905cfb56b702bbe7081b210e8987c89d2b3451f0f1ded8458c16c88fcdadec",
        ["class_fidelity_classifier_physical_integrity_verified", 'classifier_integrity.get("status") != "verified"'],
        "the evaluator independently revalidates the same conditions",
    ),
}
source_evidence: dict[str, dict] = {}
for name, (path, expected_sha, snippets, condition) in source_specs.items():
    identity = file_identity(path)
    require(identity["sha256"] == expected_sha, f"{name} SHA differs")
    text = Path(path).read_text(encoding="utf-8")
    require(all(snippet in text for snippet in snippets), f"{name} required fail-closed source is missing")
    source_evidence[name] = {
        **identity,
        "required_snippets": snippets,
        "required_snippets_present": True,
        "fail_closed_condition": condition,
    }

launcher_specs = {
    "terminal_guard_producer": (
        checkout_specs["terminal_guard_producer"][0] + "/scripts/run_generation_terminal_system_claim_guard_waiter.py",
        "c0b9db1acb4f7d46be04ff191c7272a893cb48faae039aa2703bafba922ab561",
    ),
    "terminal_completion_consumer": (
        checkout_specs["terminal_completion_consumer"][0] + "/scripts/wait_generation_quality_bridge_terminal_completion_audit.py",
        "931f910987a478af6029bd1045ae3ddb4f6bde227b53ee1825e598d0757a4f1b",
    ),
    "factorization_consumer": (
        checkout_specs["factorization_consumer"][0] + "/scripts/run_generation_factorization_quality_regression_supervisor.py",
        "3f253b257685f958e04c469259d249d91b9cee8032f0fc0d1474b2c8656bcbbf",
    ),
    "conditioning_consumer": (
        checkout_specs["conditioning_consumer"][0] + "/scripts/run_generation_conditioning_ranking_probe_supervisor.py",
        "c90b3bee114ff7d69d11c32ede4969074e9811aa1abd83ac604480c79564492f",
    ),
    "comparison_consumer": (
        checkout_specs["comparison_consumer"][0] + "/scripts/wait_generation_quality_bridge_comparison.py",
        "db8333674547b0aa5b6bea7489b9c884dde561d5d99f3b2acadd2c1fbb9dc150",
    ),
    "random_token_wrapper": (
        "/tmp/cofitok-random-token-waiter-eae02a6.sh",
        "d569a3019dfd4709de9b86650f11a590596e53c4b362b08e0c934d174c5a9b66",
    ),
}
launcher_evidence: dict[str, dict] = {}
for name, (path, expected_sha) in launcher_specs.items():
    identity = file_identity(path)
    require(identity["sha256"] == expected_sha, f"{name} launcher SHA differs")
    launcher_evidence[name] = identity

process_specs = {
    "terminal_guard_producer": ("scripts/run_generation_terminal_system_claim_guard_waiter.py", checkout_specs["terminal_guard_producer"][0]),
    "terminal_completion_consumer": ("scripts/wait_generation_quality_bridge_terminal_completion_audit.py", checkout_specs["terminal_completion_consumer"][0]),
    "factorization_consumer": ("scripts/run_generation_factorization_quality_regression_supervisor.py", checkout_specs["factorization_consumer"][0]),
    "conditioning_consumer": ("scripts/run_generation_conditioning_ranking_probe_supervisor.py", checkout_specs["conditioning_consumer"][0]),
    "comparison_consumer": ("scripts/wait_generation_quality_bridge_comparison.py", checkout_specs["comparison_consumer"][0]),
    "random_token_wrapper": ("/tmp/cofitok-random-token-waiter-eae02a6.sh", checkout_specs["random_token_consumer"][0]),
}
processes: dict[str, dict] = {}
for name, needles in process_specs.items():
    evidence = process_identity(discover_one(*needles))
    require(evidence["ppid"] == 1, f"{name} not detached under PID 1")
    require(evidence["nice"] == 10 and evidence["ionice"] == "idle", f"{name} CPU scheduling differs")
    require(evidence["environment"].get("OMP_NUM_THREADS") == "1", f"{name} OMP differs")
    require(evidence["environment"].get("MKL_NUM_THREADS") == "1", f"{name} MKL differs")
    processes[name] = evidence

controller = process_identity(discover_one("/tmp/cofitok-quality-bridge-execution-cf0e5fa/dense_only_resume_20260822.py"))
watchdog = process_identity(discover_one("scripts/run_generation_training_watchdog.py", "dense_rollout_x0_u2_ema_teacher"))
pair_monitor_process = process_identity(discover_one("scripts/monitor_generation_pair.py", "generation_stability_full_data_quality_bridge_100k"))
require(controller["ppid"] == 1, "controller parent differs")
require(watchdog["ppid"] == controller["pid"], "watchdog parent differs")
require(pair_monitor_process["ppid"] == controller["pid"], "pair monitor parent differs")

gpu_rows = run(
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
)
gpu_processes: list[dict] = []
for line in gpu_rows.splitlines() if gpu_rows else []:
    fields = [field.strip() for field in line.split(",")]
    require(len(fields) == 3, f"malformed GPU row: {line}")
    gpu_processes.append(
        {"pid": int(fields[0]), "process_name": fields[1], "used_memory_mib": int(fields[2])}
    )
require(len(gpu_processes) == 1, f"expected sole GPU owner, got {gpu_processes}")
trainer = process_identity(gpu_processes[0]["pid"])
require("scripts/train_generation.py" in trainer["cmdline"], "GPU owner is not trainer")
require("imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json" in trainer["cmdline"], "GPU owner is not exact dense trainer")
require(trainer["ppid"] == watchdog["pid"], "trainer parent differs")
for name, evidence in processes.items():
    require(evidence["pid"] != trainer["pid"], f"{name} unexpectedly owns GPU")
    require(trainer["pid"] not in evidence["children"], f"{name} owns trainer child")

pair_path = BASE / "pair_monitor.json"
pair = read_json(pair_path)
require(pair.get("status") == "running", "pair monitor not running")
require(pair.get("stage") == "dense_identity_training", "pair stage differs")
require(pair.get("issues") == [], "pair issues non-empty")
require(pair.get("git") == {"branch": TRAINING_BRANCH, "revision": TRAINING_REVISION, "tracked_dirty": False}, "pair Git differs")
require(pair.get("gpu_contention", {}).get("current", {}).get("unrelated_gpu_processes") == [], "unrelated GPU process observed")

cofitok_metric = last_jsonl(BASE / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher/train_metrics.jsonl")
dense_metric = last_jsonl(BASE / "dense_rollout_x0_u2_ema_teacher/train_metrics.jsonl")
require(cofitok_metric.get("step") == 100000 and cofitok_metric.get("samples_seen") == 6400000, "CoFiTok terminal metric differs")
require(50000 < int(dense_metric.get("step", 0)) < 90000, "dense is outside expected pre-90K interval")
require(dense_metric.get("samples_seen") == dense_metric.get("step") * 64, "dense sample accounting differs")
require(finite_numbers(cofitok_metric) and finite_numbers(dense_metric), "non-finite training metric")

audit_dir = BASE / "reports/checkpoint_audits"
checkpoint_waiters: dict[str, dict] = {}
for alias in ("cofitok", "dense"):
    for step in (90000, 95000, 100000):
        key = f"{alias}_{step}"
        status_path = audit_dir / f"{alias}_checkpoint_step_{step:08d}_waiter_status.json"
        status = read_json(status_path)
        evidence = {
            "status_report": file_identity(status_path),
            "status": status.get("status"),
            "detail": status.get("detail"),
            "pid": status.get("pid"),
        }
        audit_path = audit_dir / f"{alias}_checkpoint_step_{step:08d}_physical_integrity_audit.json"
        if alias == "cofitok":
            require(status.get("status") == "pass", f"{key} waiter not pass")
            audit = read_json(audit_path)
            require(audit.get("status") == "pass", f"{key} audit not pass")
            require(audit.get("checkpoint", {}).get("physical_sha256_verified") is True, f"{key} physical SHA not verified")
            require(audit.get("latest_pointer", {}).get("exact_target_binding") is True, f"{key} latest binding differs")
            require(audit.get("metrics", {}).get("strictly_increasing") is True, f"{key} metrics not increasing")
            require(audit.get("metrics", {}).get("samples_seen_binding_verified") is True, f"{key} sample binding differs")
            target = audit.get("metrics", {}).get("target_row", {})
            require(target.get("step") == step and target.get("samples_seen") == step * 64, f"{key} target row differs")
            integrity = audit.get("checkpoint", {}).get("integrity", {})
            require(integrity.get("git_revision") == TRAINING_REVISION, f"{key} Git differs")
            require(integrity.get("dataset_identity_sha256") == DATASET_SHA256, f"{key} dataset differs")
            require(integrity.get("runtime_environment_sha256") == RUNTIME_SHA256, f"{key} runtime differs")
            evidence.update(
                {
                    "audit": file_identity(audit_path),
                    "physical_sha256_verified": True,
                    "latest_exact_target_binding": True,
                    "metrics_strictly_increasing": True,
                    "samples_seen": target["samples_seen"],
                }
            )
        else:
            require(status.get("status") == "waiting" and status.get("detail") == "checkpoint_missing", f"{key} waiter state differs")
            require(not audit_path.exists(), f"{key} audit unexpectedly exists")
            waiter = process_identity(int(status["pid"]))
            require(waiter["ppid"] == 1 and waiter["nice"] == 10 and waiter["ionice"] == "idle", f"{key} runtime differs")
            require(waiter["environment"].get("CUDA_VISIBLE_DEVICES") == "", f"{key} CUDA visibility differs")
            require(waiter["environment"].get("OMP_NUM_THREADS") == "1", f"{key} OMP differs")
            require(waiter["environment"].get("MKL_NUM_THREADS") == "1", f"{key} MKL differs")
            evidence.update({"waiter_process": waiter, "audit_absent": True})
        checkpoint_waiters[key] = evidence

replay_status_path = audit_dir / "dense_checkpoint_integrity_replay_waiter_status.json"
replay_status = read_json(replay_status_path)
require(replay_status.get("status") == "waiting", "replay waiter not waiting")
require(replay_status.get("detail") == "waiting_for_source_audit_step_00090000", "replay detail differs")
replay_process = process_identity(int(replay_status["pid"]))
require(replay_process["ppid"] == 1 and replay_process["nice"] == 19 and replay_process["ionice"] == "idle", "replay waiter runtime differs")
require(replay_process["environment"].get("CUDA_VISIBLE_DEVICES") == "", "replay CUDA visibility differs")

status_paths = {
    "runtime_fairness": BASE / "reports/runtime_compute_fairness/waiter_status.json",
    "runtime_claim_guard": BASE / "reports/runtime_compute_claim_guard_v1/waiter_status.json",
    "terminal_system_guard": BASE / "reports/terminal_system_claim_guard_v1/waiter_status.json",
    "terminal_completion": BASE / "reports/terminal_completion_audit_v1/waiter_status.json",
    "quality_comparison": BASE / "reports/quality_bridge_comparison_v1/waiter_status.json",
    "factorization_supervisor": BASE / "reports/factorization_quality_regression_supervisor_v1/supervisor_status.json",
    "conditioning_supervisor": Path(
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/preparations/"
        "conditioning_ranking_four_arm_probe1k_standing_auth_v1/"
        "fc78b85ac78faa1498d749c6a24d78289cde8438/supervisor_status.json"
    ),
}
statuses: dict[str, dict] = {}
for name, path in status_paths.items():
    value = read_json(path)
    require(value.get("status") == "waiting", f"{name} is not waiting")
    require(value.get("child_pid") in (None, 0), f"{name} has unexpected child")
    statuses[name] = {
        "identity": file_identity(path),
        "status": value.get("status"),
        "detail": value.get("detail"),
        "pid": value.get("pid"),
        "child_pid": value.get("child_pid"),
    }

terminal_outputs = [
    BASE / "reports/quality_bridge_result.json",
    BASE / "reports/runtime_compute_fairness/final_report.json",
    BASE / "reports/runtime_compute_claim_guard_v1/runtime_compute_claim_guard.json",
    BASE / "reports/terminal_system_claim_guard_v1/terminal_system_claim_guard.json",
    BASE / "reports/quality_bridge_comparison_v1/quality_bridge_comparison.json",
    BASE / "reports/terminal_completion_audit_v1/terminal_completion_audit.json",
    BASE / "reports/followup_experiment_decision_exposure_aware_v2.json",
]
require(all(not path.exists() for path in terminal_outputs), "terminal output appeared during audit")

filesystem = os.statvfs("/root/autodl-tmp")
report = {
    "schema_version": 1,
    "role": "generation_terminal_guard_consumer_integrity_coverage_audit",
    "status": "pass",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "generation_advantage_proven": False,
    "scientific_state": {
        "matched_terminal_evidence_complete": False,
        "generation_advantage_proven": False,
        "reason": "dense training and terminal evaluations/claim guards remain incomplete",
    },
    "scope": {
        "cpu_only_evidence_audit": True,
        "diagnostic_non_authorizing": True,
        "training_launch_allowed": False,
        "sampling_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
        "inference_export_authorization_allowed": False,
        "process_signals_allowed": False,
        "checkpoint_payload_loading_allowed": False,
        "gpu_execution_allowed": False,
        "upstream_decisions_modified": False,
    },
    "training_chain": {
        "git": training_git,
        "fixed_sources": fixed_source_evidence,
        "controller": controller,
        "watchdog": watchdog,
        "trainer": trainer,
        "pair_monitor_process": pair_monitor_process,
        "pair_monitor": {
            "identity": file_identity(pair_path),
            "status": pair["status"],
            "stage": pair["stage"],
            "issues": pair["issues"],
            "unrelated_gpu_processes": pair["gpu_contention"]["current"]["unrelated_gpu_processes"],
        },
        "metrics": {
            "cofitok": {"step": cofitok_metric["step"], "samples_seen": cofitok_metric["samples_seen"], "finite": True},
            "dense_identity": {"step": dense_metric["step"], "samples_seen": dense_metric["samples_seen"], "finite": True},
        },
        "gpu_compute_processes": gpu_processes,
        "sole_gpu_owner_is_exact_dense_trainer": True,
        "disk_free_bytes": filesystem.f_bavail * filesystem.f_frsize,
    },
    "checkpoint_integrity": {
        "physical_waiters": checkpoint_waiters,
        "dense_replay_waiter": {
            "status_report": file_identity(replay_status_path),
            "status": replay_status["status"],
            "detail": replay_status["detail"],
            "process": replay_process,
        },
    },
    "terminal_guard_coverage": {
        "producer_and_consumers": [
            "terminal_guard_producer",
            "terminal_completion_consumer",
            "factorization_consumer",
            "conditioning_consumer",
            "comparison_consumer",
            "random_token_wrapper_and_evaluator",
        ],
        "checkouts": checkouts,
        "launcher_sources": launcher_evidence,
        "live_processes": processes,
        "validation_sources": source_evidence,
        "all_live_waiters_detached_under_pid1": True,
        "all_live_waiters_nice10_ionice_idle": True,
        "no_terminal_guard_consumer_owns_gpu": True,
        "classifier_policy_true_required_by_all_downstream_routes": True,
        "classifier_evidence_verified_required_by_all_downstream_routes": True,
        "completion_audit_rehashes_classifier_before_and_after": True,
        "random_token_preflight_and_evaluator_both_revalidate_integrity": True,
    },
    "downstream_statuses": statuses,
    "terminal_outputs": {
        "all_absent": True,
        "paths": [path.as_posix() for path in terminal_outputs],
    },
    "decision": "continue_existing_serial_chain_without_additional_launch",
    "limitations": [
        "This audit proves only live source/process coverage of classifier-integrity fail-closed conditions.",
        "It does not establish generation quality, a CoFiTok advantage, route eligibility, or authorization for an experiment.",
        "Dense 90K/95K/100K physical audits and replay remain pending until their checkpoints exist.",
    ],
}

REPORT_DIR.mkdir(parents=True, exist_ok=False)
payload = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
descriptor = os.open(REPORT_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
try:
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
except Exception:
    REPORT_PATH.unlink(missing_ok=True)
    raise
directory_descriptor = os.open(REPORT_DIR, os.O_RDONLY)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)

identity = file_identity(REPORT_PATH)
require(read_json(REPORT_PATH) == report, "written report does not replay")
print(
    json.dumps(
        {
            "status": "pass",
            "path": identity["path"],
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "dense_step": dense_metric["step"],
            "dense_samples_seen": dense_metric["samples_seen"],
            "gpu_processes": gpu_processes,
            "generation_advantage_proven": False,
        },
        sort_keys=True,
    )
)
