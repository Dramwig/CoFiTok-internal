from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
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
AUDIT = REPORT_ROOT / "prelaunch_contract_audit.json"
REVISION = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
TREE = "6cef27723196fd363379bca2e7b85b1678ebd777"
BRANCH = "scale/generation-stability-quality-bridge-100k"
FORMAL_REVISION = "1ebcc15210e63a776a2ba448481cbd8bb94a4066"
FORMAL_BRANCH = "scale/generative-system"
FORMAL_PORCELAIN_COUNT = 87
FORMAL_PORCELAIN_SHA256 = (
    "a7e1a2daf19f77e5d92efaa09b2768f36f5df2e989475480f7d22e54b46e9497"
)


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def run_bytes(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.check_output(args, cwd=cwd)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    observed = {
        "path": path.as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload),
    }
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
        raise RuntimeError(f"identity mismatch: {path}: {observed}")
    return observed


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(recursive_values(child, key))
    return found


files = {
    "bundle": identity(
        Path("/tmp/cofitok-quality-bridge-cf0e5fa.bundle"),
        40130783,
        "3392e2a66c48237b8bfa0bde519245d4d17d055bb21b4e86321537fe5c3a871a",
    ),
    "standing_authorization": identity(
        BASE / "standing_authorization.json",
        865,
        "5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df",
    ),
    "execution_approval": identity(
        BASE / "execution_approval.json",
        1754,
        "e9da52a4e7ff1b4700b70aadaa8703ee40fb1a9862e847dcfb6e75933d295a4b",
    ),
    "preparation": identity(
        REPORT_ROOT / "preparation.json",
        23398,
        "7398d9a6f096ea9c178295c9016bb56fd38e28dff30f26662ae4225aded208ea",
    ),
    "config_validation": identity(
        REPORT_ROOT / "config_validation.json",
        14827,
        "cce5afbccac509903fceb93cf5bb3c7bcfa53a7637430545f6d0ae6b317b0fe2",
    ),
    "source_gate_validation": identity(
        REPORT_ROOT / "source_gate_validation.json",
        1932,
        "915dc497ee921d9768271ce60ac7cb8d1186c3e6bf9b6ca70d07ec58bf493918",
    ),
    "storage_preparation": identity(
        REPORT_ROOT / "storage_capacity_preparation.json",
        1089,
        "85fe1197797811bab6d1cb9e09cca54517e4596e4c9636540425e6c9150b11f6",
    ),
    "confirmation": identity(
        Path(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "stability_scaling_50k_sampling_confirmation_10k_v1/confirmation_report.json"
        ),
        13003,
        "bf25ae5efa0c4e478568d9ee27f9cdc09aec02a7cac9313993dd2420f75e4032",
    ),
    "frozen_gate": identity(
        Path(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "stability_scaling_50k_ema_teacher/reports/promotion_gate.json"
        ),
        31870,
        "2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90",
    ),
    "cofitok_config": identity(
        PROJECT
        / "configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json",
        3019,
        "40b5815c1aa92c1a22c585ddbe57d5232e57a031ece238bd27d1a0a1422fa14e",
    ),
    "dense_config": identity(
        PROJECT
        / "configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json",
        2459,
        "f8f1c1967468bef4db1e9a00e8f80c3da5af22e417d8ab4864afcb4fb68f884c",
    ),
    "execution_runbook": identity(
        PROJECT
        / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh",
        27943,
        "f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73",
    ),
    "milestone_runbook": identity(
        PROJECT / "artifacts/runbooks/generation_full_milestone_eval.sh",
        1889,
        "cf0f53c5c6bcfd7ca861e5f583b2d503213d633275b5751c7ba5964bd3a0cb56",
    ),
    "trainer": identity(
        PROJECT / "scripts/train_generation.py",
        40169,
        "6377b3cb6c4a8bdd13d6572e19ddd394ded716537bfd09367ae2efdce9ed5131",
    ),
    "sampling_preflight": identity(
        PROJECT / "scripts/preflight_generation_sampling.py",
        12893,
        "c82ad8432334c8cbb42c2eea889af1d90de62a813f8eea7cc45144401572a614",
    ),
    "sampler": identity(
        PROJECT / "scripts/generate_samples.py",
        16054,
        "3beddb8203af4c7c5cd070fb6780abba0c361a26c2fbe8364d35625f091c3238",
    ),
    "contract_test_status": identity(
        BASE / "prelaunch_contract_tests_cpu_status.json",
        1206,
        "86f1bf711112c00c27bcc2331d756a07f41b2788ba5c93c8068948d3a3ab24d9",
    ),
    "contract_test_log": identity(
        BASE / "prelaunch_contract_tests_cpu.log",
        160,
        "ea00a3bc20061f3e1451afe7a511de601c578a7c8543a3ac101e0884a38051e5",
    ),
}

if run("git", "rev-parse", "HEAD", cwd=PROJECT) != REVISION:
    raise RuntimeError("execution checkout revision changed")
if run("git", "rev-parse", "HEAD^{tree}", cwd=PROJECT) != TREE:
    raise RuntimeError("execution checkout tree changed")
if run("git", "branch", "--show-current", cwd=PROJECT) != BRANCH:
    raise RuntimeError("execution checkout branch changed")
if run_bytes("git", "status", "--porcelain=v1", cwd=PROJECT):
    raise RuntimeError("execution checkout became dirty")

formal_porcelain = run_bytes("git", "status", "--porcelain=v1", cwd=FORMAL)
if run("git", "rev-parse", "HEAD", cwd=FORMAL) != FORMAL_REVISION:
    raise RuntimeError("formal checkout revision changed")
if run("git", "branch", "--show-current", cwd=FORMAL) != FORMAL_BRANCH:
    raise RuntimeError("formal checkout branch changed")
if len(formal_porcelain.splitlines()) != FORMAL_PORCELAIN_COUNT:
    raise RuntimeError("formal checkout porcelain count changed")
if sha256(formal_porcelain) != FORMAL_PORCELAIN_SHA256:
    raise RuntimeError("formal checkout porcelain hash changed")

standing = load_json(Path(files["standing_authorization"]["path"]))
approval = load_json(Path(files["execution_approval"]["path"]))
preparation = load_json(Path(files["preparation"]["path"]))
config_validation = load_json(Path(files["config_validation"]["path"]))
source_gate_validation = load_json(Path(files["source_gate_validation"]["path"]))
storage = load_json(Path(files["storage_preparation"]["path"]))
confirmation = load_json(Path(files["confirmation"]["path"]))
frozen_gate = load_json(Path(files["frozen_gate"]["path"]))
test_status = load_json(Path(files["contract_test_status"]["path"]))
test_log = Path(files["contract_test_log"]["path"]).read_text(encoding="utf-8")

if standing.get("status") != "active":
    raise RuntimeError("standing authorization is not active")
if standing.get("instruction", {}).get("exact_text") != "之后不要我授权你直接运行需要的实验":
    raise RuntimeError("standing authorization instruction changed")
if approval.get("status") != "approved" or approval.get("scope") != "stability_quality_bridge_100k_execution_only":
    raise RuntimeError("quality bridge execution approval changed")
approval_boundary = approval.get("authorization_boundary", {})
if approval_boundary != {
    "quality_bridge_execution_allowed": True,
    "scope_limited_to_quality_bridge": True,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "scaling_authorization_created": False,
    "report_is_promotion_gate": False,
}:
    raise RuntimeError("quality bridge authorization boundary changed")

if preparation.get("status") != "prepared":
    raise RuntimeError("preparation is not prepared")
selection = preparation.get("selection", {})
if (
    selection.get("dataset") != "imagenet_256"
    or selection.get("steps") != 100000
    or selection.get("milestone_steps") != [50000, 100000]
    or selection.get("effective_batch_size") != 64
    or selection.get("images_seen_per_method") != 6400000
    or selection.get("capacity_change_allowed") is not False
    or selection.get("qualified_model_and_loss_recipe_preserved") is not True
):
    raise RuntimeError("quality bridge selection changed")
evaluation = preparation.get("evaluation_contract", {})
for name in ("milestones", "terminal"):
    protocol = evaluation.get(name, {})
    if protocol.get("guidance_scale") != 1.5 or protocol.get("guidance_rescale") != 0.0:
        raise RuntimeError(f"quality bridge {name} protocol changed")

if config_validation.get("status") != "pass":
    raise RuntimeError("config validation is not pass")
pair = preparation["matched_training_contract"]["config_validation"]
if pair.get("status") != "pass" or pair.get("mismatches") != []:
    raise RuntimeError("matched pair contract failed")
if pair.get("relative_parameter_gap", 1.0) > 0.02:
    raise RuntimeError("matched parameter gap exceeds two percent")
if pair.get("training_recipe", {}).get("valid") is not True:
    raise RuntimeError("training recipe is not valid")

if (
    source_gate_validation.get("stage") != "scaling"
    or source_gate_validation.get("sources", {}).get("status") != "verified"
    or source_gate_validation.get("sources", {}).get("stage") != "scaling"
    or source_gate_validation.get("sources", {}).get("source_profile")
    != "stability_scaling"
):
    raise RuntimeError("source gate evidence is not verified")
if frozen_gate.get("status") != "fail" or frozen_gate.get("decision") != "hold":
    raise RuntimeError("frozen gate identity or decision changed")
frozen_rescales = recursive_values(frozen_gate, "guidance_rescale")
frozen_scales = recursive_values(frozen_gate, "guidance_scale")
if not frozen_rescales or set(frozen_rescales) != {0}:
    raise RuntimeError("frozen gate does not bind guidance rescale zero")
if not frozen_scales or set(frozen_scales) != {1.5}:
    raise RuntimeError("frozen gate does not bind guidance scale 1.5")
if (
    confirmation.get("status") != "hold"
    or confirmation.get("decision") != "candidate_quality_not_confirmed"
    or confirmation.get("next_boundary", {}).get("candidate_protocol_formalized") is not False
    or confirmation.get("claim_boundary", {}).get("replaces_frozen_promotion_gate") is not False
):
    raise RuntimeError("candidate sampling protocol boundary changed")

preflight_text = Path(files["sampling_preflight"]["path"]).read_text(encoding="utf-8")
sampler_text = Path(files["sampler"]["path"]).read_text(encoding="utf-8")
milestone_text = Path(files["milestone_runbook"]["path"]).read_text(encoding="utf-8")
execution_text = Path(files["execution_runbook"]["path"]).read_text(encoding="utf-8")
trainer_text = Path(files["trainer"]["path"]).read_text(encoding="utf-8")
if 'parser.add_argument("--guidance-rescale", type=float, default=0.0)' not in preflight_text:
    raise RuntimeError("sampling preflight rescale default changed")
if 'parser.add_argument("--guidance-rescale", type=float, default=0.0)' not in sampler_text:
    raise RuntimeError("sampler rescale default changed")
if "--guidance-rescale" in milestone_text:
    raise RuntimeError("milestone runbook unexpectedly overrides frozen rescale default")
if execution_text.count("--guidance-rescale 0.0") < 2:
    raise RuntimeError("terminal runbook does not explicitly bind frozen rescale")
if "loop_end = min(loop_end, start_step + args.stop_after_steps)" not in trainer_text:
    raise RuntimeError("stop-after-steps is no longer resume-relative")

if (
    test_status.get("status") != "passed"
    or test_status.get("exit_code") != 0
    or test_status.get("cuda_visible_devices") != "-1"
    or test_status.get("pytest_cache_disabled") is not True
    or test_status.get("git", {}).get("revision") != REVISION
    or test_status.get("git", {}).get("tree") != TREE
    or test_status.get("git", {}).get("tracked_dirty") is not False
):
    raise RuntimeError("prelaunch contract test status failed")
passed_markers = test_log.count(".")
if passed_markers != 142 or "[100%]" not in test_log:
    raise RuntimeError(f"unexpected contract test summary: {passed_markers}")

if storage.get("status") != "pass":
    raise RuntimeError("storage preparation did not pass")
required_free = int(storage["plan"]["required_free_bytes"])
current_free = shutil.disk_usage(OUTPUT_ROOT).free
if current_free < required_free:
    raise RuntimeError("current free space fell below quality bridge requirement")

launch_receipt = REPORT_ROOT / "launch_receipt.json"
cofitok_run = OUTPUT_ROOT / "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
dense_run = OUTPUT_ROOT / "dense_rollout_x0_u2_ema_teacher"
if launch_receipt.exists():
    raise RuntimeError("launch receipt appeared during prelaunch audit")
for run_dir in (cofitok_run, dense_run):
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"unreceipted training state appeared: {run_dir}")

gpu_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
gpu_rows = [row.strip() for row in gpu_query.stdout.splitlines() if row.strip()]
waiter_status_path = BASE / "idle_waiter_status.json"
waiter = load_json(waiter_status_path)
if waiter.get("status") != "waiting" or waiter.get("detail") != "waiting_for_gpu_idle":
    raise RuntimeError("idle waiter is not in the expected prelaunch state")

report = {
    "schema_version": 1,
    "role": "stability_full_data_quality_bridge_100k_prelaunch_contract_audit",
    "status": "pass_waiting_for_gpu_idle" if gpu_rows else "pass_idle_confirmation_pending",
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "execution_checkout": {
        "path": PROJECT.as_posix(),
        "revision": REVISION,
        "tree": TREE,
        "branch": BRANCH,
        "tracked_dirty": False,
    },
    "formal_checkout": {
        "path": FORMAL.as_posix(),
        "revision": FORMAL_REVISION,
        "branch": FORMAL_BRANCH,
        "porcelain_count": FORMAL_PORCELAIN_COUNT,
        "porcelain_sha256": FORMAL_PORCELAIN_SHA256,
    },
    "authorization": {
        "standing_authorization_active": True,
        "quality_bridge_execution_allowed": True,
        "scope": "stability_quality_bridge_100k_execution_only",
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "release_authorization_allowed": False,
        "report_is_promotion_gate": False,
    },
    "protocol_resolution": {
        "candidate_confirmation": {
            "status": "hold",
            "decision": "candidate_quality_not_confirmed",
            "guidance_scale": 1.5,
            "guidance_rescale": 1.0,
            "candidate_protocol_formalized": False,
            "replaces_frozen_promotion_gate": False,
        },
        "frozen_formal_protocol": {
            "gate_status": "fail",
            "gate_decision": "hold",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
        },
        "quality_bridge_protocol": {
            "milestone": {"sample_steps": 50, "samples": 2048, "guidance_scale": 1.5, "guidance_rescale": 0.0},
            "terminal": {"sample_steps": 100, "samples": 10000, "guidance_scale": 1.5, "guidance_rescale": 0.0},
            "decision": "preserve_frozen_formal_protocol",
            "reason": "The rescale-1.0 candidate was not formalized and does not replace the frozen gate.",
        },
    },
    "matched_training": {
        "dataset": selection["dataset"],
        "steps_per_method": selection["steps"],
        "milestone_steps": selection["milestone_steps"],
        "effective_batch_size": selection["effective_batch_size"],
        "images_seen_per_method": selection["images_seen_per_method"],
        "equivalent_epochs": selection["equivalent_epochs"],
        "cofitok_parameters": pair["cofitok"]["parameter_count"],
        "dense_parameters": pair["dense"]["parameter_count"],
        "relative_parameter_gap": pair["relative_parameter_gap"],
        "pair_contract_valid": True,
        "capacity_change_allowed": False,
    },
    "recovery_contract": {
        "stop_after_steps_resume_relative": True,
        "exact_resume_config_equality_required": True,
        "metrics_reconciliation_tested": True,
        "protected_checkpoint_retention_tested": True,
        "watchdog_and_monitor_tested": True,
        "pytest_exit_code": 0,
        "passed_test_count": passed_markers,
        "test_files": test_status["tests"],
    },
    "storage": {
        "status": "pass",
        "required_free_bytes": required_free,
        "preparation_free_bytes": storage["filesystem"]["free_bytes"],
        "current_free_bytes": current_free,
        "current_headroom_bytes": current_free - required_free,
    },
    "runtime_state": {
        "idle_waiter_status": waiter["status"],
        "idle_waiter_detail": waiter["detail"],
        "idle_waiter_pid": int((BASE / "idle_waiter.pid").read_text().strip()),
        "required_consecutive_idle_polls": waiter["required_idle_polls"],
        "observed_gpu_compute_rows": gpu_rows,
        "launch_receipt_present": False,
        "training_state_present": False,
    },
    "evidence": files,
    "issues": [],
}

REPORT_ROOT.mkdir(parents=True, exist_ok=True)
temporary = AUDIT.with_name(f".{AUDIT.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(AUDIT)
print(AUDIT.as_posix())
