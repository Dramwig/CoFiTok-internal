"""Run the explicitly authorized, bounded 100K to 110K continuation.

The controller is intentionally single-purpose: it consumes an already created
source-bound execution authorization, resumes CoFiTok and dense_identity in
sequence, evaluates both outputs with one matched protocol, and leaves the
scientific result at ``hold``.  It never creates authorization and never sends
signals to unrelated processes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - controller is POSIX-only
    fcntl = None  # type: ignore[assignment]

from cofitok.generation.exposure_capacity_authorization import (
    SOURCE_REVISION,
    TARGET_STEP,
    checkout_identity,
    identity,
    read_object,
    validate_authorization_contract,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256
from cofitok.training.checkpointing import resolve_latest_checkpoint, verify_training_checkpoint
try:
    from scripts.exposure_capacity_result_cli import layout as result_layout
except ModuleNotFoundError:  # pragma: no cover - direct script invocation fallback
    from exposure_capacity_result_cli import layout as result_layout


ROLE = "generation_exposure_capacity_continuation_controller"
SOURCE_STEP = 100_000
EFFECTIVE_BATCH_SIZE = 64
SAMPLE_COUNT = 10_000
SAMPLE_STEPS = 100
SAMPLE_SEED = 2027
ROLLOUT_SEED = 2029
COFITOK_PREFIX = 8
DENSE_PREFIX = 1
MIN_FREE_BYTES = 120 * 1024**3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the source-bound 100K to 110K exposure continuation."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--stage-authorization", type=Path, required=True)
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val"),
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/CoFiTok/checkpoints/evaluators/torchvision/"
            "resnet50-11ad3fa6.pth"
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/checkpoints/eval_cache/torch_fidelity"),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Recover an explicitly interrupted continuation output root after "
            "revalidating its controller status and checkpoint state."
        ),
    )
    return parser.parse_args(argv)


def _read(path: Path, name: str) -> dict[str, Any]:
    return read_object(reject_symlink_chain(path, name=name), name=name)


def _assert_sha(path: Path, expected: str, name: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA256 changed: expected {expected}, found {actual}")


def _gpu_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: list[dict[str, Any]] = []
    for line in inventory_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError("nvidia-smi GPU inventory is malformed")
        inventory.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    compute_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute: list[dict[str, Any]] = []
    for line in compute_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError("nvidia-smi compute inventory is malformed")
        compute.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return inventory, compute


def _conflicting_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    current = os.getpid()
    markers = (
        "run_generation_exposure_capacity_continuation.py",
        "train_generation.py",
        "generate_samples.py",
        "evaluate_generation_metrics.py",
        "evaluate_generation_class_fidelity.py",
        "evaluate_generation_checkpoint.py",
        "evaluate_generation_rollout_stability.py",
    )
    root_text = output_root.as_posix()
    conflicts: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=2)
        try:
            pid = int(fields[0])
        except (IndexError, ValueError):
            continue
        if pid == current:
            continue
        command = fields[-1]
        if any(marker in command for marker in markers) and (
            root_text in command or "CoFiTok" in command
        ):
            conflicts.append({"pid": pid, "command": stripped})
    return conflicts


def _assert_idle_runtime(output_root: Path) -> None:
    inventory, compute = _gpu_inventory()
    if len(inventory) != 1:
        raise RuntimeError("continuation requires exactly one target GPU")
    if compute:
        raise RuntimeError("continuation refuses to start while a GPU process is active")
    if any(
        row["memory_used_mib"] > 16 or row["utilization_percent"] > 5
        for row in inventory
    ):
        raise RuntimeError("continuation target GPU is not idle")
    conflicts = _conflicting_processes(output_root)
    if conflicts:
        raise RuntimeError("matching CoFiTok process is already active")


def _write_status(path: Path, *, status: str, stage: str, detail: str, auth: Mapping[str, Any] | None, exit_code: int | None = None) -> None:
    payload = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "stage": stage,
        "detail": detail,
        "exit_code": exit_code,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "hostname": socket.gethostname(),
        "updated_at_unix": time.time(),
        "authorization": dict(auth or {}),
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "release_allowed": False,
        "process_signals_allowed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _execution_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise RuntimeError("bounded continuation requires POSIX flock")
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another bounded continuation controller holds the lock") from exc
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _claimed_output_root(
    output_root: Path,
    lock_path: Path,
    *,
    resume: bool = False,
) -> Iterator[None]:
    """Claim a new root or reopen a verified partial root under the lock."""

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with _execution_lock(lock_path):
        if resume:
            if not output_root.exists():
                raise FileNotFoundError(
                    f"cannot recover a missing continuation output root: {output_root}"
                )
            reject_symlink_chain(output_root, name="continuation output root")
            if not output_root.is_dir():
                raise ValueError(
                    f"continuation output root is not a directory: {output_root}"
                )
        else:
            if output_root.exists():
                raise FileExistsError(
                    f"refusing to reuse an existing continuation output root: {output_root}"
                )
            output_root.mkdir(parents=True, exist_ok=False)
        yield


def _validate_resumable_output_root(
    output_root: Path,
    status_path: Path,
    authorization_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only an explicitly incomplete root created by this authorization."""

    resolved_root = output_root.resolve()
    resolved_status = reject_symlink_chain(
        status_path,
        name="continuation controller status",
    ).resolve()
    if resolved_status.parent != resolved_root:
        raise ValueError(
            "continuation recovery requires controller_status.json inside the output root"
        )
    if not resolved_status.is_file():
        raise FileNotFoundError(
            f"continuation recovery status is missing: {resolved_status}"
        )
    status = _read(resolved_status, "continuation controller status")
    if status.get("role") != ROLE:
        raise ValueError("continuation recovery status belongs to another controller")
    if status.get("status") not in {"running", "failed"}:
        raise ValueError(
            "continuation recovery requires a non-terminal running or failed status"
        )
    if status.get("authorization") != dict(authorization_identity):
        raise ValueError("continuation recovery authorization identity differs")
    if status.get("terminal_status") != "hold":
        raise ValueError("continuation recovery status weakens the terminal hold")
    for field in (
        "generation_advantage_proven",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "release_allowed",
        "process_signals_allowed",
    ):
        if status.get(field) is not False:
            raise ValueError(f"continuation recovery status permits {field}")
    result_path = resolved_root / "exposure_capacity_result.json"
    if result_path.exists():
        raise ValueError(
            "continuation recovery refuses a root that already contains a result"
        )
    allowed_names = {
        "controller.log",
        "controller_status.json",
        "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
        "dense_rollout_x0_u2_ema_teacher",
        "evaluations",
    }
    unexpected = sorted(
        child.name for child in resolved_root.iterdir() if child.name not in allowed_names
    )
    if unexpected:
        raise ValueError(
            "continuation recovery output root contains unexpected entries: "
            + ", ".join(unexpected)
        )
    return status


def _training_resume_action(
    run_dir: Path,
    *,
    source_checkpoint: Path,
    config_path: Path,
    execution_checkout: Mapping[str, Any],
    resume_requested: bool,
) -> str:
    """Return ``source``, ``auto``, or ``skip`` after checking a partial run."""

    if run_dir.is_symlink():
        raise ValueError("continuation training output must not be a symlink")
    if not run_dir.exists():
        return "source"
    run = reject_symlink_chain(run_dir, name="continuation training output").resolve()
    if not run.is_dir():
        raise ValueError(f"continuation training output is not a directory: {run}")
    latest = run / "latest.json"
    if not latest.is_file() or latest.is_symlink():
        raise RuntimeError(
            "existing continuation training output has no verified latest.json; "
            "refusing implicit reconstruction"
        )
    checkpoint = resolve_latest_checkpoint(run).resolve()
    integrity = verify_training_checkpoint(checkpoint)
    step = int(integrity.get("step", -1))
    if not SOURCE_STEP <= step <= TARGET_STEP:
        raise ValueError(
            f"continuation checkpoint step {step} is outside [{SOURCE_STEP}, {TARGET_STEP}]"
        )
    if step == SOURCE_STEP:
        if not resume_requested:
            raise RuntimeError(
                "source-step continuation output exists; rerun the controller with --resume"
            )
        return "source"
    if step == TARGET_STEP:
        report_path = run / "training_report.json"
        if report_path.is_file() and not report_path.is_symlink():
            report = _read(report_path, "continuation training report")
            report_git = report.get("git")
            if (
                report.get("training_complete") is True
                and report.get("completed_steps") == TARGET_STEP
                and report.get("target_steps") == TARGET_STEP
                and report.get("config_path") == config_path.resolve().as_posix()
                and isinstance(report_git, Mapping)
                and report_git.get("revision") == execution_checkout.get("revision")
                and report_git.get("branch") == execution_checkout.get("branch")
                and report_git.get("dirty") is False
            ):
                return "skip"
            if report.get("training_complete") is True:
                raise ValueError(
                    "completed continuation training report does not match the current execution identity"
                )
    if not resume_requested:
        raise RuntimeError(
            "partial continuation output exists; rerun the controller with --resume"
        )
    return "auto"


def _layout(root: Path, method: str) -> dict[str, Path]:
    return result_layout(root, method)


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_handle: Any,
) -> None:
    printable = shlex.join(str(value) for value in command)
    log_handle.write(f"\n$ {printable}\n")
    log_handle.flush()
    subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=dict(env),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _common_python_env(project_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    source = [project_root.as_posix(), (project_root / "src").as_posix()]
    previous = env.get("PYTHONPATH")
    if previous:
        source.append(previous)
    env["PYTHONPATH"] = os.pathsep.join(source)
    return env


def _result_command(args: argparse.Namespace, root: Path) -> list[str | Path]:
    builder = args.project_root / "scripts" / "build_generation_exposure_capacity_result.py"
    command: list[str | Path] = [args.python, builder, "--result", root / "exposure_capacity_result.json"]
    command.extend(
        [
            "--authorization", args.authorization,
            "--expected-authorization-sha256", args.expected_authorization_sha256,
            "--gate", args.gate,
            "--expected-gate-sha256", args.expected_gate_sha256,
            "--preparation", args.preparation,
            "--expected-preparation-sha256", args.expected_preparation_sha256,
            "--standing-authorization", args.standing_authorization,
            "--expected-standing-authorization-sha256", args.expected_standing_authorization_sha256,
            "--stage-authorization", args.stage_authorization,
            "--expected-stage-authorization-sha256", args.expected_stage_authorization_sha256,
            "--source-project-root", args.source_project_root,
            "--execution-project-root", args.project_root,
            "--cofitok-config", args.cofitok_config,
            "--dense-config", args.dense_config,
            "--output-root", root,
            "--real-dir", args.real_dir,
            "--classifier-checkpoint", args.classifier_checkpoint,
            "--cache-root", args.cache_root,
        ]
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = reject_symlink_chain(args.project_root, name="execution project root").resolve()
    source_project_root = reject_symlink_chain(
        args.source_project_root, name="source project root"
    ).resolve()
    output_root = reject_symlink_chain(args.output_root, name="continuation output root").resolve()
    status_path = args.status_output or output_root / "controller_status.json"
    log_path = args.log_output or output_root / "controller.log"
    auth_id = identity(args.authorization)
    gate_id = identity(args.gate)
    preparation_id = identity(args.preparation)
    standing_id = identity(args.standing_authorization)
    stage_authorization_id = identity(args.stage_authorization)
    authorization = _read(args.authorization, "execution authorization")
    gate = _read(args.gate, "execution gate")
    preparation = _read(args.preparation, "preparation")
    standing = _read(args.standing_authorization, "standing authorization")
    stage_authorization = _read(args.stage_authorization, "stage authorization")
    _assert_sha(args.authorization, args.expected_authorization_sha256, "authorization")
    _assert_sha(args.gate, args.expected_gate_sha256, "execution gate")
    _assert_sha(args.preparation, args.expected_preparation_sha256, "preparation")
    _assert_sha(args.standing_authorization, args.expected_standing_authorization_sha256, "standing authorization")
    _assert_sha(args.stage_authorization, args.expected_stage_authorization_sha256, "stage authorization")
    source_checkout = checkout_identity(source_project_root)
    execution_checkout = checkout_identity(project_root)
    config_ids = {"cofitok": identity(args.cofitok_config), "dense_identity": identity(args.dense_config)}
    validated = validate_authorization_contract(
        authorization,
        gate=gate,
        preparation=preparation,
        preparation_identity=preparation_id,
        gate_identity=gate_id,
        standing_identity=standing_id,
        execution_checkout=execution_checkout,
        source_checkout=source_checkout,
        config_identities=config_ids,
    )
    if validated.get("stage_authorization_identity") != stage_authorization_id:
        raise ValueError("execution authorization binds another stage authorization")
    contract = validated["qualification_contract"]
    if contract["output_root"] != output_root.as_posix():
        raise ValueError("controller output root differs from authorized target")
    if validated["target"]["target_step"] != TARGET_STEP:
        raise ValueError("controller target horizon differs from authorization")
    if not args.python.is_file():
        raise FileNotFoundError(f"continuation Python executable is missing: {args.python}")

    authorized_lock = reject_symlink_chain(
        Path(str(contract["execution_lock"])), name="continuation execution lock"
    ).resolve()
    expected_lock = output_root.parent / f".{output_root.name}.exposure_execution.lock"
    if authorized_lock != expected_lock:
        raise ValueError("authorization execution lock is not the sibling controller lock")
    root_lock = authorized_lock
    auth_summary = {"path": auth_id["path"], "bytes": auth_id["bytes"], "sha256": auth_id["sha256"]}
    try:
        # Lock the sibling path before creating the output root, so a second
        # controller cannot race the output-root absence check.
        with _claimed_output_root(output_root, root_lock, resume=args.resume):
            if args.resume:
                _validate_resumable_output_root(output_root, status_path, auth_summary)
            _assert_idle_runtime(output_root)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                env = _common_python_env(project_root)
                _write_status(status_path, status="running", stage="preflight", detail="authorization and runtime preflight passed", auth=auth_summary)
                source_checkpoints = validated["source_checkpoints"]
                for method, config, prefix in (
                    ("cofitok", args.cofitok_config, COFITOK_PREFIX),
                    ("dense_identity", args.dense_config, DENSE_PREFIX),
                ):
                    layout = _layout(output_root, method)
                    source_checkpoint = Path(source_checkpoints[method]["checkpoint"]["path"])
                    training_action = _training_resume_action(
                        layout["run"],
                        source_checkpoint=source_checkpoint,
                        config_path=config,
                        execution_checkout=execution_checkout,
                        resume_requested=args.resume,
                    )
                    _assert_idle_runtime(output_root)
                    _assert_sha(args.authorization, args.expected_authorization_sha256, "authorization")
                    if training_action == "skip":
                        _write_status(
                            status_path,
                            status="running",
                            stage=f"training_{method}",
                            detail=f"reusing verified completed {method} continuation training",
                            auth=auth_summary,
                        )
                        continue
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"training_{method}",
                        detail=f"exact-resume training {method} to 110K",
                        auth=auth_summary,
                    )
                    command: list[str | Path] = [
                        args.python,
                        project_root / "scripts" / "train_generation.py",
                        "--config",
                        config,
                        "--output-dir",
                        layout["run"],
                    ]
                    if training_action == "source":
                        command.extend(
                            [
                                "--resume",
                                source_checkpoint,
                                "--resume-source-revision",
                                SOURCE_REVISION,
                            ]
                        )
                    else:
                        command.extend(["--resume", "auto"])
                    command.extend(["--resume-target-steps", str(TARGET_STEP)])
                    _run(command, cwd=project_root, env=env, log_handle=log_handle)

                for method, prefix in (("cofitok", COFITOK_PREFIX), ("dense_identity", DENSE_PREFIX)):
                    layout = _layout(output_root, method)
                    checkpoint = layout["run"] / f"checkpoint_step_{TARGET_STEP:08d}.pt"
                    _assert_idle_runtime(output_root)
                    _write_status(status_path, status="running", stage=f"sampling_{method}", detail=f"matched EMA DDIM-100 sampling for {method}", auth=auth_summary)
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "generate_samples.py",
                            "--checkpoint", checkpoint,
                            "--output-dir", layout["sampling"],
                            "--num-samples", str(SAMPLE_COUNT),
                            "--batch-size", "32",
                            "--sample-steps", str(SAMPLE_STEPS),
                            "--prefix-budgets", str(prefix),
                            "--guidance-scale", "1.5",
                            "--guidance-rescale", "0.0",
                            "--cfg-batch-mode", "batched",
                            "--eta", "0.0",
                            "--seed", str(SAMPLE_SEED),
                            "--start-index", "0",
                            "--weights", "ema",
                            "--precision", "bf16",
                            *( ["--resume"] if args.resume else [] ),
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    _write_status(status_path, status="running", stage=f"metrics_{method}", detail=f"FID/IS/precision/recall for {method}", auth=auth_summary)
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_metrics.py",
                            "--real-dir", args.real_dir,
                            "--generated-dir", layout["generated"],
                            "--sampling-report", layout["sampling"] / "sampling_report.json",
                            "--output-dir", layout["metrics"],
                            "--batch-size", "64",
                            "--prc-batch-size", str(SAMPLE_COUNT),
                            "--min-samples", str(SAMPLE_COUNT),
                            "--seed", str(SAMPLE_SEED),
                            "--cache-root", args.cache_root,
                            "--real-cache-name", "imagenet256_val_50k_torch_fidelity_v04",
                            *( ["--resume"] if args.resume else [] ),
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    _write_status(status_path, status="running", stage=f"class_fidelity_{method}", detail=f"class fidelity for {method}", auth=auth_summary)
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_class_fidelity.py",
                            "--generated-dir", layout["generated"],
                            "--sampling-report", layout["sampling"] / "sampling_report.json",
                            "--output-dir", layout["class_fidelity"],
                            "--classifier-checkpoint", args.classifier_checkpoint,
                            "--batch-size", "64",
                            "--num-workers", "8",
                            "--min-samples", str(SAMPLE_COUNT),
                            *( ["--resume"] if args.resume else [] ),
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    _write_status(status_path, status="running", stage=f"mechanism_{method}", detail=f"checkpoint mechanism evaluation for {method}", auth=auth_summary)
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_checkpoint.py",
                            "--checkpoint", checkpoint,
                            "--output-dir", layout["mechanism"],
                            "--num-images", "1024",
                            "--timestep", "500",
                            "--random-orders", "4" if method == "cofitok" else "0",
                            "--seed", str(SAMPLE_SEED),
                            "--weights", "ema",
                            "--precision", "bf16",
                            *( ["--resume"] if args.resume else [] ),
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    if args.resume and layout["rollout"].exists():
                        if (
                            not layout["rollout_report"].is_file()
                            or layout["rollout_report"].is_symlink()
                        ):
                            raise RuntimeError(
                                "incomplete rollout output cannot be recovered safely; "
                                "refusing to rerun without an explicit evaluator resume contract"
                            )
                        _write_status(
                            status_path,
                            status="running",
                            stage=f"rollout_{method}",
                            detail=f"reusing verified completed {method} rollout stability report",
                            auth=auth_summary,
                        )
                        continue
                    _write_status(status_path, status="running", stage=f"rollout_{method}", detail=f"DDIM-100 rollout stability for {method}", auth=auth_summary)
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_rollout_stability.py",
                            "--checkpoint", checkpoint,
                            "--output-dir", layout["rollout"],
                            "--num-images", "64",
                            "--batch-size", "4",
                            "--sample-steps", str(SAMPLE_STEPS),
                            "--seed", str(ROLLOUT_SEED),
                            "--weights", "ema",
                            "--precision", "bf16",
                            "--guidance-scale", "1.5",
                            "--guidance-rescale", "0.0",
                            "--teacher-guidance-scale", "1.0",
                            "--cfg-batch-mode", "batched",
                            "--clip-x0",
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )

                _write_status(status_path, status="running", stage="result", detail="building and validating bounded continuation result", auth=auth_summary)
                _run(_result_command(args, output_root), cwd=project_root, env=env, log_handle=log_handle)
                _run(
                    [
                        args.python,
                        project_root / "scripts" / "validate_generation_exposure_capacity_result.py",
                        "--result", output_root / "exposure_capacity_result.json",
                        "--authorization", args.authorization,
                        "--expected-authorization-sha256", args.expected_authorization_sha256,
                        "--gate", args.gate,
                        "--expected-gate-sha256", args.expected_gate_sha256,
                        "--preparation", args.preparation,
                        "--expected-preparation-sha256", args.expected_preparation_sha256,
                        "--standing-authorization", args.standing_authorization,
                        "--expected-standing-authorization-sha256", args.expected_standing_authorization_sha256,
                        "--source-project-root", source_project_root,
                        "--execution-project-root", project_root,
                        "--cofitok-config", args.cofitok_config,
                        "--dense-config", args.dense_config,
                        "--output-root", output_root,
                        "--real-dir", args.real_dir,
                        "--classifier-checkpoint", args.classifier_checkpoint,
                        "--cache-root", args.cache_root,
                    ],
                    cwd=project_root,
                    env=env,
                    log_handle=log_handle,
                )
                _write_status(status_path, status="completed", stage="complete", detail="bounded continuation completed; terminal scientific hold preserved", auth=auth_summary)
        return 0
    except BaseException as error:
        # A lock/preflight failure must not create a candidate output root as
        # a side effect.  An explicitly external status path may still record
        # that failure before the root exists.
        if output_root.exists() or args.status_output is not None:
            try:
                _write_status(status_path, status="failed", stage="failed", detail=f"{type(error).__name__}: {error}", auth=auth_summary, exit_code=1)
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
