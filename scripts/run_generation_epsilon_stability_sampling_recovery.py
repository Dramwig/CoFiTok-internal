from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cofitok.generation import (
    EPSILON_STABILITY_EXECUTION_ACTIONS,
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    build_epsilon_stability_case_observation,
    build_epsilon_stability_execution_authorization,
    build_epsilon_stability_observation_manifest,
    build_epsilon_stability_preparation,
    build_epsilon_stability_real_artifact_reference,
    build_epsilon_stability_sampling_design,
    materialize_epsilon_stability_case_protocol,
)
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLE = "generation_epsilon_stability_sampling_recovery_controller"
SCHEMA_VERSION = 1
METHODS = ("cofitok", "dense_identity")
TRACKED_RUNTIME_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "PYTORCH_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _read_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return payload


def _source(identity: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    expected = dict(identity)
    path = reject_symlink_chain(
        str(expected.get("path", "")), name=f"epsilon-stability {label}"
    )
    actual = gate_source_report_identity(path)
    if actual != expected:
        raise ValueError(f"epsilon-stability bound source changed: {label}")
    return {"identity": actual, "payload": _read_object(actual["path"])}


def _identity(path: str | Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=f"epsilon-stability {label}")
    actual = gate_source_report_identity(source)
    if actual["sha256"] != expected_sha256:
        raise ValueError(f"epsilon-stability {label} SHA256 differs")
    return actual


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def _full_status(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain"],
        text=True,
    ).strip()


def validate_control_git(
    project: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    observed = {**git_provenance(project), "tree": _tree(project)}
    expected = {
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if observed != expected or _full_status(project):
        raise ValueError("epsilon-stability controller requires an exact clean checkout")
    return observed


def parse_gpu_process_pids(output: str) -> list[int]:
    pids = []
    for line in output.splitlines():
        row = line.strip()
        if not row or row.lower().startswith("no running"):
            continue
        first = row.split(",", 1)[0].strip()
        if not first.isdigit():
            raise ValueError(f"unparseable nvidia-smi compute process row: {row}")
        pids.append(int(first))
    return sorted(set(pids))


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_process_pids(result.stdout)


def matching_process_pids(
    *,
    markers: Sequence[str],
    own_pid: int | None = None,
    proc_root: str | Path = "/proc",
) -> list[int]:
    own = os.getpid() if own_pid is None else own_pid
    matches = []
    root = Path(proc_root)
    if not root.is_dir():
        return matches
    for candidate in root.iterdir():
        if not candidate.name.isdigit() or int(candidate.name) == own:
            continue
        try:
            command = (candidate / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if all(marker in command for marker in markers):
            matches.append(int(candidate.name))
    return sorted(matches)


def duplicate_controller_pids(
    *,
    output_root: Path,
    own_pid: int | None = None,
    proc_root: str | Path = "/proc",
) -> list[int]:
    return matching_process_pids(
        markers=(
            "run_generation_epsilon_stability_sampling_recovery.py",
            output_root.resolve().as_posix(),
        ),
        own_pid=own_pid,
        proc_root=proc_root,
    )


def _status_payload(
    *,
    status: str,
    detail: str,
    args: argparse.Namespace,
    execution_identity: Mapping[str, Any] | None = None,
    stage: str | None = None,
    case_id: str | None = None,
    method: str | None = None,
    child_pid: int | None = None,
    completed_arms: int = 0,
    result_identity: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "child_pid": child_pid,
        "project": args.project.resolve().as_posix(),
        "output_root": args.output_root.resolve().as_posix(),
        "stage": stage,
        "case_id": case_id,
        "method": method,
        "completed_arms": completed_arms,
        "total_arms": 16,
        "execution_authorization": dict(execution_identity or {}),
        "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
        "result_authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
        "generation_advantage_proven": False,
    }
    if result_identity is not None:
        payload["result"] = dict(result_identity)
    if error is not None:
        payload["error"] = error
    return payload


def _write_status(args: argparse.Namespace, **kwargs: Any) -> None:
    write_json_report(args.status_output, _status_payload(args=args, **kwargs))


def _replay_preparation(
    preparation: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    source_bindings = preparation.get("source_bindings")
    methods = preparation.get("methods")
    if not isinstance(source_bindings, Mapping) or not isinstance(methods, Mapping):
        raise ValueError("epsilon-stability preparation source bindings are missing")
    design_source = _source(preparation["design_identity"], label="sampling design")
    if design_source["payload"] != dict(design):
        raise ValueError("epsilon-stability preparation design payload differs")
    method_sources: dict[str, Any] = {}
    for method in METHODS:
        row = methods[method]
        method_sources[method] = {
            "checkpoint": _identity(
                row["checkpoint"]["path"],
                row["checkpoint"]["sha256"],
                label=f"{method} checkpoint",
            ),
            "integrity_sidecar": _source(
                row["integrity_sidecar"], label=f"{method} integrity sidecar"
            ),
            "latest": _source(row["latest"], label=f"{method} latest"),
            "training_report": _source(
                row["training_report"], label=f"{method} training report"
            ),
        }
        if method_sources[method]["checkpoint"] != row["checkpoint"]:
            raise ValueError(f"epsilon-stability {method} checkpoint identity differs")
    replayed = build_epsilon_stability_preparation(
        design_source=design_source,
        decision_source=_source(
            source_bindings["post_reconciliation_decision"],
            label="post-reconciliation decision",
        ),
        decision_verification_source=_source(
            source_bindings["post_reconciliation_verification"],
            label="post-reconciliation verification",
        ),
        reconciliation_source=_source(
            source_bindings["cross_protocol_reconciliation"],
            label="cross-protocol reconciliation",
        ),
        quality_bridge_result_source=_source(
            source_bindings["quality_bridge_result"],
            label="quality bridge result",
        ),
        training_pair_report_source=_source(
            source_bindings["training_pair_report"],
            label="training pair report",
        ),
        method_sources=method_sources,
        dataset_identity=_identity(
            source_bindings["dataset"]["path"],
            source_bindings["dataset"]["sha256"],
            label="dataset manifest",
        ),
        real_set_source=_source(source_bindings["real_set"], label="real set"),
        runtime_binding_source=_source(
            source_bindings["runtime_environment"], label="runtime binding"
        ),
        evaluator_source=_source(
            source_bindings["evaluator"], label="evaluator manifest"
        ),
        classifier_identity=_identity(
            source_bindings["classifier"]["path"],
            source_bindings["classifier"]["sha256"],
            label="classifier weights",
        ),
        classifier_report_source=_source(
            source_bindings["classifier_report"], label="classifier report"
        ),
        repair_git=preparation["repair_git"],
        seed=int(preparation["random_stream"]["seed"]),
        random_stream_namespace=str(preparation["random_stream"]["namespace"]),
        output_root=str(preparation["output_root"]),
    )
    if replayed != dict(preparation):
        raise ValueError("epsilon-stability preparation does not physically replay")
    return replayed


def replay_execution_gate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    design_identity = _identity(
        args.design,
        args.expected_design_sha256,
        label="sampling design",
    )
    design = _read_object(design_identity["path"])
    if design != build_epsilon_stability_sampling_design():
        raise ValueError("epsilon-stability sampling design is not canonical")
    preparation_identity = _identity(
        args.preparation,
        args.expected_preparation_sha256,
        label="preparation",
    )
    preparation = _read_object(preparation_identity["path"])
    _replay_preparation(preparation, design=design)
    user_identity = _identity(
        args.user_authorization,
        args.expected_user_authorization_sha256,
        label="user authorization",
    )
    user_authorization = _read_object(user_identity["path"])
    expected = build_epsilon_stability_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_identity,
        design=design,
        user_authorization=user_authorization,
        user_authorization_identity=user_identity,
    )
    execution_identity = _identity(
        args.execution_authorization,
        args.expected_execution_authorization_sha256,
        label="execution authorization",
    )
    execution = _read_object(execution_identity["path"])
    if execution != expected:
        raise ValueError("epsilon-stability execution authorization does not replay")
    if execution["output_root"] != args.output_root.resolve().as_posix():
        raise ValueError("epsilon-stability controller output root differs from gate")
    if execution["git"] != {
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("epsilon-stability execution gate binds another checkout")
    if execution["source_bindings"]["real_set"]["path"] != str(
        args.real_set_contract.resolve()
    ):
        raise ValueError("epsilon-stability controller real-set source differs")
    if execution["real_set"]["root"] != args.real_dir.resolve().as_posix():
        raise ValueError("epsilon-stability controller real directory differs")
    if execution["source_bindings"]["classifier"]["path"] != str(
        args.classifier_checkpoint.resolve()
    ):
        raise ValueError("epsilon-stability controller classifier differs")
    return execution_identity, execution


def replay_evaluator_manifest(
    *,
    execution: Mapping[str, Any],
    control_git: Mapping[str, Any],
) -> dict[str, Any]:
    source = _source(
        execution["source_bindings"]["evaluator"],
        label="evaluator manifest",
    )
    manifest = source["payload"]
    rows = manifest.get("sources")
    if (
        manifest.get("schema")
        != "cofitok_matched_epsilon_stability_evaluator_manifest_v1"
        or manifest.get("status") != "pass"
        or manifest.get("git") != dict(control_git)
        or not isinstance(rows, Mapping)
        or not rows
    ):
        raise ValueError("epsilon-stability evaluator manifest contract differs")
    for relative, expected in rows.items():
        path = reject_symlink_chain(
            PROJECT_ROOT / str(relative),
            name=f"epsilon-stability evaluator source {relative}",
        )
        actual = gate_source_report_identity(path)
        if actual != dict(expected):
            raise ValueError(
                f"epsilon-stability evaluator source changed: {relative}"
            )
    return source["identity"]


def sampling_command(
    *,
    python: Path,
    project: Path,
    execution: Mapping[str, Any],
    design: Mapping[str, Any],
    case_id: str,
    method: str,
    sample_root: Path,
) -> list[str]:
    protocol = materialize_epsilon_stability_case_protocol(
        dict(design),
        case_id,
        method,
        seed=int(execution["random_stream"]["seed"]),
        start_index=int(execution["random_stream"]["start_index"]),
    )
    controls = next(
        row["sampling_controls"]
        for row in design["cases"]
        if row["case_id"] == case_id
    )
    command = [
        str(python),
        str(project / "scripts/generate_samples.py"),
        "--checkpoint",
        str(execution["methods"][method]["checkpoint"]["path"]),
        "--output-dir",
        str(sample_root),
        "--num-samples",
        str(protocol["num_samples"]),
        "--batch-size",
        str(protocol["batch_size"]),
        "--sample-steps",
        str(protocol["sample_steps"]),
        "--prefix-budgets",
        str(execution["methods"][method]["prefix_budget"]),
        "--guidance-scale",
        str(protocol["guidance_scale"]),
        "--guidance-rescale",
        str(protocol["guidance_rescale"]),
        "--cfg-batch-mode",
        str(protocol["cfg_batch_mode"]),
        "--eta",
        str(protocol["eta"]),
        "--x0-constraint",
        str(protocol["x0_constraint"]),
        "--dynamic-threshold-percentile",
        str(protocol["dynamic_threshold_percentile"]),
        "--seed",
        str(protocol["seed"]),
        "--start-index",
        str(protocol["start_index"]),
        "--weights",
        "ema",
        "--precision",
        str(protocol["precision"]),
        "--resume",
    ]
    if controls["requested_start_timestep"] is not None:
        command.extend(
            ["--start-timestep", str(controls["requested_start_timestep"])]
        )
    if controls["scale_initial_noise_by_sigma"]:
        command.append("--scale-initial-noise-by-sigma")
    if controls["recompute_epsilon_after_x0_constraint"]:
        command.append("--recompute-epsilon-after-x0-constraint")
    return command


def evaluation_commands(
    *,
    python: Path,
    project: Path,
    args: argparse.Namespace,
    execution: Mapping[str, Any],
    case_id: str,
    method: str,
    arm_root: Path,
) -> dict[str, list[str]]:
    budget = int(execution["methods"][method]["prefix_budget"])
    sample_root = arm_root / "samples_1000_ddim100"
    generated = sample_root / f"prefix_{budget}"
    sampling_report = sample_root / "sampling_report.json"
    return {
        "metrics": [
            str(python),
            str(project / "scripts/evaluate_generation_metrics.py"),
            "--real-dir",
            str(args.real_dir),
            "--generated-dir",
            str(generated),
            "--sampling-report",
            str(sampling_report),
            "--output-dir",
            str(arm_root / "metrics"),
            "--cache-root",
            str(args.eval_cache),
            "--min-samples",
            "1000",
            "--skip-prc",
            "--resume",
        ],
        "class_fidelity": [
            str(python),
            str(project / "scripts/evaluate_generation_class_fidelity.py"),
            "--generated-dir",
            str(generated),
            "--sampling-report",
            str(sampling_report),
            "--output-dir",
            str(arm_root / "class_fidelity"),
            "--classifier-checkpoint",
            str(args.classifier_checkpoint),
            "--min-samples",
            "1000",
            "--resume",
        ],
        "artifact": [
            str(python),
            str(project / "scripts/evaluate_generation_epsilon_stability_artifacts.py"),
            "--image-dir",
            str(generated),
            "--source-kind",
            "generated",
            "--sampling-report",
            str(sampling_report),
            "--expected-sampling-report-sha256",
            file_sha256(sampling_report),
            "--output",
            str(arm_root / "artifact_report.json"),
            "--resume",
        ],
    }


def _run_child(
    command: Sequence[str],
    *,
    args: argparse.Namespace,
    detail: str,
    stage: str,
    execution_identity: Mapping[str, Any],
    case_id: str | None,
    method: str | None,
    completed_arms: int,
) -> None:
    process = subprocess.Popen(list(command), cwd=args.project)
    _write_status(
        args,
        status="running",
        detail=detail,
        stage=stage,
        case_id=case_id,
        method=method,
        child_pid=process.pid,
        completed_arms=completed_arms,
        execution_identity=execution_identity,
    )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _write_or_replay(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if _read_object(path) != payload:
            raise ValueError(f"existing epsilon-stability report differs: {path}")
        return
    write_json_report(path, payload)


def _prepare_real_reference(
    *,
    args: argparse.Namespace,
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    completed_arms: int,
) -> dict[str, Any]:
    real_root = args.output_root / "real_artifact_reference"
    images = real_root / "images"
    subset_manifest = real_root / "subset_manifest.json"
    _run_child(
        [
            str(args.python),
            str(
                args.project
                / "scripts/prepare_generation_epsilon_stability_real_artifact_subset.py"
            ),
            "--real-dir",
            str(args.real_dir),
            "--real-set-contract",
            str(args.real_set_contract),
            "--expected-real-set-contract-sha256",
            execution["source_bindings"]["real_set"]["sha256"],
            "--output-dir",
            str(images),
            "--output",
            str(subset_manifest),
            "--resume",
        ],
        args=args,
        detail="preparing_physical_real_artifact_reference",
        stage="real_reference_subset",
        execution_identity=execution_identity,
        case_id=None,
        method=None,
        completed_arms=completed_arms,
    )
    artifact_report = real_root / "artifact_report.json"
    _run_child(
        [
            str(args.python),
            str(args.project / "scripts/evaluate_generation_epsilon_stability_artifacts.py"),
            "--image-dir",
            str(images),
            "--source-kind",
            "real_reference",
            "--real-set-contract",
            str(args.real_set_contract),
            "--expected-real-set-contract-sha256",
            execution["source_bindings"]["real_set"]["sha256"],
            "--subset-manifest",
            str(subset_manifest),
            "--expected-subset-manifest-sha256",
            file_sha256(subset_manifest),
            "--output",
            str(artifact_report),
            "--resume",
        ],
        args=args,
        detail="evaluating_real_artifact_reference",
        stage="real_reference_artifact",
        execution_identity=execution_identity,
        case_id=None,
        method=None,
        completed_arms=completed_arms,
    )
    artifact_source = {
        "identity": gate_source_report_identity(artifact_report),
        "payload": _read_object(artifact_report),
    }
    reference = build_epsilon_stability_real_artifact_reference(
        design=design,
        design_identity=design_identity,
        execution_authorization=execution,
        execution_authorization_identity=execution_identity,
        artifact_report_source=artifact_source,
    )
    reference_path = real_root / "real_artifact_reference.json"
    _write_or_replay(reference_path, reference)
    return {
        "path": reference_path,
        "identity": gate_source_report_identity(reference_path),
        "payload": reference,
    }


def _build_observation(
    *,
    args: argparse.Namespace,
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    case_id: str,
    method: str,
    arm_root: Path,
) -> tuple[Path, dict[str, Any]]:
    sample_root = arm_root / "samples_1000_ddim100"
    reports = {
        "sampling_report": sample_root / "sampling_report.json",
        "metrics_report": arm_root / "metrics/generation_metrics_report.json",
        "class_fidelity_report": arm_root
        / "class_fidelity/class_fidelity_report.json",
        "artifact_report": arm_root / "artifact_report.json",
    }
    report_sources = {
        name: {
            "identity": gate_source_report_identity(path),
            "payload": _read_object(path),
        }
        for name, path in reports.items()
    }
    observation = build_epsilon_stability_case_observation(
        design=design,
        design_identity=design_identity,
        execution_authorization=execution,
        execution_authorization_identity=execution_identity,
        case_id=case_id,
        method=method,
        report_sources=report_sources,
    )
    path = args.output_root / "reports/observations" / f"{case_id}__{method}.json"
    _write_or_replay(path, observation)
    return path, observation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one serial, source-bound matched 8-case x 2-method "
            "1000-sample epsilon-stability recovery diagnostic."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--user-authorization", type=Path, required=True)
    parser.add_argument("--expected-user-authorization-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--expected-execution-authorization-sha256", required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--real-set-contract", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--required-idle-polls", type=int, default=3)
    parser.add_argument("--idle-poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    args.project = args.project.resolve()
    args.python = args.python.resolve()
    args.design = args.design.resolve()
    args.preparation = args.preparation.resolve()
    args.user_authorization = args.user_authorization.resolve()
    args.execution_authorization = args.execution_authorization.resolve()
    args.real_dir = args.real_dir.resolve()
    args.real_set_contract = args.real_set_contract.resolve()
    args.eval_cache = args.eval_cache.resolve()
    args.classifier_checkpoint = args.classifier_checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    args.status_output = args.status_output.resolve()
    args.lock = args.lock.resolve()
    execution_identity: dict[str, Any] = {}
    completed_arms = 0
    lock_created = False
    try:
        _write_status(
            args,
            status="preflight",
            detail="replaying_execution_gate_and_physical_sources",
            stage="preflight",
            completed_arms=0,
        )
        if any(name in os.environ for name in TRACKED_RUNTIME_VARIABLES):
            raise ValueError(
                "epsilon-stability controller requires the bound unset runtime variables"
            )
        if not args.python.is_file() or not os.access(args.python, os.X_OK):
            raise FileNotFoundError("epsilon-stability Python runtime is unavailable")
        if args.project != PROJECT_ROOT.resolve():
            raise ValueError("epsilon-stability controller project path differs")
        control_git = validate_control_git(
            args.project,
            expected_revision=args.expected_revision,
            expected_tree=args.expected_tree,
            expected_branch=args.expected_branch,
        )
        execution_identity, execution = replay_execution_gate(args)
        if execution["git"] != control_git:
            raise ValueError("epsilon-stability controller Git differs from execution gate")
        replay_evaluator_manifest(execution=execution, control_git=control_git)
        duplicates = duplicate_controller_pids(output_root=args.output_root)
        if duplicates:
            raise RuntimeError(
                f"duplicate epsilon-stability controller exists: {duplicates}"
            )
        samplers = matching_process_pids(markers=("scripts/generate_samples.py",))
        if samplers:
            raise RuntimeError(f"another generation sampler exists: {samplers}")
        if args.output_root.exists() or args.lock.exists():
            raise FileExistsError("epsilon-stability output root or lock already exists")
        if args.required_idle_polls != 3 or args.idle_poll_seconds <= 0.0:
            raise ValueError("epsilon-stability idle-GPU preflight contract differs")
        for poll in range(args.required_idle_polls):
            pids = gpu_compute_pids()
            if pids:
                raise RuntimeError(f"GPU is not idle; compute PIDs: {pids}")
            if poll + 1 < args.required_idle_polls:
                time.sleep(args.idle_poll_seconds)
        args.lock.mkdir(parents=False)
        lock_created = True
        args.output_root.mkdir(parents=False)
        args.eval_cache.mkdir(parents=True, exist_ok=True)
        design = _read_object(args.design)
        design_identity = gate_source_report_identity(args.design)
        write_json_report(
            args.output_root / "reports/controller_receipt.json",
            {
                "schema_version": 1,
                "role": "generation_epsilon_stability_controller_receipt",
                "status": "pass",
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "git": control_git,
                "execution_authorization": execution_identity,
                "output_root": args.output_root.as_posix(),
                "serial_case_order": [row["case_id"] for row in design["cases"]],
                "serial_method_order": list(METHODS),
                "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
                "result_authorization_boundary": dict(
                    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
                ),
            },
        )
        real_reference = _prepare_real_reference(
            args=args,
            design=design,
            design_identity=design_identity,
            execution=execution,
            execution_identity=execution_identity,
            completed_arms=completed_arms,
        )
        observation_rows: list[tuple[Path, dict[str, Any]]] = []
        for case in design["cases"]:
            case_id = str(case["case_id"])
            for method in METHODS:
                if gpu_compute_pids():
                    raise RuntimeError("foreign GPU process appeared before diagnostic arm")
                arm_root = args.output_root / "cases" / case_id / method
                sample_root = arm_root / "samples_1000_ddim100"
                _run_child(
                    sampling_command(
                        python=args.python,
                        project=args.project,
                        execution=execution,
                        design=design,
                        case_id=case_id,
                        method=method,
                        sample_root=sample_root,
                    ),
                    args=args,
                    detail="running_matched_1000_sample_arm",
                    stage="sampling",
                    execution_identity=execution_identity,
                    case_id=case_id,
                    method=method,
                    completed_arms=completed_arms,
                )
                commands = evaluation_commands(
                    python=args.python,
                    project=args.project,
                    args=args,
                    execution=execution,
                    case_id=case_id,
                    method=method,
                    arm_root=arm_root,
                )
                for stage in ("metrics", "class_fidelity", "artifact"):
                    if gpu_compute_pids():
                        raise RuntimeError(
                            f"foreign GPU process appeared before {stage} evaluation"
                        )
                    _run_child(
                        commands[stage],
                        args=args,
                        detail=f"running_{stage}_evaluation",
                        stage=stage,
                        execution_identity=execution_identity,
                        case_id=case_id,
                        method=method,
                        completed_arms=completed_arms,
                    )
                observation_rows.append(
                    _build_observation(
                        args=args,
                        design=design,
                        design_identity=design_identity,
                        execution=execution,
                        execution_identity=execution_identity,
                        case_id=case_id,
                        method=method,
                        arm_root=arm_root,
                    )
                )
                completed_arms += 1
                _write_status(
                    args,
                    status="running",
                    detail="diagnostic_arm_completed",
                    stage="observation",
                    case_id=case_id,
                    method=method,
                    completed_arms=completed_arms,
                    execution_identity=execution_identity,
                )

        observation_sources = [
            {
                "identity": gate_source_report_identity(path),
                "payload": payload,
            }
            for path, payload in observation_rows
        ]
        manifest = build_epsilon_stability_observation_manifest(
            design_identity=design_identity,
            execution_authorization_identity=execution_identity,
            observation_sources=observation_sources,
        )
        manifest_path = args.output_root / "reports/observation_manifest.json"
        _write_or_replay(manifest_path, manifest)
        result_path = args.output_root / "reports/epsilon_stability_sampling_result.json"
        _run_child(
            [
                str(args.python),
                str(
                    args.project
                    / "scripts/build_generation_epsilon_stability_sampling_result.py"
                ),
                "--design",
                str(args.design),
                "--expected-design-sha256",
                args.expected_design_sha256,
                "--execution-authorization",
                str(args.execution_authorization),
                "--expected-execution-authorization-sha256",
                args.expected_execution_authorization_sha256,
                "--observation-manifest",
                str(manifest_path),
                "--expected-observation-manifest-sha256",
                file_sha256(manifest_path),
                "--real-artifact-reference",
                str(real_reference["path"]),
                "--expected-real-artifact-reference-sha256",
                real_reference["identity"]["sha256"],
                "--output",
                str(result_path),
            ],
            args=args,
            detail="building_non_authorizing_sampling_result",
            stage="result",
            execution_identity=execution_identity,
            case_id=None,
            method=None,
            completed_arms=completed_arms,
        )
        result_identity = gate_source_report_identity(result_path)
        _run_child(
            [
                str(args.python),
                str(
                    args.project
                    / "scripts/validate_generation_epsilon_stability_sampling_result.py"
                ),
                "--result",
                str(result_path),
                "--expected-result-sha256",
                result_identity["sha256"],
                "--design",
                str(args.design),
                "--expected-design-sha256",
                args.expected_design_sha256,
                "--execution-authorization",
                str(args.execution_authorization),
                "--expected-execution-authorization-sha256",
                args.expected_execution_authorization_sha256,
                "--observation-manifest",
                str(manifest_path),
                "--expected-observation-manifest-sha256",
                file_sha256(manifest_path),
                "--real-artifact-reference",
                str(real_reference["path"]),
                "--expected-real-artifact-reference-sha256",
                real_reference["identity"]["sha256"],
            ],
            args=args,
            detail="replaying_non_authorizing_sampling_result",
            stage="result_replay",
            execution_identity=execution_identity,
            case_id=None,
            method=None,
            completed_arms=completed_arms,
        )
        result = _read_object(result_path)
        if result.get("generation_advantage_proven") is not False:
            raise ValueError("epsilon-stability result crossed its claim boundary")
        _write_status(
            args,
            status="completed",
            detail="matched_1000_sample_sampling_recovery_diagnostic_completed",
            stage="completed",
            completed_arms=completed_arms,
            execution_identity=execution_identity,
            result_identity=result_identity,
        )
        args.lock.rmdir()
        lock_created = False
        return 0
    except BaseException as error:
        _write_status(
            args,
            status="failed",
            detail="epsilon_stability_diagnostic_failed_closed",
            stage="failed",
            completed_arms=completed_arms,
            execution_identity=execution_identity,
            error=f"{type(error).__name__}: {error}",
        )
        if lock_created:
            write_json_report(
                args.lock / "failure.json",
                {
                    "schema_version": 1,
                    "role": "generation_epsilon_stability_fail_closed_lock",
                    "status": "failed",
                    "pid": os.getpid(),
                    "error": f"{type(error).__name__}: {error}",
                    "process_signals_allowed": False,
                },
            )
        raise


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
