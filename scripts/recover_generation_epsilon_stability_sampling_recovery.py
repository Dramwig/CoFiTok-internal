from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from cofitok.generation import (
    EPSILON_STABILITY_EXECUTION_ACTIONS,
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    build_epsilon_stability_execution_authorization,
    build_epsilon_stability_observation_manifest,
    build_epsilon_stability_real_artifact_reference,
    build_epsilon_stability_sampling_design,
    validate_epsilon_stability_execution_authorization,
    validate_epsilon_stability_user_authorization_receipt,
)
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, git_provenance, write_json_report
from scripts import run_generation_epsilon_stability_sampling_recovery as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCHEMA = "cofitok_epsilon_stability_pre_gpu_recovery_authorization_v1"
RECOVERY_ROLE = "generation_epsilon_stability_pre_gpu_recovery_controller"
RECOVERY_REASON = "real_reference_jpeg_materialization_contract_fix"
FAILED_LOG_MARKER = "real artifact source image contract differs"
FAILED_HELPER = "prepare_generation_epsilon_stability_real_artifact_subset.py"
MATERIALIZER_RELATIVE_PATH = (
    "scripts/prepare_generation_epsilon_stability_real_artifact_subset.py"
)
RECOVERY_CONTROLLER_RELATIVE_PATH = (
    "scripts/recover_generation_epsilon_stability_sampling_recovery.py"
)
RECOVERY_RUNBOOK_RELATIVE_PATH = (
    "artifacts/runbooks/"
    "generation_epsilon_stability_sampling_recovery_pre_gpu_recovery_v1.sh"
)
RECOVERY_DIFF_CONTRACT = {
    RECOVERY_RUNBOOK_RELATIVE_PATH: "A",
    "docs/records/"
    "2026-08-24_generation_epsilon_stability_sampling_recovery_execution.md": "M",
    MATERIALIZER_RELATIVE_PATH: "M",
    RECOVERY_CONTROLLER_RELATIVE_PATH: "A",
    "tests/test_generation_epsilon_stability_controller.py": "M",
    "tests/test_generation_epsilon_stability_recovery.py": "A",
}
TRACKED_RUNTIME_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "PYTORCH_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _read_object(path: str | Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="epsilon-stability recovery JSON")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {source}")
    return payload


def _identity(
    path: str | Path,
    expected_sha256: str | None = None,
    *,
    label: str,
) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=f"epsilon-stability recovery {label}")
    identity = gate_source_report_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"epsilon-stability recovery {label} SHA256 differs")
    return identity


def _source(
    path: str | Path,
    expected_sha256: str | None = None,
    *,
    label: str,
) -> dict[str, Any]:
    identity = _identity(path, expected_sha256, label=label)
    return {"identity": identity, "payload": _read_object(identity["path"])}


def _git_identity(path: Path) -> dict[str, Any]:
    root = path.resolve()
    observed = {
        **git_provenance(root),
        "tree": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
            text=True,
        ).strip(),
        "path": root.as_posix(),
    }
    full_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )
    if observed["tracked_dirty"] or full_status:
        raise ValueError(f"epsilon-stability recovery checkout is not clean: {root}")
    return {**observed, "full_status_clean": True}


def _validate_recovery_git_diff(
    recovery_project: Path,
    *,
    execution_revision: str,
) -> dict[str, Any]:
    root = recovery_project.resolve()
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "merge-base",
            "--is-ancestor",
            execution_revision,
            "HEAD",
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("epsilon-stability recovery is not based on original execution")
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-status",
            "--no-renames",
            f"{execution_revision}..HEAD",
        ],
        text=True,
    )
    observed: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M"}:
            raise ValueError("epsilon-stability recovery Git diff is unsupported")
        status, relative = fields
        if relative in observed:
            raise ValueError("epsilon-stability recovery Git diff has duplicates")
        observed[relative] = status
    if observed != RECOVERY_DIFF_CONTRACT:
        raise ValueError("epsilon-stability recovery Git diff exceeds its contract")
    return {
        "base_revision": execution_revision,
        "changes": [
            {"path": relative, "status": observed[relative]}
            for relative in sorted(observed)
        ],
        "only_existing_runtime_source_changed": MATERIALIZER_RELATIVE_PATH,
        "new_runtime_sources": [
            RECOVERY_CONTROLLER_RELATIVE_PATH,
            RECOVERY_RUNBOOK_RELATIVE_PATH,
        ],
        "status": "pass",
    }


def _inventory(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"epsilon-stability recovery inventory root is invalid: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"epsilon-stability recovery inventory has symlink: {path}")
        if path.is_dir():
            rows.append({"path": relative, "type": "directory"})
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "identity": gate_source_report_identity(path),
                }
            )
        else:
            raise ValueError(f"unsupported recovery inventory entry: {path}")
    return rows


def _is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _validate_recovery_state_paths(
    *,
    output_root: Path,
    failed_lock: Path,
    recovery_lock: Path,
    recovery_status: Path,
) -> dict[str, str]:
    protected = {
        "failed output": output_root.resolve(),
        "failed lock": failed_lock.resolve(),
    }
    recovery = {
        "recovery lock": recovery_lock.resolve(),
        "recovery status": recovery_status.resolve(),
    }
    if recovery["recovery lock"] == recovery["recovery status"]:
        raise ValueError("epsilon-stability recovery lock and status must be separate")
    for recovery_label, recovery_path in recovery.items():
        for protected_label, protected_path in protected.items():
            if _is_within(recovery_path, protected_path):
                raise ValueError(
                    f"epsilon-stability {recovery_label} is inside {protected_label}"
                )
    if _is_within(recovery["recovery status"], recovery["recovery lock"]):
        raise ValueError("epsilon-stability recovery status is inside recovery lock")
    if _is_within(recovery["recovery lock"], recovery["recovery status"]):
        raise ValueError("epsilon-stability recovery lock is inside recovery status")
    return {
        "failed_output": protected["failed output"].as_posix(),
        "failed_lock": protected["failed lock"].as_posix(),
        "recovery_lock": recovery["recovery lock"].as_posix(),
        "recovery_status": recovery["recovery status"].as_posix(),
    }


def _expected_failed_output_inventory(
    controller_receipt_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"path": "real_artifact_reference", "type": "directory"},
        {"path": "real_artifact_reference/images", "type": "directory"},
        {"path": "reports", "type": "directory"},
        {
            "path": "reports/controller_receipt.json",
            "type": "file",
            "identity": dict(controller_receipt_identity),
        },
    ]


def _validate_original_evaluator(
    execution_project: Path,
    execution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _source(
        execution["source_bindings"]["evaluator"]["path"],
        execution["source_bindings"]["evaluator"]["sha256"],
        label="original evaluator manifest",
    )
    manifest = source["payload"]
    rows = manifest.get("sources")
    expected_git = {**execution["git"]}
    if (
        manifest.get("schema")
        != "cofitok_matched_epsilon_stability_evaluator_manifest_v1"
        or manifest.get("status") != "pass"
        or manifest.get("git") != expected_git
        or not isinstance(rows, Mapping)
        or not rows
    ):
        raise ValueError("epsilon-stability original evaluator manifest differs")
    for relative, expected in rows.items():
        actual = gate_source_report_identity(
            reject_symlink_chain(
                execution_project / str(relative),
                name=f"epsilon-stability original evaluator source {relative}",
            )
        )
        if actual != dict(expected):
            raise ValueError(
                f"epsilon-stability original evaluator source changed: {relative}"
            )
    return source["identity"], manifest


def _validate_recovery_evaluator_equivalence(
    *,
    recovery_project: Path,
    evaluator_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = evaluator_manifest.get("sources")
    if not isinstance(rows, Mapping) or MATERIALIZER_RELATIVE_PATH not in rows:
        raise ValueError("epsilon-stability evaluator source set is incomplete")
    drift: list[str] = []
    for relative, expected in rows.items():
        if not isinstance(expected, Mapping):
            raise ValueError("epsilon-stability evaluator source identity is malformed")
        observed = gate_source_report_identity(
            reject_symlink_chain(
                recovery_project / str(relative),
                name=f"epsilon-stability recovery evaluator source {relative}",
            )
        )
        if (
            observed["bytes"] != expected.get("bytes")
            or observed["sha256"] != expected.get("sha256")
        ):
            drift.append(str(relative))
    if drift != [MATERIALIZER_RELATIVE_PATH]:
        raise ValueError(
            "epsilon-stability recovery evaluator drift is not materializer-only"
        )
    materializer = gate_source_report_identity(
        recovery_project / MATERIALIZER_RELATIVE_PATH
    )
    return {
        "status": "pass",
        "evaluator_source_count": len(rows),
        "byte_identical_non_materializer_source_count": len(rows) - 1,
        "only_drifted_evaluator_source": MATERIALIZER_RELATIVE_PATH,
        "recovery_materializer": materializer,
    }


def _replay_original_execution(
    *,
    execution_project: Path,
    design_source: Mapping[str, Any],
    preparation_source: Mapping[str, Any],
    user_source: Mapping[str, Any],
    execution_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    design = design_source["payload"]
    preparation = preparation_source["payload"]
    user = user_source["payload"]
    execution = execution_source["payload"]
    if design != build_epsilon_stability_sampling_design():
        raise ValueError("epsilon-stability recovery design is not canonical")
    validate_epsilon_stability_user_authorization_receipt(user)
    expected = build_epsilon_stability_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_source["identity"],
        design=design,
        user_authorization=user,
        user_authorization_identity=user_source["identity"],
    )
    if execution != expected:
        raise ValueError("epsilon-stability original execution authorization differs")
    validate_epsilon_stability_execution_authorization(
        execution,
        design_identity=design_source["identity"],
        design=design,
    )
    observed_git = _git_identity(execution_project)
    if {
        key: observed_git[key]
        for key in ("revision", "tree", "branch", "tracked_dirty")
    } != execution["git"]:
        raise ValueError("epsilon-stability original execution checkout differs")
    evaluator, evaluator_manifest = _validate_original_evaluator(
        execution_project, execution
    )
    return execution, evaluator, evaluator_manifest


def _validate_failed_attempt(
    *,
    output_root: Path,
    failed_lock: Path,
    status_source: Mapping[str, Any],
    log_source: Mapping[str, Any],
    controller_receipt_source: Mapping[str, Any],
    failure_source: Mapping[str, Any],
    launch_source: Mapping[str, Any],
    prelaunch_source: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    status = status_source["payload"]
    receipt = controller_receipt_source["payload"]
    failure = failure_source["payload"]
    launch = launch_source["payload"]
    prelaunch = prelaunch_source["payload"]
    log_text = Path(log_source["identity"]["path"]).read_text(
        encoding="utf-8", errors="replace"
    )
    if (
        status.get("status") != "failed"
        or status.get("stage") != "failed"
        or status.get("completed_arms") != 0
        or status.get("child_pid") is not None
        or status.get("output_root") != output_root.resolve().as_posix()
        or status.get("execution_authorization") != dict(execution_identity)
        or status.get("generation_advantage_proven") is not False
    ):
        raise ValueError("epsilon-stability failed controller status differs")
    if FAILED_LOG_MARKER not in log_text or FAILED_HELPER not in log_text:
        raise ValueError("epsilon-stability failed controller log is not the JPEG bug")
    if "scripts/generate_samples.py" in log_text:
        raise ValueError("epsilon-stability failed attempt reached sampling")
    if (
        receipt.get("status") != "pass"
        or receipt.get("execution_authorization") != dict(execution_identity)
        or receipt.get("output_root") != output_root.resolve().as_posix()
        or receipt.get("authorized_actions") != EPSILON_STABILITY_EXECUTION_ACTIONS
        or receipt.get("result_authorization_boundary")
        != EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
    ):
        raise ValueError("epsilon-stability original controller receipt differs")
    if (
        failure.get("status") != "failed"
        or failure.get("pid") != status.get("pid")
        or failure.get("error") != status.get("error")
        or failure.get("process_signals_allowed") is not False
    ):
        raise ValueError("epsilon-stability failed lock record differs")
    if (
        launch.get("status") != "launched"
        or launch.get("pid") != status.get("pid")
        or launch.get("output_root") != output_root.resolve().as_posix()
        or launch.get("generation_advantage_proven") is not False
    ):
        raise ValueError("epsilon-stability original launch receipt differs")
    if prelaunch.get("status") != "pass":
        raise ValueError("epsilon-stability original prelaunch receipt differs")
    expected_inventory = _expected_failed_output_inventory(
        controller_receipt_source["identity"]
    )
    if _inventory(output_root) != expected_inventory:
        raise ValueError("epsilon-stability failed output inventory differs")
    if _inventory(failed_lock) != [
        {
            "path": "failure.json",
            "type": "file",
            "identity": failure_source["identity"],
        }
    ]:
        raise ValueError("epsilon-stability failed lock inventory differs")
    return {
        "status": "failed_before_gpu_sampling",
        "completed_arms": 0,
        "failure_kind": RECOVERY_REASON,
        "original_controller_pid": int(status["pid"]),
        "output_inventory": expected_inventory,
        "failed_lock_inventory": _inventory(failed_lock),
    }


def build_live_recovery_authorization(
    *,
    recovery_project: Path,
    expected_recovery_revision: str,
    expected_recovery_tree: str,
    expected_recovery_branch: str,
    execution_project: Path,
    python: Path,
    design_path: Path,
    expected_design_sha256: str,
    preparation_path: Path,
    expected_preparation_sha256: str,
    user_authorization_path: Path,
    expected_user_authorization_sha256: str,
    execution_authorization_path: Path,
    expected_execution_authorization_sha256: str,
    output_root: Path,
    failed_lock: Path,
    recovery_lock: Path,
    recovery_status: Path,
    eval_cache: Path,
    controller_status_path: Path,
    expected_controller_status_sha256: str,
    controller_log_path: Path,
    expected_controller_log_sha256: str,
    controller_receipt_path: Path,
    expected_controller_receipt_sha256: str,
    failure_record_path: Path,
    expected_failure_record_sha256: str,
    launch_receipt_path: Path,
    expected_launch_receipt_sha256: str,
    prelaunch_receipt_path: Path,
    expected_prelaunch_receipt_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    if recovery_project.resolve() != PROJECT_ROOT.resolve():
        raise ValueError("epsilon-stability recovery project path differs")
    recovery_git = _git_identity(recovery_project)
    expected_recovery_git = {
        "revision": expected_recovery_revision,
        "tree": expected_recovery_tree,
        "branch": expected_recovery_branch,
        "tracked_dirty": False,
    }
    if {
        key: recovery_git[key]
        for key in ("revision", "tree", "branch", "tracked_dirty")
    } != expected_recovery_git:
        raise ValueError("epsilon-stability recovery Git differs")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError("epsilon-stability recovery Python is unavailable")
    design_source = _source(
        design_path, expected_design_sha256, label="sampling design"
    )
    preparation_source = _source(
        preparation_path,
        expected_preparation_sha256,
        label="original preparation",
    )
    user_source = _source(
        user_authorization_path,
        expected_user_authorization_sha256,
        label="user authorization",
    )
    execution_source = _source(
        execution_authorization_path,
        expected_execution_authorization_sha256,
        label="original execution authorization",
    )
    (
        execution,
        original_evaluator_identity,
        evaluator_manifest,
    ) = _replay_original_execution(
        execution_project=execution_project,
        design_source=design_source,
        preparation_source=preparation_source,
        user_source=user_source,
        execution_source=execution_source,
    )
    if output_root.resolve().as_posix() != execution["output_root"]:
        raise ValueError("epsilon-stability recovery output root differs")
    if failed_lock.resolve().as_posix() != f"{execution['output_root']}.lock":
        raise ValueError("epsilon-stability recovery failed lock differs")
    state_paths = _validate_recovery_state_paths(
        output_root=output_root,
        failed_lock=failed_lock,
        recovery_lock=recovery_lock,
        recovery_status=recovery_status,
    )
    if recovery_lock.exists() or recovery_status.exists():
        raise FileExistsError("epsilon-stability recovery state already exists")
    if (output_root / "reports/controller_recovery_receipt.json").exists():
        raise FileExistsError("epsilon-stability recovery receipt already exists")

    status_source = _source(
        controller_status_path,
        expected_controller_status_sha256,
        label="failed controller status",
    )
    log_identity = _identity(
        controller_log_path,
        expected_controller_log_sha256,
        label="failed controller log",
    )
    log_source = {"identity": log_identity, "payload": None}
    receipt_source = _source(
        controller_receipt_path,
        expected_controller_receipt_sha256,
        label="original controller receipt",
    )
    failure_source = _source(
        failure_record_path,
        expected_failure_record_sha256,
        label="failed lock record",
    )
    launch_source = _source(
        launch_receipt_path,
        expected_launch_receipt_sha256,
        label="original launch receipt",
    )
    prelaunch_source = _source(
        prelaunch_receipt_path,
        expected_prelaunch_receipt_sha256,
        label="original prelaunch receipt",
    )
    failed_state = _validate_failed_attempt(
        output_root=output_root,
        failed_lock=failed_lock,
        status_source=status_source,
        log_source=log_source,
        controller_receipt_source=receipt_source,
        failure_source=failure_source,
        launch_source=launch_source,
        prelaunch_source=prelaunch_source,
        execution_identity=execution_source["identity"],
    )
    recovery_git_diff = _validate_recovery_git_diff(
        recovery_project,
        execution_revision=str(execution["git"]["revision"]),
    )
    evaluator_equivalence = _validate_recovery_evaluator_equivalence(
        recovery_project=recovery_project,
        evaluator_manifest=evaluator_manifest,
    )
    recovery_sources = {
        "controller": _identity(
            recovery_project / RECOVERY_CONTROLLER_RELATIVE_PATH,
            label="recovery controller",
        ),
        "runbook": _identity(
            recovery_project / RECOVERY_RUNBOOK_RELATIVE_PATH,
            label="recovery runbook",
        ),
        "real_reference_materializer": _identity(
            recovery_project / MATERIALIZER_RELATIVE_PATH,
            label="JPEG materializer",
        ),
    }
    if (
        evaluator_equivalence["recovery_materializer"]
        != recovery_sources["real_reference_materializer"]
    ):
        raise ValueError("epsilon-stability recovery materializer binding differs")
    return {
        "schema": RECOVERY_SCHEMA,
        "status": "approved",
        "scope": "matched_1000_sample_epsilon_stability_sampling_diagnostic_only",
        "reason": RECOVERY_REASON,
        "created_at": created_at,
        "hostname": socket.gethostname(),
        "recovery_git": recovery_git,
        "execution_git": {
            **execution["git"],
            "path": execution_project.resolve().as_posix(),
            "full_status_clean": True,
        },
        "original_sources": {
            "design": design_source["identity"],
            "preparation": preparation_source["identity"],
            "user_authorization": user_source["identity"],
            "execution_authorization": execution_source["identity"],
            "evaluator_manifest": original_evaluator_identity,
        },
        "recovery_sources": recovery_sources,
        "recovery_git_diff": recovery_git_diff,
        "evaluator_equivalence": evaluator_equivalence,
        "failure_evidence": {
            "controller_status": status_source["identity"],
            "controller_log": log_identity,
            "controller_receipt": receipt_source["identity"],
            "failure_record": failure_source["identity"],
            "launch_receipt": launch_source["identity"],
            "prelaunch_receipt": prelaunch_source["identity"],
        },
        "failed_state": failed_state,
        "state_path_contract": state_paths,
        "paths": {
            "python": python.resolve().as_posix(),
            "recovery_project": recovery_project.resolve().as_posix(),
            "execution_project": execution_project.resolve().as_posix(),
            "output_root": output_root.resolve().as_posix(),
            "failed_lock": failed_lock.resolve().as_posix(),
            "recovery_lock": recovery_lock.resolve().as_posix(),
            "recovery_status": recovery_status.resolve().as_posix(),
            "eval_cache": eval_cache.resolve().as_posix(),
        },
        "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
        "limitations": {
            "same_diagnostic_output_root_only": True,
            "original_failed_output_and_lock_preserved": True,
            "recovery_attempt_count": 1,
            "materializer_only_code_change": True,
            "sampling_and_evaluator_checkout_unchanged": True,
            "independent_10000_confirmation_allowed": False,
            "training_allowed": False,
            "full_300k_allowed": False,
            "promotion_allowed": False,
            "export_allowed": False,
            "release_allowed": False,
            "process_signals_allowed": False,
        },
        "generation_advantage_proven": False,
    }


def replay_live_recovery_authorization(
    authorization_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity(
        authorization_path,
        expected_sha256,
        label="recovery authorization",
    )
    actual = _read_object(identity["path"])
    paths = actual.get("paths")
    original = actual.get("original_sources")
    failure = actual.get("failure_evidence")
    recovery_git = actual.get("recovery_git")
    if not all(isinstance(value, Mapping) for value in (paths, original, failure, recovery_git)):
        raise ValueError("epsilon-stability recovery authorization is malformed")
    expected = build_live_recovery_authorization(
        recovery_project=Path(str(paths["recovery_project"])),
        expected_recovery_revision=str(recovery_git["revision"]),
        expected_recovery_tree=str(recovery_git["tree"]),
        expected_recovery_branch=str(recovery_git["branch"]),
        execution_project=Path(str(paths["execution_project"])),
        python=Path(str(paths["python"])),
        design_path=Path(str(original["design"]["path"])),
        expected_design_sha256=str(original["design"]["sha256"]),
        preparation_path=Path(str(original["preparation"]["path"])),
        expected_preparation_sha256=str(original["preparation"]["sha256"]),
        user_authorization_path=Path(str(original["user_authorization"]["path"])),
        expected_user_authorization_sha256=str(
            original["user_authorization"]["sha256"]
        ),
        execution_authorization_path=Path(
            str(original["execution_authorization"]["path"])
        ),
        expected_execution_authorization_sha256=str(
            original["execution_authorization"]["sha256"]
        ),
        output_root=Path(str(paths["output_root"])),
        failed_lock=Path(str(paths["failed_lock"])),
        recovery_lock=Path(str(paths["recovery_lock"])),
        recovery_status=Path(str(paths["recovery_status"])),
        eval_cache=Path(str(paths["eval_cache"])),
        controller_status_path=Path(str(failure["controller_status"]["path"])),
        expected_controller_status_sha256=str(
            failure["controller_status"]["sha256"]
        ),
        controller_log_path=Path(str(failure["controller_log"]["path"])),
        expected_controller_log_sha256=str(failure["controller_log"]["sha256"]),
        controller_receipt_path=Path(str(failure["controller_receipt"]["path"])),
        expected_controller_receipt_sha256=str(
            failure["controller_receipt"]["sha256"]
        ),
        failure_record_path=Path(str(failure["failure_record"]["path"])),
        expected_failure_record_sha256=str(failure["failure_record"]["sha256"]),
        launch_receipt_path=Path(str(failure["launch_receipt"]["path"])),
        expected_launch_receipt_sha256=str(failure["launch_receipt"]["sha256"]),
        prelaunch_receipt_path=Path(str(failure["prelaunch_receipt"]["path"])),
        expected_prelaunch_receipt_sha256=str(
            failure["prelaunch_receipt"]["sha256"]
        ),
        created_at=str(actual["created_at"]),
    )
    if actual != expected:
        raise ValueError("epsilon-stability recovery authorization does not replay")
    return identity, actual


def _matching_process_pids(markers: Sequence[str]) -> list[int]:
    return base.matching_process_pids(markers=markers)


def duplicate_recovery_controller_pids(
    *,
    authorization_path: Path,
    own_pid: int | None = None,
    proc_root: str | Path = "/proc",
) -> list[int]:
    return base.matching_process_pids(
        markers=(
            RECOVERY_CONTROLLER_RELATIVE_PATH,
            authorization_path.resolve().as_posix(),
        ),
        own_pid=own_pid,
        proc_root=proc_root,
    )


def _status_payload(
    *,
    authorization: Mapping[str, Any],
    authorization_identity: Mapping[str, Any],
    status: str,
    detail: str,
    stage: str,
    child_pid: int | None = None,
    case_id: str | None = None,
    method: str | None = None,
    completed_arms: int = 0,
    result_identity: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": RECOVERY_ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "child_pid": child_pid,
        "stage": stage,
        "case_id": case_id,
        "method": method,
        "completed_arms": completed_arms,
        "total_arms": 16,
        "output_root": authorization["paths"]["output_root"],
        "execution_project": authorization["paths"]["execution_project"],
        "recovery_project": authorization["paths"]["recovery_project"],
        "recovery_authorization": dict(authorization_identity),
        "original_execution_authorization": authorization["original_sources"][
            "execution_authorization"
        ],
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


def _write_status(
    authorization: Mapping[str, Any],
    authorization_identity: Mapping[str, Any],
    **kwargs: Any,
) -> None:
    write_json_report(
        Path(str(authorization["paths"]["recovery_status"])),
        _status_payload(
            authorization=authorization,
            authorization_identity=authorization_identity,
            **kwargs,
        ),
    )


def _run_child(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    authorization: Mapping[str, Any],
    authorization_identity: Mapping[str, Any],
    detail: str,
    stage: str,
    case_id: str | None,
    method: str | None,
    completed_arms: int,
) -> None:
    process = subprocess.Popen(list(command), cwd=cwd, env=dict(env))
    _write_status(
        authorization,
        authorization_identity,
        status="running",
        detail=detail,
        stage=stage,
        child_pid=process.pid,
        case_id=case_id,
        method=method,
        completed_arms=completed_arms,
    )
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(command))


def _child_environment(project: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in TRACKED_RUNTIME_VARIABLES:
        environment.pop(name, None)
    environment["PYTHONPATH"] = f"{project / 'src'}:{project}"
    return environment


def _prepare_real_reference(
    *,
    args: SimpleNamespace,
    recovery_project: Path,
    recovery_env: Mapping[str, str],
    execution_env: Mapping[str, str],
    design: Mapping[str, Any],
    design_identity: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_identity: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorization_identity: Mapping[str, Any],
    completed_arms: int,
) -> dict[str, Any]:
    real_root = args.output_root / "real_artifact_reference"
    images = real_root / "images"
    subset_manifest = real_root / "subset_manifest.json"
    _run_child(
        [
            str(args.python),
            str(recovery_project / MATERIALIZER_RELATIVE_PATH),
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
        cwd=recovery_project,
        env=recovery_env,
        authorization=authorization,
        authorization_identity=authorization_identity,
        detail="recovering_physical_real_artifact_reference",
        stage="real_reference_subset",
        case_id=None,
        method=None,
        completed_arms=completed_arms,
    )
    artifact_report = real_root / "artifact_report.json"
    _run_child(
        [
            str(args.python),
            str(
                args.project
                / "scripts/evaluate_generation_epsilon_stability_artifacts.py"
            ),
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
        cwd=args.project,
        env=execution_env,
        authorization=authorization,
        authorization_identity=authorization_identity,
        detail="evaluating_recovered_real_artifact_reference",
        stage="real_reference_artifact",
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
    base._write_or_replay(reference_path, reference)
    return {
        "path": reference_path,
        "identity": gate_source_report_identity(reference_path),
        "payload": reference,
    }


def run_recovery(
    authorization_path: Path,
    expected_authorization_sha256: str,
    *,
    required_idle_polls: int,
    idle_poll_seconds: float,
) -> int:
    authorization_identity: dict[str, Any] = {}
    authorization: dict[str, Any] = {}
    completed_arms = 0
    lock_created = False
    try:
        if any(name in os.environ for name in TRACKED_RUNTIME_VARIABLES):
            raise ValueError(
                "epsilon-stability recovery requires the bound unset runtime variables"
            )
        authorization_identity, authorization = replay_live_recovery_authorization(
            authorization_path,
            expected_authorization_sha256,
        )
        _write_status(
            authorization,
            authorization_identity,
            status="preflight",
            detail="replaying_recovery_authorization_and_original_execution",
            stage="preflight",
            completed_arms=0,
        )
        paths = authorization["paths"]
        recovery_project = Path(str(paths["recovery_project"])).resolve()
        execution_project = Path(str(paths["execution_project"])).resolve()
        output_root = Path(str(paths["output_root"])).resolve()
        recovery_lock = Path(str(paths["recovery_lock"])).resolve()
        python = Path(str(paths["python"])).resolve()
        original = authorization["original_sources"]
        design_source = _source(
            original["design"]["path"],
            original["design"]["sha256"],
            label="sampling design",
        )
        preparation_source = _source(
            original["preparation"]["path"],
            original["preparation"]["sha256"],
            label="original preparation",
        )
        user_source = _source(
            original["user_authorization"]["path"],
            original["user_authorization"]["sha256"],
            label="user authorization",
        )
        execution_source = _source(
            original["execution_authorization"]["path"],
            original["execution_authorization"]["sha256"],
            label="original execution authorization",
        )
        execution, evaluator_identity, _ = _replay_original_execution(
            execution_project=execution_project,
            design_source=design_source,
            preparation_source=preparation_source,
            user_source=user_source,
            execution_source=execution_source,
        )
        if evaluator_identity != original["evaluator_manifest"]:
            raise ValueError("epsilon-stability original evaluator identity differs")
        if recovery_project != PROJECT_ROOT.resolve():
            raise ValueError("epsilon-stability running recovery checkout differs")
        duplicates = sorted(
            set(
                base.duplicate_controller_pids(output_root=output_root)
                + duplicate_recovery_controller_pids(
                    authorization_path=authorization_path,
                )
            )
            - {os.getpid()}
        )
        if duplicates:
            raise RuntimeError(
                f"duplicate epsilon-stability recovery controller exists: {duplicates}"
            )
        samplers = _matching_process_pids(("scripts/generate_samples.py",))
        if samplers:
            raise RuntimeError(f"another generation sampler exists: {samplers}")
        if required_idle_polls != 3 or idle_poll_seconds <= 0.0:
            raise ValueError("epsilon-stability recovery idle-GPU contract differs")
        for poll in range(required_idle_polls):
            pids = base.gpu_compute_pids()
            if pids:
                raise RuntimeError(f"GPU is not idle; compute PIDs: {pids}")
            if poll + 1 < required_idle_polls:
                time.sleep(idle_poll_seconds)
        recovery_lock.mkdir(parents=False)
        lock_created = True

        args = SimpleNamespace(
            project=execution_project,
            python=python,
            real_dir=Path(str(execution["real_set"]["root"])).resolve(),
            real_set_contract=Path(
                str(execution["source_bindings"]["real_set"]["path"])
            ).resolve(),
            eval_cache=Path(str(paths["eval_cache"])).resolve(),
            classifier_checkpoint=Path(
                str(execution["source_bindings"]["classifier"]["path"])
            ).resolve(),
            output_root=output_root,
        )
        recovery_env = _child_environment(recovery_project)
        execution_env = _child_environment(execution_project)
        design = design_source["payload"]
        recovery_receipt = {
            "schema_version": 1,
            "role": "generation_epsilon_stability_controller_recovery_receipt",
            "status": "pass",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "recovery_authorization": authorization_identity,
            "original_execution_authorization": execution_source["identity"],
            "recovery_git": authorization["recovery_git"],
            "execution_git": authorization["execution_git"],
            "output_root": output_root.as_posix(),
            "failed_lock_preserved": authorization["paths"]["failed_lock"],
            "recovery_lock": recovery_lock.as_posix(),
            "serial_case_order": [row["case_id"] for row in design["cases"]],
            "serial_method_order": list(base.METHODS),
            "authorized_actions": dict(EPSILON_STABILITY_EXECUTION_ACTIONS),
            "result_authorization_boundary": dict(
                EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
            ),
            "generation_advantage_proven": False,
        }
        base._write_or_replay(
            output_root / "reports/controller_recovery_receipt.json",
            recovery_receipt,
        )
        real_reference = _prepare_real_reference(
            args=args,
            recovery_project=recovery_project,
            recovery_env=recovery_env,
            execution_env=execution_env,
            design=design,
            design_identity=design_source["identity"],
            execution=execution,
            execution_identity=execution_source["identity"],
            authorization=authorization,
            authorization_identity=authorization_identity,
            completed_arms=completed_arms,
        )
        observation_rows: list[tuple[Path, dict[str, Any]]] = []
        for case in design["cases"]:
            case_id = str(case["case_id"])
            for method in base.METHODS:
                if base.gpu_compute_pids():
                    raise RuntimeError(
                        "foreign GPU process appeared before diagnostic arm"
                    )
                arm_root = output_root / "cases" / case_id / method
                sample_root = arm_root / "samples_1000_ddim100"
                _run_child(
                    base.sampling_command(
                        python=python,
                        project=execution_project,
                        execution=execution,
                        design=design,
                        case_id=case_id,
                        method=method,
                        sample_root=sample_root,
                    ),
                    cwd=execution_project,
                    env=execution_env,
                    authorization=authorization,
                    authorization_identity=authorization_identity,
                    detail="running_recovered_matched_1000_sample_arm",
                    stage="sampling",
                    case_id=case_id,
                    method=method,
                    completed_arms=completed_arms,
                )
                commands = base.evaluation_commands(
                    python=python,
                    project=execution_project,
                    args=args,
                    execution=execution,
                    case_id=case_id,
                    method=method,
                    arm_root=arm_root,
                )
                for stage in ("metrics", "class_fidelity", "artifact"):
                    if base.gpu_compute_pids():
                        raise RuntimeError(
                            f"foreign GPU process appeared before {stage} evaluation"
                        )
                    _run_child(
                        commands[stage],
                        cwd=execution_project,
                        env=execution_env,
                        authorization=authorization,
                        authorization_identity=authorization_identity,
                        detail=f"running_recovered_{stage}_evaluation",
                        stage=stage,
                        case_id=case_id,
                        method=method,
                        completed_arms=completed_arms,
                    )
                observation_rows.append(
                    base._build_observation(
                        args=args,
                        design=design,
                        design_identity=design_source["identity"],
                        execution=execution,
                        execution_identity=execution_source["identity"],
                        case_id=case_id,
                        method=method,
                        arm_root=arm_root,
                    )
                )
                completed_arms += 1
                _write_status(
                    authorization,
                    authorization_identity,
                    status="running",
                    detail="recovered_diagnostic_arm_completed",
                    stage="observation",
                    case_id=case_id,
                    method=method,
                    completed_arms=completed_arms,
                )

        observation_sources = [
            {
                "identity": gate_source_report_identity(path),
                "payload": payload,
            }
            for path, payload in observation_rows
        ]
        manifest = build_epsilon_stability_observation_manifest(
            design_identity=design_source["identity"],
            execution_authorization_identity=execution_source["identity"],
            observation_sources=observation_sources,
        )
        manifest_path = output_root / "reports/observation_manifest.json"
        base._write_or_replay(manifest_path, manifest)
        result_path = output_root / "reports/epsilon_stability_sampling_result.json"
        _run_child(
            [
                str(python),
                str(
                    execution_project
                    / "scripts/build_generation_epsilon_stability_sampling_result.py"
                ),
                "--design",
                str(design_source["identity"]["path"]),
                "--expected-design-sha256",
                design_source["identity"]["sha256"],
                "--execution-authorization",
                str(execution_source["identity"]["path"]),
                "--expected-execution-authorization-sha256",
                execution_source["identity"]["sha256"],
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
            cwd=execution_project,
            env=execution_env,
            authorization=authorization,
            authorization_identity=authorization_identity,
            detail="building_recovered_non_authorizing_sampling_result",
            stage="result",
            case_id=None,
            method=None,
            completed_arms=completed_arms,
        )
        result_identity = gate_source_report_identity(result_path)
        _run_child(
            [
                str(python),
                str(
                    execution_project
                    / "scripts/validate_generation_epsilon_stability_sampling_result.py"
                ),
                "--result",
                str(result_path),
                "--expected-result-sha256",
                result_identity["sha256"],
                "--design",
                str(design_source["identity"]["path"]),
                "--expected-design-sha256",
                design_source["identity"]["sha256"],
                "--execution-authorization",
                str(execution_source["identity"]["path"]),
                "--expected-execution-authorization-sha256",
                execution_source["identity"]["sha256"],
                "--observation-manifest",
                str(manifest_path),
                "--expected-observation-manifest-sha256",
                file_sha256(manifest_path),
                "--real-artifact-reference",
                str(real_reference["path"]),
                "--expected-real-artifact-reference-sha256",
                real_reference["identity"]["sha256"],
            ],
            cwd=execution_project,
            env=execution_env,
            authorization=authorization,
            authorization_identity=authorization_identity,
            detail="replaying_recovered_non_authorizing_sampling_result",
            stage="result_replay",
            case_id=None,
            method=None,
            completed_arms=completed_arms,
        )
        result = _read_object(result_path)
        if result.get("generation_advantage_proven") is not False:
            raise ValueError("epsilon-stability recovery crossed its claim boundary")
        _write_status(
            authorization,
            authorization_identity,
            status="completed",
            detail="matched_1000_sample_sampling_recovery_diagnostic_completed",
            stage="completed",
            completed_arms=completed_arms,
            result_identity=result_identity,
        )
        recovery_lock.rmdir()
        lock_created = False
        return 0
    except BaseException as error:
        if authorization:
            _write_status(
                authorization,
                authorization_identity,
                status="failed",
                detail="epsilon_stability_recovery_failed_closed",
                stage="failed",
                completed_arms=completed_arms,
                error=f"{type(error).__name__}: {error}",
            )
        if lock_created and authorization:
            write_json_report(
                Path(str(authorization["paths"]["recovery_lock"]))
                / "failure.json",
                {
                    "schema_version": 1,
                    "role": "generation_epsilon_stability_recovery_fail_closed_lock",
                    "status": "failed",
                    "pid": os.getpid(),
                    "error": f"{type(error).__name__}: {error}",
                    "process_signals_allowed": False,
                },
            )
        raise


def _add_authorization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recovery-project", type=Path, required=True)
    parser.add_argument("--expected-recovery-revision", required=True)
    parser.add_argument("--expected-recovery-tree", required=True)
    parser.add_argument("--expected-recovery-branch", required=True)
    parser.add_argument("--execution-project", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--user-authorization", type=Path, required=True)
    parser.add_argument("--expected-user-authorization-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument(
        "--expected-execution-authorization-sha256", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failed-lock", type=Path, required=True)
    parser.add_argument("--recovery-lock", type=Path, required=True)
    parser.add_argument("--recovery-status", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--controller-status", type=Path, required=True)
    parser.add_argument("--expected-controller-status-sha256", required=True)
    parser.add_argument("--controller-log", type=Path, required=True)
    parser.add_argument("--expected-controller-log-sha256", required=True)
    parser.add_argument("--controller-receipt", type=Path, required=True)
    parser.add_argument("--expected-controller-receipt-sha256", required=True)
    parser.add_argument("--failure-record", type=Path, required=True)
    parser.add_argument("--expected-failure-record-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument("--prelaunch-receipt", type=Path, required=True)
    parser.add_argument("--expected-prelaunch-receipt-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, validate, or run the single source-bound recovery from the "
            "pre-GPU JPEG real-reference failure."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    _add_authorization_arguments(authorize)
    authorize.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--authorization", type=Path, required=True)
    validate.add_argument("--expected-authorization-sha256", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--expected-authorization-sha256", required=True)
    run.add_argument("--required-idle-polls", type=int, default=3)
    run.add_argument("--idle-poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "authorize":
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = build_live_recovery_authorization(
            recovery_project=args.recovery_project,
            expected_recovery_revision=args.expected_recovery_revision,
            expected_recovery_tree=args.expected_recovery_tree,
            expected_recovery_branch=args.expected_recovery_branch,
            execution_project=args.execution_project,
            python=args.python,
            design_path=args.design,
            expected_design_sha256=args.expected_design_sha256,
            preparation_path=args.preparation,
            expected_preparation_sha256=args.expected_preparation_sha256,
            user_authorization_path=args.user_authorization,
            expected_user_authorization_sha256=(
                args.expected_user_authorization_sha256
            ),
            execution_authorization_path=args.execution_authorization,
            expected_execution_authorization_sha256=(
                args.expected_execution_authorization_sha256
            ),
            output_root=args.output_root,
            failed_lock=args.failed_lock,
            recovery_lock=args.recovery_lock,
            recovery_status=args.recovery_status,
            eval_cache=args.eval_cache,
            controller_status_path=args.controller_status,
            expected_controller_status_sha256=(
                args.expected_controller_status_sha256
            ),
            controller_log_path=args.controller_log,
            expected_controller_log_sha256=args.expected_controller_log_sha256,
            controller_receipt_path=args.controller_receipt,
            expected_controller_receipt_sha256=(
                args.expected_controller_receipt_sha256
            ),
            failure_record_path=args.failure_record,
            expected_failure_record_sha256=args.expected_failure_record_sha256,
            launch_receipt_path=args.launch_receipt,
            expected_launch_receipt_sha256=args.expected_launch_receipt_sha256,
            prelaunch_receipt_path=args.prelaunch_receipt,
            expected_prelaunch_receipt_sha256=(
                args.expected_prelaunch_receipt_sha256
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json_report(args.output, payload)
        print(f"wrote {args.output}")
        return
    if args.command == "validate":
        _, payload = replay_live_recovery_authorization(
            args.authorization,
            args.expected_authorization_sha256,
        )
        print(json.dumps(payload, sort_keys=True))
        return
    raise SystemExit(
        run_recovery(
            args.authorization,
            args.expected_authorization_sha256,
            required_idle_polls=args.required_idle_polls,
            idle_poll_seconds=args.idle_poll_seconds,
        )
    )


if __name__ == "__main__":
    main()
