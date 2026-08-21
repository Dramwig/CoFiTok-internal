from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows unit-test import path
    fcntl = None

from cofitok.generation_cost import (
    resume_compute_adjustment_source_identity,
    training_cost_summary,
    verify_resume_compute_adjustment_source,
)
from cofitok.training.completion import validate_completed_generation_training
from scripts.build_generation_resume_compute_adjustment import (
    build_adjustment_report,
)
from scripts.validate_generation_training_pair import validate_training_pair


ROLE = "quality_bridge_runtime_compute_fairness_audit"
WAITER_ROLE = "quality_bridge_runtime_compute_fairness_waiter"
DEPLOYMENT_ROLE = "quality_bridge_runtime_compute_fairness_deployment_receipt"
EXPECTED_EFFECTIVE_BATCH = 64
EXPECTED_DATASET = "imagenet_256"
EXPECTED_RECIPE_STAGE = "stability_quality_bridge"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    os.replace(temporary, path)


def write_once_or_verify(
    path: Path,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    expected = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError(f"existing {label} is not byte-equivalent")
        return
    write_json_atomic(path, payload)


def git_value(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_checkout(
    checkout: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    revision = git_value(checkout, "rev-parse", "HEAD")
    tree = git_value(checkout, "rev-parse", "HEAD^{tree}")
    branch = git_value(checkout, "branch", "--show-current")
    tracked_status = git_value(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if revision != expected_revision:
        raise ValueError("auditor checkout revision differs")
    if tree != expected_tree:
        raise ValueError("auditor checkout tree differs")
    if branch != expected_branch:
        raise ValueError("auditor checkout branch differs")
    if tracked_status:
        raise ValueError("auditor checkout has tracked modifications")
    return {
        "path": checkout.resolve().as_posix(),
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def require_source(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if len(expected_sha256) != 64:
        raise ValueError(f"expected {label} SHA256 is malformed")
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return identity


def _verify_runbook_runtime_binding(runbook: Path) -> dict[str, Any]:
    source = runbook.read_text(encoding="utf-8")
    required_fragments = (
        "read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION",
        '--micro-batch-size "$SELECTED_MICRO_BATCH"',
        '--gradient-accumulation-steps "$SELECTED_ACCUMULATION"',
        '--expected-micro-batch-size "$SELECTED_MICRO_BATCH"',
        '--expected-gradient-accumulation-steps "$SELECTED_ACCUMULATION"',
        'train_to_milestone "$COFITOK_CONFIG" "$COFITOK_RUN"',
        'train_to_milestone "$DENSE_CONFIG" "$DENSE_RUN"',
        'require_complete "$COFITOK_RUN/training_report.json"',
        'require_complete "$DENSE_RUN/training_report.json"',
        "scripts/validate_generation_training_pair.py",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise ValueError(
            "quality bridge runbook lacks runtime binding fragments: "
            + "; ".join(missing)
        )
    if source.count("read -r SELECTED_MICRO_BATCH SELECTED_ACCUMULATION") != 1:
        raise ValueError("quality bridge runbook runtime selection is not read once")
    return {
        "selection_read_once": True,
        "shared_train_function_uses_selected_runtime": True,
        "both_methods_use_shared_train_function": True,
        "both_completion_reports_validate_selected_runtime": True,
        "post_completion_pair_validation_required": True,
    }


def validate_static_contract(
    *,
    launch_receipt: Path,
    expected_launch_receipt_sha256: str,
    config_validation: Path,
    expected_config_validation_sha256: str,
    runtime_selection: Path,
    expected_runtime_selection_sha256: str,
    active_runbook: Path,
    expected_active_runbook_sha256: str,
    cofitok_config: Path,
    dense_config: Path,
    expected_training_revision: str,
    expected_training_branch: str,
    expected_steps: int,
) -> dict[str, Any]:
    identities = {
        "launch_receipt": require_source(
            launch_receipt,
            expected_sha256=expected_launch_receipt_sha256,
            label="launch receipt",
        ),
        "config_validation": require_source(
            config_validation,
            expected_sha256=expected_config_validation_sha256,
            label="config validation",
        ),
        "runtime_selection": require_source(
            runtime_selection,
            expected_sha256=expected_runtime_selection_sha256,
            label="runtime selection",
        ),
        "active_runbook": require_source(
            active_runbook,
            expected_sha256=expected_active_runbook_sha256,
            label="active runbook",
        ),
        "cofitok_config": file_identity(cofitok_config),
        "dense_config": file_identity(dense_config),
    }
    launch = read_object(launch_receipt, label="launch receipt")
    validation = read_object(config_validation, label="config validation")
    runtime = read_object(runtime_selection, label="runtime selection")
    git = launch.get("git")
    git = git if isinstance(git, dict) else {}
    selected = launch.get("runtime_selection")
    selected = selected if isinstance(selected, dict) else {}
    selection = launch.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    micro_batch = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    effective_batch = int(selected.get("effective_batch_size", -1))
    if (
        launch.get("status") != "pass"
        or git.get("revision") != expected_training_revision
        or git.get("branch") != expected_training_branch
        or git.get("tracked_dirty") is not False
        or int(selection.get("steps", -1)) != expected_steps
        or selection.get("dataset") != EXPECTED_DATASET
    ):
        raise ValueError("quality bridge launch receipt identity differs")
    if (
        micro_batch < 1
        or accumulation < 1
        or effective_batch != EXPECTED_EFFECTIVE_BATCH
        or micro_batch * accumulation != effective_batch
    ):
        raise ValueError("quality bridge launch runtime is invalid")
    if validation.get("status") != "pass" or validation.get("mismatches") != []:
        raise ValueError("quality bridge config validation did not pass cleanly")
    relative_gap = float(validation.get("relative_parameter_gap", math.nan))
    cofitok_parameters = int(
        validation.get("cofitok", {}).get("parameter_count", -1)
    )
    dense_parameters = int(validation.get("dense", {}).get("parameter_count", -1))
    if (
        cofitok_parameters < 1
        or dense_parameters < 1
        or not math.isfinite(relative_gap)
        or abs(relative_gap) > 0.02
    ):
        raise ValueError("quality bridge parameter gap is invalid")
    runtime_environment_sha256 = str(
        selected.get("runtime_environment_sha256", "")
    )
    if len(runtime_environment_sha256) != 64:
        raise ValueError("quality bridge runtime environment SHA256 is malformed")
    selected_runtime = runtime.get("selected")
    selected_runtime = selected_runtime if isinstance(selected_runtime, dict) else {}
    if any(
        int(selected_runtime.get(key, -1)) != expected
        for key, expected in (
            ("micro_batch_size", micro_batch),
            ("gradient_accumulation_steps", accumulation),
            ("effective_batch_size", effective_batch),
        )
    ):
        raise ValueError("runtime selection and launch receipt differ")
    launch_sources = launch.get("source_reports")
    launch_sources = launch_sources if isinstance(launch_sources, dict) else {}
    expected_bound_sources = {
        "config_validation": identities["config_validation"],
        "runtime_selection": identities["runtime_selection"],
        "cofitok_config": identities["cofitok_config"],
        "dense_config": identities["dense_config"],
    }
    for name, expected in expected_bound_sources.items():
        if launch_sources.get(name) != expected:
            raise ValueError(f"launch receipt bound {name} differs")
    runbook_contract = _verify_runbook_runtime_binding(active_runbook)
    output_root = Path(str(launch.get("output_root", ""))).resolve()
    training_run_dirs = launch.get("training_run_dirs")
    if (
        not str(launch.get("output_root", "")).strip()
        or not isinstance(training_run_dirs, list)
        or len(training_run_dirs) != 2
        or any(
            not Path(str(path)).resolve().is_relative_to(output_root)
            for path in training_run_dirs
        )
    ):
        raise ValueError("quality bridge launch output paths are invalid")
    return {
        "status": "verified",
        "sources": identities,
        "training_git": {
            "revision": expected_training_revision,
            "branch": expected_training_branch,
            "tracked_dirty": False,
        },
        "dataset": EXPECTED_DATASET,
        "output_root": output_root.as_posix(),
        "training_run_dirs": [
            Path(str(path)).resolve().as_posix() for path in training_run_dirs
        ],
        "target_steps_per_method": expected_steps,
        "runtime": {
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
            "runtime_environment_sha256": selected.get(
                "runtime_environment_sha256"
            ),
        },
        "parameters": {
            "cofitok": cofitok_parameters,
            "dense_identity": dense_parameters,
            "relative_gap": relative_gap,
        },
        "runbook_contract": runbook_contract,
    }


def training_state(run_dir: Path, *, expected_steps: int) -> dict[str, Any]:
    latest_path = run_dir / "latest.json"
    report_path = run_dir / "training_report.json"
    latest_step = 0
    if latest_path.is_file():
        try:
            latest_step = int(
                read_object(latest_path, label="latest checkpoint pointer").get(
                    "step", 0
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            latest_step = -1
    if not report_path.is_file():
        return {
            "run_dir": run_dir.resolve().as_posix(),
            "report_exists": False,
            "training_complete": False,
            "completed_steps": latest_step,
            "target_steps": expected_steps,
            "ready": False,
        }
    report = read_object(report_path, label="training report")
    completed_steps = int(report.get("completed_steps", latest_step))
    target_steps = int(report.get("target_steps", -1))
    complete = report.get("training_complete") is True
    return {
        "run_dir": run_dir.resolve().as_posix(),
        "report_exists": True,
        "report": file_identity(report_path),
        "training_complete": complete,
        "completed_steps": completed_steps,
        "target_steps": target_steps,
        "ready": complete
        and completed_steps == expected_steps
        and target_steps == expected_steps,
    }


def _orphan_metrics(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.glob(
            "train_metrics_orphaned_at_resume_????????_????????????.jsonl"
        ),
        key=lambda path: path.as_posix(),
    )


def inspect_existing_adjustment_coverage(
    *,
    method: str,
    run_dir: Path,
    adjustment_path: Path,
) -> dict[str, Any]:
    discovered_identities = [file_identity(path) for path in _orphan_metrics(run_dir)]
    discovered = {
        _normalized_path(identity["path"]): identity
        for identity in discovered_identities
    }
    if not adjustment_path.is_file():
        return {
            "status": "absent",
            "valid": True,
            "method": method,
            "provided": False,
            "terminal_adjustment_required": bool(discovered),
            "adjustment": None,
            "discovered_orphan_archive_count": len(discovered),
            "covered_orphan_archive_count": 0,
            "discovered_orphan_archives": discovered_identities,
            "issues": [],
        }

    adjustment_identity = resume_compute_adjustment_source_identity(adjustment_path)
    adjustment = read_object(
        adjustment_path,
        label=f"{method} resume-compute adjustment",
    )
    events = adjustment.get("recovery_events")
    issues: list[str] = []
    covered: dict[str, dict[str, Any]] = {}
    observed_paths: set[str] = set()
    if not isinstance(events, list):
        issues.append("existing adjustment recovery events are malformed")
        events = []
    for event in events:
        if not isinstance(event, dict):
            issues.append("existing adjustment recovery event is malformed")
            continue
        identity = event.get("orphan_metrics")
        if not isinstance(identity, dict):
            issues.append("existing adjustment orphan identity is malformed")
            continue
        normalized = _normalized_path(identity.get("path"))
        if not normalized:
            issues.append("existing adjustment orphan path is missing")
            continue
        if normalized in observed_paths:
            issues.append("existing adjustment repeats an orphan metrics source")
            continue
        observed_paths.add(normalized)
        expected = discovered.get(normalized)
        if expected is None:
            continue
        try:
            bytes_match = int(identity.get("bytes", -1)) == int(expected["bytes"])
        except (TypeError, ValueError):
            bytes_match = False
        if not bytes_match or identity.get("sha256") != expected["sha256"]:
            issues.append("existing adjustment orphan identity differs from disk")
            continue
        covered[normalized] = identity

    missing = sorted(set(discovered) - set(covered))
    extra = sorted(observed_paths - set(discovered))
    if missing or extra:
        issues.append(
            "existing adjustment does not cover the current physical orphan archive set"
        )
    return {
        "status": "verified" if not issues else "stale",
        "valid": not issues,
        "method": method,
        "provided": True,
        "terminal_adjustment_required": bool(discovered),
        "adjustment": adjustment_identity,
        "discovered_orphan_archive_count": len(discovered),
        "covered_orphan_archive_count": len(covered),
        "discovered_orphan_archives": discovered_identities,
        "missing_orphan_archives": missing,
        "extra_orphan_archives": extra,
        "issues": issues,
    }


def require_current_adjustment_coverage(
    adjustment_states: dict[str, dict[str, Any]],
) -> None:
    stale_adjustments = sorted(
        method
        for method, state in adjustment_states.items()
        if state.get("valid") is not True
    )
    if stale_adjustments:
        raise ValueError(
            "existing resume-compute adjustment is stale for: "
            + ", ".join(stale_adjustments)
        )


def resolve_training_cost(
    *,
    method: str,
    report: dict[str, Any],
    run_dir: Path,
    adjustment_path: Path,
    expected_steps: int,
) -> dict[str, Any]:
    if Path(str(report.get("output_dir", ""))).resolve() != run_dir.resolve():
        raise ValueError(f"{method} training output directory differs")
    unadjusted = training_cost_summary(report)
    adjustment_state = unadjusted["resume_compute_adjustment"]
    adjustment_required = adjustment_state["required"] is True
    adjustment_identity = None
    adjustment_verification = None
    if adjustment_required:
        orphans = _orphan_metrics(run_dir)
        if not orphans:
            raise ValueError(f"{method} requires recovery adjustment without archives")
        if not adjustment_path.exists():
            report_payload = build_adjustment_report(
                canonical_metrics=run_dir / "train_metrics.jsonl",
                orphan_metrics=orphans,
                effective_batch=int(unadjusted["effective_batch_size"]),
                continuity_end_step=expected_steps,
            )
            write_json_atomic(adjustment_path, report_payload)
        adjustment_identity = resume_compute_adjustment_source_identity(
            adjustment_path
        )
        adjustment_verification = verify_resume_compute_adjustment_source(
            adjustment_identity,
            method=method,
            training_report=report,
        )
        cost = adjustment_verification["training_cost"]
    else:
        if adjustment_path.exists():
            raise ValueError(f"{method} has an unexpected recovery adjustment")
        cost = unadjusted
    if cost.get("valid") is not True:
        raise ValueError(f"{method} training cost is invalid")
    return {
        "status": "verified",
        "training_cost": cost,
        "resume_compute_adjustment": adjustment_identity,
        "resume_compute_adjustment_verification": adjustment_verification,
    }


def _relative_change(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("runtime comparison values are invalid")
    return (value - baseline) / baseline


def build_terminal_audit(
    *,
    static_contract: dict[str, Any],
    auditor_checkout: dict[str, Any],
    waiter_source: dict[str, Any],
    cofitok_config: Path,
    dense_config: Path,
    cofitok_run_dir: Path,
    dense_run_dir: Path,
    cofitok_adjustment_path: Path,
    dense_adjustment_path: Path,
    expected_steps: int,
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    runtime = static_contract["runtime"]
    completion_arguments = {
        "expected_steps": expected_steps,
        "expected_revision": expected_training_revision,
        "expected_branch": expected_training_branch,
        "expected_micro_batch_size": int(runtime["micro_batch_size"]),
        "expected_gradient_accumulation_steps": int(
            runtime["gradient_accumulation_steps"]
        ),
    }
    cofitok_report = validate_completed_generation_training(
        report_path=cofitok_run_dir / "training_report.json",
        config_path=cofitok_config,
        **completion_arguments,
    )
    dense_report = validate_completed_generation_training(
        report_path=dense_run_dir / "training_report.json",
        config_path=dense_config,
        **completion_arguments,
    )
    pair_validation = validate_training_pair(
        cofitok_report,
        dense_report,
        expected_steps=expected_steps,
        expected_revision=expected_training_revision,
        expected_branch=expected_training_branch,
        expected_dataset=EXPECTED_DATASET,
        expected_recipe_stage=EXPECTED_RECIPE_STAGE,
    )
    expected_parameters = static_contract["parameters"]
    if (
        int(cofitok_report.get("parameter_count", -1))
        != int(expected_parameters["cofitok"])
        or int(dense_report.get("parameter_count", -1))
        != int(expected_parameters["dense_identity"])
        or not math.isclose(
            float(pair_validation["relative_parameter_gap"]),
            float(expected_parameters["relative_gap"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("terminal parameter evidence differs from launch validation")
    costs = {
        "cofitok": resolve_training_cost(
            method="cofitok",
            report=cofitok_report,
            run_dir=cofitok_run_dir,
            adjustment_path=cofitok_adjustment_path,
            expected_steps=expected_steps,
        ),
        "dense_identity": resolve_training_cost(
            method="dense_identity",
            report=dense_report,
            run_dir=dense_run_dir,
            adjustment_path=dense_adjustment_path,
            expected_steps=expected_steps,
        ),
    }
    expected_samples = expected_steps * EXPECTED_EFFECTIVE_BATCH
    for method, evidence in costs.items():
        cost = evidence["training_cost"]
        if (
            int(cost["micro_batch_size"]) != int(runtime["micro_batch_size"])
            or int(cost["gradient_accumulation_steps"])
            != int(runtime["gradient_accumulation_steps"])
            or int(cost["effective_batch_size"]) != EXPECTED_EFFECTIVE_BATCH
            or int(cost["samples_seen"]) != expected_samples
        ):
            raise ValueError(f"{method} observed runtime differs from launch")
    cofitok_cost = costs["cofitok"]["training_cost"]
    dense_cost = costs["dense_identity"]["training_cost"]
    return {
        "schema_version": 1,
        "role": ROLE,
        "status": "pass",
        "scope": {
            "cpu_only": True,
            "checkpoint_payload_loading_allowed": False,
            "gpu_execution_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "auditor": {
            "checkout": auditor_checkout,
            "source": waiter_source,
        },
        "contract": static_contract,
        "terminal_sources": {
            "cofitok_training": file_identity(
                cofitok_run_dir / "training_report.json"
            ),
            "dense_training": file_identity(
                dense_run_dir / "training_report.json"
            ),
            "cofitok_latest": file_identity(cofitok_run_dir / "latest.json"),
            "dense_latest": file_identity(dense_run_dir / "latest.json"),
        },
        "pair_validation": pair_validation,
        "observed_runtime_parity": {
            "status": "proven",
            "micro_batch_size": int(runtime["micro_batch_size"]),
            "gradient_accumulation_steps": int(
                runtime["gradient_accumulation_steps"]
            ),
            "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
            "canonical_images_per_method": expected_samples,
        },
        "methods": costs,
        "descriptive_comparison": {
            "role": "measured_outcomes_not_predeclared_advantage",
            "cofitok_minus_dense_adjusted_elapsed_seconds": float(
                cofitok_cost["elapsed_seconds"]
            )
            - float(dense_cost["elapsed_seconds"]),
            "cofitok_adjusted_elapsed_relative_change": _relative_change(
                float(cofitok_cost["elapsed_seconds"]),
                float(dense_cost["elapsed_seconds"]),
            ),
            "cofitok_throughput_relative_change": _relative_change(
                float(cofitok_cost["images_per_second"]),
                float(dense_cost["images_per_second"]),
            ),
            "cofitok_peak_vram_relative_change": _relative_change(
                float(cofitok_cost["peak_vram_bytes"]),
                float(dense_cost["peak_vram_bytes"]),
            ),
        },
        "claim_boundary": {
            "matched_runtime_contract_established": True,
            "observed_pair_runtime_parity_established": True,
            "physical_recovery_compute_included": True,
            "training_speed_advantage_predeclared": False,
            "memory_advantage_predeclared": False,
            "sample_quality_established": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def build_deployment_receipt(
    *,
    static_contract: dict[str, Any],
    auditor_checkout: dict[str, Any],
    waiter_source: dict[str, Any],
    audit_output: Path,
    status_output: Path,
    cofitok_run_dir: Path,
    dense_run_dir: Path,
    cofitok_adjustment_path: Path,
    dense_adjustment_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "auditor": {
            "checkout": auditor_checkout,
            "source": waiter_source,
        },
        "contract_sources": static_contract["sources"],
        "expected_runtime": static_contract["runtime"],
        "targets": {
            "cofitok_run_dir": cofitok_run_dir.resolve().as_posix(),
            "dense_run_dir": dense_run_dir.resolve().as_posix(),
            "cofitok_resume_compute_adjustment": (
                cofitok_adjustment_path.resolve().as_posix()
            ),
            "dense_resume_compute_adjustment": (
                dense_adjustment_path.resolve().as_posix()
            ),
            "audit_output": audit_output.resolve().as_posix(),
            "status_output": status_output.resolve().as_posix(),
        },
        "scope": {
            "cpu_only_waiter": True,
            "may_create_missing_method_specific_adjustment": True,
            "may_create_terminal_fairness_audit": True,
            "checkpoint_payload_loading_allowed": False,
            "gpu_execution_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_or_release_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def validate_target_paths(
    *,
    static_contract: dict[str, Any],
    cofitok_run_dir: Path,
    dense_run_dir: Path,
    writable_paths: tuple[Path, ...],
) -> None:
    output_root = Path(static_contract["output_root"]).resolve()
    expected_runs = [
        Path(path).resolve() for path in static_contract["training_run_dirs"]
    ]
    actual_runs = [cofitok_run_dir.resolve(), dense_run_dir.resolve()]
    if actual_runs != expected_runs:
        raise ValueError("waiter run directories differ from launch receipt")
    for path in writable_paths:
        if not path.resolve().is_relative_to(output_root):
            raise ValueError("waiter output path escapes the quality bridge root")


def waiter_status(
    *,
    status: str,
    detail: str,
    started_at: str,
    deployment_receipt: Path,
    audit_output: Path,
    training_states: dict[str, Any],
    adjustment_states: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": WAITER_ROLE,
        "status": status,
        "detail": detail,
        "error": error,
        "started_at": started_at,
        "updated_at": utc_now(),
        "pid": os.getpid(),
        "deployment_receipt": (
            file_identity(deployment_receipt)
            if deployment_receipt.is_file()
            else None
        ),
        "audit_output": (
            file_identity(audit_output) if audit_output.is_file() else None
        ),
        "training": training_states,
        "resume_compute_adjustments": adjustment_states,
        "scope": {
            "cpu_only": True,
            "gpu_execution_allowed": False,
            "checkpoint_payload_loading_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the full-data quality bridge training pair and publish a "
            "source-bound runtime/physical-compute fairness audit."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--expected-config-validation-sha256", required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--expected-runtime-selection-sha256", required=True)
    parser.add_argument("--active-runbook", type=Path, required=True)
    parser.add_argument("--expected-active-runbook-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--cofitok-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, required=True)
    parser.add_argument(
        "--cofitok-resume-compute-adjustment", type=Path, required=True
    )
    parser.add_argument(
        "--dense-resume-compute-adjustment", type=Path, required=True
    )
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-steps", type=int, default=100000)
    parser.add_argument("--expected-auditor-revision", required=True)
    parser.add_argument("--expected-auditor-tree", required=True)
    parser.add_argument("--expected-auditor-branch", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2592000.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_steps < 1:
        raise ValueError("expected steps must be positive")
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("poll and timeout seconds must be positive")
    project = args.project.resolve()
    waiter_source = require_source(
        Path(__file__),
        expected_sha256=args.expected_source_sha256,
        label="waiter source",
    )
    auditor_checkout = verify_checkout(
        project,
        expected_revision=args.expected_auditor_revision,
        expected_tree=args.expected_auditor_tree,
        expected_branch=args.expected_auditor_branch,
    )
    static_arguments = {
        "launch_receipt": args.launch_receipt,
        "expected_launch_receipt_sha256": args.expected_launch_receipt_sha256,
        "config_validation": args.config_validation,
        "expected_config_validation_sha256": (
            args.expected_config_validation_sha256
        ),
        "runtime_selection": args.runtime_selection,
        "expected_runtime_selection_sha256": (
            args.expected_runtime_selection_sha256
        ),
        "active_runbook": args.active_runbook,
        "expected_active_runbook_sha256": args.expected_active_runbook_sha256,
        "cofitok_config": args.cofitok_config,
        "dense_config": args.dense_config,
        "expected_training_revision": args.expected_training_revision,
        "expected_training_branch": args.expected_training_branch,
        "expected_steps": args.expected_steps,
    }
    static_contract = validate_static_contract(**static_arguments)
    validate_target_paths(
        static_contract=static_contract,
        cofitok_run_dir=args.cofitok_run_dir,
        dense_run_dir=args.dense_run_dir,
        writable_paths=(
            args.cofitok_resume_compute_adjustment,
            args.dense_resume_compute_adjustment,
            args.audit_output,
            args.status_output,
            args.deployment_receipt_output,
        ),
    )
    deployment = build_deployment_receipt(
        static_contract=static_contract,
        auditor_checkout=auditor_checkout,
        waiter_source=waiter_source,
        audit_output=args.audit_output,
        status_output=args.status_output,
        cofitok_run_dir=args.cofitok_run_dir,
        dense_run_dir=args.dense_run_dir,
        cofitok_adjustment_path=args.cofitok_resume_compute_adjustment,
        dense_adjustment_path=args.dense_resume_compute_adjustment,
    )
    write_once_or_verify(
        args.deployment_receipt_output,
        deployment,
        label="runtime/compute fairness deployment receipt",
    )

    lock_path = args.status_output.with_suffix(args.status_output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("runtime/compute fairness waiter is already active")
            return

    started_at = utc_now()
    deadline = time.monotonic() + args.timeout_seconds
    states: dict[str, Any] = {}
    adjustment_states: dict[str, Any] = {}
    try:
        while True:
            static_contract = validate_static_contract(**static_arguments)
            states = {
                "cofitok": training_state(
                    args.cofitok_run_dir, expected_steps=args.expected_steps
                ),
                "dense_identity": training_state(
                    args.dense_run_dir, expected_steps=args.expected_steps
                ),
            }
            adjustment_states = {
                "cofitok": inspect_existing_adjustment_coverage(
                    method="cofitok",
                    run_dir=args.cofitok_run_dir,
                    adjustment_path=args.cofitok_resume_compute_adjustment,
                ),
                "dense_identity": inspect_existing_adjustment_coverage(
                    method="dense_identity",
                    run_dir=args.dense_run_dir,
                    adjustment_path=args.dense_resume_compute_adjustment,
                ),
            }
            require_current_adjustment_coverage(adjustment_states)
            if all(state["ready"] for state in states.values()):
                audit = build_terminal_audit(
                    static_contract=static_contract,
                    auditor_checkout=auditor_checkout,
                    waiter_source=waiter_source,
                    cofitok_config=args.cofitok_config,
                    dense_config=args.dense_config,
                    cofitok_run_dir=args.cofitok_run_dir,
                    dense_run_dir=args.dense_run_dir,
                    cofitok_adjustment_path=(
                        args.cofitok_resume_compute_adjustment
                    ),
                    dense_adjustment_path=args.dense_resume_compute_adjustment,
                    expected_steps=args.expected_steps,
                    expected_training_revision=args.expected_training_revision,
                    expected_training_branch=args.expected_training_branch,
                )
                write_once_or_verify(
                    args.audit_output,
                    audit,
                    label="runtime/compute fairness audit",
                )
                write_json_atomic(
                    args.status_output,
                    waiter_status(
                        status="pass",
                        detail="terminal_runtime_compute_fairness_verified",
                        started_at=started_at,
                        deployment_receipt=args.deployment_receipt_output,
                        audit_output=args.audit_output,
                        training_states=states,
                        adjustment_states=adjustment_states,
                    ),
                )
                print(args.audit_output)
                return

            write_json_atomic(
                args.status_output,
                waiter_status(
                    status="waiting",
                    detail="waiting_for_both_exact_100k_training_reports",
                    started_at=started_at,
                    deployment_receipt=args.deployment_receipt_output,
                    audit_output=args.audit_output,
                    training_states=states,
                    adjustment_states=adjustment_states,
                ),
            )
            if args.once:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("runtime/compute fairness waiter timed out")
            time.sleep(args.poll_seconds)
    except Exception as error:
        write_json_atomic(
            args.status_output,
            waiter_status(
                status="failed",
                detail="terminal_runtime_compute_fairness_failed",
                error=f"{type(error).__name__}: {error}",
                started_at=started_at,
                deployment_receipt=args.deployment_receipt_output,
                audit_output=args.audit_output,
                training_states=states,
                adjustment_states=adjustment_states,
            ),
        )
        raise


if __name__ == "__main__":
    main()
