from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from cofitok.generation_pair import generation_pair_contract
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_stability_50k_convergence_and_next_action_audit"


@dataclass(frozen=True)
class ConvergenceAuditContract:
    training_revision: str = "2c2c1f5166b73d4f28df93b276901671ac1a7836"
    training_branch: str = "scale/generation-stability-50k-preflight"
    dataset: str = "imagenet_256_10pct"
    dataset_identity_sha256: str = (
        "97cfec247a6991d3fcda6ff14bc75a89c07063836fd9cbe99fa58a41ab867741"
    )
    runtime_environment_sha256: str = (
        "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"
    )
    target_steps: int = 50_000
    effective_batch: int = 64
    promotion_gate_sha256: str = (
        "2f763913d5a07ba5563f5bc75548780678fcdf9da4e119e03262f10daf84dd90"
    )
    support_audit_sha256: str = (
        "6b3acfa62780b951b0adb45f817335b8086bafbed9dc7dcb90ba0afa07d0d9eb"
    )
    pair_summary_sha256: str = (
        "e04d7a503d16fa9bbdf5204d0bf4a0b0adbcfe4dc4e805c96931bffb3bbfa192"
    )
    cofitok_checkpoint_sha256: str = (
        "ec7b9a0981f1d45420a9a86cdb80339d6d87b87fa77891c234db3d1b84376c2a"
    )
    dense_checkpoint_sha256: str = (
        "325da25f9fd228ab224abd977da7f9e1d000ba36e3e8136b55edae9c9f47c716"
    )
    sampling_recovery_revision: str = "11d8f954030915f1d8848594683ac70518ce39bc"
    sampling_recovery_tree: str = "95a5e656b1e0a8b772ce5d1fe1badc487ef98d16"
    sampling_execution_branch: str = "scale/generation-large-capacity"
    quality_bridge_revision: str = "cf0e5faa94bf4ab38d947b921935b3b765b5537a"
    quality_bridge_tree: str = "6cef27723196fd363379bca2e7b85b1678ebd777"
    quality_bridge_branch: str = "scale/generation-stability-quality-bridge-100k"
    descriptive_flat_band: float = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CPU-only, source-bound convergence and next-evidence audit "
            "for the completed stability 50K CoFiTok/dense pair."
        )
    )
    parser.add_argument("--scaling-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--expected-audit-revision", required=True)
    parser.add_argument("--expected-audit-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _is_hex(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_contract(contract: ConvergenceAuditContract) -> None:
    for name in (
        "training_revision",
        "sampling_recovery_revision",
        "sampling_recovery_tree",
        "quality_bridge_revision",
        "quality_bridge_tree",
    ):
        if not _is_hex(getattr(contract, name), length=40):
            raise ValueError(f"convergence contract {name} is malformed")
    for name in (
        "dataset_identity_sha256",
        "runtime_environment_sha256",
        "promotion_gate_sha256",
        "support_audit_sha256",
        "pair_summary_sha256",
        "cofitok_checkpoint_sha256",
        "dense_checkpoint_sha256",
    ):
        if not _is_hex(getattr(contract, name), length=64):
            raise ValueError(f"convergence contract {name} is malformed")
    if contract.target_steps < 100 or contract.effective_batch < 1:
        raise ValueError("convergence contract step and batch budgets are invalid")
    if not 0.0 < contract.descriptive_flat_band < 1.0:
        raise ValueError("descriptive flat band must lie strictly inside (0, 1)")


def _json_source(path: Path, *, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    safe = reject_symlink_chain(path, name=name)
    if not safe.is_file():
        raise FileNotFoundError(f"{name} is missing: {safe}")
    return read_json_object(safe, name=name), file_identity(safe)


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_nested_finite(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_nested_finite(item, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_nested_finite(item, label=f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite(value, label=label)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    label: str,
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    git = manifest.get("git")
    if git != {
        "revision": contract.training_revision,
        "branch": contract.training_branch,
        "dirty": False,
    }:
        raise ValueError(f"{label} manifest Git identity differs")
    provenance = manifest.get("dataset_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") != "pass"
        or provenance.get("formal") is not True
        or provenance.get("dataset") != contract.dataset
        or provenance.get("identity_sha256") != contract.dataset_identity_sha256
    ):
        raise ValueError(f"{label} manifest dataset identity differs")
    if manifest.get("runtime_environment_sha256") != contract.runtime_environment_sha256:
        raise ValueError(f"{label} manifest runtime identity differs")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} manifest resolved config is missing")
    try:
        target_steps = int(config["runtime"]["steps"])
        evaluation_interval = int(config["runtime"]["evaluation_interval"])
        micro_batch = int(config["data"]["batch_size"])
        accumulation = int(config["optimization"]["gradient_accumulation_steps"])
        log_interval = int(config["optimization"]["log_interval"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} manifest runtime schedule is malformed") from error
    if target_steps != contract.target_steps:
        raise ValueError(f"{label} target step differs")
    if micro_batch * accumulation != contract.effective_batch:
        raise ValueError(f"{label} effective batch differs")
    if log_interval < 1 or evaluation_interval < 1:
        raise ValueError(f"{label} logging or validation interval is invalid")
    parameter_count = int(manifest.get("parameter_count", 0))
    if parameter_count < 1:
        raise ValueError(f"{label} parameter count is invalid")
    return {
        "config": config,
        "parameter_count": parameter_count,
        "log_interval": log_interval,
        "evaluation_interval": evaluation_interval,
        "micro_batch": micro_batch,
        "gradient_accumulation_steps": accumulation,
    }


def _resume_extra_steps(
    manifest: dict[str, Any],
    *,
    label: str,
    log_interval: int,
    target_steps: int,
) -> list[int]:
    resume = manifest.get("resume")
    reconciliation = manifest.get("metrics_resume_reconciliation")
    if resume is None and reconciliation is None:
        return []
    if not isinstance(resume, str) or not isinstance(reconciliation, dict):
        raise ValueError(f"{label} resume evidence is incomplete")
    match = re.search(r"checkpoint_step_(\d{8})\.pt$", resume)
    if match is None:
        raise ValueError(f"{label} resume checkpoint name is malformed")
    resume_step = int(match.group(1))
    if (
        reconciliation.get("schema_version") != 1
        or reconciliation.get("status") not in {"unchanged", "reconciled"}
        or int(reconciliation.get("resume_step", -1)) != resume_step
        or int(reconciliation.get("orphaned_rows", -1)) < 0
    ):
        raise ValueError(f"{label} metrics resume reconciliation differs")
    extra = resume_step + 1
    if extra > target_steps or extra % log_interval == 0:
        return []
    return [extra]


def _expected_steps(
    *,
    target_steps: int,
    log_interval: int,
    extra_steps: list[int],
) -> list[int]:
    if target_steps % log_interval:
        raise ValueError("target steps must be divisible by the log interval")
    return sorted({1, *range(log_interval, target_steps + 1, log_interval), *extra_steps})


def _read_metrics(
    path: Path,
    *,
    manifest: dict[str, Any],
    validated_manifest: dict[str, Any],
    label: str,
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    safe = reject_symlink_chain(path, name=f"{label} metrics")
    if not safe.is_file():
        raise FileNotFoundError(f"{label} metrics are missing: {safe}")
    raw = safe.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        if not raw_line.strip():
            raise ValueError(f"{label} metrics line {line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} metrics line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} metrics line {line_number} is not an object")
        _validate_nested_finite(row, label=f"{label}.metrics[{line_number}]")
        rows.append(row)
    extra_steps = _resume_extra_steps(
        manifest,
        label=label,
        log_interval=validated_manifest["log_interval"],
        target_steps=contract.target_steps,
    )
    expected_steps = _expected_steps(
        target_steps=contract.target_steps,
        log_interval=validated_manifest["log_interval"],
        extra_steps=extra_steps,
    )
    observed_steps: list[int] = []
    required_fields = (
        "epsilon",
        "total",
        "rollout_consistency",
        "rollout_consistency_scale",
        "ema_teacher_consistency",
        "ema_teacher_consistency_scale",
        "grad_norm",
        "learning_rate",
    )
    for row in rows:
        try:
            step = int(row["step"])
            samples_seen = int(row["samples_seen"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} metrics step accounting is malformed") from error
        observed_steps.append(step)
        if samples_seen != step * contract.effective_batch:
            raise ValueError(f"{label} samples_seen differs at step {step}")
        for field in required_fields:
            _finite(row.get(field), label=f"{label} step {step} {field}")
    if observed_steps != expected_steps:
        raise ValueError(f"{label} metrics do not follow the exact logging/resume schedule")
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    expected_validation_steps = list(
        range(
            validated_manifest["evaluation_interval"],
            contract.target_steps + 1,
            validated_manifest["evaluation_interval"],
        )
    )
    if [int(row["step"]) for row in validation_rows] != expected_validation_steps:
        raise ValueError(f"{label} validation events do not follow the exact schedule")
    for event_index, row in enumerate(validation_rows):
        step = int(row["step"])
        if (
            int(row.get("validation_event_index", -1)) != event_index
            or int(row.get("validation_batch_index", -1)) != event_index
            or int(row.get("validation_num_images", 0)) < 1
            or int(row.get("validation_noise_seed", -1)) < 0
            or _finite(
                row.get("validation_epsilon_mse"),
                label=f"{label} step {step} validation epsilon MSE",
            )
            < 0.0
        ):
            raise ValueError(f"{label} validation provenance differs at step {step}")
    return {
        "rows": rows,
        "validation_rows": validation_rows,
        "identity": file_identity(safe),
        "row_count": len(rows),
        "first_step": observed_steps[0],
        "last_step": observed_steps[-1],
        "resume_extra_steps": extra_steps,
        "all_numeric_metrics_finite": True,
        "steps_strict_and_exact": True,
        "samples_seen_exact": True,
        "validation_event_count": len(validation_rows),
    }


def _statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("metric summary window is empty")
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _linear_slope(
    rows: list[dict[str, Any]],
    *,
    field: str,
    low_exclusive: int,
    high_inclusive: int,
) -> dict[str, Any]:
    points = [
        (float(row["step"]), float(row[field]))
        for row in rows
        if low_exclusive < int(row["step"]) <= high_inclusive
    ]
    if len(points) < 2:
        raise ValueError(f"{field} slope window has fewer than two points")
    x_mean = mean(point[0] for point in points)
    y_mean = mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    return {
        "count": len(points),
        "low_exclusive": low_exclusive,
        "high_inclusive": high_inclusive,
        "window_mean": y_mean,
        "slope_per_1000_steps": slope * 1_000.0,
        "relative_change_over_window_at_mean": (
            slope * (high_inclusive - low_exclusive) / y_mean if y_mean else None
        ),
    }


def _training_windows(
    rows: list[dict[str, Any]],
    *,
    target_steps: int,
    flat_band: float,
) -> dict[str, Any]:
    def step_at(fraction: float) -> int:
        return int(round(target_steps * fraction))

    bounds = {
        "early_stable_10_20pct": (step_at(0.10), step_at(0.20)),
        "mid_40_50pct": (step_at(0.40), step_at(0.50)),
        "late_80_90pct": (step_at(0.80), step_at(0.90)),
        "late_90_100pct": (step_at(0.90), target_steps),
        "terminal_98_100pct": (step_at(0.98), target_steps),
    }
    fields = (
        "epsilon",
        "total",
        "rollout_consistency",
        "ema_teacher_consistency",
        "grad_norm",
        "learning_rate",
    )
    summaries: dict[str, Any] = {}
    for name, (low, high) in bounds.items():
        window_rows = [row for row in rows if low < int(row["step"]) <= high]
        summaries[name] = {
            "low_exclusive": low,
            "high_inclusive": high,
            "metrics": {
                field: _statistics([float(row[field]) for row in window_rows])
                for field in fields
            },
        }
    late_prior = summaries["late_80_90pct"]["metrics"]["epsilon"]["mean"]
    late_terminal = summaries["late_90_100pct"]["metrics"]["epsilon"]["mean"]
    mid = summaries["mid_40_50pct"]["metrics"]["epsilon"]["mean"]
    slope = _linear_slope(
        rows,
        field="epsilon",
        low_exclusive=step_at(0.80),
        high_inclusive=target_steps,
    )
    ratio = late_terminal / late_prior
    flat = (
        abs(ratio - 1.0) <= flat_band
        and abs(float(slope["relative_change_over_window_at_mean"])) <= flat_band
    )
    return {
        "windows": summaries,
        "epsilon_late_90_100_over_80_90": ratio,
        "epsilon_late_80_90_over_mid_40_50": late_prior / mid,
        "epsilon_last_20pct_linear_slope": slope,
        "post_hoc_descriptive_flat_signature": {
            "band": flat_band,
            "inside_band": flat,
            "thresholded_gate": False,
            "preregistered": False,
            "quality_claim_allowed": False,
        },
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired validation correlation requires matched vectors")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    if denominator == 0.0:
        raise ValueError("paired validation correlation is undefined")
    return numerator / denominator


def _paired_validation(
    cofitok_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(cofitok_rows) != len(dense_rows) or not cofitok_rows:
        raise ValueError("paired validation event count differs")
    events: list[dict[str, Any]] = []
    fields = (
        "step",
        "validation_event_index",
        "validation_batch_index",
        "validation_num_images",
        "validation_noise_seed",
    )
    for cofitok, dense in zip(cofitok_rows, dense_rows):
        if any(cofitok.get(field) != dense.get(field) for field in fields):
            raise ValueError("paired validation provenance differs")
        c_value = float(cofitok["validation_epsilon_mse"])
        d_value = float(dense["validation_epsilon_mse"])
        events.append(
            {
                **{field: cofitok[field] for field in fields},
                "cofitok_epsilon_mse": c_value,
                "dense_epsilon_mse": d_value,
                "cofitok_minus_dense": c_value - d_value,
            }
        )
    cofitok_values = [event["cofitok_epsilon_mse"] for event in events]
    dense_values = [event["dense_epsilon_mse"] for event in events]
    differences = [event["cofitok_minus_dense"] for event in events]
    return {
        "event_count": len(events),
        "all_step_seed_batch_and_count_metadata_equal": True,
        "cofitok_mean": mean(cofitok_values),
        "dense_mean": mean(dense_values),
        "ratio_of_means_delta": mean(cofitok_values) / mean(dense_values) - 1.0,
        "mean_difference_cofitok_minus_dense": mean(differences),
        "median_difference_cofitok_minus_dense": median(differences),
        "cofitok_lower_event_count": sum(value < 0.0 for value in differences),
        "dense_lower_event_count": sum(value > 0.0 for value in differences),
        "pearson_correlation": _pearson(cofitok_values, dense_values),
        "temporal_convergence_claim_allowed": False,
        "temporal_claim_rejection_reason": (
            "Each scheduled event advances to a different validation_batch_index; "
            "the stream is source-matched between methods but does not hold the "
            "validation batch constant across training time."
        ),
        "matched_method_trajectory_claim_allowed": True,
        "events": events,
    }


def _validate_run_evidence(
    run_dir: Path,
    *,
    label: str,
    manifest: dict[str, Any],
    expected_checkpoint_sha256: str,
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    training_report, training_identity = _json_source(
        run_dir / "training_report.json", name=f"{label} training report"
    )
    latest, latest_identity = _json_source(
        run_dir / "latest.json", name=f"{label} latest checkpoint pointer"
    )
    if (
        training_report.get("training_complete") is not True
        or int(training_report.get("completed_steps", -1)) != contract.target_steps
        or int(training_report.get("target_steps", -1)) != contract.target_steps
        or training_report.get("stop_requested") is not False
        or training_report.get("git") != manifest.get("git")
        or training_report.get("config") != manifest.get("config")
        or training_report.get("dataset_provenance")
        != manifest.get("dataset_provenance")
        or training_report.get("runtime_environment_sha256")
        != contract.runtime_environment_sha256
    ):
        raise ValueError(f"{label} terminal training report differs")
    final_metrics = training_report.get("final_metrics")
    if (
        not isinstance(final_metrics, dict)
        or int(final_metrics.get("step", -1)) != contract.target_steps
        or int(final_metrics.get("samples_seen", -1))
        != contract.target_steps * contract.effective_batch
    ):
        raise ValueError(f"{label} final training metrics differ")
    if training_report.get("latest_checkpoint") != latest:
        raise ValueError(f"{label} training report and latest pointer differ")
    if (
        int(latest.get("step", -1)) != contract.target_steps
        or latest.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError(f"{label} latest checkpoint identity differs")
    integrity_name = str(latest.get("integrity_manifest", ""))
    if Path(integrity_name).name != integrity_name or not integrity_name:
        raise ValueError(f"{label} latest integrity manifest name is invalid")
    integrity, integrity_identity = _json_source(
        run_dir / integrity_name, name=f"{label} checkpoint integrity sidecar"
    )
    for key, value in integrity.items():
        if latest.get(key) != value:
            raise ValueError(f"{label} latest pointer differs from sidecar field {key}")
    checkpoint_name = str(integrity.get("checkpoint", ""))
    if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name:
        raise ValueError(f"{label} checkpoint filename is invalid")
    checkpoint = reject_symlink_chain(
        run_dir / checkpoint_name, name=f"{label} physical checkpoint"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} physical checkpoint is missing")
    physical_bytes = checkpoint.stat().st_size
    if physical_bytes != int(integrity.get("checkpoint_bytes", -1)):
        raise ValueError(f"{label} physical checkpoint byte count differs")
    if (
        integrity.get("checkpoint_sha256") != expected_checkpoint_sha256
        or integrity.get("dataset_identity_sha256")
        != contract.dataset_identity_sha256
        or integrity.get("runtime_environment_sha256")
        != contract.runtime_environment_sha256
        or integrity.get("git_revision") != contract.training_revision
        or integrity.get("git_branch") != contract.training_branch
        or integrity.get("git_dirty") is not False
    ):
        raise ValueError(f"{label} checkpoint sidecar provenance differs")
    return {
        "training_report": training_identity,
        "latest": latest_identity,
        "integrity_sidecar": integrity_identity,
        "checkpoint": {
            "path": checkpoint.resolve().as_posix(),
            "bytes": physical_bytes,
            "sha256_from_required_integrity_sidecar": expected_checkpoint_sha256,
            "physical_checkpoint_rehashed_by_this_audit": False,
        },
        "elapsed_seconds": _finite(
            training_report.get("elapsed_seconds"), label=f"{label} elapsed seconds"
        ),
        "peak_vram_bytes": int(training_report.get("peak_vram_bytes", 0)),
        "final_learning_rate": _finite(
            final_metrics.get("learning_rate"), label=f"{label} final learning rate"
        ),
    }


def _validate_gate(
    gate: dict[str, Any],
    identity: dict[str, Any],
    *,
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    if identity["sha256"] != contract.promotion_gate_sha256:
        raise ValueError("frozen promotion gate SHA256 differs")
    failed = sorted(
        row.get("name")
        for row in gate.get("gates", [])
        if isinstance(row, dict) and row.get("passed") is False
    )
    if (
        gate.get("schema_version") != 2
        or gate.get("status") != "fail"
        or gate.get("decision") != "hold"
        or failed != ["absolute_fid_quality"]
    ):
        raise ValueError("frozen promotion gate terminal state differs")
    summary = gate.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("frozen promotion gate summary is missing")
    values = {
        name: _finite(summary.get(name), label=f"promotion gate {name}")
        for name in (
            "cofitok_fid",
            "dense_fid",
            "cofitok_precision",
            "dense_precision",
            "cofitok_recall",
            "dense_recall",
        )
    }
    return {
        "identity": identity,
        "status": "fail",
        "decision": "hold",
        "sole_failed_gate": "absolute_fid_quality",
        **values,
    }


def _validate_support_audit(
    support: dict[str, Any],
    identity: dict[str, Any],
    *,
    gate_identity: dict[str, Any],
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    if identity["sha256"] != contract.support_audit_sha256:
        raise ValueError("frozen sample-support audit SHA256 differs")
    boundary = support.get("claim_boundary")
    interpretation = support.get("interpretation_policy")
    if (
        support.get("schema_version") != 1
        or support.get("status") != "completed"
        or support.get("role") != "generation_frozen_existing_sample_support_audit"
        or not isinstance(boundary, dict)
        or boundary.get("causal_attribution_allowed") is not False
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("gpu_execution_authorized") is not False
        or boundary.get("new_sampling_performed") is not False
        or boundary.get("new_training_performed") is not False
        or not isinstance(interpretation, dict)
        or interpretation.get("thresholded_gate") is not False
    ):
        raise ValueError("frozen sample-support audit claim boundary differs")
    source_gate = support.get("sources", {}).get("promotion_gate")
    if source_gate != gate_identity:
        raise ValueError("sample-support audit binds another promotion gate")
    statistics = support.get("support_statistics")
    contrasts = support.get("contrasts_to_real")
    matched = support.get("matched_method_pair")
    formal = interpretation.get("formal_distribution_support_remains_authoritative")
    if not all(isinstance(value, dict) for value in (statistics, contrasts, matched, formal)):
        raise ValueError("sample-support audit evidence is incomplete")
    duplicate_counts: dict[str, Any] = {}
    for method in ("real_matched_subset", "cofitok", "dense_identity"):
        row = statistics.get(method)
        if not isinstance(row, dict):
            raise ValueError(f"sample-support audit {method} row is missing")
        duplicate_counts[method] = {
            "decoded_pixel_duplicate_count": int(
                row.get("decoded_pixel_duplicate_count", -1)
            ),
            "dhash_duplicate_count": int(row.get("dhash_duplicate_count", -1)),
        }
        if min(duplicate_counts[method].values()) < 0:
            raise ValueError(f"sample-support audit {method} duplicate count differs")
    return {
        "identity": identity,
        "duplicate_counts": duplicate_counts,
        "formal_distribution_support": dict(formal),
        "contrasts_to_real": {
            method: dict(contrasts[method]) for method in ("cofitok", "dense_identity")
        },
        "matched_method_pair": {
            "pairing": matched.get("pairing"),
            "paired_lowres_rms_mean": matched.get("paired_lowres_rms", {}).get("mean"),
            "paired_lowres_rms_over_cofitok_same_class_rms": matched.get(
                "paired_lowres_rms_over_cofitok_same_class_rms"
            ),
            "paired_lowres_rms_over_dense_same_class_rms": matched.get(
                "paired_lowres_rms_over_dense_same_class_rms"
            ),
        },
        "causal_attribution_allowed": False,
    }


def _validate_terminal_chain(
    scaling_root: Path,
    full_root: Path,
    *,
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    pair_monitor, pair_monitor_identity = _json_source(
        scaling_root / "pair_monitor.json", name="stability pair monitor"
    )
    pair_summary, pair_summary_identity = _json_source(
        scaling_root / "reports/pair_summary.json", name="stability pair summary"
    )
    dense_recovery, dense_recovery_identity = _json_source(
        scaling_root / "reports/dense_recovery_status.json",
        name="dense recovery status",
    )
    posteval, posteval_identity = _json_source(
        scaling_root / "reports/posteval_waiter.json", name="posteval waiter"
    )
    supplemental, supplemental_identity = _json_source(
        scaling_root / "reports/frozen_posteval_supplemental/supplemental_waiter.json",
        name="supplemental waiter",
    )
    readiness, readiness_identity = _json_source(
        full_root / "reports/readiness_waiter.json", name="full readiness waiter"
    )
    if pair_monitor.get("status") != "pass" or pair_monitor.get("stage") != "complete":
        raise ValueError("stability pair monitor is not terminal pass")
    if (
        pair_summary_identity["sha256"] != contract.pair_summary_sha256
        or pair_summary.get("status") != "completed"
        or int(pair_summary.get("completed_steps_per_method", -1))
        != contract.target_steps
        or int(pair_summary.get("images_seen_per_method", -1))
        != contract.target_steps * contract.effective_batch
        or pair_summary.get("formal_300k_authorization_allowed") is not False
    ):
        raise ValueError("stability pair summary differs")
    if (
        dense_recovery.get("status") != "pass"
        or dense_recovery.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("dense recovery terminal status differs")
    if (
        posteval.get("status") != "pass"
        or posteval.get("detail") != "formal_ema_postevaluation_completed"
        or int(posteval.get("child_exit_code", -1)) != 0
    ):
        raise ValueError("formal posteval terminal status differs")
    if (
        supplemental.get("status") != "failed"
        or "post-evaluation status is stale" not in str(supplemental.get("detail", ""))
        or supplemental.get("supplemental_non_authorizing") is not True
        or supplemental.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("supplemental waiter terminal status differs")
    if (
        readiness.get("status") != "failed"
        or "did not authorize" not in str(readiness.get("detail", ""))
        or readiness.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("full readiness waiter terminal status differs")
    supplemental_qualification = (
        scaling_root
        / "reports/frozen_posteval_supplemental/supplemental_qualification.json"
    )
    class_fidelity = (
        scaling_root / "reports/frozen_posteval_class_fidelity/qualification_report.json"
    )
    return {
        "pair_monitor": {"identity": pair_monitor_identity, "status": "pass"},
        "pair_summary": {"identity": pair_summary_identity, "status": "completed"},
        "dense_recovery": {"identity": dense_recovery_identity, "status": "pass"},
        "posteval": {
            "identity": posteval_identity,
            "status": "pass",
            "detail": "formal_ema_postevaluation_completed",
        },
        "supplemental": {
            "identity": supplemental_identity,
            "status": "failed",
            "detail": supplemental["detail"],
            "coordination_failure_inference": (
                "The posteval source is terminal pass while the waiter failed only "
                "its stale-status heartbeat check; this is not model-quality evidence."
            ),
            "supplemental_qualification_present": supplemental_qualification.exists(),
        },
        "class_fidelity": {
            "qualification_present": class_fidelity.exists(),
            "quality_pass_claim_allowed": False,
        },
        "readiness": {
            "identity": readiness_identity,
            "status": "failed",
            "detail": readiness["detail"],
            "full_training_launch_allowed": False,
        },
    }


def _validate_next_action_sources(
    project_root: Path,
    *,
    contract: ConvergenceAuditContract,
) -> dict[str, Any]:
    plan_path = (
        project_root
        / "configs/generation/diagnostics/stability_50k_sampling_recovery_v1.json"
    )
    sampling_receipt_path = (
        project_root
        / "artifacts/reports/generation/"
        "sampling_confirmation_distribution_support_rehearsal_2026-08-05.json"
    )
    bridge_receipt_path = (
        project_root
        / "artifacts/reports/generation/"
        "stability_full_data_quality_bridge_100k_linux_rehearsal_2026-08-05/"
        "rehearsal_summary.json"
    )
    plan, plan_identity = _json_source(plan_path, name="sampling recovery plan")
    sampling_receipt, sampling_receipt_identity = _json_source(
        sampling_receipt_path, name="sampling recovery Linux rehearsal receipt"
    )
    bridge_receipt, bridge_receipt_identity = _json_source(
        bridge_receipt_path, name="quality bridge Linux rehearsal receipt"
    )
    diagnostic = plan.get("diagnostic")
    confirmation = plan.get("confirmation")
    selection = plan.get("selection_policy")
    boundary = plan.get("claim_boundary")
    if (
        plan.get("schema_version") != 1
        or not isinstance(diagnostic, dict)
        or int(diagnostic.get("sample_count_per_case", -1)) != 1_000
        or int(diagnostic.get("class_count", -1)) != 1_000
        or diagnostic.get("balanced_modulo_exact_coverage_required") is not True
        or len(diagnostic.get("cases", [])) != 5
        or not isinstance(confirmation, dict)
        or int(confirmation.get("sample_count", -1)) != 10_000
        or not isinstance(selection, dict)
        or selection.get("eligibility") != "strict_fid_improvement_for_both_methods"
        or not isinstance(boundary, dict)
        or boundary.get("full_training_launch_allowed") is not False
        or boundary.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("sampling recovery plan contract differs")
    candidate = sampling_receipt.get("candidate")
    execution_boundary = sampling_receipt.get("execution_boundary")
    if (
        sampling_receipt.get("status") != "pass"
        or not isinstance(candidate, dict)
        or candidate.get("revision") != contract.sampling_recovery_revision
        or candidate.get("tree") != contract.sampling_recovery_tree
        or candidate.get("execution_branch") != contract.sampling_execution_branch
        or not isinstance(execution_boundary, dict)
        or execution_boundary.get("sampling_recovery_executed") is not False
        or execution_boundary.get("sampling_confirmation_executed") is not False
        or execution_boundary.get("training_executed") is not False
        or execution_boundary.get("full_300k_authorized") is not False
    ):
        raise ValueError("sampling recovery rehearsal receipt differs")
    target = bridge_receipt.get("execution_target")
    bridge_boundary = bridge_receipt.get("safety_boundary")
    if (
        bridge_receipt.get("status") != "pass"
        or not isinstance(target, dict)
        or target.get("revision") != contract.quality_bridge_revision
        or target.get("tree") != contract.quality_bridge_tree
        or target.get("branch") != contract.quality_bridge_branch
        or not isinstance(bridge_boundary, dict)
        or bridge_boundary.get("quality_bridge_launched") is not False
        or bridge_boundary.get("full_training_launch_allowed") is not False
        or bridge_boundary.get("full_300k_launch_allowed") is not False
        or bridge_boundary.get("explicit_user_execution_approval_required") is not True
    ):
        raise ValueError("quality bridge rehearsal receipt differs")
    return {
        "sampling_recovery": {
            "plan": plan_identity,
            "rehearsal_receipt": sampling_receipt_identity,
            "execution_revision": contract.sampling_recovery_revision,
            "execution_branch": contract.sampling_execution_branch,
            "sample_count_per_case": 1_000,
            "case_count": 5,
            "method_count": 2,
            "total_diagnostic_images": 10_000,
            "exact_approval_text": (
                "Approve the non-authorizing matched 1000-sample "
                "sampling-recovery diagnostic only."
            ),
            "execution_authorized": False,
        },
        "quality_bridge": {
            "rehearsal_receipt": bridge_receipt_identity,
            "execution_revision": contract.quality_bridge_revision,
            "execution_branch": contract.quality_bridge_branch,
            "target_steps_per_method": 100_000,
            "dataset": "imagenet_256",
            "execution_authorized": False,
        },
    }


def build_convergence_audit(
    *,
    scaling_root: Path,
    full_root: Path,
    project_root: Path,
    audit_git: dict[str, Any],
    contract: ConvergenceAuditContract = ConvergenceAuditContract(),
) -> dict[str, Any]:
    _require_contract(contract)
    scaling_root = reject_symlink_chain(scaling_root, name="scaling root")
    full_root = reject_symlink_chain(full_root, name="full readiness root")
    if not scaling_root.is_dir() or not full_root.is_dir():
        raise FileNotFoundError("convergence audit roots are missing")
    if (
        not _is_hex(audit_git.get("revision"), length=40)
        or not str(audit_git.get("branch", ""))
        or audit_git.get("tracked_dirty") is not False
    ):
        raise ValueError("convergence audit Git identity must be exact and clean")
    run_specs = {
        "cofitok": {
            "directory": "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
            "checkpoint_sha256": contract.cofitok_checkpoint_sha256,
        },
        "dense_identity": {
            "directory": "dense_rollout_x0_u2_ema_teacher",
            "checkpoint_sha256": contract.dense_checkpoint_sha256,
        },
    }
    loaded: dict[str, Any] = {}
    for label, spec in run_specs.items():
        run_dir = scaling_root / spec["directory"]
        manifest, manifest_identity = _json_source(
            run_dir / "run_manifest.json", name=f"{label} run manifest"
        )
        validated_manifest = _validate_manifest(
            manifest, label=label, contract=contract
        )
        metrics = _read_metrics(
            run_dir / "train_metrics.jsonl",
            manifest=manifest,
            validated_manifest=validated_manifest,
            label=label,
            contract=contract,
        )
        loaded[label] = {
            "run_dir": run_dir,
            "manifest": manifest,
            "manifest_identity": manifest_identity,
            "validated_manifest": validated_manifest,
            "metrics": metrics,
            "terminal": _validate_run_evidence(
                run_dir,
                label=label,
                manifest=manifest,
                expected_checkpoint_sha256=spec["checkpoint_sha256"],
                contract=contract,
            ),
        }
    pair_contract = generation_pair_contract(
        loaded["cofitok"]["manifest"]["config"],
        loaded["dense_identity"]["manifest"]["config"],
    )
    if pair_contract.get("valid") is not True:
        raise ValueError("generation pair contract failed: " + "; ".join(pair_contract["issues"]))
    dense_parameters = loaded["dense_identity"]["validated_manifest"]["parameter_count"]
    parameter_gap = (
        loaded["cofitok"]["validated_manifest"]["parameter_count"]
        - dense_parameters
    ) / dense_parameters
    if abs(parameter_gap) > 0.02:
        raise ValueError("matched parameter gap exceeds 2 percent")
    paired_validation = _paired_validation(
        loaded["cofitok"]["metrics"]["validation_rows"],
        loaded["dense_identity"]["metrics"]["validation_rows"],
    )
    training = {}
    for label in ("cofitok", "dense_identity"):
        metrics = loaded[label]["metrics"]
        training[label] = {
            "manifest": loaded[label]["manifest_identity"],
            "metrics": metrics["identity"],
            "row_integrity": {
                key: metrics[key]
                for key in (
                    "row_count",
                    "first_step",
                    "last_step",
                    "resume_extra_steps",
                    "all_numeric_metrics_finite",
                    "steps_strict_and_exact",
                    "samples_seen_exact",
                    "validation_event_count",
                )
            },
            "training_windows": _training_windows(
                metrics["rows"],
                target_steps=contract.target_steps,
                flat_band=contract.descriptive_flat_band,
            ),
            "terminal": loaded[label]["terminal"],
        }
    gate, gate_identity = _json_source(
        scaling_root / "reports/promotion_gate.json", name="frozen promotion gate"
    )
    frozen_quality = _validate_gate(gate, gate_identity, contract=contract)
    support, support_identity = _json_source(
        scaling_root
        / "reports/frozen_existing_sample_support_audit/support_audit.json",
        name="frozen existing-sample support audit",
    )
    support_summary = _validate_support_audit(
        support,
        support_identity,
        gate_identity=gate_identity,
        contract=contract,
    )
    terminal_chain = _validate_terminal_chain(
        scaling_root, full_root, contract=contract
    )
    next_sources = _validate_next_action_sources(project_root, contract=contract)
    both_flat = all(
        training[label]["training_windows"]["post_hoc_descriptive_flat_signature"][
            "inside_band"
        ]
        for label in ("cofitok", "dense_identity")
    )
    duplicate_collapse_rejected = all(
        row["decoded_pixel_duplicate_count"] == 0
        and row["dhash_duplicate_count"] == 0
        for row in support_summary["duplicate_counts"].values()
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass",
        "role": REPORT_ROLE,
        "source_profile": "stability_scaling",
        "audit_git": dict(audit_git),
        "contract": {
            "training_revision": contract.training_revision,
            "training_branch": contract.training_branch,
            "dataset": contract.dataset,
            "dataset_identity_sha256": contract.dataset_identity_sha256,
            "runtime_environment_sha256": contract.runtime_environment_sha256,
            "target_steps": contract.target_steps,
            "effective_batch": contract.effective_batch,
            "images_seen_per_method": contract.target_steps * contract.effective_batch,
            "cofitok_parameter_count": loaded["cofitok"]["validated_manifest"][
                "parameter_count"
            ],
            "dense_parameter_count": dense_parameters,
            "relative_parameter_gap": parameter_gap,
            "generation_pair_contract": pair_contract,
        },
        "training": training,
        "paired_validation": paired_validation,
        "frozen_quality": frozen_quality,
        "frozen_sample_support": support_summary,
        "terminal_evidence_chain": terminal_chain,
        "static_next_action_sources": next_sources,
        "evidence_synthesis": {
            "facts": {
                "matched_primary_validation_correlation": paired_validation[
                    "pearson_correlation"
                ],
                "matched_primary_validation_ratio_of_means_delta": paired_validation[
                    "ratio_of_means_delta"
                ],
                "both_methods_last_20pct_inside_post_hoc_5pct_flat_band": both_flat,
                "frozen_duplicate_collapse_rejected": duplicate_collapse_rejected,
                "both_methods_formal_recall_below_0_01": (
                    frozen_quality["cofitok_recall"] < 0.01
                    and frozen_quality["dense_recall"] < 0.01
                ),
                "cofitok_fid_better_than_dense": (
                    frozen_quality["cofitok_fid"] < frozen_quality["dense_fid"]
                ),
            },
            "supported_interpretations": [
                "The 50K pair is finite, source-matched, and checkpoint-auditable.",
                "The shared primary denoising trajectory does not show a CoFiTok-only optimization failure.",
                "The frozen sample failure is shared distribution-support collapse rather than decoded duplicate collapse.",
                "The late training epsilon stream is descriptively flat under a post-hoc 5% band while the cosine schedule is at its floor.",
            ],
            "unsupported_interpretations": [
                "The changing validation-batch stream proves temporal convergence.",
                "The existing evidence causally separates sampling-policy failure from checkpoint undertraining.",
                "The supplemental stale-status failure is model-quality evidence.",
                "A CoFiTok-specific factorization collapse has been established.",
            ],
            "decision": "sampling_recovery_first_then_full_data_scale_if_not_recovered",
            "ordered_next_evidence": [
                {
                    "order": 1,
                    "stage": "matched_1000_sample_per_case_sampling_recovery",
                    "reason": (
                        "It is the smallest source-matched experiment that can test "
                        "whether CFG/rescale policy explains the shared frozen support collapse."
                    ),
                    "execution_revision": contract.sampling_recovery_revision,
                    "execution_authorized": False,
                    "requires_exact_user_approval": True,
                    "requires_idle_gpu": True,
                },
                {
                    "order": 2,
                    "stage": "independent_matched_10000_sampling_confirmation",
                    "condition": (
                        "Only if one shared non-baseline case strictly improves both "
                        "methods in stage 1."
                    ),
                    "execution_authorized": False,
                    "requires_separate_exact_user_approval": True,
                },
                {
                    "order": 3,
                    "stage": "fresh_full_data_matched_100k_quality_bridge",
                    "condition": (
                        "Use when stage 1 finds no shared recovery, or stage 2 fails "
                        "full FID and distribution-support confirmation."
                    ),
                    "execution_revision": contract.quality_bridge_revision,
                    "execution_authorized": False,
                    "requires_exact_user_approval": True,
                    "requires_idle_gpu": True,
                },
            ],
        },
        "claim_boundary": {
            "read_only_source_audit": True,
            "gpu_work_performed": False,
            "new_sampling_performed": False,
            "new_training_performed": False,
            "causal_attribution_allowed": False,
            "report_is_promotion_gate": False,
            "sampling_execution_allowed": False,
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
            "new_gate_required": True,
        },
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"convergence audit output already exists: {args.output}")
    audit_git = git_provenance(PROJECT_ROOT)
    if audit_git != {
        "revision": args.expected_audit_revision,
        "branch": args.expected_audit_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("convergence audit checkout identity differs")
    report = build_convergence_audit(
        scaling_root=args.scaling_root,
        full_root=args.full_root,
        project_root=PROJECT_ROOT,
        audit_git=audit_git,
    )
    builder = Path(__file__).resolve()
    report["builder"] = {
        "path": "scripts/build_generation_stability_50k_convergence_audit.py",
        "bytes": builder.stat().st_size,
        "sha256": file_sha256(builder),
    }
    write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "status": report["status"],
                "decision": report["evidence_synthesis"]["decision"],
                "full_training_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
