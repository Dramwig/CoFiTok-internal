from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from cofitok.generation_pair import generation_pair_contract
from cofitok.reporting import file_sha256, write_json_report


_SHARED_SCHEDULES = (
    "rollout_consistency",
    "ema_teacher_consistency",
)
_SCHEDULE_PHASES = (
    "disabled",
    "inactive",
    "warmup",
    "full_scale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound diagnostic comparison of matched CoFiTok and "
            "dense training trajectories through an exact cutoff step."
        )
    )
    parser.add_argument("--cofitok-metrics", type=Path, required=True)
    parser.add_argument("--dense-metrics", type=Path, required=True)
    parser.add_argument("--cofitok-manifest", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--cutoff-step", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--cofitok-origin")
    parser.add_argument("--dense-origin")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help=(
            "Optionally write exact metrics-prefix snapshots into this directory and "
            "bind the report to those snapshots instead of the growing inputs."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": path.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _is_hex_digest(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_metrics_prefix(path: Path, *, cutoff_step: int) -> dict[str, Any]:
    if cutoff_step < 1:
        raise ValueError("cutoff_step must be positive")
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    prefix_parts: list[bytes] = []
    seen_after_cutoff = False
    for line_number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number} has an invalid step") from error
        if step <= cutoff_step:
            if seen_after_cutoff:
                raise ValueError(f"{path} contains a cutoff row after a later step")
            rows.append(row)
            prefix_parts.append(raw_line)
        else:
            seen_after_cutoff = True
    if not rows or int(rows[-1]["step"]) != cutoff_step:
        raise ValueError(f"{path} does not contain the exact cutoff step {cutoff_step}")
    prefix = b"".join(prefix_parts)
    return {
        "rows": rows,
        "raw_prefix": prefix,
        "identity": {
            "row_count": len(rows),
            "last_step": cutoff_step,
            "bytes": len(prefix),
            "sha256": hashlib.sha256(prefix).hexdigest(),
        },
        "observed_file": _source(path),
    }


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    label: str,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    git = manifest.get("git")
    if not isinstance(git, dict):
        raise ValueError(f"{label} manifest lacks Git identity")
    if git.get("revision") != expected_revision:
        raise ValueError(f"{label} manifest revision does not match the expected revision")
    if git.get("branch") != expected_branch:
        raise ValueError(f"{label} manifest branch does not match the expected branch")
    if git.get("dirty") is not False:
        raise ValueError(f"{label} manifest used a dirty tracked worktree")
    provenance = manifest.get("dataset_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") != "pass"
        or provenance.get("formal") is not True
    ):
        raise ValueError(f"{label} manifest lacks passing dataset provenance")
    dataset_identity = provenance.get("identity_sha256")
    runtime_identity = manifest.get("runtime_environment_sha256")
    if not _is_hex_digest(dataset_identity, length=64):
        raise ValueError(f"{label} manifest has an invalid dataset identity")
    if not _is_hex_digest(runtime_identity, length=64):
        raise ValueError(f"{label} manifest has an invalid runtime identity")
    parameters = int(manifest.get("parameter_count", 0))
    if parameters < 1:
        raise ValueError(f"{label} manifest lacks a positive parameter count")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} manifest lacks its resolved config")
    if config.get("data", {}).get("dataset") != provenance.get("dataset"):
        raise ValueError(f"{label} manifest config and provenance datasets differ")
    return {
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
            "dirty": False,
        },
        "dataset": provenance.get("dataset"),
        "dataset_identity_sha256": dataset_identity,
        "runtime_environment_sha256": runtime_identity,
        "parameter_count": parameters,
        "config": config,
    }


def _expected_steps(*, cutoff_step: int, log_interval: int) -> list[int]:
    if log_interval < 1:
        raise ValueError("log_interval must be positive")
    if cutoff_step % log_interval:
        raise ValueError("cutoff_step must be divisible by the matched log interval")
    return [1, *range(log_interval, cutoff_step + 1, log_interval)]


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    cutoff_step: int,
    log_interval: int,
    evaluation_interval: int,
    effective_batch: int,
) -> dict[str, Any]:
    steps = [int(row["step"]) for row in rows]
    expected = _expected_steps(cutoff_step=cutoff_step, log_interval=log_interval)
    if steps != expected:
        raise ValueError(f"{label} metrics do not follow the exact logging schedule")
    for row in rows:
        step = int(row["step"])
        if int(row.get("samples_seen", -1)) != step * effective_batch:
            raise ValueError(f"{label} samples_seen is invalid at step {step}")
        for field, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _finite_number(value, label=f"{label} step {step} {field}")
        for field in (
            "epsilon",
            "rollout_consistency",
            "rollout_consistency_scale",
            "ema_teacher_consistency",
            "ema_teacher_consistency_scale",
        ):
            _finite_number(row.get(field), label=f"{label} step {step} {field}")
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    expected_validation_steps = list(
        range(evaluation_interval, cutoff_step + 1, evaluation_interval)
    )
    if [int(row["step"]) for row in validation_rows] != expected_validation_steps:
        raise ValueError(f"{label} validation events do not follow the exact schedule")
    for event_index, row in enumerate(validation_rows):
        step = int(row["step"])
        mse = _finite_number(
            row.get("validation_epsilon_mse"),
            label=f"{label} step {step} validation_epsilon_mse",
        )
        if mse < 0.0:
            raise ValueError(f"{label} validation MSE is negative at step {step}")
        expected_metadata = {
            "validation_event_index": event_index,
            "validation_batch_index": event_index,
        }
        for field, expected_value in expected_metadata.items():
            if int(row.get(field, -1)) != expected_value:
                raise ValueError(f"{label} {field} is invalid at step {step}")
        if int(row.get("validation_num_images", 0)) < 1:
            raise ValueError(f"{label} validation_num_images is invalid at step {step}")
        if int(row.get("validation_noise_seed", -1)) < 0:
            raise ValueError(f"{label} validation_noise_seed is invalid at step {step}")
    endpoint = rows[-1]
    return {
        "row_count": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "samples_seen": int(endpoint["samples_seen"]),
        "all_numeric_metrics_finite": True,
        "steps_exact": True,
        "samples_seen_exact": True,
        "validation_event_count": len(validation_rows),
        "validation_steps": expected_validation_steps,
        "validation_provenance_complete": True,
        "endpoint": {
            "epsilon": float(endpoint["epsilon"]),
            "rollout_consistency": float(endpoint["rollout_consistency"]),
            "rollout_consistency_scale": float(endpoint["rollout_consistency_scale"]),
        },
        "validation_rows": validation_rows,
    }


def _schedule_contract(config: dict[str, Any], *, label: str) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, dict):
        raise ValueError(f"{label} config lacks its loss schedule")
    result: dict[str, Any] = {}
    for name in _SHARED_SCHEDULES:
        weight = _finite_number(
            loss.get(f"{name}_weight"),
            label=f"{label} {name}_weight",
        )
        if weight < 0.0:
            raise ValueError(f"{label} {name}_weight must be nonnegative")
        try:
            start_step = int(loss[f"{name}_start_step"])
            warmup_steps = int(loss[f"{name}_warmup_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} {name} schedule is incomplete") from error
        if start_step < 0 or warmup_steps < 0:
            raise ValueError(f"{label} {name} schedule steps must be nonnegative")
        enabled = weight > 0.0
        result[name] = {
            "enabled": enabled,
            "weight": weight,
            "start_step": start_step,
            "warmup_steps": warmup_steps,
            "full_scale_step": start_step + warmup_steps if enabled else None,
        }
    return result


def _schedule_phase(step: int, contract: dict[str, Any]) -> str:
    if contract["enabled"] is not True:
        return "disabled"
    start_step = int(contract["start_step"])
    warmup_steps = int(contract["warmup_steps"])
    if step < start_step:
        return "inactive"
    if warmup_steps > 0 and step < start_step + warmup_steps:
        return "warmup"
    return "full_scale"


def _validate_observed_schedule_scale(
    value: Any,
    *,
    label: str,
    phase: str,
) -> float:
    scale = _finite_number(value, label=label)
    if scale < 0.0 or scale > 1.0:
        raise ValueError(f"{label} must be between zero and one")
    if phase in {"disabled", "inactive"} and scale != 0.0:
        raise ValueError(f"{label} must be zero during the {phase} phase")
    if phase == "warmup" and scale >= 1.0:
        raise ValueError(f"{label} must remain below one during warmup")
    if phase == "full_scale" and not math.isclose(scale, 1.0, abs_tol=1e-8):
        raise ValueError(f"{label} must equal one during the full-scale phase")
    return scale


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot summarize an empty validation regime")
    cofitok_values = [event["cofitok_epsilon_mse"] for event in events]
    dense_values = [event["dense_epsilon_mse"] for event in events]
    deltas = [event["cofitok_minus_dense"] for event in events]
    relative_deltas = [
        event["relative_delta"]
        for event in events
        if event["relative_delta"] is not None
    ]
    cofitok_mean = mean(cofitok_values)
    dense_mean = mean(dense_values)
    return {
        "event_count": len(events),
        "cofitok_mean_epsilon_mse": cofitok_mean,
        "dense_mean_epsilon_mse": dense_mean,
        "mean_cofitok_minus_dense": mean(deltas),
        "relative_delta_of_means": (
            (cofitok_mean - dense_mean) / dense_mean if dense_mean != 0.0 else None
        ),
        "mean_absolute_delta": mean(abs(delta) for delta in deltas),
        "median_absolute_delta": median(abs(delta) for delta in deltas),
        "max_absolute_relative_delta": (
            max(abs(value) for value in relative_deltas) if relative_deltas else None
        ),
        "cofitok_lower_event_count": sum(delta < 0.0 for delta in deltas),
        "dense_lower_event_count": sum(delta > 0.0 for delta in deltas),
        "tie_event_count": sum(delta == 0.0 for delta in deltas),
        "endpoint_relative_delta": events[-1]["relative_delta"],
    }


def _schedule_regime_summaries(
    events: list[dict[str, Any]],
    *,
    contracts: dict[str, Any],
) -> dict[str, Any]:
    by_schedule: dict[str, Any] = {}
    for name, contract in contracts.items():
        phase_summaries = []
        for phase in _SCHEDULE_PHASES:
            selected = [
                event
                for event in events
                if event["schedule_phases"][name] == phase
            ]
            if not selected:
                continue
            phase_summaries.append(
                {
                    "phase": phase,
                    "first_step": selected[0]["step"],
                    "last_step": selected[-1]["step"],
                    "event_steps": [event["step"] for event in selected],
                    "summary": _event_summary(selected),
                }
            )
        by_schedule[name] = phase_summaries
    return {
        "basis": "predeclared shared loss schedules from both resolved manifests",
        "contracts": contracts,
        "by_schedule": by_schedule,
        "individual_regime_significance_claim_allowed": False,
    }


def _paired_validation(
    cofitok_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    *,
    schedule_contracts: dict[str, Any],
) -> dict[str, Any]:
    if len(cofitok_rows) != len(dense_rows):
        raise ValueError("matched validation event counts differ")
    events: list[dict[str, Any]] = []
    for cofitok, dense in zip(cofitok_rows, dense_rows, strict=True):
        step = int(cofitok["step"])
        metadata_fields = (
            "step",
            "validation_event_index",
            "validation_batch_index",
            "validation_num_images",
            "validation_noise_seed",
        )
        if any(cofitok.get(field) != dense.get(field) for field in metadata_fields):
            raise ValueError(f"validation provenance differs between methods at step {step}")
        cofitok_mse = float(cofitok["validation_epsilon_mse"])
        dense_mse = float(dense["validation_epsilon_mse"])
        delta = cofitok_mse - dense_mse
        relative = delta / dense_mse if dense_mse != 0.0 else None
        schedule_phases = {
            name: _schedule_phase(step, contract)
            for name, contract in schedule_contracts.items()
        }
        observed_schedule_scales: dict[str, float] = {}
        for name, phase in schedule_phases.items():
            field = f"{name}_scale"
            cofitok_scale = _validate_observed_schedule_scale(
                cofitok.get(field),
                label=f"cofitok step {step} {field}",
                phase=phase,
            )
            dense_scale = _validate_observed_schedule_scale(
                dense.get(field),
                label=f"dense_identity step {step} {field}",
                phase=phase,
            )
            if not math.isclose(cofitok_scale, dense_scale, abs_tol=1e-12):
                raise ValueError(f"matched {field} differs at step {step}")
            observed_schedule_scales[name] = cofitok_scale
        events.append(
            {
                "step": step,
                "validation_event_index": int(cofitok["validation_event_index"]),
                "validation_batch_index": int(cofitok["validation_batch_index"]),
                "validation_num_images": int(cofitok["validation_num_images"]),
                "validation_noise_seed": int(cofitok["validation_noise_seed"]),
                "cofitok_epsilon_mse": cofitok_mse,
                "dense_epsilon_mse": dense_mse,
                "cofitok_minus_dense": delta,
                "relative_delta": relative,
                "schedule_phases": schedule_phases,
                "observed_schedule_scales": observed_schedule_scales,
                "lower_mse": (
                    "cofitok" if delta < 0.0 else "dense_identity" if delta > 0.0 else "tie"
                ),
            }
        )
    return {
        "pairing_basis": (
            "same validation steps, event indices, batch indices, image counts, "
            "and fixed noise seed"
        ),
        "events": events,
        "summary": _event_summary(events),
        "schedule_regimes": _schedule_regime_summaries(
            events,
            contracts=schedule_contracts,
        ),
    }


def build_report(
    *,
    cofitok_metrics: dict[str, Any],
    dense_metrics: dict[str, Any],
    cofitok_manifest: dict[str, Any],
    dense_manifest: dict[str, Any],
    cutoff_step: int,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if not _is_hex_digest(expected_revision, length=40):
        raise ValueError("expected_revision must be a full 40-character revision")
    validated_cofitok = _validate_manifest(
        cofitok_manifest,
        label="cofitok",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    validated_dense = _validate_manifest(
        dense_manifest,
        label="dense_identity",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    for field in ("dataset", "dataset_identity_sha256", "runtime_environment_sha256"):
        if validated_cofitok[field] != validated_dense[field]:
            raise ValueError(f"matched manifests differ in {field}")
    contract = generation_pair_contract(
        validated_cofitok["config"], validated_dense["config"]
    )
    if contract["valid"] is not True:
        raise ValueError("training pair contract failed: " + "; ".join(contract["issues"]))
    cofitok_config = validated_cofitok["config"]
    dense_config = validated_dense["config"]
    schedule_contracts = _schedule_contract(cofitok_config, label="cofitok")
    dense_schedule_contracts = _schedule_contract(
        dense_config,
        label="dense_identity",
    )
    if schedule_contracts != dense_schedule_contracts:
        raise ValueError("matched manifests differ in shared loss schedules")
    optimization = cofitok_config["optimization"]
    data = cofitok_config["data"]
    runtime = cofitok_config["runtime"]
    effective_batch = int(data["batch_size"]) * int(
        optimization["gradient_accumulation_steps"]
    )
    log_interval = int(optimization["log_interval"])
    evaluation_interval = int(runtime["evaluation_interval"])
    if effective_batch < 1 or evaluation_interval < 1:
        raise ValueError("matched effective batch and evaluation interval must be positive")
    cofitok_trajectory = _validate_rows(
        cofitok_metrics["rows"],
        label="cofitok",
        cutoff_step=cutoff_step,
        log_interval=log_interval,
        evaluation_interval=evaluation_interval,
        effective_batch=effective_batch,
    )
    dense_trajectory = _validate_rows(
        dense_metrics["rows"],
        label="dense_identity",
        cutoff_step=cutoff_step,
        log_interval=log_interval,
        evaluation_interval=evaluation_interval,
        effective_batch=effective_batch,
    )
    paired = _paired_validation(
        cofitok_trajectory.pop("validation_rows"),
        dense_trajectory.pop("validation_rows"),
        schedule_contracts=schedule_contracts,
    )
    dense_parameters = validated_dense["parameter_count"]
    parameter_gap = (
        validated_cofitok["parameter_count"] - dense_parameters
    ) / dense_parameters
    if abs(parameter_gap) > 0.02:
        raise ValueError("matched parameter gap exceeds 2%")
    return {
        "schema_version": 2,
        "status": "pass",
        "role": "matched_training_trajectory_diagnostic",
        "cutoff_step": cutoff_step,
        "images_seen_per_method": cutoff_step * effective_batch,
        "contract": {
            "git": validated_cofitok["git"],
            "dataset": validated_cofitok["dataset"],
            "dataset_identity_sha256": validated_cofitok["dataset_identity_sha256"],
            "runtime_environment_sha256": validated_cofitok[
                "runtime_environment_sha256"
            ],
            "effective_batch_size": effective_batch,
            "log_interval": log_interval,
            "evaluation_interval": evaluation_interval,
            "cofitok_parameter_count": validated_cofitok["parameter_count"],
            "dense_parameter_count": dense_parameters,
            "relative_parameter_gap": parameter_gap,
            "generation_pair_contract": contract,
        },
        "trajectories": {
            "cofitok": cofitok_trajectory,
            "dense_identity": dense_trajectory,
        },
        "paired_fixed_validation": paired,
        "comparison_policy": {
            "shared_primary_training_epsilon_reported_descriptively": True,
            "shared_rollout_consistency_reported_descriptively": True,
            "total_loss_comparison_allowed": False,
            "training_wall_clock_comparison_allowed": False,
            "reason": (
                "CoFiTok total loss includes factorization-only auxiliaries, and the "
                "dense run experienced externally observed GPU contention."
            ),
        },
        "claim_boundary": {
            "supports": [
                "matched resolved training contract through the cutoff",
                "source-bound fixed-validation epsilon trajectory through the cutoff",
                "predeclared fixed-validation summaries by shared schedule regime",
            ],
            "schedule_regime_quality_claim_allowed": False,
            "quality_claim_allowed": False,
            "formal_50k_gate_substitute": False,
            "promotion_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "sample_quality_metrics_present": False,
            "required_next_evidence": (
                "exact healthy matched 50K completion followed by formal EMA post-eval"
            ),
        },
    }


def main() -> None:
    args = _parse_args()
    cofitok_metrics = _read_metrics_prefix(
        args.cofitok_metrics, cutoff_step=args.cutoff_step
    )
    dense_metrics = _read_metrics_prefix(
        args.dense_metrics, cutoff_step=args.cutoff_step
    )
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_paths = {
            "cofitok": args.snapshot_dir
            / f"cofitok_train_metrics_through_step_{args.cutoff_step:08d}.jsonl",
            "dense": args.snapshot_dir
            / f"dense_train_metrics_through_step_{args.cutoff_step:08d}.jsonl",
        }
        snapshot_paths["cofitok"].write_bytes(cofitok_metrics["raw_prefix"])
        snapshot_paths["dense"].write_bytes(dense_metrics["raw_prefix"])
        cofitok_metrics = _read_metrics_prefix(
            snapshot_paths["cofitok"], cutoff_step=args.cutoff_step
        )
        dense_metrics = _read_metrics_prefix(
            snapshot_paths["dense"], cutoff_step=args.cutoff_step
        )
    cofitok_manifest = _read_object(args.cofitok_manifest)
    dense_manifest = _read_object(args.dense_manifest)
    report = build_report(
        cofitok_metrics=cofitok_metrics,
        dense_metrics=dense_metrics,
        cofitok_manifest=cofitok_manifest,
        dense_manifest=dense_manifest,
        cutoff_step=args.cutoff_step,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    builder_path = Path(__file__).resolve()
    report["builder"] = {
        "path": "scripts/build_generation_matched_training_trajectory.py",
        "bytes": builder_path.stat().st_size,
        "sha256": file_sha256(builder_path),
    }
    report["sources"] = {
        "cofitok_metrics": {
            "origin": args.cofitok_origin,
            "bound_prefix": cofitok_metrics["identity"],
            "observed_file": cofitok_metrics["observed_file"],
        },
        "dense_metrics": {
            "origin": args.dense_origin,
            "bound_prefix": dense_metrics["identity"],
            "observed_file": dense_metrics["observed_file"],
        },
        "cofitok_manifest": _source(args.cofitok_manifest),
        "dense_manifest": _source(args.dense_manifest),
    }
    write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "cutoff_step": report["cutoff_step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
