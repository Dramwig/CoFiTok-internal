from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from cofitok.generation_pair import generation_pair_contract
from cofitok.reporting import file_sha256, write_json_report
from scripts import build_generation_matched_training_trajectory as trajectory


EXPECTED_GAMMA = 5.0
AUTHORIZATION_BOUNDARY = {
    "continuation_beyond_50000_allowed": False,
    "evaluation_launch_allowed": False,
    "export_allowed": False,
    "full_300k_launch_allowed": False,
    "full_training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "inference_export_allowed": False,
    "process_signals_allowed": False,
    "promotion_allowed": False,
    "release_allowed": False,
    "sampling_launch_allowed": False,
    "training_launch_allowed": False,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound, non-authorizing Min-SNR fixed-validation "
            "early-warning report."
        )
    )
    parser.add_argument("--gamma5-metrics", type=Path, required=True)
    parser.add_argument("--gamma5-manifest", type=Path, required=True)
    parser.add_argument("--gamma0-cofitok-metrics", type=Path, required=True)
    parser.add_argument("--gamma0-cofitok-report", type=Path, required=True)
    parser.add_argument("--gamma0-dense-metrics", type=Path, required=True)
    parser.add_argument("--gamma0-dense-report", type=Path, required=True)
    parser.add_argument("--cutoff-step", type=int, required=True)
    parser.add_argument("--expected-gamma5-revision", required=True)
    parser.add_argument("--expected-gamma5-branch", required=True)
    parser.add_argument("--expected-gamma0-revision", required=True)
    parser.add_argument("--expected-gamma0-branch", required=True)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _effective_batch(config: dict[str, Any]) -> int:
    return int(config["data"]["batch_size"]) * int(
        config["optimization"]["gradient_accumulation_steps"]
    )


def _treatment_normalized_config(
    config: dict[str, Any],
    *,
    label: str,
    expected_gamma: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = deepcopy(config)
    name = normalized.pop("name", None)
    loss = normalized.get("loss")
    if not isinstance(loss, dict):
        raise ValueError(f"{label} config lacks its loss section")
    gamma = trajectory._finite_number(
        loss.get("min_snr_gamma", 0.0),
        label=f"{label} min_snr_gamma",
    )
    if not math.isclose(gamma, expected_gamma, abs_tol=1e-12):
        raise ValueError(f"{label} min_snr_gamma differs")
    loss.pop("min_snr_gamma", None)
    return normalized, {"name": name, "min_snr_gamma": gamma}


def _trajectory_contract(
    metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    label: str,
    cutoff_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    optimization = config.get("optimization", {})
    runtime = config.get("runtime", {})
    effective_batch = _effective_batch(config)
    log_interval = int(optimization.get("log_interval", 0))
    evaluation_interval = int(runtime.get("evaluation_interval", 0))
    validated = trajectory._validate_rows(
        metrics["rows"],
        label=label,
        cutoff_step=cutoff_step,
        log_interval=log_interval,
        evaluation_interval=evaluation_interval,
        effective_batch=effective_batch,
    )
    validation_rows = validated.pop("validation_rows")
    return validated, validation_rows, {
        "effective_batch": effective_batch,
        "log_interval": log_interval,
        "evaluation_interval": evaluation_interval,
    }


def _validate_min_snr_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = []
    weighted_ratios = []
    for row in rows:
        step = int(row["step"])
        weight = trajectory._finite_number(
            row.get("min_snr_weight_mean"),
            label=f"gamma5_cofitok step {step} min_snr_weight_mean",
        )
        weighted = trajectory._finite_number(
            row.get("epsilon"),
            label=f"gamma5_cofitok step {step} epsilon",
        )
        unweighted = trajectory._finite_number(
            row.get("epsilon_unweighted"),
            label=f"gamma5_cofitok step {step} epsilon_unweighted",
        )
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"gamma5_cofitok step {step} has invalid Min-SNR weight")
        if unweighted < 0.0 or weighted < 0.0 or weighted > unweighted + 1e-12:
            raise ValueError(
                f"gamma5_cofitok step {step} weighted epsilon exceeds unweighted"
            )
        weights.append(weight)
        weighted_ratios.append(weighted / unweighted if unweighted else 0.0)
    below_one = sum(value < 0.999999 for value in weights)
    if below_one == 0:
        raise ValueError("gamma5_cofitok metrics never apply Min-SNR downweighting")
    return {
        "required_fields_present": True,
        "weight_range_verified": True,
        "weighted_epsilon_not_above_unweighted": True,
        "downweighted_row_count": below_one,
        "row_count": len(rows),
        "weight_mean": mean(weights),
        "weighted_to_unweighted_epsilon_ratio_mean": mean(weighted_ratios),
    }


def _paired_events(
    gamma5: list[dict[str, Any]],
    gamma0_cofitok: list[dict[str, Any]],
    gamma0_dense: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not len(gamma5) == len(gamma0_cofitok) == len(gamma0_dense):
        raise ValueError("fixed-validation event counts differ")
    metadata = (
        "step",
        "validation_event_index",
        "validation_batch_index",
        "validation_num_images",
        "validation_noise_seed",
    )
    events = []
    for treatment, cofitok, dense in zip(
        gamma5, gamma0_cofitok, gamma0_dense, strict=True
    ):
        step = int(treatment["step"])
        treatment_identity = tuple(treatment.get(field) for field in metadata)
        if (
            tuple(cofitok.get(field) for field in metadata) != treatment_identity
            or tuple(dense.get(field) for field in metadata) != treatment_identity
        ):
            raise ValueError(f"fixed-validation provenance differs at step {step}")
        values = {
            "gamma5_cofitok": float(treatment["validation_epsilon_mse"]),
            "gamma0_cofitok": float(cofitok["validation_epsilon_mse"]),
            "gamma0_dense": float(dense["validation_epsilon_mse"]),
        }
        events.append(
            {
                "step": step,
                "validation_event_index": int(treatment["validation_event_index"]),
                "validation_batch_index": int(treatment["validation_batch_index"]),
                "validation_num_images": int(treatment["validation_num_images"]),
                "validation_noise_seed": int(treatment["validation_noise_seed"]),
                **values,
                "relative_vs_gamma0_cofitok": (
                    values["gamma5_cofitok"] / values["gamma0_cofitok"] - 1.0
                    if values["gamma0_cofitok"]
                    else None
                ),
                "relative_vs_gamma0_dense": (
                    values["gamma5_cofitok"] / values["gamma0_dense"] - 1.0
                    if values["gamma0_dense"]
                    else None
                ),
            }
        )
    if not events:
        raise ValueError("fixed-validation trajectory is empty")
    return events


def _reference_summary(events: list[dict[str, Any]], reference: str) -> dict[str, Any]:
    treatment_values = [event["gamma5_cofitok"] for event in events]
    reference_values = [event[reference] for event in events]
    treatment_mean = mean(treatment_values)
    reference_mean = mean(reference_values)
    return {
        "event_count": len(events),
        "gamma5_mean_epsilon_mse": treatment_mean,
        "reference_mean_epsilon_mse": reference_mean,
        "relative_delta_of_means": (
            treatment_mean / reference_mean - 1.0 if reference_mean else None
        ),
        "gamma5_lower_event_count": sum(
            treatment < control
            for treatment, control in zip(
                treatment_values, reference_values, strict=True
            )
        ),
        "gamma5_higher_event_count": sum(
            treatment > control
            for treatment, control in zip(
                treatment_values, reference_values, strict=True
            )
        ),
        "endpoint_relative_delta": (
            treatment_values[-1] / reference_values[-1] - 1.0
            if reference_values[-1]
            else None
        ),
    }


def build_report(
    *,
    gamma5_metrics: dict[str, Any],
    gamma5_manifest: dict[str, Any],
    gamma0_cofitok_metrics: dict[str, Any],
    gamma0_cofitok_report: dict[str, Any],
    gamma0_dense_metrics: dict[str, Any],
    gamma0_dense_report: dict[str, Any],
    cutoff_step: int,
    expected_gamma5_revision: str,
    expected_gamma5_branch: str,
    expected_gamma0_revision: str,
    expected_gamma0_branch: str,
) -> dict[str, Any]:
    gamma5_source = trajectory._validate_manifest(
        gamma5_manifest,
        label="gamma5_cofitok",
        expected_revision=expected_gamma5_revision,
        expected_branch=expected_gamma5_branch,
    )
    gamma0_cofitok_source = trajectory._validate_manifest(
        gamma0_cofitok_report,
        label="gamma0_cofitok",
        expected_revision=expected_gamma0_revision,
        expected_branch=expected_gamma0_branch,
    )
    gamma0_dense_source = trajectory._validate_manifest(
        gamma0_dense_report,
        label="gamma0_dense",
        expected_revision=expected_gamma0_revision,
        expected_branch=expected_gamma0_branch,
    )
    sources = (gamma5_source, gamma0_cofitok_source, gamma0_dense_source)
    for field in ("dataset", "dataset_identity_sha256", "runtime_environment_sha256"):
        if len({source[field] for source in sources}) != 1:
            raise ValueError(f"three-source {field} differs")

    gamma5_config, gamma5_treatment = _treatment_normalized_config(
        gamma5_source["config"],
        label="gamma5_cofitok",
        expected_gamma=EXPECTED_GAMMA,
    )
    gamma0_config, gamma0_treatment = _treatment_normalized_config(
        gamma0_cofitok_source["config"],
        label="gamma0_cofitok",
        expected_gamma=0.0,
    )
    if gamma5_config != gamma0_config:
        raise ValueError("gamma5 and gamma0 CoFiTok configs differ beyond treatment/name")
    if gamma5_source["parameter_count"] != gamma0_cofitok_source["parameter_count"]:
        raise ValueError("gamma5 and gamma0 CoFiTok parameter counts differ")

    legacy_pair_contract = generation_pair_contract(
        gamma0_cofitok_source["config"], gamma0_dense_source["config"]
    )
    if legacy_pair_contract["valid"] is not True:
        raise ValueError(
            "legacy gamma0 pair contract failed: "
            + "; ".join(legacy_pair_contract["issues"])
        )

    gamma5_trajectory, gamma5_validation, gamma5_schedule = _trajectory_contract(
        gamma5_metrics,
        gamma5_source["config"],
        label="gamma5_cofitok",
        cutoff_step=cutoff_step,
    )
    cofitok_trajectory, cofitok_validation, cofitok_schedule = _trajectory_contract(
        gamma0_cofitok_metrics,
        gamma0_cofitok_source["config"],
        label="gamma0_cofitok",
        cutoff_step=cutoff_step,
    )
    dense_trajectory, dense_validation, dense_schedule = _trajectory_contract(
        gamma0_dense_metrics,
        gamma0_dense_source["config"],
        label="gamma0_dense",
        cutoff_step=cutoff_step,
    )
    if not gamma5_schedule == cofitok_schedule == dense_schedule:
        raise ValueError("three-source logging/evaluation/exposure schedules differ")
    min_snr_delivery = _validate_min_snr_rows(gamma5_metrics["rows"])
    events = _paired_events(gamma5_validation, cofitok_validation, dense_validation)

    return {
        "schema_version": 1,
        "status": "pass",
        "role": "generation_min_snr_fixed_validation_early_warning",
        "cutoff_step": cutoff_step,
        "images_seen": cutoff_step * gamma5_schedule["effective_batch"],
        "controlled_treatment": {
            "gamma5": gamma5_treatment,
            "gamma0": gamma0_treatment,
            "only_config_differences": ["name", "loss.min_snr_gamma"],
            "same_parameter_count": True,
            "same_dataset_identity": True,
            "same_runtime_environment": True,
            "legacy_pair_contract": legacy_pair_contract,
        },
        "trajectory_contract": {
            **gamma5_schedule,
            "gamma5_cofitok": gamma5_trajectory,
            "gamma0_cofitok": cofitok_trajectory,
            "gamma0_dense": dense_trajectory,
            "min_snr_delivery": min_snr_delivery,
        },
        "paired_fixed_validation": {
            "pairing_basis": (
                "same step, event index, batch index, image count, and noise seed"
            ),
            "events": events,
            "vs_gamma0_cofitok": _reference_summary(events, "gamma0_cofitok"),
            "vs_gamma0_dense": _reference_summary(events, "gamma0_dense"),
        },
        "scientific_interpretation": {
            "signal_type": "training_fixed_validation_epsilon_early_warning",
            "sample_quality_metrics_present": False,
            "generation_advantage_proven": False,
            "quality_claim_allowed": False,
            "formal_50k_result_substitute": False,
            "required_next_evidence": (
                "matched 50K completion and four terminal 10K DDIM-100 evaluation arms"
            ),
        },
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
    }


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _metrics_input(path: Path, *, cutoff_step: int) -> dict[str, Any]:
    return trajectory._read_metrics_prefix(path, cutoff_step=cutoff_step)


def main() -> None:
    args = _parse_args()
    metrics = {
        "gamma5_cofitok": _metrics_input(args.gamma5_metrics, cutoff_step=args.cutoff_step),
        "gamma0_cofitok": _metrics_input(
            args.gamma0_cofitok_metrics, cutoff_step=args.cutoff_step
        ),
        "gamma0_dense": _metrics_input(
            args.gamma0_dense_metrics, cutoff_step=args.cutoff_step
        ),
    }
    original_observed_files = {
        name: value["observed_file"] for name, value in metrics.items()
    }
    origins = {
        "gamma5_cofitok": args.gamma5_metrics,
        "gamma0_cofitok": args.gamma0_cofitok_metrics,
        "gamma0_dense": args.gamma0_dense_metrics,
    }
    if args.snapshot_dir is not None:
        snapshot_names = {
            "gamma5_cofitok": "gamma5_cofitok",
            "gamma0_cofitok": "gamma0_cofitok",
            "gamma0_dense": "gamma0_dense",
        }
        for name, stem in snapshot_names.items():
            snapshot = args.snapshot_dir / (
                f"{stem}_train_metrics_through_step_{args.cutoff_step:08d}.jsonl"
            )
            _atomic_write_bytes(snapshot, metrics[name]["raw_prefix"])
            replay = _metrics_input(snapshot, cutoff_step=args.cutoff_step)
            if replay["identity"] != metrics[name]["identity"]:
                raise ValueError(f"{name} snapshot identity differs from its source prefix")
            metrics[name] = replay

    manifests = {
        "gamma5_cofitok": trajectory._read_object(args.gamma5_manifest),
        "gamma0_cofitok": trajectory._read_object(args.gamma0_cofitok_report),
        "gamma0_dense": trajectory._read_object(args.gamma0_dense_report),
    }
    report = build_report(
        gamma5_metrics=metrics["gamma5_cofitok"],
        gamma5_manifest=manifests["gamma5_cofitok"],
        gamma0_cofitok_metrics=metrics["gamma0_cofitok"],
        gamma0_cofitok_report=manifests["gamma0_cofitok"],
        gamma0_dense_metrics=metrics["gamma0_dense"],
        gamma0_dense_report=manifests["gamma0_dense"],
        cutoff_step=args.cutoff_step,
        expected_gamma5_revision=args.expected_gamma5_revision,
        expected_gamma5_branch=args.expected_gamma5_branch,
        expected_gamma0_revision=args.expected_gamma0_revision,
        expected_gamma0_branch=args.expected_gamma0_branch,
    )
    builder = Path(__file__).resolve()
    report["builder"] = {
        "path": "scripts/build_generation_min_snr_early_warning.py",
        "bytes": builder.stat().st_size,
        "sha256": file_sha256(builder),
    }
    report["sources"] = {
        "metrics": {
            name: {
                "origin": origins[name].resolve().as_posix(),
                "origin_observed_file": original_observed_files[name],
                "bound_prefix": metrics[name]["identity"],
                "snapshot": metrics[name]["observed_file"],
            }
            for name in metrics
        },
        "gamma5_manifest": trajectory._source(args.gamma5_manifest),
        "gamma0_cofitok_report": trajectory._source(args.gamma0_cofitok_report),
        "gamma0_dense_report": trajectory._source(args.gamma0_dense_report),
    }
    write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "bytes": args.output.stat().st_size,
                "sha256": file_sha256(args.output),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
