from __future__ import annotations

import argparse
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    CONTROL_RANKING_CONFIG,
    PROBE_ROLE,
    PROBE_SCOPE,
    RANKED_RANKING_CONFIG,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_conditioning_sensitivity import (
    MANIFEST_FILENAME,
    REPORT_ROLE as SENSITIVITY_REPORT_ROLE,
    REPORT_SCHEMA_VERSION as SENSITIVITY_REPORT_SCHEMA_VERSION,
    validate_completed_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_ranking_four_arm_postevaluation"
TRAINING_STATUS_ROLE = "generation_conditioning_ranking_four_arm_probe_training_status"
EXPECTED_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_probe1k_v1"
)

RUN_NAMES = (
    "control_cofitok",
    "ranked_cofitok",
    "control_dense_identity",
    "ranked_dense_identity",
)
METHOD_ARMS = {
    "cofitok": ("control_cofitok", "ranked_cofitok"),
    "dense_identity": ("control_dense_identity", "ranked_dense_identity"),
}
PREPARATION_CONFIG_KEYS = {
    "control_cofitok": "control_cofitok",
    "ranked_cofitok": "ranked_cofitok",
    "control_dense_identity": "control_dense",
    "ranked_dense_identity": "ranked_dense",
}
CONFIG_PATHS = {
    "control_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_"
        "u2_ema_teacher_k8_probe1k.json"
    ),
    "ranked_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_"
        "u2_ema_teacher_classrank_k8_probe1k.json"
    ),
    "control_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_"
        "ema_teacher_dense_probe1k.json"
    ),
    "ranked_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_"
        "ema_teacher_classrank_dense_probe1k.json"
    ),
}
EXPECTED_REQUEST = {
    "weights": "ema",
    "num_samples": 8,
    "start_label": 64,
    "wrong_label_offset": 500,
    "timesteps": [100, 500, 900],
    "noise_seed": 204060,
    "threads": 2,
}
EXPECTED_DATASET = "imagenet_256_10pct"
EXPECTED_NUM_CLASSES = 1_000
EXPECTED_STEPS = 1_000
EXPECTED_CHECKPOINT_STEPS = [500, 750, 1_000]
EXPECTED_CHECKPOINT_INTERVAL = 250
EXPECTED_EVALUATION_INTERVAL = 250
EXPECTED_VALIDATION_EVENTS = EXPECTED_STEPS // EXPECTED_EVALUATION_INTERVAL
SIGNIFICANCE_LEVEL = 0.05
MAX_CORRECT_MSE_RATIO = 1.02
SENSITIVITY_CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "generates_new_samples": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "replaces_formal_quality_gate": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound CPU-only held-out decision for the four-arm "
            "class-conditioning-ranking probe."
        )
    )
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--training-status", required=True)
    for run in RUN_NAMES:
        option = run.replace("_", "-")
        parser.add_argument(f"--{option}-audit", required=True)
        parser.add_argument(f"--{option}-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Revalidate and reuse an exactly source-bound completed report.",
    )
    return parser.parse_args()


def _load_json_source(path: str | Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    return read_json_object(source, name=label), file_identity(source)


def _load_sensitivity_source(
    path: str | Path,
    *,
    run: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = reject_symlink_chain(path, name=f"{run} sensitivity report")
    if not report_path.is_file():
        raise FileNotFoundError(f"{run} sensitivity report is missing: {report_path}")
    report = read_json_object(report_path, name=f"{run} sensitivity report")
    if (
        report.get("schema_version") != SENSITIVITY_REPORT_SCHEMA_VERSION
        or report.get("role") != SENSITIVITY_REPORT_ROLE
        or report.get("status") != "completed"
    ):
        raise ValueError(f"{run} sensitivity report is not completed evidence")
    manifest_path = reject_symlink_chain(
        report_path.parent / MANIFEST_FILENAME,
        name=f"{run} sensitivity manifest",
    )
    manifest = read_json_object(
        manifest_path,
        name=f"{run} sensitivity manifest",
    )
    manifest_identity = file_identity(manifest_path)
    validate_completed_report(
        report,
        manifest=manifest,
        manifest_identity=manifest_identity,
    )
    return report, {
        "report": file_identity(report_path),
        "manifest": manifest_identity,
    }


def _one_sided_binomial_pvalue(successes: int, trials: int) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial count")
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Cannot average non-finite values")
    return sum(values) / len(values)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_preparation(preparation: Mapping[str, Any]) -> dict[str, Any]:
    if (
        preparation.get("schema_version") != 1
        or preparation.get("role") != PROBE_ROLE
        or preparation.get("scope") != PROBE_SCOPE
        or preparation.get("status") != "pass"
        or preparation.get("valid") is not True
        or preparation.get("issues") != []
        or preparation.get("gpu_execution_authorized") is not False
        or preparation.get("authorization_required") is not True
        or preparation.get("control_ranking_config") != CONTROL_RANKING_CONFIG
        or preparation.get("ranked_ranking_config") != RANKED_RANKING_CONFIG
    ):
        raise ValueError("Ranking-probe preparation report is not valid")
    git = preparation.get("git")
    if (
        not isinstance(git, Mapping)
        or not _is_git_revision(git.get("revision"))
        or not isinstance(git.get("branch"), str)
        or not str(git["branch"]).strip()
        or git.get("branch") != str(git["branch"]).strip()
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("Ranking-probe preparation Git is malformed")
    output_root = preparation.get("output_root")
    if (
        output_root != EXPECTED_OUTPUT_ROOT
        or not PurePosixPath(str(output_root)).is_absolute()
        or ".." in PurePosixPath(str(output_root)).parts
    ):
        raise ValueError("Ranking-probe preparation output root is malformed")
    parameter_counts = preparation.get("parameter_counts")
    configs = preparation.get("configs")
    expected_preparation_keys = set(PREPARATION_CONFIG_KEYS.values())
    if (
        not isinstance(parameter_counts, Mapping)
        or set(parameter_counts) != expected_preparation_keys
        or not isinstance(configs, Mapping)
        or set(configs) != expected_preparation_keys
    ):
        raise ValueError("Ranking-probe preparation parameter counts are missing")
    for key in expected_preparation_keys:
        if int(parameter_counts.get(key, 0)) <= 0:
            raise ValueError(f"Ranking-probe preparation parameter count {key} is invalid")
        identity = configs[key]
        if (
            not isinstance(identity, Mapping)
            or int(identity.get("bytes", 0)) <= 0
            or not _is_sha256(identity.get("sha256"))
        ):
            raise ValueError(f"Ranking-probe preparation config {key} is malformed")
    if (
        int(parameter_counts["control_cofitok"])
        != int(parameter_counts["ranked_cofitok"])
        or int(parameter_counts["control_dense"])
        != int(parameter_counts["ranked_dense"])
    ):
        raise ValueError("Ranking-probe control and ranked parameter counts differ")
    return {
        "revision": str(git["revision"]),
        "branch": str(git["branch"]),
        "output_root": output_root,
        "parameter_counts": dict(parameter_counts),
        "config_identities": {key: dict(configs[key]) for key in sorted(configs)},
        "ranking_min_timestep": int(
            preparation["ranked_ranking_config"][
                "class_conditioning_ranking_min_timestep"
            ]
        ),
    }


def _validate_training_evidence(
    *,
    preparation: Mapping[str, Any],
    training_status: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    training_audits: Mapping[str, Mapping[str, Any]],
    expected_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _validate_preparation(preparation)
    if (
        training_status.get("schema_version") != 1
        or training_status.get("role") != TRAINING_STATUS_ROLE
        or training_status.get("status") != "completed"
        or training_status.get("revision") != contract["revision"]
        or set(training_status.get("runs", {})) != set(RUN_NAMES)
    ):
        raise ValueError("Four-arm training status is not completed evidence")
    boundary = training_status.get("claim_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not False
        for field in (
            "training_quality_claim_allowed",
            "sample_quality_claim_allowed",
            "promotion_authorization_allowed",
            "followup_training_allowed",
            "full_training_launch_allowed",
        )
    ):
        raise ValueError("Four-arm training status claim boundary is malformed")

    result: dict[str, Any] = {}
    for run in RUN_NAMES:
        status_run = training_status["runs"][run]
        report = training_reports.get(run)
        audit = training_audits.get(run)
        if not isinstance(status_run, Mapping) or not isinstance(report, Mapping):
            raise ValueError(f"{run} training report is missing")
        if not isinstance(audit, Mapping):
            raise ValueError(f"{run} training audit is missing")
        expected_run_dir = f'{contract["output_root"]}/{run}'
        expected_report_path = f"{expected_run_dir}/training_report.json"
        expected_config = expected_configs[run]
        model_config = expected_config["model"]
        data_config = expected_config["data"]
        runtime_config = expected_config["runtime"]
        optimization_config = expected_config["optimization"]
        loss_config = expected_config["loss"]
        expected_samples_seen = (
            EXPECTED_STEPS
            * int(data_config["batch_size"])
            * int(optimization_config["gradient_accumulation_steps"])
        )
        if (
            status_run.get("training_report") != expected_report_path
            or report.get("training_complete") is not True
            or int(report.get("completed_steps", -1)) != EXPECTED_STEPS
            or int(report.get("target_steps", -1)) != EXPECTED_STEPS
            or report.get("output_dir") != expected_run_dir
            or report.get("config") != expected_config
            or status_run.get("final_metrics") != report.get("final_metrics")
        ):
            raise ValueError(f"{run} training report differs from the probe contract")
        report_git = report.get("git")
        if (
            not isinstance(report_git, Mapping)
            or report_git.get("revision") != contract["revision"]
            or report_git.get("branch") != contract["branch"]
            or report_git.get("dirty") is not False
        ):
            raise ValueError(f"{run} training Git differs from preparation")
        preparation_key = PREPARATION_CONFIG_KEYS[run]
        expected_parameters = int(contract["parameter_counts"][preparation_key])
        if int(report.get("parameter_count", -1)) != expected_parameters:
            raise ValueError(f"{run} parameter count differs from preparation")
        latest = report.get("latest_checkpoint")
        latest_sha = latest.get("checkpoint_sha256") if isinstance(latest, Mapping) else None
        if (
            not isinstance(latest, Mapping)
            or int(latest.get("step", -1)) != EXPECTED_STEPS
            or latest.get("checkpoint") != "checkpoint_step_00001000.pt"
            or not _is_sha256(latest_sha)
            or latest_sha != status_run.get("checkpoint_sha256")
            or int(latest.get("checkpoint_bytes", 0)) <= 0
            or latest.get("integrity_manifest")
            != "checkpoint_step_00001000.pt.integrity.json"
            or not _is_sha256(latest.get("dataset_identity_sha256"))
            or not _is_sha256(latest.get("runtime_environment_sha256"))
        ):
            raise ValueError(f"{run} latest checkpoint differs from training status")
        final_metrics = report.get("final_metrics")
        expected_ranking_scale = 1.0 if run.startswith("ranked_") else 0.0
        if not isinstance(final_metrics, Mapping):
            raise ValueError(f"{run} final metrics are missing")
        for field in ("epsilon", "class_conditioning_ranking"):
            value = float(final_metrics.get(field, math.nan))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{run} final metric {field} is invalid")
        if (
            not math.isclose(
                float(final_metrics.get("class_conditioning_ranking_scale", math.nan)),
                expected_ranking_scale,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or int(final_metrics.get("samples_seen", -1)) != expected_samples_seen
        ):
            raise ValueError(f"{run} final ranking schedule or samples seen differ")

        checkpoint_audit = audit.get("checkpoint")
        audit_latest = (
            checkpoint_audit.get("latest") if isinstance(checkpoint_audit, Mapping) else None
        )
        audit_integrity = (
            checkpoint_audit.get("latest_integrity")
            if isinstance(checkpoint_audit, Mapping)
            else None
        )
        validation = audit.get("validation")
        audit_training_report = audit.get("training_report")
        schedules = audit.get("consistency_schedules")
        if (
            audit.get("schema_version") != 2
            or audit.get("status") != "complete"
            or audit.get("issues") != []
            or int(audit.get("expected_steps", -1)) != EXPECTED_STEPS
            or int(audit.get("last_step", -1)) != EXPECTED_STEPS
            or float(audit.get("progress_fraction", -1.0)) != 1.0
            or audit.get("run_dir") != expected_run_dir
            or not isinstance(checkpoint_audit, Mapping)
            or int(checkpoint_audit.get("interval", -1)) != EXPECTED_CHECKPOINT_INTERVAL
            or checkpoint_audit.get("status") != "available"
            or checkpoint_audit.get("required_steps") != EXPECTED_CHECKPOINT_STEPS
            or checkpoint_audit.get("missing_required_steps") != []
            or not isinstance(audit_latest, Mapping)
            or int(audit_latest.get("step", -1)) != EXPECTED_STEPS
            or audit_latest.get("checkpoint_sha256") != latest_sha
            or not isinstance(audit_integrity, Mapping)
            or audit_integrity.get("status") != "verified"
            or audit_integrity.get("checkpoint_sha256") != latest_sha
            or not isinstance(audit_training_report, Mapping)
            or audit_training_report.get("path") != expected_report_path
            or audit_training_report.get("status") != "current"
            or int(audit_training_report.get("completed_steps", -1)) != EXPECTED_STEPS
            or audit_training_report.get("training_complete") is not True
            or not isinstance(validation, Mapping)
            or int(validation.get("configured_interval", -1))
            != EXPECTED_EVALUATION_INTERVAL
            or int(validation.get("event_count", -1)) != EXPECTED_VALIDATION_EVENTS
            or int(validation.get("expected_event_count", -1))
            != EXPECTED_VALIDATION_EVENTS
            or validation.get("logging_complete") is not True
            or validation.get("provenance_metadata_status") != "complete"
            or not isinstance(schedules, Mapping)
            or schedules.get("status") != "verified"
            or schedules.get("config_sha256")
            != contract["config_identities"][preparation_key]["sha256"]
        ):
            raise ValueError(f"{run} training audit did not pass the exact 1K contract")
        schedule_rows = schedules.get("schedules")
        metric_row_count = int(audit.get("metric_row_count", -1))
        if not isinstance(schedule_rows, Mapping) or metric_row_count < 1:
            raise ValueError(f"{run} training schedule audit is malformed")
        for schedule_name in (
            "rollout_consistency",
            "ema_teacher_consistency",
            "class_conditioning_ranking",
        ):
            schedule = schedule_rows.get(schedule_name)
            if not isinstance(schedule, Mapping):
                raise ValueError(f"{run} lacks schedule audit {schedule_name}")
            if schedule_name == "class_conditioning_ranking":
                prefix = "class_conditioning_ranking"
            else:
                prefix = schedule_name
            if (
                not math.isclose(
                    float(schedule.get("weight", math.nan)),
                    float(loss_config[f"{prefix}_weight"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or int(schedule.get("start_step", -1))
                != int(loss_config[f"{prefix}_start_step"])
                or int(schedule.get("warmup_steps", -1))
                != int(loss_config[f"{prefix}_warmup_steps"])
                or int(schedule.get("verified_rows", -1)) != metric_row_count
            ):
                raise ValueError(f"{run} schedule audit {schedule_name} differs")
        expected_model_contract = {
            "name": expected_config["name"],
            "predictor_type": model_config["predictor_type"],
            "synthesis_mode": model_config["synthesis_mode"],
            "token_count": int(model_config["token_count"]),
            "num_classes": int(model_config["num_classes"]),
        }
        if expected_model_contract["num_classes"] != EXPECTED_NUM_CLASSES:
            raise ValueError(f"{run} config num_classes differs from the probe contract")
        result[run] = {
            "run_dir": expected_run_dir,
            "parameter_count": expected_parameters,
            "checkpoint_sha256": latest_sha,
            "checkpoint_bytes": int(latest["checkpoint_bytes"]),
            "checkpoint_step": int(latest["step"]),
            "checkpoint_dataset_identity_sha256": latest.get(
                "dataset_identity_sha256"
            ),
            "checkpoint_runtime_environment_sha256": latest.get(
                "runtime_environment_sha256"
            ),
            "final_metrics": dict(final_metrics),
            "model_contract": expected_model_contract,
            "dataset_contract": {
                "alias": data_config["dataset"],
                "root": f'{str(data_config["root"]).rstrip("/")}/{data_config["dataset"]}',
            },
            "runtime_seed": int(runtime_config["seed"]),
        }
    return {**contract, "runs": result}


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["timestep"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def _validate_sensitivity_reports(
    *,
    reports: Mapping[str, Mapping[str, Any]],
    training_contract: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    if set(reports) != set(RUN_NAMES):
        raise ValueError("Four-arm sensitivity reports are incomplete")
    datasets = []
    checkpoint_shas = []
    canonical_row_keys: list[tuple[int, int, int, int, int]] | None = None
    for run in RUN_NAMES:
        report = reports[run]
        if (
            report.get("git") != evaluator_git
            or report.get("request") != EXPECTED_REQUEST
            or report.get("weights") != "ema"
            or report.get("claim_boundary") != SENSITIVITY_CLAIM_BOUNDARY
        ):
            raise ValueError(f"{run} sensitivity evaluator contract differs")
        runtime = report.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("device") != "cpu"
            or int(runtime.get("threads", -1)) != EXPECTED_REQUEST["threads"]
            or not math.isfinite(float(runtime.get("elapsed_seconds", math.nan)))
            or float(runtime["elapsed_seconds"]) <= 0.0
        ):
            raise ValueError(f"{run} sensitivity runtime differs")
        dataset = report.get("dataset")
        dataset_contract = training_contract["runs"][run]["dataset_contract"]
        if (
            not isinstance(dataset, Mapping)
            or dataset.get("alias") != EXPECTED_DATASET
            or dataset.get("alias") != dataset_contract["alias"]
            or dataset.get("root") != dataset_contract["root"]
        ):
            raise ValueError(f"{run} sensitivity dataset differs")
        label_map = dataset.get("label_map")
        samples = dataset.get("samples")
        if (
            not isinstance(label_map, Mapping)
            or int(label_map.get("bytes", 0)) <= 0
            or not _is_sha256(label_map.get("sha256"))
            or not isinstance(samples, list)
            or len(samples) != EXPECTED_REQUEST["num_samples"]
        ):
            raise ValueError(f"{run} sensitivity dataset identity is malformed")
        expected_sample_metadata: dict[int, dict[str, Any]] = {}
        for sample_index, sample in enumerate(samples):
            expected_label = EXPECTED_REQUEST["start_label"] + sample_index
            image = sample.get("image") if isinstance(sample, Mapping) else None
            if (
                not isinstance(sample, Mapping)
                or int(sample.get("label", -1)) != expected_label
                or not isinstance(sample.get("wnid"), str)
                or not sample["wnid"]
                or not isinstance(image, Mapping)
                or int(image.get("bytes", 0)) <= 0
                or not _is_sha256(image.get("sha256"))
            ):
                raise ValueError(f"{run} sensitivity selected sample differs")
            expected_sample_metadata[sample_index] = {
                "correct_label": expected_label,
                "correct_wnid": sample["wnid"],
                "wrong_label": (
                    expected_label + EXPECTED_REQUEST["wrong_label_offset"]
                )
                % EXPECTED_NUM_CLASSES,
                "null_label": EXPECTED_NUM_CLASSES,
                "noise_seed": EXPECTED_REQUEST["noise_seed"] + sample_index,
            }
        datasets.append(dataset)
        model = report.get("model")
        expected_model = training_contract["runs"][run]["model_contract"]
        if not isinstance(model, Mapping):
            raise ValueError(f"{run} sensitivity model metadata is malformed")
        if (
            any(model.get(field) != expected_model[field] for field in expected_model)
            or int(model.get("parameter_count", -1))
            != int(training_contract["runs"][run]["parameter_count"])
        ):
            raise ValueError(f"{run} sensitivity method identity differs")
        checkpoint = report.get("checkpoint")
        checkpoint_git = checkpoint.get("git") if isinstance(checkpoint, Mapping) else None
        checkpoint_integrity = (
            checkpoint.get("integrity_manifest")
            if isinstance(checkpoint, Mapping)
            else None
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("path")
            != f'{training_contract["runs"][run]["run_dir"]}/checkpoint_step_00001000.pt'
            or int(checkpoint.get("bytes", -1))
            != int(training_contract["runs"][run]["checkpoint_bytes"])
            or checkpoint.get("sha256")
            != training_contract["runs"][run]["checkpoint_sha256"]
            or int(checkpoint.get("step", -1)) != EXPECTED_STEPS
            or int(checkpoint.get("format_version", -1)) != 1
            or not isinstance(checkpoint_integrity, Mapping)
            or checkpoint_integrity.get("path")
            != (
                f'{training_contract["runs"][run]["run_dir"]}/'
                "checkpoint_step_00001000.pt.integrity.json"
            )
            or int(checkpoint_integrity.get("bytes", 0)) <= 0
            or not _is_sha256(checkpoint_integrity.get("sha256"))
            or checkpoint.get("dataset_identity_sha256")
            != training_contract["runs"][run]["checkpoint_dataset_identity_sha256"]
            or checkpoint.get("runtime_environment_sha256")
            != training_contract["runs"][run][
                "checkpoint_runtime_environment_sha256"
            ]
            or checkpoint_git
            != {
                "revision": training_contract["revision"],
                "branch": training_contract["branch"],
                "dirty": False,
            }
        ):
            raise ValueError(f"{run} sensitivity checkpoint differs from training")
        rows = report.get("sample_rows")
        expected_rows = EXPECTED_REQUEST["num_samples"] * len(EXPECTED_REQUEST["timesteps"])
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise ValueError(f"{run} sensitivity row count differs")
        row_keys = [_row_key(row) for row in rows]
        if len(set(row_keys)) != len(row_keys):
            raise ValueError(f"{run} sensitivity rows contain duplicate identities")
        expected_row_keys = []
        for sample_index, metadata in expected_sample_metadata.items():
            for timestep in EXPECTED_REQUEST["timesteps"]:
                expected_row_keys.append(
                    (
                        sample_index,
                        timestep,
                        metadata["correct_label"],
                        metadata["wrong_label"],
                        metadata["noise_seed"],
                    )
                )
        if sorted(row_keys) != sorted(expected_row_keys):
            raise ValueError(f"{run} sensitivity source-row identities differ")
        if canonical_row_keys is None:
            canonical_row_keys = sorted(row_keys)
        elif sorted(row_keys) != canonical_row_keys:
            raise ValueError("Four-arm sensitivity row identities differ")
        for row in rows:
            metadata = expected_sample_metadata[int(row["sample_index"])]
            if (
                row.get("correct_wnid") != metadata["correct_wnid"]
                or int(row.get("null_label", -1)) != metadata["null_label"]
                or not math.isfinite(float(row.get("forward_seconds", math.nan)))
                or float(row["forward_seconds"]) <= 0.0
            ):
                raise ValueError(f"{run} sensitivity source-row metadata differs")
            for condition in ("correct", "wrong", "null"):
                _condition_mse(row, condition)
            reported_improvement = row.get("correct_relative_mse_improvement")
            reported_better = row.get("correct_better")
            if not isinstance(reported_improvement, Mapping) or not isinstance(
                reported_better, Mapping
            ):
                raise ValueError(f"{run} sensitivity derived ranking fields are missing")
            for comparison, reference in (
                ("versus_wrong", "wrong"),
                ("versus_null", "null"),
            ):
                recomputed = _recomputed_improvement(row, reference)
                reported = float(reported_improvement.get(comparison, math.nan))
                if not math.isfinite(reported) or not math.isclose(
                    reported,
                    recomputed,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{run} sensitivity derived ranking field differs from raw MSE"
                    )
            if (
                reported_better.get("than_wrong")
                is not (_condition_mse(row, "correct") < _condition_mse(row, "wrong"))
                or reported_better.get("than_null")
                is not (_condition_mse(row, "correct") < _condition_mse(row, "null"))
            ):
                raise ValueError(f"{run} sensitivity correct-better field differs")
        checkpoint_shas.append(checkpoint["sha256"])
    if any(dataset != datasets[0] for dataset in datasets[1:]):
        raise ValueError("Four-arm sensitivity datasets or selected samples differ")
    if len(set(checkpoint_shas)) != len(checkpoint_shas):
        raise ValueError("Four-arm sensitivity reports require distinct checkpoints")
    return {
        "request": dict(EXPECTED_REQUEST),
        "dataset": datasets[0],
        "weights": "ema",
        "evaluator_git": dict(evaluator_git),
        "canonical_row_keys": [list(key) for key in canonical_row_keys or []],
    }


def _sample_identity(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def _summarize_signed(values: Sequence[float]) -> dict[str, Any]:
    positive_count = sum(value > 0.0 for value in values)
    return {
        "mean": _mean(values),
        "positive_count": positive_count,
        "positive_fraction": positive_count / len(values),
        "one_sided_sign_test_pvalue": _one_sided_binomial_pvalue(
            positive_count,
            len(values),
        ),
    }


def _condition_mse(row: Mapping[str, Any], condition: str) -> float:
    try:
        value = float(row["conditions"][condition]["epsilon_mse_to_noise"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Sensitivity row lacks {condition} epsilon MSE") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Sensitivity row {condition} epsilon MSE is invalid")
    return value


def _recomputed_improvement(row: Mapping[str, Any], reference: str) -> float:
    correct_mse = _condition_mse(row, "correct")
    reference_mse = _condition_mse(row, reference)
    return (reference_mse - correct_mse) / reference_mse


def _method_postevaluation(
    *,
    control: Mapping[str, Any],
    ranked: Mapping[str, Any],
    eligible_timesteps: Sequence[int],
) -> dict[str, Any]:
    control_rows = {_row_key(row): row for row in control["sample_rows"]}
    ranked_rows = {_row_key(row): row for row in ranked["sample_rows"]}
    if control_rows.keys() != ranked_rows.keys():
        raise ValueError("Control and ranked sensitivity row identities differ")

    per_sample: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for key in sorted(control_rows):
        timestep = key[1]
        if timestep not in eligible_timesteps:
            continue
        control_row = control_rows[key]
        ranked_row = ranked_rows[key]
        sample_key = _sample_identity(control_row)
        per_sample.setdefault(sample_key, []).append(
            {
                "timestep": timestep,
                "control": control_row,
                "ranked": ranked_row,
            }
        )
    if len(per_sample) != EXPECTED_REQUEST["num_samples"]:
        raise ValueError("Eligible sensitivity sample count differs from the contract")

    sample_rows = []
    expected_timestep_set = set(eligible_timesteps)
    for sample_key, rows in sorted(per_sample.items()):
        if {int(row["timestep"]) for row in rows} != expected_timestep_set:
            raise ValueError("A held-out sample lacks an eligible timestep")

        def improvement(arm: str, comparison: str) -> float:
            reference = {
                "versus_wrong": "wrong",
                "versus_null": "null",
            }[comparison]
            return _mean(
                [
                    _recomputed_improvement(row[arm], reference)
                    for row in rows
                ]
            )

        control_wrong = improvement("control", "versus_wrong")
        control_null = improvement("control", "versus_null")
        ranked_wrong = improvement("ranked", "versus_wrong")
        ranked_null = improvement("ranked", "versus_null")
        control_correct_mse = _mean(
            [
                _condition_mse(row["control"], "correct")
                for row in rows
            ]
        )
        ranked_correct_mse = _mean(
            [
                _condition_mse(row["ranked"], "correct")
                for row in rows
            ]
        )
        sample_rows.append(
            {
                "sample_index": sample_key[0],
                "correct_label": sample_key[1],
                "wrong_label": sample_key[2],
                "noise_seed": sample_key[3],
                "eligible_timesteps": list(eligible_timesteps),
                "control_improvement": {
                    "versus_wrong": control_wrong,
                    "versus_null": control_null,
                },
                "ranked_improvement": {
                    "versus_wrong": ranked_wrong,
                    "versus_null": ranked_null,
                },
                "ranked_minus_control_improvement": {
                    "versus_wrong": ranked_wrong - control_wrong,
                    "versus_null": ranked_null - control_null,
                },
                "correct_mse": {
                    "control": control_correct_mse,
                    "ranked": ranked_correct_mse,
                    "ranked_to_control_ratio": ranked_correct_mse
                    / control_correct_mse,
                },
            }
        )

    ranked_absolute = {
        comparison: _summarize_signed(
            [float(row["ranked_improvement"][comparison]) for row in sample_rows]
        )
        for comparison in ("versus_wrong", "versus_null")
    }
    ranked_minus_control = {
        comparison: _summarize_signed(
            [
                float(row["ranked_minus_control_improvement"][comparison])
                for row in sample_rows
            ]
        )
        for comparison in ("versus_wrong", "versus_null")
    }
    control_correct_mse_mean = _mean(
        [float(row["correct_mse"]["control"]) for row in sample_rows]
    )
    ranked_correct_mse_mean = _mean(
        [float(row["correct_mse"]["ranked"]) for row in sample_rows]
    )
    correct_mse_ratio = ranked_correct_mse_mean / control_correct_mse_mean
    gates = {
        "ranked_absolute_mean_positive": all(
            summary["mean"] > 0.0 for summary in ranked_absolute.values()
        ),
        "ranked_absolute_sample_significant": all(
            summary["one_sided_sign_test_pvalue"] < SIGNIFICANCE_LEVEL
            for summary in ranked_absolute.values()
        ),
        "ranked_minus_control_mean_positive": all(
            summary["mean"] > 0.0 for summary in ranked_minus_control.values()
        ),
        "ranked_minus_control_sample_significant": all(
            summary["one_sided_sign_test_pvalue"] < SIGNIFICANCE_LEVEL
            for summary in ranked_minus_control.values()
        ),
        "correct_mse_ratio_within_limit": correct_mse_ratio
        <= MAX_CORRECT_MSE_RATIO,
    }
    return {
        "sample_count": len(sample_rows),
        "eligible_row_count": len(sample_rows) * len(eligible_timesteps),
        "independent_unit": "held_out_validation_image",
        "timestep_rows_aggregated_before_sign_test": True,
        "ranking_metrics_recomputed_from_condition_mse": True,
        "ranked_absolute": ranked_absolute,
        "ranked_minus_control": ranked_minus_control,
        "correct_mse": {
            "control_mean": control_correct_mse_mean,
            "ranked_mean": ranked_correct_mse_mean,
            "ranked_to_control_ratio": correct_mse_ratio,
            "maximum_allowed_ratio": MAX_CORRECT_MSE_RATIO,
        },
        "gates": gates,
        "pass": all(gates.values()),
        "per_sample": sample_rows,
    }


def build_postevaluation(
    *,
    preparation: Mapping[str, Any],
    training_status: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    training_audits: Mapping[str, Mapping[str, Any]],
    expected_configs: Mapping[str, Mapping[str, Any]],
    sensitivity_reports: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Any],
    git: Mapping[str, Any],
) -> dict[str, Any]:
    training_contract = _validate_training_evidence(
        preparation=preparation,
        training_status=training_status,
        training_reports=training_reports,
        training_audits=training_audits,
        expected_configs=expected_configs,
    )
    expected_evaluator_git = {
        "revision": training_contract["revision"],
        "branch": training_contract["branch"],
        "tracked_dirty": False,
    }
    if dict(git) != expected_evaluator_git:
        raise ValueError(
            "Postevaluation requires the exact clean probe training checkout"
        )
    sensitivity_contract = _validate_sensitivity_reports(
        reports=sensitivity_reports,
        training_contract=training_contract,
        evaluator_git=git,
    )
    min_timestep = int(training_contract["ranking_min_timestep"])
    eligible_timesteps = [
        timestep
        for timestep in EXPECTED_REQUEST["timesteps"]
        if timestep >= min_timestep
    ]
    excluded_timesteps = [
        timestep
        for timestep in EXPECTED_REQUEST["timesteps"]
        if timestep < min_timestep
    ]
    if not eligible_timesteps or not excluded_timesteps:
        raise ValueError("Postevaluation requires both eligible and descriptive timesteps")

    methods = {
        method: _method_postevaluation(
            control=sensitivity_reports[arms[0]],
            ranked=sensitivity_reports[arms[1]],
            eligible_timesteps=eligible_timesteps,
        )
        for method, arms in METHOD_ARMS.items()
    }
    method_passes = {method: result["pass"] for method, result in methods.items()}
    shared_recovery = all(method_passes.values())
    if shared_recovery:
        next_action = "consider_separately_authorized_matched_sampling_validation"
    elif any(method_passes.values()):
        next_action = "reject_shared_repair_due_method_asymmetry"
    else:
        next_action = "revise_training_time_semantic_alignment_objective"

    difference_in_differences = {
        comparison: (
            methods["cofitok"]["ranked_minus_control"][comparison]["mean"]
            - methods["dense_identity"]["ranked_minus_control"][comparison]["mean"]
        )
        for comparison in ("versus_wrong", "versus_null")
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "git": dict(git),
        "sources": dict(sources),
        "training_contract": training_contract,
        "sensitivity_contract": sensitivity_contract,
        "evaluation_contract": {
            "eligible_timesteps": eligible_timesteps,
            "descriptive_only_timesteps": excluded_timesteps,
            "sample_count": EXPECTED_REQUEST["num_samples"],
            "significance_level": SIGNIFICANCE_LEVEL,
            "maximum_correct_mse_ratio": MAX_CORRECT_MSE_RATIO,
            "paired_unit": "held_out_validation_image",
            "timestep_rows_are_not_independent_units": True,
        },
        "methods": methods,
        "difference_in_differences": difference_in_differences,
        "difference_in_differences_is_descriptive_only": True,
        "decision": {
            "method_passes": method_passes,
            "shared_semantic_alignment_recovery_supported": shared_recovery,
            "cofitok_specific_advantage_claim_allowed": False,
            "recommended_next_action": next_action,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "generates_new_samples": False,
            "authorizes_training": False,
            "authorizes_sampling": False,
            "authorizes_checkpoint_promotion": False,
            "authorizes_full_training": False,
            "authorizes_release": False,
            "replaces_formal_quality_gate": False,
        },
    }


def _load_expected_configs(
    preparation: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    identities: dict[str, Any] = {}
    preparation_configs = preparation.get("configs")
    if not isinstance(preparation_configs, Mapping):
        raise ValueError("Preparation config identities are malformed")
    for run, relative_path in CONFIG_PATHS.items():
        path = reject_symlink_chain(PROJECT_ROOT / relative_path, name=f"{run} config")
        identity = file_identity(path)
        preparation_key = PREPARATION_CONFIG_KEYS[run]
        expected_identity = preparation_configs.get(preparation_key)
        if not isinstance(expected_identity, Mapping) or any(
            identity[field] != expected_identity.get(field)
            for field in ("bytes", "sha256")
        ):
            raise ValueError(f"{run} config bytes differ from the preparation report")
        configs[run] = config_to_dict(load_config(path))
        identities[run] = identity
    return configs, identities


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(args.output, name="ranking probe postevaluation output")
    if output.exists() and not output.is_file():
        raise ValueError(f"Postevaluation output is not a file: {output}")
    with exclusive_output_lock(output, role=REPORT_ROLE):
        preparation, preparation_source = _load_json_source(
            args.preparation_report,
            label="ranking probe preparation report",
        )
        training_status, training_status_source = _load_json_source(
            args.training_status,
            label="ranking probe training status",
        )
        expected_configs, config_sources = _load_expected_configs(preparation)

        training_reports: dict[str, dict[str, Any]] = {}
        training_report_sources: dict[str, Any] = {}
        training_audits: dict[str, dict[str, Any]] = {}
        training_audit_sources: dict[str, Any] = {}
        sensitivity_reports: dict[str, dict[str, Any]] = {}
        sensitivity_sources: dict[str, Any] = {}
        for run in RUN_NAMES:
            status_run = training_status.get("runs", {}).get(run)
            if not isinstance(status_run, Mapping):
                raise ValueError(f"Training status lacks run {run}")
            training_report, training_report_source = _load_json_source(
                status_run.get("training_report", ""),
                label=f"{run} training report",
            )
            audit_argument = getattr(args, f"{run}_audit")
            sensitivity_argument = getattr(args, f"{run}_report")
            audit, audit_source = _load_json_source(
                audit_argument,
                label=f"{run} training audit",
            )
            sensitivity, sensitivity_source = _load_sensitivity_source(
                sensitivity_argument,
                run=run,
            )
            training_reports[run] = training_report
            training_report_sources[run] = training_report_source
            training_audits[run] = audit
            training_audit_sources[run] = audit_source
            sensitivity_reports[run] = sensitivity
            sensitivity_sources[run] = sensitivity_source

        sources = {
            "preparation": preparation_source,
            "training_status": training_status_source,
            "configs": config_sources,
            "training_reports": training_report_sources,
            "training_audits": training_audit_sources,
            "sensitivity": sensitivity_sources,
        }
        expected = build_postevaluation(
            preparation=preparation,
            training_status=training_status,
            training_reports=training_reports,
            training_audits=training_audits,
            expected_configs=expected_configs,
            sensitivity_reports=sensitivity_reports,
            sources=sources,
            git=git_provenance(PROJECT_ROOT),
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "Postevaluation output exists; pass --resume to validate it"
                )
            existing = read_json_object(output, name="ranking probe postevaluation")
            if existing != expected:
                raise ValueError("Completed postevaluation differs from current sources")
        else:
            write_json_report(output, expected)
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
