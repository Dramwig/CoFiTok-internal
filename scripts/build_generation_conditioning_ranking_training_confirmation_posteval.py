from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_training_confirmation import (
    CLAIM_BOUNDARY as TRAINING_PREPARATION_CLAIM_BOUNDARY,
    CONFIG_RELATIVE_PATHS,
    CONTROL_RANKING_CONFIG,
    EXECUTION_BOUNDARY as TRAINING_PREPARATION_EXECUTION_BOUNDARY,
    EXPECTED_OUTPUT_ROOT,
    PREPARATION_ROLE,
    RANKED_RANKING_CONFIG,
    RUN_NAMES,
    SCOPE as TRAINING_SCOPE,
    STAGE as TRAINING_STAGE,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts import build_generation_conditioning_ranking_probe_posteval as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ROLE = "generation_conditioning_ranking_four_arm_train5k_heldout_evaluation"
STAGE = "conditioning_ranking_four_arm_train5k_heldout_evaluation_v1"
TRAINING_STATUS_ROLE = (
    "generation_conditioning_ranking_four_arm_train5k_confirmation_status"
)
TRAINING_GIT = {
    "revision": "ed9463eb85ffb97545b5506e264123e46ff2fcfc",
    "branch": "scale/generation-label-ranking-5k-training-confirmation-v1",
    "tracked_dirty": False,
}
METHOD_ARMS = {
    "cofitok": ("control_cofitok", "ranked_cofitok"),
    "dense_identity": ("control_dense_identity", "ranked_dense_identity"),
}
EXPECTED_REQUEST = {
    "weights": "ema",
    "num_samples": 16,
    "start_label": 192,
    "wrong_label_offset": 500,
    "timesteps": [100, 500, 900],
    "noise_seed": 304060,
    "threads": 2,
}
EXPECTED_DATASET = "imagenet_256_10pct"
EXPECTED_NUM_CLASSES = 1_000
EXPECTED_STEPS = 5_000
EXPECTED_CHECKPOINT_FILENAME = "checkpoint_step_00005000.pt"
EXPECTED_CHECKPOINT_STEPS = [1_250, 2_500, 5_000]
EXPECTED_CHECKPOINT_INTERVAL = 1_250
EXPECTED_EVALUATION_INTERVAL = 1_250
EXPECTED_VALIDATION_EVENTS = 4
SIGNIFICANCE_LEVEL = 0.05
MAX_CORRECT_MSE_RATIO = 1.02
TRAINING_STATUS_EXECUTION_BOUNDARY = {
    "fresh_four_arm_training_completed": True,
    "steps_per_run": 5_000,
    "effective_batch_size": 64,
    "automatic_relaunch_allowed": False,
    "sampling_allowed": False,
    "checkpoint_promotion_allowed": False,
    "followup_training_allowed": False,
    "full_100k_or_300k_launch_allowed": False,
    "release_authorization_allowed": False,
}
TRAINING_STATUS_CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "training_quality_claim_allowed": False,
    "sample_quality_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "authorizes_followup_evaluation": False,
    "authorizes_followup_training": False,
    "authorizes_full_100k_or_300k": False,
    "authorizes_release": False,
    "required_next_evidence": (
        "separately_source_bound_matched_5k_heldout_evaluation"
    ),
}
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "generates_new_samples": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_followup_training": False,
    "authorizes_full_100k_or_300k": False,
    "authorizes_release": False,
    "cofitok_specific_advantage_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "replaces_formal_quality_gate": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the independently held-out CPU-only decision for the exact "
            "four-arm 5K class-conditioning-ranking training confirmation."
        )
    )
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--training-status", required=True)
    for run in RUN_NAMES:
        option = run.replace("_", "-")
        parser.add_argument(f"--{option}-audit", required=True)
        parser.add_argument(f"--{option}-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not _is_sha256(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _load_json_source(
    path: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    return read_json_object(source, name=label), file_identity(source)


def _load_expected_configs(
    preparation: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    preparation_configs = preparation.get("configs")
    if not isinstance(preparation_configs, Mapping):
        raise ValueError("Training preparation config identities are malformed")
    configs: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for run, relative_path in CONFIG_RELATIVE_PATHS.items():
        path = reject_symlink_chain(PROJECT_ROOT / relative_path, name=f"{run} config")
        identity = file_identity(path)
        expected = preparation_configs.get(run)
        if not isinstance(expected, Mapping) or any(
            identity[field] != expected.get(field) for field in ("bytes", "sha256")
        ):
            raise ValueError(f"{run} config differs from the training preparation")
        configs[run] = config_to_dict(load_config(path))
        identities[run] = identity
    return configs, identities


def _validate_preparation(
    preparation: Mapping[str, Any],
    *,
    config_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parameter_counts = preparation.get("parameter_counts")
    preparation_configs = preparation.get("configs")
    if (
        preparation.get("schema_version") != SCHEMA_VERSION
        or preparation.get("role") != PREPARATION_ROLE
        or preparation.get("status") != "pass"
        or preparation.get("valid") is not True
        or preparation.get("stage") != TRAINING_STAGE
        or preparation.get("scope") != TRAINING_SCOPE
        or preparation.get("issues") != []
        or preparation.get("output_root") != EXPECTED_OUTPUT_ROOT
        or preparation.get("git") != TRAINING_GIT
        or preparation.get("execution_boundary")
        != TRAINING_PREPARATION_EXECUTION_BOUNDARY
        or preparation.get("claim_boundary")
        != TRAINING_PREPARATION_CLAIM_BOUNDARY
        or preparation.get("control_ranking_config")
        != CONTROL_RANKING_CONFIG
        or preparation.get("ranked_ranking_config")
        != RANKED_RANKING_CONFIG
        or preparation.get("gpu_execution_authorized") is not False
        or preparation.get("authorization_required") is not True
        or not isinstance(parameter_counts, Mapping)
        or set(parameter_counts) != set(RUN_NAMES)
        or not isinstance(preparation_configs, Mapping)
        or set(preparation_configs) != set(RUN_NAMES)
        or set(config_identities) != set(RUN_NAMES)
    ):
        raise ValueError("5K training-confirmation preparation is malformed")
    for run in RUN_NAMES:
        if (
            isinstance(parameter_counts[run], bool)
            or not isinstance(parameter_counts[run], int)
            or int(parameter_counts[run]) <= 0
        ):
            raise ValueError(f"{run} parameter count is invalid")
        expected_identity = _identity(
            preparation_configs[run],
            label=f"{run} training config",
        )
        current_identity = config_identities[run]
        if any(
            current_identity[field] != expected_identity[field]
            for field in ("bytes", "sha256")
        ):
            raise ValueError(f"{run} config identity drifted")
    if (
        parameter_counts["control_cofitok"] != parameter_counts["ranked_cofitok"]
        or parameter_counts["control_dense_identity"]
        != parameter_counts["ranked_dense_identity"]
    ):
        raise ValueError("Training control/ranked parameter counts differ")
    return {
        "revision": TRAINING_GIT["revision"],
        "branch": TRAINING_GIT["branch"],
        "output_root": EXPECTED_OUTPUT_ROOT,
        "parameter_counts": {
            run: int(parameter_counts[run]) for run in RUN_NAMES
        },
        "config_identities": {
            run: _identity(preparation_configs[run], label=f"{run} config")
            for run in RUN_NAMES
        },
        "ranking_min_timestep": int(
            RANKED_RANKING_CONFIG["class_conditioning_ranking_min_timestep"]
        ),
    }


def _validate_status_identity(
    claimed: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if _identity(claimed, label=label) != _identity(actual, label=f"physical {label}"):
        raise ValueError(f"{label} identity differs from training status")


def _validate_training_evidence(
    *,
    preparation: Mapping[str, Any],
    training_status: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    training_report_sources: Mapping[str, Mapping[str, Any]],
    training_audits: Mapping[str, Mapping[str, Any]],
    training_audit_sources: Mapping[str, Mapping[str, Any]],
    checkpoint_sources: Mapping[str, Mapping[str, Any]],
    checkpoint_integrity_sources: Mapping[str, Mapping[str, Any]],
    expected_configs: Mapping[str, Mapping[str, Any]],
    config_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _validate_preparation(
        preparation,
        config_identities=config_identities,
    )
    runs = training_status.get("runs")
    if (
        training_status.get("schema_version") != SCHEMA_VERSION
        or training_status.get("role") != TRAINING_STATUS_ROLE
        or training_status.get("status") != "completed"
        or training_status.get("stage") != TRAINING_STAGE
        or training_status.get("revision") != TRAINING_GIT["revision"]
        or training_status.get("branch") != TRAINING_GIT["branch"]
        or training_status.get("output_root") != EXPECTED_OUTPUT_ROOT
        or training_status.get("execution_boundary")
        != TRAINING_STATUS_EXECUTION_BOUNDARY
        or training_status.get("claim_boundary") != TRAINING_STATUS_CLAIM_BOUNDARY
        or not isinstance(runs, Mapping)
        or set(runs) != set(RUN_NAMES)
        or set(checkpoint_sources) != set(RUN_NAMES)
        or set(checkpoint_integrity_sources) != set(RUN_NAMES)
    ):
        raise ValueError("Four-arm 5K training status is not exact completed evidence")

    result: dict[str, Any] = {}
    for run in RUN_NAMES:
        status_run = runs[run]
        report = training_reports.get(run)
        audit = training_audits.get(run)
        config = expected_configs.get(run)
        if not all(isinstance(value, Mapping) for value in (status_run, report, audit, config)):
            raise ValueError(f"{run} training evidence is incomplete")
        assert isinstance(status_run, Mapping)
        assert isinstance(report, Mapping)
        assert isinstance(audit, Mapping)
        assert isinstance(config, Mapping)
        _validate_status_identity(
            status_run.get("training_report", {}),
            training_report_sources[run],
            label=f"{run} training report",
        )
        _validate_status_identity(
            status_run.get("training_audit", {}),
            training_audit_sources[run],
            label=f"{run} training audit",
        )
        run_dir = f"{EXPECTED_OUTPUT_ROOT}/{run}"
        latest = report.get("latest_checkpoint")
        final_metrics = report.get("final_metrics")
        report_git = report.get("git")
        expected_parameter_count = contract["parameter_counts"][run]
        expected_ranking_scale = 1.0 if run.startswith("ranked_") else 0.0
        effective_batch_size = int(config["data"]["batch_size"]) * int(
            config["optimization"]["gradient_accumulation_steps"]
        )
        expected_samples_seen = EXPECTED_STEPS * effective_batch_size
        if (
            status_run.get("run_dir") != run_dir
            or report.get("training_complete") is not True
            or int(report.get("completed_steps", -1)) != EXPECTED_STEPS
            or int(report.get("target_steps", -1)) != EXPECTED_STEPS
            or report.get("output_dir") != run_dir
            or report.get("config") != config
            or report_git
            != {
                "revision": TRAINING_GIT["revision"],
                "branch": TRAINING_GIT["branch"],
                "dirty": False,
            }
            or int(report.get("parameter_count", -1)) != expected_parameter_count
            or not isinstance(latest, Mapping)
            or latest.get("checkpoint") != EXPECTED_CHECKPOINT_FILENAME
            or int(latest.get("step", -1)) != EXPECTED_STEPS
            or int(latest.get("checkpoint_bytes", 0)) <= 0
            or not _is_sha256(latest.get("checkpoint_sha256"))
            or latest.get("integrity_manifest")
            != f"{EXPECTED_CHECKPOINT_FILENAME}.integrity.json"
            or not _is_sha256(latest.get("dataset_identity_sha256"))
            or not _is_sha256(latest.get("runtime_environment_sha256"))
            or not isinstance(final_metrics, Mapping)
            or int(final_metrics.get("step", -1)) != EXPECTED_STEPS
            or effective_batch_size != 64
            or int(final_metrics.get("samples_seen", -1))
            != expected_samples_seen
            or not math.isclose(
                float(final_metrics.get("class_conditioning_ranking_scale", math.nan)),
                expected_ranking_scale,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(f"{run} training report differs from the 5K contract")
        for field in ("epsilon", "class_conditioning_ranking"):
            value = float(final_metrics.get(field, math.nan))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{run} final metric {field} is invalid")
        if status_run.get("final_metrics") != final_metrics:
            raise ValueError(f"{run} final metrics differ from training status")

        checkpoint_identity = _identity(
            checkpoint_sources[run],
            label=f"{run} physical terminal checkpoint",
        )
        integrity_identity = _identity(
            checkpoint_integrity_sources[run],
            label=f"{run} physical checkpoint integrity manifest",
        )
        if checkpoint_identity["path"] != (
            f"{run_dir}/{EXPECTED_CHECKPOINT_FILENAME}"
        ):
            raise ValueError(f"{run} terminal checkpoint path differs")
        if integrity_identity["path"] != (
            f"{run_dir}/{EXPECTED_CHECKPOINT_FILENAME}.integrity.json"
        ):
            raise ValueError(f"{run} checkpoint integrity path differs")
        _validate_status_identity(
            status_run.get("checkpoint", {}),
            checkpoint_identity,
            label=f"{run} checkpoint",
        )
        _validate_status_identity(
            status_run.get("integrity_manifest", {}),
            integrity_identity,
            label=f"{run} checkpoint integrity",
        )
        if (
            checkpoint_identity["bytes"] != int(latest["checkpoint_bytes"])
            or checkpoint_identity["sha256"] != latest["checkpoint_sha256"]
        ):
            raise ValueError(f"{run} checkpoint differs from training report")

        checkpoint_audit = audit.get("checkpoint")
        audit_latest = (
            checkpoint_audit.get("latest")
            if isinstance(checkpoint_audit, Mapping)
            else None
        )
        audit_integrity = (
            checkpoint_audit.get("latest_integrity")
            if isinstance(checkpoint_audit, Mapping)
            else None
        )
        validation = audit.get("validation")
        schedules = audit.get("consistency_schedules")
        audit_report = audit.get("training_report")
        schedule_rows = (
            schedules.get("schedules") if isinstance(schedules, Mapping) else None
        )
        metric_row_count = int(audit.get("metric_row_count", -1))
        if (
            audit.get("schema_version") != 2
            or audit.get("status") != "complete"
            or audit.get("issues") != []
            or int(audit.get("expected_steps", -1)) != EXPECTED_STEPS
            or int(audit.get("last_step", -1)) != EXPECTED_STEPS
            or float(audit.get("progress_fraction", -1.0)) != 1.0
            or audit.get("run_dir") != run_dir
            or not isinstance(checkpoint_audit, Mapping)
            or checkpoint_audit.get("status") != "available"
            or int(checkpoint_audit.get("interval", -1))
            != EXPECTED_CHECKPOINT_INTERVAL
            or checkpoint_audit.get("required_steps") != EXPECTED_CHECKPOINT_STEPS
            or checkpoint_audit.get("missing_required_steps") != []
            or not isinstance(audit_latest, Mapping)
            or int(audit_latest.get("step", -1)) != EXPECTED_STEPS
            or audit_latest.get("checkpoint_sha256")
            != checkpoint_identity["sha256"]
            or not isinstance(audit_integrity, Mapping)
            or audit_integrity.get("status") != "verified"
            or audit_integrity.get("checkpoint_sha256")
            != checkpoint_identity["sha256"]
            or not isinstance(validation, Mapping)
            or int(validation.get("configured_interval", -1))
            != EXPECTED_EVALUATION_INTERVAL
            or int(validation.get("event_count", -1))
            != EXPECTED_VALIDATION_EVENTS
            or int(validation.get("expected_event_count", -1))
            != EXPECTED_VALIDATION_EVENTS
            or validation.get("logging_complete") is not True
            or validation.get("provenance_metadata_status") != "complete"
            or not isinstance(schedules, Mapping)
            or schedules.get("status") != "verified"
            or schedules.get("config_sha256")
            != contract["config_identities"][run]["sha256"]
            or not isinstance(schedule_rows, Mapping)
            or metric_row_count < 1
            or not isinstance(audit_report, Mapping)
            or audit_report.get("path") != f"{run_dir}/training_report.json"
            or audit_report.get("status") != "current"
            or int(audit_report.get("completed_steps", -1)) != EXPECTED_STEPS
            or audit_report.get("training_complete") is not True
        ):
            raise ValueError(f"{run} training audit differs from the 5K contract")

        loss = config["loss"]
        for schedule_name in (
            "rollout_consistency",
            "ema_teacher_consistency",
            "class_conditioning_ranking",
        ):
            schedule = schedule_rows.get(schedule_name)
            if not isinstance(schedule, Mapping):
                raise ValueError(f"{run} lacks schedule audit {schedule_name}")
            if (
                not math.isclose(
                    float(schedule.get("weight", math.nan)),
                    float(loss[f"{schedule_name}_weight"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or int(schedule.get("start_step", -1))
                != int(loss[f"{schedule_name}_start_step"])
                or int(schedule.get("warmup_steps", -1))
                != int(loss[f"{schedule_name}_warmup_steps"])
                or int(schedule.get("verified_rows", -1)) != metric_row_count
            ):
                raise ValueError(f"{run} schedule audit {schedule_name} differs")

        model = config["model"]
        data = config["data"]
        result[run] = {
            "run_dir": run_dir,
            "parameter_count": expected_parameter_count,
            "checkpoint_bytes": checkpoint_identity["bytes"],
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_integrity_manifest": integrity_identity,
            "checkpoint_dataset_identity_sha256": latest[
                "dataset_identity_sha256"
            ],
            "checkpoint_runtime_environment_sha256": latest[
                "runtime_environment_sha256"
            ],
            "final_metrics": copy.deepcopy(dict(final_metrics)),
            "model_contract": {
                "name": config["name"],
                "predictor_type": model["predictor_type"],
                "synthesis_mode": model["synthesis_mode"],
                "token_count": int(model["token_count"]),
                "num_classes": int(model["num_classes"]),
            },
            "dataset_contract": {
                "alias": data["dataset"],
                "root": f'{str(data["root"]).rstrip("/")}/{data["dataset"]}',
            },
            "runtime_seed": int(config["runtime"]["seed"]),
        }
        if result[run]["model_contract"]["num_classes"] != EXPECTED_NUM_CLASSES:
            raise ValueError(f"{run} num_classes differs from the held-out contract")
        if result[run]["dataset_contract"]["alias"] != EXPECTED_DATASET:
            raise ValueError(f"{run} dataset differs from the held-out contract")
    return {**contract, "runs": result}


def build_heldout_evaluation(
    *,
    preparation: Mapping[str, Any],
    training_status: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    training_report_sources: Mapping[str, Mapping[str, Any]],
    training_audits: Mapping[str, Mapping[str, Any]],
    training_audit_sources: Mapping[str, Mapping[str, Any]],
    checkpoint_sources: Mapping[str, Mapping[str, Any]],
    checkpoint_integrity_sources: Mapping[str, Mapping[str, Any]],
    expected_configs: Mapping[str, Mapping[str, Any]],
    config_identities: Mapping[str, Mapping[str, Any]],
    sensitivity_reports: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Any],
    evaluator_git: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not _is_git_revision(evaluator_git.get("revision"))
        or not isinstance(evaluator_git.get("branch"), str)
        or not str(evaluator_git["branch"])
        or evaluator_git.get("tracked_dirty") is not False
        or evaluator_git.get("revision") == TRAINING_GIT["revision"]
        or evaluator_git.get("branch") == TRAINING_GIT["branch"]
    ):
        raise ValueError("Held-out evaluator Git must be separate, exact, and clean")
    training_contract = _validate_training_evidence(
        preparation=preparation,
        training_status=training_status,
        training_reports=training_reports,
        training_report_sources=training_report_sources,
        training_audits=training_audits,
        training_audit_sources=training_audit_sources,
        checkpoint_sources=checkpoint_sources,
        checkpoint_integrity_sources=checkpoint_integrity_sources,
        expected_configs=expected_configs,
        config_identities=config_identities,
    )
    sensitivity_contract = base._validate_sensitivity_reports(
        reports=sensitivity_reports,
        training_contract=training_contract,
        evaluator_git=evaluator_git,
        expected_request=EXPECTED_REQUEST,
        expected_dataset=EXPECTED_DATASET,
        expected_num_classes=EXPECTED_NUM_CLASSES,
        expected_steps=EXPECTED_STEPS,
        expected_checkpoint_filename=EXPECTED_CHECKPOINT_FILENAME,
    )
    min_timestep = int(training_contract["ranking_min_timestep"])
    eligible_timesteps = [
        timestep
        for timestep in EXPECTED_REQUEST["timesteps"]
        if timestep >= min_timestep
    ]
    descriptive_timesteps = [
        timestep
        for timestep in EXPECTED_REQUEST["timesteps"]
        if timestep < min_timestep
    ]
    if not eligible_timesteps or not descriptive_timesteps:
        raise ValueError("Held-out evaluation needs eligible and descriptive timesteps")
    methods = {
        method: base._method_postevaluation(
            control=sensitivity_reports[arms[0]],
            ranked=sensitivity_reports[arms[1]],
            eligible_timesteps=eligible_timesteps,
            expected_num_samples=EXPECTED_REQUEST["num_samples"],
        )
        for method, arms in METHOD_ARMS.items()
    }
    method_passes = {method: result["pass"] for method, result in methods.items()}
    shared_recovery = all(method_passes.values())
    if shared_recovery:
        next_action = (
            "prepare_separately_source_bound_posttraining_5k_sampling_confirmation"
        )
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
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "completed",
        "stage": STAGE,
        "training_git": copy.deepcopy(TRAINING_GIT),
        "evaluator_git": copy.deepcopy(dict(evaluator_git)),
        "output_root": EXPECTED_OUTPUT_ROOT,
        "sources": copy.deepcopy(dict(sources)),
        "training_contract": training_contract,
        "sensitivity_contract": sensitivity_contract,
        "evaluation_contract": {
            "request": copy.deepcopy(EXPECTED_REQUEST),
            "eligible_timesteps": eligible_timesteps,
            "descriptive_only_timesteps": descriptive_timesteps,
            "sample_count": EXPECTED_REQUEST["num_samples"],
            "significance_level": SIGNIFICANCE_LEVEL,
            "maximum_correct_mse_ratio": MAX_CORRECT_MSE_RATIO,
            "paired_unit": "held_out_validation_image",
            "independent_from_probe1k": {
                "probe1k_start_label": 64,
                "probe1k_noise_seed": 204060,
                "train5k_start_label": EXPECTED_REQUEST["start_label"],
                "train5k_noise_seed": EXPECTED_REQUEST["noise_seed"],
            },
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
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
    }


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(
        args.output,
        name="5K held-out evaluation output",
    )
    if output.exists() and not output.is_file():
        raise ValueError(f"Held-out evaluation output is not a file: {output}")
    with exclusive_output_lock(output, role=ROLE):
        preparation, preparation_source = _load_json_source(
            args.preparation_report,
            label="5K training-confirmation preparation",
        )
        training_status, training_status_source = _load_json_source(
            args.training_status,
            label="5K training-confirmation status",
        )
        expected_configs, config_sources = _load_expected_configs(preparation)
        training_reports: dict[str, dict[str, Any]] = {}
        training_report_sources: dict[str, dict[str, Any]] = {}
        training_audits: dict[str, dict[str, Any]] = {}
        training_audit_sources: dict[str, dict[str, Any]] = {}
        checkpoint_sources: dict[str, dict[str, Any]] = {}
        checkpoint_integrity_sources: dict[str, dict[str, Any]] = {}
        sensitivity_reports: dict[str, dict[str, Any]] = {}
        sensitivity_sources: dict[str, dict[str, Any]] = {}
        status_runs = training_status.get("runs")
        if not isinstance(status_runs, Mapping):
            raise ValueError("5K training status runs are missing")
        for run in RUN_NAMES:
            status_run = status_runs.get(run)
            if not isinstance(status_run, Mapping):
                raise ValueError(f"5K training status lacks {run}")
            report_identity = status_run.get("training_report")
            if not isinstance(report_identity, Mapping):
                raise ValueError(f"5K training status lacks {run} report identity")
            report, report_source = _load_json_source(
                str(report_identity.get("path", "")),
                label=f"{run} training report",
            )
            audit, audit_source = _load_json_source(
                getattr(args, f"{run}_audit"),
                label=f"{run} training audit",
            )
            sensitivity, sensitivity_source = base._load_sensitivity_source(
                getattr(args, f"{run}_report"),
                run=run,
            )
            checkpoint_path = reject_symlink_chain(
                Path(EXPECTED_OUTPUT_ROOT) / run / EXPECTED_CHECKPOINT_FILENAME,
                name=f"{run} terminal checkpoint",
            )
            checkpoint_integrity_path = reject_symlink_chain(
                checkpoint_path.with_name(
                    f"{checkpoint_path.name}.integrity.json"
                ),
                name=f"{run} checkpoint integrity manifest",
            )
            if not checkpoint_path.is_file() or not checkpoint_integrity_path.is_file():
                raise FileNotFoundError(
                    f"{run} terminal checkpoint evidence is missing"
                )
            training_reports[run] = report
            training_report_sources[run] = report_source
            training_audits[run] = audit
            training_audit_sources[run] = audit_source
            checkpoint_sources[run] = file_identity(checkpoint_path)
            checkpoint_integrity_sources[run] = file_identity(
                checkpoint_integrity_path
            )
            sensitivity_reports[run] = sensitivity
            sensitivity_sources[run] = sensitivity_source
        sources = {
            "preparation": preparation_source,
            "training_status": training_status_source,
            "configs": config_sources,
            "training_reports": training_report_sources,
            "training_audits": training_audit_sources,
            "checkpoints": checkpoint_sources,
            "checkpoint_integrity_manifests": checkpoint_integrity_sources,
            "sensitivity": sensitivity_sources,
        }
        expected = build_heldout_evaluation(
            preparation=preparation,
            training_status=training_status,
            training_reports=training_reports,
            training_report_sources=training_report_sources,
            training_audits=training_audits,
            training_audit_sources=training_audit_sources,
            checkpoint_sources=checkpoint_sources,
            checkpoint_integrity_sources=checkpoint_integrity_sources,
            expected_configs=expected_configs,
            config_identities=config_sources,
            sensitivity_reports=sensitivity_reports,
            sources=sources,
            evaluator_git=git_provenance(PROJECT_ROOT),
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "Held-out evaluation exists; pass --resume to validate it"
                )
            if read_json_object(output, name="5K held-out evaluation") != expected:
                raise ValueError("Completed held-out evaluation differs from sources")
        else:
            write_json_report(output, expected)
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
