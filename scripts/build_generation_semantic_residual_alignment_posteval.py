from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.semantic_residual_alignment_probe import (
    CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG,
    CLAIM_BOUNDARY,
    CONFIG_PATHS,
    CONTROL_RESIDUAL_ALIGNMENT_CONFIG,
    EXECUTION_BOUNDARY,
    METHOD_ARMS,
    OUTPUT_ROOT,
    POSTEVALUATION_ROLE,
    PREPARATION_CONFIG_KEYS,
    RUN_NAMES,
    TRAINING_STATUS_ROLE,
    validate_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_conditioning_ranking_probe_posteval import (
    MAX_CORRECT_MSE_RATIO,
    SIGNIFICANCE_LEVEL,
    _is_sha256,
    _load_json_source,
    _load_sensitivity_source,
    _method_postevaluation,
    _validate_sensitivity_reports,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REQUEST = {
    "weights": "ema",
    "num_samples": 8,
    "start_label": 128,
    "wrong_label_offset": 250,
    "timesteps": [100, 500, 700, 900],
    "noise_seed": 314159,
    "threads": 2,
}
EXPECTED_STEPS = 1_000
EXPECTED_CHECKPOINT_STEPS = [500, 750, 1_000]
EXPECTED_CHECKPOINT_INTERVAL = 250
EXPECTED_EVALUATION_INTERVAL = 250
EXPECTED_VALIDATION_EVENTS = 4
EXPECTED_DATASET = "imagenet_256_10pct"
EXPECTED_NUM_CLASSES = 1_000
LEGACY_RUN_MAPPING = {
    "control_cofitok": "control_cofitok",
    "ranked_cofitok": "residual_cofitok",
    "control_dense_identity": "control_dense_identity",
    "ranked_dense_identity": "residual_dense_identity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound CPU-only held-out postevaluation for the "
            "semantic residual-alignment four-arm probe."
        )
    )
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--training-status", required=True)
    for run in RUN_NAMES:
        option = run.replace("_", "-")
        parser.add_argument(f"--{option}-audit", required=True)
        parser.add_argument(f"--{option}-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_preparation_report(
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    git = preparation.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("residual-alignment preparation Git is missing")
    validated = validate_preparation(
        preparation,
        expected_revision=str(git.get("revision", "")),
        expected_tree=str(git.get("tree", "")),
        expected_branch=str(git.get("branch", "")),
        expected_output_root=OUTPUT_ROOT,
    )
    configs = validated.get("configs")
    counts = validated.get("parameter_counts")
    expected_keys = set(PREPARATION_CONFIG_KEYS.values())
    if (
        not isinstance(configs, Mapping)
        or set(configs) != expected_keys
        or not isinstance(counts, Mapping)
        or set(counts) != expected_keys
    ):
        raise ValueError("residual-alignment preparation config inventory differs")
    for name in expected_keys:
        descriptor = configs[name]
        if (
            not isinstance(descriptor, Mapping)
            or int(descriptor.get("bytes", 0)) < 1
            or not _is_sha256(descriptor.get("sha256"))
            or int(counts.get(name, 0)) < 1
        ):
            raise ValueError(f"residual-alignment preparation config differs: {name}")
    if int(counts["control_cofitok"]) != int(counts["residual_cofitok"]):
        raise ValueError("CoFiTok control/residual parameter counts differ")
    if int(counts["control_dense"]) != int(counts["residual_dense"]):
        raise ValueError("dense control/residual parameter counts differ")
    return {
        "revision": git["revision"],
        "tree": git["tree"],
        "branch": git["branch"],
        "output_root": OUTPUT_ROOT,
        "configs": {name: dict(configs[name]) for name in sorted(configs)},
        "parameter_counts": {name: int(counts[name]) for name in counts},
        "residual_min_timestep": int(
            validated["candidate_residual_alignment_config"][
                "class_conditioning_residual_alignment_min_timestep"
            ]
        ),
    }


def _load_expected_configs(
    preparation_contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    identities: dict[str, Any] = {}
    for run, relative in CONFIG_PATHS.items():
        path = reject_symlink_chain(PROJECT_ROOT / relative, name=f"{run} config")
        identity = file_identity(path)
        preparation_key = PREPARATION_CONFIG_KEYS[run]
        expected = preparation_contract["configs"][preparation_key]
        if any(
            identity[field] != expected.get(field) for field in ("bytes", "sha256")
        ):
            raise ValueError(f"{run} config differs from preparation")
        configs[run] = config_to_dict(load_config(path))
        identities[run] = identity
    return configs, identities


def _validate_authorization(
    authorization: Mapping[str, Any],
    *,
    preparation_source: Mapping[str, Any],
    preparation_contract: Mapping[str, Any],
) -> None:
    sources = authorization.get("source_reports")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("role")
        != "generation_semantic_residual_alignment_execution_authorization"
        or authorization.get("status") != "authorized"
        or authorization.get("output_root") != OUTPUT_ROOT
        or authorization.get("execution_boundary") != EXECUTION_BOUNDARY
        or authorization.get("claim_boundary") != CLAIM_BOUNDARY
        or authorization.get("generation_advantage_proven") is not False
        or not isinstance(sources, Mapping)
        or sources.get("preparation") != preparation_source
        or authorization.get("authorized_git")
        != {
            "revision": preparation_contract["revision"],
            "tree": preparation_contract["tree"],
            "branch": preparation_contract["branch"],
            "tracked_dirty": False,
        }
    ):
        raise ValueError("residual-alignment execution authorization differs")


def _validate_required_integrity(
    audit: Mapping[str, Any],
    *,
    run: str,
    revision: str,
    branch: str,
    dataset_sha: str,
    runtime_sha: str,
) -> list[dict[str, Any]]:
    checkpoint = audit.get("checkpoint")
    required = checkpoint.get("required_integrity") if isinstance(checkpoint, Mapping) else None
    rows = required.get("checkpoints") if isinstance(required, Mapping) else None
    if (
        not isinstance(checkpoint, Mapping)
        or int(checkpoint.get("interval", -1)) != EXPECTED_CHECKPOINT_INTERVAL
        or checkpoint.get("status") != "available"
        or checkpoint.get("required_steps") != EXPECTED_CHECKPOINT_STEPS
        or checkpoint.get("missing_required_steps") != []
        or not isinstance(required, Mapping)
        or required.get("policy") != "required"
        or required.get("status") != "verified"
        or required.get("requested_steps") != EXPECTED_CHECKPOINT_STEPS
        or required.get("reached_steps") != EXPECTED_CHECKPOINT_STEPS
        or not isinstance(rows, list)
        or [row.get("step") for row in rows] != EXPECTED_CHECKPOINT_STEPS
    ):
        raise ValueError(f"{run} required checkpoint integrity differs")
    for row in rows:
        if (
            row.get("status") != "verified"
            or int(row.get("checkpoint_bytes", 0)) < 1
            or not _is_sha256(row.get("checkpoint_sha256"))
            or row.get("git_revision") != revision
            or row.get("git_branch") != branch
            or row.get("git_dirty") is not False
            or row.get("dataset_identity_sha256") != dataset_sha
            or row.get("runtime_environment_sha256") != runtime_sha
        ):
            raise ValueError(f"{run} checkpoint lineage differs at step {row.get('step')}")
    return [dict(row) for row in rows]


def _validate_training_evidence(
    *,
    preparation: Mapping[str, Any],
    training_status: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    training_audits: Mapping[str, Mapping[str, Any]],
    expected_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _validate_preparation_report(preparation)
    if (
        training_status.get("schema_version") != 1
        or training_status.get("role") != TRAINING_STATUS_ROLE
        or training_status.get("status") != "completed"
        or training_status.get("revision") != contract["revision"]
        or training_status.get("tree") != contract["tree"]
        or training_status.get("branch") != contract["branch"]
        or set(training_status.get("runs", {})) != set(RUN_NAMES)
        or training_status.get("execution_boundary") != EXECUTION_BOUNDARY
        or training_status.get("claim_boundary") != CLAIM_BOUNDARY
        or training_status.get("generation_advantage_proven") is not False
    ):
        raise ValueError("residual-alignment training status differs")

    runs: dict[str, Any] = {}
    for run in RUN_NAMES:
        status_run = training_status["runs"][run]
        report = training_reports.get(run)
        audit = training_audits.get(run)
        config = expected_configs[run]
        if not all(isinstance(value, Mapping) for value in (status_run, report, audit)):
            raise ValueError(f"{run} training evidence is missing")
        run_dir = f"{OUTPUT_ROOT}/{run}"
        report_path = f"{run_dir}/training_report.json"
        model = config["model"]
        data = config["data"]
        optimization = config["optimization"]
        loss = config["loss"]
        expected_samples = (
            EXPECTED_STEPS
            * int(data["batch_size"])
            * int(optimization["gradient_accumulation_steps"])
        )
        config_key = PREPARATION_CONFIG_KEYS[run]
        expected_parameters = contract["parameter_counts"][config_key]
        if (
            status_run.get("training_report") != report_path
            or report.get("training_complete") is not True
            or int(report.get("completed_steps", -1)) != EXPECTED_STEPS
            or int(report.get("target_steps", -1)) != EXPECTED_STEPS
            or report.get("output_dir") != run_dir
            or report.get("config") != config
            or report.get("git")
            != {
                "revision": contract["revision"],
                "branch": contract["branch"],
                "dirty": False,
            }
            or int(report.get("parameter_count", -1)) != expected_parameters
            or status_run.get("final_metrics") != report.get("final_metrics")
        ):
            raise ValueError(f"{run} training report differs")
        latest = report.get("latest_checkpoint")
        final = report.get("final_metrics")
        if not isinstance(latest, Mapping) or not isinstance(final, Mapping):
            raise ValueError(f"{run} final training evidence is missing")
        dataset_sha = str(latest.get("dataset_identity_sha256", ""))
        runtime_sha = str(latest.get("runtime_environment_sha256", ""))
        expected_scale = 1.0 if run.startswith("residual_") else 0.0
        if (
            int(latest.get("step", -1)) != EXPECTED_STEPS
            or latest.get("checkpoint") != "checkpoint_step_00001000.pt"
            or latest.get("integrity_manifest")
            != "checkpoint_step_00001000.pt.integrity.json"
            or int(latest.get("checkpoint_bytes", 0)) < 1
            or not _is_sha256(latest.get("checkpoint_sha256"))
            or latest.get("checkpoint_sha256") != status_run.get("checkpoint_sha256")
            or not _is_sha256(dataset_sha)
            or not _is_sha256(runtime_sha)
            or report.get("runtime_environment_sha256") != runtime_sha
            or report.get("dataset_provenance", {}).get("identity_sha256")
            != dataset_sha
            or int(final.get("step", -1)) != EXPECTED_STEPS
            or int(final.get("samples_seen", -1)) != expected_samples
            or not math.isclose(
                float(final.get("class_conditioning_residual_alignment_scale", math.nan)),
                expected_scale,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isfinite(
                float(final.get("class_conditioning_residual_alignment", math.nan))
            )
            or float(final["class_conditioning_residual_alignment"]) < 0.0
        ):
            raise ValueError(f"{run} final residual-alignment contract differs")
        if (
            audit.get("schema_version") != 2
            or audit.get("status") != "complete"
            or audit.get("issues") != []
            or audit.get("warnings") != []
            or int(audit.get("expected_steps", -1)) != EXPECTED_STEPS
            or int(audit.get("last_step", -1)) != EXPECTED_STEPS
            or float(audit.get("progress_fraction", -1.0)) != 1.0
            or audit.get("run_dir") != run_dir
        ):
            raise ValueError(f"{run} training audit did not complete")
        checkpoints = _validate_required_integrity(
            audit,
            run=run,
            revision=contract["revision"],
            branch=contract["branch"],
            dataset_sha=dataset_sha,
            runtime_sha=runtime_sha,
        )
        validation = audit.get("validation")
        schedules = audit.get("consistency_schedules")
        residual_schedule = (
            schedules.get("schedules", {}).get(
                "class_conditioning_residual_alignment"
            )
            if isinstance(schedules, Mapping)
            else None
        )
        expected_objective = (
            CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG
            if run.startswith("residual_")
            else CONTROL_RESIDUAL_ALIGNMENT_CONFIG
        )
        if (
            not isinstance(validation, Mapping)
            or int(validation.get("configured_interval", -1))
            != EXPECTED_EVALUATION_INTERVAL
            or int(validation.get("event_count", -1)) != EXPECTED_VALIDATION_EVENTS
            or int(validation.get("expected_event_count", -1))
            != EXPECTED_VALIDATION_EVENTS
            or validation.get("logging_complete") is not True
            or validation.get("provenance_metadata_status") != "complete"
            or not isinstance(schedules, Mapping)
            or schedules.get("status") != "verified"
            or schedules.get("config_sha256") != contract["configs"][config_key]["sha256"]
            or not isinstance(residual_schedule, Mapping)
            or not math.isclose(
                float(residual_schedule.get("weight", math.nan)),
                float(expected_objective["class_conditioning_residual_alignment_weight"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or int(residual_schedule.get("start_step", -1))
            != int(expected_objective["class_conditioning_residual_alignment_start_step"])
            or int(residual_schedule.get("warmup_steps", -1))
            != int(expected_objective["class_conditioning_residual_alignment_warmup_steps"])
            or int(residual_schedule.get("verified_rows", -1))
            != int(audit.get("metric_row_count", -2))
            or (
                run.startswith("residual_")
                and (
                    int(residual_schedule.get("active_rows", 0)) < 1
                    or int(residual_schedule.get("nonzero_loss_rows", 0)) < 1
                )
            )
        ):
            raise ValueError(f"{run} objective schedule audit differs")
        runs[run] = {
            "run_dir": run_dir,
            "parameter_count": expected_parameters,
            "checkpoint_sha256": latest["checkpoint_sha256"],
            "checkpoint_bytes": int(latest["checkpoint_bytes"]),
            "checkpoint_step": EXPECTED_STEPS,
            "checkpoint_dataset_identity_sha256": dataset_sha,
            "checkpoint_runtime_environment_sha256": runtime_sha,
            "required_checkpoint_integrity": checkpoints,
            "final_metrics": dict(final),
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
    return {**contract, "runs": runs}


def _mapped_legacy_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(contract),
        "runs": {
            legacy: contract["runs"][current]
            for legacy, current in LEGACY_RUN_MAPPING.items()
        },
    }


def _mapped_legacy_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {legacy: reports[current] for legacy, current in LEGACY_RUN_MAPPING.items()}


def build_postevaluation(
    *,
    preparation: Mapping[str, Any],
    preparation_source: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
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
    _validate_authorization(
        execution_authorization,
        preparation_source=preparation_source,
        preparation_contract=training_contract,
    )
    expected_git = {
        "revision": training_contract["revision"],
        "branch": training_contract["branch"],
        "tracked_dirty": False,
    }
    if dict(git) != expected_git:
        raise ValueError("postevaluation requires the exact clean training checkout")

    mapped_contract = _mapped_legacy_contract(training_contract)
    mapped_reports = _mapped_legacy_reports(sensitivity_reports)
    sensitivity_contract = _validate_sensitivity_reports(
        reports=mapped_reports,
        training_contract=mapped_contract,
        evaluator_git=git,
        expected_request=EXPECTED_REQUEST,
        expected_dataset=EXPECTED_DATASET,
        expected_num_classes=EXPECTED_NUM_CLASSES,
        expected_steps=EXPECTED_STEPS,
    )
    eligible = [
        timestep
        for timestep in EXPECTED_REQUEST["timesteps"]
        if timestep >= training_contract["residual_min_timestep"]
    ]
    descriptive = [
        timestep
        for timestep in EXPECTED_REQUEST["timesteps"]
        if timestep < training_contract["residual_min_timestep"]
    ]
    if not eligible or not descriptive:
        raise ValueError("postevaluation requires eligible and descriptive timesteps")
    methods = {
        method: _method_postevaluation(
            control=sensitivity_reports[arms[0]],
            ranked=sensitivity_reports[arms[1]],
            eligible_timesteps=eligible,
            expected_num_samples=EXPECTED_REQUEST["num_samples"],
        )
        for method, arms in METHOD_ARMS.items()
    }
    passes = {method: report["pass"] for method, report in methods.items()}
    shared_recovery = all(passes.values())
    if shared_recovery:
        next_action = "consider_separately_authorized_matched_sampling_validation"
    elif any(passes.values()):
        next_action = "reject_shared_repair_due_method_asymmetry"
    else:
        next_action = "reject_residual_alignment_candidate"
    difference_in_differences = {
        comparison: (
            methods["cofitok"]["ranked_minus_control"][comparison]["mean"]
            - methods["dense_identity"]["ranked_minus_control"][comparison]["mean"]
        )
        for comparison in ("versus_wrong", "versus_null")
    }
    return {
        "schema_version": 1,
        "role": POSTEVALUATION_ROLE,
        "status": "completed",
        "git": dict(git),
        "sources": dict(sources),
        "training_contract": training_contract,
        "sensitivity_contract": sensitivity_contract,
        "evaluation_contract": {
            "request": dict(EXPECTED_REQUEST),
            "eligible_timesteps": eligible,
            "descriptive_only_timesteps": descriptive,
            "sample_count": EXPECTED_REQUEST["num_samples"],
            "wrong_label_offset_is_held_out_from_training": True,
            "training_wrong_label_offsets": [1, 500],
            "significance_level": SIGNIFICANCE_LEVEL,
            "maximum_correct_mse_ratio": MAX_CORRECT_MSE_RATIO,
            "paired_unit": "held_out_validation_image",
            "timestep_rows_are_not_independent_units": True,
        },
        "arm_semantics": {
            "control": "existing matched 1K control objective",
            "ranked_fields_in_reused_statistics": (
                "semantic residual-alignment candidate arm"
            ),
        },
        "methods": methods,
        "difference_in_differences": difference_in_differences,
        "difference_in_differences_is_descriptive_only": True,
        "decision": {
            "method_passes": passes,
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
            "authorizes_followup_training": False,
            "authorizes_full_training": False,
            "authorizes_300k_training": False,
            "authorizes_export": False,
            "authorizes_release": False,
            "authorizes_process_signals": False,
            "replaces_formal_quality_gate": False,
        },
        "generation_advantage_proven": False,
    }


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(
        args.output,
        name="residual-alignment postevaluation output",
    )
    with exclusive_output_lock(output, role=POSTEVALUATION_ROLE):
        preparation, preparation_source = _load_json_source(
            args.preparation_report,
            label="residual-alignment preparation",
        )
        authorization, authorization_source = _load_json_source(
            args.execution_authorization,
            label="residual-alignment execution authorization",
        )
        training_status, training_status_source = _load_json_source(
            args.training_status,
            label="residual-alignment training status",
        )
        preparation_contract = _validate_preparation_report(preparation)
        expected_configs, config_sources = _load_expected_configs(preparation_contract)
        training_reports: dict[str, dict[str, Any]] = {}
        training_report_sources: dict[str, Any] = {}
        training_audits: dict[str, dict[str, Any]] = {}
        training_audit_sources: dict[str, Any] = {}
        sensitivity_reports: dict[str, dict[str, Any]] = {}
        sensitivity_sources: dict[str, Any] = {}
        for run in RUN_NAMES:
            status_run = training_status.get("runs", {}).get(run)
            if not isinstance(status_run, Mapping):
                raise ValueError(f"training status lacks run {run}")
            report, report_source = _load_json_source(
                status_run.get("training_report", ""),
                label=f"{run} training report",
            )
            audit, audit_source = _load_json_source(
                getattr(args, f"{run}_audit"),
                label=f"{run} training audit",
            )
            sensitivity, sensitivity_source = _load_sensitivity_source(
                getattr(args, f"{run}_report"),
                run=run,
            )
            training_reports[run] = report
            training_report_sources[run] = report_source
            training_audits[run] = audit
            training_audit_sources[run] = audit_source
            sensitivity_reports[run] = sensitivity
            sensitivity_sources[run] = sensitivity_source
        sources = {
            "preparation": preparation_source,
            "execution_authorization": authorization_source,
            "training_status": training_status_source,
            "configs": config_sources,
            "training_reports": training_report_sources,
            "training_audits": training_audit_sources,
            "sensitivity": sensitivity_sources,
        }
        expected = build_postevaluation(
            preparation=preparation,
            preparation_source=preparation_source,
            execution_authorization=authorization,
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
                    "postevaluation output exists; pass --resume to replay it"
                )
            existing = read_json_object(output, name="residual-alignment postevaluation")
            if existing != expected:
                raise ValueError("completed postevaluation differs from current sources")
        else:
            write_json_report(output, expected)
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
