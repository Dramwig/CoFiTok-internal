from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.environment import runtime_environment_sha256
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)
from scripts.build_generation_conditioning_ranking_probe_posteval import (
    METHOD_ARMS,
    RUN_NAMES,
    build_postevaluation,
    _load_expected_configs,
    _load_sensitivity_source,
)
from scripts.evaluate_generation_conditioning_ranking_samples import (
    CLAIM_BOUNDARY as PAIRED_CLASS_CLAIM_BOUNDARY,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_NUM_SAMPLES,
    METHOD_PREFIX_BUDGETS,
    REPORT_ROLE as PAIRED_CLASS_REPORT_ROLE,
    REPORT_SCHEMA_VERSION as PAIRED_CLASS_REPORT_SCHEMA_VERSION,
    paired_class_fidelity_summary,
    validate_sampling_pair,
)
from scripts.evaluate_generation_metrics import (
    GENERATION_METRICS_REPORT_ROLE,
    GENERATION_METRICS_REPORT_SCHEMA_VERSION,
    find_images,
    validate_sampling_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_ranking_four_arm_sampling_validation"
POSTEVALUATION_ROLE = "generation_conditioning_ranking_four_arm_postevaluation"
STAGE = "conditioning_ranking_four_arm_sampling5k_v1"
EXPECTED_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "conditioning_ranking_four_arm_sampling5k_v1"
)
SAMPLE_RUN_NAME = "samples_5000_ddim50_cfg15"
MAX_FID_RATIO = 1.10
EXPECTED_REAL_SET = {
    "digest_schema": "cofitok_image_tree_sha256_v1",
    "image_count": 50_000,
    "root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val",
    "sha256": "19ace4e37bee2fcaeac2b7cbd26ed785aa0e6e01015c90b4955027d163f6e44f",
}
ARM_METHOD = {
    "control_cofitok": "cofitok",
    "ranked_cofitok": "cofitok",
    "control_dense_identity": "dense_identity",
    "ranked_dense_identity": "dense_identity",
}
CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "formal_quality_gate": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "authorizes_checkpoint_promotion": False,
    "authorizes_full_training": False,
    "authorizes_full_100k_or_300k": False,
    "authorizes_release": False,
    "replaces_active_quality_bridge": False,
    "broad_generation_superiority_claim_allowed": False,
    "cofitok_specific_advantage_claim_allowed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound four-arm 5K generated-sample validation "
            "after the conditioning-ranking probe."
        )
    )
    parser.add_argument("--postevaluation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or type(size) is not int
        or size < 1
        or not _is_sha256(digest)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _clean_git(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not all(character in "0123456789abcdef" for character in revision)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _load_json_source(
    path: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    return read_json_object(source, name=label), file_identity(source)


def _load_embedded_json(
    descriptor: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _identity(descriptor, label=label)
    report, identity = _load_json_source(expected["path"], label=label)
    if identity != expected:
        raise ValueError(f"{label} identity differs")
    return report, identity


def replay_postevaluation(
    path: str | Path,
    *,
    require_sampling_selected: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    report, report_identity = _load_json_source(
        path,
        label="conditioning-ranking 1K postevaluation",
    )
    if (
        report.get("schema_version") != 1
        or report.get("role") != POSTEVALUATION_ROLE
        or report.get("status") != "completed"
    ):
        raise ValueError("conditioning-ranking 1K postevaluation is not completed")
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("conditioning-ranking postevaluation sources are missing")
    preparation, preparation_identity = _load_embedded_json(
        sources.get("preparation", {}),
        label="conditioning-ranking preparation",
    )
    training_status, training_status_identity = _load_embedded_json(
        sources.get("training_status", {}),
        label="conditioning-ranking training status",
    )
    expected_configs, config_sources = _load_expected_configs(preparation)
    if sources.get("configs") != config_sources:
        raise ValueError("conditioning-ranking config source identities differ")

    training_reports: dict[str, dict[str, Any]] = {}
    training_report_sources: dict[str, Any] = {}
    training_audits: dict[str, dict[str, Any]] = {}
    training_audit_sources: dict[str, Any] = {}
    sensitivity_reports: dict[str, dict[str, Any]] = {}
    sensitivity_sources: dict[str, Any] = {}
    for run in RUN_NAMES:
        training_report, training_report_identity = _load_embedded_json(
            sources.get("training_reports", {}).get(run, {}),
            label=f"{run} training report",
        )
        training_audit, training_audit_identity = _load_embedded_json(
            sources.get("training_audits", {}).get(run, {}),
            label=f"{run} training audit",
        )
        sensitivity_descriptor = sources.get("sensitivity", {}).get(run, {})
        if not isinstance(sensitivity_descriptor, Mapping):
            raise ValueError(f"{run} sensitivity source is missing")
        sensitivity_report_descriptor = sensitivity_descriptor.get("report")
        if not isinstance(sensitivity_report_descriptor, Mapping):
            raise ValueError(f"{run} sensitivity report identity is missing")
        sensitivity_report, sensitivity_identity = _load_sensitivity_source(
            str(sensitivity_report_descriptor.get("path", "")),
            run=run,
        )
        if sensitivity_identity != sensitivity_descriptor:
            raise ValueError(f"{run} sensitivity source identities differ")
        training_reports[run] = training_report
        training_report_sources[run] = training_report_identity
        training_audits[run] = training_audit
        training_audit_sources[run] = training_audit_identity
        sensitivity_reports[run] = sensitivity_report
        sensitivity_sources[run] = sensitivity_identity

    rebuilt_sources = {
        "preparation": preparation_identity,
        "training_status": training_status_identity,
        "configs": config_sources,
        "training_reports": training_report_sources,
        "training_audits": training_audit_sources,
        "sensitivity": sensitivity_sources,
    }
    rebuilt = build_postevaluation(
        preparation=preparation,
        training_status=training_status,
        training_reports=training_reports,
        training_audits=training_audits,
        expected_configs=expected_configs,
        sensitivity_reports=sensitivity_reports,
        sources=rebuilt_sources,
        git=report.get("git", {}),
    )
    if rebuilt != report:
        raise ValueError("conditioning-ranking 1K postevaluation replay differs")
    decision = report.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("conditioning-ranking 1K postevaluation decision is missing")
    if require_sampling_selected and (
        decision.get("method_passes")
        != {"cofitok": True, "dense_identity": True}
        or decision.get("shared_semantic_alignment_recovery_supported") is not True
        or decision.get("cofitok_specific_advantage_claim_allowed") is not False
        or decision.get("recommended_next_action")
        != "consider_separately_authorized_matched_sampling_validation"
    ):
        raise ValueError("conditioning-ranking 1K postevaluation did not select sampling")
    return report, report_identity, training_reports


def validate_training_checkpoints(
    postevaluation: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    *,
    expected_checkpoint_step: int = EXPECTED_CHECKPOINT_STEP,
) -> dict[str, Any]:
    training_contract = postevaluation.get("training_contract")
    contract_runs = (
        training_contract.get("runs")
        if isinstance(training_contract, Mapping)
        else None
    )
    if not isinstance(contract_runs, Mapping):
        raise ValueError("postevaluation training contract is missing")
    result: dict[str, Any] = {}
    for run in RUN_NAMES:
        report = training_reports.get(run)
        contract = contract_runs.get(run)
        if not isinstance(report, Mapping) or not isinstance(contract, Mapping):
            raise ValueError(f"{run} training checkpoint contract is missing")
        latest = report.get("latest_checkpoint")
        if not isinstance(latest, Mapping):
            raise ValueError(f"{run} latest checkpoint is missing")
        checkpoint = reject_symlink_chain(
            Path(str(report.get("output_dir", "")))
            / str(latest.get("checkpoint", "")),
            name=f"{run} checkpoint",
        ).resolve()
        integrity_path = reject_symlink_chain(
            checkpoint_integrity_path(checkpoint),
            name=f"{run} checkpoint integrity manifest",
        ).resolve()
        integrity = verify_training_checkpoint(checkpoint)
        checkpoint_sha = str(integrity.get("checkpoint_sha256", ""))
        if (
            int(integrity.get("step", -1)) != expected_checkpoint_step
            or checkpoint_sha != latest.get("checkpoint_sha256")
            or checkpoint_sha != contract.get("checkpoint_sha256")
            or int(integrity.get("checkpoint_bytes", -1))
            != int(latest.get("checkpoint_bytes", -2))
            or integrity.get("git_revision")
            != postevaluation.get("training_contract", {}).get("revision")
            or integrity.get("git_branch")
            != postevaluation.get("training_contract", {}).get("branch")
            or integrity.get("git_dirty") is not False
        ):
            raise ValueError(f"{run} physical checkpoint differs from training evidence")
        result[run] = {
            "checkpoint": {
                "path": checkpoint.as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": checkpoint_sha,
            },
            "integrity_manifest": file_identity(integrity_path),
            "step": expected_checkpoint_step,
            "training_report": file_identity(
                reject_symlink_chain(
                    Path(str(report["output_dir"])) / "training_report.json",
                    name=f"{run} training report",
                ).resolve()
            ),
        }
    return result


def _sampling_paths(
    output_root: Path,
    run: str,
    *,
    sample_run_name: str = SAMPLE_RUN_NAME,
) -> tuple[Path, Path]:
    sample_root = output_root / run / sample_run_name
    return sample_root, sample_root / "sampling_report.json"


def validate_sampling_evidence(
    *,
    output_root: Path,
    checkpoint_evidence: Mapping[str, Mapping[str, Any]],
    expected_git: Mapping[str, Any],
    expected_checkpoint_step: int = EXPECTED_CHECKPOINT_STEP,
    expected_num_samples: int = EXPECTED_NUM_SAMPLES,
    sample_run_name: str = SAMPLE_RUN_NAME,
    method_prefix_budgets: Mapping[str, int] = METHOD_PREFIX_BUDGETS,
    arm_method: Mapping[str, str] = ARM_METHOD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenances: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    normalized_protocols: dict[str, Any] = {}
    for run in RUN_NAMES:
        method = arm_method[run]
        expected_budget = method_prefix_budgets[method]
        sample_root, sampling_report_path = _sampling_paths(
            output_root,
            run,
            sample_run_name=sample_run_name,
        )
        report = read_json_object(sampling_report_path, name=f"{run} sampling report")
        output_dirs = report.get("output_dirs")
        if not isinstance(output_dirs, Mapping):
            raise ValueError(f"{run} sampling output directories are missing")
        generated_dir = reject_symlink_chain(
            str(output_dirs.get(str(expected_budget), "")),
            name=f"{run} generated image directory",
        ).resolve()
        expected_generated_dir = (sample_root / f"prefix_{expected_budget}").resolve()
        if generated_dir != expected_generated_dir:
            raise ValueError(f"{run} generated image directory differs")
        images = find_images(generated_dir)
        provenance = validate_sampling_provenance(
            sampling_report_path,
            generated_dir,
            images,
        )
        checkpoint = checkpoint_evidence[run]
        if (
            provenance.get("checkpoint") != checkpoint["checkpoint"]["path"]
            or provenance.get("checkpoint_sha256")
            != checkpoint["checkpoint"]["sha256"]
            or provenance.get("checkpoint_integrity_manifest")
            != checkpoint["integrity_manifest"]["path"]
            or int(provenance.get("checkpoint_step", -1))
            != expected_checkpoint_step
            or provenance.get("weights") != "ema"
            or provenance.get("git") != expected_git
            or int(provenance.get("selected_prefix_budget", -1))
            != expected_budget
            or int(provenance.get("sampling", {}).get("num_samples", -1))
            != expected_num_samples
        ):
            raise ValueError(f"{run} sampling provenance differs")
        elapsed = _finite(
            provenance.get("sampling_progress", {}).get(
                "cumulative_elapsed_seconds"
            ),
            label=f"{run} sampling elapsed seconds",
        )
        if elapsed <= 0.0:
            raise ValueError(f"{run} sampling elapsed seconds are not positive")
        normalized = dict(provenance["sampling"])
        normalized["prefix_budgets"] = []
        normalized_protocols[run] = normalized
        provenances[run] = provenance
        sources[run] = {
            "sampling_report": provenance["report_identity"],
            "sampling_manifest": provenance["manifest_identity"],
            "sampling_progress": provenance["sampling_progress"]["identity"],
            "sample_set_sha256": provenance["sample_set_sha256"],
            "generated_dir": generated_dir.as_posix(),
        }
    first = normalized_protocols[RUN_NAMES[0]]
    if any(normalized_protocols[run] != first for run in RUN_NAMES[1:]):
        raise ValueError("four-arm sampling protocols differ outside prefix budget")
    first_environment = provenances[RUN_NAMES[0]]["runtime_environment"]
    first_environment_sha = provenances[RUN_NAMES[0]][
        "runtime_environment_sha256"
    ]
    if any(
        provenances[run]["runtime_environment"] != first_environment
        or provenances[run]["runtime_environment_sha256"] != first_environment_sha
        for run in RUN_NAMES[1:]
    ):
        raise ValueError("four-arm sampling runtime environments differ")
    checkpoint_shas = {
        provenances[run]["checkpoint_sha256"] for run in RUN_NAMES
    }
    if len(checkpoint_shas) != len(RUN_NAMES):
        raise ValueError("four-arm sampling checkpoints are not distinct")
    return provenances, sources


def validate_generation_metrics_reports(
    *,
    output_root: Path,
    sampling_provenance: Mapping[str, Mapping[str, Any]],
    expected_git: Mapping[str, Any],
    expected_num_samples: int = EXPECTED_NUM_SAMPLES,
    sample_run_name: str = SAMPLE_RUN_NAME,
    expected_real_set: Mapping[str, Any] = EXPECTED_REAL_SET,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reports: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for run in RUN_NAMES:
        metrics_path = (
            output_root
            / run
            / sample_run_name
            / "metrics"
            / "generation_metrics_report.json"
        )
        report, identity = _load_json_source(
            metrics_path,
            label=f"{run} generation metrics",
        )
        metrics = report.get("metrics")
        counts = report.get("counts")
        parameters = report.get("parameters")
        if (
            report.get("schema_version") != GENERATION_METRICS_REPORT_SCHEMA_VERSION
            or report.get("role") != GENERATION_METRICS_REPORT_ROLE
            or report.get("status") != "completed"
            or report.get("protocol") != "torch_fidelity_directory_metrics"
            or report.get("git") != expected_git
            or report.get("sample_provenance") != sampling_provenance[run]
            or report.get("real_set") != expected_real_set
            or not isinstance(counts, Mapping)
            or int(counts.get("generated_image_count", -1))
            != expected_num_samples
            or int(counts.get("real_image_count", -1)) != 50_000
            or not isinstance(parameters, Mapping)
            or int(parameters.get("min_samples", -1)) != expected_num_samples
            or not isinstance(metrics, Mapping)
            or parameters.get("precision_recall_enabled") is not False
            or "precision" in metrics
            or "recall" in metrics
        ):
            raise ValueError(f"{run} generation metrics contract differs")
        environment = report.get("runtime_environment")
        if (
            not isinstance(environment, Mapping)
            or report.get("runtime_environment_sha256")
            != runtime_environment_sha256(dict(environment))
        ):
            raise ValueError(f"{run} generation evaluator environment differs")
        fid = _finite(
            metrics.get("frechet_inception_distance"),
            label=f"{run} FID",
        )
        inception_mean = _finite(
            metrics.get("inception_score_mean"),
            label=f"{run} Inception Score",
        )
        inception_std = _finite(
            metrics.get("inception_score_std"),
            label=f"{run} Inception Score std",
        )
        if fid < 0.0 or inception_mean <= 0.0 or inception_std < 0.0:
            raise ValueError(f"{run} generation metrics are outside their domains")
        reports[run] = {
            "fid": fid,
            "inception_score_mean": inception_mean,
            "inception_score_std": inception_std,
            "implementation": report.get("implementation"),
            "runtime_environment": dict(environment),
            "runtime_environment_sha256": report["runtime_environment_sha256"],
            "real_set": report["real_set"],
            "sample_provenance": report["sample_provenance"],
        }
        sources[run] = identity
    first = reports[RUN_NAMES[0]]
    if any(
        reports[run]["implementation"] != first["implementation"]
        or reports[run]["runtime_environment"]
        != first["runtime_environment"]
        or reports[run]["runtime_environment_sha256"]
        != first["runtime_environment_sha256"]
        or reports[run]["real_set"] != first["real_set"]
        for run in RUN_NAMES[1:]
    ):
        raise ValueError("four-arm generation evaluator provenance differs")
    return reports, sources


def validate_paired_class_reports(
    *,
    output_root: Path,
    sampling_provenance: Mapping[str, Mapping[str, Any]],
    expected_git: Mapping[str, Any],
    expected_num_samples: int = EXPECTED_NUM_SAMPLES,
    expected_start_index: int = 0,
    sampling_stage: str = "conditioning_ranking_four_arm_sampling5k_v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    reports: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for method, arms in METHOD_ARMS.items():
        report_path = (
            output_root
            / "reports"
            / "paired_class_fidelity"
            / method
            / "paired_class_fidelity_report.json"
        )
        report, identity = _load_json_source(
            report_path,
            label=f"{method} paired class-fidelity report",
        )
        expected_pair = validate_sampling_pair(
            sampling_provenance[arms[0]],
            sampling_provenance[arms[1]],
            method=method,
            sampling_stage=sampling_stage,
        )
        rows = report.get("sample_rows")
        if (
            report.get("schema_version") != PAIRED_CLASS_REPORT_SCHEMA_VERSION
            or report.get("role") != PAIRED_CLASS_REPORT_ROLE
            or report.get("status") != "completed"
            or report.get("protocol")
            != "torchvision_imagenet_paired_conditioning_ranking_class_fidelity"
            or report.get("git") != expected_git
            or report.get("sampling_pair") != expected_pair
            or report.get("claim_boundary") != PAIRED_CLASS_CLAIM_BOUNDARY
            or not isinstance(rows, list)
            or len(rows) != expected_num_samples
        ):
            raise ValueError(f"{method} paired class-fidelity contract differs")
        metrics = paired_class_fidelity_summary(
            rows,
            expected_start_index=expected_start_index,
        )
        if metrics != report.get("metrics"):
            raise ValueError(f"{method} paired class-fidelity metrics differ")
        environment = report.get("runtime_environment")
        if (
            not isinstance(environment, Mapping)
            or report.get("runtime_environment_sha256")
            != runtime_environment_sha256(dict(environment))
        ):
            raise ValueError(f"{method} paired evaluator environment differs")
        reports[method] = {
            "metrics": metrics,
            "classifier": report.get("classifier"),
            "runtime_environment": dict(environment),
            "runtime_environment_sha256": report["runtime_environment_sha256"],
            "sampling_pair": expected_pair,
        }
        sources[method] = identity
    first = reports["cofitok"]
    second = reports["dense_identity"]
    if (
        first["classifier"] != second["classifier"]
        or first["runtime_environment"] != second["runtime_environment"]
        or first["runtime_environment_sha256"]
        != second["runtime_environment_sha256"]
    ):
        raise ValueError("paired class-fidelity evaluator provenance differs")
    classifier = first["classifier"]
    if not isinstance(classifier, Mapping):
        raise ValueError("paired class-fidelity classifier identity is missing")
    weights_path = reject_symlink_chain(
        str(classifier.get("weights_path", "")),
        name="paired class-fidelity classifier weights",
    ).resolve()
    weights_identity = file_identity(weights_path)
    if (
        weights_identity["bytes"] != classifier.get("weights_bytes")
        or weights_identity["sha256"] != classifier.get("weights_sha256")
    ):
        raise ValueError("paired class-fidelity classifier identity differs")
    return reports, sources


def method_sampling_decision(
    *,
    method: str,
    paired_metrics: Mapping[str, Any],
    control_generation: Mapping[str, Any],
    ranked_generation: Mapping[str, Any],
) -> dict[str, Any]:
    for name in ("control", "ranked", "ranked_minus_control", "paired_sign_test", "thresholds"):
        if not isinstance(paired_metrics.get(name), Mapping):
            raise ValueError(f"{method} paired class {name} metrics are missing")
    class_gates = paired_metrics.get("gates")
    if not isinstance(class_gates, Mapping):
        raise ValueError(f"{method} paired class gates are missing")
    expected_class_gates = {
        "mean_target_log_probability_delta",
        "paired_target_log_probability_significant",
        "target_probability_ratio",
        "top1_not_regressed",
        "top5_not_regressed",
        "predicted_class_fraction_within_tolerance",
    }
    if set(class_gates) != expected_class_gates or any(
        type(class_gates[name]) is not bool for name in expected_class_gates
    ):
        raise ValueError(f"{method} paired class gates differ")
    control_fid = _finite(control_generation.get("fid"), label=f"{method} control FID")
    ranked_fid = _finite(ranked_generation.get("fid"), label=f"{method} ranked FID")
    if control_fid < 0.0 or ranked_fid < 0.0:
        raise ValueError(f"{method} FID is outside its domain")
    fid_ratio = ranked_fid / control_fid if control_fid > 0.0 else math.inf
    if control_fid == 0.0 and ranked_fid == 0.0:
        fid_ratio = 1.0
    gates = {
        **{name: bool(class_gates[name]) for name in sorted(class_gates)},
        "fid_within_tolerance": fid_ratio <= MAX_FID_RATIO,
        "exact_provenance": True,
    }
    return {
        "method": method,
        "control": {
            "fid": control_fid,
            **dict(paired_metrics["control"]),
        },
        "ranked": {
            "fid": ranked_fid,
            **dict(paired_metrics["ranked"]),
        },
        "ranked_minus_control": {
            **dict(paired_metrics["ranked_minus_control"]),
            "fid": ranked_fid - control_fid,
            "fid_ratio": fid_ratio,
        },
        "paired_sign_test": dict(paired_metrics["paired_sign_test"]),
        "thresholds": {
            **dict(paired_metrics["thresholds"]),
            "maximum_ranked_to_control_fid_ratio": MAX_FID_RATIO,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def build_sampling_validation(
    *,
    postevaluation: Mapping[str, Any],
    postevaluation_identity: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any],
    sampling_provenance: Mapping[str, Mapping[str, Any]],
    sampling_sources: Mapping[str, Any],
    generation_reports: Mapping[str, Mapping[str, Any]],
    generation_sources: Mapping[str, Any],
    paired_reports: Mapping[str, Mapping[str, Any]],
    paired_sources: Mapping[str, Any],
    git: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    clean_git = _clean_git(git, label="sampling validation builder")
    methods: dict[str, Any] = {}
    for method, arms in METHOD_ARMS.items():
        methods[method] = method_sampling_decision(
            method=method,
            paired_metrics=paired_reports[method]["metrics"],
            control_generation=generation_reports[arms[0]],
            ranked_generation=generation_reports[arms[1]],
        )
    method_passes = {method: result["pass"] for method, result in methods.items()}
    shared = all(method_passes.values())
    if shared:
        next_action = (
            "prepare_separately_bound_matched_5k_training_recipe_confirmation"
        )
    elif any(method_passes.values()):
        next_action = "reject_shared_repair_due_method_asymmetry"
    else:
        next_action = "revise_training_time_semantic_alignment_objective"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "stage": STAGE,
        "output_root": output_root,
        "git": clean_git,
        "sources": {
            "postevaluation": _identity(
                postevaluation_identity,
                label="conditioning-ranking 1K postevaluation",
            ),
            "training_checkpoints": dict(checkpoint_evidence),
            "sampling": dict(sampling_sources),
            "generation_metrics": dict(generation_sources),
            "paired_class_fidelity": dict(paired_sources),
        },
        "source_decision": dict(postevaluation["decision"]),
        "sampling_contract": {
            "sample_count_per_arm": EXPECTED_NUM_SAMPLES,
            "sample_run_name": SAMPLE_RUN_NAME,
            "methods": {
                method: {
                    "control": sampling_provenance[arms[0]],
                    "ranked": sampling_provenance[arms[1]],
                }
                for method, arms in METHOD_ARMS.items()
            },
        },
        "methods": methods,
        "difference_in_differences": {
            "mean_target_log_probability": (
                methods["cofitok"]["ranked_minus_control"][
                    "mean_target_log_probability"
                ]
                - methods["dense_identity"]["ranked_minus_control"][
                    "mean_target_log_probability"
                ]
            ),
            "fid_ratio": (
                methods["cofitok"]["ranked_minus_control"]["fid_ratio"]
                - methods["dense_identity"]["ranked_minus_control"]["fid_ratio"]
            ),
        },
        "difference_in_differences_is_descriptive_only": True,
        "decision": {
            "method_passes": method_passes,
            "shared_generated_class_alignment_recovery_supported": shared,
            "cofitok_specific_advantage_claim_allowed": False,
            "recommended_next_action": next_action,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def replay_sampling_validation(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, report_identity = _load_json_source(
        path,
        label="conditioning-ranking 5K sampling validation",
    )
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != STAGE
        or report.get("output_root") != EXPECTED_OUTPUT_ROOT
    ):
        raise ValueError("conditioning-ranking 5K sampling validation differs")
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("conditioning-ranking sampling-validation sources are missing")
    postevaluation_descriptor = sources.get("postevaluation")
    if not isinstance(postevaluation_descriptor, Mapping):
        raise ValueError("conditioning-ranking postevaluation source is missing")
    postevaluation, postevaluation_identity, training_reports = replay_postevaluation(
        str(postevaluation_descriptor.get("path", ""))
    )
    if postevaluation_identity != postevaluation_descriptor:
        raise ValueError("conditioning-ranking postevaluation identity differs")
    checkpoint_evidence = validate_training_checkpoints(
        postevaluation,
        training_reports,
    )
    if sources.get("training_checkpoints") != checkpoint_evidence:
        raise ValueError("conditioning-ranking checkpoint evidence differs")
    output_root = Path(EXPECTED_OUTPUT_ROOT).resolve()
    expected_git = _clean_git(
        report.get("git", {}),
        label="sampling validation replay Git",
    )
    sampling_provenance, sampling_sources = validate_sampling_evidence(
        output_root=output_root,
        checkpoint_evidence=checkpoint_evidence,
        expected_git=expected_git,
    )
    if sources.get("sampling") != sampling_sources:
        raise ValueError("conditioning-ranking sampling source identities differ")
    generation_reports, generation_sources = validate_generation_metrics_reports(
        output_root=output_root,
        sampling_provenance=sampling_provenance,
        expected_git=expected_git,
    )
    if sources.get("generation_metrics") != generation_sources:
        raise ValueError("conditioning-ranking generation-metrics sources differ")
    paired_reports, paired_sources = validate_paired_class_reports(
        output_root=output_root,
        sampling_provenance=sampling_provenance,
        expected_git=expected_git,
    )
    if sources.get("paired_class_fidelity") != paired_sources:
        raise ValueError("conditioning-ranking paired-class sources differ")
    expected = build_sampling_validation(
        postevaluation=postevaluation,
        postevaluation_identity=postevaluation_identity,
        checkpoint_evidence=checkpoint_evidence,
        sampling_provenance=sampling_provenance,
        sampling_sources=sampling_sources,
        generation_reports=generation_reports,
        generation_sources=generation_sources,
        paired_reports=paired_reports,
        paired_sources=paired_sources,
        git=expected_git,
        output_root=EXPECTED_OUTPUT_ROOT,
    )
    if expected != report:
        raise ValueError("conditioning-ranking sampling validation replay differs")
    return report, report_identity


def main() -> None:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="conditioning-ranking 5K sampling output root",
    ).resolve()
    if output_root.as_posix() != EXPECTED_OUTPUT_ROOT:
        raise ValueError("conditioning-ranking 5K output root differs")
    output = reject_symlink_chain(
        args.output,
        name="conditioning-ranking 5K sampling validation output",
    ).resolve()
    expected_output = output_root / "reports" / "sampling_validation.json"
    if output != expected_output:
        raise ValueError("conditioning-ranking sampling validation output differs")
    current_git = git_provenance(PROJECT_ROOT)
    _clean_git(current_git, label="sampling validation builder")
    with exclusive_output_lock(output, role=REPORT_ROLE):
        postevaluation, postevaluation_identity, training_reports = (
            replay_postevaluation(args.postevaluation)
        )
        checkpoint_evidence = validate_training_checkpoints(
            postevaluation,
            training_reports,
        )
        sampling_provenance, sampling_sources = validate_sampling_evidence(
            output_root=output_root,
            checkpoint_evidence=checkpoint_evidence,
            expected_git=current_git,
        )
        generation_reports, generation_sources = (
            validate_generation_metrics_reports(
                output_root=output_root,
                sampling_provenance=sampling_provenance,
                expected_git=current_git,
            )
        )
        paired_reports, paired_sources = validate_paired_class_reports(
            output_root=output_root,
            sampling_provenance=sampling_provenance,
            expected_git=current_git,
        )
        expected = build_sampling_validation(
            postevaluation=postevaluation,
            postevaluation_identity=postevaluation_identity,
            checkpoint_evidence=checkpoint_evidence,
            sampling_provenance=sampling_provenance,
            sampling_sources=sampling_sources,
            generation_reports=generation_reports,
            generation_sources=generation_sources,
            paired_reports=paired_reports,
            paired_sources=paired_sources,
            git=current_git,
            output_root=output_root.as_posix(),
        )
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "sampling validation exists; pass --resume to revalidate it"
                )
            existing = read_json_object(
                output,
                name="conditioning-ranking 5K sampling validation",
            )
            if existing != expected:
                raise ValueError("completed sampling validation differs from sources")
        else:
            write_json_report(output, expected)
        print(output.as_posix())


if __name__ == "__main__":
    main()
