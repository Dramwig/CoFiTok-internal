from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation.quality_bridge import RESULT_AUTHORIZATION_BOUNDARY
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CATEGORIES_SHA256,
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
    CLASS_FIDELITY_PREPROCESSING,
    CLASS_FIDELITY_QUALIFICATION_ROLE,
    CLASS_FIDELITY_QUALIFICATION_SCHEMA_VERSION,
    CLASS_FIDELITY_REPORT_ROLE,
    CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
    validate_class_fidelity_qualification,
)
from scripts import build_generation_quality_bridge_comparison as comparison

REVISION = "a" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
GIT = {"revision": REVISION, "branch": BRANCH, "tracked_dirty": False}
TRAINING_GIT = {"revision": REVISION, "branch": BRANCH, "dirty": False}
DATASET_SHA256 = "b" * 64
TRAINING_RUNTIME_SHA256 = "c" * 64
CLASS_RUNTIME_ENVIRONMENT = {"schema_version": 1, "device": {"type": "cpu"}}
EVALUATOR_RUNTIME_SHA256 = runtime_environment_sha256(CLASS_RUNTIME_ENVIRONMENT)


def _write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    return comparison.file_identity(path)


def _sampling(budget: int) -> dict:
    return {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_samples": 10_000,
        "start_index": 0,
        "batch_size": 32,
        "sample_steps": 100,
        "num_train_timesteps": 1_000,
        "actual_timesteps": select_sampling_timesteps(1_000, 100),
        "image_shape": [3, 256, 256],
        "class_schedule": "balanced_modulo",
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "prefix_budgets": [budget],
        "random_stream": {
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }


def _training_report(*, cofitok: bool) -> dict:
    return {
        "training_complete": True,
        "completed_steps": 100_000,
        "target_steps": 100_000,
        "git": copy.deepcopy(TRAINING_GIT),
        "parameter_count": 62_834_083 if cofitok else 62_824_707,
        "runtime_environment_sha256": TRAINING_RUNTIME_SHA256,
        "dataset_provenance": {"identity_sha256": DATASET_SHA256},
        "config": {
            "data": {"dataset": "imagenet_256", "batch_size": 64},
            "model": {
                "image_size": 256,
                "synthesis_mode": "fixed_basis" if cofitok else "dense_identity",
                "token_count": 8 if cofitok else 1,
                "predictor_use_feedback": cofitok,
            },
            "optimization": {"gradient_accumulation_steps": 1},
        },
        "final_metrics": {"step": 100_000, "samples_seen": 6_400_000},
    }


def _class_metrics(*, top1: float, top5: float) -> dict:
    sample_count = 10_000
    entropy = math.log(1000) * 0.8
    return {
        "sample_count": sample_count,
        "num_classes": 1000,
        "top1_correct": int(top1 * sample_count),
        "top5_correct": int(top5 * sample_count),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "mean_target_probability": 0.12,
        "target_negative_log_likelihood": 3.2,
        "requested_class_count": 1000,
        "requested_count_min": 10,
        "requested_count_max": 10,
        "predicted_class_count": 800,
        "predicted_class_fraction": 0.8,
        "predicted_class_entropy": entropy,
        "normalized_predicted_class_entropy": 0.8,
    }


def _classifier() -> dict:
    return {
        "name": CLASS_FIDELITY_CLASSIFIER_NAME,
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_bytes": CLASS_FIDELITY_CLASSIFIER_BYTES,
        "weights_sha256": CLASS_FIDELITY_CLASSIFIER_SHA256,
        "num_classes": 1000,
        "categories_sha256": CLASS_FIDELITY_CATEGORIES_SHA256,
        "preprocessing": CLASS_FIDELITY_PREPROCESSING,
    }


def _class_fidelity_report(
    *,
    method: dict,
    metrics: dict,
) -> dict:
    runtime_environment = copy.deepcopy(CLASS_RUNTIME_ENVIRONMENT)
    return {
        "schema_version": CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
        "role": CLASS_FIDELITY_REPORT_ROLE,
        "status": "completed",
        "protocol": "torchvision_imagenet_class_fidelity",
        "git": copy.deepcopy(GIT),
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256(runtime_environment),
        "paths": {},
        "classifier": _classifier(),
        "sample_provenance": {
            "git": copy.deepcopy(GIT),
            "weights": "ema",
            "checkpoint_sha256": method["checkpoint_sha256"],
            "sample_set_sha256": method["sample_set_sha256"],
            "selected_prefix_budget": method["selected_prefix_budget"],
            "sampling": copy.deepcopy(method["sampling"]),
        },
        "parameters": {
            "num_classes": 1000,
            "sample_count": 10_000,
            "target_from_filename": "int(zero_based_png_stem) mod 1000",
        },
        "metrics": copy.deepcopy(metrics),
        "runtime": {"elapsed_seconds": 1.0},
    }


def _class_fidelity(
    *,
    cofitok_method: dict,
    dense_method: dict,
    cofitok_source: dict,
    dense_source: dict,
) -> dict:
    cofitok = _class_metrics(top1=0.12, top5=0.30)
    dense = _class_metrics(top1=0.11, top5=0.29)
    thresholds = {
        "min_top1": 0.01,
        "min_top5": 0.05,
        "min_predicted_class_fraction": 0.25,
        "min_normalized_predicted_entropy": 0.50,
        "max_top1_regression": 0.05,
        "max_top5_regression": 0.05,
    }
    checks = []
    for method, metrics in (("cofitok", cofitok), ("dense_identity", dense)):
        for metric, threshold in (
            ("top1_accuracy", "min_top1"),
            ("top5_accuracy", "min_top5"),
            ("predicted_class_fraction", "min_predicted_class_fraction"),
            (
                "normalized_predicted_class_entropy",
                "min_normalized_predicted_entropy",
            ),
        ):
            checks.append(
                {
                    "name": f"{method}_{metric}",
                    "status": "pass",
                    "observed": metrics[metric],
                    "comparison": ">=",
                    "threshold": thresholds[threshold],
                }
            )
    for metric, threshold in (
        ("top1_accuracy", "max_top1_regression"),
        ("top5_accuracy", "max_top5_regression"),
    ):
        checks.append(
            {
                "name": f"cofitok_{metric}_regression_vs_dense",
                "status": "pass",
                "observed": dense[metric] - cofitok[metric],
                "comparison": "<=",
                "threshold": thresholds[threshold],
            }
        )
    sampling = _sampling(8)
    sampling.pop("prefix_budgets")
    return {
        "schema_version": CLASS_FIDELITY_QUALIFICATION_SCHEMA_VERSION,
        "role": CLASS_FIDELITY_QUALIFICATION_ROLE,
        "status": "pass",
        "stage": "scaling",
        "git": copy.deepcopy(GIT),
        "classifier": _classifier(),
        "sampling_contract": {
            "sampling": sampling,
            "cofitok_prefix_budget": 8,
            "dense_prefix_budget": 1,
            "weights": "ema",
            "sampling_git": copy.deepcopy(GIT),
            "evaluator_git": copy.deepcopy(GIT),
            "evaluator_runtime_environment_sha256": EVALUATOR_RUNTIME_SHA256,
            "sample_count_per_method": 10_000,
            "cofitok_checkpoint_sha256": cofitok_method["checkpoint_sha256"],
            "dense_checkpoint_sha256": dense_method["checkpoint_sha256"],
            "cofitok_sample_set_sha256": cofitok_method["sample_set_sha256"],
            "dense_sample_set_sha256": dense_method["sample_set_sha256"],
        },
        "thresholds": thresholds,
        "checks": checks,
        "metrics": {
            "cofitok": cofitok,
            "dense_identity": dense,
            "cofitok_minus_dense": {
                name: cofitok[name] - dense[name]
                for name in (
                    "top1_accuracy",
                    "top5_accuracy",
                    "mean_target_probability",
                    "predicted_class_fraction",
                    "normalized_predicted_class_entropy",
                )
            },
        },
        "sources": {
            "cofitok": cofitok_source,
            "dense_identity": dense_source,
        },
        "claim_boundary": {
            "class_conditional_quality_evaluated": True,
            "unconditional_distribution_quality_evaluated": False,
            "standalone_generation_quality_claim_allowed": False,
            "full_training_launch_allowed": False,
            "release_authorization_allowed": False,
        },
    }


def _terminal_method(
    tmp_path: Path,
    *,
    name: str,
    budget: int,
    sample_sha: str,
    fid: float,
) -> dict:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint_step_00100000.pt"
    checkpoint_path.write_bytes(f"{name}-checkpoint".encode())
    checkpoint_identity = comparison.file_identity(checkpoint_path)
    integrity_path = checkpoint_path.with_name(f"{checkpoint_path.name}.integrity.json")
    integrity_identity = _write_json(
        integrity_path,
        {
            "schema_version": 1,
            "checkpoint": checkpoint_path.name,
            "checkpoint_bytes": checkpoint_identity["bytes"],
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "checkpoint_format_version": 1,
            "step": 100_000,
            "git_revision": REVISION,
            "git_branch": BRANCH,
            "git_dirty": False,
            "dataset_identity_sha256": DATASET_SHA256,
            "runtime_environment_sha256": TRAINING_RUNTIME_SHA256,
        },
    )
    manifest_identity = _write_json(root / "sampling_manifest.json", {"name": name})
    progress = {
        "schema_version": 1,
        "status": "completed",
        "sampling_manifest_sha256": manifest_identity["sha256"],
        "total_samples": 10_000,
        "start_index": 0,
        "prefix_budgets": [budget],
        "completed_samples": 10_000,
        "invocation": 1,
        "cumulative_elapsed_seconds": 500.0,
        "sample_sets": {str(budget): {"count": 10_000, "sha256": sample_sha}},
        "error_type": None,
        "error": None,
    }
    progress_identity = _write_json(root / "sampling_progress.json", progress)
    report_identity = _write_json(root / "sampling_report.json", {"name": name})
    return {
        "checkpoint": checkpoint_identity["path"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_integrity_manifest": integrity_identity["path"],
        "checkpoint_step": 100_000,
        "sample_set_sha256": sample_sha,
        "sample_count": 10_000,
        "selected_prefix_budget": budget,
        "weights": "ema",
        "sampling": _sampling(budget),
        "sampling_runtime_environment_sha256": "5" * 64,
        "sampling_report": report_identity,
        "sampling_manifest": manifest_identity,
        "sampling_progress": progress_identity,
        "fid": fid,
        "inception_score": 20.0,
        "precision": 0.6,
        "recall": 0.4,
        "real_set": {
            "root": "/real/imagenet_256_val",
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": "6" * 64,
            "image_count": 50_000,
        },
        "metrics_evaluator_git": copy.deepcopy(GIT),
        "metrics_runtime_environment_sha256": EVALUATOR_RUNTIME_SHA256,
        "_physical_evidence": {
            "checkpoint": checkpoint_identity,
            "checkpoint_integrity_manifest": integrity_identity,
            "checkpoint_step": 100_000,
            "sampling_report": report_identity,
            "sampling_manifest": manifest_identity,
            "sampling_progress": progress_identity,
            "sample_set_sha256": sample_sha,
            "sample_count": 10_000,
            "real_set": {
                "root": "/real/imagenet_256_val",
                "digest_schema": "cofitok_image_tree_sha256_v1",
                "sha256": "6" * 64,
                "image_count": 50_000,
            },
        },
    }


def _cost(*, elapsed: float, peak_vram: int, adjustment_seconds: float) -> dict:
    applied = adjustment_seconds > 0.0
    orphaned_steps = 3650 if applied else 0
    event_count = 2 if applied else 0
    return {
        "valid": True,
        "target_steps": 100_000,
        "micro_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 64,
        "expected_samples_seen": 6_400_000,
        "samples_seen": 6_400_000,
        "reported_elapsed_seconds": elapsed - adjustment_seconds,
        "resume_compute_adjustment": {
            "valid": True,
            "required": applied,
            "provided": applied,
            "applied": applied,
            "seconds": adjustment_seconds,
            "hours": adjustment_seconds / 3600.0,
            "event_count": event_count,
            "orphaned_optimizer_steps_lower_bound": orphaned_steps,
            "orphaned_images_lower_bound": orphaned_steps * 64,
            "continuity_end_step": 100_000,
            "discovered_orphan_archive_count": event_count,
            "covered_orphan_archive_count": event_count,
            "required_reasons": ["orphaned_metrics_archives"] if applied else [],
            "issues": [],
        },
        "elapsed_seconds": elapsed,
        "elapsed_seconds_role": (
            "physical_lower_bound_including_orphaned_recovery_compute"
            if applied
            else "reported_training_elapsed_seconds"
        ),
        "images_per_second": 6_400_000 / elapsed,
        "peak_vram_bytes": peak_vram,
    }


def _official() -> dict:
    rows = []
    for alias, method, fid in (
        ("d_ar", "D-AR", 2.628105),
        ("mar", "MAR", 2.338494),
        ("retok", "ReTok", 2.218869),
    ):
        rows.append(
            {
                "alias": alias,
                "method": method,
                "dataset": "imagenet_256",
                "resolution": 256,
                "status": "completed_eval_only_50k",
                "sample_count": 50_000,
                "fid": fid,
                "inception_score": 200.0,
                "precision": 0.8,
                "recall": 0.6,
                "protocol": "official pretrained eval-only",
                "paper_table_role": "secondary related-method only",
                "metrics_txt": f"/official/{alias}.txt",
                "source_kind": "official_pretrained",
            }
        )
    return {"schema_version": 1, "rows": rows}


def _quality_screen(status: str) -> dict:
    failed = [] if status == "pass" else ["matched_fid_tolerance"]
    return {
        "status": status,
        "non_authorizing": True,
        "checks": [{"name": "matched_fid_tolerance", "passed": not failed}],
        "failed_checks": failed,
    }


def _graph(
    tmp_path: Path,
    *,
    guard_status: str = "hold",
    paired_kid_evaluated: bool = True,
) -> dict:
    cofitok_training_path = tmp_path / "cofitok_training.json"
    dense_training_path = tmp_path / "dense_training.json"
    cofitok_training = _training_report(cofitok=True)
    dense_training = _training_report(cofitok=False)
    cofitok_training_identity = _write_json(cofitok_training_path, cofitok_training)
    dense_training_identity = _write_json(dense_training_path, dense_training)

    cofitok_sample_sha = "7" * 64
    dense_sample_sha = "8" * 64
    cofitok_method = _terminal_method(
        tmp_path,
        name="cofitok",
        budget=8,
        sample_sha=cofitok_sample_sha,
        fid=90.0,
    )
    dense_method = _terminal_method(
        tmp_path,
        name="dense",
        budget=1,
        sample_sha=dense_sample_sha,
        fid=100.0,
    )
    cofitok_physical = cofitok_method.pop("_physical_evidence")
    dense_physical = dense_method.pop("_physical_evidence")
    cofitok_class_report = _class_fidelity_report(
        method=cofitok_method,
        metrics=_class_metrics(top1=0.12, top5=0.30),
    )
    dense_class_report = _class_fidelity_report(
        method=dense_method,
        metrics=_class_metrics(top1=0.11, top5=0.29),
    )
    cofitok_class_identity = _write_json(
        tmp_path / "cofitok_class_fidelity.json",
        cofitok_class_report,
    )
    dense_class_identity = _write_json(
        tmp_path / "dense_class_fidelity.json",
        dense_class_report,
    )
    class_fidelity = _class_fidelity(
        cofitok_method=cofitok_method,
        dense_method=dense_method,
        cofitok_source=cofitok_class_identity,
        dense_source=dense_class_identity,
    )
    class_fidelity_identity = _write_json(
        tmp_path / "class_fidelity_qualification.json",
        class_fidelity,
    )
    normalized_class_fidelity = validate_class_fidelity_qualification(
        class_fidelity,
        expected_stage="scaling",
        expected_revision=REVISION,
        expected_branch=BRANCH,
        require_pass=False,
    )

    def latest(method: dict, physical: dict) -> dict:
        return {
            "step": 100_000,
            "checkpoint": Path(method["checkpoint"]).name,
            "checkpoint_bytes": physical["checkpoint"]["bytes"],
            "checkpoint_format_version": 1,
            "checkpoint_sha256": method["checkpoint_sha256"],
            "integrity_manifest": Path(method["checkpoint_integrity_manifest"]).name,
            "dataset_identity_sha256": DATASET_SHA256,
            "runtime_environment_sha256": TRAINING_RUNTIME_SHA256,
            "git_revision": REVISION,
            "git_branch": BRANCH,
            "git_dirty": False,
        }

    cofitok_latest_identity = _write_json(
        tmp_path / "cofitok_latest.json",
        latest(cofitok_method, cofitok_physical),
    )
    dense_latest_identity = _write_json(
        tmp_path / "dense_latest.json",
        latest(dense_method, dense_physical),
    )
    screen = _quality_screen("pass" if guard_status == "pass" else "hold")
    quality = {
        "schema_version": 1,
        "role": "stability_full_data_quality_bridge_result",
        "status": "completed",
        "stage": "stability_quality_bridge",
        "git": copy.deepcopy(GIT),
        "source_reports": {
            "cofitok_training": cofitok_training_identity,
            "dense_training": dense_training_identity,
            "class_fidelity_qualification": class_fidelity_identity,
            "cofitok_class_fidelity": cofitok_class_identity,
            "dense_class_fidelity": dense_class_identity,
        },
        "terminal": {
            "physical_evidence": {
                "cofitok": cofitok_physical,
                "dense_identity": dense_physical,
            },
            "methods": {
                "cofitok": cofitok_method,
                "dense_identity": dense_method,
            },
            "class_fidelity": normalized_class_fidelity,
        },
        "quality_screen": screen,
        "authorization_boundary": copy.deepcopy(RESULT_AUTHORIZATION_BOUNDARY),
    }
    quality_identity = _write_json(tmp_path / "quality_bridge_result.json", quality)

    fairness = {
        "schema_version": 1,
        "role": "quality_bridge_runtime_compute_fairness_audit",
        "status": "pass",
        "contract": {
            "status": "verified",
            "training_git": copy.deepcopy(GIT),
            "dataset": "imagenet_256",
        },
        "terminal_sources": {
            "cofitok_training": cofitok_training_identity,
            "dense_training": dense_training_identity,
            "cofitok_latest": cofitok_latest_identity,
            "dense_latest": dense_latest_identity,
        },
        "methods": {
            "cofitok": {
                "status": "verified",
                "training_cost": _cost(
                    elapsed=20_000.0,
                    peak_vram=56 * 1024**3,
                    adjustment_seconds=100.0,
                ),
            },
            "dense_identity": {
                "status": "verified",
                "training_cost": _cost(
                    elapsed=18_000.0,
                    peak_vram=55 * 1024**3,
                    adjustment_seconds=0.0,
                ),
            },
        },
    }
    fairness_identity = _write_json(tmp_path / "runtime_fairness.json", fairness)
    pair = {
        "schema_version": 2,
        "monitor": comparison.EXPECTED_MONITOR,
        "status": "pass",
        "stage": "complete",
        "git": copy.deepcopy(GIT),
        "runs": {
            "cofitok": {"complete": True, "last_step": 100_000},
            "dense_identity": {"complete": True, "last_step": 100_000},
        },
    }
    pair_identity = _write_json(tmp_path / "pair_monitor.json", pair)
    runtime = {
        "schema_version": 1,
        "role": "generation_runtime_compute_claim_guard",
        "status": "pass",
        "decision": "runtime_cost_claims_observational_only",
        "sources": {
            "runtime_compute_fairness": fairness_identity,
            "terminal_pair_monitor": pair_identity,
        },
        "matched_training_contract": {
            "status": "verified",
            "training_git": copy.deepcopy(GIT),
            "dataset": "imagenet_256",
            "target_steps_per_method": 100_000,
        },
        "claim_policy": {
            "training_wall_clock_direct_comparison_allowed": False,
            "training_throughput_direct_comparison_allowed": False,
            "cost_efficiency_ranking_allowed": False,
            "equal_wall_clock_budget_claim_allowed": False,
            "equal_gpu_hours_budget_claim_allowed": False,
            "equal_training_flops_budget_claim_allowed": False,
            "peak_vram_advantage_claim_allowed": False,
            "quality_or_generation_advantage_claim_allowed": False,
        },
        "claim_boundary": copy.deepcopy(comparison.RUNTIME_CLAIM_BOUNDARY),
    }
    runtime_identity = _write_json(tmp_path / "runtime_guard.json", runtime)

    advantage = guard_status == "pass"
    guard = {
        "schema_version": 1,
        "role": "generation_terminal_system_claim_guard",
        "status": guard_status,
        "decision": (
            "matched_quality_advantage_qualified_with_terminal_system_evidence"
            if advantage
            else "terminal_system_evidence_complete_without_qualified_matched_advantage"
        ),
        "sources": {
            "quality_bridge_result": quality_identity,
            "statistical_claim_language_guard": {
                "path": "/guards/statistical.json",
                "bytes": 1,
                "sha256": "9" * 64,
            },
            "requested_class_visual_audit_waiter_status": {
                "path": "/guards/visual.json",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "runtime_compute_claim_guard": runtime_identity,
        },
        "evidence": {
            "quality_screen": {
                "status": screen["status"],
                "absolute_quality_passed": advantage,
                "failed_checks": screen["failed_checks"],
                "check_count": len(screen["checks"]),
            },
            "matched_statistical_advantage": {
                "replication_scope": {
                    "bound_stream_id": (
                        "quality_bridge_terminal_100k_00000000_00010000"
                    ),
                    "start_index": 0,
                    "end_index_exclusive": 10_000,
                    "sample_count": 10_000,
                    "bound_terminal_stream_count": 1,
                    "independent_replication_count": 0,
                    "independent_replication_supported": False,
                    "interpretation": (
                        comparison.REPLICATION_INTERPRETATION
                        if paired_kid_evaluated
                        else comparison.UNPAIRED_REPLICATION_INTERPRETATION
                    ),
                }
            },
            "class_fidelity_classifier_integrity": {"status": "verified"},
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "class_fidelity_classifier_physical_integrity_verified": True,
            "matched_distribution_quality_claim_allowed": advantage,
            "lower_fid_point_estimate_statement_allowed": advantage,
            "paired_kid_statistical_support_statement_allowed": advantage,
            "paired_kid_statistical_evidence_available": paired_kid_evaluated,
            "absolute_usability_claim_allowed": False,
            "fid_statistical_significance_claim_allowed": False,
            "fid_confidence_interval_claim_allowed": False,
            "cross_tier_numeric_ranking_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "independent_replication_claim_allowed": False,
            "multiple_independent_terminal_streams_claim_allowed": False,
            "replication_language_requires_distinct_bound_streams": True,
            "larger_training_launch_allowed": False,
            "inference_export_authorization_allowed": False,
            "release_authorization_allowed": False,
        },
        "claim_boundary": copy.deepcopy(comparison.TERMINAL_CLAIM_BOUNDARY),
    }
    guard_identity = _write_json(tmp_path / "terminal_guard.json", guard)
    official = _official()
    official_identity = _write_json(tmp_path / "official.json", official)
    return {
        "guard": guard,
        "guard_identity": guard_identity,
        "official": official,
        "official_identity": official_identity,
        "quality": quality,
        "runtime": runtime,
        "fairness": fairness,
        "pair": pair,
    }


def _build(graph: dict) -> dict:
    return comparison.build_report(
        terminal_guard_identity=graph["guard_identity"],
        terminal_guard=graph["guard"],
        official_identity=graph["official_identity"],
        official_related=graph["official"],
    )


def _rebind(graph: dict) -> None:
    quality_path = Path(graph["guard"]["sources"]["quality_bridge_result"]["path"])
    graph["guard"]["sources"]["quality_bridge_result"] = _write_json(
        quality_path,
        graph["quality"],
    )
    fairness_path = Path(
        graph["runtime"]["sources"]["runtime_compute_fairness"]["path"]
    )
    graph["runtime"]["sources"]["runtime_compute_fairness"] = _write_json(
        fairness_path,
        graph["fairness"],
    )
    pair_path = Path(graph["runtime"]["sources"]["terminal_pair_monitor"]["path"])
    graph["runtime"]["sources"]["terminal_pair_monitor"] = _write_json(
        pair_path,
        graph["pair"],
    )
    runtime_path = Path(
        graph["guard"]["sources"]["runtime_compute_claim_guard"]["path"]
    )
    graph["guard"]["sources"]["runtime_compute_claim_guard"] = _write_json(
        runtime_path,
        graph["runtime"],
    )
    graph["guard_identity"] = _write_json(
        Path(graph["guard_identity"]["path"]),
        graph["guard"],
    )


@pytest.mark.parametrize("guard_status", ["hold", "pass"])
def test_comparison_preserves_terminal_status_and_two_tier_policy(
    tmp_path: Path,
    guard_status: str,
) -> None:
    report = _build(_graph(tmp_path, guard_status=guard_status))
    assert report["status"] == guard_status
    assert report["scope"]["training_steps_per_method"] == 100_000
    assert report["scope"]["training_images_per_method"] == 6_400_000
    assert report["scope"]["terminal_samples_per_method"] == 10_000
    assert {row["comparison_tier"] for row in report["matched_training_rows"]} == {
        "matched_training_direct"
    }
    assert {row["method"] for row in report["official_context_rows"]} == {
        "D-AR",
        "MAR",
        "ReTok",
    }
    assert all(
        row["comparison_tier"] == "official_pretrained_contextual"
        and row["directly_comparable_to_cofitok"] is False
        for row in report["official_context_rows"]
    )
    assert report["comparison_policy"]["cross_tier_numeric_ranking_allowed"] is False
    assert report["comparison_policy"]["independent_replication_claim_allowed"] is False
    assert (
        report["comparison_policy"][
            "multiple_independent_terminal_streams_claim_allowed"
        ]
        is False
    )
    assert report["replication_scope"]["bound_terminal_stream_count"] == 1
    assert report["replication_scope"]["independent_replication_count"] == 0
    assert report["replication_scope"]["independent_replication_supported"] is False
    assert any("zero independent replications" in item for item in report["limitations"])
    assert report["authorization_boundary"] == comparison.AUTHORIZATION_BOUNDARY
    assert not any(
        value is True
        for name, value in report["authorization_boundary"].items()
        if name.endswith("_allowed")
    )
    assert (
        report["matched_training_rows"][0]["training_wall_clock_directly_comparable"]
        is False
    )
    assert comparison.verify_source_reports(report)["status"] == "verified"


def test_comparison_accepts_normalized_class_fidelity_embedding(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    qualification_path = Path(
        graph["quality"]["source_reports"]["class_fidelity_qualification"]["path"]
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    embedded = graph["quality"]["terminal"]["class_fidelity"]

    assert "checks" in qualification
    assert "checks" not in embedded
    assert "cofitok_minus_dense" in qualification["metrics"]
    assert "cofitok_minus_dense" not in embedded["metrics"]
    assert embedded["valid"] is True
    assert _build(graph)["status"] == "hold"


def test_comparison_rejects_normalized_class_fidelity_semantic_drift(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    qualification_path = Path(
        graph["quality"]["source_reports"]["class_fidelity_qualification"]["path"]
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["thresholds"]["min_top1"] = 0.02
    for check in qualification["checks"]:
        if check["name"] in {
            "cofitok_top1_accuracy",
            "dense_identity_top1_accuracy",
        }:
            check["threshold"] = 0.02
    graph["quality"]["source_reports"]["class_fidelity_qualification"] = (
        _write_json(qualification_path, qualification)
    )
    _rebind(graph)

    with pytest.raises(ValueError, match="class-fidelity qualification differs"):
        _build(graph)


def test_comparison_rejects_malformed_raw_class_fidelity_qualification(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    qualification_path = Path(
        graph["quality"]["source_reports"]["class_fidelity_qualification"]["path"]
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification.pop("checks")
    graph["quality"]["source_reports"]["class_fidelity_qualification"] = (
        _write_json(qualification_path, qualification)
    )
    _rebind(graph)

    with pytest.raises(ValueError, match="qualification checks are incomplete"):
        _build(graph)


def test_comparison_preserves_unexecuted_paired_reanalysis_boundary(
    tmp_path: Path,
) -> None:
    report = _build(_graph(tmp_path, paired_kid_evaluated=False))

    assert report["status"] == "hold"
    assert report["replication_scope"]["interpretation"] == (
        comparison.UNPAIRED_REPLICATION_INTERPRETATION
    )
    assert any(
        "paired block-KID was not evaluated" in item
        for item in report["limitations"]
    )
    assert all(
        "FID and paired block-KID reuse" not in item
        for item in report["limitations"]
    )
    rendered = comparison.render_markdown(report)
    assert "paired block-KID was not evaluated" in rendered
    assert "FID and paired block-KID re-analyze" not in rendered


def test_comparison_cannot_pass_without_paired_statistical_evidence(
    tmp_path: Path,
) -> None:
    graph = _graph(
        tmp_path,
        guard_status="pass",
        paired_kid_evaluated=False,
    )

    with pytest.raises(ValueError, match="system guard status and policy differ"):
        _build(graph)


@pytest.mark.parametrize("mutation", ["missing_policy", "unverified_evidence"])
def test_comparison_requires_terminal_classifier_physical_integrity(
    tmp_path: Path,
    mutation: str,
) -> None:
    graph = _graph(tmp_path)
    if mutation == "missing_policy":
        graph["guard"]["claim_policy"].pop(
            "class_fidelity_classifier_physical_integrity_verified"
        )
    else:
        graph["guard"]["evidence"]["class_fidelity_classifier_integrity"][
            "status"
        ] = "unverified"
    graph["guard_identity"] = _write_json(
        Path(graph["guard_identity"]["path"]),
        graph["guard"],
    )

    with pytest.raises(ValueError, match="terminal system claim guard contract differs"):
        _build(graph)


@pytest.mark.parametrize(
    "mutation",
    ["independent_policy", "replication_count", "sample_window", "interpretation"],
)
def test_comparison_rejects_terminal_replication_overclaim(
    tmp_path: Path,
    mutation: str,
) -> None:
    graph = _graph(tmp_path)
    if mutation == "independent_policy":
        graph["guard"]["claim_policy"][
            "independent_replication_claim_allowed"
        ] = True
    else:
        replication = graph["guard"]["evidence"][
            "matched_statistical_advantage"
        ]["replication_scope"]
        if mutation == "replication_count":
            replication["independent_replication_count"] = 1
            replication["independent_replication_supported"] = True
        elif mutation == "sample_window":
            replication["end_index_exclusive"] = 20_000
            replication["sample_count"] = 20_000
        else:
            replication["interpretation"] = (
                comparison.UNPAIRED_REPLICATION_INTERPRETATION
            )
    graph["guard_identity"] = _write_json(
        Path(graph["guard_identity"]["path"]),
        graph["guard"],
    )

    with pytest.raises(ValueError, match="terminal system claim guard contract differs"):
        _build(graph)


def test_comparison_normalizes_training_git_without_weakening_identity(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    report = _build(graph)
    assert report["scope"]["training_git"] == GIT

    training_path = Path(graph["quality"]["source_reports"]["cofitok_training"]["path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["git"] = {**GIT, "dirty": False}
    graph["quality"]["source_reports"]["cofitok_training"] = _write_json(
        training_path,
        training,
    )
    graph["fairness"]["terminal_sources"]["cofitok_training"] = graph["quality"][
        "source_reports"
    ]["cofitok_training"]
    _rebind(graph)
    with pytest.raises(ValueError, match="Git identity is invalid"):
        _build(graph)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("training_steps", "training report differs"),
        ("runtime_samples", "training budget"),
        ("sampling_progress", "sampling progress"),
        ("real_set", "physical evidence differs"),
        ("evaluator", "different evaluator evidence"),
    ],
)
def test_comparison_rejects_mismatched_terminal_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    graph = _graph(tmp_path)
    if mutation == "training_steps":
        path = Path(graph["quality"]["source_reports"]["dense_training"]["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["completed_steps"] = 99_999
        graph["quality"]["source_reports"]["dense_training"] = _write_json(
            path, payload
        )
        graph["fairness"]["terminal_sources"]["dense_training"] = graph["quality"][
            "source_reports"
        ]["dense_training"]
    elif mutation == "runtime_samples":
        graph["fairness"]["methods"]["dense_identity"]["training_cost"][
            "samples_seen"
        ] = 6_399_936
    elif mutation == "sampling_progress":
        method = graph["quality"]["terminal"]["methods"]["cofitok"]
        path = Path(method["sampling_progress"]["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["sample_sets"]["8"]["sha256"] = "f" * 64
        method["sampling_progress"] = _write_json(path, payload)
    elif mutation == "real_set":
        graph["quality"]["terminal"]["methods"]["dense_identity"]["real_set"][
            "sha256"
        ] = "f" * 64
    elif mutation == "evaluator":
        graph["quality"]["terminal"]["methods"]["dense_identity"][
            "metrics_runtime_environment_sha256"
        ] = "f" * 64
    _rebind(graph)
    with pytest.raises(ValueError, match=message):
        _build(graph)


def test_comparison_source_verifier_rejects_post_build_drift(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    report = _build(graph)
    source = Path(report["source_reports"]["official_related_methods"]["path"])
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after binding"):
        comparison.verify_source_reports(report)


@pytest.mark.parametrize("source", ["checkpoint", "checkpoint_integrity_manifest"])
def test_comparison_rejects_physical_checkpoint_drift(
    tmp_path: Path,
    source: str,
) -> None:
    graph = _graph(tmp_path)
    physical = graph["quality"]["terminal"]["physical_evidence"]["cofitok"]
    path = Path(physical[source]["path"])
    path.write_bytes(path.read_bytes() + b"drift")
    _rebind(graph)
    with pytest.raises(ValueError, match="changed after binding"):
        _build(graph)


def test_comparison_binds_physical_checkpoint_identities_to_rows(
    tmp_path: Path,
) -> None:
    report = _build(_graph(tmp_path))
    for row in report["matched_training_rows"]:
        assert row["checkpoint_identity"]["sha256"] == row["checkpoint_sha256"]
        assert (
            row["checkpoint_integrity_manifest_identity"]["path"]
            == row["checkpoint_integrity_manifest"]
        )
    verified = comparison.verify_source_reports(report)
    assert set(verified["physical_sources"]) == {"CoFiTok K=8", "Dense identity"}


@pytest.mark.parametrize("mutation", ["checkpoint", "sample_set", "evaluator"])
def test_comparison_rejects_class_fidelity_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    graph = _graph(tmp_path)
    qualification_path = Path(
        graph["quality"]["source_reports"]["class_fidelity_qualification"]["path"]
    )
    qualification = graph["quality"]["terminal"]["class_fidelity"]
    if mutation == "checkpoint":
        qualification["sampling_contract"]["cofitok_checkpoint_sha256"] = "f" * 64
    elif mutation == "sample_set":
        qualification["sampling_contract"]["dense_sample_set_sha256"] = "f" * 64
    else:
        qualification["sampling_contract"]["evaluator_runtime_environment_sha256"] = (
            "f" * 64
        )
    graph["quality"]["source_reports"]["class_fidelity_qualification"] = _write_json(
        qualification_path,
        qualification,
    )
    _rebind(graph)
    with pytest.raises(ValueError, match="class.fidelity"):
        _build(graph)


def test_comparison_rejects_raw_class_fidelity_source_drift(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    source = Path(graph["quality"]["source_reports"]["cofitok_class_fidelity"]["path"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["metrics"]["top1_correct"] += 1
    source.write_text(json.dumps(payload), encoding="utf-8")
    _rebind(graph)
    with pytest.raises(ValueError, match="changed after binding"):
        _build(graph)


@pytest.mark.parametrize("dataset_provenance", [None, [], "imagenet_256"])
def test_comparison_rejects_malformed_dataset_provenance(
    tmp_path: Path,
    dataset_provenance: object,
) -> None:
    graph = _graph(tmp_path)
    training_path = Path(graph["quality"]["source_reports"]["cofitok_training"]["path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["dataset_provenance"] = dataset_provenance
    identity = _write_json(training_path, training)
    graph["quality"]["source_reports"]["cofitok_training"] = identity
    graph["fairness"]["terminal_sources"]["cofitok_training"] = identity
    _rebind(graph)
    with pytest.raises(ValueError, match="training report differs"):
        _build(graph)


def test_main_is_idempotent_and_rejects_output_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph(tmp_path)
    output = tmp_path / "comparison_output"
    argv = [
        "build_generation_quality_bridge_comparison.py",
        "--terminal-system-guard",
        graph["guard_identity"]["path"],
        "--expected-terminal-system-guard-sha256",
        graph["guard_identity"]["sha256"],
        "--official-related",
        graph["official_identity"]["path"],
        "--expected-official-related-sha256",
        graph["official_identity"]["sha256"],
        "--output-dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    comparison.main()
    first = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    comparison.main()
    second = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    assert first == second
    assert set(first) == {
        "quality_bridge_comparison.json",
        "quality_bridge_comparison.md",
        "quality_bridge_comparison.csv",
    }

    (output / "quality_bridge_comparison.md").write_text(
        "drift\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="existing comparison output differs"):
        comparison.main()
