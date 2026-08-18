from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    AUTHORIZATION_BOUNDARY,
    CLAIM_BOUNDARY,
    EXECUTION_RECEIPT_ROLE,
    EXPECTED_CHECKPOINT_FILENAME,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_HELDOUT_DECISION,
    EXPECTED_OUTPUT_ROOT,
    HELDOUT_EVALUATOR_GIT,
    HELDOUT_ROLE,
    HELDOUT_STAGE,
    IDLE_GPU_EVIDENCE_ROLE,
    METHOD_PREFIX_BUDGETS,
    PREPARATION_ROLE,
    RUN_METHODS,
    RUN_NAMES,
    SAMPLE_RUN_NAME,
    SAMPLING_PROTOCOL,
    SCHEMA_VERSION,
    STAGE,
    TRAINING_EXECUTION_BOUNDARY,
    TRAINING_EXECUTION_CLAIM_BOUNDARY,
    TRAINING_EXECUTION_RECEIPT_ROLE,
    TRAINING_GIT,
    TRAINING_OUTPUT_ROOT,
    TRAINING_STAGE,
    TRAINING_STATUS_ROLE,
    build_sampling_execution_receipt,
    build_sampling_preparation,
    validate_idle_gpu_evidence,
    validate_stream_independence,
)
from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
)
from cofitok.inference_replay import file_identity
from scripts.build_generation_conditioning_ranking_posttraining_sampling_confirmation import (
    build_posttraining_sampling_confirmation,
)
from scripts.evaluate_generation_conditioning_ranking_samples import (
    LEGACY_SAMPLING_STAGE,
    paired_class_fidelity_summary,
    validate_sampling_pair,
)
from scripts.select_generation_conditioning_ranking_sampling_batch import (
    REPORT_ROLE as SELECTION_ROLE,
    select_four_arm_sampling_batch,
    validate_completed_selection,
)


def _identity(path: str, character: str = "a", *, size: int = 123) -> dict:
    return {"path": path, "bytes": size, "sha256": character * 64}


def _git() -> dict:
    return {
        "revision": "a" * 40,
        "branch": (
            "scale/generation-label-ranking-posttraining-"
            "5k-sampling-confirmation-v1"
        ),
        "tracked_dirty": False,
    }


def _standing_authorization() -> dict:
    return {
        "schema_version": 1,
        "role": "cofitok_standing_experiment_authorization_record",
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_TEXT,
            "received_at": "2026-08-13T00:11:00+08:00",
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _classifier() -> dict:
    return {
        "name": CLASS_FIDELITY_CLASSIFIER_NAME,
        "weights_path": "/weights/resnet50-11ad3fa6.pth",
        "weights_bytes": CLASS_FIDELITY_CLASSIFIER_BYTES,
        "weights_sha256": CLASS_FIDELITY_CLASSIFIER_SHA256,
        "num_classes": 1_000,
        "categories_sha256": "c" * 64,
        "preprocessing": {"crop_size": [224]},
    }


def _source_bundle() -> dict:
    status_identity = _identity("/evidence/training_status.json", "1")
    standing_identity = _identity("/evidence/standing.json", "2")
    runs = {}
    status_runs = {}
    checkpoints = {}
    heldout_checkpoints = {}
    heldout_integrity = {}
    heldout_reports = {}
    digest_characters = ("3", "4", "5", "6")
    for index, (run, character) in enumerate(
        zip(RUN_NAMES, digest_characters), start=1
    ):
        run_dir = f"{TRAINING_OUTPUT_ROOT}/{run}"
        checkpoint = _identity(
            f"{run_dir}/{EXPECTED_CHECKPOINT_FILENAME}",
            character,
            size=1_000 + index,
        )
        integrity = _identity(
            f"{checkpoint['path']}.integrity.json",
            f"{index + 6:x}",
            size=200 + index,
        )
        training_report = _identity(
            f"{run_dir}/training_report.json",
            f"{index + 10:x}",
            size=300 + index,
        )
        runs[run] = {
            "run_dir": run_dir,
            "checkpoint_bytes": checkpoint["bytes"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_integrity_manifest": copy.deepcopy(integrity),
            "final_metrics": {"step": EXPECTED_CHECKPOINT_STEP},
        }
        status_runs[run] = {
            "checkpoint": copy.deepcopy(checkpoint),
            "integrity_manifest": copy.deepcopy(integrity),
            "training_report": copy.deepcopy(training_report),
        }
        checkpoints[run] = {
            "checkpoint": copy.deepcopy(checkpoint),
            "integrity_manifest": copy.deepcopy(integrity),
            "training_report": copy.deepcopy(training_report),
            "step": EXPECTED_CHECKPOINT_STEP,
        }
        heldout_checkpoints[run] = copy.deepcopy(checkpoint)
        heldout_integrity[run] = copy.deepcopy(integrity)
        heldout_reports[run] = copy.deepcopy(training_report)

    training_status = {
        "schema_version": SCHEMA_VERSION,
        "role": TRAINING_STATUS_ROLE,
        "status": "completed",
        "stage": TRAINING_STAGE,
        "revision": TRAINING_GIT["revision"],
        "branch": TRAINING_GIT["branch"],
        "output_root": TRAINING_OUTPUT_ROOT,
        "runs": status_runs,
    }
    training_receipt = {
        "schema_version": SCHEMA_VERSION,
        "role": TRAINING_EXECUTION_RECEIPT_ROLE,
        "status": "authorized",
        "authorization_mode": "active_standing_experiment_authorization",
        "stage": TRAINING_STAGE,
        "authorized_revision": TRAINING_GIT["revision"],
        "authorized_branch": TRAINING_GIT["branch"],
        "output_root": TRAINING_OUTPUT_ROOT,
        "authorization_boundary": copy.deepcopy(TRAINING_EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(TRAINING_EXECUTION_CLAIM_BOUNDARY),
        "source_reports": {
            "standing_authorization": copy.deepcopy(standing_identity)
        },
    }
    heldout = {
        "schema_version": SCHEMA_VERSION,
        "role": HELDOUT_ROLE,
        "status": "completed",
        "stage": HELDOUT_STAGE,
        "training_git": copy.deepcopy(TRAINING_GIT),
        "evaluator_git": copy.deepcopy(HELDOUT_EVALUATOR_GIT),
        "output_root": TRAINING_OUTPUT_ROOT,
        "sources": {
            "training_status": copy.deepcopy(status_identity),
            "checkpoints": heldout_checkpoints,
            "checkpoint_integrity_manifests": heldout_integrity,
            "training_reports": heldout_reports,
        },
        "training_contract": {
            "revision": TRAINING_GIT["revision"],
            "branch": TRAINING_GIT["branch"],
            "output_root": TRAINING_OUTPUT_ROOT,
            "runs": runs,
        },
        "methods": {
            "cofitok": {"pass": True},
            "dense_identity": {"pass": True},
        },
        "decision": copy.deepcopy(EXPECTED_HELDOUT_DECISION),
    }
    return {
        "heldout": heldout,
        "heldout_identity": _identity("/evidence/heldout.json", "d"),
        "training_status": training_status,
        "training_status_identity": status_identity,
        "training_receipt": training_receipt,
        "training_receipt_identity": _identity(
            f"{TRAINING_OUTPUT_ROOT}/reports/execution_receipt.json", "e"
        ),
        "standing": _standing_authorization(),
        "standing_identity": standing_identity,
        "checkpoints": checkpoints,
    }


def _preparation() -> dict:
    bundle = _source_bundle()
    return build_sampling_preparation(
        heldout_evaluation=bundle["heldout"],
        heldout_evaluation_identity=bundle["heldout_identity"],
        training_status=bundle["training_status"],
        training_status_identity=bundle["training_status_identity"],
        training_execution_receipt=bundle["training_receipt"],
        training_execution_receipt_identity=bundle[
            "training_receipt_identity"
        ],
        checkpoint_evidence=bundle["checkpoints"],
        standing_authorization=bundle["standing"],
        standing_authorization_identity=bundle["standing_identity"],
        classifier=_classifier(),
        runbook_identity=_identity("/checkout/artifacts/runbooks/runbook.sh", "f"),
        builder_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )


def _idle_evidence() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": IDLE_GPU_EVIDENCE_ROLE,
        "status": "pass",
        "stage": STAGE,
        "output_root": EXPECTED_OUTPUT_ROOT,
        "git": _git(),
        "required_consecutive_idle_polls": 5,
        "observations": [
            {
                "poll_index": index + 1,
                "observed_at_unix": 1_000.0 + index,
                "gpu_compute_pids": [],
            }
            for index in range(5)
        ],
    }


def _minimal_selection(preparation: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": SELECTION_ROLE,
        "status": "selected",
        "stage": STAGE,
        "output_root": EXPECTED_OUTPUT_ROOT,
        "git": _git(),
        "sampling_protocol": copy.deepcopy(SAMPLING_PROTOCOL),
        "checkpoints": copy.deepcopy(
            preparation["source_reports"]["training_checkpoints"]
        ),
        "selected": {
            "batch_size": 32,
            "selection_score_images_per_second": 10.0,
            "max_memory_fraction": 0.5,
            "estimated_speedup_over_baseline": 1.5,
        },
    }


def _sampling_source(*, method: str, checkpoint_character: str) -> dict:
    budget = METHOD_PREFIX_BUDGETS[method]
    sampling = {
        "num_samples": SAMPLING_PROTOCOL["num_samples_per_arm"],
        "start_index": SAMPLING_PROTOCOL["start_index"],
        "sample_steps": SAMPLING_PROTOCOL["sample_steps"],
        "num_train_timesteps": SAMPLING_PROTOCOL["num_train_timesteps"],
        "guidance_scale": SAMPLING_PROTOCOL["guidance_scale"],
        "guidance_rescale": SAMPLING_PROTOCOL["guidance_rescale"],
        "cfg_batch_mode": SAMPLING_PROTOCOL["cfg_batch_mode"],
        "eta": SAMPLING_PROTOCOL["eta"],
        "seed": SAMPLING_PROTOCOL["seed"],
        "precision": SAMPLING_PROTOCOL["precision"],
        "class_schedule": SAMPLING_PROTOCOL["class_schedule"],
        "sampler": SAMPLING_PROTOCOL["sampler"],
        "clip_x0": SAMPLING_PROTOCOL["clip_x0"],
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "prefix_budgets": [budget],
        "image_shape": [3, 256, 256],
        "actual_timesteps": select_sampling_timesteps(
            SAMPLING_PROTOCOL["num_train_timesteps"],
            SAMPLING_PROTOCOL["sample_steps"],
        ),
    }
    return {
        "sampling": sampling,
        "weights": "ema",
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "selected_prefix_budget": budget,
        "checkpoint_sha256": checkpoint_character * 64,
        "git": _git(),
        "runtime_environment": {"torch": "test"},
        "runtime_environment_sha256": "b" * 64,
    }


def _paired_row(index: int, *, delta: float = 0.1) -> dict:
    control_log = -7.0
    return {
        "sample_index": index,
        "requested_class": index % 1_000,
        "control": {
            "target_log_probability": control_log,
            "target_probability": 0.001,
            "predicted_class": index % 1_000,
            "top1_correct": index % 4 == 0,
            "top5_correct": index % 2 == 0,
        },
        "ranked": {
            "target_log_probability": control_log + delta,
            "target_probability": 0.0011,
            "predicted_class": index % 1_000,
            "top1_correct": index % 4 == 0,
            "top5_correct": index % 2 == 0,
        },
        "ranked_minus_control_target_log_probability": delta,
    }


def test_posttraining_stream_is_exactly_independent_and_drift_is_rejected() -> None:
    stream = validate_stream_independence(SAMPLING_PROTOCOL)

    assert stream["global_index_interval_inclusive"] == [5_000, 9_999]
    assert stream["per_sample_seed_interval_inclusive"] == [511_020, 516_019]
    assert stream["global_index_overlap_count"] == 0
    assert stream["per_sample_seed_overlap_count"] == 0
    assert stream["balanced_class_count_per_class"] == 5

    drift = copy.deepcopy(SAMPLING_PROTOCOL)
    drift["random_stream"]["global_index_interval_inclusive"] = [0, 4_999]
    with pytest.raises(ValueError, match="protocol differs"):
        validate_stream_independence(drift)


def test_preparation_binds_exact_step5000_sources_and_stays_non_authorizing() -> None:
    report = _preparation()

    assert report["role"] == PREPARATION_ROLE
    assert report["sampling_protocol"] == SAMPLING_PROTOCOL
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert report["authorization_boundary"]["training_allowed"] is False
    assert report["authorization_boundary"]["full_100k_or_300k_launch_allowed"] is False
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    for run in RUN_NAMES:
        checkpoint = report["source_reports"]["training_checkpoints"][run]
        assert checkpoint["step"] == EXPECTED_CHECKPOINT_STEP
        assert checkpoint["checkpoint"]["path"].endswith(
            f"/{EXPECTED_CHECKPOINT_FILENAME}"
        )

    bundle = _source_bundle()
    bundle["checkpoints"]["ranked_cofitok"]["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint evidence differs"):
        build_sampling_preparation(
            heldout_evaluation=bundle["heldout"],
            heldout_evaluation_identity=bundle["heldout_identity"],
            training_status=bundle["training_status"],
            training_status_identity=bundle["training_status_identity"],
            training_execution_receipt=bundle["training_receipt"],
            training_execution_receipt_identity=bundle[
                "training_receipt_identity"
            ],
            checkpoint_evidence=bundle["checkpoints"],
            standing_authorization=bundle["standing"],
            standing_authorization_identity=bundle["standing_identity"],
            classifier=_classifier(),
            runbook_identity=_identity("/checkout/runbook.sh", "f"),
            builder_git=_git(),
            expected_revision=_git()["revision"],
            expected_branch=_git()["branch"],
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )


def test_posttraining_paired_evaluator_uses_global_indices_5000_to_9999() -> None:
    rows = [_paired_row(index) for index in range(5_000, 5_008)]
    metrics = paired_class_fidelity_summary(rows, expected_start_index=5_000)
    assert metrics["paired_sign_test"]["positive"] == 8
    with pytest.raises(ValueError, match="indices"):
        paired_class_fidelity_summary(rows)

    control = _sampling_source(method="cofitok", checkpoint_character="1")
    ranked = _sampling_source(method="cofitok", checkpoint_character="2")
    pair = validate_sampling_pair(
        control,
        ranked,
        method="cofitok",
        sampling_stage=STAGE,
    )
    assert pair["sampling_stage"] == STAGE
    assert pair["checkpoint_step"] == 5_000
    assert pair["sampling"]["start_index"] == 5_000
    assert pair["sampling"]["seed"] == 506_020

    legacy_control = copy.deepcopy(control)
    legacy_ranked = copy.deepcopy(ranked)
    for source, character in ((legacy_control, "3"), (legacy_ranked, "4")):
        source["sampling"].update(
            {
                "start_index": 0,
                "seed": 406_020,
            }
        )
        source["checkpoint_step"] = 1_000
        source["checkpoint_sha256"] = character * 64
    legacy_pair = validate_sampling_pair(
        legacy_control,
        legacy_ranked,
        method="cofitok",
        sampling_stage=LEGACY_SAMPLING_STAGE,
    )
    assert "sampling_stage" not in legacy_pair
    assert legacy_pair["sampling"]["start_index"] == 0


def _preflight_report(
    preparation: dict,
    *,
    run: str,
    batch_size: int,
    throughput: float,
) -> dict:
    checkpoint = preparation["source_reports"]["training_checkpoints"][run]
    environment = {"device": {"name": "test-gpu"}, "torch": {"version": "test"}}
    return {
        "status": "passed",
        "git": copy.deepcopy(preparation["git"]),
        "checkpoint": checkpoint["checkpoint"]["path"],
        "checkpoint_sha256": checkpoint["checkpoint"]["sha256"],
        "checkpoint_integrity_manifest": checkpoint["integrity_manifest"]["path"],
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "weights": "ema",
        "request": {
            "batch_size": batch_size,
            "prefix_budget": METHOD_PREFIX_BUDGETS[RUN_METHODS[run]],
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "precision": "bf16",
            "warmup_forwards": 2,
            "measured_forwards": 5,
        },
        "result": {
            "output_images_per_second": throughput,
            "device_total_memory_bytes": 100_000,
            "cuda_memory_after_forward": {"peak_allocated_bytes": 50_000},
        },
        "runtime_environment": environment,
        "runtime_environment_sha256": runtime_environment_sha256(environment),
    }


def test_selector_accepts_posttraining_stage_and_replays_physical_preflights(
    tmp_path: Path,
) -> None:
    preparation = _preparation()
    preparation_identity = _identity("/evidence/preparation.json", "1")
    rows = []
    for batch_size, base_speed in ((16, 10.0), (32, 20.0), (64, 15.0)):
        arms = {}
        for run_index, run in enumerate(RUN_NAMES):
            report = _preflight_report(
                preparation,
                run=run,
                batch_size=batch_size,
                throughput=base_speed - run_index,
            )
            report_path = tmp_path / f"{run}_{batch_size}.json"
            report_path.write_text(
                json.dumps(report, sort_keys=True),
                encoding="utf-8",
            )
            arms[run] = {
                "report": report,
                "identity": file_identity(report_path),
            }
        rows.append({"batch_size": batch_size, "arms": arms})
    selected = select_four_arm_sampling_batch(
        rows,
        baseline_batch_size=SAMPLING_PROTOCOL["baseline_batch_size"],
        max_memory_fraction=SAMPLING_PROTOCOL["maximum_memory_fraction"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "role": SELECTION_ROLE,
        "status": "selected",
        "stage": STAGE,
        "output_root": EXPECTED_OUTPUT_ROOT,
        "git": copy.deepcopy(preparation["git"]),
        "source_reports": {"preparation": preparation_identity},
        "sampling_protocol": copy.deepcopy(SAMPLING_PROTOCOL),
        "checkpoints": copy.deepcopy(
            preparation["source_reports"]["training_checkpoints"]
        ),
        **selected,
    }

    replayed = validate_completed_selection(
        report,
        preparation=preparation,
        preparation_identity=preparation_identity,
    )
    assert replayed["selected"]["batch_size"] == 32
    assert replayed["stage"] == STAGE
    assert SAMPLE_RUN_NAME.endswith("independent_stream_v1")

    drift = copy.deepcopy(report)
    drift["sampling_protocol"]["start_index"] = 0
    with pytest.raises(ValueError, match="selection differs"):
        validate_completed_selection(
            drift,
            preparation=preparation,
            preparation_identity=preparation_identity,
        )


def test_execution_receipt_requires_five_indexed_idle_polls_and_is_non_authorizing() -> None:
    preparation = _preparation()
    idle = _idle_evidence()
    validate_idle_gpu_evidence(
        idle,
        expected_git=_git(),
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )
    receipt = build_sampling_execution_receipt(
        preparation=preparation,
        preparation_identity=_identity("/evidence/preparation.json", "1"),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=preparation["source_reports"][
            "standing_authorization"
        ],
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=_identity("/evidence/idle.json", "2"),
        batch_selection=_minimal_selection(preparation),
        batch_selection_identity=_identity("/evidence/selection.json", "3"),
        receipt_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )

    assert receipt["role"] == EXECUTION_RECEIPT_ROLE
    assert receipt["status"] == "authorized"
    assert receipt["authorization_boundary"]["training_allowed"] is False
    assert receipt["authorization_boundary"]["checkpoint_promotion_allowed"] is False
    assert receipt["claim_boundary"]["authorizes_release"] is False

    wrong_index = _idle_evidence()
    wrong_index["observations"][2]["poll_index"] = 99
    with pytest.raises(ValueError, match="observation differs"):
        validate_idle_gpu_evidence(
            wrong_index,
            expected_git=_git(),
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )


def _final_inputs() -> dict:
    bundle = _source_bundle()
    preparation = _preparation()
    selection = _minimal_selection(preparation)
    receipt = build_sampling_execution_receipt(
        preparation=preparation,
        preparation_identity=_identity("/evidence/preparation.json", "1"),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=preparation["source_reports"][
            "standing_authorization"
        ],
        idle_gpu_evidence=_idle_evidence(),
        idle_gpu_evidence_identity=_identity("/evidence/idle.json", "2"),
        batch_selection=selection,
        batch_selection_identity=_identity("/evidence/selection.json", "3"),
        receipt_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )
    metrics = paired_class_fidelity_summary(
        [_paired_row(index) for index in range(5_000, 5_008)],
        expected_start_index=5_000,
    )
    generation = {
        "control_cofitok": {"fid": 100.0},
        "ranked_cofitok": {"fid": 105.0},
        "control_dense_identity": {"fid": 90.0},
        "ranked_dense_identity": {"fid": 95.0},
    }
    return {
        "heldout_evaluation": bundle["heldout"],
        "heldout_identity": bundle["heldout_identity"],
        "preparation": preparation,
        "preparation_identity": _identity("/evidence/preparation.json", "1"),
        "execution_receipt": receipt,
        "execution_receipt_identity": _identity("/evidence/receipt.json", "4"),
        "checkpoint_evidence": preparation["source_reports"][
            "training_checkpoints"
        ],
        "sampling_provenance": {run: {"run": run} for run in RUN_NAMES},
        "sampling_sources": {
            run: _identity(f"/evidence/{run}_sampling.json", "5")
            for run in RUN_NAMES
        },
        "generation_reports": generation,
        "generation_sources": {
            run: _identity(f"/evidence/{run}_metrics.json", "6")
            for run in RUN_NAMES
        },
        "paired_reports": {
            "cofitok": {"metrics": copy.deepcopy(metrics)},
            "dense_identity": {"metrics": copy.deepcopy(metrics)},
        },
        "paired_sources": {
            "cofitok": _identity("/evidence/cofitok_paired.json", "7"),
            "dense_identity": _identity("/evidence/dense_paired.json", "8"),
        },
        "git": _git(),
        "output_root": EXPECTED_OUTPUT_ROOT,
    }


def test_final_confirmation_routes_shared_asymmetric_and_failed_evidence() -> None:
    shared = build_posttraining_sampling_confirmation(**_final_inputs())
    assert shared["decision"] == {
        "method_passes": {"cofitok": True, "dense_identity": True},
        "shared_posttraining_generated_class_alignment_recovery_confirmed": True,
        "cofitok_specific_advantage_claim_allowed": False,
        "recommended_next_action": (
            "retain_shared_conditioning_ranking_recipe_for_separately_"
            "authorized_future_scaling"
        ),
    }
    assert shared["claim_boundary"] == CLAIM_BOUNDARY

    asymmetric_inputs = _final_inputs()
    asymmetric_inputs["generation_reports"]["ranked_cofitok"]["fid"] = 112.0
    asymmetric = build_posttraining_sampling_confirmation(**asymmetric_inputs)
    assert asymmetric["decision"]["method_passes"] == {
        "cofitok": False,
        "dense_identity": True,
    }
    assert asymmetric["decision"]["recommended_next_action"] == (
        "reject_shared_repair_due_posttraining_method_asymmetry"
    )

    failed_inputs = _final_inputs()
    failing_metrics = paired_class_fidelity_summary(
        [_paired_row(index, delta=0.001) for index in range(5_000, 5_008)],
        expected_start_index=5_000,
    )
    failed_inputs["paired_reports"] = {
        "cofitok": {"metrics": copy.deepcopy(failing_metrics)},
        "dense_identity": {"metrics": copy.deepcopy(failing_metrics)},
    }
    failed = build_posttraining_sampling_confirmation(**failed_inputs)
    assert failed["decision"]["method_passes"] == {
        "cofitok": False,
        "dense_identity": False,
    }
    assert failed["decision"]["recommended_next_action"] == (
        "revise_training_time_semantic_alignment_objective"
    )
