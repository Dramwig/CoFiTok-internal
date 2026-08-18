from __future__ import annotations

import copy

import pytest

from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from cofitok.generation.conditioning_ranking_sampling import (
    AUTHORIZATION_BOUNDARY,
    CLAIM_BOUNDARY,
    EXECUTION_RECEIPT_ROLE,
    EXPECTED_OUTPUT_ROOT,
    EXPECTED_POSTEVALUATION_DECISION,
    IDLE_GPU_EVIDENCE_ROLE,
    PREPARATION_ROLE,
    RUN_NAMES,
    SAMPLING_PROTOCOL,
    SCHEMA_VERSION,
    STAGE,
    build_sampling_execution_receipt,
    build_sampling_preparation,
    validate_idle_gpu_evidence,
)
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
)
from scripts.select_generation_conditioning_ranking_sampling_batch import (
    REPORT_ROLE as SELECTION_ROLE,
    select_four_arm_sampling_batch,
)


def _identity(name: str, character: str = "a") -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 123,
        "sha256": character * 64,
    }


def _git() -> dict:
    return {
        "revision": "b" * 40,
        "branch": "scale/generation-label-ranking-standing-authorization-v1",
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


def _postevaluation_and_checkpoints() -> tuple[dict, dict]:
    runs = {}
    evidence = {}
    for index, run in enumerate(RUN_NAMES, start=1):
        run_dir = f"/runs/{run}"
        digest = f"{index:x}" * 64
        runs[run] = {
            "run_dir": run_dir,
            "checkpoint_bytes": 1_000 + index,
            "checkpoint_sha256": digest,
        }
        evidence[run] = {
            "checkpoint": {
                "path": f"{run_dir}/checkpoint_step_00001000.pt",
                "bytes": 1_000 + index,
                "sha256": digest,
            },
            "integrity_manifest": {
                "path": f"{run_dir}/checkpoint_step_00001000.pt.integrity.json",
                "bytes": 200 + index,
                "sha256": f"{index + 4:x}" * 64,
            },
            "training_report": {
                "path": f"{run_dir}/training_report.json",
                "bytes": 300 + index,
                "sha256": f"{index + 8:x}" * 64,
            },
            "step": 1_000,
        }
    return (
        {
            "schema_version": 1,
            "role": "generation_conditioning_ranking_four_arm_postevaluation",
            "status": "completed",
            "decision": copy.deepcopy(EXPECTED_POSTEVALUATION_DECISION),
            "training_contract": {
                "revision": "1" * 40,
                "branch": "scale/generation-label-ranking-standing-authorization-v1",
                "runs": runs,
            },
        },
        evidence,
    )


def _classifier() -> dict:
    return {
        "name": CLASS_FIDELITY_CLASSIFIER_NAME,
        "weights_path": "/weights/resnet50.pth",
        "weights_bytes": CLASS_FIDELITY_CLASSIFIER_BYTES,
        "weights_sha256": CLASS_FIDELITY_CLASSIFIER_SHA256,
        "num_classes": 1_000,
        "categories_sha256": "c" * 64,
        "preprocessing": {"crop_size": [224]},
    }


def _preparation() -> dict:
    postevaluation, checkpoints = _postevaluation_and_checkpoints()
    return build_sampling_preparation(
        postevaluation=postevaluation,
        postevaluation_identity=_identity("postevaluation"),
        checkpoint_evidence=checkpoints,
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=_identity("standing", "d"),
        classifier=_classifier(),
        runbook_identity=_identity("runbook", "e"),
        builder_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )


def test_sampling_preparation_binds_sources_and_stays_non_authorizing() -> None:
    report = _preparation()

    assert report["role"] == PREPARATION_ROLE
    assert report["status"] == "pass"
    assert report["sampling_protocol"] == SAMPLING_PROTOCOL
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert report["authorization_boundary"]["training_allowed"] is False
    assert report["claim_boundary"] == CLAIM_BOUNDARY


def test_sampling_preparation_rejects_wrong_decision_or_checkpoint() -> None:
    postevaluation, checkpoints = _postevaluation_and_checkpoints()
    postevaluation["decision"]["method_passes"]["dense_identity"] = False
    with pytest.raises(ValueError, match="did not authorize sampling"):
        build_sampling_preparation(
            postevaluation=postevaluation,
            postevaluation_identity=_identity("postevaluation"),
            checkpoint_evidence=checkpoints,
            standing_authorization=_standing_authorization(),
            standing_authorization_identity=_identity("standing", "d"),
            classifier=_classifier(),
            runbook_identity=_identity("runbook", "e"),
            builder_git=_git(),
            expected_revision=_git()["revision"],
            expected_branch=_git()["branch"],
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )

    postevaluation, checkpoints = _postevaluation_and_checkpoints()
    checkpoints[RUN_NAMES[0]]["checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint evidence differs"):
        build_sampling_preparation(
            postevaluation=postevaluation,
            postevaluation_identity=_identity("postevaluation"),
            checkpoint_evidence=checkpoints,
            standing_authorization=_standing_authorization(),
            standing_authorization_identity=_identity("standing", "d"),
            classifier=_classifier(),
            runbook_identity=_identity("runbook", "e"),
            builder_git=_git(),
            expected_revision=_git()["revision"],
            expected_branch=_git()["branch"],
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )


def _preflight_report(run: str, batch_size: int, throughput: float) -> dict:
    return {
        "status": "passed",
        "request": {
            "batch_size": batch_size,
            "prefix_budget": 8 if "cofitok" in run else 1,
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
        "runtime_environment": {"device": {"name": "test"}},
        "runtime_environment_sha256": "unused-by-pure-selector",
    }


def test_four_arm_selector_uses_worst_arm_and_requires_baseline() -> None:
    import scripts.select_generation_conditioning_ranking_sampling_batch as selector

    original = selector._validated_runtime_environment_sha
    selector._validated_runtime_environment_sha = lambda report: "f" * 64
    try:
        rows = []
        for batch_size, base_speed in ((16, 10.0), (32, 18.0), (64, 15.0)):
            rows.append(
                {
                    "batch_size": batch_size,
                    "arms": {
                        run: {
                            "report": _preflight_report(
                                run,
                                batch_size,
                                base_speed - index,
                            ),
                            "identity": None,
                        }
                        for index, run in enumerate(RUN_NAMES)
                    },
                }
            )
        selected = select_four_arm_sampling_batch(
            rows,
            baseline_batch_size=16,
            max_memory_fraction=0.9,
        )
        assert selected["selected"]["batch_size"] == 32

        rows[0]["arms"][RUN_NAMES[0]]["report"]["status"] = "failed"
        with pytest.raises(ValueError, match="baseline did not pass"):
            select_four_arm_sampling_batch(
                rows,
                baseline_batch_size=16,
                max_memory_fraction=0.9,
            )
    finally:
        selector._validated_runtime_environment_sha = original


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
            {"observed_at_unix": 1_000.0 + index, "gpu_compute_pids": []}
            for index in range(5)
        ],
    }


def test_execution_receipt_binds_idle_evidence_and_selected_batch() -> None:
    preparation = _preparation()
    idle = _idle_evidence()
    validate_idle_gpu_evidence(
        idle,
        expected_git=_git(),
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )
    selection = {
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
    receipt = build_sampling_execution_receipt(
        preparation=preparation,
        preparation_identity=_identity("preparation", "1"),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=_identity("standing", "d"),
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=_identity("idle", "2"),
        batch_selection=selection,
        batch_selection_identity=_identity("selection", "3"),
        receipt_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )

    assert receipt["role"] == EXECUTION_RECEIPT_ROLE
    assert receipt["status"] == "authorized"
    assert receipt["selected_batch"]["batch_size"] == 32
    assert receipt["authorization_boundary"]["training_allowed"] is False
    assert receipt["claim_boundary"]["cofitok_specific_advantage_claim_allowed"] is False


def test_idle_gpu_evidence_rejects_busy_or_nonconsecutive_rows() -> None:
    busy = _idle_evidence()
    busy["observations"][3]["gpu_compute_pids"] = [123]
    with pytest.raises(ValueError, match="observation differs"):
        validate_idle_gpu_evidence(
            busy,
            expected_git=_git(),
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )

    reversed_time = _idle_evidence()
    reversed_time["observations"][4]["observed_at_unix"] = 900.0
    with pytest.raises(ValueError, match="observation differs"):
        validate_idle_gpu_evidence(
            reversed_time,
            expected_git=_git(),
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )
