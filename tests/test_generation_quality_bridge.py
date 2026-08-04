from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.diffusion import select_sampling_timesteps
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation.quality_bridge import (
    EXECUTION_APPROVAL_BOUNDARY,
    LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY,
    QUALITY_BRIDGE_APPROVAL_ROLE,
    QUALITY_BRIDGE_APPROVAL_SCOPE,
    QUALITY_BRIDGE_LAUNCH_RECEIPT_ROLE,
    QUALITY_BRIDGE_RECIPE_STAGE,
    RESULT_AUTHORIZATION_BOUNDARY,
    _quality_screen,
    _terminal_method_row,
    _validate_training_audit,
    build_quality_bridge_launch_receipt,
    build_quality_bridge_preparation,
    build_quality_bridge_result,
    validate_quality_bridge_execution_approval,
    validate_quality_bridge_launch_receipt_contract,
)
from cofitok.generation_recipe import (
    generation_training_recipe_contract,
    infer_generation_training_stage,
)
from scripts.validate_generation_configs import validate_pair
from scripts.validate_generation_training_pair import validate_training_pair
from scripts import build_generation_quality_bridge_result as result_builder


ROOT = Path(__file__).resolve().parents[1]
COFITOK_CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
)
DENSE_CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json"
)
SOURCE_COFITOK_CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_50k.json"
)
SOURCE_DENSE_CONFIG = (
    ROOT
    / "configs/generation/"
    "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_50k.json"
)

QUALITY_BRIDGE_DIRECT_ENTRYPOINTS = (
    "build_generation_quality_bridge_preparation.py",
    "validate_generation_quality_bridge_preparation.py",
    "validate_generation_quality_bridge_execution_approval.py",
    "build_generation_quality_bridge_launch_receipt.py",
    "validate_generation_quality_bridge_launch_receipt.py",
    "build_generation_quality_bridge_result.py",
    "verify_generation_quality_bridge_result.py",
)


def _identity(path: str) -> dict[str, object]:
    return {"path": path, "bytes": 123, "sha256": "a" * 64}


def _provenance(dataset: str) -> dict[str, object]:
    spec = FORMAL_GENERATION_DATASETS[dataset]
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": dataset,
        "dataset_root": f"/root/autodl-tmp/CoFiTok/datasets/{dataset}",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": spec.manifest_bytes,
            "sha256": spec.manifest_sha256,
        },
        "splits": {"train": spec.train_images, "val": spec.val_images},
        "issues": [],
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def _source_training(path: Path, *, parameters: int) -> dict[str, object]:
    config = config_to_dict(load_config(path))
    return {
        "training_complete": True,
        "completed_steps": 50_000,
        "target_steps": 50_000,
        "parameter_count": parameters,
        "config": config,
        "dataset_provenance": _provenance("imagenet_256_10pct"),
        "final_metrics": {"samples_seen": 3_200_000},
    }


def _completed_bridge_training(path: Path, *, parameters: int) -> dict[str, object]:
    config = config_to_dict(load_config(path))
    provenance = _provenance("imagenet_256")
    return {
        "training_complete": True,
        "completed_steps": 100_000,
        "target_steps": 100_000,
        "parameter_count": parameters,
        "git": {
            "dirty": False,
            "revision": "b" * 40,
            "branch": "scale/generation-quality-bridge",
        },
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00100000.pt",
            "step": 100_000,
            "dataset_identity_sha256": provenance["identity_sha256"],
        },
        "config": config,
        "dataset_provenance": provenance,
        "training_authorization": None,
    }


def _gate() -> dict[str, object]:
    source_reports = {
        "cofitok_training": _identity("/evidence/cofitok/training_report.json"),
        "dense_training": _identity("/evidence/dense/training_report.json"),
        "cofitok_generation": _identity(
            "/evidence/cofitok/metrics/generation_metrics_report.json"
        ),
        "dense_generation": _identity(
            "/evidence/dense/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": _identity(
            "/evidence/cofitok/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": _identity(
            "/evidence/dense/checkpoint_evaluation_report.json"
        ),
    }
    return {
        "schema_version": 2,
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "status": "fail",
        "decision": "hold",
        "thresholds": {
            "max_absolute_fid": 100.0,
            "max_endpoint_regression": 0.05,
            "max_fid_regression": 0.05,
            "max_precision_regression": 0.05,
            "max_recall_regression": 0.05,
            "min_coarse_token_energy_ratio": 0.05,
            "min_precision": 0.3,
            "min_recall": 0.3,
            "min_samples": 10_000,
        },
        "summary": {
            "cofitok_fid": 138.29702495267782,
            "dense_fid": 151.4476773464495,
            "cofitok_recall": 0.008679999969899654,
            "dense_recall": 0.00977999996393919,
        },
        "gates": [
            {"name": "fid_within_tolerance", "passed": True},
            {"name": "ordered_prefix_path", "passed": True},
            {"name": "coarse_token_utilization", "passed": True},
            {"name": "absolute_fid_quality", "passed": False},
        ],
        "source_reports": source_reports,
    }


def _preparation() -> dict[str, object]:
    cofitok = load_config(COFITOK_CONFIG)
    dense = load_config(DENSE_CONFIG)
    validation = validate_pair(
        cofitok,
        dense,
        max_parameter_gap=0.02,
        stage="stability_quality_bridge",
    )
    gate = _gate()
    verification = {
        "status": "verified",
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "source_reports": gate["source_reports"],
    }
    return build_quality_bridge_preparation(
        promotion_gate=gate,
        promotion_gate_identity=_identity("/evidence/promotion_gate.json"),
        gate_source_verification=verification,
        source_cofitok_training=_source_training(
            SOURCE_COFITOK_CONFIG,
            parameters=validation["cofitok"]["parameter_count"],
        ),
        source_dense_training=_source_training(
            SOURCE_DENSE_CONFIG,
            parameters=validation["dense"]["parameter_count"],
        ),
        cofitok_config=config_to_dict(cofitok),
        dense_config=config_to_dict(dense),
        cofitok_config_identity=_identity(COFITOK_CONFIG.as_posix()),
        dense_config_identity=_identity(DENSE_CONFIG.as_posix()),
        config_validation=validation,
    )


def test_quality_bridge_recipe_is_matched_full_data_base128_100k() -> None:
    cofitok = config_to_dict(load_config(COFITOK_CONFIG))
    dense = config_to_dict(load_config(DENSE_CONFIG))

    assert infer_generation_training_stage(cofitok, dense) == (
        "stability_quality_bridge"
    )
    contract = generation_training_recipe_contract(
        cofitok,
        dense,
        stage="stability_quality_bridge",
    )

    assert contract["valid"] is True, contract["issues"]
    assert contract["expected_shared"]["data.dataset"] == "imagenet_256"
    assert contract["expected_shared"]["model.base_channels"] == 128
    assert contract["expected_shared"]["runtime.steps"] == 100_000
    assert contract["expected_shared"]["runtime.protected_checkpoint_steps"] == [
        50_000,
        100_000,
    ]
    assert contract["expected_shared"]["loss.rollout_consistency_warmup_steps"] == 10_000
    assert contract["expected_shared"]["loss.ema_teacher_consistency_start_step"] == 30_000
    assert contract["effective_batches"]["cofitok"]["effective_batch_size"] == 64


def test_quality_bridge_preparation_selects_100k_without_authorizing_300k() -> None:
    report = _preparation()

    assert report["status"] == "prepared"
    assert report["selection"]["steps"] == 100_000
    assert report["selection"]["base_channels"] == 128
    assert report["selection"]["milestone_steps"] == [50_000, 100_000]
    assert report["selection"]["milestone_equivalent_epochs"]["50000"] == pytest.approx(
        2.497722389792744
    )
    assert report["selection"]["equivalent_epochs"] == pytest.approx(
        4.995444779585488
    )
    assert report["source_quality_hold"]["source_equivalent_epochs"] == pytest.approx(
        24.96859419012024
    )
    assert report["evaluation_contract"]["terminal"]["samples_per_method"] == 10_000
    assert report["evaluation_contract"]["terminal"]["skip_precision_recall_allowed"] is False
    assert report["authorization_boundary"] == {
        "quality_bridge_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "report_is_promotion_gate": False,
        "new_gate_required": True,
        "explicit_execution_approval_required": True,
    }


def test_completed_quality_bridge_pair_is_valid_without_a_scaling_authorization() -> None:
    config_validation = validate_pair(
        load_config(COFITOK_CONFIG),
        load_config(DENSE_CONFIG),
        max_parameter_gap=0.02,
        stage="stability_quality_bridge",
    )

    report = validate_training_pair(
        _completed_bridge_training(
            COFITOK_CONFIG,
            parameters=config_validation["cofitok"]["parameter_count"],
        ),
        _completed_bridge_training(
            DENSE_CONFIG,
            parameters=config_validation["dense"]["parameter_count"],
        ),
        expected_steps=100_000,
        expected_revision="b" * 40,
        expected_branch="scale/generation-quality-bridge",
        expected_dataset="imagenet_256",
        expected_recipe_stage="stability_quality_bridge",
    )

    assert report["status"] == "pass"
    assert report["authorization_gate_identity_sha256"] is None
    assert report["cofitok"]["formal_full_training"] is False
    assert report["dense"]["formal_full_training"] is False


def test_quality_bridge_rejects_a_second_failed_scientific_gate() -> None:
    gate = _gate()
    gate["gates"].append({"name": "ordered_prefix_path_regression", "passed": False})
    cofitok = load_config(COFITOK_CONFIG)
    dense = load_config(DENSE_CONFIG)
    validation = validate_pair(
        cofitok,
        dense,
        max_parameter_gap=0.02,
        stage="stability_quality_bridge",
    )

    with pytest.raises(ValueError, match="only failed gate"):
        build_quality_bridge_preparation(
            promotion_gate=gate,
            promotion_gate_identity=_identity("/evidence/promotion_gate.json"),
            gate_source_verification={
                "status": "verified",
                "source_profile": "stability_scaling",
                "source_reports": gate["source_reports"],
            },
            source_cofitok_training=_source_training(
                SOURCE_COFITOK_CONFIG,
                parameters=validation["cofitok"]["parameter_count"],
            ),
            source_dense_training=_source_training(
                SOURCE_DENSE_CONFIG,
                parameters=validation["dense"]["parameter_count"],
            ),
            cofitok_config=config_to_dict(cofitok),
            dense_config=config_to_dict(dense),
            cofitok_config_identity=_identity(COFITOK_CONFIG.as_posix()),
            dense_config_identity=_identity(DENSE_CONFIG.as_posix()),
            config_validation=validation,
        )


def test_quality_bridge_rejects_a_capacity_change() -> None:
    report = _preparation()
    assert report["matched_training_contract"]["source_parameter_counts"]["cofitok"] > 0

    cofitok = load_config(COFITOK_CONFIG)
    dense = load_config(DENSE_CONFIG)
    validation = validate_pair(
        cofitok,
        dense,
        max_parameter_gap=0.02,
        stage="stability_quality_bridge",
    )
    gate = _gate()
    source = _source_training(
        SOURCE_COFITOK_CONFIG,
        parameters=validation["cofitok"]["parameter_count"] + 1,
    )
    with pytest.raises(ValueError, match="changed model capacity"):
        build_quality_bridge_preparation(
            promotion_gate=gate,
            promotion_gate_identity=_identity("/evidence/promotion_gate.json"),
            gate_source_verification={
                "status": "verified",
                "source_profile": "stability_scaling",
                "source_reports": gate["source_reports"],
            },
            source_cofitok_training=source,
            source_dense_training=_source_training(
                SOURCE_DENSE_CONFIG,
                parameters=validation["dense"]["parameter_count"],
            ),
            cofitok_config=config_to_dict(cofitok),
            dense_config=config_to_dict(dense),
            cofitok_config_identity=_identity(COFITOK_CONFIG.as_posix()),
            dense_config_identity=_identity(DENSE_CONFIG.as_posix()),
            config_validation=validation,
        )


def test_quality_bridge_prepare_runbook_never_launches_training() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_prepare.sh"
    ).read_text(encoding="utf-8")

    assert "--sources-only" in source
    assert "--stage stability_quality_bridge" in source
    assert "build_generation_quality_bridge_preparation.py" in source
    assert "quality bridge remains non-authorizing" in source
    assert "train_generation.py" not in source
    assert "full_matched_300k" not in source


EXPECTED_REVISION = "b" * 40
EXPECTED_BRANCH = "scale/generation-quality-bridge"
OUTPUT_ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/quality_bridge"
STORAGE_PATH = "/root/autodl-tmp/CoFiTok/checkpoints/generation"
RUN_DIRS = [f"{OUTPUT_ROOT}/cofitok", f"{OUTPUT_ROOT}/dense"]
BENCHMARK_ROOT = f"{OUTPUT_ROOT}/runtime_preflight/training"


def _execution_approval(preparation_identity: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": QUALITY_BRIDGE_APPROVAL_ROLE,
        "status": "approved",
        "scope": QUALITY_BRIDGE_APPROVAL_SCOPE,
        "preparation": preparation_identity,
        "git": {
            "revision": EXPECTED_REVISION,
            "branch": EXPECTED_BRANCH,
            "tracked_dirty": False,
        },
        "selection": {
            "dataset": "imagenet_256",
            "steps": 100_000,
            "milestone_steps": [50_000, 100_000],
            "effective_batch_size": 64,
        },
        "approval_record": {
            "approved_by": "user",
            "approved_at": "2026-08-05T00:00:00Z",
        },
        "output_root": OUTPUT_ROOT,
        "authorization_boundary": copy.deepcopy(EXECUTION_APPROVAL_BOUNDARY),
    }


def _launch_receipt() -> dict[str, object]:
    preparation = _preparation()
    preparation_identity = _identity("/evidence/preparation.json")
    approval = _execution_approval(preparation_identity)
    config_validation = preparation["matched_training_contract"]["config_validation"]
    source_identities = {
        "preparation": preparation_identity,
        "execution_approval": _identity("/evidence/execution_approval.json"),
        "cofitok_config": preparation["matched_training_contract"][
            "cofitok_config"
        ],
        "dense_config": preparation["matched_training_contract"]["dense_config"],
        "config_validation": _identity("/evidence/config_validation.json"),
        "storage_capacity": _identity("/evidence/storage_capacity.json"),
        "runtime_selection": _identity("/evidence/runtime_selection.json"),
    }
    expected_git = approval["git"]
    storage = {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": "stability_full_data_quality_bridge_100k_execution",
        "status": "pass",
        "git": expected_git,
        "filesystem": {"path": STORAGE_PATH, "free_bytes": 10**12},
        "plan": {
            "sample_count": 30_000,
            "checkpoint_count": 10,
            "required_free_bytes": 10**9,
        },
        "headroom_bytes": 10**11,
    }
    config_sha256 = {
        "cofitok": source_identities["cofitok_config"]["sha256"],
        "dense_identity": source_identities["dense_config"]["sha256"],
    }
    runtime = {
        "status": "selected",
        "git_revision": EXPECTED_REVISION,
        "config_sha256": config_sha256,
        "benchmark_root": BENCHMARK_ROOT,
        "runtime_environment_sha256": "c" * 64,
        "selected": {
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 4,
        },
        "selection_lock": {
            "training_run_dirs": RUN_DIRS,
            "training_target_steps": 100_000,
            "expected_effective_batch_size": 64,
            "git": expected_git,
            "config_sha256": config_sha256,
            "benchmark_root": BENCHMARK_ROOT,
        },
    }
    return build_quality_bridge_launch_receipt(
        preparation=preparation,
        execution_approval=approval,
        config_validation=config_validation,
        storage_capacity=storage,
        runtime_selection=runtime,
        source_identities=source_identities,
        expected_revision=EXPECTED_REVISION,
        expected_branch=EXPECTED_BRANCH,
        output_root=OUTPUT_ROOT,
        storage_path=STORAGE_PATH,
        training_run_dirs=RUN_DIRS,
        benchmark_root=BENCHMARK_ROOT,
        training_state_absent_at_launch=True,
    )


def _training_audit(run_dir: str) -> dict[str, object]:
    checkpoint = "checkpoint_step_00100000.pt"
    return {
        "schema_version": 2,
        "status": "complete",
        "issues": [],
        "warnings": [],
        "run_dir": run_dir,
        "expected_steps": 100_000,
        "last_step": 100_000,
        "metric_row_count": 100_000,
        "validation_event_count": 100,
        "training_report": {
            "path": f"{run_dir}/training_report.json",
            "status": "current",
            "completed_steps": 100_000,
            "training_complete": True,
        },
        "checkpoint": {
            "status": "available",
            "steps": [50_000, 100_000],
            "required_steps": [50_000, 100_000],
            "missing_required_steps": [],
            "latest": {"checkpoint": checkpoint, "step": 100_000},
            "latest_integrity": {
                "policy": "required",
                "status": "verified",
                "checkpoint": checkpoint,
                "step": 100_000,
                "integrity_manifest": f"{checkpoint}.integrity.json",
                "checkpoint_bytes": 123,
                "checkpoint_sha256": "d" * 64,
            },
        },
    }


def _terminal_generation(*, budget: int, prc: bool = True) -> dict[str, object]:
    checkpoint_sha = ("d" if budget == 8 else "e") * 64
    sample_sha = ("f" if budget == 8 else "0") * 64
    expected_git = {
        "revision": EXPECTED_REVISION,
        "branch": EXPECTED_BRANCH,
        "tracked_dirty": False,
    }
    sampling = {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_samples": 10_000,
        "start_index": 0,
        "batch_size": 32,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, 100),
        "prefix_budgets": [budget],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "image_shape": [3, 256, 256],
        "class_schedule": "balanced_modulo",
        "random_stream": {
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }
    checkpoint = f"/checkpoints/method_{budget}.pt"
    return {
        "schema_version": 3,
        "role": "generation_directory_metrics_report",
        "protocol": "torch_fidelity_directory_metrics",
        "status": "completed",
        "git": expected_git,
        "runtime_environment_sha256": "a" * 64,
        "counts": {"generated_image_count": 10_000},
        "parameters": {"precision_recall_enabled": prc},
        "real_set": {
            "root": "/datasets/imagenet_256/val",
            "sha256": "b" * 64,
            "image_count": 50_000,
        },
        "sample_provenance": {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_integrity_manifest": f"{checkpoint}.integrity.json",
            "checkpoint_step": 100_000,
            "weights": "ema",
            "selected_prefix_budget": budget,
            "sample_set_sha256": sample_sha,
            "git": expected_git,
            "runtime_environment_sha256": "a" * 64,
            "report_identity": _identity(f"/samples/{budget}/sampling_report.json"),
            "manifest_identity": _identity(
                f"/samples/{budget}/sampling_manifest.json"
            ),
            "sampling_progress": {
                "identity": _identity(
                    f"/samples/{budget}/sampling_progress.json"
                ),
                "status": "completed",
                "completed_samples": 10_000,
                "cumulative_elapsed_seconds": 100.0,
            },
            "sampling": sampling,
        },
        "metrics": {
            "frechet_inception_distance": 90.0,
            "inception_score_mean": 10.0,
            "precision": 0.6,
            "recall": 0.4,
        },
    }


def _terminal_checkpoint_eval(*, budget: int) -> dict[str, object]:
    checkpoint_sha = ("d" if budget == 8 else "e") * 64
    checkpoint = f"/checkpoints/method_{budget}.pt"
    random_orders = 4 if budget == 8 else 0
    metrics = {
        "evaluated_images": 256,
        "orders": {
            "ordered": {
                "endpoint_clean_mse": 0.1,
                "prefix_path_mse_auc": 0.2,
            }
        },
        "ordered_rank_by_path_auc": 1,
        "order_count": 6 if budget == 8 else 1,
        "zero_token_max_abs": 0.0,
        "shuffled_to_ordered_endpoint_ratio": 1.5,
    }
    if budget == 8:
        metrics["component_energy_ratio_per_sample_mean"] = [
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.44,
            0.44,
        ]
    return {
        "schema_version": 2,
        "role": "generation_checkpoint_evaluation_report",
        "status": "completed",
        "git": {
            "revision": EXPECTED_REVISION,
            "branch": EXPECTED_BRANCH,
            "tracked_dirty": False,
        },
        "request": {
            "num_images": 256,
            "timestep": 500,
            "random_orders": random_orders,
            "weights": "ema",
            "precision": "bf16",
        },
        "checkpoint_step": 100_000,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_integrity_manifest": f"{checkpoint}.integrity.json",
        "weights": "ema",
        "config": {
            "model": {
                "token_count": budget,
                "token_spatial_strides": (
                    [4, 4, 2, 2, 2, 2, 1, 1] if budget == 8 else [1]
                ),
            }
        },
        "metrics": metrics,
    }


def test_execution_approval_is_exactly_scoped_and_non_authorizing() -> None:
    preparation_identity = _identity("/evidence/preparation.json")
    approval = _execution_approval(preparation_identity)
    evidence = validate_quality_bridge_execution_approval(
        approval,
        preparation_identity=preparation_identity,
        expected_revision=EXPECTED_REVISION,
        expected_branch=EXPECTED_BRANCH,
        expected_output_root=OUTPUT_ROOT,
    )
    assert evidence["authorization_boundary"] == EXECUTION_APPROVAL_BOUNDARY
    assert evidence["authorization_boundary"]["full_300k_launch_allowed"] is False

    approval["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="boundary differs"):
        validate_quality_bridge_execution_approval(
            approval,
            preparation_identity=preparation_identity,
            expected_revision=EXPECTED_REVISION,
            expected_branch=EXPECTED_BRANCH,
            expected_output_root=OUTPUT_ROOT,
        )


def test_launch_receipt_is_strictly_replayable_and_non_authorizing() -> None:
    receipt = _launch_receipt()
    evidence = validate_quality_bridge_launch_receipt_contract(
        receipt,
        expected_revision=EXPECTED_REVISION,
        expected_branch=EXPECTED_BRANCH,
    )
    assert receipt["role"] == QUALITY_BRIDGE_LAUNCH_RECEIPT_ROLE
    assert receipt["stage"] == QUALITY_BRIDGE_RECIPE_STAGE
    assert receipt["authorization_boundary"] == LAUNCH_RECEIPT_AUTHORIZATION_BOUNDARY
    assert evidence["training_run_dirs"] == RUN_DIRS

    drifted = copy.deepcopy(receipt)
    drifted["source_reports"].pop("runtime_selection")
    with pytest.raises(ValueError, match="source set differs"):
        validate_quality_bridge_launch_receipt_contract(
            drifted,
            expected_revision=EXPECTED_REVISION,
            expected_branch=EXPECTED_BRANCH,
        )

    drifted = copy.deepcopy(receipt)
    drifted["authorization_boundary"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="authorization boundary differs"):
        validate_quality_bridge_launch_receipt_contract(
            drifted,
            expected_revision=EXPECTED_REVISION,
            expected_branch=EXPECTED_BRANCH,
        )


def test_training_audit_uses_real_checkpoint_and_integrity_status_fields() -> None:
    audit = _training_audit(RUN_DIRS[0])
    evidence = _validate_training_audit(
        audit,
        label="CoFiTok",
        expected_run_dir=RUN_DIRS[0],
    )
    assert evidence["latest_integrity"]["status"] == "verified"

    wrong_status = copy.deepcopy(audit)
    wrong_status["checkpoint"]["status"] = "verified"
    with pytest.raises(ValueError, match="training audit differs"):
        _validate_training_audit(
            wrong_status,
            label="CoFiTok",
            expected_run_dir=RUN_DIRS[0],
        )

    invalid_integrity = copy.deepcopy(audit)
    invalid_integrity["checkpoint"]["latest_integrity"]["status"] = "invalid"
    with pytest.raises(ValueError, match="training audit differs"):
        _validate_training_audit(
            invalid_integrity,
            label="CoFiTok",
            expected_run_dir=RUN_DIRS[0],
        )


def test_terminal_method_rejects_skip_prc_and_binds_sampling_receipts() -> None:
    expected_git = {
        "revision": EXPECTED_REVISION,
        "branch": EXPECTED_BRANCH,
        "tracked_dirty": False,
    }
    with pytest.raises(ValueError, match="terminal generation identity differs"):
        _terminal_method_row(
            _terminal_generation(budget=8, prc=False),
            _terminal_checkpoint_eval(budget=8),
            label="CoFiTok",
            expected_prefix_budget=8,
            expected_random_orders=4,
            expected_git=expected_git,
        )

    row = _terminal_method_row(
        _terminal_generation(budget=8),
        _terminal_checkpoint_eval(budget=8),
        label="CoFiTok",
        expected_prefix_budget=8,
        expected_random_orders=4,
        expected_git=expected_git,
    )
    assert row["sampling_report"]["path"].endswith("sampling_report.json")
    assert row["sampling_manifest"]["path"].endswith("sampling_manifest.json")
    assert row["sampling_progress"]["path"].endswith("sampling_progress.json")
    assert row["coarse_token_utilization"]["coarse_token_energy_ratio"] == pytest.approx(
        0.12
    )


def test_quality_screen_requires_distribution_mechanism_and_class_fidelity() -> None:
    source_hold = _preparation()["source_quality_hold"]
    cofitok = {
        "fid": 90.0,
        "precision": 0.62,
        "recall": 0.40,
        "endpoint_clean_mse": 0.10,
        "ordered_rank_by_path_auc": 1,
        "order_count": 6,
        "zero_token_max_abs": 0.0,
        "shuffled_to_ordered_endpoint_ratio": 1.5,
        "coarse_token_utilization": {"coarse_token_energy_ratio": 0.12},
    }
    dense = {
        "fid": 92.0,
        "precision": 0.64,
        "recall": 0.42,
        "endpoint_clean_mse": 0.10,
    }
    report = _quality_screen(
        cofitok=cofitok,
        dense=dense,
        class_fidelity={"valid": True, "status": "pass"},
        source_hold=source_hold,
    )
    assert report["status"] == "pass"
    assert report["failed_checks"] == []
    assert report["non_authorizing"] is True
    assert RESULT_AUTHORIZATION_BOUNDARY["full_300k_launch_allowed"] is False

    degraded = copy.deepcopy(cofitok)
    degraded.update(
        {
            "recall": 0.1,
            "ordered_rank_by_path_auc": 2,
            "zero_token_max_abs": 0.01,
        }
    )
    report = _quality_screen(
        cofitok=degraded,
        dense=dense,
        class_fidelity={"valid": False, "status": "hold"},
        source_hold=source_hold,
    )
    assert report["status"] == "hold"
    assert {
        "cofitok_recall_floor",
        "ordered_prefix_rank",
        "restricted_synthesis_zero_token",
        "class_fidelity",
    }.issubset(report["failed_checks"])


def test_physical_terminal_verifier_replays_samples_real_set_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_dir = tmp_path / "generated"
    real_dir = tmp_path / "real"
    generated_dir.mkdir()
    real_dir.mkdir()
    checkpoint = tmp_path / "checkpoint_step_00100000.pt"
    checkpoint.write_bytes(b"checkpoint")
    integrity_path = tmp_path / "checkpoint_step_00100000.pt.integrity.json"
    integrity_path.write_text("{}", encoding="utf-8")
    sampling_report = tmp_path / "sampling_report.json"
    sampling_report.write_text("{}", encoding="utf-8")
    generated_images = [generated_dir / "000000.png"]
    real_images = [real_dir / "000000.png"]
    provenance = {
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": "a" * 64,
        "checkpoint_integrity_manifest": integrity_path.resolve().as_posix(),
        "checkpoint_step": 100_000,
        "sample_set_sha256": "b" * 64,
        "report_identity": _identity(sampling_report.resolve().as_posix()),
        "manifest_identity": _identity((tmp_path / "sampling_manifest.json").as_posix()),
        "sampling_progress": {
            "identity": _identity((tmp_path / "sampling_progress.json").as_posix())
        },
    }
    report = {
        "paths": {
            "generated_dir": generated_dir.as_posix(),
            "real_dir": real_dir.as_posix(),
            "sampling_report": sampling_report.as_posix(),
        },
        "sample_provenance": provenance,
        "real_set": {
            "root": real_dir.resolve().as_posix(),
            "sha256": "c" * 64,
            "image_count": 1,
        },
    }
    monkeypatch.setattr(
        result_builder,
        "find_images",
        lambda path: generated_images if path == generated_dir.resolve() else real_images,
    )
    monkeypatch.setattr(
        result_builder,
        "validate_sampling_provenance",
        lambda report_path, directory, images: provenance,
    )
    monkeypatch.setattr(
        result_builder,
        "image_tree_sha256",
        lambda images, root: "c" * 64,
    )
    monkeypatch.setattr(
        result_builder,
        "verify_training_checkpoint",
        lambda path: {
            "checkpoint_bytes": len(b"checkpoint"),
            "checkpoint_sha256": "a" * 64,
            "step": 100_000,
        },
    )
    evidence = result_builder._verify_physical_generation_evidence(
        report,
        label="CoFiTok terminal",
    )
    assert evidence["sample_set_sha256"] == "b" * 64
    assert evidence["real_set"]["sha256"] == "c" * 64

    drifted = copy.deepcopy(provenance)
    drifted["sample_set_sha256"] = "d" * 64
    monkeypatch.setattr(
        result_builder,
        "validate_sampling_provenance",
        lambda report_path, directory, images: drifted,
    )
    with pytest.raises(ValueError, match="physical sampling provenance differs"):
        result_builder._verify_physical_generation_evidence(
            report,
            label="CoFiTok terminal",
        )


def _sampling_preflight(*, budget: int) -> dict[str, object]:
    checkpoint = f"/checkpoints/method_{budget}.pt"
    return {
        "schema_version": 1,
        "status": "passed",
        "git": {
            "revision": EXPECTED_REVISION,
            "branch": EXPECTED_BRANCH,
            "tracked_dirty": False,
        },
        "checkpoint_step": 100_000,
        "checkpoint_sha256": ("d" if budget == 8 else "e") * 64,
        "checkpoint_integrity_manifest": f"{checkpoint}.integrity.json",
        "weights": "ema",
        "requested_weights": "ema",
        "runtime_environment_sha256": "a" * 64,
        "request": {
            "batch_size": 32,
            "prefix_budget": budget,
            "precision": "bf16",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "image_shape": [3, 256, 256],
        },
        "result": {
            "output_finite": True,
            "output_shape": [32, 3, 256, 256],
            "mean_forward_seconds": 1.0,
            "cuda_memory_after_forward": {"max_allocated_bytes": 123},
        },
    }


def _physical_evidence_from_method(method: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint": {
            "path": method["checkpoint"],
            "bytes": 123,
            "sha256": method["checkpoint_sha256"],
        },
        "checkpoint_integrity_manifest": _identity(
            method["checkpoint_integrity_manifest"]
        ),
        "checkpoint_step": method["checkpoint_step"],
        "sampling_report": method["sampling_report"],
        "sampling_manifest": method["sampling_manifest"],
        "sampling_progress": method["sampling_progress"],
        "sample_set_sha256": method["sample_set_sha256"],
        "sample_count": method["sample_count"],
        "real_set": method["real_set"],
    }


def test_terminal_result_is_source_bound_complete_and_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _preparation()
    launch_receipt = _launch_receipt()
    cofitok_generation = _terminal_generation(budget=8)
    dense_generation = _terminal_generation(budget=1)
    dense_generation["metrics"]["frechet_inception_distance"] = 92.0
    dense_generation["metrics"]["precision"] = 0.62
    dense_generation["metrics"]["recall"] = 0.42
    cofitok_checkpoint = _terminal_checkpoint_eval(budget=8)
    dense_checkpoint = _terminal_checkpoint_eval(budget=1)
    expected_git = {
        "revision": EXPECTED_REVISION,
        "branch": EXPECTED_BRANCH,
        "tracked_dirty": False,
    }
    cofitok_method = _terminal_method_row(
        cofitok_generation,
        cofitok_checkpoint,
        label="CoFiTok",
        expected_prefix_budget=8,
        expected_random_orders=4,
        expected_git=expected_git,
    )
    dense_method = _terminal_method_row(
        dense_generation,
        dense_checkpoint,
        label="dense",
        expected_prefix_budget=1,
        expected_random_orders=0,
        expected_git=expected_git,
    )
    source_identities = {
        "preparation": launch_receipt["source_reports"]["preparation"],
        "launch_receipt": _identity("/evidence/launch_receipt.json"),
        "cofitok_training": _identity(f"{RUN_DIRS[0]}/training_report.json"),
        "dense_training": _identity(f"{RUN_DIRS[1]}/training_report.json"),
        "training_pair_validation": _identity(
            "/evidence/training_pair_validation.json"
        ),
        "cofitok_training_audit": _identity("/evidence/cofitok_audit.json"),
        "dense_training_audit": _identity("/evidence/dense_audit.json"),
        "milestone_50000": _identity("/evidence/milestone_50000.json"),
        "milestone_100000": _identity("/evidence/milestone_100000.json"),
        "cofitok_sampling_preflight": _identity(
            "/evidence/cofitok_sampling_preflight.json"
        ),
        "dense_sampling_preflight": _identity(
            "/evidence/dense_sampling_preflight.json"
        ),
        "cofitok_generation": _identity("/evidence/cofitok_generation.json"),
        "dense_generation": _identity("/evidence/dense_generation.json"),
        "cofitok_checkpoint_eval": _identity(
            "/evidence/cofitok_checkpoint_eval.json"
        ),
        "dense_checkpoint_eval": _identity(
            "/evidence/dense_checkpoint_eval.json"
        ),
        "class_fidelity_qualification": _identity(
            "/evidence/class_fidelity_qualification.json"
        ),
        "cofitok_class_fidelity": _identity(
            "/evidence/cofitok_class_fidelity.json"
        ),
        "dense_class_fidelity": _identity(
            "/evidence/dense_class_fidelity.json"
        ),
    }
    class_fidelity = {
        "valid": True,
        "status": "pass",
        "sources": {
            "cofitok": source_identities["cofitok_class_fidelity"],
            "dense_identity": source_identities["dense_class_fidelity"],
        },
        "sampling_contract": {
            "cofitok_checkpoint_sha256": cofitok_method["checkpoint_sha256"],
            "dense_checkpoint_sha256": dense_method["checkpoint_sha256"],
            "cofitok_sample_set_sha256": cofitok_method["sample_set_sha256"],
            "dense_sample_set_sha256": dense_method["sample_set_sha256"],
        },
    }
    monkeypatch.setattr(
        "cofitok.generation.quality_bridge.validate_class_fidelity_qualification",
        lambda *args, **kwargs: class_fidelity,
    )
    pair_validation = {
        "status": "pass",
        "expected_steps": 100_000,
        "expected_revision": EXPECTED_REVISION,
        "expected_branch": EXPECTED_BRANCH,
        "expected_dataset": "imagenet_256",
        "authorization_gate_identity_sha256": None,
        "training_recipe": {
            "stage": "stability_quality_bridge",
            "valid": True,
        },
        "cofitok": {"formal_full_training": False},
        "dense": {"formal_full_training": False},
    }
    report = build_quality_bridge_result(
        preparation=preparation,
        launch_receipt=launch_receipt,
        training_pair_validation=pair_validation,
        cofitok_training_audit=_training_audit(RUN_DIRS[0]),
        dense_training_audit=_training_audit(RUN_DIRS[1]),
        milestone_evidence={
            50_000: {"status": "verified"},
            100_000: {"status": "verified"},
        },
        cofitok_sampling_preflight=_sampling_preflight(budget=8),
        dense_sampling_preflight=_sampling_preflight(budget=1),
        cofitok_generation=cofitok_generation,
        dense_generation=dense_generation,
        cofitok_checkpoint_eval=cofitok_checkpoint,
        dense_checkpoint_eval=dense_checkpoint,
        class_fidelity_qualification={"status": "pass"},
        physical_evidence={
            "cofitok": _physical_evidence_from_method(cofitok_method),
            "dense_identity": _physical_evidence_from_method(dense_method),
        },
        source_identities=source_identities,
        expected_revision=EXPECTED_REVISION,
        expected_branch=EXPECTED_BRANCH,
    )
    assert report["status"] == "completed"
    assert report["quality_screen"]["status"] == "pass"
    assert report["authorization_boundary"] == RESULT_AUTHORIZATION_BOUNDARY
    assert report["authorization_boundary"]["full_training_launch_allowed"] is False
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False

    authorized_pair = copy.deepcopy(pair_validation)
    authorized_pair["authorization_gate_identity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="matched training validation differs"):
        build_quality_bridge_result(
            preparation=preparation,
            launch_receipt=launch_receipt,
            training_pair_validation=authorized_pair,
            cofitok_training_audit=_training_audit(RUN_DIRS[0]),
            dense_training_audit=_training_audit(RUN_DIRS[1]),
            milestone_evidence={
                50_000: {"status": "verified"},
                100_000: {"status": "verified"},
            },
            cofitok_sampling_preflight=_sampling_preflight(budget=8),
            dense_sampling_preflight=_sampling_preflight(budget=1),
            cofitok_generation=cofitok_generation,
            dense_generation=dense_generation,
            cofitok_checkpoint_eval=cofitok_checkpoint,
            dense_checkpoint_eval=dense_checkpoint,
            class_fidelity_qualification={"status": "pass"},
            physical_evidence={
                "cofitok": _physical_evidence_from_method(cofitok_method),
                "dense_identity": _physical_evidence_from_method(dense_method),
            },
            source_identities=source_identities,
            expected_revision=EXPECTED_REVISION,
            expected_branch=EXPECTED_BRANCH,
        )


def test_quality_bridge_execute_runbook_is_receipted_and_never_authorizes_300k() -> None:
    prepare = (
        ROOT
        / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_prepare.sh"
    ).read_text(encoding="utf-8")
    execute = (
        ROOT
        / "artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh"
    ).read_text(encoding="utf-8")
    expected_pythonpath = 'PYTHONPATH="$PROJECT:$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"'
    assert expected_pythonpath in prepare
    assert expected_pythonpath in execute
    assert "QUALITY_BRIDGE_EXECUTION_APPROVAL" in execute
    assert "validate_generation_quality_bridge_execution_approval.py" in execute
    assert "build_generation_quality_bridge_launch_receipt.py" in execute
    assert "validate_generation_quality_bridge_launch_receipt.py" in execute
    assert "build_generation_quality_bridge_result.py" in execute
    assert "verify_generation_quality_bridge_result.py" in execute
    assert "--required-checkpoint-steps 50000,100000" in execute
    assert "--skip-prc" not in execute
    assert "full_300k_launch_allowed\": False" in execute
    assert "full_matched_300k" not in execute
    assert "refusing an unreceipted or duplicate quality bridge pair monitor" in execute


def test_quality_bridge_direct_entrypoints_import_under_runbook_pythonpath() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    for script in QUALITY_BRIDGE_DIRECT_ENTRYPOINTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"
