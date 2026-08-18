from __future__ import annotations

import copy

import pytest

from scripts import (
    build_generation_conditioning_ranking_training_confirmation_posteval as target,
)
from scripts.build_generation_conditioning_ranking_probe_posteval import (
    SENSITIVITY_CLAIM_BOUNDARY,
)


EVALUATOR_GIT = {
    "revision": "a" * 40,
    "branch": "scale/generation-label-ranking-5k-heldout-evaluation-v1",
    "tracked_dirty": False,
}
PARAMETER_COUNTS = {
    "control_cofitok": 101,
    "ranked_cofitok": 101,
    "control_dense_identity": 99,
    "ranked_dense_identity": 99,
}
IDENTITY_CHARACTERS = {
    "control_cofitok": "1",
    "ranked_cofitok": "2",
    "control_dense_identity": "3",
    "ranked_dense_identity": "4",
}


def _identity(path: str, *, size: int, character: str) -> dict:
    return {"path": path, "bytes": size, "sha256": character * 64}


def _config_sha(run: str) -> str:
    return {run_name: f"{index + 5:x}" * 64 for index, run_name in enumerate(target.RUN_NAMES)}[run]


def _preparation() -> dict:
    configs = {
        run: _identity(
            f"/training-checkout/{target.CONFIG_RELATIVE_PATHS[run]}",
            size=1_000 + index,
            character=f"{index + 5:x}",
        )
        for index, run in enumerate(target.RUN_NAMES)
    }
    return {
        "schema_version": target.SCHEMA_VERSION,
        "role": target.PREPARATION_ROLE,
        "status": "pass",
        "valid": True,
        "stage": target.TRAINING_STAGE,
        "scope": target.TRAINING_SCOPE,
        "issues": [],
        "output_root": target.EXPECTED_OUTPUT_ROOT,
        "git": copy.deepcopy(target.TRAINING_GIT),
        "execution_boundary": copy.deepcopy(
            target.TRAINING_PREPARATION_EXECUTION_BOUNDARY
        ),
        "claim_boundary": copy.deepcopy(
            target.TRAINING_PREPARATION_CLAIM_BOUNDARY
        ),
        "control_ranking_config": copy.deepcopy(target.CONTROL_RANKING_CONFIG),
        "ranked_ranking_config": copy.deepcopy(target.RANKED_RANKING_CONFIG),
        "gpu_execution_authorized": False,
        "authorization_required": True,
        "parameter_counts": copy.deepcopy(PARAMETER_COUNTS),
        "configs": configs,
    }


def _expected_configs() -> dict[str, dict]:
    configs = {}
    for run in target.RUN_NAMES:
        ranked = run.startswith("ranked_")
        cofitok = "cofitok" in run
        ranking = (
            target.RANKED_RANKING_CONFIG
            if ranked
            else target.CONTROL_RANKING_CONFIG
        )
        configs[run] = {
            "name": run,
            "data": {
                "dataset": target.EXPECTED_DATASET,
                "root": "/root/autodl-tmp/CoFiTok/datasets",
                "batch_size": 64,
            },
            "model": {
                "predictor_type": "scalable_unet",
                "synthesis_mode": "fixed_basis" if cofitok else "dense_identity",
                "token_count": 8 if cofitok else 1,
                "num_classes": target.EXPECTED_NUM_CLASSES,
            },
            "loss": {
                "rollout_consistency_weight": 0.1,
                "rollout_consistency_start_step": 0,
                "rollout_consistency_warmup_steps": 1_000,
                "ema_teacher_consistency_weight": 0.25,
                "ema_teacher_consistency_start_step": 3_000,
                "ema_teacher_consistency_warmup_steps": 1_000,
                **copy.deepcopy(ranking),
            },
            "runtime": {
                "steps": target.EXPECTED_STEPS,
                "seed": 2027,
                "checkpoint_interval": target.EXPECTED_CHECKPOINT_INTERVAL,
                "evaluation_interval": target.EXPECTED_EVALUATION_INTERVAL,
            },
            "optimization": {"gradient_accumulation_steps": 1},
        }
    return configs


def _training_evidence() -> dict:
    preparation = _preparation()
    configs = _expected_configs()
    config_identities = {
        run: {
            "path": f"/evaluation-checkout/{target.CONFIG_RELATIVE_PATHS[run]}",
            "bytes": preparation["configs"][run]["bytes"],
            "sha256": preparation["configs"][run]["sha256"],
        }
        for run in target.RUN_NAMES
    }
    status_runs = {}
    reports = {}
    report_sources = {}
    audits = {}
    audit_sources = {}
    checkpoint_sources = {}
    integrity_sources = {}
    for index, run in enumerate(target.RUN_NAMES):
        character = IDENTITY_CHARACTERS[run]
        run_dir = f"{target.EXPECTED_OUTPUT_ROOT}/{run}"
        report_source = _identity(
            f"{run_dir}/training_report.json",
            size=10_000 + index,
            character=chr(ord("b") + index),
        )
        audit_source = _identity(
            f"{target.EXPECTED_OUTPUT_ROOT}/reports/{run}_training_audit.json",
            size=20_000 + index,
            character=("f", "a", "b", "c")[index],
        )
        checkpoint_source = _identity(
            f"{run_dir}/{target.EXPECTED_CHECKPOINT_FILENAME}",
            size=1_000_000 + index,
            character=character,
        )
        integrity_source = _identity(
            f"{run_dir}/{target.EXPECTED_CHECKPOINT_FILENAME}.integrity.json",
            size=500 + index,
            character=chr(ord("a") + index),
        )
        final_metrics = {
            "step": target.EXPECTED_STEPS,
            "samples_seen": 320_000,
            "epsilon": 0.03,
            "class_conditioning_ranking": 0.01 if run.startswith("ranked_") else 0.0,
            "class_conditioning_ranking_scale": (
                1.0 if run.startswith("ranked_") else 0.0
            ),
        }
        status_runs[run] = {
            "run_dir": run_dir,
            "training_report": copy.deepcopy(report_source),
            "training_audit": copy.deepcopy(audit_source),
            "checkpoint": copy.deepcopy(checkpoint_source),
            "integrity_manifest": copy.deepcopy(integrity_source),
            "final_metrics": copy.deepcopy(final_metrics),
        }
        reports[run] = {
            "training_complete": True,
            "completed_steps": target.EXPECTED_STEPS,
            "target_steps": target.EXPECTED_STEPS,
            "output_dir": run_dir,
            "config": copy.deepcopy(configs[run]),
            "git": {
                "revision": target.TRAINING_GIT["revision"],
                "branch": target.TRAINING_GIT["branch"],
                "dirty": False,
            },
            "parameter_count": PARAMETER_COUNTS[run],
            "latest_checkpoint": {
                "checkpoint": target.EXPECTED_CHECKPOINT_FILENAME,
                "step": target.EXPECTED_STEPS,
                "checkpoint_sha256": checkpoint_source["sha256"],
                "checkpoint_bytes": checkpoint_source["bytes"],
                "integrity_manifest": (
                    f"{target.EXPECTED_CHECKPOINT_FILENAME}.integrity.json"
                ),
                "dataset_identity_sha256": "d" * 64,
                "runtime_environment_sha256": "e" * 64,
            },
            "final_metrics": copy.deepcopy(final_metrics),
        }
        loss = configs[run]["loss"]
        audits[run] = {
            "schema_version": 2,
            "status": "complete",
            "issues": [],
            "expected_steps": target.EXPECTED_STEPS,
            "last_step": target.EXPECTED_STEPS,
            "progress_fraction": 1.0,
            "metric_row_count": 100,
            "run_dir": run_dir,
            "checkpoint": {
                "status": "available",
                "interval": target.EXPECTED_CHECKPOINT_INTERVAL,
                "required_steps": list(target.EXPECTED_CHECKPOINT_STEPS),
                "missing_required_steps": [],
                "latest": {
                    "step": target.EXPECTED_STEPS,
                    "checkpoint_sha256": checkpoint_source["sha256"],
                },
                "latest_integrity": {
                    "status": "verified",
                    "checkpoint_sha256": checkpoint_source["sha256"],
                },
            },
            "training_report": {
                "path": report_source["path"],
                "status": "current",
                "completed_steps": target.EXPECTED_STEPS,
                "training_complete": True,
            },
            "validation": {
                "configured_interval": target.EXPECTED_EVALUATION_INTERVAL,
                "event_count": target.EXPECTED_VALIDATION_EVENTS,
                "expected_event_count": target.EXPECTED_VALIDATION_EVENTS,
                "logging_complete": True,
                "provenance_metadata_status": "complete",
            },
            "consistency_schedules": {
                "status": "verified",
                "config_sha256": _config_sha(run),
                "schedules": {
                    name: {
                        "weight": loss[f"{name}_weight"],
                        "start_step": loss[f"{name}_start_step"],
                        "warmup_steps": loss[f"{name}_warmup_steps"],
                        "verified_rows": 100,
                    }
                    for name in (
                        "rollout_consistency",
                        "ema_teacher_consistency",
                        "class_conditioning_ranking",
                    )
                },
            },
        }
        report_sources[run] = report_source
        audit_sources[run] = audit_source
        checkpoint_sources[run] = checkpoint_source
        integrity_sources[run] = integrity_source
    status = {
        "schema_version": target.SCHEMA_VERSION,
        "role": target.TRAINING_STATUS_ROLE,
        "status": "completed",
        "stage": target.TRAINING_STAGE,
        "revision": target.TRAINING_GIT["revision"],
        "branch": target.TRAINING_GIT["branch"],
        "output_root": target.EXPECTED_OUTPUT_ROOT,
        "runs": status_runs,
        "execution_boundary": copy.deepcopy(target.TRAINING_STATUS_EXECUTION_BOUNDARY),
        "claim_boundary": copy.deepcopy(target.TRAINING_STATUS_CLAIM_BOUNDARY),
    }
    return {
        "preparation": preparation,
        "training_status": status,
        "training_reports": reports,
        "training_report_sources": report_sources,
        "training_audits": audits,
        "training_audit_sources": audit_sources,
        "checkpoint_sources": checkpoint_sources,
        "checkpoint_integrity_sources": integrity_sources,
        "expected_configs": configs,
        "config_identities": config_identities,
    }


def _sensitivity_report(
    run: str,
    evidence: dict,
    *,
    recovered: bool,
    correct_mse_ratio: float = 0.95,
) -> dict:
    ranked = run.startswith("ranked_")
    cofitok = "cofitok" in run
    rows = []
    for sample_index in range(target.EXPECTED_REQUEST["num_samples"]):
        for timestep in target.EXPECTED_REQUEST["timesteps"]:
            if ranked and recovered:
                correct_mse = correct_mse_ratio
                wrong_mse = correct_mse / 0.94
                null_mse = correct_mse / 0.95
            elif ranked:
                correct_mse = correct_mse_ratio
                wrong_mse = correct_mse / 1.02
                null_mse = correct_mse / 1.03
            else:
                correct_mse = 1.0
                wrong_mse = 0.99
                null_mse = 0.98
            rows.append(
                {
                    "sample_index": sample_index,
                    "timestep": timestep,
                    "correct_label": target.EXPECTED_REQUEST["start_label"] + sample_index,
                    "correct_wnid": f"n{192 + sample_index:08d}",
                    "wrong_label": (692 + sample_index) % 1_000,
                    "null_label": 1_000,
                    "noise_seed": target.EXPECTED_REQUEST["noise_seed"] + sample_index,
                    "forward_seconds": 0.1,
                    "conditions": {
                        "correct": {"epsilon_mse_to_noise": correct_mse},
                        "wrong": {"epsilon_mse_to_noise": wrong_mse},
                        "null": {"epsilon_mse_to_noise": null_mse},
                    },
                    "correct_relative_mse_improvement": {
                        "versus_wrong": (wrong_mse - correct_mse) / wrong_mse,
                        "versus_null": (null_mse - correct_mse) / null_mse,
                    },
                    "correct_better": {
                        "than_wrong": correct_mse < wrong_mse,
                        "than_null": correct_mse < null_mse,
                    },
                }
            )
    checkpoint = evidence["checkpoint_sources"][run]
    integrity = evidence["checkpoint_integrity_sources"][run]
    return {
        "git": copy.deepcopy(EVALUATOR_GIT),
        "request": copy.deepcopy(target.EXPECTED_REQUEST),
        "dataset": {
            "alias": target.EXPECTED_DATASET,
            "root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256_10pct",
            "label_map": {
                "path": "/dataset/metadata/label_to_wnid.json",
                "bytes": 100,
                "sha256": "9" * 64,
            },
            "samples": [
                {
                    "label": 192 + sample_index,
                    "wnid": f"n{192 + sample_index:08d}",
                    "image": {
                        "path": f"/dataset/{192 + sample_index}.png",
                        "bytes": 100 + sample_index,
                        "sha256": f"{sample_index + 1:064x}",
                    },
                }
                for sample_index in range(16)
            ],
        },
        "weights": "ema",
        "checkpoint": {
            **copy.deepcopy(checkpoint),
            "format_version": 1,
            "step": target.EXPECTED_STEPS,
            "integrity_manifest": copy.deepcopy(integrity),
            "git": {
                "revision": target.TRAINING_GIT["revision"],
                "branch": target.TRAINING_GIT["branch"],
                "dirty": False,
            },
            "dataset_identity_sha256": "d" * 64,
            "runtime_environment_sha256": "e" * 64,
        },
        "model": {
            "name": run,
            "predictor_type": "scalable_unet",
            "synthesis_mode": "fixed_basis" if cofitok else "dense_identity",
            "token_count": 8 if cofitok else 1,
            "num_classes": 1_000,
            "parameter_count": PARAMETER_COUNTS[run],
        },
        "sample_rows": rows,
        "runtime": {"device": "cpu", "threads": 2, "elapsed_seconds": 1.0},
        "claim_boundary": copy.deepcopy(SENSITIVITY_CLAIM_BOUNDARY),
    }


def _build_inputs(
    *,
    cofitok_recovered: bool = True,
    dense_recovered: bool = True,
) -> dict:
    evidence = _training_evidence()
    evidence["sensitivity_reports"] = {
        "control_cofitok": _sensitivity_report(
            "control_cofitok", evidence, recovered=False, correct_mse_ratio=1.0
        ),
        "ranked_cofitok": _sensitivity_report(
            "ranked_cofitok", evidence, recovered=cofitok_recovered
        ),
        "control_dense_identity": _sensitivity_report(
            "control_dense_identity", evidence, recovered=False, correct_mse_ratio=1.0
        ),
        "ranked_dense_identity": _sensitivity_report(
            "ranked_dense_identity", evidence, recovered=dense_recovered
        ),
    }
    evidence["sources"] = {"synthetic": True}
    evidence["evaluator_git"] = copy.deepcopy(EVALUATOR_GIT)
    return evidence


def test_exact_5k_heldout_source_pass_is_non_authorizing() -> None:
    report = target.build_heldout_evaluation(**_build_inputs())

    assert report["role"] == target.ROLE
    assert report["training_git"] == target.TRAINING_GIT
    assert report["evaluator_git"] == EVALUATOR_GIT
    assert report["decision"]["shared_semantic_alignment_recovery_supported"]
    assert report["decision"]["recommended_next_action"] == (
        "prepare_separately_source_bound_posttraining_5k_sampling_confirmation"
    )
    assert report["evaluation_contract"]["eligible_timesteps"] == [500, 900]
    assert report["evaluation_contract"]["descriptive_only_timesteps"] == [100]
    assert report["evaluation_contract"]["sample_count"] == 16
    assert report["evaluation_contract"]["independent_from_probe1k"] == {
        "probe1k_start_label": 64,
        "probe1k_noise_seed": 204060,
        "train5k_start_label": 192,
        "train5k_noise_seed": 304060,
    }
    for method in ("cofitok", "dense_identity"):
        assert report["methods"][method]["pass"] is True
        assert report["methods"][method]["sample_count"] == 16
        assert report["methods"][method]["ranked_minus_control"]["versus_wrong"][
            "one_sided_sign_test_pvalue"
        ] == pytest.approx(1 / 65_536)
    assert all(
        value is False
        for key, value in report["claim_boundary"].items()
        if key.startswith("authorizes_")
    )
    assert report["claim_boundary"]["diagnostic_only"] is True


def test_heldout_decision_rejects_method_asymmetry_and_shared_failure() -> None:
    asymmetric = target.build_heldout_evaluation(
        **_build_inputs(cofitok_recovered=True, dense_recovered=False)
    )
    assert asymmetric["methods"]["cofitok"]["pass"] is True
    assert asymmetric["methods"]["dense_identity"]["pass"] is False
    assert asymmetric["decision"]["recommended_next_action"] == (
        "reject_shared_repair_due_method_asymmetry"
    )

    failed = target.build_heldout_evaluation(
        **_build_inputs(cofitok_recovered=False, dense_recovered=False)
    )
    assert failed["decision"]["recommended_next_action"] == (
        "revise_training_time_semantic_alignment_objective"
    )


def test_heldout_rejects_request_checkpoint_and_integrity_drift() -> None:
    request_drift = _build_inputs()
    request_drift["sensitivity_reports"]["ranked_cofitok"]["request"][
        "noise_seed"
    ] += 1
    with pytest.raises(ValueError, match="evaluator contract"):
        target.build_heldout_evaluation(**request_drift)

    checkpoint_drift = _build_inputs()
    checkpoint_drift["sensitivity_reports"]["ranked_dense_identity"][
        "checkpoint"
    ]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint differs from training"):
        target.build_heldout_evaluation(**checkpoint_drift)

    integrity_drift = _build_inputs()
    integrity_drift["sensitivity_reports"]["ranked_dense_identity"][
        "checkpoint"
    ]["integrity_manifest"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint differs from training"):
        target.build_heldout_evaluation(**integrity_drift)


def test_heldout_rejects_config_audit_and_physical_source_drift() -> None:
    config_drift = _build_inputs()
    config_drift["training_reports"]["ranked_cofitok"]["config"]["name"] = (
        "drifted"
    )
    with pytest.raises(ValueError, match="training report"):
        target.build_heldout_evaluation(**config_drift)

    audit_drift = _build_inputs()
    audit_drift["training_audits"]["ranked_cofitok"]["consistency_schedules"][
        "schedules"
    ]["class_conditioning_ranking"]["weight"] = 0.04
    with pytest.raises(ValueError, match="schedule audit"):
        target.build_heldout_evaluation(**audit_drift)

    physical_drift = _build_inputs()
    physical_drift["checkpoint_sources"]["control_cofitok"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity differs from training status"):
        target.build_heldout_evaluation(**physical_drift)


def test_heldout_requires_a_separate_clean_evaluator_revision() -> None:
    same_revision = _build_inputs()
    same_revision["evaluator_git"]["revision"] = target.TRAINING_GIT["revision"]
    for report in same_revision["sensitivity_reports"].values():
        report["git"] = copy.deepcopy(same_revision["evaluator_git"])
    with pytest.raises(ValueError, match="separate, exact, and clean"):
        target.build_heldout_evaluation(**same_revision)

    dirty = _build_inputs()
    dirty["evaluator_git"]["tracked_dirty"] = True
    for report in dirty["sensitivity_reports"].values():
        report["git"] = copy.deepcopy(dirty["evaluator_git"])
    with pytest.raises(ValueError, match="separate, exact, and clean"):
        target.build_heldout_evaluation(**dirty)


def test_heldout_rejects_nonindependent_labels_noise_and_preparation_recipe() -> None:
    label_drift = _build_inputs()
    for report in label_drift["sensitivity_reports"].values():
        report["request"]["start_label"] = 64
    with pytest.raises(ValueError, match="evaluator contract"):
        target.build_heldout_evaluation(**label_drift)

    preparation_drift = _build_inputs()
    preparation_drift["preparation"]["ranked_ranking_config"][
        "class_conditioning_ranking_weight"
    ] = 0.04
    with pytest.raises(ValueError, match="preparation is malformed"):
        target.build_heldout_evaluation(**preparation_drift)
