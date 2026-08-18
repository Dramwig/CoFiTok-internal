from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.build_generation_conditioning_ranking_probe_posteval import (
    EXPECTED_EVALUATOR_BRANCH,
    EXPECTED_CHECKPOINT_STEPS,
    EXPECTED_OUTPUT_ROOT,
    EXPECTED_REQUEST,
    EXPECTED_TRAINING_BRANCH,
    EXPECTED_TRAINING_REVISION,
    REPORT_ROLE,
    RUN_NAMES,
    SENSITIVITY_CLAIM_BOUNDARY,
    build_postevaluation,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_REVISION = EXPECTED_TRAINING_REVISION
EVALUATOR_GIT = {
    "revision": "b" * 40,
    "branch": EXPECTED_EVALUATOR_BRANCH,
    "tracked_dirty": False,
}
OUTPUT_ROOT = EXPECTED_OUTPUT_ROOT
PARAMETER_COUNTS = {
    "control_cofitok": 101,
    "ranked_cofitok": 101,
    "control_dense": 99,
    "ranked_dense": 99,
}
PREPARATION_KEYS = {
    "control_cofitok": "control_cofitok",
    "ranked_cofitok": "ranked_cofitok",
    "control_dense_identity": "control_dense",
    "ranked_dense_identity": "ranked_dense",
}


def _checkpoint_sha(run: str) -> str:
    character = {
        "control_cofitok": "1",
        "ranked_cofitok": "2",
        "control_dense_identity": "3",
        "ranked_dense_identity": "4",
    }[run]
    return character * 64


def _preparation() -> dict:
    return {
        "schema_version": 1,
        "role": "generation_conditioning_ranking_four_arm_probe_preparation",
        "scope": "imagenet256_10pct_four_arm_class_ranking_probe1k_only",
        "status": "pass",
        "valid": True,
        "issues": [],
        "git": {
            "revision": TRAINING_REVISION,
            "branch": EXPECTED_TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "output_root": OUTPUT_ROOT,
        "gpu_execution_authorized": False,
        "authorization_required": True,
        "control_ranking_config": {
            "class_conditioning_ranking_weight": 0.0,
            "class_conditioning_ranking_start_step": 0,
            "class_conditioning_ranking_warmup_steps": 0,
            "class_conditioning_ranking_batch_fraction": 0.0625,
            "class_conditioning_ranking_margin": 0.0,
            "class_conditioning_ranking_wrong_label_offset": 1,
            "class_conditioning_ranking_min_timestep": 0,
        },
        "ranked_ranking_config": {
            "class_conditioning_ranking_weight": 0.05,
            "class_conditioning_ranking_start_step": 100,
            "class_conditioning_ranking_warmup_steps": 200,
            "class_conditioning_ranking_batch_fraction": 0.0625,
            "class_conditioning_ranking_margin": 0.01,
            "class_conditioning_ranking_wrong_label_offset": 500,
            "class_conditioning_ranking_min_timestep": 500,
        },
        "configs": {
            key: {"bytes": 10, "sha256": "c" * 64}
            for key in PARAMETER_COUNTS
        },
        "parameter_counts": dict(PARAMETER_COUNTS),
    }


def _expected_configs() -> dict[str, dict]:
    configs = {}
    for run in RUN_NAMES:
        ranked = run.startswith("ranked_")
        cofitok = "cofitok" in run
        ranking = (
            _preparation()["ranked_ranking_config"]
            if ranked
            else _preparation()["control_ranking_config"]
        )
        configs[run] = {
            "name": run,
            "data": {
                "dataset": "imagenet_256_10pct",
                "root": "/root/autodl-tmp/CoFiTok/datasets",
                "batch_size": 16,
            },
            "model": {
                "predictor_type": "scalable_unet",
                "synthesis_mode": "fixed_basis" if cofitok else "dense_identity",
                "token_count": 8 if cofitok else 1,
                "num_classes": 1_000,
            },
            "loss": {
                "rollout_consistency_weight": 0.1,
                "rollout_consistency_start_step": 0,
                "rollout_consistency_warmup_steps": 200,
                "ema_teacher_consistency_weight": 0.25,
                "ema_teacher_consistency_start_step": 600,
                "ema_teacher_consistency_warmup_steps": 300,
                **ranking,
            },
            "runtime": {
                "steps": 1_000,
                "seed": 2027,
                "checkpoint_interval": 250,
                "evaluation_interval": 250,
            },
            "optimization": {
                "gradient_accumulation_steps": 4,
            },
        }
    return configs


def _training_evidence() -> tuple[dict, dict[str, dict], dict[str, dict]]:
    expected_configs = _expected_configs()
    status_runs = {}
    reports = {}
    audits = {}
    for run in RUN_NAMES:
        run_dir = f"{OUTPUT_ROOT}/{run}"
        sha = _checkpoint_sha(run)
        ranking = expected_configs[run]["loss"]
        status_runs[run] = {
            "training_report": f"{run_dir}/training_report.json",
            "checkpoint_sha256": sha,
            "final_metrics": {
                "epsilon": 0.03,
                "class_conditioning_ranking": 0.01 if run.startswith("ranked_") else 0.0,
                "class_conditioning_ranking_scale": (
                    1.0 if run.startswith("ranked_") else 0.0
                ),
                "samples_seen": 64_000,
            },
        }
        reports[run] = {
            "training_complete": True,
            "completed_steps": 1_000,
            "target_steps": 1_000,
            "output_dir": run_dir,
            "config": expected_configs[run],
            "git": {
                "revision": TRAINING_REVISION,
                "branch": EXPECTED_TRAINING_BRANCH,
                "dirty": False,
            },
            "parameter_count": PARAMETER_COUNTS[PREPARATION_KEYS[run]],
            "latest_checkpoint": {
                "checkpoint": "checkpoint_step_00001000.pt",
                "step": 1_000,
                "checkpoint_sha256": sha,
                "checkpoint_bytes": 1_234,
                "integrity_manifest": "checkpoint_step_00001000.pt.integrity.json",
                "dataset_identity_sha256": "d" * 64,
                "runtime_environment_sha256": "e" * 64,
            },
            "final_metrics": copy.deepcopy(status_runs[run]["final_metrics"]),
        }
        audits[run] = {
            "schema_version": 2,
            "status": "complete",
            "issues": [],
            "expected_steps": 1_000,
            "last_step": 1_000,
            "progress_fraction": 1.0,
            "metric_row_count": 40,
            "run_dir": run_dir,
            "checkpoint": {
                "interval": 250,
                "status": "available",
                "steps": [500, 750, 1_000],
                "required_steps": list(EXPECTED_CHECKPOINT_STEPS),
                "missing_required_steps": [],
                "latest": {
                    "step": 1_000,
                    "checkpoint_sha256": sha,
                },
                "latest_integrity": {
                    "status": "verified",
                    "checkpoint_sha256": sha,
                },
            },
            "training_report": {
                "path": f"{run_dir}/training_report.json",
                "status": "current",
                "completed_steps": 1_000,
                "training_complete": True,
            },
            "validation": {
                "configured_interval": 250,
                "event_count": 4,
                "expected_event_count": 4,
                "logging_complete": True,
                "provenance_metadata_status": "complete",
            },
            "consistency_schedules": {
                "status": "verified",
                "config_sha256": "c" * 64,
                "schedules": {
                    "rollout_consistency": {
                        "weight": 0.1,
                        "start_step": 0,
                        "warmup_steps": 200,
                        "verified_rows": 40,
                    },
                    "ema_teacher_consistency": {
                        "weight": 0.25,
                        "start_step": 600,
                        "warmup_steps": 300,
                        "verified_rows": 40,
                    },
                    "class_conditioning_ranking": {
                        "weight": ranking["class_conditioning_ranking_weight"],
                        "start_step": ranking[
                            "class_conditioning_ranking_start_step"
                        ],
                        "warmup_steps": ranking[
                            "class_conditioning_ranking_warmup_steps"
                        ],
                        "verified_rows": 40,
                    },
                },
            },
        }
    status = {
        "schema_version": 1,
        "role": "generation_conditioning_ranking_four_arm_probe_training_status",
        "status": "completed",
        "revision": TRAINING_REVISION,
        "runs": status_runs,
        "claim_boundary": {
            "training_quality_claim_allowed": False,
            "sample_quality_claim_allowed": False,
            "promotion_authorization_allowed": False,
            "followup_training_allowed": False,
            "full_training_launch_allowed": False,
            "required_next_evidence": "held-out matched conditioning sensitivity evaluation",
        },
    }
    return status, reports, audits


def _sensitivity_report(
    run: str,
    *,
    recovered: bool,
    correct_mse_ratio: float = 0.95,
) -> dict:
    ranked = run.startswith("ranked_")
    cofitok = "cofitok" in run
    rows = []
    for sample_index in range(8):
        for timestep in EXPECTED_REQUEST["timesteps"]:
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
                    "correct_label": 64 + sample_index,
                    "correct_wnid": f"n{64 + sample_index:08d}",
                    "wrong_label": 564 + sample_index,
                    "null_label": 1_000,
                    "noise_seed": 204060 + sample_index,
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
    return {
        "git": dict(EVALUATOR_GIT),
        "request": copy.deepcopy(EXPECTED_REQUEST),
        "dataset": {
            "alias": "imagenet_256_10pct",
            "root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256_10pct",
            "label_map": {
                "path": "/dataset/metadata/label_to_wnid.json",
                "bytes": 100,
                "sha256": "a" * 64,
            },
            "samples": [
                {
                    "label": 64 + sample_index,
                    "wnid": f"n{64 + sample_index:08d}",
                    "image": {
                        "path": f"/dataset/{sample_index}.png",
                        "bytes": 100 + sample_index,
                        "sha256": f"{sample_index + 1:x}" * 64,
                    },
                }
                for sample_index in range(8)
            ],
        },
        "weights": "ema",
        "checkpoint": {
            "path": f"{OUTPUT_ROOT}/{run}/checkpoint_step_00001000.pt",
            "bytes": 1_234,
            "sha256": _checkpoint_sha(run),
            "format_version": 1,
            "step": 1_000,
            "integrity_manifest": {
                "path": f"{OUTPUT_ROOT}/{run}/checkpoint_step_00001000.pt.integrity.json",
                "bytes": 100,
                "sha256": "f" * 64,
            },
            "git": {
                "revision": TRAINING_REVISION,
                "branch": EXPECTED_TRAINING_BRANCH,
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
            "parameter_count": PARAMETER_COUNTS[PREPARATION_KEYS[run]],
        },
        "sample_rows": rows,
        "runtime": {
            "device": "cpu",
            "threads": 2,
            "elapsed_seconds": 1.0,
        },
        "claim_boundary": copy.deepcopy(SENSITIVITY_CLAIM_BOUNDARY),
    }


def _build_inputs(
    *,
    cofitok_recovered: bool = True,
    dense_recovered: bool = True,
) -> dict:
    status, reports, audits = _training_evidence()
    sensitivities = {
        "control_cofitok": _sensitivity_report(
            "control_cofitok",
            recovered=False,
            correct_mse_ratio=1.0,
        ),
        "ranked_cofitok": _sensitivity_report(
            "ranked_cofitok",
            recovered=cofitok_recovered,
        ),
        "control_dense_identity": _sensitivity_report(
            "control_dense_identity",
            recovered=False,
            correct_mse_ratio=1.0,
        ),
        "ranked_dense_identity": _sensitivity_report(
            "ranked_dense_identity",
            recovered=dense_recovered,
        ),
    }
    return {
        "preparation": _preparation(),
        "training_status": status,
        "training_reports": reports,
        "training_audits": audits,
        "expected_configs": _expected_configs(),
        "sensitivity_reports": sensitivities,
        "sources": {"synthetic": True},
        "git": dict(EVALUATOR_GIT),
    }


def _set_raw_improvement(row: dict, *, versus_wrong: float, versus_null: float) -> None:
    correct_mse = float(row["conditions"]["correct"]["epsilon_mse_to_noise"])
    wrong_mse = correct_mse / (1.0 - versus_wrong)
    null_mse = correct_mse / (1.0 - versus_null)
    row["conditions"]["wrong"]["epsilon_mse_to_noise"] = wrong_mse
    row["conditions"]["null"]["epsilon_mse_to_noise"] = null_mse
    row["correct_relative_mse_improvement"] = {
        "versus_wrong": versus_wrong,
        "versus_null": versus_null,
    }
    row["correct_better"] = {
        "than_wrong": correct_mse < wrong_mse,
        "than_null": correct_mse < null_mse,
    }


def test_postevaluation_requires_shared_absolute_and_control_relative_recovery() -> None:
    report = build_postevaluation(**_build_inputs())

    assert report["role"] == REPORT_ROLE
    assert report["decision"]["shared_semantic_alignment_recovery_supported"] is True
    assert report["decision"]["cofitok_specific_advantage_claim_allowed"] is False
    assert report["decision"]["recommended_next_action"] == (
        "consider_separately_authorized_matched_sampling_validation"
    )
    for method in ("cofitok", "dense_identity"):
        result = report["methods"][method]
        assert result["pass"] is True
        assert result["sample_count"] == 8
        assert result["eligible_row_count"] == 16
        assert result["independent_unit"] == "held_out_validation_image"
        assert result["ranking_metrics_recomputed_from_condition_mse"] is True
        assert result["ranked_absolute"]["versus_null"]["positive_count"] == 8
        assert result["ranked_minus_control"]["versus_wrong"][
            "one_sided_sign_test_pvalue"
        ] == pytest.approx(1 / 256)
    assert report["evaluation_contract"]["eligible_timesteps"] == [500, 900]
    assert report["evaluation_contract"]["descriptive_only_timesteps"] == [100]
    assert report["claim_boundary"]["authorizes_sampling"] is False


def test_postevaluation_rejects_method_asymmetry_as_a_shared_repair() -> None:
    report = build_postevaluation(
        **_build_inputs(cofitok_recovered=True, dense_recovered=False)
    )

    assert report["methods"]["cofitok"]["pass"] is True
    assert report["methods"]["dense_identity"]["pass"] is False
    assert report["decision"]["shared_semantic_alignment_recovery_supported"] is False
    assert report["decision"]["recommended_next_action"] == (
        "reject_shared_repair_due_method_asymmetry"
    )


def test_postevaluation_rejects_correct_mse_regression_even_with_better_ranking() -> None:
    inputs = _build_inputs()
    inputs["sensitivity_reports"]["ranked_cofitok"] = _sensitivity_report(
        "ranked_cofitok",
        recovered=True,
        correct_mse_ratio=1.03,
    )

    report = build_postevaluation(**inputs)

    assert report["methods"]["cofitok"]["gates"][
        "correct_mse_ratio_within_limit"
    ] is False
    assert report["methods"]["cofitok"]["pass"] is False


def test_postevaluation_rejects_request_or_checkpoint_drift() -> None:
    request_drift = _build_inputs()
    request_drift["sensitivity_reports"]["ranked_cofitok"]["request"][
        "noise_seed"
    ] += 1
    with pytest.raises(ValueError, match="evaluator contract"):
        build_postevaluation(**request_drift)

    checkpoint_drift = _build_inputs()
    checkpoint_drift["sensitivity_reports"]["ranked_dense_identity"][
        "checkpoint"
    ]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint differs from training"):
        build_postevaluation(**checkpoint_drift)


def test_postevaluation_rejects_incomplete_training_audit() -> None:
    inputs = _build_inputs()
    inputs["training_audits"]["ranked_cofitok"]["checkpoint"][
        "missing_required_steps"
    ] = [750]

    with pytest.raises(ValueError, match="training audit"):
        build_postevaluation(**inputs)


def test_postevaluation_rejects_preparation_config_and_schedule_audit_drift() -> None:
    preparation_drift = _build_inputs()
    preparation_drift["preparation"]["git"]["revision"] = "f" * 40
    with pytest.raises(ValueError, match="preparation Git"):
        build_postevaluation(**preparation_drift)

    config_drift = _build_inputs()
    config_drift["training_reports"]["ranked_dense_identity"]["config"]["name"] = (
        "drifted"
    )
    with pytest.raises(ValueError, match="probe contract"):
        build_postevaluation(**config_drift)

    audit_drift = _build_inputs()
    audit_drift["training_audits"]["ranked_cofitok"]["consistency_schedules"][
        "config_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="training audit"):
        build_postevaluation(**audit_drift)


def test_postevaluation_rejects_nonfinite_raw_mse_and_derived_field_tampering() -> None:
    nonfinite = _build_inputs()
    nonfinite["sensitivity_reports"]["ranked_cofitok"]["sample_rows"][0][
        "conditions"
    ]["wrong"]["epsilon_mse_to_noise"] = float("nan")
    with pytest.raises(ValueError, match="epsilon MSE is invalid"):
        build_postevaluation(**nonfinite)

    derived_drift = _build_inputs()
    derived_drift["sensitivity_reports"]["ranked_cofitok"]["sample_rows"][0][
        "correct_relative_mse_improvement"
    ]["versus_wrong"] = 0.99
    with pytest.raises(ValueError, match="derived ranking field differs from raw MSE"):
        build_postevaluation(**derived_drift)


def test_postevaluation_rejects_cross_method_source_row_identity_drift() -> None:
    inputs = _build_inputs()
    for run in ("control_dense_identity", "ranked_dense_identity"):
        inputs["sensitivity_reports"][run]["sample_rows"][0]["noise_seed"] += 1

    with pytest.raises(ValueError, match="source-row identities differ"):
        build_postevaluation(**inputs)

    metadata_drift = _build_inputs()
    for run in ("control_dense_identity", "ranked_dense_identity"):
        metadata_drift["sensitivity_reports"][run]["sample_rows"][0][
            "correct_wnid"
        ] = "n00000000"
    with pytest.raises(ValueError, match="source-row metadata differs"):
        build_postevaluation(**metadata_drift)


def test_postevaluation_aggregates_timesteps_before_the_sign_test() -> None:
    inputs = _build_inputs()
    rows = inputs["sensitivity_reports"]["ranked_cofitok"]["sample_rows"]
    for row in rows:
        if row["timestep"] == 100:
            continue
        if row["sample_index"] < 6:
            improvement = 0.02
        elif row["timestep"] == 500:
            improvement = 0.01
        else:
            improvement = -0.10
        _set_raw_improvement(
            row,
            versus_wrong=improvement,
            versus_null=improvement,
        )

    report = build_postevaluation(**inputs)
    cofitok = report["methods"]["cofitok"]

    assert cofitok["ranked_absolute"]["versus_wrong"]["positive_count"] == 6
    assert cofitok["ranked_absolute"]["versus_wrong"][
        "one_sided_sign_test_pvalue"
    ] == pytest.approx(37 / 256)
    assert cofitok["gates"]["ranked_absolute_sample_significant"] is False
    assert cofitok["pass"] is False
    assert report["evaluation_contract"]["timestep_rows_are_not_independent_units"]


def test_postevaluation_runbook_is_cpu_only_and_non_authorizing() -> None:
    runbook = (
        ROOT
        / "artifacts"
        / "runbooks"
        / "generation_conditioning_ranking_four_arm_posteval_v1.sh"
    ).read_text(encoding="utf-8")

    assert '[[ -z "$(git status --porcelain)" ]]' in runbook
    assert "export CUDA_VISIBLE_DEVICES=-1" in runbook
    assert "pgrep -af 'scripts/train_generation.py'" in runbook
    assert "generate_samples.py" not in runbook
    assert "train_generation.py --config" not in runbook
    assert "EXECUTION_APPROVAL" not in runbook
    assert runbook.count("--resume") >= 2
    assert runbook.count('sha256sum "$TRAINING_STATUS"') >= 2
    assert "chmod 0444" in runbook
