from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_EXACT_TEXT,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
)
from cofitok.generation.factorization_quality_regression import (
    CLAIM_BOUNDARY,
    CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION,
    DIAGNOSTIC_REPORT_ROLE,
    EXECUTION_BOUNDARY,
    FOLLOWUP_DECISION_BUILDER_GIT,
    FOLLOWUP_DECISION_CATEGORY,
    FOLLOWUP_DECISION_ID,
    OUTPUT_ROOT,
    PREPARATION_BOUNDARY,
    PROTOCOL,
    QUALITY_BRIDGE_GIT,
    QUALITY_BRIDGE_RESULT_ROLE,
    RUN_DIRS,
    SEEDS,
    SOURCE_BINDING_BOUNDARY,
    build_diagnostic_report,
    build_execution_authorization,
    build_preparation,
    build_source_binding,
    classify_followup_decision,
    validate_execution_authorization,
)
from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    EXPECTED_CHECKS,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
)
from cofitok.generation.stability_qualification import build_stability_qualification
from cofitok.inference_replay import file_identity
from scripts import (
    build_generation_factorization_quality_regression_source_binding as source_binding_builder,
)


REVISION = "a" * 40
BRANCH = "scale/generation-factorization-quality-regression-v1"


def _identity(path: str, character: str = "b") -> dict[str, object]:
    return {"path": path, "bytes": 123, "sha256": character * 64}


def _git() -> dict[str, object]:
    return {"revision": REVISION, "branch": BRANCH, "tracked_dirty": False}


def _standing() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_EXACT_TEXT,
            "received_at": "2026-08-13T00:11:00+08:00",
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _training_exposure_context(
    source: dict[str, object] | None = None,
) -> dict[str, object]:
    train_image_count = 1_281_167
    full_epochs = 6_400_000 / train_image_count
    reference_epochs = 3_200_000 / 128_161
    return {
        "source_report": copy.deepcopy(
            source or _identity("/evidence/training_exposure.json", "8")
        ),
        "dataset": "imagenet_256",
        "dataset_identity_sha256": "7" * 64,
        "train_image_count": train_image_count,
        "steps_per_method": 100_000,
        "effective_batch_size": 64,
        "images_seen_per_method": 6_400_000,
        "full_data_equivalent_epochs": full_epochs,
        "historical_10pct_reference_equivalent_epochs": reference_epochs,
        "full_to_reference_equivalent_epoch_ratio": full_epochs / reference_epochs,
        "below_historical_reference_exposure": True,
        "insufficient_exposure_is_live_hypothesis": True,
        "causal_status": "not_identified_by_exposure_alone",
        "quality_metric_comparison_allowed": False,
        "terminal_result_content_bound": True,
        "terminal_checkpoint_binding_verified": True,
    }


def _decision(
    *,
    route: str = FOLLOWUP_DECISION_ID,
    exposure_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    ordered_checks = sorted(EXPECTED_CHECKS)
    failed = ["matched_fid_tolerance"]
    exposure_source = exposure_identity or _identity(
        "/evidence/training_exposure.json", "8"
    )
    recommendation = {
        "id": route,
        "category": (
            FOLLOWUP_DECISION_CATEGORY if route == FOLLOWUP_DECISION_ID else "other"
        ),
        "execution_ready": False,
        "gpu_execution_allowed": False,
        "full_300k_launch_allowed": False,
        "release_authorization_allowed": False,
        "trigger": {"failed_checks": failed},
    }
    return {
        "schema_version": FOLLOWUP_DECISION_SCHEMA_VERSION,
        "role": "stability_quality_bridge_followup_experiment_decision",
        "status": "completed",
        "decision_builder_git": copy.deepcopy(FOLLOWUP_DECISION_BUILDER_GIT),
        "quality_bridge_execution_git": copy.deepcopy(QUALITY_BRIDGE_GIT),
        "authorization_boundary": copy.deepcopy(FOLLOWUP_AUTHORIZATION_BOUNDARY),
        "recommended_next_stage": recommendation,
        "terminal_quality": {
            "status": "hold",
            "checks": [
                {"name": name, "passed": name not in failed} for name in ordered_checks
            ],
            "failed_checks": failed,
        },
        "source_reports": {
            "quality_bridge_result": _identity("/evidence/result.json"),
            "milestones": {
                "50000": _identity("/evidence/milestone_50000.json", "5"),
                "100000": _identity("/evidence/milestone_100000.json", "6"),
            },
            "terminal_training_exposure": copy.deepcopy(exposure_source),
        },
        "training_exposure": _training_exposure_context(exposure_source),
    }


def _terminal_guard() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "generation_terminal_system_claim_guard",
        "status": "hold",
        "sources": {"quality_bridge_result": _identity("/evidence/result.json")},
        "evidence": {
            "quality_screen": {"failed_checks": ["matched_fid_tolerance"]}
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "larger_training_launch_allowed": False,
            "release_authorization_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
        },
        "claim_boundary": {
            "training_launch_allowed": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def _config(*, cofitok: bool) -> dict[str, object]:
    return {
        "name": "cofitok" if cofitok else "dense",
        "data": {"dataset": "imagenet_256"},
        "diffusion": {"num_train_timesteps": 1000},
        "runtime": {"steps": 100_000},
        "optimization": {"learning_rate": 1e-4},
        "model": {
            "token_count": 8 if cofitok else 1,
            "predictor_use_feedback": cofitok,
            "synthesis_mode": "fixed_basis" if cofitok else "dense_identity",
        },
        "loss": {"epsilon_weight": 1.0},
    }


def _checkpoint_reference(*, cofitok: bool) -> dict[str, object]:
    method = "cofitok" if cofitok else "dense_identity"
    character = "c" if cofitok else "d"
    return {
        "checkpoint": f"{RUN_DIRS[method]}/checkpoint_step_00100000.pt",
        "checkpoint_bytes": 1_000_000_000,
        "checkpoint_sha256": character * 64,
        "checkpoint_step": 100_000,
        "integrity_manifest": _identity(f"/{method}/integrity.json", character),
    }


def _training(*, cofitok: bool) -> dict[str, object]:
    reference = _checkpoint_reference(cofitok=cofitok)
    method = "cofitok" if cofitok else "dense_identity"
    return {
        "training_complete": True,
        "completed_steps": 100_000,
        "target_steps": 100_000,
        "git": {
            "revision": QUALITY_BRIDGE_GIT["revision"],
            "branch": QUALITY_BRIDGE_GIT["branch"],
            "dirty": False,
        },
        "config": _config(cofitok=cofitok),
        "output_dir": RUN_DIRS[method],
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00100000.pt",
            "checkpoint_bytes": reference["checkpoint_bytes"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "integrity_manifest": "checkpoint_step_00100000.pt.integrity.json",
            "step": 100_000,
        },
        "final_metrics": {"validation_epsilon_mse": 1.01 if cofitok else 1.0},
    }


def _checkpoint_eval(
    *,
    cofitok: bool,
    evaluation_git: dict[str, object] | None = None,
) -> dict[str, object]:
    reference = _checkpoint_reference(cofitok=cofitok)
    return {
        "schema_version": CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION,
        "role": "generation_checkpoint_evaluation_report",
        "status": "completed",
        "weights": "ema",
        "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "checkpoint_step": 100_000,
        "git": copy.deepcopy(evaluation_git or QUALITY_BRIDGE_GIT),
        "config": _config(cofitok=cofitok),
        "metrics": {
            "orders": {"ordered": {"endpoint_clean_mse": 1.02 if cofitok else 1.0}},
            "component_energy_ratio": (
                [0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]
                if cofitok
                else [1.0]
            ),
            "evaluated_images": 256,
            "ordered_rank_by_path_auc": 1,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 10.0,
            "timestep": 500,
        },
    }


def _source_fixture(
    *,
    physical_exposure_identity: dict[str, object] | None = None,
    decision_exposure_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    preparation = build_preparation(execution_git=_git())
    preparation_identity = _identity("/control/preparation.json", "1")
    decision = _decision(exposure_identity=decision_exposure_identity)
    terminal = _terminal_guard()
    cofitok_training = _training(cofitok=True)
    dense_training = _training(cofitok=False)
    cofitok_eval = _checkpoint_eval(cofitok=True)
    dense_eval = _checkpoint_eval(cofitok=False)
    identities = {
        "quality_result": _identity("/evidence/result.json"),
        "followup_decision": _identity("/evidence/decision.json", "2"),
        "terminal_training_exposure": physical_exposure_identity
        or _identity("/evidence/training_exposure.json", "8"),
        "terminal_system_guard": _identity("/evidence/terminal.json", "3"),
        "cofitok_training": _identity("/evidence/cofitok_training.json", "4"),
        "dense_training": _identity("/evidence/dense_training.json", "5"),
        "cofitok_checkpoint_evaluation": _identity("/evidence/cofitok_eval.json", "6"),
        "dense_checkpoint_evaluation": _identity("/evidence/dense_eval.json", "7"),
        "cofitok_checkpoint_integrity": _identity("/cofitok/integrity.json", "c"),
        "dense_checkpoint_integrity": _identity("/dense_identity/integrity.json", "d"),
    }
    quality_result = {
        "schema_version": 1,
        "role": QUALITY_BRIDGE_RESULT_ROLE,
        "status": "completed",
        "source_reports": {
            "cofitok_training": identities["cofitok_training"],
            "dense_training": identities["dense_training"],
            "cofitok_checkpoint_eval": identities["cofitok_checkpoint_evaluation"],
            "dense_checkpoint_eval": identities["dense_checkpoint_evaluation"],
        },
    }
    binding = build_source_binding(
        preparation=preparation,
        preparation_identity=preparation_identity,
        followup_decision=decision,
        followup_decision_identity=identities["followup_decision"],
        terminal_system_guard=terminal,
        terminal_system_guard_identity=identities["terminal_system_guard"],
        quality_result=quality_result,
        quality_result_identity=identities["quality_result"],
        training_reports={
            "cofitok": cofitok_training,
            "dense_identity": dense_training,
        },
        checkpoint_evaluations={
            "cofitok": cofitok_eval,
            "dense_identity": dense_eval,
        },
        checkpoint_references={
            "cofitok": _checkpoint_reference(cofitok=True),
            "dense_identity": _checkpoint_reference(cofitok=False),
        },
        source_identities=identities,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    return {
        "preparation": preparation,
        "preparation_identity": preparation_identity,
        "binding": binding,
        "binding_identity": _identity("/control/source_binding.json", "8"),
        "training": {
            "cofitok": cofitok_training,
            "dense_identity": dense_training,
        },
    }


def _rollout(*, cofitok: bool, seed: int, high_frequency: float = 0.3) -> dict:
    reference = _checkpoint_reference(cofitok=cofitok)
    protocol = {
        "num_images": PROTOCOL["num_images_per_seed"],
        "batch_size": PROTOCOL["batch_size"],
        "teacher_timesteps": PROTOCOL["teacher_timesteps"],
        "sample_steps": PROTOCOL["sample_steps"],
        "guidance_scale": PROTOCOL["guidance_scale"],
        "teacher_guidance_scale": PROTOCOL["teacher_guidance_scale"],
        "guidance_rescale": PROTOCOL["guidance_rescale"],
        "cfg_batch_mode": PROTOCOL["cfg_batch_mode"],
        "clip_x0": PROTOCOL["clip_x0"],
        "precision": PROTOCOL["precision"],
        "seed": seed,
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "git": _git(),
        "weights": "ema",
        "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "checkpoint_step": 100_000,
        "config": _config(cofitok=cofitok),
        "protocol": protocol,
        "teacher_forced": [
            {
                "timestep": timestep,
                "epsilon_mse": 1.01 if cofitok else 1.0,
                "clipped_x0_mse": 1.02 if cofitok else 1.0,
            }
            for timestep in PROTOCOL["teacher_timesteps"]
        ],
        "free_sampling_rollout": {
            "steps": [
                {
                    "timestep": timestep,
                    "predicted_x0_high_frequency_ratio": high_frequency,
                }
                for timestep in (595, 394, 192, 91)
            ]
        },
        "reconstruction_rollout": {
            "summary": {
                "final_clipped_x0_mse": 1.03 if cofitok else 1.0,
                "final_to_best_x0_mse_amplification": 1.1,
            }
        },
        "runtime": {
            "elapsed_seconds": 12.0,
            "cuda_peak_memory_bytes": 1024,
            "device": "cuda",
        },
    }


def _diagnostic_inputs(*, failing_seeds: set[int] | None = None) -> dict[str, object]:
    source = _source_fixture()
    authorization = build_execution_authorization(
        preparation=source["preparation"],
        preparation_identity=source["preparation_identity"],
        source_binding=source["binding"],
        source_binding_identity=source["binding_identity"],
        standing_authorization=_standing(),
        standing_authorization_identity=_identity("/control/standing.json", "9"),
        execution_git=_git(),
        expected_revision=REVISION,
        expected_branch=BRANCH,
    )
    diagnostic_evals = {
        "cofitok": _checkpoint_eval(cofitok=True, evaluation_git=_git()),
        "dense_identity": _checkpoint_eval(cofitok=False, evaluation_git=_git()),
    }
    identities: dict[str, dict[str, object]] = {
        "cofitok_training": _identity("/evidence/cofitok_training.json", "4"),
        "dense_training": _identity("/evidence/dense_training.json", "5"),
        "cofitok_diagnostic_checkpoint_evaluation": _identity(
            "/diagnostic/cofitok_eval.json", "a"
        ),
        "dense_diagnostic_checkpoint_evaluation": _identity(
            "/diagnostic/dense_eval.json", "b"
        ),
    }
    rollouts: dict[int, dict[str, dict]] = {}
    qualifications: dict[int, dict] = {}
    for index, seed in enumerate(SEEDS):
        high_frequency = 0.5 if failing_seeds and seed in failing_seeds else 0.3
        cofitok = _rollout(cofitok=True, seed=seed, high_frequency=high_frequency)
        dense = _rollout(cofitok=False, seed=seed, high_frequency=0.25)
        rollouts[seed] = {"cofitok": cofitok, "dense_identity": dense}
        identities[f"cofitok_rollout_seed_{seed}"] = _identity(
            f"/diagnostic/cofitok_{seed}.json", str(index + 1)
        )
        identities[f"dense_rollout_seed_{seed}"] = _identity(
            f"/diagnostic/dense_{seed}.json", str(index + 3)
        )
        identities[f"qualification_seed_{seed}"] = _identity(
            f"/diagnostic/qualification_{seed}.json", str(index + 5)
        )
        qualification = build_stability_qualification(
            cofitok_training=source["training"]["cofitok"],
            dense_training=source["training"]["dense_identity"],
            cofitok_checkpoint=diagnostic_evals["cofitok"],
            dense_checkpoint=diagnostic_evals["dense_identity"],
            cofitok_rollout=cofitok,
            dense_rollout=dense,
            expected_weights="ema",
            expected_evaluation_revision=REVISION,
            expected_evaluation_branch=BRANCH,
        )
        qualification["sources"] = {
            "cofitok_training": identities["cofitok_training"],
            "dense_training": identities["dense_training"],
            "cofitok_checkpoint": identities[
                "cofitok_diagnostic_checkpoint_evaluation"
            ],
            "dense_checkpoint": identities[
                "dense_diagnostic_checkpoint_evaluation"
            ],
            "cofitok_rollout": identities[f"cofitok_rollout_seed_{seed}"],
            "dense_rollout": identities[f"dense_rollout_seed_{seed}"],
        }
        qualifications[seed] = qualification
    return {
        "preparation": source["preparation"],
        "preparation_identity": source["preparation_identity"],
        "source_binding": source["binding"],
        "source_binding_identity": source["binding_identity"],
        "execution_authorization": authorization,
        "execution_authorization_identity": _identity("/control/auth.json", "f"),
        "training_reports": source["training"],
        "checkpoint_evaluations": diagnostic_evals,
        "rollout_reports": rollouts,
        "qualification_reports": qualifications,
        "source_identities": identities,
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
    }


def test_preparation_is_permanently_non_authorizing() -> None:
    report = build_preparation(execution_git=_git())

    assert report["status"] == "prepared"
    assert report["protocol"] == PROTOCOL
    assert report["output_root"] == OUTPUT_ROOT
    assert report["authorization_boundary"] == PREPARATION_BOUNDARY
    assert report["authorization_boundary"]["gpu_execution_authorized"] is False


def test_followup_route_requires_exact_matched_failure() -> None:
    assert classify_followup_decision(_decision()) == "selected"
    assert classify_followup_decision(_decision(route="another_route")) == "not_selected"

    drifted = _decision()
    drifted["recommended_next_stage"]["trigger"]["failed_checks"] = []
    with pytest.raises(ValueError, match="trigger differs"):
        classify_followup_decision(drifted)

    exposure_drift = _decision()
    exposure_drift["training_exposure"]["images_seen_per_method"] = 6_399_999
    with pytest.raises(ValueError, match="training exposure differs"):
        classify_followup_decision(exposure_drift)

    exposure_source_drift = _decision()
    exposure_source_drift["training_exposure"]["source_report"]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="training exposure source differs"):
        classify_followup_decision(exposure_source_drift)

    boundary_drift = _decision()
    boundary_drift["authorization_boundary"]["recommended_stage_execution_allowed"] = True
    with pytest.raises(ValueError, match="decision is malformed"):
        classify_followup_decision(boundary_drift)


def test_source_binding_rejects_terminal_training_exposure_identity_drift() -> None:
    with pytest.raises(
        ValueError,
        match="terminal_training_exposure source identity differs",
    ):
        _source_fixture(
            physical_exposure_identity=_identity(
                "/evidence/training_exposure.json", "9"
            )
        )


def test_source_binding_builder_passes_terminal_exposure_to_v2_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text(
        json.dumps(build_preparation(execution_git=_git())),
        encoding="utf-8",
    )
    result_path = tmp_path / "quality_result.json"
    exposure_path = tmp_path / "training_exposure.json"
    decision = _decision(
        exposure_identity={
            "path": exposure_path.as_posix(),
            "bytes": 321,
            "sha256": "8" * 64,
        }
    )
    decision["source_reports"]["quality_bridge_result"] = {
        "path": result_path.as_posix(),
        "bytes": 123,
        "sha256": "b" * 64,
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    captured: dict[str, object] = {}

    def stop_after_capture(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        raise RuntimeError("captured v2 replay arguments")

    monkeypatch.setattr(
        source_binding_builder.followup_builder,
        "build_from_sources",
        stop_after_capture,
    )
    with pytest.raises(RuntimeError, match="captured v2 replay arguments"):
        source_binding_builder.build_from_paths(
            preparation_path=preparation_path,
            expected_preparation_sha256=file_identity(preparation_path)["sha256"],
            followup_decision_path=decision_path,
            terminal_system_guard_path=tmp_path / "terminal.json",
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )

    assert captured["quality_bridge_result_path"] == result_path
    assert captured["expected_quality_bridge_result_sha256"] == "b" * 64
    assert captured["training_exposure_report_path"] == exposure_path
    assert captured["expected_training_exposure_report_sha256"] == "8" * 64


def test_source_binding_and_standing_authorization_do_not_authorize_training() -> None:
    source = _source_fixture()
    assert source["binding"]["authorization_boundary"] == SOURCE_BINDING_BOUNDARY
    assert source["binding"]["selected_route"]["id"] == FOLLOWUP_DECISION_ID

    kwargs = {
        "preparation": source["preparation"],
        "preparation_identity": source["preparation_identity"],
        "source_binding": source["binding"],
        "source_binding_identity": source["binding_identity"],
        "standing_authorization": _standing(),
        "standing_authorization_identity": _identity("/control/standing.json", "9"),
        "execution_git": _git(),
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
    }
    authorization = build_execution_authorization(**kwargs)

    assert authorization["authorization_boundary"] == EXECUTION_BOUNDARY
    assert authorization["authorization_boundary"]["gpu_execution_allowed"] is True
    assert authorization["authorization_boundary"]["training_launch_allowed"] is False
    assert validate_execution_authorization(authorization, **kwargs) == authorization

    drifted = copy.deepcopy(authorization)
    drifted["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="authorization differs"):
        validate_execution_authorization(drifted, **kwargs)


def test_diagnostic_report_cannot_make_causal_or_quality_claim() -> None:
    report = build_diagnostic_report(**_diagnostic_inputs())

    assert report["role"] == DIAGNOSTIC_REPORT_ROLE
    assert report["decision"]["id"] == (
        "factorization_rollout_regression_not_reproduced"
    )
    assert report["decision"]["next_stage_execution_allowed"] is False
    assert report["decision"]["causal_factorization_attribution"] is False
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["claim_boundary"]["quality_advantage_claim_allowed"] is False


def test_diagnostic_distinguishes_consistent_and_seed_sensitive_instability() -> None:
    seed_sensitive = build_diagnostic_report(
        **_diagnostic_inputs(failing_seeds={SEEDS[0]})
    )
    consistent = build_diagnostic_report(
        **_diagnostic_inputs(failing_seeds=set(SEEDS))
    )

    assert seed_sensitive["decision"]["id"] == (
        "seed_sensitive_factorization_rollout_regression_detected"
    )
    assert "predicted_x0_high_frequency" in seed_sensitive["decision"][
        "failed_gate_union"
    ]
    assert seed_sensitive["decision"]["failed_gate_intersection"] == []
    assert consistent["decision"]["id"] == (
        "consistent_factorization_rollout_regression_detected"
    )
    assert "predicted_x0_high_frequency" in consistent["decision"][
        "failed_gate_intersection"
    ]


def test_diagnostic_rejects_protocol_drift() -> None:
    inputs = _diagnostic_inputs()
    inputs["rollout_reports"][SEEDS[0]]["cofitok"]["protocol"][
        "guidance_scale"
    ] = 2.0

    with pytest.raises(ValueError, match="rollout report differs"):
        build_diagnostic_report(**inputs)
