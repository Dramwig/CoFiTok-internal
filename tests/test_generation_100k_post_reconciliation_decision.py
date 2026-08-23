from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cofitok.generation.post_reconciliation_decision import (
    AUTHORIZATION_BOUNDARY,
    EXPECTED_CHECKS,
    EXPECTED_FAILED_CHECKS,
    POST_RECONCILIATION_DECISION_ROLE,
    TRAINING_BRANCH,
    TRAINING_REVISION,
    TRAINING_TREE,
    build_post_reconciliation_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _identity(name: str, digest: str) -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": digest * 64,
    }


def _check(name: str, passed: bool) -> dict:
    return {
        "name": name,
        "passed": passed,
        "observed": 0.0 if passed else 1.0,
        "comparison": "test",
        "threshold": 0.0,
    }


def _inputs() -> dict:
    identities = {
        "decision": _identity("decision", "a"),
        "reconciliation": _identity("reconciliation", "b"),
        "quality": _identity("quality", "c"),
        "exposure": _identity("exposure", "d"),
        "cofitok_generation": _identity("cofitok_generation", "e"),
        "dense_generation": _identity("dense_generation", "f"),
        "cofitok_class": _identity("cofitok_class", "1"),
        "dense_class": _identity("dense_class", "2"),
    }
    checks = [
        _check(name, name not in EXPECTED_FAILED_CHECKS)
        for name in (
            "cofitok_absolute_fid",
            "matched_fid_tolerance",
            "cofitok_precision_floor",
            "cofitok_recall_floor",
            "matched_precision_tolerance",
            "matched_recall_tolerance",
            "matched_endpoint_tolerance",
            "ordered_prefix_rank",
            "coarse_token_utilization",
            "restricted_synthesis_zero_token",
            "shuffle_mismatch",
            "class_fidelity",
        )
    ]
    assert {row["name"] for row in checks} == EXPECTED_CHECKS
    class_metrics = {
        "cofitok": {
            "sample_count": 10_000,
            "top1_accuracy": 0.0024,
            "top5_accuracy": 0.0106,
        },
        "dense_identity": {
            "sample_count": 10_000,
            "top1_accuracy": 0.0017,
            "top5_accuracy": 0.0081,
        },
    }
    methods = {
        "cofitok": {
            "fid": 115.26217262363417,
            "precision": 0.7555999755859375,
            "recall": 0.008320000022649765,
            "checkpoint_sha256": "3" * 64,
            "sample_set_sha256": "4" * 64,
        },
        "dense_identity": {
            "fid": 123.02103114594166,
            "precision": 0.6653000116348267,
            "recall": 0.009999999776482582,
            "checkpoint_sha256": "5" * 64,
            "sample_set_sha256": "6" * 64,
        },
    }
    decision = {
        "schema_version": 2,
        "status": "completed",
        "role": "stability_quality_bridge_followup_experiment_decision",
        "recommended_next_stage": {
            "id": "reconcile_100k_cross_protocol_evidence",
            "category": "evidence_conflict",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
        "authorization_boundary": {
            "recommended_stage_execution_allowed": False,
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_authorization_allowed": False,
        },
    }
    reconciliation = {
        "schema_version": 1,
        "role": "generation_100k_cross_protocol_reconciliation",
        "status": "completed",
        "operational_status": "pass",
        "terminal_status": "hold",
        "source_evidence": {
            "decision": identities["decision"],
            "quality_result": identities["quality"],
            "terminal_training_exposure": identities["exposure"],
        },
        "comparison": {
            "protocol_rows": {
                "ddim50_2048": {"lower_fid_method": "dense_identity"},
                "ddim100_2048": {"lower_fid_method": "cofitok"},
                "ddim100_10000": {"lower_fid_method": "cofitok"},
            },
            "per_method_effects": {
                "cofitok": {
                    "sampler_step_effect_at_2048": {
                        "fid_delta_ddim100_minus_ddim50": -91.3256
                    }
                },
                "dense_identity": {
                    "sampler_step_effect_at_2048": {
                        "fid_delta_ddim100_minus_ddim50": 2.7287
                    }
                },
            },
            "ranking_reversal_explanation": (
                "sampler_step_effect_dominates_observed_ranking_reversal"
            ),
            "sampler_step_changes_matched_ranking": True,
            "sample_count_changes_matched_ranking": False,
            "terminal_protocol_controls_quality_status": True,
            "terminal_quality_status": "hold",
            "generation_advantage_proven": False,
        },
        "claim_boundary": {
            "generation_advantage_proven": False,
            "quality_or_generation_advantage_claim_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "gpu_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_authorization_allowed": False,
            "export_authorization_allowed": False,
            "release_authorization_allowed": False,
            "process_signals_allowed": False,
        },
    }
    quality = {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_full_data_quality_bridge_result",
        "stage": "stability_quality_bridge",
        "git": {
            "revision": TRAINING_REVISION,
            "branch": TRAINING_BRANCH,
            "tracked_dirty": False,
        },
        "authorization_boundary": {
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_authorization_allowed": False,
        },
        "source_reports": {
            "cofitok_generation": identities["cofitok_generation"],
            "dense_generation": identities["dense_generation"],
            "cofitok_class_fidelity": identities["cofitok_class"],
            "dense_class_fidelity": identities["dense_class"],
        },
        "quality_screen": {
            "status": "hold",
            "non_authorizing": True,
            "failed_checks": list(EXPECTED_FAILED_CHECKS),
            "checks": checks,
            "thresholds": {
                "max_absolute_fid": 100.0,
                "min_recall": 0.3,
            },
        },
        "terminal": {
            "methods": methods,
            "class_fidelity": {
                "status": "hold",
                "valid": False,
                "metrics": class_metrics,
                "sources": {
                    "cofitok": identities["cofitok_class"],
                    "dense_identity": identities["dense_class"],
                },
            },
        },
    }
    exposure_rows = {
        method: {
            "status": "complete",
            "training_complete": True,
            "dataset": "imagenet_256",
            "target_steps": 100_000,
            "completed_steps": 100_000,
            "effective_batch_size": 64,
            "samples_seen": 6_400_000,
            "train_image_count": 1_281_167,
            "dataset_identity_sha256": "7" * 64,
            "completed_equivalent_epochs": 6_400_000 / 1_281_167,
            "git": {
                "revision": TRAINING_REVISION,
                "branch": TRAINING_BRANCH,
                "dirty": False,
            },
        }
        for method in ("cofitok", "dense_identity")
    }
    exposure = {
        "schema_version": 1,
        "exposure_schema_version": 1,
        "role": "generation_training_exposure_audit",
        "status": "pass",
        "rows": exposure_rows,
        "comparison": {
            "same_dataset": True,
            "same_dataset_identity": True,
            "same_effective_batch_size": True,
            "same_completed_steps": True,
            "same_images_seen": True,
            "same_equivalent_epochs": True,
            "same_dataset_normalized_exposure": True,
        },
        "terminal_binding": {
            "terminal_result_binding_verified": True,
            "active_runbook_verification_completed": True,
            "terminal_result": identities["quality"],
        },
    }
    sampling_common = {
        "sampler": "ddim",
        "sample_steps": 100,
        "num_samples": 10_000,
        "start_index": 0,
        "seed": 0,
        "class_schedule": "balanced_modulo",
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "clip_x0": True,
        "eta": 0.0,
        "precision": "bf16",
    }
    terminal_evidence = {}
    for method, prefix, checkpoint_sha, sample_sha, generation_identity, class_identity in (
        (
            "cofitok",
            8,
            "3" * 64,
            "4" * 64,
            identities["cofitok_generation"],
            identities["cofitok_class"],
        ),
        (
            "dense_identity",
            1,
            "5" * 64,
            "6" * 64,
            identities["dense_generation"],
            identities["dense_class"],
        ),
    ):
        terminal_evidence[method] = {
            "generation_report": generation_identity,
            "class_fidelity_report": class_identity,
            "checkpoint_payload": _identity(f"{method}_checkpoint", "8"),
            "checkpoint_sidecar": _identity(f"{method}_sidecar", "9"),
            "latest": _identity(f"{method}_latest", "0"),
            "physical_sha256_verified": True,
            "sidecar_latest_reconciled": True,
            "checkpoint_step": 100_000,
            "checkpoint_sha256": checkpoint_sha,
            "sample_set_sha256": sample_sha,
            "prefix_budget": prefix,
            "generation_metrics": {
                "fid": methods[method]["fid"],
                "precision": methods[method]["precision"],
                "recall": methods[method]["recall"],
                "inception_score_mean": 9.0,
            },
            "class_fidelity_metrics": class_metrics[method],
            "sampling": {
                **sampling_common,
                "prefix_budgets": [prefix],
            },
        }
    return {
        "authoritative_decision": decision,
        "authoritative_decision_identity": identities["decision"],
        "reconciliation": reconciliation,
        "reconciliation_identity": identities["reconciliation"],
        "quality_result": quality,
        "quality_result_identity": identities["quality"],
        "training_exposure": exposure,
        "training_exposure_identity": identities["exposure"],
        "terminal_evidence": terminal_evidence,
        "builder_git": {
            "revision": "a" * 40,
            "tree": "b" * 40,
            "branch": "analysis/test-post-reconciliation",
            "tracked_dirty": False,
            "path": "/checkout",
        },
    }


def test_post_reconciliation_decision_selects_bounded_sampling_discriminator() -> None:
    report = build_post_reconciliation_decision(**_inputs())

    assert report["role"] == POST_RECONCILIATION_DECISION_ROLE
    assert report["status"] == "completed"
    assert report["operational_status"] == "pass"
    assert report["terminal_status"] == "hold"
    assert report["generation_advantage_proven"] is False
    assert report["scientific_resolution"]["cross_protocol_conflict"] == {
        "status": "resolved",
        "explanation": "sampler_step_effect_dominates_observed_ranking_reversal",
        "sampler_step_changes_matched_ranking": True,
        "sample_count_changes_matched_ranking": False,
        "ddim50_2048_winner": "dense_identity",
        "ddim100_2048_winner": "cofitok",
        "ddim100_10000_winner": "cofitok",
    }
    resolution = report["scientific_resolution"]["terminal_quality"]
    assert resolution["matched_quality_checks_pass"] is True
    assert resolution["factorization_mechanism_checks_pass"] is True
    assert resolution["both_methods_absolute_fid_above_threshold"] is True
    assert resolution["both_methods_recall_below_floor"] is True
    assert report["recommended_next_stage"]["id"] == (
        "prepare_matched_100k_epsilon_stability_sampling_diagnostic"
    )
    assert report["recommended_next_stage"]["execution_ready"] is False
    assert report["recommended_next_stage"]["screening_design"][
        "sample_count_per_method_case"
    ] == 1_000


def test_post_reconciliation_decision_disables_legacy_supervisors() -> None:
    report = build_post_reconciliation_decision(**_inputs())

    disposition = report["legacy_route_disposition"]
    assert disposition["factorization_quality_regression_supervisor"][
        "eligible"
    ] is False
    assert disposition["conditioning_only_supervisor"]["eligible"] is False
    assert disposition["legacy_supervisor_execution_authorizations_must_not_be_created"]
    assert report["scientific_resolution"]["failure_classification"] == {
        "matched_quality_only_failure": False,
        "factorization_mechanism_failure": False,
        "class_only_failure": False,
        "mixed_shared_absolute_quality_support_and_class_failure": True,
        "insufficient_exposure_is_live_hypothesis": True,
        "exposure_causal_status": "not_identified_by_exposure_alone",
    }


def test_post_reconciliation_boundary_is_permanently_non_authorizing() -> None:
    report = build_post_reconciliation_decision(**_inputs())

    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    for key, value in AUTHORIZATION_BOUNDARY.items():
        if key == "new_source_bound_execution_gate_required":
            assert value is True
        else:
            assert value is False
    stage = report["recommended_next_stage"]
    assert stage["gpu_execution_allowed"] is False
    assert stage["sampling_launch_allowed"] is False
    assert stage["training_launch_allowed"] is False
    assert stage["full_300k_launch_allowed"] is False


def test_post_reconciliation_defers_capacity_exposure_and_recipe() -> None:
    report = build_post_reconciliation_decision(**_inputs())

    audit = report["candidate_asset_audit"]
    assert audit["matched_100k_epsilon_stability_sampling"]["selected"] is True
    for name in (
        "matched_capacity_qualification",
        "matched_exposure_qualification",
        "matched_training_recipe_intervention",
    ):
        assert audit[name]["selected"] is False
    assert report["scientific_resolution"]["training_exposure"][
        "full_data_equivalent_epochs"
    ] == pytest.approx(4.995445558619602)


def test_runbook_is_cpu_only_non_authorizing_and_versioned() -> None:
    text = (
        ROOT
        / "artifacts/runbooks/generation_100k_post_reconciliation_decision_v1.sh"
    ).read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=-1" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "MKL_NUM_THREADS=1" in text
    assert "100k_post_reconciliation_decision_v1_20260824" in text
    assert "build_generation_100k_post_reconciliation_decision.py" in text
    assert "verify_generation_100k_post_reconciliation_decision.py" in text
    assert "scripts/generate_samples.py" not in text
    assert "scripts/train_generation.py" not in text
    assert "nvidia-smi" not in text
    assert "kill " not in text


def test_rejects_unresolved_cross_protocol_conclusion() -> None:
    inputs = _inputs()
    inputs["reconciliation"]["comparison"]["ranking_reversal_explanation"] = (
        "sample_count_effect_dominates_observed_ranking_reversal"
    )

    with pytest.raises(ValueError, match="conclusion differs"):
        build_post_reconciliation_decision(**inputs)


def test_rejects_class_only_route_misclassification() -> None:
    inputs = _inputs()
    screen = inputs["quality_result"]["quality_screen"]
    for row in screen["checks"]:
        row["passed"] = row["name"] != "class_fidelity"
    screen["failed_checks"] = ["class_fidelity"]

    with pytest.raises(ValueError, match="failure classification differs"):
        build_post_reconciliation_decision(**inputs)


def test_rejects_factorization_or_matched_quality_failure() -> None:
    inputs = _inputs()
    screen = inputs["quality_result"]["quality_screen"]
    failed = set(EXPECTED_FAILED_CHECKS) | {"matched_fid_tolerance"}
    for row in screen["checks"]:
        row["passed"] = row["name"] not in failed
    screen["failed_checks"] = [
        row["name"] for row in screen["checks"] if row["passed"] is False
    ]

    with pytest.raises(ValueError, match="failure classification differs"):
        build_post_reconciliation_decision(**inputs)


def test_rejects_checkpoint_payload_mismatch() -> None:
    inputs = _inputs()
    inputs["terminal_evidence"]["cofitok"]["checkpoint_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="terminal cofitok evidence contract differs"):
        build_post_reconciliation_decision(**inputs)


def test_rejects_mismatched_sampling_protocol() -> None:
    inputs = _inputs()
    inputs["terminal_evidence"]["dense_identity"]["sampling"][
        "guidance_rescale"
    ] = 1.0

    with pytest.raises(ValueError, match="sampling protocol differs"):
        build_post_reconciliation_decision(**inputs)


@pytest.mark.parametrize(
    "script",
    (
        "build_generation_100k_post_reconciliation_decision.py",
        "verify_generation_100k_post_reconciliation_decision.py",
    ),
)
def test_entrypoints_import_under_runbook_pythonpath(script: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
