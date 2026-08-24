from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.semantic_residual_alignment_probe import (
    CLAIM_BOUNDARY,
    CONFIG_PATHS,
    EXECUTION_BOUNDARY,
    METHOD_ARMS,
    OUTPUT_ROOT,
    POSTEVALUATION_ROLE,
    PREPARATION_CONFIG_KEYS,
    RUN_NAMES,
    TRAINING_STATUS_ROLE,
    semantic_residual_alignment_probe_contract,
)
from scripts import build_generation_semantic_residual_alignment_posteval as posteval


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "analysis/generation-semantic-residual-alignment-test"
GIT = {"revision": REVISION, "branch": BRANCH, "tracked_dirty": False}


def _configs() -> dict[str, dict]:
    return {
        run: config_to_dict(load_config(ROOT / path))
        for run, path in CONFIG_PATHS.items()
    }


def _parameter_counts() -> dict[str, int]:
    return {
        "control_cofitok": 101,
        "residual_cofitok": 101,
        "control_dense": 99,
        "residual_dense": 99,
    }


def _preparation(configs: dict[str, dict]) -> dict:
    contract = semantic_residual_alignment_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        residual_cofitok=configs["residual_cofitok"],
        residual_dense=configs["residual_dense_identity"],
    )
    assert contract["valid"]
    return {
        **contract,
        "git": {**GIT, "tree": TREE},
        "output_root": OUTPUT_ROOT,
        "configs": {
            key: {
                "path": f"configs/{key}.json",
                "bytes": 100,
                "sha256": "c" * 64,
            }
            for key in PREPARATION_CONFIG_KEYS.values()
        },
        "parameter_counts": _parameter_counts(),
    }


def _checkpoint_sha(run: str, step: int = 1_000) -> str:
    character = {
        "control_cofitok": "1",
        "residual_cofitok": "2",
        "control_dense_identity": "3",
        "residual_dense_identity": "4",
    }[run]
    return character * 62 + f"{step // 250:02d}"


def _training_evidence(
    configs: dict[str, dict],
) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    status_runs: dict[str, dict] = {}
    reports: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    for run in RUN_NAMES:
        residual = run.startswith("residual_")
        run_dir = f"{OUTPUT_ROOT}/{run}"
        checkpoint_sha = _checkpoint_sha(run)
        final = {
            "step": 1_000,
            "samples_seen": 64_000,
            "epsilon": 0.03,
            "class_conditioning_residual_alignment": 0.02 if residual else 0.0,
            "class_conditioning_residual_alignment_scale": 1.0 if residual else 0.0,
        }
        status_runs[run] = {
            "training_report": f"{run_dir}/training_report.json",
            "checkpoint_sha256": checkpoint_sha,
            "final_metrics": copy.deepcopy(final),
        }
        config_key = PREPARATION_CONFIG_KEYS[run]
        reports[run] = {
            "training_complete": True,
            "completed_steps": 1_000,
            "target_steps": 1_000,
            "output_dir": run_dir,
            "config": configs[run],
            "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
            "parameter_count": _parameter_counts()[config_key],
            "latest_checkpoint": {
                "checkpoint": "checkpoint_step_00001000.pt",
                "step": 1_000,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_bytes": 1_000,
                "integrity_manifest": "checkpoint_step_00001000.pt.integrity.json",
                "dataset_identity_sha256": "d" * 64,
                "runtime_environment_sha256": "e" * 64,
            },
            "runtime_environment_sha256": "e" * 64,
            "dataset_provenance": {"identity_sha256": "d" * 64},
            "final_metrics": copy.deepcopy(final),
        }
        objective = configs[run]["loss"]
        checkpoints = []
        for step in (500, 750, 1_000):
            checkpoints.append(
                {
                    "status": "verified",
                    "step": step,
                    "checkpoint_bytes": 1_000 + step,
                    "checkpoint_sha256": _checkpoint_sha(run, step),
                    "git_revision": REVISION,
                    "git_branch": BRANCH,
                    "git_dirty": False,
                    "dataset_identity_sha256": "d" * 64,
                    "runtime_environment_sha256": "e" * 64,
                }
            )
        audits[run] = {
            "schema_version": 2,
            "status": "complete",
            "issues": [],
            "warnings": [],
            "expected_steps": 1_000,
            "last_step": 1_000,
            "progress_fraction": 1.0,
            "metric_row_count": 40,
            "run_dir": run_dir,
            "checkpoint": {
                "interval": 250,
                "status": "available",
                "required_steps": [500, 750, 1_000],
                "missing_required_steps": [],
                "required_integrity": {
                    "policy": "required",
                    "status": "verified",
                    "requested_steps": [500, 750, 1_000],
                    "reached_steps": [500, 750, 1_000],
                    "checkpoints": checkpoints,
                }
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
                    "class_conditioning_residual_alignment": {
                        "weight": objective[
                            "class_conditioning_residual_alignment_weight"
                        ],
                        "start_step": objective[
                            "class_conditioning_residual_alignment_start_step"
                        ],
                        "warmup_steps": objective[
                            "class_conditioning_residual_alignment_warmup_steps"
                        ],
                        "verified_rows": 40,
                        "active_rows": 30 if residual else 0,
                        "nonzero_loss_rows": 30 if residual else 0,
                    }
                },
            },
        }
    status = {
        "schema_version": 1,
        "role": TRAINING_STATUS_ROLE,
        "status": "completed",
        "revision": REVISION,
        "tree": TREE,
        "branch": BRANCH,
        "runs": status_runs,
        "execution_boundary": EXECUTION_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "generation_advantage_proven": False,
    }
    return status, reports, audits


def test_training_evidence_requires_all_twelve_physical_checkpoints() -> None:
    configs = _configs()
    status, reports, audits = _training_evidence(configs)

    result = posteval._validate_training_evidence(
        preparation=_preparation(configs),
        training_status=status,
        training_reports=reports,
        training_audits=audits,
        expected_configs=configs,
    )

    assert set(result["runs"]) == set(RUN_NAMES)
    assert sum(
        len(run["required_checkpoint_integrity"])
        for run in result["runs"].values()
    ) == 12
    assert result["runs"]["residual_cofitok"]["final_metrics"][
        "class_conditioning_residual_alignment_scale"
    ] == 1.0


def test_training_evidence_rejects_physical_lineage_drift() -> None:
    configs = _configs()
    status, reports, audits = _training_evidence(configs)
    audits["residual_dense_identity"]["checkpoint"]["required_integrity"][
        "checkpoints"
    ][1]["git_revision"] = "f" * 40

    with pytest.raises(ValueError, match="checkpoint lineage"):
        posteval._validate_training_evidence(
            preparation=_preparation(configs),
            training_status=status,
            training_reports=reports,
            training_audits=audits,
            expected_configs=configs,
        )


def _row(sample: int, timestep: int, *, recovered: bool) -> dict:
    correct = 0.95 if recovered else 1.0
    wrong = correct / (0.95 if recovered else 1.01)
    null = correct / (0.96 if recovered else 1.02)
    return {
        "sample_index": sample,
        "timestep": timestep,
        "correct_label": 128 + sample,
        "wrong_label": 378 + sample,
        "noise_seed": 314159 + sample,
        "conditions": {
            "correct": {"epsilon_mse_to_noise": correct},
            "wrong": {"epsilon_mse_to_noise": wrong},
            "null": {"epsilon_mse_to_noise": null},
        },
    }


def _sensitivity(recovered: bool) -> dict:
    return {
        "sample_rows": [
            _row(sample, timestep, recovered=recovered)
            for sample in range(8)
            for timestep in (100, 500, 700, 900)
        ]
    }


def test_postevaluation_pass_is_still_non_authorizing(monkeypatch) -> None:
    training_contract = {
        "revision": REVISION,
        "tree": TREE,
        "branch": BRANCH,
        "residual_min_timestep": 500,
        "runs": {run: {} for run in RUN_NAMES},
    }
    monkeypatch.setattr(
        posteval,
        "_validate_training_evidence",
        lambda **kwargs: training_contract,
    )
    monkeypatch.setattr(posteval, "_validate_authorization", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        posteval,
        "_validate_sensitivity_reports",
        lambda **kwargs: {"request": copy.deepcopy(posteval.EXPECTED_REQUEST)},
    )
    reports = {
        "control_cofitok": _sensitivity(False),
        "residual_cofitok": _sensitivity(True),
        "control_dense_identity": _sensitivity(False),
        "residual_dense_identity": _sensitivity(True),
    }

    result = posteval.build_postevaluation(
        preparation={},
        preparation_source={},
        execution_authorization={},
        training_status={},
        training_reports={},
        training_audits={},
        expected_configs={},
        sensitivity_reports=reports,
        sources={"synthetic": True},
        git=GIT,
    )

    assert result["role"] == POSTEVALUATION_ROLE
    assert result["decision"]["shared_semantic_alignment_recovery_supported"] is True
    assert result["decision"]["cofitok_specific_advantage_claim_allowed"] is False
    assert result["generation_advantage_proven"] is False
    assert result["evaluation_contract"][
        "wrong_label_offset_is_held_out_from_training"
    ] is True
    assert result["evaluation_contract"]["eligible_timesteps"] == [500, 700, 900]
    for field, value in result["claim_boundary"].items():
        if field.startswith("authorizes_"):
            assert value is False


def test_method_mapping_remains_matched() -> None:
    assert METHOD_ARMS == {
        "cofitok": ("control_cofitok", "residual_cofitok"),
        "dense_identity": ("control_dense_identity", "residual_dense_identity"),
    }
