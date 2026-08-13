from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_PREPARATION_BOUNDARY,
    CAPACITY_PROBE_REFERENCE_REASON,
    CAPACITY_PROBE_SOURCE_DECISION_BRANCH,
    CAPACITY_PROBE_SOURCE_DECISION_REVISION,
    build_capacity_probe_preparation,
)
from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    QUALITY_BRIDGE_EXECUTION_BRANCH,
    QUALITY_BRIDGE_EXECUTION_REVISION,
)
from scripts.validate_generation_configs import validate_pair


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "generation"
BASE_CONFIGS = {
    "cofitok": (
        "imagenet256_stability_quality_bridge_"
        "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
    ),
    "dense_identity": (
        "imagenet256_stability_quality_bridge_"
        "rollout_x0_u2_ema_teacher_dense_100k.json"
    ),
}
CAPACITY_CONFIGS = {
    "cofitok": (
        "imagenet256_stability_capacity_probe_"
        "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
    ),
    "dense_identity": (
        "imagenet256_stability_capacity_probe_"
        "rollout_x0_u2_ema_teacher_dense_100k.json"
    ),
}
PREPARATION_GIT = {
    "revision": "a" * 40,
    "branch": "scale/generation-capacity-probe-v1",
    "tracked_dirty": False,
}


def _identity(name: str, character: str = "b") -> dict[str, object]:
    return {
        "path": f"/evidence/{name}",
        "bytes": 123,
        "sha256": character * 64,
    }


def _decision() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "completed",
        "role": FOLLOWUP_DECISION_ROLE,
        "decision_builder_git": {
            "revision": CAPACITY_PROBE_SOURCE_DECISION_REVISION,
            "branch": CAPACITY_PROBE_SOURCE_DECISION_BRANCH,
            "tracked_dirty": False,
        },
        "source_reports": {
            "quality_bridge_result": _identity("result.json", "2"),
        },
        "quality_bridge_execution_git": {
            "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
            "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
            "tracked_dirty": False,
        },
        "source_replay": {
            "quality_bridge_result_rebuilt_byte_equivalent": True,
            "physical_checkpoint_sample_and_real_set_reverified": True,
            "milestone_source_reports_reverified": True,
        },
        "terminal_quality": {
            "status": "hold",
            "failed_checks": ["cofitok_absolute_fid", "cofitok_recall_floor"],
        },
        "milestone_trend": {
            "shared_quality_trend_strictly_improved": True,
        },
        "recommended_next_stage": {
            "id": "prepare_matched_250m_capacity_qualification_probe",
            "category": "capacity_qualification",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(FOLLOWUP_AUTHORIZATION_BOUNDARY),
    }


def _loaded(names: dict[str, str]) -> tuple[dict[str, object], dict[str, object]]:
    configs = {
        method: load_config(CONFIG_ROOT / name) for method, name in names.items()
    }
    return configs, {
        method: config_to_dict(config) for method, config in configs.items()
    }


def _reference(method: str, character: str) -> dict[str, object]:
    checkpoint_name = "checkpoint_step_00010000.pt"
    checkpoint = {
        "path": f"/references/{method}/{checkpoint_name}",
        "bytes": 1000,
        "sha256": character * 64,
    }
    integrity = {
        "path": f"/references/{method}/{checkpoint_name}.integrity.json",
        "bytes": 500,
        "sha256": ("e" if character != "e" else "f") * 64,
    }
    return {
        "receipt": _identity(f"{method}_reference.json", character),
        "report": {
            "schema_version": 1,
            "status": "preserved",
            "role": "generation_checkpoint_hardlink_reference",
            "reason": CAPACITY_PROBE_REFERENCE_REASON,
            "expected": {
                "step": 10_000,
                "git": {
                    "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
                    "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
                    "tracked_dirty": False,
                },
            },
            "reference": {
                "directory": f"/references/{method}",
                "checkpoint": checkpoint,
                "integrity": integrity,
            },
            "storage": {
                "same_filesystem": True,
                "checkpoint_same_inode": True,
                "integrity_same_inode": True,
                "additional_checkpoint_data_blocks_required": False,
            },
            "authorization_boundary": {
                "training_launch_allowed": False,
                "gpu_use_allowed": False,
                "checkpoint_mutation_allowed": False,
                "checkpoint_deletion_allowed": False,
                "full_300k_launch_allowed": False,
                "release_allowed": False,
            },
        },
    }


def _kwargs() -> dict[str, object]:
    base_objects, base = _loaded(BASE_CONFIGS)
    capacity_objects, capacity = _loaded(CAPACITY_CONFIGS)
    return {
        "followup_decision": _decision(),
        "followup_decision_identity": _identity("decision.json", "1"),
        "quality_bridge_result_identity": _identity("result.json", "2"),
        "quality_bridge_preparation_identity": _identity("bridge.json", "3"),
        "base_configs": base,
        "capacity_configs": capacity,
        "base_config_identities": {
            method: _identity(name, "4") for method, name in BASE_CONFIGS.items()
        },
        "capacity_config_identities": {
            method: _identity(name, "5")
            for method, name in CAPACITY_CONFIGS.items()
        },
        "base_config_validation": validate_pair(
            base_objects["cofitok"],
            base_objects["dense_identity"],
            max_parameter_gap=0.02,
            stage="stability_quality_bridge",
        ),
        "capacity_config_validation": validate_pair(
            capacity_objects["cofitok"],
            capacity_objects["dense_identity"],
            max_parameter_gap=0.02,
            stage="stability_capacity_probe",
        ),
        "checkpoint_references": {
            "cofitok": _reference("cofitok", "6"),
            "dense_identity": _reference("dense_identity", "7"),
        },
        "preparation_git": PREPARATION_GIT,
        "output_root": "/checkpoints/generation/capacity_probe_250m_10k_v1",
    }


def test_capacity_probe_preparation_is_bounded_and_causal() -> None:
    report = build_capacity_probe_preparation(**_kwargs())

    assert report["status"] == "prepared"
    assert report["selection"]["stop_after_step"] == 10_000
    assert report["selection"]["configured_training_horizon"] == 100_000
    assert report["selection"]["images_seen_per_training_arm"] == 640_000
    assert report["selection"]["training_intervention"] == ["model.base_channels"]
    assert report["matched_training_contract"]["base128"][
        "parameter_counts"
    ] == {"cofitok": 62_834_083, "dense_identity": 62_824_707}
    assert report["matched_training_contract"]["base256"][
        "parameter_counts"
    ] == {"cofitok": 250_153_763, "dense_identity": 250_135_043}
    assert report["evaluation_contract"]["samples_per_arm"] == 2_048
    assert report["authorization_boundary"] == CAPACITY_PROBE_PREPARATION_BOUNDARY
    assert report["authorization_boundary"]["capacity_probe_execution_allowed"] is False
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_capacity_probe_rejects_non_capacity_decision() -> None:
    values = _kwargs()
    values["followup_decision"]["recommended_next_stage"]["id"] = (
        "run_matched_factorization_mechanism_recovery_probe"
    )

    with pytest.raises(ValueError, match="does not select"):
        build_capacity_probe_preparation(**values)


def test_capacity_probe_rejects_another_decision_builder() -> None:
    values = _kwargs()
    values["followup_decision"]["decision_builder_git"]["revision"] = "f" * 40

    with pytest.raises(ValueError, match="exact follow-up decision builder"):
        build_capacity_probe_preparation(**values)


def test_capacity_probe_rejects_cross_bound_bridge_result() -> None:
    values = _kwargs()
    values["quality_bridge_result_identity"] = _identity("other.json", "9")

    with pytest.raises(ValueError, match="binds another quality bridge result"):
        build_capacity_probe_preparation(**values)


def test_capacity_probe_rejects_objective_or_schedule_confound() -> None:
    values = _kwargs()
    values["capacity_configs"]["cofitok"]["loss"][
        "rollout_consistency_weight"
    ] = 0.2

    with pytest.raises(ValueError, match="capacity intervention differs"):
        build_capacity_probe_preparation(**values)


def test_capacity_probe_rejects_unverified_reference_boundary() -> None:
    values = _kwargs()
    values["checkpoint_references"]["cofitok"]["report"][
        "authorization_boundary"
    ]["checkpoint_mutation_allowed"] = True

    with pytest.raises(ValueError, match="checkpoint reference contract differs"):
        build_capacity_probe_preparation(**values)


@pytest.mark.parametrize(
    "entrypoint",
    (
        "scripts/build_generation_capacity_probe_preparation.py",
        "scripts/verify_generation_capacity_probe_preparation.py",
    ),
)
def test_capacity_probe_preparation_entrypoints_import(entrypoint: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "capacity" in result.stdout.lower()
