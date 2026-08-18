from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    EXECUTION_BOUNDARY,
    PROBE_AUTHORIZATION_TEXT,
    PROBE_SCOPE,
    conditioning_ranking_probe_contract,
    validate_conditioning_ranking_probe_approval,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "generation"


def _configs() -> dict[str, dict]:
    names = {
        "control_cofitok": (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_"
            "k8_probe1k.json"
        ),
        "control_dense": (
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_"
            "dense_probe1k.json"
        ),
        "ranked_cofitok": (
            "imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_"
            "classrank_k8_probe1k.json"
        ),
        "ranked_dense": (
            "imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_"
            "classrank_dense_probe1k.json"
        ),
    }
    return {
        key: config_to_dict(load_config(CONFIG_ROOT / name))
        for key, name in names.items()
    }


def test_four_arm_probe_changes_only_the_shared_ranking_contract() -> None:
    configs = _configs()
    report = conditioning_ranking_probe_contract(**configs)

    assert report["valid"] is True, report["issues"]
    assert report["pair_contracts"]["control"]["valid"] is True
    assert report["pair_contracts"]["ranked"]["valid"] is True
    assert report["ranked_ranking_config"][
        "class_conditioning_ranking_weight"
    ] == 0.05
    assert report["execution_boundary"]["steps_per_run"] == 1_000
    assert report["gpu_execution_authorized"] is False


def test_four_arm_probe_rejects_nonranking_recipe_drift() -> None:
    configs = _configs()
    configs["ranked_dense"] = copy.deepcopy(configs["ranked_dense"])
    configs["ranked_dense"]["optimization"]["learning_rate"] = 2e-4

    report = conditioning_ranking_probe_contract(**configs)

    assert report["valid"] is False
    assert any("outside the ranking fields" in issue for issue in report["issues"])


def test_probe_approval_is_exact_and_non_authorizing_beyond_1k() -> None:
    revision = "a" * 40
    preparation_sha = "b" * 64
    output_root = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "conditioning_ranking_four_arm_probe1k_v1"
    )
    approval = {
        "schema_version": 1,
        "role": "generation_conditioning_ranking_probe_execution_approval",
        "status": "approved",
        "scope": PROBE_SCOPE,
        "user_authorization_text": PROBE_AUTHORIZATION_TEXT,
        "authorized_revision": revision,
        "preparation_report_sha256": preparation_sha,
        "output_root": output_root,
        "authorization_boundary": EXECUTION_BOUNDARY,
    }

    validated = validate_conditioning_ranking_probe_approval(
        approval,
        expected_revision=revision,
        expected_preparation_sha256=preparation_sha,
        expected_output_root=output_root,
    )

    assert validated["authorization_boundary"]["full_training_launch_allowed"] is False
    assert validated["authorization_boundary"]["followup_training_allowed"] is False

    approval["authorization_boundary"] = {
        **EXECUTION_BOUNDARY,
        "followup_training_allowed": True,
    }
    with pytest.raises(ValueError, match="exact scope"):
        validate_conditioning_ranking_probe_approval(
            approval,
            expected_revision=revision,
            expected_preparation_sha256=preparation_sha,
            expected_output_root=output_root,
        )


def test_probe_runbook_requires_fully_clean_idle_checkout() -> None:
    runbook = (
        ROOT
        / "artifacts"
        / "runbooks"
        / "generation_conditioning_ranking_four_arm_probe1k_v1.sh"
    ).read_text(encoding="utf-8")

    assert '[[ -z "$(git status --porcelain)" ]]' in runbook
    assert "--untracked-files=no" not in runbook
    assert "nvidia-smi --query-compute-apps=pid" in runbook
    assert 'test ! -e "$OUTPUT_ROOT"' in runbook
