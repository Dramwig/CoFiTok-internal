from __future__ import annotations

import copy
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from cofitok.generation.semantic_residual_alignment_probe import (
    AUTHORIZATION_ROLE,
    CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG,
    CLAIM_BOUNDARY,
    CONFIG_PATHS,
    CONTROL_RESIDUAL_ALIGNMENT_CONFIG,
    EXECUTION_BOUNDARY,
    EXPECTED_FAILED_CHECKS,
    EXPECTED_SOURCE_SHA256,
    OUTPUT_ROOT,
    PREPARATION_CONFIG_KEYS,
    PREPARATION_ROLE,
    QUALITY_BRIDGE_GIT,
    QUALITY_BRIDGE_RESULT_PATH,
    RUN_NAMES,
    SCOPE,
    SOURCE_POSTEVALUATION_GIT,
    SOURCE_POSTEVALUATION_PATH,
    STAGE,
    build_execution_authorization,
    semantic_residual_alignment_probe_contract,
)


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40
TREE = "b" * 40
BRANCH = "analysis/generation-semantic-residual-alignment-test"


def _configs() -> dict[str, dict]:
    return {
        name: config_to_dict(load_config(ROOT / relative))
        for name, relative in CONFIG_PATHS.items()
    }


def _identity(
    path: str,
    *,
    digest: str = "c" * 64,
    size: int = 100,
) -> dict:
    return {"path": path, "bytes": size, "sha256": digest}


def _preparation(configs: dict[str, dict]) -> dict:
    contract = semantic_residual_alignment_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        residual_cofitok=configs["residual_cofitok"],
        residual_dense=configs["residual_dense_identity"],
    )
    assert contract["valid"]
    identities = {
        key: _identity(f"configs/{key}.json", digest=str(index + 1) * 64)
        for index, key in enumerate(PREPARATION_CONFIG_KEYS.values())
    }
    return {
        **contract,
        "git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "output_root": OUTPUT_ROOT,
        "configs": identities,
        "parameter_counts": {
            "control_cofitok": 62_834_083,
            "residual_cofitok": 62_834_083,
            "control_dense": 62_824_707,
            "residual_dense": 62_824_707,
        },
    }


def _standing() -> dict:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_TEXT,
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
            "received_at": "2026-08-13T00:11:00+08:00",
        },
        "preserved_safety_boundaries": STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    }


def _quality() -> dict:
    return {
        "schema_version": 1,
        "role": "stability_full_data_quality_bridge_result",
        "status": "completed",
        "git": QUALITY_BRIDGE_GIT,
        "quality_screen": {
            "status": "hold",
            "failed_checks": EXPECTED_FAILED_CHECKS,
            "non_authorizing": True,
        },
        "terminal": {"class_fidelity": {"status": "hold"}},
        "authorization_boundary": {
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_authorization_allowed": False,
        },
    }


def _source_postevaluation() -> dict:
    return {
        "schema_version": 1,
        "role": "generation_conditioning_ranking_four_arm_postevaluation",
        "status": "completed",
        "git": SOURCE_POSTEVALUATION_GIT,
        "decision": {
            "method_passes": {"cofitok": False, "dense_identity": False},
            "shared_semantic_alignment_recovery_supported": False,
            "cofitok_specific_advantage_claim_allowed": False,
            "recommended_next_action": (
                "revise_training_time_semantic_alignment_objective"
            ),
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "authorizes_training": False,
            "authorizes_sampling": False,
            "authorizes_checkpoint_promotion": False,
            "authorizes_full_training": False,
            "authorizes_release": False,
        },
    }


def _authorization_arguments() -> dict:
    configs = _configs()
    return {
        "preparation": _preparation(configs),
        "preparation_identity": _identity("/evidence/preparation.json"),
        "standing_authorization": _standing(),
        "standing_authorization_identity": _identity(
            "/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json"
        ),
        "quality_bridge_result": _quality(),
        "quality_bridge_result_identity": _identity(
            QUALITY_BRIDGE_RESULT_PATH,
            digest=EXPECTED_SOURCE_SHA256["quality_bridge_result"],
        ),
        "source_postevaluation": _source_postevaluation(),
        "source_postevaluation_identity": _identity(
            SOURCE_POSTEVALUATION_PATH,
            digest=EXPECTED_SOURCE_SHA256["source_postevaluation"],
        ),
        "runbook_identity": _identity("/checkout/artifacts/runbooks/probe.sh"),
        "authorization_git": {
            "revision": REVISION,
            "tree": TREE,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "expected_revision": REVISION,
        "expected_tree": TREE,
        "expected_branch": BRANCH,
        "expected_output_root": OUTPUT_ROOT,
    }


def test_actual_four_arm_configs_match_exact_residual_contract() -> None:
    configs = _configs()
    report = semantic_residual_alignment_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        residual_cofitok=configs["residual_cofitok"],
        residual_dense=configs["residual_dense_identity"],
    )

    assert report["status"] == "pass"
    assert report["issues"] == []
    assert report["control_residual_alignment_config"] == (
        CONTROL_RESIDUAL_ALIGNMENT_CONFIG
    )
    assert report["candidate_residual_alignment_config"] == (
        CANDIDATE_RESIDUAL_ALIGNMENT_CONFIG
    )
    assert report["execution_boundary"] == EXECUTION_BOUNDARY
    assert report["claim_boundary"] == CLAIM_BOUNDARY


def test_probe_rejects_drift_outside_objective_fields() -> None:
    configs = _configs()
    configs["residual_cofitok"]["optimization"]["learning_rate"] = 0.0002

    report = semantic_residual_alignment_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        residual_cofitok=configs["residual_cofitok"],
        residual_dense=configs["residual_dense_identity"],
    )

    assert report["status"] == "fail"
    assert any("outside objective fields" in issue for issue in report["issues"])


def test_probe_rejects_legacy_ranking_fields() -> None:
    configs = _configs()
    configs["residual_dense_identity"]["loss"][
        "class_conditioning_ranking_weight"
    ] = 0.05

    report = semantic_residual_alignment_probe_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        residual_cofitok=configs["residual_cofitok"],
        residual_dense=configs["residual_dense_identity"],
    )

    assert report["status"] == "fail"
    assert any("legacy class-ranking" in issue for issue in report["issues"])


def test_authorization_is_exact_and_permanently_non_promoting() -> None:
    report = build_execution_authorization(**_authorization_arguments())

    assert report["role"] == AUTHORIZATION_ROLE
    assert report["status"] == "authorized"
    assert report["scope"] == SCOPE
    assert report["stage"] == STAGE
    assert report["execution_boundary"] == EXECUTION_BOUNDARY
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["generation_advantage_proven"] is False


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda args: args["source_postevaluation"]["decision"].__setitem__(
                "shared_semantic_alignment_recovery_supported", True
            ),
            "postevaluation",
        ),
        (
            lambda args: args["quality_bridge_result"]["quality_screen"].__setitem__(
                "status", "pass"
            ),
            "quality-bridge",
        ),
        (
            lambda args: args.__setitem__("expected_output_root", "/tmp/arbitrary"),
            "preparation contract|output root",
        ),
    ],
)
def test_authorization_rejects_route_or_scope_drift(mutator, match: str) -> None:
    arguments = _authorization_arguments()
    mutator(arguments)

    with pytest.raises(ValueError, match=match):
        build_execution_authorization(**arguments)


def test_contract_inventory_is_stable() -> None:
    assert RUN_NAMES == (
        "control_cofitok",
        "residual_cofitok",
        "control_dense_identity",
        "residual_dense_identity",
    )
    assert PREPARATION_ROLE == (
        "generation_semantic_residual_alignment_probe_preparation"
    )
