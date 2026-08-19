from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_full_data_bridge import (
    BASE_CONFIG_RELATIVE_PATHS,
    CLAIM_BOUNDARY,
    EXPECTED_OUTPUT_ROOT,
    PLANNED_EXECUTION,
    POSTTRAINING_DECISION,
    POSTTRAINING_GIT,
    POSTTRAINING_REPORT_ROLE,
    PREPARATION_BOUNDARY,
    RANKED_CONFIG_RELATIVE_PATHS,
    RANKED_RANKING_CONFIG,
    SCOPE,
    STAGE,
    build_full_data_ranked_bridge_preparation,
    full_data_ranked_bridge_contract,
    validate_full_data_ranked_bridge_preparation,
    validate_posttraining_sampling_confirmation,
)
from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    CLAIM_BOUNDARY as POSTTRAINING_CLAIM_BOUNDARY,
    EXPECTED_OUTPUT_ROOT as POSTTRAINING_OUTPUT_ROOT,
    STAGE as POSTTRAINING_STAGE,
)
from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from scripts import (
    prepare_generation_conditioning_ranking_full_data_bridge as prepare_cli,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRANCH = "scale/generation-conditioning-ranking-full-data-100k-v1"
REVISION = "b" * 40


def _configs() -> dict[str, dict]:
    return {
        "base_cofitok": config_to_dict(
            load_config(PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATHS["cofitok"])
        ),
        "base_dense_identity": config_to_dict(
            load_config(PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATHS["dense_identity"])
        ),
        "ranked_cofitok": config_to_dict(
            load_config(PROJECT_ROOT / RANKED_CONFIG_RELATIVE_PATHS["cofitok"])
        ),
        "ranked_dense_identity": config_to_dict(
            load_config(PROJECT_ROOT / RANKED_CONFIG_RELATIVE_PATHS["dense_identity"])
        ),
    }


def _identity(name: str, character: str = "a") -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 123,
        "sha256": character * 64,
    }


def _standing_authorization() -> dict:
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
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _followup() -> dict:
    quality = _identity("quality", "a")
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_quality_bridge_followup_experiment_decision",
        "quality_bridge_execution_git": {
            "revision": "cf0e5faa94bf4ab38d947b921935b3b765b5537a",
            "branch": "scale/generation-stability-quality-bridge-100k",
            "tracked_dirty": False,
        },
        "source_reports": {"quality_bridge_result": quality},
        "terminal_quality": {
            "failed_checks": ["class_fidelity"],
            "checks": [{"name": "class_fidelity", "passed": False}],
        },
        "recommended_next_stage": {
            "id": "run_class_conditioning_fidelity_diagnostic",
            "category": "class_conditioning_recovery",
            "execution_ready": False,
            "gpu_execution_allowed": False,
            "full_300k_launch_allowed": False,
        },
    }


def _posttraining_confirmation() -> dict:
    return {
        "schema_version": 1,
        "role": POSTTRAINING_REPORT_ROLE,
        "status": "completed",
        "stage": POSTTRAINING_STAGE,
        "output_root": POSTTRAINING_OUTPUT_ROOT,
        "git": copy.deepcopy(POSTTRAINING_GIT),
        "sources": {},
        "methods": {
            "cofitok": {"pass": True},
            "dense_identity": {"pass": True},
        },
        "decision": copy.deepcopy(POSTTRAINING_DECISION),
        "claim_boundary": copy.deepcopy(POSTTRAINING_CLAIM_BOUNDARY),
    }


def _preparation_kwargs() -> dict:
    configs = _configs()
    return {
        "posttraining_confirmation": _posttraining_confirmation(),
        "posttraining_confirmation_identity": _identity("posttraining", "b"),
        "quality_bridge_followup": _followup(),
        "quality_bridge_followup_identity": _identity("followup", "c"),
        "standing_authorization": _standing_authorization(),
        "standing_authorization_identity": _identity("standing", "d"),
        **configs,
        "config_identities": {
            name: _identity(name, "e") for name in configs
        },
        "parameter_counts": {
            "base_cofitok": 100,
            "ranked_cofitok": 100,
            "base_dense_identity": 100,
            "ranked_dense_identity": 100,
        },
        "builder_git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "expected_revision": REVISION,
        "expected_branch": BRANCH,
        "expected_output_root": EXPECTED_OUTPUT_ROOT,
    }


def test_full_data_ranked_configs_match_base_outside_ranking_fields() -> None:
    contract = full_data_ranked_bridge_contract(**_configs())

    assert contract["status"] == "pass"
    assert contract["valid"] is True
    assert contract["issues"] == []
    assert contract["ranked_ranking_config"] == RANKED_RANKING_CONFIG
    assert contract["planned_execution"] == PLANNED_EXECUTION
    assert contract["preparation_boundary"] == PREPARATION_BOUNDARY
    assert contract["claim_boundary"] == CLAIM_BOUNDARY


def test_full_data_ranked_contract_rejects_nonranking_drift() -> None:
    configs = _configs()
    configs["ranked_cofitok"]["optimization"]["learning_rate"] = 2e-4

    contract = full_data_ranked_bridge_contract(**configs)

    assert contract["valid"] is False
    assert any("outside ranking fields" in issue for issue in contract["issues"])


def test_posttraining_source_requires_shared_generated_sample_pass() -> None:
    report = _posttraining_confirmation()
    assert validate_posttraining_sampling_confirmation(report) == report

    asymmetric = copy.deepcopy(report)
    asymmetric["decision"]["method_passes"]["dense_identity"] = False
    with pytest.raises(ValueError, match="did not select scaling"):
        validate_posttraining_sampling_confirmation(asymmetric)

    authorizing = copy.deepcopy(report)
    authorizing["claim_boundary"]["authorizes_training"] = True
    with pytest.raises(ValueError, match="did not select scaling"):
        validate_posttraining_sampling_confirmation(authorizing)


def test_preparation_is_source_bound_fresh_and_non_authorizing() -> None:
    report = build_full_data_ranked_bridge_preparation(**_preparation_kwargs())

    assert report["status"] == "pass"
    assert report["stage"] == STAGE
    assert report["scope"] == SCOPE
    assert report["planned_execution"]["fresh_start_required"] is True
    assert (
        report["planned_execution"][
            "resume_from_existing_quality_bridge_checkpoint_allowed"
        ]
        is False
    )
    assert report["gpu_execution_authorized"] is False
    assert report["preparation_boundary"]["training_allowed"] is False
    assert report["claim_boundary"]["authorizes_full_300k"] is False
    assert report["source_decisions"]["quality_bridge_failed_checks"] == [
        "class_fidelity"
    ]
    assert validate_full_data_ranked_bridge_preparation(
        report,
        expected_revision=REVISION,
        expected_branch=BRANCH,
    ) == report


def test_preparation_rejects_wrong_followup_or_parameter_drift() -> None:
    kwargs = _preparation_kwargs()
    kwargs["quality_bridge_followup"]["recommended_next_stage"]["id"] = (
        "run_capacity_probe"
    )
    with pytest.raises(ValueError, match="class-conditioning follow-up differs"):
        build_full_data_ranked_bridge_preparation(**kwargs)

    kwargs = _preparation_kwargs()
    kwargs["parameter_counts"]["ranked_cofitok"] += 1
    with pytest.raises(ValueError, match="CoFiTok parameter count differs"):
        build_full_data_ranked_bridge_preparation(**kwargs)

    kwargs = _preparation_kwargs()
    kwargs["parameter_counts"]["base_cofitok"] = 200
    kwargs["parameter_counts"]["ranked_cofitok"] = 200
    with pytest.raises(ValueError, match="parameter gap exceeds 2%"):
        build_full_data_ranked_bridge_preparation(**kwargs)


def test_preparation_replay_rejects_contract_tampering() -> None:
    report = build_full_data_ranked_bridge_preparation(**_preparation_kwargs())

    tampered = copy.deepcopy(report)
    tampered["source_decisions"]["posttraining_sampling"][
        "cofitok_specific_advantage_claim_allowed"
    ] = True
    with pytest.raises(ValueError, match="preparation differs"):
        validate_full_data_ranked_bridge_preparation(
            tampered,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )

    tampered = copy.deepcopy(report)
    tampered["parameter_relative_gap"] = 0.01
    with pytest.raises(ValueError, match="parameter gap differs"):
        validate_full_data_ranked_bridge_preparation(
            tampered,
            expected_revision=REVISION,
            expected_branch=BRANCH,
        )


def test_prepare_cli_writes_once_and_replays_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posttraining_path = tmp_path / "posttraining.json"
    followup_path = tmp_path / "followup.json"
    standing_path = tmp_path / "standing.json"
    output = tmp_path / "preparation.json"
    for path, payload in (
        (posttraining_path, _posttraining_confirmation()),
        (followup_path, _followup()),
        (standing_path, _standing_authorization()),
    ):
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        prepare_cli,
        "replay_posttraining_sampling_confirmation",
        lambda path: (
            json.loads(path.read_text(encoding="utf-8")),
            prepare_cli.file_identity(path),
        ),
    )
    monkeypatch.setattr(
        prepare_cli,
        "git_provenance",
        lambda _root: {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        prepare_cli,
        "_parameter_count",
        lambda _path: 100,
    )
    arguments = [
        "prepare_generation_conditioning_ranking_full_data_bridge.py",
        "--posttraining-confirmation",
        str(posttraining_path),
        "--expected-posttraining-confirmation-sha256",
        prepare_cli.file_identity(posttraining_path)["sha256"],
        "--quality-bridge-followup",
        str(followup_path),
        "--expected-quality-bridge-followup-sha256",
        prepare_cli.file_identity(followup_path)["sha256"],
        "--standing-authorization",
        str(standing_path),
        "--expected-standing-authorization-sha256",
        prepare_cli.file_identity(standing_path)["sha256"],
        "--base-cofitok",
        str(PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATHS["cofitok"]),
        "--base-dense",
        str(PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATHS["dense_identity"]),
        "--ranked-cofitok",
        str(PROJECT_ROOT / RANKED_CONFIG_RELATIVE_PATHS["cofitok"]),
        "--ranked-dense",
        str(PROJECT_ROOT / RANKED_CONFIG_RELATIVE_PATHS["dense_identity"]),
        "--expected-revision",
        REVISION,
        "--expected-branch",
        BRANCH,
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    prepare_cli.main()
    first = output.read_bytes()

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    prepare_cli.main()

    assert output.read_bytes() == first
    report = json.loads(first)
    assert report["gpu_execution_authorized"] is False
    assert report["source_reports"]["posttraining_sampling_confirmation"] == (
        prepare_cli.file_identity(posttraining_path)
    )
