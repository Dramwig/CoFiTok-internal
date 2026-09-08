from __future__ import annotations

import copy

import pytest

from cofitok.generation.terminal_snr_large_capacity import (
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_EVALUATION_CONTRACT,
    METHODS,
    MILESTONE_STEPS,
    TARGET_STEPS,
)
from cofitok.generation.terminal_snr_large_capacity_execution import (
    GOAL_BINDING,
    STAGE_BOUNDARY,
    build_terminal_snr_large_capacity_stage_authorization,
    validate_terminal_snr_large_capacity_stage_authorization,
)
from test_generation_terminal_snr_large_capacity import _build as _build_preparation


OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "terminal_snr_endpoint0975_large_capacity_300k_v1"
)
PREPARATION_ID = {
    "path": "/evidence/terminal_snr_large_capacity_preparation.json",
    "bytes": 123,
    "sha256": "a" * 64,
}
EXECUTION_GIT = {
    "revision": "b" * 40,
    "tree": "c" * 40,
    "branch": "scale/generation-terminal-snr-capacity-full-v1",
    "tracked_dirty": False,
}


def _identity(path: str, character: str) -> dict[str, object]:
    return {"path": path, "bytes": 123, "sha256": character * 64}


def _preparation() -> dict:
    return {
        "selection": {
            "output_root": OUTPUT_ROOT,
            "fresh_initialization_required": True,
            "resume_checkpoint_allowed": False,
            "condition": "endpoint0975",
            "methods": list(METHODS),
            "configured_training_steps": TARGET_STEPS,
        },
        "source_evidence": {
            "configs": {
                "cofitok": _identity("/configs/cofitok.json", "d"),
                "dense_identity": _identity("/configs/dense.json", "e"),
            }
        },
    }


def _patch_preparation(monkeypatch: pytest.MonkeyPatch) -> dict:
    prepared = _preparation()
    monkeypatch.setattr(
        "cofitok.generation.terminal_snr_large_capacity_execution."
        "validate_terminal_snr_large_capacity_preparation_contract",
        lambda value: value,
    )
    return prepared


def _build(monkeypatch: pytest.MonkeyPatch) -> dict:
    return build_terminal_snr_large_capacity_stage_authorization(
        preparation=_patch_preparation(monkeypatch),
        preparation_identity=PREPARATION_ID,
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
        approved_by="user",
        approved_at="2026-09-08T00:00:00+00:00",
        source_instruction=(
            "Continue toward the explicitly requested 250M/300K matched "
            "training, per-method 50K DDIM-250 formal evaluation, and terminal audit."
        ),
    )


def test_stage_authorization_binds_exact_fresh_300k_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    selection = report["selection"]
    assert selection["preparation"] == PREPARATION_ID
    assert selection["execution_checkout"] == EXECUTION_GIT
    assert selection["output_root"] == OUTPUT_ROOT
    assert selection["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert selection["effective_batch_size"] == EFFECTIVE_BATCH_SIZE
    assert selection["configured_training_steps"] == TARGET_STEPS
    assert selection["milestone_steps"] == list(MILESTONE_STEPS)
    assert selection["fresh_initialization_required"] is True
    assert selection["resume_allowed"] is False
    assert selection["formal_evaluation"] == FORMAL_EVALUATION_CONTRACT
    assert report["approval_record"]["goal_binding"] == GOAL_BINDING
    assert report["authorization_boundary"] == STAGE_BOUNDARY
    assert report["next_stage"]["execution_ready"] is False
    assert report["next_stage"]["training_launch_allowed"] is False


def test_stage_authorization_accepts_the_canonical_preparation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = _build_preparation(monkeypatch)
    report = build_terminal_snr_large_capacity_stage_authorization(
        preparation=preparation,
        preparation_identity=PREPARATION_ID,
        execution_checkout=EXECUTION_GIT,
        output_root=OUTPUT_ROOT,
        approved_by="user",
        approved_at="2026-09-08T00:00:00+00:00",
        source_instruction=(
            "Explicit 250M/300K matched training, per-method 50K DDIM-250 "
            "formal evaluation, and terminal completion audit."
        ),
    )
    assert report["selection"]["output_root"] == OUTPUT_ROOT
    assert report["authorization_boundary"]["training_launch_allowed"] is False


def test_stage_authorization_is_not_execution_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(monkeypatch)
    assert report["authorization_boundary"]["decision_is_execution_authorization"] is False
    assert report["authorization_boundary"]["gpu_runtime_preflight_allowed"] is True
    assert report["authorization_boundary"]["training_launch_allowed"] is False
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert report["authorization_boundary"]["process_signals_allowed"] is False


def test_stage_authorization_rejects_a_vague_source_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="explicit 250M/300K"):
        build_terminal_snr_large_capacity_stage_authorization(
            preparation=_patch_preparation(monkeypatch),
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
            approved_by="user",
            approved_at="now",
            source_instruction="continue the experiment",
        )


def test_stage_authorization_rejects_resume_or_changed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    prepared["selection"]["resume_checkpoint_allowed"] = True
    with pytest.raises(ValueError, match="selection differs"):
        build_terminal_snr_large_capacity_stage_authorization(
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            output_root=OUTPUT_ROOT,
            approved_by="user",
            approved_at="now",
            source_instruction="explicit goal",
        )


def test_stage_authorization_rejects_implicit_launch_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered["next_stage"]["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="authorization differs"):
        validate_terminal_snr_large_capacity_stage_authorization(
            altered,
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            expected_output_root=OUTPUT_ROOT,
        )


@pytest.mark.parametrize(
    "section",
    ["selection", "approval_record", "next_stage", "authorization_boundary"],
)
def test_stage_authorization_rejects_unexpected_nested_fields(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    report = _build(monkeypatch)
    altered = copy.deepcopy(report)
    altered[section]["unexpected_authority"] = True
    with pytest.raises(ValueError, match="authorization differs"):
        validate_terminal_snr_large_capacity_stage_authorization(
            altered,
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=EXECUTION_GIT,
            expected_output_root=OUTPUT_ROOT,
        )


def test_stage_authorization_requires_exact_clean_execution_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _patch_preparation(monkeypatch)
    dirty = {**EXECUTION_GIT, "tracked_dirty": True}
    with pytest.raises(ValueError, match="exact clean checkout"):
        build_terminal_snr_large_capacity_stage_authorization(
            preparation=prepared,
            preparation_identity=PREPARATION_ID,
            execution_checkout=dirty,
            output_root=OUTPUT_ROOT,
            approved_by="user",
            approved_at="now",
            source_instruction="explicit goal",
        )
