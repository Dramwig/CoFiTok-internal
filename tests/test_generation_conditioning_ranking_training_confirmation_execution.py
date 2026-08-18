from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from cofitok.generation.conditioning_ranking_training_confirmation import (
    CLAIM_BOUNDARY,
    EXECUTION_AUTHORIZATION_BOUNDARY,
    EXECUTION_RECEIPT_ROLE,
    EXPECTED_OUTPUT_ROOT,
    IDLE_GPU_EVIDENCE_ROLE,
    RUN_NAMES,
    SOURCE_CLAIM_BOUNDARY,
    SOURCE_METHOD_GATES,
    SOURCE_SAMPLING_GIT,
    SOURCE_SAMPLING_OUTPUT_ROOT,
    SOURCE_SAMPLING_ROLE,
    SOURCE_SAMPLING_STAGE,
    STAGE,
    build_training_confirmation_execution_receipt,
    build_training_confirmation_preparation,
    sampling_validation_route,
    validate_idle_gpu_evidence,
)
from cofitok.inference_replay import file_identity
from scripts import (
    build_generation_conditioning_ranking_training_confirmation_execution_receipt as build_cli,
)
from scripts import (
    verify_generation_conditioning_ranking_training_confirmation_execution_receipt as verify_cli,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "control_cofitok": PROJECT_ROOT
    / "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_k8_probe5k.json",
    "control_dense_identity": PROJECT_ROOT
    / "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_dense_probe5k.json",
    "ranked_cofitok": PROJECT_ROOT
    / "configs/generation/imagenet256_10pct_stability_rgbtail3_rollout_x0_u2_ema_teacher_classrank_k8_confirm5k.json",
    "ranked_dense_identity": PROJECT_ROOT
    / "configs/generation/imagenet256_10pct_stability_rollout_x0_u2_ema_teacher_classrank_dense_confirm5k.json",
}


def _identity(name: str, character: str) -> dict:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": character * 64,
    }


def _git() -> dict:
    return {
        "revision": "b" * 40,
        "branch": "scale/generation-label-ranking-5k-training-confirmation-v1",
        "tracked_dirty": False,
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
            "received_at": "2026-08-19T00:00:00+08:00",
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def _sampling_validation(
    *,
    cofitok_pass: bool = True,
    dense_pass: bool = True,
) -> dict:
    method_passes = {
        "cofitok": cofitok_pass,
        "dense_identity": dense_pass,
    }
    methods = {}
    for name, passed in method_passes.items():
        gates = {gate: True for gate in SOURCE_METHOD_GATES}
        if not passed:
            gates["paired_target_log_probability_significant"] = False
        methods[name] = {"pass": passed, "gates": gates}
    shared = all(method_passes.values())
    if shared:
        action = "prepare_separately_bound_matched_5k_training_recipe_confirmation"
    elif any(method_passes.values()):
        action = "reject_shared_repair_due_method_asymmetry"
    else:
        action = "revise_training_time_semantic_alignment_objective"
    return {
        "schema_version": 1,
        "role": SOURCE_SAMPLING_ROLE,
        "status": "completed",
        "stage": SOURCE_SAMPLING_STAGE,
        "output_root": SOURCE_SAMPLING_OUTPUT_ROOT,
        "git": copy.deepcopy(SOURCE_SAMPLING_GIT),
        "sampling_contract": {
            "sample_count_per_arm": 5_000,
            "sample_run_name": "samples_5000_ddim50_cfg15",
            "methods": {"cofitok": {}, "dense_identity": {}},
        },
        "methods": methods,
        "decision": {
            "method_passes": method_passes,
            "shared_generated_class_alignment_recovery_supported": shared,
            "cofitok_specific_advantage_claim_allowed": False,
            "recommended_next_action": action,
        },
        "claim_boundary": copy.deepcopy(SOURCE_CLAIM_BOUNDARY),
    }


def _configs() -> dict[str, dict]:
    return {
        name: config_to_dict(load_config(path))
        for name, path in CONFIG_PATHS.items()
    }


def _config_identities() -> dict[str, dict]:
    return {
        name: _identity(f"{name}_config", f"{index:x}")
        for index, name in enumerate(RUN_NAMES, start=1)
    }


def _preparation() -> dict:
    configs = _configs()
    return build_training_confirmation_preparation(
        sampling_validation=_sampling_validation(),
        sampling_validation_identity=_identity("sampling", "a"),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=_identity("standing", "b"),
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense_identity"],
        config_identities=_config_identities(),
        parameter_counts={
            "control_cofitok": 100,
            "ranked_cofitok": 100,
            "control_dense_identity": 90,
            "ranked_dense_identity": 90,
        },
        builder_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )


def _idle_evidence() -> dict:
    return {
        "schema_version": 1,
        "role": IDLE_GPU_EVIDENCE_ROLE,
        "status": "pass",
        "stage": STAGE,
        "output_root": EXPECTED_OUTPUT_ROOT,
        "git": _git(),
        "required_consecutive_idle_polls": 5,
        "observations": [
            {
                "poll_index": index + 1,
                "observed_at_unix": 1_000.0 + index,
                "gpu_compute_pids": [],
            }
            for index in range(5)
        ],
    }


def test_sampling_validation_route_selects_only_the_exact_shared_pass() -> None:
    assert sampling_validation_route(_sampling_validation()) == "selected"
    assert (
        sampling_validation_route(
            _sampling_validation(cofitok_pass=False, dense_pass=True)
        )
        == "not_selected"
    )
    assert (
        sampling_validation_route(
            _sampling_validation(cofitok_pass=False, dense_pass=False)
        )
        == "not_selected"
    )
    malformed = _sampling_validation(cofitok_pass=False, dense_pass=True)
    malformed["decision"]["recommended_next_action"] = (
        "prepare_separately_bound_matched_5k_training_recipe_confirmation"
    )
    assert sampling_validation_route(malformed) == "invalid"


def test_execution_receipt_binds_every_source_and_only_authorizes_exact_training() -> None:
    idle = _idle_evidence()
    receipt = build_training_confirmation_execution_receipt(
        preparation=_preparation(),
        preparation_identity=_identity("preparation", "c"),
        sampling_validation=_sampling_validation(),
        sampling_validation_identity=_identity("sampling", "a"),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=_identity("standing", "b"),
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=_identity("idle", "d"),
        config_identities=_config_identities(),
        runbook_identity=_identity("runbook", "e"),
        receipt_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )

    assert receipt["role"] == EXECUTION_RECEIPT_ROLE
    assert receipt["status"] == "authorized"
    assert receipt["authorization_boundary"] == EXECUTION_AUTHORIZATION_BOUNDARY
    assert receipt["authorization_boundary"]["training_allowed"] is True
    assert receipt["authorization_boundary"]["runbook_launch_count_maximum"] == 1
    assert receipt["authorization_boundary"]["sampling_allowed"] is False
    assert receipt["authorization_boundary"][
        "full_100k_or_300k_launch_allowed"
    ] is False
    assert receipt["authorization_boundary"]["release_authorization_allowed"] is False
    assert receipt["claim_boundary"] == CLAIM_BOUNDARY


def test_execution_receipt_rejects_source_config_or_preparation_drift() -> None:
    kwargs = {
        "preparation": _preparation(),
        "preparation_identity": _identity("preparation", "c"),
        "sampling_validation": _sampling_validation(),
        "sampling_validation_identity": _identity("sampling", "a"),
        "standing_authorization": _standing_authorization(),
        "standing_authorization_identity": _identity("standing", "b"),
        "idle_gpu_evidence": _idle_evidence(),
        "idle_gpu_evidence_identity": _identity("idle", "d"),
        "config_identities": _config_identities(),
        "runbook_identity": _identity("runbook", "e"),
        "receipt_git": _git(),
        "expected_revision": _git()["revision"],
        "expected_branch": _git()["branch"],
        "expected_output_root": EXPECTED_OUTPUT_ROOT,
    }
    kwargs["sampling_validation"] = _sampling_validation(
        cofitok_pass=False,
        dense_pass=True,
    )
    with pytest.raises(ValueError, match="did not select"):
        build_training_confirmation_execution_receipt(**kwargs)

    kwargs["sampling_validation"] = _sampling_validation()
    kwargs["config_identities"] = _config_identities()
    kwargs["config_identities"]["ranked_cofitok"] = _identity("wrong", "f")
    with pytest.raises(ValueError, match="preparation differs"):
        build_training_confirmation_execution_receipt(**kwargs)

    kwargs["config_identities"] = _config_identities()
    kwargs["preparation"] = copy.deepcopy(_preparation())
    kwargs["preparation"]["gpu_execution_authorized"] = True
    with pytest.raises(ValueError, match="preparation differs"):
        build_training_confirmation_execution_receipt(**kwargs)


def test_idle_gpu_evidence_requires_five_ordered_empty_observations() -> None:
    validate_idle_gpu_evidence(
        _idle_evidence(),
        expected_git=_git(),
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )
    busy = _idle_evidence()
    busy["observations"][-1]["gpu_compute_pids"] = [123]
    with pytest.raises(ValueError, match="observation differs"):
        validate_idle_gpu_evidence(
            busy,
            expected_git=_git(),
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )
    skipped = _idle_evidence()
    skipped["observations"][-1]["poll_index"] = 7
    with pytest.raises(ValueError, match="observation differs"):
        validate_idle_gpu_evidence(
            skipped,
            expected_git=_git(),
            expected_output_root=EXPECTED_OUTPUT_ROOT,
        )


def test_execution_receipt_cli_builds_and_replays_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_path = tmp_path / "sampling_validation.json"
    standing_path = tmp_path / "standing_authorization.json"
    idle_path = tmp_path / "idle_gpu_evidence.json"
    preparation_path = tmp_path / "preparation.json"
    receipt_path = tmp_path / "execution_receipt.json"
    sampling_path.write_text(
        json.dumps(_sampling_validation(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    standing_path.write_text(
        json.dumps(_standing_authorization(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    idle_path.write_text(
        json.dumps(_idle_evidence(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    configs = _configs()
    config_identities = {
        name: file_identity(path) for name, path in CONFIG_PATHS.items()
    }
    preparation = build_training_confirmation_preparation(
        sampling_validation=_sampling_validation(),
        sampling_validation_identity=file_identity(sampling_path),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=file_identity(standing_path),
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense_identity"],
        config_identities=config_identities,
        parameter_counts={
            "control_cofitok": 100,
            "ranked_cofitok": 100,
            "control_dense_identity": 90,
            "ranked_dense_identity": 90,
        },
        builder_git=_git(),
        expected_revision=_git()["revision"],
        expected_branch=_git()["branch"],
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )
    preparation_path.write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runbook = (
        PROJECT_ROOT
        / "artifacts/runbooks/generation_conditioning_ranking_four_arm_train5k_confirmation_v1.sh"
    )
    monkeypatch.setattr(build_cli, "git_provenance", lambda _root: _git())
    monkeypatch.setattr(verify_cli, "git_provenance", lambda _root: _git())
    monkeypatch.setattr(
        build_cli,
        "replay_sampling_validation",
        lambda path: (
            json.loads(path.read_text(encoding="utf-8")),
            file_identity(path),
        ),
    )
    common = [
        "--preparation",
        str(preparation_path),
        "--expected-preparation-sha256",
        file_identity(preparation_path)["sha256"],
        "--sampling-validation",
        str(sampling_path),
        "--expected-sampling-validation-sha256",
        file_identity(sampling_path)["sha256"],
        "--standing-authorization",
        str(standing_path),
        "--expected-standing-authorization-sha256",
        file_identity(standing_path)["sha256"],
        "--idle-gpu-evidence",
        str(idle_path),
        "--expected-idle-gpu-evidence-sha256",
        file_identity(idle_path)["sha256"],
        "--control-cofitok",
        str(CONFIG_PATHS["control_cofitok"]),
        "--control-dense",
        str(CONFIG_PATHS["control_dense_identity"]),
        "--ranked-cofitok",
        str(CONFIG_PATHS["ranked_cofitok"]),
        "--ranked-dense",
        str(CONFIG_PATHS["ranked_dense_identity"]),
        "--runbook",
        str(runbook),
        "--expected-runbook-sha256",
        file_identity(runbook)["sha256"],
        "--expected-revision",
        _git()["revision"],
        "--expected-branch",
        _git()["branch"],
        "--expected-output-root",
        EXPECTED_OUTPUT_ROOT,
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_training_confirmation_execution_receipt.py",
            *common,
            "--output",
            str(receipt_path),
        ],
    )
    build_cli.main()
    receipt_sha = file_identity(receipt_path)["sha256"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_training_confirmation_execution_receipt.py",
            "--receipt",
            str(receipt_path),
            "--expected-receipt-sha256",
            receipt_sha,
            *common,
        ],
    )
    verify_cli.main()
    assert file_identity(receipt_path)["sha256"] == receipt_sha
