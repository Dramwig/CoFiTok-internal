from __future__ import annotations

import copy
from typing import Any

import pytest

from cofitok.environment import runtime_environment_sha256
from cofitok.generation import (
    EPSILON_STABILITY_EXECUTION_ACTIONS,
    EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS,
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    EPSILON_STABILITY_PREPARATION_SCHEMA,
    EPSILON_STABILITY_SCOPE,
    EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE,
    build_epsilon_stability_execution_authorization,
    build_epsilon_stability_preparation,
    build_epsilon_stability_sampling_design,
    build_epsilon_stability_user_authorization_receipt,
    validate_epsilon_stability_user_authorization_receipt,
)


def _identity(name: str, fill: str = "a") -> dict[str, Any]:
    return {
        "path": f"/bound/{name}",
        "bytes": 1,
        "sha256": fill * 64,
    }


def _preparation(design_identity: dict[str, Any]) -> dict[str, Any]:
    methods = {}
    for method, prefix_budget in (("cofitok", 8), ("dense_identity", 1)):
        methods[method] = {
            "checkpoint_step": 100_000,
            "prefix_budget": prefix_budget,
            "checkpoint": _identity(f"{method}.pt"),
            "integrity_sidecar": _identity(f"{method}.pt.integrity.json"),
            "latest": _identity(f"{method}.latest.json"),
            "training_report": _identity(f"{method}.training_report.json"),
        }
    source_bindings = {
        name: _identity(f"{name}.json")
        for name in EPSILON_STABILITY_EXECUTION_SOURCE_BINDINGS
    }
    return {
        "schema": EPSILON_STABILITY_PREPARATION_SCHEMA,
        "status": "prepared",
        "execution_ready": False,
        "scope": EPSILON_STABILITY_SCOPE,
        "design_identity": design_identity,
        "terminal_route_receipt_identity": _identity("route.json"),
        "repair_git": {
            "revision": "b" * 40,
            "tree": "c" * 40,
            "branch": "scale/generation-quality-repair-epsilon-stability-v1",
            "tracked_dirty": False,
        },
        "random_stream": {
            "seed": 2_026_082_401,
            "start_index": 0,
            "namespace": "epsilon-stability-v1-20260824",
            "fresh": True,
        },
        "output_root": (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "matched_epsilon_stability_1k_v1"
        ),
        "output_root_non_overlapping": True,
        "evaluator_git": {
            "revision": "b" * 40,
            "branch": "scale/generation-quality-repair-epsilon-stability-v1",
            "tracked_dirty": False,
        },
        "runtime_environment_sha256s": {
            role: character * 64
            for role, character in zip(
                ("sampling", "metrics", "class_fidelity", "artifact"),
                "def0",
                strict=True,
            )
        },
        "training_runtime_environment_sha256s": ["1" * 64],
        "real_set": {
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": "2" * 64,
            "root": "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val",
            "image_count": 50_000,
        },
        "methods": methods,
        "source_bindings": source_bindings,
        "authorization_boundary": dict(
            EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
        ),
    }


def test_user_authorization_receipt_is_exact_and_non_authorizing() -> None:
    receipt = build_epsilon_stability_user_authorization_receipt(
        task_id="019fda19-c552-78b1-8460-463ee5ae9c7a",
        user_message=EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE,
    )

    assert validate_epsilon_stability_user_authorization_receipt(receipt) == receipt
    assert receipt["authorized_actions"] == EPSILON_STABILITY_EXECUTION_ACTIONS
    assert receipt["authorized_actions"]["matched_1000_sample_sampling"] is True
    assert receipt["authorized_actions"]["associated_non_formal_evaluation"] is True
    assert all(
        receipt["authorized_actions"][name] is False
        for name in (
            "independent_10000_confirmation",
            "training",
            "full_300k",
            "promotion",
            "inference_export",
            "release",
            "process_signals",
        )
    )

    changed = copy.deepcopy(receipt)
    changed["authorized_actions"]["training"] = True
    with pytest.raises(ValueError, match="authorization contract differs"):
        validate_epsilon_stability_user_authorization_receipt(changed)


def test_execution_gate_binds_preparation_latest_and_exact_actions() -> None:
    design = build_epsilon_stability_sampling_design()
    design_identity = _identity("design.json")
    preparation = _preparation(design_identity)
    user = build_epsilon_stability_user_authorization_receipt(
        task_id="019fda19-c552-78b1-8460-463ee5ae9c7a",
        user_message=EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE,
    )

    gate = build_epsilon_stability_execution_authorization(
        preparation=preparation,
        preparation_identity=_identity("preparation.json"),
        design=design,
        user_authorization=user,
        user_authorization_identity=_identity("user_authorization.json"),
    )

    assert gate["status"] == "approved"
    assert gate["scope"] == EPSILON_STABILITY_SCOPE
    assert gate["methods"]["cofitok"]["latest"] == (
        preparation["methods"]["cofitok"]["latest"]
    )
    assert gate["authorized_actions"] == EPSILON_STABILITY_EXECUTION_ACTIONS
    assert gate["output_root_non_overlapping"] is True

    missing_latest = copy.deepcopy(preparation)
    del missing_latest["methods"]["dense_identity"]["latest"]
    with pytest.raises(ValueError, match="execution source is malformed"):
        build_epsilon_stability_execution_authorization(
            preparation=missing_latest,
            preparation_identity=_identity("preparation.json"),
            design=design,
            user_authorization=user,
            user_authorization_identity=_identity("user_authorization.json"),
        )


def test_preparation_replays_terminal_checkpoint_and_runtime_sources() -> None:
    design = build_epsilon_stability_sampling_design()
    design_identity = _identity("design.json")
    reconciliation_identity = _identity("reconciliation.json")
    quality_identity = _identity("quality_bridge_result.json")
    decision_identity = _identity("decision.json")
    next_stage = {
        "id": "prepare_matched_100k_epsilon_stability_sampling_diagnostic",
        "execution_ready": False,
        "gpu_execution_allowed": False,
        "sampling_launch_allowed": False,
        "training_launch_allowed": False,
        "full_300k_launch_allowed": False,
    }
    boundary = {
        "new_source_bound_execution_gate_required": True,
        "sampling_launch_allowed": False,
        "training_launch_allowed": False,
        "process_signals_allowed": False,
    }
    repair_git = {
        "revision": "b" * 40,
        "tree": "c" * 40,
        "branch": "scale/generation-quality-repair-epsilon-stability-v1",
        "tracked_dirty": False,
    }
    dataset_identity = {
        "path": "/dataset/metadata/image_manifest.jsonl",
        "bytes": 123,
        "sha256": "3" * 64,
    }
    dataset_provenance = {
        "status": "pass",
        "formal": True,
        "identity_sha256": "4" * 64,
        "dataset_root": "/dataset",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": 123,
            "sha256": "3" * 64,
        },
    }
    training_git = {
        "revision": "5" * 40,
        "branch": "scale/generation-stability-quality-bridge-100k",
        "dirty": False,
    }
    training_runtime = "6" * 64
    method_sources: dict[str, Any] = {}
    terminal_methods: dict[str, Any] = {}
    for method, prefix_budget, checkpoint_digest in (
        ("cofitok", 8, "7" * 64),
        ("dense_identity", 1, "8" * 64),
    ):
        checkpoint = {
            "path": f"/runs/{method}/checkpoint_step_00100000.pt",
            "bytes": 1_000,
            "sha256": checkpoint_digest,
        }
        sidecar_identity = _identity(f"{method}.sidecar.json")
        latest_identity = _identity(f"{method}.latest.json")
        training_identity = _identity(f"{method}.training.json")
        sidecar = {
            "schema_version": 1,
            "step": 100_000,
            "checkpoint": "checkpoint_step_00100000.pt",
            "checkpoint_bytes": 1_000,
            "checkpoint_sha256": checkpoint_digest,
            "dataset_identity_sha256": "4" * 64,
            "runtime_environment_sha256": training_runtime,
            "git_revision": training_git["revision"],
            "git_branch": training_git["branch"],
        }
        latest = {
            **sidecar,
            "integrity_manifest": f"{method}.sidecar.json",
        }
        latest["checkpoint"] = "checkpoint_step_00100000.pt"
        training = {
            "training_complete": True,
            "completed_steps": 100_000,
            "target_steps": 100_000,
            "latest_checkpoint": latest,
            "git": training_git,
            "dataset_provenance": dataset_provenance,
            "runtime_environment_sha256": training_runtime,
        }
        method_sources[method] = {
            "checkpoint": checkpoint,
            "integrity_sidecar": {
                "identity": sidecar_identity,
                "payload": sidecar,
            },
            "latest": {"identity": latest_identity, "payload": latest},
            "training_report": {
                "identity": training_identity,
                "payload": training,
            },
        }
        terminal_methods[method] = {
            "checkpoint_payload": checkpoint,
            "checkpoint_sidecar": sidecar_identity,
            "latest": latest_identity,
        }
        assert design["matched_methods"][method]["prefix_budget"] == prefix_budget

    decision = {
        "schema_version": 1,
        "role": "generation_100k_post_reconciliation_experiment_decision",
        "status": "completed",
        "operational_status": "pass",
        "generation_advantage_proven": False,
        "recommended_next_stage": next_stage,
        "authorization_boundary": boundary,
        "source_evidence": {
            "cross_protocol_reconciliation": reconciliation_identity,
            "quality_bridge_result": quality_identity,
            "terminal_methods": terminal_methods,
        },
    }
    verification = {
        "schema_version": 1,
        "role": "generation_100k_post_reconciliation_decision_verification",
        "status": "verified",
        "decision": decision_identity,
        "recommended_next_stage": next_stage,
        "authorization_boundary": boundary,
    }
    runtime_environment = {"schema_version": 1, "device": {"type": "cpu"}}
    runtime_digest = runtime_environment_sha256(runtime_environment)
    runtime_binding = {
        "schema": "cofitok_matched_epsilon_stability_runtime_binding_v1",
        "status": "pass",
        "runtime_environments": {
            role: runtime_environment
            for role in ("sampling", "metrics", "class_fidelity", "artifact")
        },
        "runtime_environment_sha256s": {
            role: runtime_digest
            for role in ("sampling", "metrics", "class_fidelity", "artifact")
        },
    }
    real_set = {
        "digest_schema": "cofitok_image_tree_sha256_v1",
        "sha256": "9" * 64,
        "root": "/dataset/val",
        "image_count": 50_000,
    }
    evaluator_manifest = {
        "schema": "cofitok_matched_epsilon_stability_evaluator_manifest_v1",
        "status": "pass",
        "git": repair_git,
        "sources": {"evaluator.py": _identity("evaluator.py")},
    }
    classifier = {
        "path": "/weights/resnet50.pth",
        "bytes": 200,
        "sha256": "a" * 64,
    }
    classifier_report = {
        "classifier": {
            "weights_path": classifier["path"],
            "weights_bytes": classifier["bytes"],
            "weights_sha256": classifier["sha256"],
        }
    }

    preparation = build_epsilon_stability_preparation(
        design_source={"identity": design_identity, "payload": design},
        decision_source={"identity": decision_identity, "payload": decision},
        decision_verification_source={
            "identity": _identity("verification.json"),
            "payload": verification,
        },
        reconciliation_source={
            "identity": reconciliation_identity,
            "payload": {
                "role": "generation_100k_cross_protocol_reconciliation",
                "status": "completed",
                "operational_status": "pass",
                "terminal_status": "hold",
            },
        },
        quality_bridge_result_source={
            "identity": quality_identity,
            "payload": {
                "role": "stability_full_data_quality_bridge_result",
                "status": "completed",
                "authorization_boundary": {"report_is_promotion_gate": False},
            },
        },
        training_pair_report_source={
            "identity": _identity("pair_monitor.json"),
            "payload": {"status": "pass", "stage": "complete", "issues": []},
        },
        method_sources=method_sources,
        dataset_identity=dataset_identity,
        real_set_source={
            "identity": _identity("real_set_source.json"),
            "payload": {"real_set": real_set},
        },
        runtime_binding_source={
            "identity": _identity("runtime_binding.json"),
            "payload": runtime_binding,
        },
        evaluator_source={
            "identity": _identity("evaluator_manifest.json"),
            "payload": evaluator_manifest,
        },
        classifier_identity=classifier,
        classifier_report_source={
            "identity": _identity("classifier_report.json"),
            "payload": classifier_report,
        },
        repair_git=repair_git,
        seed=2_026_082_401,
        random_stream_namespace="epsilon-stability-v1-20260824",
        output_root=(
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "matched_epsilon_stability_1k_v1"
        ),
    )

    assert preparation["status"] == "prepared"
    assert preparation["execution_ready"] is False
    assert preparation["methods"]["cofitok"]["latest"] == (
        method_sources["cofitok"]["latest"]["identity"]
    )
    assert preparation["source_bindings"]["cross_protocol_reconciliation"] == (
        reconciliation_identity
    )
    assert preparation["authorization_boundary"] == (
        EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
    )
