from __future__ import annotations

from typing import Any

from cofitok.generation.distribution_support import (
    STABILITY_DISTRIBUTION_SUPPORT_ROLE,
    STABILITY_DISTRIBUTION_SUPPORT_SCHEMA_VERSION,
)


FROZEN_POSTEVAL_VERIFICATION_SCHEMA_VERSION = 1
FROZEN_POSTEVAL_VERIFICATION_ROLE = (
    "generation_stability_frozen_posteval_verification"
)
FROZEN_SUPPLEMENTAL_SCHEMA_VERSION = 1
FROZEN_SUPPLEMENTAL_ROLE = "generation_stability_frozen_supplemental_qualification"
POSTEVAL_WAITER_ROLE = "generation_stability_50k_posteval_waiter"


def _sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _git_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and len(str(value.get("revision", ""))) == 40
        and all(
            character in "0123456789abcdef"
            for character in str(value.get("revision", ""))
        )
        and bool(str(value.get("branch", "")))
        and value.get("tracked_dirty") is False
    )


def _source_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(str(value.get("path", "")))
        and isinstance(value.get("bytes"), int)
        and not isinstance(value.get("bytes"), bool)
        and int(value["bytes"]) > 0
        and _sha256(value.get("sha256"))
    )


def _expected_provenance(
    *,
    training_revision: str,
    training_branch: str,
    frozen_evaluation_revision: str,
    frozen_evaluation_branch: str,
) -> dict[str, str]:
    return {
        "training_revision": training_revision,
        "training_branch": training_branch,
        "evaluation_revision": frozen_evaluation_revision,
        "evaluation_branch": frozen_evaluation_branch,
    }


def build_frozen_posteval_verification(
    *,
    posteval_status: dict[str, Any],
    source: dict[str, Any],
    verifier_git: dict[str, Any],
    expected_training_revision: str,
    expected_training_branch: str,
    expected_evaluation_revision: str,
    expected_evaluation_branch: str,
) -> dict[str, Any]:
    if not _source_identity(source):
        raise ValueError("post-evaluation status source identity is invalid")
    if not _git_identity(verifier_git):
        raise ValueError("post-evaluation verifier Git identity is not clean")
    expected = posteval_status.get("expected")
    required = {
        "training_revision": expected_training_revision,
        "training_branch": expected_training_branch,
        "evaluation_revision": expected_evaluation_revision,
        "evaluation_branch": expected_evaluation_branch,
        "formal_300k_allowed": False,
    }
    if (
        posteval_status.get("schema_version") != 1
        or posteval_status.get("role") != POSTEVAL_WAITER_ROLE
        or posteval_status.get("status") != "pass"
        or posteval_status.get("detail") != "formal_ema_postevaluation_completed"
        or int(posteval_status.get("child_exit_code", -1)) != 0
        or not isinstance(expected, dict)
        or any(expected.get(name) != value for name, value in required.items())
    ):
        raise ValueError("frozen stability post-evaluation is not an exact completed run")
    return {
        "schema_version": FROZEN_POSTEVAL_VERIFICATION_SCHEMA_VERSION,
        "role": FROZEN_POSTEVAL_VERIFICATION_ROLE,
        "status": "verified",
        "source": source,
        "verifier_git": verifier_git,
        "expected": required,
        "posteval": {
            "status": "pass",
            "detail": "formal_ema_postevaluation_completed",
            "child_exit_code": 0,
            "updated_at": posteval_status.get("updated_at"),
        },
        "claim_boundary": {
            "supplemental_execution_allowed": True,
            "replaces_postevaluation": False,
            "full_training_launch_allowed": False,
        },
    }


def _validate_posteval_verification(
    report: dict[str, Any],
    *,
    source: dict[str, Any],
    expected: dict[str, Any],
    expected_verifier_git: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != FROZEN_POSTEVAL_VERIFICATION_SCHEMA_VERSION
        or report.get("role") != FROZEN_POSTEVAL_VERIFICATION_ROLE
        or report.get("status") != "verified"
        or not _source_identity(source)
        or report.get("source") != source
        or report.get("expected") != expected
        or report.get("verifier_git") != expected_verifier_git
        or not _git_identity(report.get("verifier_git"))
        or report.get("claim_boundary")
        != {
            "supplemental_execution_allowed": True,
            "replaces_postevaluation": False,
            "full_training_launch_allowed": False,
        }
    ):
        raise ValueError("frozen post-evaluation verification contract differs")


def _validate_distribution_support(
    report: dict[str, Any],
    *,
    promotion_gate_source: dict[str, Any],
) -> bool:
    boundary = report.get("claim_boundary")
    decision = report.get("decision_boundary")
    if (
        report.get("schema_version") != STABILITY_DISTRIBUTION_SUPPORT_SCHEMA_VERSION
        or report.get("role") != STABILITY_DISTRIBUTION_SUPPORT_ROLE
        or report.get("status") not in {"pass", "fail"}
        or report.get("sources", {}).get("promotion_gate") != promotion_gate_source
        or not _git_identity(report.get("builder_git"))
        or not isinstance(boundary, dict)
        or boundary.get("supplemental_non_authorizing") is not True
        or boundary.get("replaces_generation_gate") is not False
        or boundary.get("replaces_rollout_stability_qualification") is not False
        or boundary.get("full_training_launch_allowed") is not False
        or not isinstance(decision, dict)
        or decision.get("scaling_authorization_evaluated") is not False
        or decision.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("distribution-support supplemental contract differs")
    return report["status"] == "pass"


def _validate_rollout_stability(
    report: dict[str, Any],
    *,
    expected_training_revision: str,
    expected_supplemental_revision: str,
    expected_supplemental_branch: str,
) -> bool:
    protocol = report.get("protocol")
    rollout = protocol.get("rollout", {}) if isinstance(protocol, dict) else {}
    identity = report.get("identity")
    sources = report.get("sources")
    if (
        report.get("schema_version") != 2
        or report.get("status") not in {"pass", "fail"}
        or not isinstance(protocol, dict)
        or protocol.get("weights") != "ema"
        or int(protocol.get("checkpoint_step", -1)) != 50_000
        or int(protocol.get("checkpoint_evaluated_images", -1)) != 1_024
        or int(protocol.get("checkpoint_timestep", -1)) != 500
        or not isinstance(rollout, dict)
        or int(rollout.get("num_images", -1)) != 64
        or int(rollout.get("sample_steps", -1)) != 100
        or int(rollout.get("seed", -1)) != 2029
        or float(rollout.get("guidance_scale", -1.0)) != 1.5
        or float(rollout.get("guidance_rescale", -1.0)) != 0.0
        or float(rollout.get("teacher_guidance_scale", -1.0)) != 1.0
        or rollout.get("cfg_batch_mode") != "batched"
        or rollout.get("clip_x0") is not True
        or rollout.get("precision") != "bf16"
        or not isinstance(identity, dict)
        or identity.get("training_git_revision") != expected_training_revision
        or identity.get("evaluation_git_revision") != expected_supplemental_revision
        or identity.get("evaluation_git_branch") != expected_supplemental_branch
        or not _sha256(identity.get("cofitok_checkpoint_sha256"))
        or not _sha256(identity.get("dense_checkpoint_sha256"))
        or not isinstance(sources, dict)
        or set(sources)
        != {
            "cofitok_training",
            "dense_training",
            "cofitok_checkpoint",
            "dense_checkpoint",
            "cofitok_rollout",
            "dense_rollout",
        }
        or not all(_source_identity(value) for value in sources.values())
    ):
        raise ValueError("EMA rollout-stability supplemental contract differs")
    return report["status"] == "pass"


def build_frozen_stability_supplemental_qualification(
    *,
    promotion_gate: dict[str, Any],
    posteval_verification: dict[str, Any],
    distribution_support: dict[str, Any],
    rollout_stability: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    builder_git: dict[str, Any],
    expected_training_revision: str,
    expected_training_branch: str,
    expected_frozen_evaluation_revision: str,
    expected_frozen_evaluation_branch: str,
    expected_supplemental_revision: str,
    expected_supplemental_branch: str,
) -> dict[str, Any]:
    expected_sources = {
        "promotion_gate",
        "posteval_verification",
        "distribution_support",
        "rollout_stability",
    }
    if set(sources) != expected_sources or not all(
        _source_identity(value) for value in sources.values()
    ):
        raise ValueError("frozen supplemental source identities are incomplete")
    expected_builder = {
        "revision": expected_supplemental_revision,
        "branch": expected_supplemental_branch,
        "tracked_dirty": False,
    }
    if builder_git != expected_builder or not _git_identity(builder_git):
        raise ValueError("frozen supplemental builder Git identity differs")
    provenance = _expected_provenance(
        training_revision=expected_training_revision,
        training_branch=expected_training_branch,
        frozen_evaluation_revision=expected_frozen_evaluation_revision,
        frozen_evaluation_branch=expected_frozen_evaluation_branch,
    )
    if (
        promotion_gate.get("stage") != "scaling"
        or promotion_gate.get("source_profile") != "stability_scaling"
        or int(promotion_gate.get("schema_version", -1)) not in {2, 3, 4}
        or promotion_gate.get("status") not in {"pass", "fail"}
        or promotion_gate.get("provenance_contract") != provenance
    ):
        raise ValueError("frozen promotion gate provenance differs")
    posteval_expected = {
        **provenance,
        "formal_300k_allowed": False,
    }
    _validate_posteval_verification(
        posteval_verification,
        source=posteval_verification.get("source"),
        expected=posteval_expected,
        expected_verifier_git=expected_builder,
    )
    if sources["promotion_gate"] != distribution_support.get("sources", {}).get(
        "promotion_gate"
    ):
        raise ValueError("distribution-support gate source differs")
    distribution_passed = _validate_distribution_support(
        distribution_support,
        promotion_gate_source=sources["promotion_gate"],
    )
    rollout_passed = _validate_rollout_stability(
        rollout_stability,
        expected_training_revision=expected_training_revision,
        expected_supplemental_revision=expected_supplemental_revision,
        expected_supplemental_branch=expected_supplemental_branch,
    )
    base_gate_passed = (
        promotion_gate.get("status") == "pass"
        and promotion_gate.get("decision") == "promote_to_full_imagenet256"
    )
    passed = base_gate_passed and distribution_passed and rollout_passed
    return {
        "schema_version": FROZEN_SUPPLEMENTAL_SCHEMA_VERSION,
        "role": FROZEN_SUPPLEMENTAL_ROLE,
        "status": "pass" if passed else "hold",
        "decision": "supplemental_quality_complete" if passed else "hold",
        "sources": sources,
        "builder_git": builder_git,
        "provenance_contract": {
            **provenance,
            "supplemental_revision": expected_supplemental_revision,
            "supplemental_branch": expected_supplemental_branch,
        },
        "checks": {
            "base_gate_passed": base_gate_passed,
            "distribution_support_passed": distribution_passed,
            "ema_rollout_stability_passed": rollout_passed,
            "all_supplemental_quality_checks_passed": passed,
        },
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_readiness": False,
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        },
    }
