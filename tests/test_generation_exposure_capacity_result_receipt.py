from __future__ import annotations

import copy

import pytest

from cofitok.generation.exposure_capacity_result import (
    RESULT_ROLE,
    RESULT_SCHEMA,
    VALIDATION_RECEIPT_BOUNDARY,
    build_validation_receipt,
    validate_validation_receipt,
)


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": (name.encode("utf-8").hex() + "0" * 64)[:64],
    }


def _git() -> dict[str, object]:
    return {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "analysis/exposure",
        "tracked_dirty": False,
    }


def _result() -> dict[str, object]:
    methods = ("cofitok", "dense_identity")
    return {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization": _identity("authorization"),
        "candidate_gate": _identity("candidate_gate"),
        "preparation": _identity("preparation"),
        "standing_authorization": _identity("standing_authorization"),
        "execution_checkout": _git(),
        "configs": {method: _identity(f"{method}_config") for method in methods},
        "training": {
            method: {
                "report": _identity(f"{method}_training_report"),
                "checkpoint": _identity(f"{method}_checkpoint"),
                "integrity_manifest": _identity(f"{method}_integrity"),
                "latest": _identity(f"{method}_latest"),
                "metrics": _identity(f"{method}_metrics"),
            }
            for method in methods
        },
        "sampling": {
            method: {"report": _identity(f"{method}_sampling_report")}
            for method in methods
        },
        "quality": {
            method: {"report": _identity(f"{method}_generation_metrics")}
            for method in methods
        },
        "class_fidelity": {
            method: {"report": _identity(f"{method}_class_fidelity")}
            for method in methods
        },
        "mechanism": {
            method: {"report": _identity(f"{method}_mechanism")}
            for method in methods
        },
        "rollout": {
            method: {"report": _identity(f"{method}_rollout")}
            for method in methods
        },
        "claim_guards": {
            "terminal_hold_preserved": True,
            "generation_advantage_proven": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
        },
    }


def test_validation_receipt_is_content_addressed_and_non_authorizing() -> None:
    result = _result()
    result_identity = _identity("result")

    receipt = build_validation_receipt(
        result=result,
        result_identity=result_identity,
        validator_git=_git(),
    )

    assert receipt["status"] == "pass"
    assert len(str(receipt["validation_basis_sha256"])) == 64
    assert receipt["authorization_boundary"] == VALIDATION_RECEIPT_BOUNDARY
    assert not any(VALIDATION_RECEIPT_BOUNDARY.values())
    assert validate_validation_receipt(
        receipt,
        result=result,
        result_identity=result_identity,
    ) == receipt


def test_validation_receipt_rejects_a_changed_result_identity() -> None:
    result = _result()
    receipt = build_validation_receipt(
        result=result,
        result_identity=_identity("result"),
        validator_git=_git(),
    )

    with pytest.raises(ValueError, match="not reproducible"):
        validate_validation_receipt(
            receipt,
            result=result,
            result_identity=_identity("another_result"),
        )


def test_validation_receipt_rejects_source_evidence_tampering() -> None:
    result = _result()
    receipt = build_validation_receipt(
        result=result,
        result_identity=_identity("result"),
        validator_git=_git(),
    )
    tampered = copy.deepcopy(receipt)
    tampered["source_evidence"]["methods"]["cofitok"]["rollout_stability_report"][
        "sha256"
    ] = "f" * 64

    with pytest.raises(ValueError, match="not reproducible"):
        validate_validation_receipt(
            tampered,
            result=result,
            result_identity=_identity("result"),
        )


def test_validation_receipt_requires_a_clean_validator_and_separately_binds_execution() -> None:
    result = _result()
    dirty = _git()
    dirty["tracked_dirty"] = True
    with pytest.raises(ValueError, match="exact clean checkout"):
        build_validation_receipt(
            result=result,
            result_identity=_identity("result"),
            validator_git=dirty,
        )

    different = _git()
    different["revision"] = "c" * 40
    receipt = build_validation_receipt(
        result=result,
        result_identity=_identity("result"),
        validator_git=different,
    )
    assert receipt["execution_checkout"] == result["execution_checkout"]
    assert receipt["validator_git"] == different


def test_validation_receipt_rejects_a_rewritten_execution_checkout() -> None:
    result = _result()
    receipt = build_validation_receipt(
        result=result,
        result_identity=_identity("result"),
        validator_git=_git(),
    )
    receipt["execution_checkout"]["revision"] = "c" * 40

    with pytest.raises(ValueError, match="not reproducible"):
        validate_validation_receipt(
            receipt,
            result=result,
            result_identity=_identity("result"),
        )
