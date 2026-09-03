from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from cofitok.generation.capacity_confirmation_result import (
    RESULT_BOUNDARY,
    RESULT_ROLE,
    RESULT_SCHEMA,
)
from cofitok.generation.capacity_scaling_decision import (
    CAPACITY_SCALING_DECISION_BOUNDARY,
    CAPACITY_SCALING_HOLD_ID,
    CAPACITY_SCALING_RECOMMENDATION_ID,
    build_capacity_scaling_decision,
    validate_capacity_scaling_decision,
)
from cofitok.generation.capacity_screen import ARM_NAMES, ARM_SPECS
from cofitok.generation.exposure_capacity_authorization import identity
from scripts import build_generation_capacity_scaling_decision as builder


ROOT = Path(__file__).resolve().parents[1]
SCREEN_ROOT = "/root/autodl-tmp/CoFiTok/checkpoints/generation/capacity_qualification_v1"
CONFIRMATION_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}
DECISION_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "scale/generation-capacity-full-source-compatible-v1",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": len(name) + 100,
        "sha256": sha256(name.encode("utf-8")).hexdigest(),
    }


def _arm(arm: str) -> dict[str, object]:
    spec = ARM_SPECS[arm]
    run_dir = f"{SCREEN_ROOT}/{arm}"
    checkpoint = {
        "path": f"{run_dir}/checkpoint_step_00010000.pt",
        "bytes": 1_000_000 + int(spec["parameter_count"]),
        "sha256": sha256(f"checkpoint-{arm}".encode()).hexdigest(),
        "step": 10_000,
        "integrity_manifest": {
            "path": f"{run_dir}/checkpoint_step_00010000.pt.integrity.json",
            "bytes": 600,
            "sha256": sha256(f"integrity-{arm}".encode()).hexdigest(),
        },
    }
    config = _identity(f"config-{arm}")
    training_report = _identity(f"training-{arm}")
    screen_training = {
        "stage": spec["recipe_stage"],
        "run_dir": run_dir,
        "training_report": training_report["path"],
        "config": config["path"],
        "configured_steps": 100_000,
        "completed_steps": 10_000,
        "training_complete": False,
        "intentional_partial_stop": True,
        "effective_batch_size": 64,
        "images_seen": 640_000,
        "parameter_count": spec["parameter_count"],
        "runtime_environment_sha256": "e" * 64,
        "dataset_identity_sha256": "f" * 64,
        "checkpoint": checkpoint,
    }
    return {
        "arm": arm,
        "capacity": spec["capacity"],
        "method": spec["method"],
        "execution_git": CONFIRMATION_GIT,
        "sources": {
            "screen_arm_validation": _identity(f"screen-validation-{arm}"),
        },
        "frozen_training": {
            "training_performed": False,
            "config": config,
            "training_report": training_report,
            "checkpoint": checkpoint,
            "screen_training": screen_training,
        },
    }


def _inputs(
    monkeypatch: pytest.MonkeyPatch, *, passed: bool = True
) -> dict[str, object]:
    arm_ids = {arm: _identity(f"confirmation-{arm}") for arm in ARM_NAMES}
    result = {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "scientific_status": "confirmation_pass" if passed else "hold",
        "result_git": CONFIRMATION_GIT,
        "checks": [{"name": "support", "passed": passed}],
        "failed_checks": [] if passed else ["support"],
        "next_stage": {"support_collapse_resolved": passed},
        "source_evidence": {"arm_validations": arm_ids},
        "authorization_boundary": RESULT_BOUNDARY,
    }
    arms = {arm: _arm(arm) for arm in ARM_NAMES}
    monkeypatch.setattr(
        "cofitok.generation.capacity_scaling_decision.validate_capacity_confirmation_result_contract",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cofitok.generation.capacity_scaling_decision.validate_capacity_confirmation_arm_validation",
        lambda value: value,
    )
    return {
        "capacity_confirmation_result": result,
        "capacity_confirmation_result_identity": _identity("confirmation-result"),
        "arm_validations": arms,
        "arm_validation_identities": arm_ids,
        "confirmation_checkout": CONFIRMATION_GIT,
        "decision_git": DECISION_GIT,
    }


def _decision(
    monkeypatch: pytest.MonkeyPatch, *, passed: bool = True
) -> dict[str, object]:
    return build_capacity_scaling_decision(
        **_inputs(monkeypatch, passed=passed)
    )


def test_pass_selects_only_separate_10k_to_50k_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _decision(monkeypatch)
    assert report["next_stage"]["route"] == "capacity_scaling_preparation"
    assert report["next_stage"]["capacity_scaling_preparation_allowed"] is True
    assert report["next_stage"]["training_launch_allowed"] is False
    assert report["next_stage"]["full_300k_launch_allowed"] is False
    assert report["selection"]["resume_from_step"] == 10_000
    assert report["selection"]["stop_after_step"] == 50_000
    assert report["selection"]["output_root"].endswith(
        "/capacity_scaling_50000"
    )
    assert "execution_authorization" not in report
    assert report["authorization_boundary"] == CAPACITY_SCALING_DECISION_BOUNDARY
    evidence = validate_capacity_scaling_decision(
        report,
        expected_decision_revision=DECISION_GIT["revision"],
        expected_decision_tree=DECISION_GIT["tree"],
        expected_decision_branch=DECISION_GIT["branch"],
    )
    assert evidence["capacity_scaling_preparation_allowed"] is True


def test_failed_confirmation_holds_without_gpu_or_training_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _decision(monkeypatch, passed=False)
    assert report["next_stage"]["route"] == "hold"
    assert report["next_stage"]["capacity_scaling_preparation_allowed"] is False
    assert report["next_stage"]["training_launch_allowed"] is False
    assert report["scientific_status"] == "hold"
    assert CAPACITY_SCALING_HOLD_ID.startswith("hold_capacity_scaling")


def test_decision_rejects_confirmation_or_arm_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["capacity_confirmation_result"] = copy.deepcopy(
        inputs["capacity_confirmation_result"]
    )
    inputs["capacity_confirmation_result"]["result_git"] = {
        **CONFIRMATION_GIT,
        "tree": "0" * 40,
    }
    with pytest.raises(ValueError, match="Git identity differs"):
        build_capacity_scaling_decision(**inputs)

    inputs = _inputs(monkeypatch)
    inputs["arm_validation_identities"] = copy.deepcopy(
        inputs["arm_validation_identities"]
    )
    inputs["arm_validation_identities"]["base256_cofitok"] = _identity("other")
    with pytest.raises(ValueError, match="differs from confirmation result"):
        build_capacity_scaling_decision(**inputs)


def test_decision_requires_exact_frozen_step_10k_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(monkeypatch)
    inputs["arm_validations"] = copy.deepcopy(inputs["arm_validations"])
    frozen = inputs["arm_validations"]["base256_cofitok"]["frozen_training"]
    frozen["checkpoint"]["step"] = 10_001
    frozen["screen_training"]["checkpoint"]["step"] = 10_001
    with pytest.raises(ValueError, match="exact step-10K"):
        build_capacity_scaling_decision(**inputs)


def test_contract_rejects_implicit_training_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _decision(monkeypatch)
    altered = copy.deepcopy(report)
    altered["next_stage"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        validate_capacity_scaling_decision(
            altered,
            expected_decision_revision=DECISION_GIT["revision"],
            expected_decision_tree=DECISION_GIT["tree"],
            expected_decision_branch=DECISION_GIT["branch"],
        )


def test_confirmation_result_replay_rejects_nonreproducible_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation = tmp_path / "preparation.json"
    launch = tmp_path / "launch.json"
    arm_paths = {arm: tmp_path / f"{arm}.json" for arm in ARM_NAMES}
    for path in (preparation, launch, *arm_paths.values()):
        path.write_text(json.dumps({"path": path.name}), encoding="utf-8")
    result = {
        "result_git": CONFIRMATION_GIT,
        "source_evidence": {
            "preparation": identity(preparation),
            "launch_receipt": identity(launch),
            "arm_validations": {
                arm: identity(path) for arm, path in arm_paths.items()
            },
        },
    }
    result_path = tmp_path / "capacity_confirmation_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(
        builder, "validate_capacity_confirmation_result_contract", lambda value: value
    )
    monkeypatch.setattr(
        builder, "build_capacity_confirmation_result", lambda **kwargs: copy.deepcopy(result)
    )
    actual, result_id, arms, arm_ids = builder.replay_capacity_confirmation_result(
        result_path,
        expected_sha256=identity(result_path)["sha256"],
        expected_confirmation_checkout=CONFIRMATION_GIT,
    )
    assert actual == result
    assert result_id == identity(result_path)
    assert set(arms) == set(ARM_NAMES)
    assert set(arm_ids) == set(ARM_NAMES)

    monkeypatch.setattr(
        builder,
        "build_capacity_confirmation_result",
        lambda **kwargs: {**result, "tampered": True},
    )
    with pytest.raises(ValueError, match="not reproducible"):
        builder.replay_capacity_confirmation_result(
            result_path,
            expected_sha256=identity(result_path)["sha256"],
            expected_confirmation_checkout=CONFIRMATION_GIT,
        )


@pytest.mark.parametrize(
    "entrypoint",
    (
        "scripts/build_generation_capacity_scaling_decision.py",
        "scripts/verify_generation_capacity_scaling_decision.py",
    ),
)
def test_capacity_scaling_entrypoints_import_under_runbook_pythonpath(
    entrypoint: str,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "capacity" in result.stdout.lower()
    assert "50k" in result.stdout.lower()


def test_recommendation_identifier_is_preparation_only() -> None:
    assert CAPACITY_SCALING_RECOMMENDATION_ID.startswith("prepare_exact")
