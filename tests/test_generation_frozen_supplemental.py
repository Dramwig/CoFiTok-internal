from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cofitok.generation.distribution_support import (
    STABILITY_DISTRIBUTION_SUPPORT_ROLE,
    STABILITY_DISTRIBUTION_SUPPORT_SCHEMA_VERSION,
)
from cofitok.generation.frozen_supplemental import (
    FROZEN_POSTEVAL_VERIFICATION_ROLE,
    FROZEN_SUPPLEMENTAL_ROLE,
    build_frozen_posteval_verification,
    build_frozen_stability_supplemental_qualification,
)
from cofitok.reporting import file_sha256
from scripts import build_generation_stability_frozen_supplemental as cli


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
FROZEN_REVISION = "2" * 40
FROZEN_BRANCH = "scale/generation-stability-50k-posteval-v4"
SUPPLEMENTAL_REVISION = "3" * 40
SUPPLEMENTAL_BRANCH = "scale/generation-large-capacity"


def _git(revision: str = SUPPLEMENTAL_REVISION) -> dict:
    return {
        "revision": revision,
        "branch": SUPPLEMENTAL_BRANCH,
        "tracked_dirty": False,
    }


def _identity(label: str) -> dict:
    return {
        "path": f"/evidence/{label}.json",
        "bytes": 100,
        "sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(),
    }


def _posteval_status() -> dict:
    return {
        "schema_version": 1,
        "role": "generation_stability_50k_posteval_waiter",
        "status": "pass",
        "detail": "formal_ema_postevaluation_completed",
        "child_exit_code": 0,
        "expected": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": FROZEN_REVISION,
            "evaluation_branch": FROZEN_BRANCH,
            "formal_300k_allowed": False,
        },
        "updated_at": "2026-08-03T00:00:00+00:00",
    }


def _promotion_gate(*, passed: bool = True) -> dict:
    return {
        "schema_version": 2,
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "status": "pass" if passed else "fail",
        "decision": "promote_to_full_imagenet256" if passed else "hold",
        "provenance_contract": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": FROZEN_REVISION,
            "evaluation_branch": FROZEN_BRANCH,
        },
    }


def _distribution(gate_source: dict, *, passed: bool = True) -> dict:
    return {
        "schema_version": STABILITY_DISTRIBUTION_SUPPORT_SCHEMA_VERSION,
        "role": STABILITY_DISTRIBUTION_SUPPORT_ROLE,
        "status": "pass" if passed else "fail",
        "sources": {
            "promotion_gate": gate_source,
            "cofitok_generation": _identity("cofitok_generation"),
            "dense_generation": _identity("dense_generation"),
        },
        "builder_git": _git(),
        "decision_boundary": {
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        },
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_rollout_stability_qualification": False,
            "full_training_launch_allowed": False,
        },
    }


def _rollout(*, passed: bool = True) -> dict:
    return {
        "schema_version": 2,
        "status": "pass" if passed else "fail",
        "protocol": {
            "weights": "ema",
            "checkpoint_step": 50_000,
            "checkpoint_evaluated_images": 1_024,
            "checkpoint_timestep": 500,
            "rollout": {
                "num_images": 64,
                "batch_size": 8,
                "sample_steps": 100,
                "guidance_scale": 1.5,
                "teacher_guidance_scale": 1.0,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "clip_x0": True,
                "precision": "bf16",
                "seed": 2029,
            },
        },
        "identity": {
            "training_git_revision": TRAINING_REVISION,
            "evaluation_git_revision": SUPPLEMENTAL_REVISION,
            "evaluation_git_branch": SUPPLEMENTAL_BRANCH,
            "cofitok_checkpoint_sha256": "a" * 64,
            "dense_checkpoint_sha256": "b" * 64,
        },
        "sources": {
            name: _identity(name)
            for name in (
                "cofitok_training",
                "dense_training",
                "cofitok_checkpoint",
                "dense_checkpoint",
                "cofitok_rollout",
                "dense_rollout",
            )
        },
    }


def _inputs(*, base_pass: bool = True, distribution_pass: bool = True, rollout_pass: bool = True) -> dict:
    gate_source = _identity("promotion_gate")
    posteval = build_frozen_posteval_verification(
        posteval_status=_posteval_status(),
        source=_identity("posteval_status"),
        verifier_git=_git(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=FROZEN_REVISION,
        expected_evaluation_branch=FROZEN_BRANCH,
    )
    return {
        "promotion_gate": _promotion_gate(passed=base_pass),
        "posteval_verification": posteval,
        "distribution_support": _distribution(
            gate_source, passed=distribution_pass
        ),
        "rollout_stability": _rollout(passed=rollout_pass),
        "sources": {
            "promotion_gate": gate_source,
            "posteval_verification": _identity("posteval_verification"),
            "distribution_support": _identity("distribution_support"),
            "rollout_stability": _identity("rollout_stability"),
        },
        "builder_git": _git(),
        "expected_training_revision": TRAINING_REVISION,
        "expected_training_branch": TRAINING_BRANCH,
        "expected_frozen_evaluation_revision": FROZEN_REVISION,
        "expected_frozen_evaluation_branch": FROZEN_BRANCH,
        "expected_supplemental_revision": SUPPLEMENTAL_REVISION,
        "expected_supplemental_branch": SUPPLEMENTAL_BRANCH,
    }


def test_frozen_posteval_verification_requires_exact_completed_waiter() -> None:
    report = build_frozen_posteval_verification(
        posteval_status=_posteval_status(),
        source=_identity("posteval_status"),
        verifier_git=_git(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=FROZEN_REVISION,
        expected_evaluation_branch=FROZEN_BRANCH,
    )
    assert report["role"] == FROZEN_POSTEVAL_VERIFICATION_ROLE
    assert report["claim_boundary"]["full_training_launch_allowed"] is False

    running = _posteval_status()
    running["status"] = "running"
    with pytest.raises(ValueError, match="not an exact completed run"):
        build_frozen_posteval_verification(
            posteval_status=running,
            source=_identity("posteval_status"),
            verifier_git=_git(),
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
            expected_evaluation_revision=FROZEN_REVISION,
            expected_evaluation_branch=FROZEN_BRANCH,
        )


def test_frozen_supplemental_combines_all_quality_checks_without_authorizing() -> None:
    report = build_frozen_stability_supplemental_qualification(**_inputs())

    assert report["role"] == FROZEN_SUPPLEMENTAL_ROLE
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["claim_boundary"] == {
        "supplemental_non_authorizing": True,
        "replaces_generation_gate": False,
        "replaces_readiness": False,
        "scaling_authorization_evaluated": False,
        "full_training_launch_allowed": False,
    }


@pytest.mark.parametrize(
    "field",
    ("base_pass", "distribution_pass", "rollout_pass"),
)
def test_frozen_supplemental_holds_when_any_quality_check_fails(field: str) -> None:
    kwargs = {field: False}
    report = build_frozen_stability_supplemental_qualification(**_inputs(**kwargs))
    assert report["status"] == "hold"
    assert report["checks"]["all_supplemental_quality_checks_passed"] is False
    assert report["claim_boundary"]["full_training_launch_allowed"] is False


def test_frozen_supplemental_rejects_weakened_rollout_protocol() -> None:
    inputs = _inputs()
    inputs["rollout_stability"] = deepcopy(inputs["rollout_stability"])
    inputs["rollout_stability"]["protocol"]["rollout"]["sample_steps"] = 50
    with pytest.raises(ValueError, match="rollout-stability supplemental contract"):
        build_frozen_stability_supplemental_qualification(**inputs)


def _write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def test_bound_cli_rehashes_nested_supplemental_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = {}
    for name in (
        "cofitok_generation",
        "dense_generation",
        "cofitok_training",
        "dense_training",
        "cofitok_checkpoint",
        "dense_checkpoint",
        "cofitok_rollout",
        "dense_rollout",
    ):
        nested[name] = _write_json(tmp_path / "nested" / f"{name}.json", {"name": name})
    posteval_path = tmp_path / "posteval_waiter.json"
    posteval_source = _write_json(posteval_path, _posteval_status())
    posteval_verification = build_frozen_posteval_verification(
        posteval_status=_posteval_status(),
        source=posteval_source,
        verifier_git=_git(),
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_evaluation_revision=FROZEN_REVISION,
        expected_evaluation_branch=FROZEN_BRANCH,
    )
    gate_path = tmp_path / "promotion_gate.json"
    gate_source = _write_json(gate_path, _promotion_gate())
    distribution = _distribution(gate_source)
    distribution["sources"]["cofitok_generation"] = nested["cofitok_generation"]
    distribution["sources"]["dense_generation"] = nested["dense_generation"]
    rollout = _rollout()
    for name in rollout["sources"]:
        rollout["sources"][name] = nested[name]
    paths = {
        "promotion_gate": gate_path,
        "posteval_verification": tmp_path / "posteval_verification.json",
        "distribution_support": tmp_path / "distribution_support.json",
        "rollout_stability": tmp_path / "rollout_stability.json",
    }
    _write_json(paths["posteval_verification"], posteval_verification)
    _write_json(paths["distribution_support"], distribution)
    _write_json(paths["rollout_stability"], rollout)
    args = SimpleNamespace(
        **paths,
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        expected_frozen_evaluation_revision=FROZEN_REVISION,
        expected_frozen_evaluation_branch=FROZEN_BRANCH,
        expected_supplemental_revision=SUPPLEMENTAL_REVISION,
        expected_supplemental_branch=SUPPLEMENTAL_BRANCH,
    )
    monkeypatch.setattr(cli, "git_provenance", lambda project: _git())

    report = cli.build_bound_report(args)
    assert report["status"] == "pass"

    Path(nested["cofitok_rollout"]["path"]).write_text("changed", encoding="ascii")
    with pytest.raises(ValueError, match="rollout-stability source changed"):
        cli.build_bound_report(args)


def test_frozen_supplemental_runbook_is_diagnostic_only_and_resumable() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "artifacts/runbooks/generation_stability_frozen_50k_supplemental_after_posteval.sh"
    ).read_text(encoding="utf-8")

    assert source.count("scripts/run_generation_stage_once.py") == 5
    assert source.count("scripts/evaluate_generation_checkpoint.py") == 2
    assert source.count("scripts/evaluate_generation_rollout_stability.py") == 2
    assert "scripts/validate_generation_stability_frozen_posteval.py" in source
    assert "scripts/build_generation_stability_distribution_support.py" in source
    assert "scripts/build_generation_stability_frozen_supplemental.py" in source
    assert source.index("validate_generation_stability_frozen_posteval.py") < source.index(
        "nvidia-smi"
    )
    assert "--sample-steps 100" in source
    assert "--weights ema" in source
    assert "--num-images 64" in source
    assert "--num-images 1024" in source
    assert "--require-pass" not in source
    assert "train_generation.py" not in source
    assert "full_training_launch_allowed=true" not in source
