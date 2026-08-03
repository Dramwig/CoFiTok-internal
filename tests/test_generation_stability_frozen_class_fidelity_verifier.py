from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from scripts import verify_generation_stability_frozen_class_fidelity as verifier


EVALUATOR_GIT = {
    "revision": "a" * 40,
    "branch": "scale/frozen-class-fidelity",
    "tracked_dirty": False,
}
SAMPLING_GIT = {
    "revision": "b" * 40,
    "branch": "scale/frozen-sampling",
    "tracked_dirty": False,
}


def _metrics(*, top1: float, top5: float) -> dict:
    sample_count = 10_000
    entropy = 6.2
    return {
        "sample_count": sample_count,
        "num_classes": 1000,
        "top1_correct": int(top1 * sample_count),
        "top5_correct": int(top5 * sample_count),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "mean_target_probability": 0.15,
        "target_negative_log_likelihood": 3.0,
        "requested_class_count": 1000,
        "requested_count_min": 10,
        "requested_count_max": 10,
        "predicted_class_count": 800,
        "predicted_class_fraction": 0.8,
        "predicted_class_entropy": entropy,
        "normalized_predicted_class_entropy": entropy / math.log(1000),
    }


def _sampling(*, prefix_budget: int | None = None) -> dict:
    sampling = {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "inference_api": {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        },
        "sampler": "ddim",
        "num_samples": 10_000,
        "start_index": 0,
        "batch_size": 16,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "actual_timesteps": verifier._timesteps(),
        "image_shape": [3, 256, 256],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "class_schedule": "balanced_modulo",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "random_stream": {
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }
    if prefix_budget is not None:
        sampling["prefix_budgets"] = [prefix_budget]
    return sampling


def _raw_report(
    *,
    prefix_budget: int,
    checkpoint_sha256: str,
    sample_set_sha256: str,
    top1: float,
    top5: float,
) -> dict:
    runtime_environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "test"},
    }
    runtime_sha256 = verifier._runtime_environment_sha256(runtime_environment)
    return {
        "schema_version": 2,
        "role": verifier.REPORT_ROLE,
        "status": "completed",
        "protocol": "torchvision_imagenet_class_fidelity",
        "git": EVALUATOR_GIT,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_sha256,
        "classifier": verifier.CLASSIFIER,
        "sample_provenance": {
            "git": SAMPLING_GIT,
            "weights": "ema",
            "checkpoint_sha256": checkpoint_sha256,
            "sample_set_sha256": sample_set_sha256,
            "selected_prefix_budget": prefix_budget,
            "sampling": _sampling(prefix_budget=prefix_budget),
        },
        "parameters": {
            "num_classes": 1000,
            "sample_count": 10_000,
            "target_from_filename": "int(zero_based_png_stem) mod 1000",
        },
        "metrics": _metrics(top1=top1, top5=top5),
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _passing_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path, Path]:
    cofitok = _raw_report(
        prefix_budget=8,
        checkpoint_sha256="c" * 64,
        sample_set_sha256="d" * 64,
        top1=0.21,
        top5=0.42,
    )
    dense = _raw_report(
        prefix_budget=1,
        checkpoint_sha256="e" * 64,
        sample_set_sha256="f" * 64,
        top1=0.22,
        top5=0.43,
    )
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    _write(cofitok_path, cofitok)
    _write(dense_path, dense)

    metric_names = (
        "top1_accuracy",
        "top5_accuracy",
        "mean_target_probability",
        "predicted_class_fraction",
        "normalized_predicted_class_entropy",
    )
    cofitok_metrics = cofitok["metrics"]
    dense_metrics = dense["metrics"]
    contract = {
        "sampling": _sampling(),
        "sampling_git": SAMPLING_GIT,
        "evaluator_git": EVALUATOR_GIT,
        "evaluator_runtime_environment_sha256": cofitok[
            "runtime_environment_sha256"
        ],
        "cofitok_checkpoint_sha256": "c" * 64,
        "dense_checkpoint_sha256": "e" * 64,
        "cofitok_sample_set_sha256": "d" * 64,
        "dense_sample_set_sha256": "f" * 64,
        "cofitok_prefix_budget": 8,
        "dense_prefix_budget": 1,
        "weights": "ema",
        "sample_count_per_method": 10_000,
    }
    report = {
        "schema_version": 2,
        "role": verifier.ROLE,
        "status": "pass",
        "stage": "scaling",
        "git": EVALUATOR_GIT,
        "classifier": verifier.CLASSIFIER,
        "thresholds": verifier.THRESHOLDS,
        "claim_boundary": {
            "class_conditional_quality_evaluated": True,
            "unconditional_distribution_quality_evaluated": False,
            "standalone_generation_quality_claim_allowed": False,
            "full_training_launch_allowed": False,
            "release_authorization_allowed": False,
        },
        "sampling_contract": contract,
        "sources": {
            "cofitok": verifier._identity(cofitok_path),
            "dense_identity": verifier._identity(dense_path),
        },
        "metrics": {
            "cofitok": cofitok_metrics,
            "dense_identity": dense_metrics,
            "cofitok_minus_dense": {
                name: float(cofitok_metrics[name]) - float(dense_metrics[name])
                for name in metric_names
            },
        },
        "checks": verifier._expected_checks(cofitok_metrics, dense_metrics),
    }
    report_path = tmp_path / "qualification.json"
    _write(report_path, report)
    gate = {
        "source_profile": "stability_scaling",
        "provenance_contract": {
            "evaluation_revision": SAMPLING_GIT["revision"],
            "evaluation_branch": SAMPLING_GIT["branch"],
        },
        "gates": [
            {
                "name": "matched_sampling_provenance",
                "passed": True,
                "evidence": {
                    name: contract[name]
                    for name in (
                        "cofitok_checkpoint_sha256",
                        "dense_checkpoint_sha256",
                        "cofitok_sample_set_sha256",
                        "dense_sample_set_sha256",
                    )
                },
            }
        ],
    }
    gate_path = tmp_path / "promotion_gate.json"
    _write(gate_path, gate)
    monkeypatch.setattr(
        verifier,
        "verify_generation_gate_source_reports",
        lambda value: {},
    )
    monkeypatch.setattr(
        verifier,
        "validate_generation_gate_authorization",
        lambda value, expected_stage: {"stage": expected_stage},
    )
    return report, report_path, gate_path


def _verify(report: dict, report_path: Path, gate_path: Path) -> dict:
    return verifier.verify_frozen_class_fidelity_qualification(
        report,
        report_path=report_path,
        expected_report_sha256=verifier.file_sha256(report_path),
        promotion_gate_path=gate_path,
        expected_evaluator_revision=EVALUATOR_GIT["revision"],
        expected_evaluator_branch=EVALUATOR_GIT["branch"],
    )


def test_frozen_class_fidelity_verifier_replays_cross_revision_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_qualification(
        tmp_path, monkeypatch
    )
    evidence = _verify(report, report_path, gate_path)

    assert evidence["class_fidelity_passed"] is True
    assert evidence["sampling_git"] == SAMPLING_GIT
    assert evidence["evaluator_git"] == EVALUATOR_GIT
    assert evidence["required_for_full_training_launch"] is True
    assert evidence["full_training_launch_allowed"] is False


def test_frozen_class_fidelity_verifier_rejects_raw_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_qualification(
        tmp_path, monkeypatch
    )
    Path(report["sources"]["cofitok"]["path"]).write_text(
        '{"replaced":true}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source changed"):
        _verify(report, report_path, gate_path)


def test_frozen_class_fidelity_verifier_rejects_gate_sample_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_qualification(
        tmp_path, monkeypatch
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gates"][0]["evidence"]["cofitok_sample_set_sha256"] = "9" * 64
    _write(gate_path, gate)

    with pytest.raises(ValueError, match="sample identity differs from the gate"):
        _verify(report, report_path, gate_path)


def test_frozen_class_fidelity_verifier_rejects_forged_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_qualification(
        tmp_path, monkeypatch
    )
    raw_path = Path(report["sources"]["cofitok"]["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["runtime_environment"]["device"]["type"] = "forged"
    _write(raw_path, raw)
    report["sources"]["cofitok"] = verifier._identity(raw_path)
    _write(report_path, report)

    with pytest.raises(ValueError, match="raw report contract differs"):
        _verify(report, report_path, gate_path)


def test_frozen_class_fidelity_verifier_recomputes_threshold_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_qualification(
        tmp_path, monkeypatch
    )
    report["checks"][0]["threshold"] = 0.0
    _write(report_path, report)

    with pytest.raises(ValueError, match="checks did not pass"):
        _verify(report, report_path, gate_path)


def test_frozen_class_fidelity_verifier_cli_exposes_source_bindings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["verify-frozen-class-fidelity", "--help"])

    with pytest.raises(SystemExit) as error:
        verifier.main()

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--expected-report-sha256" in help_text
    assert "--promotion-gate" in help_text
    assert "--expected-evaluator-revision" in help_text
    assert "--expected-evaluator-branch" in help_text
