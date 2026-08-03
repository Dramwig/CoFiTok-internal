from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CATEGORIES_SHA256,
    CLASS_FIDELITY_REPORT_ROLE,
    CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
    ClassFidelityAccumulator,
    validate_class_fidelity_qualification,
    validate_class_fidelity_report,
)
from scripts import build_generation_class_fidelity_qualification as qualification
from scripts import evaluate_generation_class_fidelity as evaluator
from scripts import verify_generation_stability_frozen_class_fidelity as frozen_verify
from scripts.audit_large_scale_generation_completion import (
    _class_fidelity_comparison_evidence,
)
from scripts.build_generation_gate_report import (
    _class_fidelity_evidence as gate_class_fidelity_evidence,
)


EXPECTED_GIT = {
    "revision": "a" * 40,
    "branch": "scale/test",
    "tracked_dirty": False,
}


def _metrics(*, top1: float, top5: float, sample_count: int) -> dict:
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
        "requested_count_min": sample_count // 1000,
        "requested_count_max": sample_count // 1000,
        "predicted_class_count": 800,
        "predicted_class_fraction": 0.8,
        "predicted_class_entropy": entropy,
        "normalized_predicted_class_entropy": entropy / math.log(1000),
    }


def _classifier() -> dict:
    return {
        "name": "torchvision_resnet50_imagenet1k_v2",
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_url": evaluator.CLASSIFIER_URL,
        "weights_path": "/checkpoints/resnet50-11ad3fa6.pth",
        "weights_bytes": 102540417,
        "weights_sha256": evaluator.CLASSIFIER_SHA256,
        "num_classes": 1000,
        "categories_sha256": CLASS_FIDELITY_CATEGORIES_SHA256,
        "preprocessing": {
            "resize_size": [232],
            "crop_size": [224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear",
            "antialias": True,
        },
    }


def _report(
    *, method: str, top1: float, top5: float, stage: str = "full"
) -> dict:
    budget = 8 if method == "cofitok" else 1
    sample_count = 50_000 if stage == "full" else 10_000
    sample_steps = 250 if stage == "full" else 100
    runtime_environment = {"schema_version": 1, "device": {"type": "cpu"}}
    return {
        "schema_version": CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
        "role": CLASS_FIDELITY_REPORT_ROLE,
        "status": "completed",
        "protocol": "torchvision_imagenet_class_fidelity",
        "git": EXPECTED_GIT,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha256(runtime_environment),
        "paths": {},
        "classifier": _classifier(),
        "sample_provenance": {
            "git": EXPECTED_GIT,
            "weights": "ema",
            "checkpoint_sha256": ("d" if method == "cofitok" else "e") * 64,
            "sample_set_sha256": ("f" if method == "cofitok" else "0") * 64,
            "selected_prefix_budget": budget,
            "sampling": {
                "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_samples": sample_count,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": sample_steps,
                "num_train_timesteps": 1000,
                "actual_timesteps": select_sampling_timesteps(1000, sample_steps),
                "image_shape": [3, 256, 256],
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "class_schedule": "balanced_modulo",
                "prefix_budgets": [budget],
                "eta": 0.0,
                "clip_x0": True,
                "seed": 0,
                "precision": "bf16",
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
            },
        },
        "parameters": {
            "num_classes": 1000,
            "sample_count": sample_count,
            "target_from_filename": "int(zero_based_png_stem) mod 1000",
        },
        "metrics": _metrics(
            top1=top1,
            top5=top5,
            sample_count=sample_count,
        ),
        "runtime": {"elapsed_seconds": 1.0, "torch_version": torch.__version__},
    }


def _thresholds() -> dict[str, float]:
    return {
        "min_top1": 0.10,
        "min_top5": 0.25,
        "min_predicted_class_fraction": 0.50,
        "min_normalized_predicted_entropy": 0.70,
        "max_top1_regression": 0.05,
        "max_top5_regression": 0.05,
    }


def _passing_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    monkeypatch.setattr(qualification, "git_provenance", lambda root: EXPECTED_GIT)
    cofitok = _report(method="cofitok", top1=0.21, top5=0.42)
    dense = _report(method="dense", top1=0.22, top5=0.43)
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    cofitok_path.write_text(json.dumps(cofitok), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    return qualification.build_qualification(
        cofitok=cofitok,
        dense=dense,
        cofitok_path=cofitok_path,
        dense_path=dense_path,
        stage="full",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
        thresholds=_thresholds(),
    )


def _passing_frozen_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, Path, Path]:
    sampling_git = {
        "revision": "b" * 40,
        "branch": "scale/frozen-sampling",
        "tracked_dirty": False,
    }
    cofitok = _report(
        method="cofitok", top1=0.21, top5=0.42, stage="scaling"
    )
    dense = _report(method="dense", top1=0.22, top5=0.43, stage="scaling")
    cofitok["sample_provenance"]["git"] = sampling_git
    dense["sample_provenance"]["git"] = sampling_git
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    cofitok_path.write_text(json.dumps(cofitok), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    monkeypatch.setattr(qualification, "git_provenance", lambda root: EXPECTED_GIT)
    report = qualification.build_qualification(
        cofitok=cofitok,
        dense=dense,
        cofitok_path=cofitok_path,
        dense_path=dense_path,
        stage="scaling",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
        thresholds={
            "min_top1": 0.01,
            "min_top5": 0.05,
            "min_predicted_class_fraction": 0.25,
            "min_normalized_predicted_entropy": 0.50,
            "max_top1_regression": 0.05,
            "max_top5_regression": 0.05,
        },
    )
    report_path = tmp_path / "qualification.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    gate = {
        "source_profile": "stability_scaling",
        "provenance_contract": {
            "training_revision": "c" * 40,
            "training_branch": "scale/training",
            "evaluation_revision": sampling_git["revision"],
            "evaluation_branch": sampling_git["branch"],
        },
        "gates": [
            {
                "name": "matched_sampling_provenance",
                "passed": True,
                "evidence": {
                    "cofitok_checkpoint_sha256": report["sampling_contract"][
                        "cofitok_checkpoint_sha256"
                    ],
                    "dense_checkpoint_sha256": report["sampling_contract"][
                        "dense_checkpoint_sha256"
                    ],
                    "cofitok_sample_set_sha256": report["sampling_contract"][
                        "cofitok_sample_set_sha256"
                    ],
                    "dense_sample_set_sha256": report["sampling_contract"][
                        "dense_sample_set_sha256"
                    ],
                },
            }
        ],
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    monkeypatch.setattr(
        frozen_verify,
        "verify_generation_gate_source_reports",
        lambda value: {},
    )
    monkeypatch.setattr(
        frozen_verify,
        "validate_generation_gate_authorization",
        lambda value, expected_stage: {"stage": expected_stage},
    )
    return report, report_path, gate_path


def test_accumulator_computes_streaming_class_fidelity() -> None:
    accumulator = ClassFidelityAccumulator(num_classes=5)
    logits = torch.tensor(
        [
            [9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0, 0.0],
            [7.0, 6.0, 5.0, 4.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 6.0],
        ]
    )
    accumulator.update(logits[:2], torch.tensor([0, 1]))
    accumulator.update(logits[2:], torch.tensor([1, 3]))
    metrics = accumulator.finalize()

    assert metrics["sample_count"] == 4
    assert metrics["top1_correct"] == 2
    assert metrics["top5_correct"] == 4
    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["top5_accuracy"] == pytest.approx(1.0)
    assert metrics["requested_class_count"] == 3
    assert metrics["predicted_class_count"] == 3


def test_accumulator_rejects_nonfinite_logits() -> None:
    accumulator = ClassFidelityAccumulator(num_classes=5)
    with pytest.raises(ValueError, match="finite"):
        accumulator.update(
            torch.tensor([[float("nan"), 0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([0]),
        )


def test_validate_class_fidelity_report_rejects_raw_weights() -> None:
    report = _report(method="cofitok", top1=0.2, top5=0.4)
    report["sample_provenance"]["weights"] = "model"
    with pytest.raises(ValueError, match="source contract"):
        validate_class_fidelity_report(report)


def test_class_fidelity_v2_separates_sampler_and_evaluator_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_git = {
        "revision": "b" * 40,
        "branch": "scale/frozen-sampling",
        "tracked_dirty": False,
    }
    cofitok = _report(
        method="cofitok", top1=0.21, top5=0.42, stage="scaling"
    )
    dense = _report(method="dense", top1=0.22, top5=0.43, stage="scaling")
    cofitok["sample_provenance"]["git"] = sampling_git
    dense["sample_provenance"]["git"] = sampling_git
    validate_class_fidelity_report(cofitok)
    validate_class_fidelity_report(dense)
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    cofitok_path.write_text(json.dumps(cofitok), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    monkeypatch.setattr(qualification, "git_provenance", lambda root: EXPECTED_GIT)

    report = qualification.build_qualification(
        cofitok=cofitok,
        dense=dense,
        cofitok_path=cofitok_path,
        dense_path=dense_path,
        stage="scaling",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
        thresholds={
            "min_top1": 0.01,
            "min_top5": 0.05,
            "min_predicted_class_fraction": 0.25,
            "min_normalized_predicted_entropy": 0.50,
            "max_top1_regression": 0.05,
            "max_top5_regression": 0.05,
        },
    )

    assert report["sampling_contract"]["sampling_git"] == sampling_git
    assert report["sampling_contract"]["evaluator_git"] == EXPECTED_GIT
    assert validate_class_fidelity_qualification(
        report,
        expected_stage="scaling",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
    )["valid"] is True


def test_class_fidelity_qualification_rejects_mismatched_sampling_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cofitok = _report(
        method="cofitok", top1=0.21, top5=0.42, stage="scaling"
    )
    dense = _report(method="dense", top1=0.22, top5=0.43, stage="scaling")
    dense["sample_provenance"]["git"] = {
        "revision": "b" * 40,
        "branch": "scale/other-sampling",
        "tracked_dirty": False,
    }
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    cofitok_path.write_text(json.dumps(cofitok), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    monkeypatch.setattr(qualification, "git_provenance", lambda root: EXPECTED_GIT)

    with pytest.raises(ValueError, match="sampling Git identities"):
        qualification.build_qualification(
            cofitok=cofitok,
            dense=dense,
            cofitok_path=cofitok_path,
            dense_path=dense_path,
            stage="scaling",
            expected_revision=EXPECTED_GIT["revision"],
            expected_branch=EXPECTED_GIT["branch"],
            thresholds={
                "min_top1": 0.01,
                "min_top5": 0.05,
                "min_predicted_class_fraction": 0.25,
                "min_normalized_predicted_entropy": 0.50,
                "max_top1_regression": 0.05,
                "max_top5_regression": 0.05,
            },
        )


def test_frozen_class_fidelity_verifier_replays_cross_revision_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_frozen_qualification(
        tmp_path, monkeypatch
    )
    evidence = frozen_verify.verify_frozen_class_fidelity_qualification(
        report,
        report_path=report_path,
        expected_report_sha256=frozen_verify.file_sha256(report_path),
        promotion_gate_path=gate_path,
        expected_evaluator_revision=EXPECTED_GIT["revision"],
        expected_evaluator_branch=EXPECTED_GIT["branch"],
    )

    assert evidence["class_fidelity_passed"] is True
    assert evidence["sampling_git"]["revision"] == "b" * 40
    assert evidence["evaluator_git"] == EXPECTED_GIT
    assert evidence["required_for_full_training_launch"] is True
    assert evidence["full_training_launch_allowed"] is False


def test_frozen_class_fidelity_verifier_rejects_raw_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_frozen_qualification(
        tmp_path, monkeypatch
    )
    Path(report["sources"]["cofitok"]["path"]).write_text(
        '{"replaced":true}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source changed"):
        frozen_verify.verify_frozen_class_fidelity_qualification(
            report,
            report_path=report_path,
            expected_report_sha256=frozen_verify.file_sha256(report_path),
            promotion_gate_path=gate_path,
            expected_evaluator_revision=EXPECTED_GIT["revision"],
            expected_evaluator_branch=EXPECTED_GIT["branch"],
        )


def test_frozen_class_fidelity_verifier_rejects_gate_sample_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_frozen_qualification(
        tmp_path, monkeypatch
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["gates"][0]["evidence"]["cofitok_sample_set_sha256"] = "9" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="sample identity differs from the gate"):
        frozen_verify.verify_frozen_class_fidelity_qualification(
            report,
            report_path=report_path,
            expected_report_sha256=frozen_verify.file_sha256(report_path),
            promotion_gate_path=gate_path,
            expected_evaluator_revision=EXPECTED_GIT["revision"],
            expected_evaluator_branch=EXPECTED_GIT["branch"],
        )


def test_frozen_class_fidelity_verifier_rejects_forged_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_frozen_qualification(
        tmp_path, monkeypatch
    )
    raw_path = Path(report["sources"]["cofitok"]["path"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["runtime_environment"]["device"]["type"] = "forged"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    report["sources"]["cofitok"] = frozen_verify._identity(raw_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="raw report contract differs"):
        frozen_verify.verify_frozen_class_fidelity_qualification(
            report,
            report_path=report_path,
            expected_report_sha256=frozen_verify.file_sha256(report_path),
            promotion_gate_path=gate_path,
            expected_evaluator_revision=EXPECTED_GIT["revision"],
            expected_evaluator_branch=EXPECTED_GIT["branch"],
        )


def test_frozen_class_fidelity_verifier_recomputes_threshold_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, report_path, gate_path = _passing_frozen_qualification(
        tmp_path, monkeypatch
    )
    report["checks"][0]["threshold"] = 0.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="checks did not pass"):
        frozen_verify.verify_frozen_class_fidelity_qualification(
            report,
            report_path=report_path,
            expected_report_sha256=frozen_verify.file_sha256(report_path),
            promotion_gate_path=gate_path,
            expected_evaluator_revision=EXPECTED_GIT["revision"],
            expected_evaluator_branch=EXPECTED_GIT["branch"],
        )


def test_frozen_class_fidelity_runbook_is_source_bound_and_nontraining() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/"
        "generation_stability_frozen_50k_class_fidelity_after_supplemental.sh"
    ).read_text(encoding="utf-8")

    assert "frozen_posteval_class_fidelity.lock" in source
    assert "flock -n 6" in source
    assert "posteval_waiter.json" in source
    assert "supplemental_waiter.json" in source
    assert "supplemental_qualification.json" in source
    assert 'nested.get("status") != "pass"' in source
    assert "validate_generation_gate_report.py" in source
    assert "verify_generation_stability_frozen_supplemental.py" in source
    assert source.count("scripts/evaluate_generation_class_fidelity.py") == 2
    assert "scripts/build_generation_class_fidelity_qualification.py" in source
    assert "scripts/verify_generation_stability_frozen_class_fidelity.py" in source
    assert "--min-samples 10000" in source
    assert "--stage scaling" in source
    assert "--allow-hold" in source
    assert '--output-file "$QUALIFICATION"' in source
    assert '--output-tree "$CLASS_FIDELITY_ROOT"' not in source
    assert "scripts/train_generation.py" not in source
    assert "full_matched_300k" not in source


def test_classifier_identity_binds_bytes_sha_and_preprocessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "classifier.pth"
    checkpoint.write_bytes(b"fixed")
    monkeypatch.setattr(evaluator, "CLASSIFIER_BYTES", 5)
    monkeypatch.setattr(
        evaluator,
        "CLASSIFIER_SHA256",
        hashlib.sha256(b"fixed").hexdigest(),
    )
    identity = evaluator.classifier_identity(checkpoint)
    assert identity["weights_bytes"] == 5
    assert identity["num_classes"] == 1000
    assert identity["preprocessing"]["crop_size"] == [224]


def test_calculate_class_fidelity_consumes_filename_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"{index:06d}.png"
        Image.new("RGB", (256, 256), color).save(path)
        paths.append(path)

    class FakeClassifier(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((batch.shape[0], 1000), device=batch.device)
            logits[0, 0] = 10.0
            logits[1, 1] = 10.0
            return logits

    monkeypatch.setattr(
        evaluator,
        "_load_classifier",
        lambda checkpoint, device: FakeClassifier().to(device),
    )
    metrics = evaluator.calculate_class_fidelity(
        images=paths,
        classifier_checkpoint=tmp_path / "unused.pth",
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
    )
    assert metrics["sample_count"] == 2
    assert metrics["top1_accuracy"] == pytest.approx(1.0)
    assert metrics["top5_accuracy"] == pytest.approx(1.0)


def test_paired_qualification_passes_absolute_and_relative_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _passing_qualification(tmp_path, monkeypatch)
    assert report["status"] == "pass"
    assert len(report["checks"]) == 10
    assert report["sampling_contract"]["cofitok_prefix_budget"] == 8
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert validate_class_fidelity_qualification(
        report,
        expected_stage="full",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
    )["valid"] is True


def test_paired_qualification_rejects_tampered_metric_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _passing_qualification(tmp_path, monkeypatch)
    report["metrics"]["cofitok_minus_dense"]["top1_accuracy"] += 0.01

    with pytest.raises(ValueError, match="paired metric delta differs"):
        validate_class_fidelity_qualification(
            report,
            expected_stage="full",
            expected_revision=EXPECTED_GIT["revision"],
            expected_branch=EXPECTED_GIT["branch"],
        )


def test_completion_evidence_independently_recomputes_class_fidelity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_report = _passing_qualification(tmp_path, monkeypatch)
    validated = validate_class_fidelity_qualification(
        qualification_report,
        expected_stage="full",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
    )
    contract = validated["sampling_contract"]
    gate_evidence = gate_class_fidelity_evidence(
        qualification_report,
        stage="full",
        expected_evaluation_revision=EXPECTED_GIT["revision"],
        expected_evaluation_branch=EXPECTED_GIT["branch"],
        expected_sampling_protocol=contract["sampling"],
        expected_cofitok_checkpoint_sha256=contract[
            "cofitok_checkpoint_sha256"
        ],
        expected_dense_checkpoint_sha256=contract["dense_checkpoint_sha256"],
        expected_cofitok_sample_set_sha256=contract[
            "cofitok_sample_set_sha256"
        ],
        expected_dense_sample_set_sha256=contract[
            "dense_sample_set_sha256"
        ],
    )
    final_gate = {
        "provenance_contract": {
            "evaluation_revision": EXPECTED_GIT["revision"],
            "evaluation_branch": EXPECTED_GIT["branch"],
        },
        "gates": [
            {
                "name": "class_conditional_fidelity",
                "passed": True,
                "evidence": gate_evidence,
            }
        ],
    }
    report = {
        "class_fidelity": {
            "status": validated["status"],
            "valid": validated["valid"],
            "classifier": validated["classifier"],
            "thresholds": validated["thresholds"],
            "sampling_contract": validated["sampling_contract"],
            "claim_boundary": validated["claim_boundary"],
        },
        "matched_summary": {
            "cofitok_minus_dense_class_top1": -0.01,
            "cofitok_minus_dense_class_top5": -0.01,
        },
    }
    rows = []
    for method, metric_key, checkpoint_key, sample_key in (
        (
            "CoFiTok K=8",
            "cofitok",
            "cofitok_checkpoint_sha256",
            "cofitok_sample_set_sha256",
        ),
        (
            "Dense identity",
            "dense_identity",
            "dense_checkpoint_sha256",
            "dense_sample_set_sha256",
        ),
    ):
        metrics = validated["metrics"][metric_key]
        rows.append(
            {
                "method": method,
                "checkpoint_sha256": contract[checkpoint_key],
                "sample_set_sha256": contract[sample_key],
                "class_fidelity_sample_count": metrics["sample_count"],
                "class_top1_accuracy": metrics["top1_accuracy"],
                "class_top5_accuracy": metrics["top5_accuracy"],
                "class_mean_target_probability": metrics[
                    "mean_target_probability"
                ],
                "class_target_negative_log_likelihood": metrics[
                    "target_negative_log_likelihood"
                ],
                "class_predicted_class_fraction": metrics[
                    "predicted_class_fraction"
                ],
                "class_normalized_predicted_entropy": metrics[
                    "normalized_predicted_class_entropy"
                ],
            }
        )

    evidence = _class_fidelity_comparison_evidence(
        report,
        rows,
        qualification_report,
        final_gate,
    )
    assert evidence["status"] == "pass"
    assert evidence["sample_count_per_method"] == 50_000
    assert evidence["release_authorization_allowed"] is False

    tampered_rows = copy.deepcopy(rows)
    tampered_rows[0]["class_top1_accuracy"] += 0.01
    with pytest.raises(ValueError, match="class_top1_accuracy differs"):
        _class_fidelity_comparison_evidence(
            report,
            tampered_rows,
            qualification_report,
            final_gate,
        )


def test_paired_qualification_holds_on_ignored_conditioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "git_provenance", lambda root: EXPECTED_GIT)
    cofitok = _report(
        method="cofitok", top1=0.005, top5=0.02, stage="scaling"
    )
    dense = _report(
        method="dense", top1=0.22, top5=0.43, stage="scaling"
    )
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    cofitok_path.write_text(json.dumps(cofitok), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")

    report = qualification.build_qualification(
        cofitok=cofitok,
        dense=dense,
        cofitok_path=cofitok_path,
        dense_path=dense_path,
        stage="scaling",
        expected_revision=EXPECTED_GIT["revision"],
        expected_branch=EXPECTED_GIT["branch"],
        thresholds=_thresholds(),
    )
    assert report["status"] == "hold"
    failed = {row["name"] for row in report["checks"] if row["status"] == "hold"}
    assert "cofitok_top1_accuracy" in failed
    assert "cofitok_top1_accuracy_regression_vs_dense" in failed


def test_paired_qualification_rejects_unmatched_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qualification, "git_provenance", lambda root: EXPECTED_GIT)
    cofitok = _report(
        method="cofitok", top1=0.2, top5=0.4, stage="scaling"
    )
    dense = _report(method="dense", top1=0.2, top5=0.4, stage="scaling")
    dense["sample_provenance"]["sampling"]["guidance_scale"] = 2.0
    cofitok_path = tmp_path / "cofitok.json"
    dense_path = tmp_path / "dense.json"
    cofitok_path.write_text(json.dumps(cofitok), encoding="utf-8")
    dense_path.write_text(json.dumps(dense), encoding="utf-8")

    with pytest.raises(ValueError, match="not matched"):
        qualification.build_qualification(
            cofitok=cofitok,
            dense=dense,
            cofitok_path=cofitok_path,
            dense_path=dense_path,
            stage="scaling",
            expected_revision=EXPECTED_GIT["revision"],
            expected_branch=EXPECTED_GIT["branch"],
            thresholds=_thresholds(),
        )
