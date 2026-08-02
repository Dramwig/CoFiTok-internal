from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cofitok.generation_gate import (
    GENERATION_GATE_SCHEMA_VERSION,
    REQUIRED_GENERATION_GATES,
    validate_generation_gate_authorization,
)
from cofitok.generation_gate_sources import (
    GATE_SOURCE_SUFFIXES,
    build_generation_gate_source_reports,
)
from cofitok.training import ExponentialMovingAverage
from cofitok.training.authorization import (
    capture_generation_training_authorization,
    validate_checkpoint_training_authorization,
)
from cofitok.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
    verify_training_checkpoint,
)


def _gate(stage: str = "scaling", *, source_profile: str | None = None) -> dict:
    full = stage == "full"
    summary = {
        "cofitok_fid": 19.5 if full else 90.0,
        "dense_fid": 19.0 if full else 89.0,
        "cofitok_endpoint_mse": 0.104,
        "dense_endpoint_mse": 0.1,
        "cofitok_precision": 0.35,
        "dense_precision": 0.36,
        "cofitok_recall": 0.34,
        "dense_recall": 0.35,
        "ordered_rank": 1,
        "order_count": 24,
        "coarse_token_energy_ratio": 0.06,
    }
    thresholds = {
        "min_samples": 50_000 if full else 10_000,
        "max_fid_regression": 0.05,
        "max_absolute_fid": 20.0 if full else 100.0,
        "max_endpoint_regression": 0.05,
        "min_coarse_token_energy_ratio": 0.05,
        "min_precision": 0.30,
        "min_recall": 0.30,
        "max_precision_regression": 0.05,
        "max_recall_regression": 0.05,
    }

    def evidence(name: str) -> dict:
        if name == "fid_within_tolerance":
            return {
                "cofitok_fid": summary["cofitok_fid"],
                "dense_fid": summary["dense_fid"],
                "max_regression": thresholds["max_fid_regression"],
            }
        if name == "absolute_fid_quality":
            return {
                "cofitok_fid": summary["cofitok_fid"],
                "max_absolute_fid": thresholds["max_absolute_fid"],
            }
        if name == "endpoint_within_tolerance":
            return {
                "cofitok_endpoint_mse": summary["cofitok_endpoint_mse"],
                "dense_endpoint_mse": summary["dense_endpoint_mse"],
                "max_regression": thresholds["max_endpoint_regression"],
            }
        if name == "ordered_prefix_path":
            return {"rank": summary["ordered_rank"], "order_count": summary["order_count"]}
        if name == "coarse_token_utilization":
            return {
                "valid": True,
                "source_metric": "component_energy_ratio_per_sample_mean",
                "token_count": 8,
                "coarse_token_count": 6,
                "component_energy_ratios": [0.01] * 6 + [0.30, 0.64],
                "coarse_token_energy_ratio": summary["coarse_token_energy_ratio"],
                "min_coarse_token_energy_ratio": thresholds[
                    "min_coarse_token_energy_ratio"
                ],
            }
        if name == "restricted_synthesis_contract":
            return {"zero_token_max_abs": 0.0}
        if name == "shuffle_mismatch":
            return {"shuffled_to_ordered_endpoint_ratio": 1.2}
        if name == "matched_sampling_provenance":
            return {
                "cofitok_checkpoint_step": 50_000,
                "dense_checkpoint_step": 50_000,
                "cofitok_checkpoint_sha256": "a" * 64,
                "dense_checkpoint_sha256": "b" * 64,
            }
        if name == "matched_checkpoint_evaluator_code_provenance":
            return {
                "expected_revision": "d" * 40,
                "expected_branch": "scale/generative-system",
            }
        if name == "full_precision_recall_quality":
            return {
                "enforced": True,
                **{key: thresholds[key] for key in (
                    "min_precision",
                    "min_recall",
                    "max_precision_regression",
                    "max_recall_regression",
                )},
                **{key: summary[key] for key in (
                    "cofitok_precision",
                    "dense_precision",
                    "cofitok_recall",
                    "dense_recall",
                )},
            }
        if name == "scaling_precision_recall_quality":
            return {
                "enforced": True,
                **{key: thresholds[key] for key in (
                    "min_precision",
                    "min_recall",
                    "max_precision_regression",
                    "max_recall_regression",
                )},
                **{key: summary[key] for key in (
                    "cofitok_precision",
                    "dense_precision",
                    "cofitok_recall",
                    "dense_recall",
                )},
            }
        return {}

    required_gates = set(REQUIRED_GENERATION_GATES[stage])
    if source_profile == "stability_scaling":
        required_gates.add("scaling_precision_recall_quality")
    gate = {
        "schema_version": GENERATION_GATE_SCHEMA_VERSION,
        "stage": stage,
        "status": "pass",
        "decision": (
            "large_scale_generation_ready" if full else "promote_to_full_imagenet256"
        ),
        "thresholds": thresholds,
        "gates": [
            {"name": name, "passed": True, "evidence": evidence(name)}
            for name in sorted(required_gates)
        ],
        "summary": summary,
    }
    if source_profile is not None:
        gate["source_profile"] = source_profile
    return gate


def _bind_gate_sources(gate: dict, tmp_path: Path) -> dict[str, Path]:
    stage = gate["stage"]
    paths = {}
    for index, (name, suffix) in enumerate(GATE_SOURCE_SUFFIXES[stage].items()):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"name": name, "index": index}),
            encoding="utf-8",
        )
        paths[name] = path
    gate["source_reports"] = build_generation_gate_source_reports(
        stage=stage,
        paths=paths,
    )
    return paths


def _bind_rollout_stability(gate: dict) -> dict:
    evidence = {
        "valid": True,
        "checks": {
            "schema": True,
            "status": True,
            "weights": True,
            "checkpoint_step": True,
            "checkpoint_images": True,
            "checkpoint_identity": True,
            "qualification_gates": True,
            "pair_contract": True,
            "rollout_protocol": True,
        },
        "schema_version": 2,
        "status": "pass",
        "weights": "ema",
        "checkpoint_step": 50_000,
        "checkpoint_evaluated_images": 1024,
        "cofitok_checkpoint_sha256": "a" * 64,
        "dense_checkpoint_sha256": "b" * 64,
        "evaluation_git_revision": "d" * 40,
        "evaluation_git_branch": "scale/generative-system",
        "failed_gates": [],
        "pair_contract_valid": True,
        "rollout_protocol": {
            "num_images": 64,
            "sample_steps": 100,
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "clip_x0": True,
            "precision": "bf16",
        },
    }
    gate["gates"].append(
        {
            "name": "rollout_stability_diagnostic",
            "passed": True,
            "evidence": evidence,
        }
    )
    gate["diagnostic_reports"] = {
        "rollout_stability_qualification": {
            "path": "/reports/ema_rollout_stability/qualification_report.json",
            "bytes": 123,
            "sha256": "c" * 64,
        }
    }
    return evidence


def test_scaling_gate_authorizes_only_the_locked_contract() -> None:
    evidence = validate_generation_gate_authorization(_gate(), expected_stage="scaling")

    assert evidence["decision"] == "promote_to_full_imagenet256"
    assert evidence["validated_thresholds"]["max_absolute_fid"] == 100.0


def test_legacy_schema_v2_stability_gate_remains_replayable() -> None:
    gate = _gate()
    gate["schema_version"] = 2
    gate["source_profile"] = "stability_scaling"

    evidence = validate_generation_gate_authorization(
        gate, expected_stage="scaling"
    )

    assert evidence["schema_version"] == 2


def test_schema_v4_stability_gate_requires_rollout_diagnostic() -> None:
    gate = _gate(source_profile="stability_scaling")

    with pytest.raises(ValueError, match="rollout_stability_diagnostic"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_legacy_schema_v3_stability_gate_remains_replayable_without_new_quality_gate() -> None:
    gate = _gate(source_profile="stability_scaling")
    gate["schema_version"] = 3
    gate["gates"] = [
        row
        for row in gate["gates"]
        if row["name"] != "scaling_precision_recall_quality"
    ]
    _bind_rollout_stability(gate)

    evidence = validate_generation_gate_authorization(
        gate, expected_stage="scaling"
    )

    assert evidence["schema_version"] == 3


@pytest.mark.parametrize(
    ("threshold", "value"),
    (
        ("min_samples", 9_999),
        ("max_absolute_fid", 100.01),
        ("max_fid_regression", 0.051),
        ("min_coarse_token_energy_ratio", 0.049),
    ),
)
def test_scaling_gate_rejects_weakened_thresholds(threshold: str, value: float) -> None:
    gate = _gate()
    gate["thresholds"][threshold] = value

    with pytest.raises(ValueError, match="weaker"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_rejects_missing_required_check() -> None:
    gate = _gate()
    gate["gates"] = [
        row for row in gate["gates"] if row["name"] != "absolute_fid_quality"
    ]

    with pytest.raises(ValueError, match="absolute_fid_quality"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_recomputes_summary_thresholds() -> None:
    gate = _gate()
    gate["summary"]["cofitok_fid"] = 100.1

    with pytest.raises(ValueError, match="absolute FID"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_rejects_collapsed_coarse_token_evidence() -> None:
    gate = _gate()
    gate["summary"]["coarse_token_energy_ratio"] = 0.005
    row = next(
        item for item in gate["gates"] if item["name"] == "coarse_token_utilization"
    )
    row["evidence"]["component_energy_ratios"] = [0.0] * 5 + [0.005, 0.335, 0.66]
    row["evidence"]["coarse_token_energy_ratio"] = 0.005

    with pytest.raises(ValueError, match="coarse-token utilization"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_accepts_roundoff_in_derived_coarse_energy_sum() -> None:
    gate = _gate()
    row = next(
        item for item in gate["gates"] if item["name"] == "coarse_token_utilization"
    )
    derived = sum(row["evidence"]["component_energy_ratios"][:6])
    reported = derived + 1e-15
    row["evidence"]["coarse_token_energy_ratio"] = reported
    gate["summary"]["coarse_token_energy_ratio"] = reported

    validate_generation_gate_authorization(gate, expected_stage="scaling")

    row["evidence"]["coarse_token_energy_ratio"] = derived + 1e-9
    with pytest.raises(ValueError, match="differs from its summary"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_validates_optional_ema_rollout_stability() -> None:
    gate = _gate(source_profile="stability_scaling")
    evidence = _bind_rollout_stability(gate)

    validate_generation_gate_authorization(gate, expected_stage="scaling")

    evidence["rollout_protocol"]["guidance_scale"] = 1.0
    with pytest.raises(ValueError, match="rollout-stability protocol"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


@pytest.mark.parametrize(
    ("threshold", "value"),
    (
        ("min_precision", 0.099),
        ("min_recall", 0.099),
        ("max_precision_regression", 0.051),
        ("max_recall_regression", 0.051),
    ),
)
def test_schema_v4_stability_scaling_rejects_weakened_distribution_support_thresholds(
    threshold: str, value: float
) -> None:
    gate = _gate(source_profile="stability_scaling")
    _bind_rollout_stability(gate)
    gate["thresholds"][threshold] = value

    with pytest.raises(ValueError, match=threshold):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_schema_v4_stability_scaling_recomputes_distribution_support() -> None:
    gate = _gate(source_profile="stability_scaling")
    _bind_rollout_stability(gate)
    gate["summary"]["cofitok_recall"] = 0.09
    quality = next(
        row
        for row in gate["gates"]
        if row["name"] == "scaling_precision_recall_quality"
    )
    quality["evidence"]["cofitok_recall"] = 0.09

    with pytest.raises(ValueError, match="minimum recall"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_scaling_gate_validates_rgbtail3_stride_partition() -> None:
    gate = _gate()
    row = next(
        item for item in gate["gates"] if item["name"] == "coarse_token_utilization"
    )
    row["evidence"].update(
        {
            "partition_schema": "token_spatial_stride_suffix_v1",
            "coarse_token_count": 5,
            "full_resolution_tail_token_count": 3,
            "token_spatial_strides": [16, 16, 8, 8, 4, 1, 1, 1],
            "component_energy_ratios": [0.012] * 5 + [0.30, 0.30, 0.34],
        }
    )

    validate_generation_gate_authorization(gate, expected_stage="scaling")

    row["evidence"]["coarse_token_count"] = 6
    with pytest.raises(ValueError, match="partition is invalid"):
        validate_generation_gate_authorization(gate, expected_stage="scaling")


def test_full_gate_rejects_weakened_precision_floor() -> None:
    gate = copy.deepcopy(_gate("full"))
    gate["thresholds"]["min_precision"] = 0.29

    with pytest.raises(ValueError, match="min_precision"):
        validate_generation_gate_authorization(gate, expected_stage="full")


def test_training_authorization_binds_gate_file_and_checkpoint_before_load(
    tmp_path, monkeypatch
) -> None:
    gate = _gate()
    source_paths = _bind_gate_sources(gate, tmp_path)
    gate_path = tmp_path / "promotion_gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    authorization = capture_generation_training_authorization(gate_path)

    model = torch.nn.Linear(2, 2)
    ema = ExponentialMovingAverage(model, warmup_steps=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "checkpoint_step_00000001.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        step=1,
        config={"name": "authorization-test"},
        extra_state={"training_authorization": authorization},
    )
    integrity = verify_training_checkpoint(checkpoint)
    assert (
        integrity["authorization_gate_identity_sha256"]
        == authorization["gate_identity_sha256"]
    )
    restored = load_training_checkpoint(
        checkpoint,
        model=torch.nn.Linear(2, 2),
        expected_training_authorization=authorization,
        restore_rng=False,
    )
    assert restored["extra_state"]["training_authorization"] == authorization

    changed_gate = copy.deepcopy(gate)
    changed_gate["gates"].append(
        {"name": "additional_audit_note", "passed": True, "evidence": {}}
    )
    changed_path = tmp_path / "changed_promotion_gate.json"
    changed_path.write_text(json.dumps(changed_gate, sort_keys=True), encoding="utf-8")
    changed_authorization = capture_generation_training_authorization(changed_path)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before authorization validation")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="expected promotion gate"):
        load_training_checkpoint(
            checkpoint,
            model=torch.nn.Linear(2, 2),
            expected_training_authorization=changed_authorization,
        )

    source_paths["cofitok_generation"].write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="source report changed after binding"):
        capture_generation_training_authorization(gate_path)


def test_full_training_runbook_binds_authorization_to_every_segment() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_full_matched_300k_after_gate.sh"
    ).read_text(encoding="utf-8")

    assert '--authorization-gate "$GATE"' in runbook
    assert "training_pair_validation.json" in runbook


def test_full_posteval_revalidates_gate_sources_and_training_pair() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_full_posteval_50k.sh"
    ).read_text(encoding="utf-8")

    assert "validate_generation_gate_report.py" in runbook
    assert '--gate "$SCALING_GATE" --stage scaling' in runbook
    assert "validate_generation_training_pair.py" in runbook
    assert '--authorization-gate "$SCALING_GATE"' in runbook
    assert "posteval_training_pair_validation.json" in runbook
    assert "generation_full_matched_300k_monitor.json" in runbook
    assert '--training-contention "$TRAINING_CONTENTION"' in runbook
    assert (
        "--num-images 1024 --timestep 500 --random-orders 16 "
        "--weights ema --precision bf16 --resume"
    ) in runbook
    assert (
        "--num-images 1024 --timestep 500 --random-orders 0 "
        "--weights ema --precision bf16 --resume"
    ) in runbook
    assert runbook.count("scripts/evaluate_generation_metrics.py") == 2
    assert runbook.count("--cache-root \"$EVAL_CACHE\" --min-samples 50000 --resume") == 2


def test_formal_full_trainer_refuses_to_start_without_authorization(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train_generation.py"),
            "--config",
            str(root / "configs/generation/imagenet256_cofitok_k8_300k.json"),
            "--output-dir",
            str(tmp_path / "must-not-start"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires --authorization-gate" in result.stderr
    assert not (tmp_path / "must-not-start").exists()


def test_formal_full_checkpoint_payload_requires_authorization_binding() -> None:
    checkpoint = {
        "config": {
            "data": {"dataset": "imagenet_256"},
            "runtime": {"steps": 300_000},
        },
        "extra_state": {},
    }

    with pytest.raises(ValueError, match="lacks its scaling-gate authorization"):
        validate_checkpoint_training_authorization(checkpoint, {})
