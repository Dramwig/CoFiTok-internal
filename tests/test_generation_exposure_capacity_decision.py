from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from cofitok.generation.exposure_capacity_authorization import identity
from cofitok.generation.exposure_capacity_decision import (
    AUTHORIZATION_BOUNDARY,
    build_decision,
    validate_decision_contract,
)
from cofitok.generation.exposure_capacity_result import (
    RESULT_ROLE,
    RESULT_SCHEMA,
    build_validation_receipt,
)
from cofitok.reporting import file_sha256, write_json_report
from scripts import build_generation_exposure_capacity_decision as builder
from scripts import exposure_capacity_decision_cli as decision_cli
from scripts import validate_generation_exposure_capacity_decision as validator


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("cofitok", "dense_identity")
EXECUTION_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/generation-exposure-capacity-source-compatible-20260901",
    "tracked_dirty": False,
}
DECISION_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "scale/generation-capacity-source-compatible-v1",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100 + len(name),
        "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
    }


def _config(*, cofitok: bool) -> dict:
    return {
        "data": {"dataset": "imagenet_256"},
        "diffusion": {"num_train_timesteps": 1000},
        "runtime": {"steps": 110_000},
        "optimization": {"learning_rate": 1e-5},
        "model": {
            "token_count": 8 if cofitok else 1,
            "predictor_use_feedback": cofitok,
            "synthesis_mode": "fixed_basis" if cofitok else "dense_identity",
        },
        "loss": {"epsilon_weight": 1.0},
    }


def _training(*, cofitok: bool, validation: float = 1.0) -> dict:
    return {
        "training_complete": True,
        "completed_steps": 110_000,
        "target_steps": 110_000,
        "git": {"revision": EXECUTION_GIT["revision"]},
        "config": _config(cofitok=cofitok),
        "final_metrics": {"validation_epsilon_mse": validation},
    }


def _checkpoint(*, cofitok: bool, endpoint: float = 1.0) -> dict:
    metrics = {
        "orders": {"ordered": {"endpoint_clean_mse": endpoint}},
        "component_energy_ratio": (
            [0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25]
            if cofitok
            else [1.0]
        ),
        "evaluated_images": 1024,
        "ordered_rank_by_path_auc": 1,
        "zero_token_max_abs": 0.0,
        "shuffled_to_ordered_endpoint_ratio": 10.0,
        "timestep": 500,
    }
    return {
        "status": "completed",
        "weights": "ema",
        "checkpoint_sha256": "e" * 64 if cofitok else "f" * 64,
        "checkpoint_step": 110_000,
        "git": copy.deepcopy(EXECUTION_GIT),
        "config": _config(cofitok=cofitok),
        "metrics": metrics,
    }


def _rollout(
    *,
    cofitok: bool,
    high_frequency: float = 0.3,
    final_mse: float = 1.0,
) -> dict:
    return {
        "status": "completed",
        "weights": "ema",
        "checkpoint_sha256": "e" * 64 if cofitok else "f" * 64,
        "checkpoint_step": 110_000,
        "git": copy.deepcopy(EXECUTION_GIT),
        "config": _config(cofitok=cofitok),
        "protocol": {
            "num_images": 64,
            "batch_size": 8,
            "sample_steps": 100,
            "guidance_scale": 1.5,
            "seed": 2029,
        },
        "free_sampling_rollout": {
            "steps": [
                {
                    "timestep": timestep,
                    "predicted_x0_high_frequency_ratio": high_frequency,
                }
                for timestep in (595, 394, 192, 91)
            ]
        },
        "reconstruction_rollout": {
            "summary": {
                "final_clipped_x0_mse": final_mse,
                "final_to_best_x0_mse_amplification": 1.1,
            }
        },
    }


def _raw_reports(*, rollout_failure: bool = False) -> dict[str, dict]:
    return {
        "cofitok_training": _training(cofitok=True, validation=1.01),
        "dense_training": _training(cofitok=False, validation=1.0),
        "cofitok_checkpoint": _checkpoint(cofitok=True, endpoint=1.02),
        "dense_checkpoint": _checkpoint(cofitok=False, endpoint=1.0),
        "cofitok_rollout": _rollout(
            cofitok=True,
            high_frequency=0.6 if rollout_failure else 0.3,
            final_mse=1.03,
        ),
        "dense_rollout": _rollout(
            cofitok=False,
            high_frequency=0.25,
            final_mse=1.0,
        ),
    }


def _metric_row(
    *,
    fid: float,
    precision: float,
    recall: float,
    top1: float,
    top5: float,
    coverage: float,
    entropy: float,
) -> dict[str, float]:
    return {
        "fid": fid,
        "precision": precision,
        "recall": recall,
        "top1": top1,
        "top5": top5,
        "predicted_class_fraction": coverage,
        "normalized_predicted_class_entropy": entropy,
    }


def _qualified_metrics() -> dict[str, dict[str, float]]:
    return {
        "cofitok": _metric_row(
            fid=80.0,
            precision=0.45,
            recall=0.20,
            top1=0.02,
            top5=0.08,
            coverage=0.60,
            entropy=0.75,
        ),
        "dense_identity": _metric_row(
            fid=82.0,
            precision=0.43,
            recall=0.19,
            top1=0.018,
            top5=0.075,
            coverage=0.58,
            entropy=0.72,
        ),
    }


def _shared_collapse_metrics() -> dict[str, dict[str, float]]:
    return {
        "cofitok": _metric_row(
            fid=115.0,
            precision=0.70,
            recall=0.008,
            top1=0.0024,
            top5=0.010,
            coverage=0.73,
            entropy=0.78,
        ),
        "dense_identity": _metric_row(
            fid=123.0,
            precision=0.66,
            recall=0.010,
            top1=0.0017,
            top5=0.008,
            coverage=0.69,
            entropy=0.75,
        ),
    }


def _quality_bridge_result(metrics: dict[str, dict[str, float]]) -> dict:
    return {
        "schema_version": 1,
        "role": "stability_full_data_quality_bridge_result",
        "status": "completed",
        "stage": "stability_quality_bridge",
        "terminal": {
            "methods": {
                method: {
                    "fid": row["fid"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                }
                for method, row in metrics.items()
            },
            "class_fidelity": {
                "metrics": {
                    method: {
                        "top1_accuracy": row["top1"],
                        "top5_accuracy": row["top5"],
                        "predicted_class_fraction": row[
                            "predicted_class_fraction"
                        ],
                        "normalized_predicted_class_entropy": row[
                            "normalized_predicted_class_entropy"
                        ],
                    }
                    for method, row in metrics.items()
                }
            },
        },
    }


def _preparation(quality_identity: dict[str, object]) -> dict:
    preparation = json.loads(
        (ROOT / "preparation.json").read_text(encoding="utf-8")
    )
    preparation["sources"]["quality_bridge_result"] = copy.deepcopy(
        quality_identity
    )
    return preparation


def _result(
    *,
    metrics: dict[str, dict[str, float]],
    preparation_identity: dict[str, object],
    raw_identities: dict[str, dict[str, object]],
) -> dict:
    result = {
        "schema_version": RESULT_SCHEMA,
        "role": RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization": _identity("authorization"),
        "candidate_gate": _identity("candidate_gate"),
        "preparation": copy.deepcopy(preparation_identity),
        "standing_authorization": _identity("standing_authorization"),
        "execution_checkout": copy.deepcopy(EXECUTION_GIT),
        "configs": {
            method: _identity(f"{method}_config") for method in METHODS
        },
        "training": {},
        "sampling": {},
        "quality": {},
        "class_fidelity": {},
        "mechanism": {},
        "rollout": {},
        "claim_guards": {
            "terminal_hold_preserved": True,
            "generation_advantage_proven": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "release_allowed": False,
        },
    }
    for method, prefix in (("cofitok", "cofitok"), ("dense_identity", "dense")):
        result["training"][method] = {
            "report": copy.deepcopy(raw_identities[f"{prefix}_training"]),
            "checkpoint": _identity(f"{method}_checkpoint_payload"),
            "integrity_manifest": _identity(f"{method}_checkpoint_integrity"),
            "latest": _identity(f"{method}_latest"),
            "metrics": _identity(f"{method}_training_metrics"),
        }
        result["sampling"][method] = {
            "report": _identity(f"{method}_sampling_report")
        }
        result["quality"][method] = {
            "report": _identity(f"{method}_generation_metrics"),
            "metrics": {
                "frechet_inception_distance": metrics[method]["fid"],
                "precision": metrics[method]["precision"],
                "recall": metrics[method]["recall"],
            },
        }
        result["class_fidelity"][method] = {
            "report": _identity(f"{method}_class_fidelity"),
            "metrics": {
                "top1_accuracy": metrics[method]["top1"],
                "top5_accuracy": metrics[method]["top5"],
                "predicted_class_fraction": metrics[method][
                    "predicted_class_fraction"
                ],
                "normalized_predicted_class_entropy": metrics[method][
                    "normalized_predicted_class_entropy"
                ],
            },
        }
        result["mechanism"][method] = {
            "report": copy.deepcopy(raw_identities[f"{prefix}_checkpoint"])
        }
        result["rollout"][method] = {
            "report": copy.deepcopy(raw_identities[f"{prefix}_rollout"])
        }
    return result


def _inputs(
    *,
    metrics: dict[str, dict[str, float]],
    rollout_failure: bool = False,
) -> dict:
    raw_reports = _raw_reports(rollout_failure=rollout_failure)
    raw_identities = {
        name: _identity(name) for name in raw_reports
    }
    quality = _quality_bridge_result(_shared_collapse_metrics())
    quality_identity = _identity("quality_bridge_result")
    preparation = _preparation(quality_identity)
    preparation_identity = _identity("preparation")
    result = _result(
        metrics=metrics,
        preparation_identity=preparation_identity,
        raw_identities=raw_identities,
    )
    result_identity = _identity("exposure_result")
    receipt = build_validation_receipt(
        result=result,
        result_identity=result_identity,
        validator_git=EXECUTION_GIT,
    )
    return {
        "exposure_result": result,
        "exposure_result_identity": result_identity,
        "validation_receipt": receipt,
        "validation_receipt_identity": _identity("validation_receipt"),
        "preparation": preparation,
        "preparation_identity": preparation_identity,
        "quality_bridge_result": quality,
        "quality_bridge_result_identity": quality_identity,
        "raw_source_reports": raw_reports,
        "raw_source_identities": raw_identities,
        "decision_git": copy.deepcopy(DECISION_GIT),
    }


def test_all_scaling_checks_route_to_exposure_qualified() -> None:
    report = build_decision(**_inputs(metrics=_qualified_metrics()))

    assert report["decision"] == "exposure_qualified"
    assert report["failed_checks"] == []
    assert report["next_stage"]["large_capacity_readiness_preparation_allowed"]
    assert validate_decision_contract(report) == report


def test_shared_support_collapse_routes_to_capacity_screen() -> None:
    report = build_decision(**_inputs(metrics=_shared_collapse_metrics()))

    assert report["decision"] == "capacity_screen"
    assert report["shared_collapse"]["present"] is True
    assert report["cofitok_specific_or_mechanism_failures"] == []
    assert report["next_stage"]["capacity_screen_preparation_allowed"] is True
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert not any(AUTHORIZATION_BOUNDARY.values())


def test_rollout_failure_routes_to_hold() -> None:
    report = build_decision(
        **_inputs(metrics=_shared_collapse_metrics(), rollout_failure=True)
    )

    assert report["decision"] == "hold"
    assert report["stability_qualification"]["status"] == "fail"
    assert "mechanism_and_rollout_stability" in report["failed_checks"]


def test_decision_rejects_receipt_or_preparation_mismatch() -> None:
    inputs = _inputs(metrics=_qualified_metrics())
    inputs["exposure_result"]["quality"]["cofitok"]["metrics"][
        "frechet_inception_distance"
    ] = 81.0
    inputs["exposure_result_identity"] = _identity("changed_exposure_result")
    with pytest.raises(ValueError, match="receipt is not reproducible"):
        build_decision(**inputs)

    inputs = _inputs(metrics=_qualified_metrics())
    inputs["exposure_result"]["preparation"] = _identity("other_preparation")
    inputs["exposure_result_identity"] = _identity("preparation_mismatch_result")
    inputs["validation_receipt"] = build_validation_receipt(
        result=inputs["exposure_result"],
        result_identity=inputs["exposure_result_identity"],
        validator_git=EXECUTION_GIT,
    )
    with pytest.raises(ValueError, match="binds another preparation"):
        build_decision(**inputs)


def test_decision_rejects_raw_source_identity_tampering() -> None:
    inputs = _inputs(metrics=_qualified_metrics())
    inputs["raw_source_identities"]["cofitok_training"] = _identity(
        "changed_training"
    )

    with pytest.raises(ValueError, match="raw sources differ"):
        build_decision(**inputs)


def test_decision_contract_rejects_authorization_tampering() -> None:
    report = build_decision(**_inputs(metrics=_qualified_metrics()))
    report["authorization_boundary"]["gpu_execution_allowed"] = True

    with pytest.raises(ValueError, match="contract differs"):
        validate_decision_contract(report)


def _write_cli_sources(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
    paths: dict[str, Path] = {}
    raw_reports = _raw_reports()
    raw_identities: dict[str, dict[str, object]] = {}
    for name, report in raw_reports.items():
        path = tmp_path / f"{name}.json"
        write_json_report(path, report)
        paths[name] = path
        raw_identities[name] = identity(path)

    quality_path = tmp_path / "quality_bridge_result.json"
    write_json_report(
        quality_path,
        _quality_bridge_result(_shared_collapse_metrics()),
    )
    paths["quality_bridge_result"] = quality_path
    quality_identity = identity(quality_path)

    preparation_path = tmp_path / "preparation.json"
    write_json_report(preparation_path, _preparation(quality_identity))
    paths["preparation"] = preparation_path
    preparation_identity = identity(preparation_path)

    result_path = tmp_path / "exposure_capacity_result.json"
    result = _result(
        metrics=_shared_collapse_metrics(),
        preparation_identity=preparation_identity,
        raw_identities=raw_identities,
    )
    write_json_report(result_path, result)
    paths["exposure_result"] = result_path

    receipt_path = tmp_path / "exposure_capacity_result.validation.json"
    write_json_report(
        receipt_path,
        build_validation_receipt(
            result=result,
            result_identity=identity(result_path),
            validator_git=EXECUTION_GIT,
        ),
    )
    paths["validation_receipt"] = receipt_path
    return paths, raw_identities


def _common_cli_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--exposure-result",
        str(paths["exposure_result"]),
        "--expected-exposure-result-sha256",
        file_sha256(paths["exposure_result"]),
        "--validation-receipt",
        str(paths["validation_receipt"]),
        "--expected-validation-receipt-sha256",
        file_sha256(paths["validation_receipt"]),
        "--preparation",
        str(paths["preparation"]),
        "--expected-preparation-sha256",
        file_sha256(paths["preparation"]),
        "--quality-bridge-result",
        str(paths["quality_bridge_result"]),
        "--expected-quality-bridge-result-sha256",
        file_sha256(paths["quality_bridge_result"]),
        "--cofitok-training-report",
        str(paths["cofitok_training"]),
        "--dense-training-report",
        str(paths["dense_training"]),
        "--cofitok-checkpoint-report",
        str(paths["cofitok_checkpoint"]),
        "--dense-checkpoint-report",
        str(paths["dense_checkpoint"]),
        "--cofitok-rollout-report",
        str(paths["cofitok_rollout"]),
        "--dense-rollout-report",
        str(paths["dense_rollout"]),
        "--decision-project-root",
        str(ROOT),
    ]


def test_builder_and_validator_replay_the_exact_capacity_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _write_cli_sources(tmp_path)
    decision_path = tmp_path / "decision.json"
    monkeypatch.setattr(decision_cli, "checkout_identity", lambda _: DECISION_GIT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_exposure_capacity_decision.py",
            "--decision",
            str(decision_path),
            *_common_cli_args(paths),
        ],
    )

    builder.main()

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "capacity_screen"
    validation_path = tmp_path / "decision.validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_decision.py",
            "--decision",
            str(decision_path),
            "--expected-decision-sha256",
            file_sha256(decision_path),
            "--output",
            str(validation_path),
            *_common_cli_args(paths),
        ],
    )

    validator.main()

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "pass"
    assert validation["scientific_route"] == "capacity_screen"
    assert not any(validation["authorization_boundary"].values())


def test_validator_rejects_a_changed_raw_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _write_cli_sources(tmp_path)
    decision_path = tmp_path / "decision.json"
    monkeypatch.setattr(decision_cli, "checkout_identity", lambda _: DECISION_GIT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_exposure_capacity_decision.py",
            "--decision",
            str(decision_path),
            *_common_cli_args(paths),
        ],
    )
    builder.main()

    checkpoint = json.loads(
        paths["cofitok_checkpoint"].read_text(encoding="utf-8")
    )
    checkpoint["metrics"]["zero_token_max_abs"] = 0.1
    write_json_report(paths["cofitok_checkpoint"], checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_decision.py",
            "--decision",
            str(decision_path),
            "--expected-decision-sha256",
            file_sha256(decision_path),
            *_common_cli_args(paths),
        ],
    )

    with pytest.raises(ValueError, match="raw sources differ"):
        validator.main()


def test_builder_rejects_an_expected_primary_source_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _write_cli_sources(tmp_path)
    arguments = _common_cli_args(paths)
    sha_index = arguments.index("--expected-exposure-result-sha256") + 1
    arguments[sha_index] = "0" * 64
    monkeypatch.setattr(decision_cli, "checkout_identity", lambda _: DECISION_GIT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_exposure_capacity_decision.py",
            "--decision",
            str(tmp_path / "decision.json"),
            *arguments,
        ],
    )

    with pytest.raises(ValueError, match="exposure result SHA256 differs"):
        builder.main()


def test_validator_rejects_a_rewritten_decision_even_with_its_new_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = _write_cli_sources(tmp_path)
    decision_path = tmp_path / "decision.json"
    monkeypatch.setattr(decision_cli, "checkout_identity", lambda _: DECISION_GIT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_exposure_capacity_decision.py",
            "--decision",
            str(decision_path),
            *_common_cli_args(paths),
        ],
    )
    builder.main()

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["next_stage"]["full_300k_launch_allowed"] = True
    write_json_report(decision_path, decision)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_generation_exposure_capacity_decision.py",
            "--decision",
            str(decision_path),
            "--expected-decision-sha256",
            file_sha256(decision_path),
            *_common_cli_args(paths),
        ],
    )

    with pytest.raises(ValueError, match="does not match revalidated source evidence"):
        validator.main()
