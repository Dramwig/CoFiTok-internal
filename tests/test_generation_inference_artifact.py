from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cofitok.configs import (
    DataConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    config_to_dict,
)
from cofitok.generation import (
    GenerationRequest,
    GenerationSession,
    export_ema_inference_artifact,
    inference_export_manifest_path,
    verify_inference_artifact,
    verify_inference_export_manifest,
    verify_generation_release_receipt,
    write_generation_release_receipt,
)
from cofitok.generation_gate import (
    GENERATION_GATE_SCHEMA_VERSION,
    REQUIRED_GENERATION_GATES,
)
from cofitok.generation_cost import training_cost_summary
from cofitok.generation_gate_sources import (
    GATE_SOURCE_SUFFIXES,
    build_generation_gate_source_reports,
)
import cofitok.generation.release as generation_release
from cofitok.generation.release import (
    _COMPLETION_PROFILES,
    _COMPLETION_REQUIRED_CHECKS,
)
from cofitok.environment import runtime_environment_sha256
from cofitok.models import CoFiTokTiny
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import checkpoint_integrity_path


ROOT = Path(__file__).resolve().parents[1]
from scripts.preflight_generation_sampling import run_sampling_preflight


SOURCE_ENVIRONMENT_SHA = "e" * 64
SOURCE_GIT = {
    "revision": "a" * 40,
    "branch": "scale/generative-system",
    "dirty": False,
}
EXPORT_GIT = {
    "revision": "d" * 40,
    "branch": "analysis/generation-inference-export-v1",
    "tracked_dirty": False,
}
EXPORT_ENVIRONMENT = {
    "schema_version": 1,
    "python": {"implementation": "CPython", "version": "3.10"},
}
EXPORT_EXECUTION = {
    "git": EXPORT_GIT,
    "runtime_environment": EXPORT_ENVIRONMENT,
    "runtime_environment_sha256": runtime_environment_sha256(
        EXPORT_ENVIRONMENT
    ),
}


def _capacity_training_authorization(tmp_path) -> dict:
    return {
        "schema_version": 1,
        "status": "pass",
        "stage": "capacity_full_experimental",
        "decision": "authorize_fresh_matched_300k_training",
        "gate_path": (
            tmp_path / "capacity_full_training_launch_receipt.json"
        ).resolve().as_posix(),
        "gate_bytes": 23_456,
        "gate_sha256": "6" * 64,
        "gate_identity_sha256": "7" * 64,
        "validated_thresholds": {
            "target_start_step": 0,
            "target_steps": 300_000,
            "effective_batch_size": 64,
            "capacity_100k_checkpoint_resume_allowed": False,
            "formal_generation_claim_allowed": False,
            "release_authorization_allowed": False,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 1,
        },
    }


def _training_authorization(tmp_path) -> dict:
    return {
        "schema_version": 1,
        "status": "pass",
        "stage": "scaling",
        "decision": "promote_to_full_imagenet256",
        "gate_path": (tmp_path / "promotion_gate.json").resolve().as_posix(),
        "gate_bytes": 12_345,
        "gate_sha256": "b" * 64,
        "gate_identity_sha256": "c" * 64,
        "validated_thresholds": {
            "min_samples": 10_000.0,
            "max_fid_regression": 0.05,
            "max_absolute_fid": 100.0,
            "max_endpoint_regression": 0.05,
        },
    }


def _gate_training_report() -> dict:
    return {
        "target_steps": 300_000,
        "completed_steps": 300_000,
        "config": {
            "data": {"batch_size": 16},
            "optimization": {"gradient_accumulation_steps": 4},
            "runtime": {"device": "cuda"},
        },
        "final_metrics": {"samples_seen": 19_200_000},
        "elapsed_seconds": 100_000.0,
        "peak_vram_bytes": 24 * 1024**3,
    }


def _full_gate() -> dict:
    cofitok_training_cost = training_cost_summary(_gate_training_report())
    dense_training_cost = training_cost_summary(_gate_training_report())
    summary = {
        "cofitok_fid": 19.5,
        "dense_fid": 19.0,
        "cofitok_endpoint_mse": 0.104,
        "dense_endpoint_mse": 0.1,
        "cofitok_precision": 0.35,
        "dense_precision": 0.36,
        "cofitok_recall": 0.34,
        "dense_recall": 0.35,
        "ordered_rank": 1,
        "order_count": 24,
        "coarse_token_energy_ratio": 0.06,
        "cofitok_training_cost": cofitok_training_cost,
        "dense_training_cost": dense_training_cost,
    }
    thresholds = {
        "min_samples": 50_000,
        "max_fid_regression": 0.05,
        "max_absolute_fid": 20.0,
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
            return {
                "rank": summary["ordered_rank"],
                "order_count": summary["order_count"],
            }
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
        if name == "full_precision_recall_quality":
            return {
                "enforced": True,
                **{
                    key: thresholds[key]
                    for key in (
                        "min_precision",
                        "min_recall",
                        "max_precision_regression",
                        "max_recall_regression",
                    )
                },
                **{
                    key: summary[key]
                    for key in (
                        "cofitok_precision",
                        "dense_precision",
                        "cofitok_recall",
                        "dense_recall",
                    )
                },
            }
        if name == "training_cost_accounting":
            return {
                "cofitok": cofitok_training_cost,
                "dense": dense_training_cost,
            }
        return {}

    return {
        "schema_version": GENERATION_GATE_SCHEMA_VERSION,
        "stage": "full",
        "status": "pass",
        "decision": "large_scale_generation_ready",
        "resume_compute_adjustments": {},
        "thresholds": thresholds,
        "gates": [
            {"name": name, "passed": True, "evidence": evidence(name)}
            for name in sorted(REQUIRED_GENERATION_GATES["full"])
        ],
        "summary": summary,
    }


def _release_gate_path(tmp_path):
    gate = _full_gate()
    paths = {}
    for index, (name, suffix) in enumerate(GATE_SOURCE_SUFFIXES["full"].items()):
        path = tmp_path / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            _gate_training_report()
            if name in {"cofitok_training", "dense_training"}
            else {"name": name, "index": index}
        )
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        paths[name] = path
    gate["source_reports"] = build_generation_gate_source_reports(
        stage="full",
        paths=paths,
    )
    path = tmp_path / "final_generation_gate.json"
    path.write_text(json.dumps(gate, sort_keys=True), encoding="utf-8")
    return path


def _training_checkpoint(
    tmp_path,
    *,
    include_provenance: bool = True,
    include_authorization: bool = False,
    step: int = 31,
    authorization_root: Path | None = None,
    training_authorization: dict | None = None,
):
    config = ExperimentConfig(
        name="inference_export_cpu",
        data=DataConfig(image_size=8, channels=3),
        diffusion=DiffusionConfig(num_train_timesteps=4, schedule_type="cosine"),
        model=ModelConfig(
            image_channels=3,
            image_size=8,
            token_count=2,
            token_channels=4,
            base_channels=8,
            predictor_type="scalable_unet",
            predictor_channel_multipliers=[1],
            predictor_num_res_blocks=1,
            predictor_attention_resolutions=[],
            predictor_num_heads=1,
            num_classes=5,
            synthesis_active_token_channels=[2, 4],
        ),
        runtime=RuntimeConfig(device="cpu", precision="fp32"),
        optimization=OptimizationConfig(ema_warmup_steps=0),
    )
    model = CoFiTokTiny(config.model)
    ema = ExponentialMovingAverage(model, warmup_steps=0)
    with torch.no_grad():
        for value in ema.shadow.values():
            if torch.is_floating_point(value):
                value.add_(0.01)
    path = tmp_path / "training.pt"
    payload = {
        "format_version": 1,
        "config": config_to_dict(config),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "step": step,
    }
    authorization = (
        training_authorization
        if training_authorization is not None
        else _training_authorization(authorization_root or tmp_path)
    )
    if include_provenance:
        payload["extra_state"] = {
            "runtime_environment_sha256": SOURCE_ENVIRONMENT_SHA,
            "git": SOURCE_GIT,
        }
        if include_authorization:
            payload["extra_state"]["training_authorization"] = authorization
    torch.save(payload, path)
    integrity = {
        "schema_version": 1,
        "checkpoint": path.name,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_format_version": 1,
        "step": step,
    }
    if include_provenance:
        integrity.update(
            runtime_environment_sha256=SOURCE_ENVIRONMENT_SHA,
            git_revision=SOURCE_GIT["revision"],
            git_branch=SOURCE_GIT["branch"],
            git_dirty=SOURCE_GIT["dirty"],
        )
        if include_authorization:
            integrity.update(
                authorization_stage=authorization["stage"],
                authorization_decision=authorization["decision"],
                authorization_gate_bytes=authorization["gate_bytes"],
                authorization_gate_sha256=authorization["gate_sha256"],
                authorization_gate_identity_sha256=authorization[
                    "gate_identity_sha256"
                ],
            )
    write_json_report(
        checkpoint_integrity_path(path),
        integrity,
    )
    return path


def _completion_audit_path(
    tmp_path: Path,
    *,
    cofitok_export: dict,
    dense_export: dict,
) -> Path:
    def evidence(report: dict, *, smoke_count: int) -> dict:
        execution = report.get("execution")
        execution_git = (
            None if not isinstance(execution, dict) else execution["git"]
        )
        execution_environment_sha = (
            None
            if not isinstance(execution, dict)
            else execution["runtime_environment_sha256"]
        )
        return {
            "artifact_path": report["artifact"],
            "artifact_sha256": report["artifact_sha256"],
            "artifact_bytes": report["artifact_bytes"],
            "export_manifest": report["export_manifest"],
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            "source_checkpoint_bytes": report["source_checkpoint_bytes"],
            "source_runtime_environment_sha256": report[
                "source_runtime_environment_sha256"
            ],
            "source_git": report["source_git"],
            "execution_git": execution_git,
            "export_runtime_environment_sha256": execution_environment_sha,
            "execution_runtime_environment_sha256": execution_environment_sha,
            "training_authorization": report["source_training_authorization"],
            "release_authorization": report["release_authorization"],
            "smoke_output_count": smoke_count,
            "smoke_output_sha256": ["9" * 64] * smoke_count,
        }

    profile = "large_scale_generation_v1"
    inference_evidence = {
        "cofitok": evidence(cofitok_export, smoke_count=4),
        "dense_identity": evidence(dense_export, smoke_count=2),
    }
    contract = _COMPLETION_PROFILES[profile]
    special_evidence = {
        "ten_percent_matched_training": {"expected_revision": "2" * 40},
        "controlled_revision_transition": {
            "training_revision": "1" * 40,
            "target_revision": SOURCE_GIT["revision"],
        },
        "full_matched_training": {
            "expected_revision": SOURCE_GIT["revision"],
            "expected_branch": "scale/generative-system",
        },
        "formal_50k_generation": {
            method: {
                "checkpoint_sha256": row["source_checkpoint_sha256"],
            }
            for method, row in inference_evidence.items()
        },
    }
    audit = {
        "schema_version": 1,
        "profile": profile,
        "status": "complete",
        "complete": True,
        "expected_revisions": {
            "deployment_source": "1" * 40,
            "ten_percent_training": "2" * 40,
            "full_training": SOURCE_GIT["revision"],
        },
        "checks": [
            {
                "name": name,
                "status": "pass",
                "evidence": (
                    inference_evidence
                    if name == contract["check"]
                    else special_evidence.get(name, {"verified": True})
                ),
            }
            for name in _COMPLETION_REQUIRED_CHECKS[profile]
        ],
        "failed_checks": [],
        "missing_checks": [],
        "warnings": [],
    }
    path = tmp_path / "completion_audit.json"
    write_json_report(path, audit)
    return path


def _release_receipt_fixture(
    tmp_path: Path,
    *,
    completion_profile: str = "large_scale_generation_v1",
) -> tuple[Path, Path, Path]:
    cofitok_source_root = tmp_path / "cofitok_source"
    dense_source_root = tmp_path / "dense_source"
    cofitok_source_root.mkdir()
    dense_source_root.mkdir()
    training_authorization = (
        _capacity_training_authorization(tmp_path)
        if completion_profile == "capacity_full_generation_system_v1"
        else _training_authorization(tmp_path)
    )
    cofitok_source = _training_checkpoint(
        cofitok_source_root,
        include_authorization=True,
        step=300_000,
        authorization_root=tmp_path,
        training_authorization=training_authorization,
    )
    dense_source = _training_checkpoint(
        dense_source_root,
        include_authorization=True,
        step=300_000,
        authorization_root=tmp_path,
        training_authorization=training_authorization,
    )
    release_gate = _release_gate_path(tmp_path)
    cofitok_artifact = tmp_path / "cofitok_ema_inference.pt"
    dense_artifact = tmp_path / "dense_ema_inference.pt"
    cofitok_export = export_ema_inference_artifact(
        cofitok_source,
        cofitok_artifact,
        release_gate=release_gate,
        execution=EXPORT_EXECUTION,
    )
    dense_export = export_ema_inference_artifact(
        dense_source,
        dense_artifact,
        release_gate=release_gate,
        execution=EXPORT_EXECUTION,
    )
    audit = _completion_audit_path(
        tmp_path,
        cofitok_export=cofitok_export,
        dense_export=dense_export,
    )
    if completion_profile != "large_scale_generation_v1":
        write_json_report(
            audit,
            _completion_profile_payload(audit, completion_profile),
        )
    receipt = tmp_path / "release_receipt.json"
    write_generation_release_receipt(audit, receipt)
    return cofitok_artifact, audit, receipt


def _completion_profile_payload(audit: Path, profile: str) -> dict:
    payload = json.loads(audit.read_text(encoding="utf-8"))
    if profile == "large_scale_generation_v1":
        return payload
    payload["profile"] = profile
    payload["status"] = "pass"
    payload.pop("expected_revisions")
    inference_evidence = payload["checks"][
        _COMPLETION_REQUIRED_CHECKS["large_scale_generation_v1"].index(
            "deployable_ema_inference_artifacts"
        )
    ]["evidence"]
    contract = _COMPLETION_PROFILES[profile]
    final_gate_sha256 = inference_evidence["cofitok"][
        "release_authorization"
    ]["gate_sha256"]
    if profile == "stability_generation_system_v1":
        payload["expectations"] = {
            "decision_source_revision": "1" * 40,
            "scaling_training_revision": "2" * 40,
            "scaling_evaluation_revision": "3" * 40,
            "full_training_revision": SOURCE_GIT["revision"],
            "full_evaluation_revision": "5" * 40,
            "export_revision": EXPORT_GIT["revision"],
            "decision_sha256": "1" * 64,
            "scaling_gate_sha256": "b" * 64,
            "full_readiness_sha256": "3" * 64,
            "full_launch_receipt_sha256": "4" * 64,
            "final_gate_sha256": final_gate_sha256,
            "scaling_training_branch": "scale/generation-stability-50k",
            "scaling_evaluation_branch": "analysis/generation-stability-eval",
            "full_training_branch": SOURCE_GIT["branch"],
            "full_evaluation_branch": "analysis/generation-full-eval",
            "export_branch": EXPORT_GIT["branch"],
        }
        special_evidence = {
            "stability_5k_authorization": {
                "source_revision": payload["expectations"][
                    "decision_source_revision"
                ],
                "decision_sha256": payload["expectations"]["decision_sha256"],
            },
            "stability_scaling_gate": {
                "gate_sha256": payload["expectations"]["scaling_gate_sha256"],
                "provenance": {
                    "training_revision": "2" * 40,
                    "training_branch": "scale/generation-stability-50k",
                    "evaluation_revision": "3" * 40,
                    "evaluation_branch": "analysis/generation-stability-eval",
                },
            },
            "stability_full_training_readiness": {
                "readiness_sha256": payload["expectations"][
                    "full_readiness_sha256"
                ],
            },
            "stability_full_launch_receipt": {
                "launch_receipt_sha256": payload["expectations"][
                    "full_launch_receipt_sha256"
                ],
                "readiness_sha256": payload["expectations"][
                    "full_readiness_sha256"
                ],
            },
            "stability_full_formal_generation": {
                "methods": {
                    method: {
                        "checkpoint_sha256": row["source_checkpoint_sha256"],
                    }
                    for method, row in inference_evidence.items()
                },
            },
            "stability_final_gate": {
                "gate_sha256": final_gate_sha256,
                "provenance": {
                    "training_revision": SOURCE_GIT["revision"],
                    "training_branch": SOURCE_GIT["branch"],
                    "evaluation_revision": "5" * 40,
                    "evaluation_branch": "analysis/generation-full-eval",
                },
            },
        }
    elif profile == "capacity_full_generation_system_v1":
        payload["expectations"] = {
            "training_revision": SOURCE_GIT["revision"],
            "training_tree": "6" * 40,
            "training_branch": SOURCE_GIT["branch"],
            "evaluation_revision": "7" * 40,
            "evaluation_tree": "8" * 40,
            "evaluation_branch": "analysis/generation-capacity-eval",
            "export_revision": EXPORT_GIT["revision"],
            "export_branch": EXPORT_GIT["branch"],
            "training_supervisor_deployment_sha256": "1" * 64,
            "posteval_supervisor_deployment_sha256": "2" * 64,
            "training_launch_receipt_sha256": inference_evidence["cofitok"][
                "training_authorization"
            ]["gate_sha256"],
            "final_gate_sha256": final_gate_sha256,
        }
        training_git = {
            "revision": payload["expectations"]["training_revision"],
            "tree": payload["expectations"]["training_tree"],
            "branch": payload["expectations"]["training_branch"],
            "tracked_dirty": False,
        }
        evaluation_git = {
            "revision": payload["expectations"]["evaluation_revision"],
            "tree": payload["expectations"]["evaluation_tree"],
            "branch": payload["expectations"]["evaluation_branch"],
            "tracked_dirty": False,
        }
        training_authorization = inference_evidence["cofitok"][
            "training_authorization"
        ]
        special_evidence = {
            "capacity_full_training_supervisor_deployment": {
                "identity": {
                    "path": audit.parent.joinpath(
                        "training_supervisor_deployment.json"
                    )
                    .resolve()
                    .as_posix(),
                    "bytes": 12_345,
                    "sha256": payload["expectations"][
                        "training_supervisor_deployment_sha256"
                    ],
                },
                "git": training_git,
            },
            "capacity_full_training_completion": {
                "training_launch_receipt": {
                    "path": audit.parent.joinpath(
                        "capacity_full_training_launch_receipt.json"
                    )
                    .resolve()
                    .as_posix(),
                    "bytes": training_authorization["gate_bytes"],
                    "sha256": payload["expectations"][
                        "training_launch_receipt_sha256"
                    ],
                },
                "authorization": training_authorization,
            },
            "capacity_full_posteval_supervisor_deployment": {
                "identity": {
                    "path": audit.parent.joinpath(
                        "posteval_supervisor_deployment.json"
                    )
                    .resolve()
                    .as_posix(),
                    "bytes": 23_456,
                    "sha256": payload["expectations"][
                        "posteval_supervisor_deployment_sha256"
                    ],
                },
                "git": evaluation_git,
            },
            "capacity_full_formal_generation": {
                "methods": {
                    method: {
                        "checkpoint_sha256": row["source_checkpoint_sha256"],
                    }
                    for method, row in inference_evidence.items()
                },
            },
            "capacity_full_final_gate": {
                "gate_sha256": final_gate_sha256,
                "provenance": {
                    "training_revision": SOURCE_GIT["revision"],
                    "training_branch": SOURCE_GIT["branch"],
                    "evaluation_revision": "7" * 40,
                    "evaluation_branch": "analysis/generation-capacity-eval",
                },
            }
        }
    else:
        raise AssertionError(f"unsupported test completion profile: {profile}")
    payload["checks"] = [
        {
            "name": name,
            "status": "pass",
            "evidence": (
                inference_evidence
                if name == contract["check"]
                else special_evidence.get(name, {"verified": True})
            ),
        }
        for name in _COMPLETION_REQUIRED_CHECKS[profile]
    ]
    return payload


def _assert_receipt_rejected_before_artifact_verification(
    tmp_path: Path,
    monkeypatch,
    audit: Path,
    payload: dict,
    *,
    match: str,
) -> None:
    write_json_report(audit, payload)

    def fail_if_artifact_verified(*args, **kwargs):
        raise AssertionError("artifact was verified before provenance rejection")

    monkeypatch.setattr(
        generation_release,
        "verify_inference_artifact",
        fail_if_artifact_verified,
    )
    with pytest.raises(ValueError, match=match):
        write_generation_release_receipt(
            audit,
            tmp_path / "invalid_provenance_release_receipt.json",
        )


def test_ema_export_is_smaller_verified_and_sample_equivalent(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    release_gate = _release_gate_path(tmp_path)

    report = export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )
    reused = export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )

    assert report["status"] == "completed"
    assert report["schema_version"] == 4
    assert report["weights"] == "ema_export"
    assert report["verified"] is True
    assert report["artifact_bytes"] < report["source_checkpoint_bytes"]
    assert reused["reused"] is True
    verified = verify_inference_artifact(artifact)
    assert verified["schema_version"] == 4
    assert verified["artifact_format_version"] == 4
    assert verified["artifact_sha256"] == report["artifact_sha256"]
    assert report["source_runtime_environment_sha256"] == SOURCE_ENVIRONMENT_SHA
    assert report["source_git"] == SOURCE_GIT
    assert report["source_training_authorization"] == _training_authorization(
        tmp_path
    )
    assert verified["source_training_authorization"] == _training_authorization(
        tmp_path
    )
    assert report["release_authorization"] == verified["release_authorization"]
    assert report["release_authorization"]["stage"] == "full"
    assert report["release_authorization"]["decision"] == (
        "large_scale_generation_ready"
    )
    manifest_path = inference_export_manifest_path(artifact)
    manifest = verify_inference_export_manifest(
        manifest_path,
        expected_artifact=artifact,
    )
    assert report["export_manifest"] == {
        "path": manifest_path.resolve().as_posix(),
        "bytes": manifest_path.stat().st_size,
        "sha256": file_sha256(manifest_path),
    }
    assert manifest["source"]["sha256"] == report["source_checkpoint_sha256"]
    assert manifest["target"]["artifact"] == artifact.resolve().as_posix()
    assert manifest["release_authorization"] == report["release_authorization"]

    request = GenerationRequest(
        seeds=(9,),
        class_labels=(2,),
        sample_steps=1,
        guidance_scale=1.0,
        precision="fp32",
    )
    source_result = GenerationSession.from_checkpoint(source, weights="ema").generate(
        request
    )
    export_session = GenerationSession.from_checkpoint(
        artifact,
        weights="ema",
        require_release_authorization=True,
    )
    export_result = export_session.generate(request)
    torch.testing.assert_close(
        source_result.images,
        export_result.images,
        rtol=0.0,
        atol=0.0,
    )
    assert export_result.metadata["weights"] == "ema_export"
    assert export_result.metadata["source_checkpoint_sha256"] == report[
        "source_checkpoint_sha256"
    ]
    assert (
        export_result.metadata["source_runtime_environment_sha256"]
        == SOURCE_ENVIRONMENT_SHA
    )
    assert export_result.metadata["source_git"] == SOURCE_GIT
    assert export_result.metadata["training_authorization"] == (
        _training_authorization(tmp_path)
    )
    assert export_result.metadata["release_authorization"] == report[
        "release_authorization"
    ]
    assert export_result.metadata["release_authorization_required"] is True
    preflight = run_sampling_preflight(
        artifact,
        batch_size=1,
        prefix_budget=2,
        guidance_scale=1.0,
        precision="fp32",
        require_release_authorization=True,
    )
    assert preflight["status"] == "passed"
    assert preflight["release_authorization_required"] is True
    assert preflight["training_authorization"] == _training_authorization(
        tmp_path
    )
    assert preflight["release_authorization"] == report[
        "release_authorization"
    ]

    cli_report_path = tmp_path / "cli_export_report.json"
    cli_artifact = tmp_path / "cli_cofitok_ema_inference.pt"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_generation_inference_artifact.py"),
            "--checkpoint",
            str(source),
            "--output",
            str(cli_artifact),
            "--release-gate",
            str(release_gate),
            "--resume",
            "--report",
            str(cli_report_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cli_report = json.loads(cli_report_path.read_text(encoding="utf-8"))
    execution = cli_report["execution"]
    assert len(execution["git"]["revision"]) == 40
    assert execution["runtime_environment"]["device"]["type"] == "cpu"
    assert execution["runtime_environment_sha256"] == (
        runtime_environment_sha256(execution["runtime_environment"])
    )
    cli_manifest = verify_inference_export_manifest(
        inference_export_manifest_path(cli_artifact),
        expected_artifact=cli_artifact,
    )
    assert cli_manifest["execution"] == execution
    assert cli_report["resume_requested"] is True
    assert cli_report["partial_outputs_recovered"] is False


def test_ema_export_rejects_source_without_deployment_provenance(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_provenance=False)

    with pytest.raises(ValueError, match="runtime environment provenance"):
        export_ema_inference_artifact(
            source,
            tmp_path / "unprovenanced_inference.pt",
        )


def test_completed_export_replay_preserves_all_bound_bytes(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    first = export_ema_inference_artifact(source, artifact)
    paths = (
        artifact,
        checkpoint_integrity_path(artifact),
        inference_export_manifest_path(artifact),
    )
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size, file_sha256(path))
        for path in paths
    }

    replay = export_ema_inference_artifact(source, artifact, resume=True)

    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size, file_sha256(path))
        for path in paths
    }
    assert replay["reused"] is True
    assert replay["resume_requested"] is True
    assert replay["partial_outputs_recovered"] is False
    assert replay["artifact_sha256"] == first["artifact_sha256"]
    assert after == before


def test_partial_export_requires_manifest_bound_explicit_resume(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    checkpoint_integrity_path(artifact).unlink()
    stale = artifact.parent / f"{artifact.name}.tmp-stale"
    stale.write_bytes(b"partial temporary")

    with pytest.raises(ValueError, match="requires explicit resume"):
        export_ema_inference_artifact(source, artifact)
    assert artifact.is_file()
    assert stale.is_file()

    recovered = export_ema_inference_artifact(source, artifact, resume=True)

    assert recovered["reused"] is False
    assert recovered["resume_requested"] is True
    assert recovered["partial_outputs_recovered"] is True
    assert not stale.exists()
    verify_inference_artifact(artifact)
    verify_inference_export_manifest(
        inference_export_manifest_path(artifact),
        expected_artifact=artifact,
    )


def test_partial_export_without_manifest_is_never_deleted(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    checkpoint_integrity_path(artifact).unlink()
    inference_export_manifest_path(artifact).unlink()
    artifact_sha = file_sha256(artifact)

    with pytest.raises(ValueError, match="lacks a matching export manifest"):
        export_ema_inference_artifact(source, artifact, resume=True)

    assert artifact.is_file()
    assert file_sha256(artifact) == artifact_sha
    assert not checkpoint_integrity_path(artifact).exists()
    assert not inference_export_manifest_path(artifact).exists()


def test_partial_export_rejects_manifest_drift_before_deserialization(
    tmp_path,
    monkeypatch,
) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    checkpoint_integrity_path(artifact).unlink()
    manifest_path = inference_export_manifest_path(artifact)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution"] = {"unbound": "environment"}
    write_json_report(manifest_path, manifest)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before manifest rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="manifest request differs"):
        export_ema_inference_artifact(source, artifact, resume=True)
    assert artifact.is_file()


def test_complete_but_tampered_export_is_not_repaired_by_resume(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    tampered_sha = file_sha256(artifact)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        export_ema_inference_artifact(source, artifact, resume=True)

    assert file_sha256(artifact) == tampered_sha
    assert checkpoint_integrity_path(artifact).is_file()
    assert inference_export_manifest_path(artifact).is_file()


def test_concurrent_export_fails_closed_before_creating_outputs(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    report = tmp_path / "export_report.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    with exclusive_output_lock(artifact, role="test_export_owner"):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_generation_inference_artifact.py"),
                "--checkpoint",
                str(source),
                "--output",
                str(artifact),
                "--report",
                str(report),
                "--resume",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert "already locked" in result.stderr
    assert not artifact.exists()
    assert not checkpoint_integrity_path(artifact).exists()
    assert not inference_export_manifest_path(artifact).exists()
    assert not report.exists()


def test_inference_artifact_rejects_model_weights_and_tampering(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)

    with pytest.raises(ValueError, match="only exported EMA"):
        GenerationSession.from_checkpoint(artifact, weights="model")

    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_production_mode_rejects_unreleased_inference_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("inference artifact was deserialized before policy rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="release-authorized inference artifact"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            require_release_authorization=True,
        )


def test_inference_artifact_rejects_source_sidecar_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(source, artifact)
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["source_git_revision"] = "b" * 40
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="source Git provenance mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_inference_artifact_rejects_training_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=_release_gate_path(tmp_path),
    )
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["source_training_authorization"][
        "gate_identity_sha256"
    ] = "d" * 64
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="training authorization mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_export_reuse_rejects_source_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    release_gate = _release_gate_path(tmp_path)
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )
    source_integrity_path = checkpoint_integrity_path(source)
    source_integrity = json.loads(
        source_integrity_path.read_text(encoding="utf-8")
    )
    source_integrity["authorization_gate_identity_sha256"] = "d" * 64
    write_json_report(source_integrity_path, source_integrity)

    with pytest.raises(ValueError, match="authorization differs"):
        export_ema_inference_artifact(
            source,
            artifact,
            release_gate=release_gate,
        )


def test_formal_inference_export_requires_release_gate(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)

    with pytest.raises(ValueError, match="requires a full release gate"):
        export_ema_inference_artifact(
            source,
            tmp_path / "unauthorized_inference.pt",
        )


def test_completion_receipt_authorizes_consumer_load_and_preflight(tmp_path) -> None:
    artifact, audit, receipt = _release_receipt_fixture(tmp_path)

    authorization = verify_generation_release_receipt(receipt, artifact)
    session = GenerationSession.from_checkpoint(
        artifact,
        weights="ema",
        completion_receipt=receipt,
        require_completion_authorization=True,
    )
    result = session.generate(
        GenerationRequest(
            seeds=(7,),
            class_labels=(2,),
            sample_steps=1,
            guidance_scale=1.0,
            precision="fp32",
        )
    )
    preflight = run_sampling_preflight(
        artifact,
        batch_size=1,
        prefix_budget=2,
        guidance_scale=1.0,
        precision="fp32",
        completion_receipt=receipt,
        require_completion_authorization=True,
    )

    assert authorization["method"] == "cofitok"
    assert authorization["completion_audit"]["path"] == audit.resolve().as_posix()
    assert result.metadata["completion_authorization"] == authorization
    assert result.metadata["completion_authorization_required"] is True
    assert result.metadata["release_authorization_required"] is True
    assert preflight["status"] == "passed"
    assert preflight["completion_authorization"] == authorization
    assert preflight["completion_authorization_required"] is True


def test_release_receipt_requires_bound_export_manifest_at_consumption(tmp_path) -> None:
    artifact, _, receipt = _release_receipt_fixture(tmp_path)
    manifest = inference_export_manifest_path(artifact)
    manifest.unlink()

    with pytest.raises(FileNotFoundError, match="release source is missing"):
        verify_generation_release_receipt(receipt, artifact)


def test_release_artifact_remains_portable_after_source_checkpoint_archival(
    tmp_path,
) -> None:
    artifact, _, receipt = _release_receipt_fixture(tmp_path)
    manifest = json.loads(
        inference_export_manifest_path(artifact).read_text(encoding="utf-8")
    )
    source = Path(manifest["source"]["path"])
    checkpoint_integrity_path(source).unlink()
    source.unlink()

    authorization = verify_generation_release_receipt(receipt, artifact)

    assert authorization["method"] == "cofitok"


def test_stability_completion_profile_publishes_release_receipt(tmp_path) -> None:
    artifact, audit, receipt = _release_receipt_fixture(
        tmp_path,
        completion_profile="stability_generation_system_v1",
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))

    written = write_generation_release_receipt(audit, receipt)
    authorization = verify_generation_release_receipt(receipt, artifact)

    assert written["completion_profile"] == "stability_generation_system_v1"
    assert written["completion_expectations"] == payload["expectations"]
    assert authorization["completion_profile"] == (
        "stability_generation_system_v1"
    )


def test_capacity_full_completion_profile_publishes_release_receipt(tmp_path) -> None:
    artifact, audit, receipt = _release_receipt_fixture(
        tmp_path,
        completion_profile="capacity_full_generation_system_v1",
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))

    written = write_generation_release_receipt(audit, receipt)
    authorization = verify_generation_release_receipt(receipt, artifact)

    assert written["completion_profile"] == (
        "capacity_full_generation_system_v1"
    )
    assert written["completion_expectations"] == payload["expectations"]
    assert authorization["completion_profile"] == (
        "capacity_full_generation_system_v1"
    )


@pytest.mark.parametrize(
    ("profile", "field", "replacement", "message"),
    (
        (
            "large_scale_generation_v1",
            "full_training",
            "f" * 40,
            "deployment revision differs",
        ),
        (
            "stability_generation_system_v1",
            "full_training_revision",
            "f" * 40,
            "artifact source Git differs",
        ),
        (
            "stability_generation_system_v1",
            "export_revision",
            "e" * 40,
            "artifact execution Git differs",
        ),
        (
            "stability_generation_system_v1",
            "scaling_gate_sha256",
            "c" * 64,
            "training gate differs",
        ),
        (
            "capacity_full_generation_system_v1",
            "training_branch",
            "scale/another-training-branch",
            "capacity training deployment differs",
        ),
        (
            "capacity_full_generation_system_v1",
            "export_branch",
            "analysis/another-export-branch",
            "artifact execution Git differs",
        ),
        (
            "capacity_full_generation_system_v1",
            "final_gate_sha256",
            "f" * 64,
            "release gate differs",
        ),
    ),
)
def test_completion_expectations_cross_bind_artifact_provenance(
    tmp_path,
    monkeypatch,
    profile,
    field,
    replacement,
    message,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile=profile,
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    expectation_key = (
        "expected_revisions"
        if profile == "large_scale_generation_v1"
        else "expectations"
    )
    payload[expectation_key][field] = replacement

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match=message,
    )


def test_completion_expectations_reject_unknown_fields_before_artifact_verification(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile="stability_generation_system_v1",
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["expectations"]["unbound_revision"] = "f" * 40

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="expectations differ",
    )


@pytest.mark.parametrize(
    ("profile", "field"),
    (
        ("large_scale_generation_v1", "full_training"),
        ("stability_generation_system_v1", "final_gate_sha256"),
        ("capacity_full_generation_system_v1", "training_revision"),
    ),
)
def test_completion_expectation_digests_require_string_types_before_artifact_verification(
    tmp_path,
    monkeypatch,
    profile,
    field,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile=profile,
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    expectation_key = (
        "expected_revisions"
        if profile == "large_scale_generation_v1"
        else "expectations"
    )
    payload[expectation_key][field] = int(
        payload[expectation_key][field],
        16,
    )

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match=f"expectation {field} is invalid",
    )


@pytest.mark.parametrize(
    ("profile", "check", "path", "replacement", "message"),
    (
        (
            "stability_generation_system_v1",
            "stability_5k_authorization",
            ("decision_sha256",),
            "f" * 64,
            "stability decision differs",
        ),
        (
            "stability_generation_system_v1",
            "stability_full_training_readiness",
            ("readiness_sha256",),
            "f" * 64,
            "stability readiness differs",
        ),
        (
            "stability_generation_system_v1",
            "stability_full_launch_receipt",
            ("launch_receipt_sha256",),
            "f" * 64,
            "stability launch receipt differs",
        ),
        (
            "capacity_full_generation_system_v1",
            "capacity_full_training_supervisor_deployment",
            ("identity", "sha256"),
            "f" * 64,
            "capacity training deployment differs",
        ),
        (
            "capacity_full_generation_system_v1",
            "capacity_full_posteval_supervisor_deployment",
            ("identity", "sha256"),
            "f" * 64,
            "capacity post-eval deployment differs",
        ),
        (
            "capacity_full_generation_system_v1",
            "capacity_full_training_completion",
            ("training_launch_receipt", "sha256"),
            "f" * 64,
            "capacity training authorization differs",
        ),
    ),
)
def test_completion_expectations_cross_bind_supporting_check_evidence(
    tmp_path,
    monkeypatch,
    profile,
    check,
    path,
    replacement,
    message,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile=profile,
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    check_row = next(row for row in payload["checks"] if row["name"] == check)
    target = check_row["evidence"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match=message,
    )


def test_large_scale_completion_rejects_source_branch_drift_before_artifact_verification(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    inference = next(
        row
        for row in payload["checks"]
        if row["name"] == "deployable_ema_inference_artifacts"
    )["evidence"]
    for method in ("cofitok", "dense_identity"):
        inference[method]["source_git"]["branch"] = "scale/forged-branch"

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="artifact source branch differs",
    )


def test_capacity_completion_cross_binds_artifact_training_authorization_to_completion_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile="capacity_full_generation_system_v1",
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    completion = next(
        row
        for row in payload["checks"]
        if row["name"] == "capacity_full_training_completion"
    )["evidence"]
    completion["authorization"]["gate_identity_sha256"] = "f" * 64

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="capacity training authorization evidence differs",
    )


@pytest.mark.parametrize(
    ("profile", "check", "methods_field"),
    (
        ("large_scale_generation_v1", "formal_50k_generation", None),
        (
            "stability_generation_system_v1",
            "stability_full_formal_generation",
            "methods",
        ),
        (
            "capacity_full_generation_system_v1",
            "capacity_full_formal_generation",
            "methods",
        ),
    ),
)
def test_completion_receipt_cross_binds_evaluated_and_exported_checkpoints(
    tmp_path,
    monkeypatch,
    profile,
    check,
    methods_field,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile=profile,
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    formal = next(row for row in payload["checks"] if row["name"] == check)[
        "evidence"
    ]
    methods = formal if methods_field is None else formal[methods_field]
    methods["cofitok"]["checkpoint_sha256"] = "f" * 64

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="cofitok evaluated checkpoint differs",
    )


def test_capacity_completion_cross_binds_training_receipt_identity(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile="capacity_full_generation_system_v1",
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    completion = next(
        row
        for row in payload["checks"]
        if row["name"] == "capacity_full_training_completion"
    )["evidence"]
    completion["training_launch_receipt"]["bytes"] += 1

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="capacity training receipt identity differs",
    )


@pytest.mark.parametrize(
    ("authorization_field", "message"),
    (
        ("training_authorization", "training authorization is invalid"),
        ("release_authorization", "release authorization is invalid"),
    ),
)
def test_completion_receipt_validates_authorization_schemas_before_artifact_verification(
    tmp_path,
    monkeypatch,
    authorization_field,
    message,
) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    inference = next(
        row
        for row in payload["checks"]
        if row["name"] == "deployable_ema_inference_artifacts"
    )["evidence"]
    for method in ("cofitok", "dense_identity"):
        inference[method][authorization_field].pop("gate_identity_sha256")

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match=message,
    )


def test_completion_receipt_rejects_non_string_artifact_digest_before_artifact_verification(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    inference = next(
        row
        for row in payload["checks"]
        if row["name"] == "deployable_ema_inference_artifacts"
    )["evidence"]
    inference["cofitok"]["artifact_sha256"] = int(
        inference["cofitok"]["artifact_sha256"],
        16,
    )

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="cofitok completion artifact evidence is malformed",
    )


def test_completion_receipt_rejects_final_gate_evidence_drift_before_artifact_verification(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(
        tmp_path,
        completion_profile="capacity_full_generation_system_v1",
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    final_gate = next(
        row
        for row in payload["checks"]
        if row["name"] == "capacity_full_final_gate"
    )
    final_gate["evidence"]["gate_sha256"] = "f" * 64

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="final-gate evidence differs",
    )


def test_completion_receipt_rejects_cross_method_provenance_drift_before_artifact_verification(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    payload = _completion_profile_payload(audit, "large_scale_generation_v1")
    inference = next(
        row
        for row in payload["checks"]
        if row["name"] == "deployable_ema_inference_artifacts"
    )["evidence"]
    inference["dense_identity"]["source_runtime_environment_sha256"] = "f" * 64

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="different source_runtime_environment_sha256",
    )


def test_completion_receipt_rejects_reused_source_checkpoint_before_artifact_verification(
    tmp_path,
    monkeypatch,
) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    payload = _completion_profile_payload(audit, "large_scale_generation_v1")
    inference = next(
        row
        for row in payload["checks"]
        if row["name"] == "deployable_ema_inference_artifacts"
    )["evidence"]
    inference["dense_identity"]["source_checkpoint_sha256"] = inference[
        "cofitok"
    ]["source_checkpoint_sha256"]

    _assert_receipt_rejected_before_artifact_verification(
        tmp_path,
        monkeypatch,
        audit,
        payload,
        match="distinct source checkpoints",
    )


def test_completion_authorization_requires_receipt_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    artifact, _, _ = _release_receipt_fixture(tmp_path)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("artifact was deserialized before receipt policy rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="requires a release receipt"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            require_completion_authorization=True,
        )


def test_completion_receipt_rejects_changed_terminal_audit_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    artifact, audit, receipt = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["warnings"] = ["post-release mutation"]
    write_json_report(audit, changed)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("artifact was deserialized before audit drift rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="changed after release"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            completion_receipt=receipt,
            require_completion_authorization=True,
        )


def test_release_receipt_rejects_incomplete_completion_audit(tmp_path) -> None:
    artifact, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed.update(status="in_progress", complete=False)
    changed["missing_checks"] = ["formal_50k_generation"]
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="did not pass"):
        write_generation_release_receipt(
            audit,
            tmp_path / "invalid_release_receipt.json",
        )

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="not authorized"):
        verify_generation_release_receipt(
            _release_receipt_fixture(other)[2],
            artifact,
        )


def test_release_receipt_rejects_forged_minimal_completion_audit(tmp_path) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"] = [
        row
        for row in changed["checks"]
        if row["name"] == "deployable_ema_inference_artifacts"
    ]
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="missing required checks"):
        write_generation_release_receipt(
            audit,
            tmp_path / "forged_minimal_release_receipt.json",
        )


def test_release_receipt_rejects_duplicate_completion_check(tmp_path) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"].append(dict(changed["checks"][0]))
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="duplicate check"):
        write_generation_release_receipt(
            audit,
            tmp_path / "duplicate_check_release_receipt.json",
        )


def test_release_receipt_rejects_non_passing_check_hidden_by_summary(tmp_path) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"][0]["status"] = "fail"
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="requires every check to pass"):
        write_generation_release_receipt(
            audit,
            tmp_path / "hidden_failed_check_release_receipt.json",
        )


def test_release_receipt_rejects_passing_check_without_evidence(tmp_path) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"][0]["evidence"] = None
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="lacks structured check evidence"):
        write_generation_release_receipt(
            audit,
            tmp_path / "missing_check_evidence_release_receipt.json",
        )


def test_release_receipt_rejects_unknown_completion_check(tmp_path) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"].append(
        {
            "name": "forged_terminal_check",
            "status": "pass",
            "evidence": {"verified": True},
        }
    )
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="unknown checks"):
        write_generation_release_receipt(
            audit,
            tmp_path / "unknown_check_release_receipt.json",
        )


def test_release_receipt_rejects_reordered_completion_contract(tmp_path) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"][0], changed["checks"][1] = (
        changed["checks"][1],
        changed["checks"][0],
    )
    write_json_report(audit, changed)

    with pytest.raises(ValueError, match="check order differs"):
        write_generation_release_receipt(
            audit,
            tmp_path / "reordered_check_release_receipt.json",
        )


def test_invalid_completion_contract_is_rejected_before_artifact_verification(
    tmp_path, monkeypatch
) -> None:
    _, audit, _ = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"].pop(0)
    write_json_report(audit, changed)

    def fail_if_artifact_verified(*args, **kwargs):
        raise AssertionError("artifact was verified before completion rejection")

    monkeypatch.setattr(
        generation_release,
        "verify_inference_artifact",
        fail_if_artifact_verified,
    )
    with pytest.raises(ValueError, match="missing required checks"):
        write_generation_release_receipt(
            audit,
            tmp_path / "preverification_rejection_receipt.json",
        )


def test_invalid_completion_contract_is_rejected_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    artifact, audit, receipt = _release_receipt_fixture(tmp_path)
    changed = json.loads(audit.read_text(encoding="utf-8"))
    changed["checks"].pop(0)
    write_json_report(audit, changed)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["completion_audit"] = {
        "path": audit.resolve().as_posix(),
        "bytes": audit.stat().st_size,
        "sha256": file_sha256(audit),
    }
    write_json_report(receipt, receipt_payload)

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("artifact was deserialized before completion rejection")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="missing required checks"):
        GenerationSession.from_checkpoint(
            artifact,
            weights="ema",
            completion_receipt=receipt,
            require_completion_authorization=True,
        )


def test_formal_export_rejects_release_source_drift_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    release_gate = _release_gate_path(tmp_path)
    gate = json.loads(release_gate.read_text(encoding="utf-8"))
    source_report = Path(gate["source_reports"]["dense_generation"]["path"])
    source_report.write_text("changed\n", encoding="ascii")

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before gate-source validation")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="source report changed after binding"):
        export_ema_inference_artifact(
            source,
            tmp_path / "source_drift_inference.pt",
            release_gate=release_gate,
        )


def test_inference_artifact_rejects_release_authorization_drift(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=_release_gate_path(tmp_path),
    )
    integrity_path = checkpoint_integrity_path(artifact)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    integrity["release_authorization"]["gate_identity_sha256"] = "d" * 64
    write_json_report(integrity_path, integrity)

    with pytest.raises(ValueError, match="release authorization mismatch"):
        GenerationSession.from_checkpoint(artifact, weights="ema")


def test_export_reuse_rejects_changed_release_gate(tmp_path) -> None:
    source = _training_checkpoint(tmp_path, include_authorization=True)
    artifact = tmp_path / "cofitok_ema_inference.pt"
    release_gate = _release_gate_path(tmp_path)
    export_ema_inference_artifact(
        source,
        artifact,
        release_gate=release_gate,
    )
    changed_gate = json.loads(release_gate.read_text(encoding="utf-8"))
    changed_gate["gates"].append(
        {"name": "additional_release_audit", "passed": True, "evidence": {}}
    )
    release_gate.write_text(
        json.dumps(changed_gate, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="release authorization differs"):
        export_ema_inference_artifact(
            source,
            artifact,
            release_gate=release_gate,
        )


def test_formal_export_runbook_binds_full_release_gate() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runbooks/generation_export_inference_artifacts.sh"
    ).read_text(encoding="utf-8")

    assert runbook.count('--release-gate "$FINAL_GATE"') == 2
    assert runbook.count("--resume") == 2
    assert runbook.count(".export_manifest.json") == 2
    assert runbook.count('test -f "$COFITOK_EXPORT_MANIFEST"') == 1
    assert runbook.count('test -f "$DENSE_EXPORT_MANIFEST"') == 1
