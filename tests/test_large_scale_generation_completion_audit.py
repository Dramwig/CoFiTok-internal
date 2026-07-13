from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

from scripts.audit_large_scale_generation_completion import (
    MILESTONE_STEPS,
    build_completion_audit,
)
from cofitok.environment import runtime_environment_sha256


TEN_REVISION = "a" * 40
FULL_REVISION = "b" * 40
REAL_SET_SHA = "e" * 64
ROOT = Path(__file__).resolve().parents[1]


def _runtime_environment() -> dict:
    return {
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.10.20",
            "executable": "/env/bin/python",
        },
        "platform": {"system": "Linux", "release": "test", "machine": "x86_64"},
        "packages": {
            "numpy": "2.0.0",
            "pillow": "11.0.0",
            "torch": "2.7.1",
            "torchvision": "0.22.1",
            "tqdm": "4.67.0",
        },
        "torch": {
            "version": "2.7.1+cu128",
            "cuda_version": "12.8",
            "cudnn_version": 90701,
        },
        "device": {"type": "cuda", "name": "NVIDIA RTX PRO 6000"},
        "environment_variables": {},
        "project_files": {
            "pyproject.toml": {"sha256": "1" * 64},
            "uv.lock": {"sha256": "2" * 64},
        },
    }


def _training(
    *, steps: int, dataset: str, revision: str, parameters: int, checkpoint_sha: str
) -> dict:
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    return {
        "training_complete": True,
        "completed_steps": steps,
        "target_steps": steps,
        "parameter_count": parameters,
        "elapsed_seconds": 100_000.0,
        "peak_vram_bytes": 24 * 1024**3,
        "final_metrics": {"samples_seen": steps * 64},
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": environment_sha,
        "git": {
            "dirty": False,
            "revision": revision,
            "branch": "scale/generative-system",
        },
        "latest_checkpoint": {
            "checkpoint": f"checkpoint_step_{steps:08d}.pt",
            "step": steps,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": 1_000,
            "integrity_manifest": f"checkpoint_step_{steps:08d}.pt.integrity.json",
            "runtime_environment_sha256": environment_sha,
            "git_revision": revision,
            "git_branch": "scale/generative-system",
            "git_dirty": False,
        },
        "config": {
            "data": {"dataset": dataset, "batch_size": 16},
            "diffusion": {"schedule_type": "cosine"},
            "runtime": {"precision": "bf16", "device": "cuda"},
            "optimization": {"gradient_accumulation_steps": 4},
            "model": {
                "token_count": 8 if parameters > 100_000 else 1,
                "token_channels": 64 if parameters > 100_000 else 3,
                "image_channels": 3,
                "image_size": 256,
                "base_channels": 128,
                "predictor_type": "scalable_unet",
                "predictor_use_feedback": parameters > 100_000,
                "num_classes": 1000,
                "class_dropout_prob": 0.1,
                "synthesis_mode": (
                    "restricted" if parameters > 100_000 else "dense_identity"
                ),
            },
            "loss": {
                "epsilon_weight": 1.0,
                "denoise_path_component_weight": 0.1 if parameters > 100_000 else 0.0,
            },
        },
    }


def _gate(stage: str) -> dict:
    decision = (
        "promote_to_full_imagenet256"
        if stage == "scaling"
        else "large_scale_generation_ready"
    )
    environment_sha = runtime_environment_sha256(_runtime_environment())
    gates = [{"name": "all_evidence", "passed": True}]
    if stage == "full":
        gates.extend(
            {
                "name": name,
                "passed": True,
                "evidence": (
                    {
                        "cofitok": {
                            "sha256": environment_sha
                        },
                        "dense_identity": {
                            "sha256": environment_sha
                        },
                    }
                    if name
                    in {
                        "matched_sampling_runtime_environment",
                        "matched_evaluator_runtime_environment",
                    }
                    else (
                        {
                            method: {
                                "valid": True,
                                "digest_schema": "cofitok_image_tree_sha256_v1",
                                "sha256": REAL_SET_SHA,
                                "root": "/datasets/imagenet_256/validation",
                                "image_count": 50_000,
                                "real_cache_name": (
                                    "imagenet256_val__cofitok_" + REAL_SET_SHA[:16]
                                ),
                            }
                            for method in ("cofitok", "dense_identity")
                        }
                        if name == "matched_real_set_provenance"
                        else {}
                    )
                ),
            }
            for name in (
                "generation_metrics_complete",
                "matched_real_set_provenance",
                "matched_sampling_code_provenance",
                "matched_sampling_runtime_environment",
                "matched_evaluator_code_provenance",
                "matched_evaluator_runtime_environment",
                "matched_checkpoint_evaluator_code_provenance",
                "distribution_metric_ranges",
                "fid_within_tolerance",
                "absolute_fid_quality",
                "full_precision_recall_quality",
                "endpoint_within_tolerance",
                "ordered_prefix_path",
                "restricted_synthesis_contract",
                "shuffle_mismatch",
                "full_training_checkpoint_integrity",
            )
        )
        gates.append(
            {
                "name": "matched_sampling_provenance",
                "passed": True,
                "evidence": {
                    "cofitok_checkpoint_sha256": "a" * 64,
                    "dense_checkpoint_sha256": "b" * 64,
                    "cofitok_sample_set_sha256": "A" * 64,
                    "dense_sample_set_sha256": "B" * 64,
                },
            }
        )
    return {
        "stage": stage,
        "status": "pass",
        "decision": decision,
        "gates": gates,
        "thresholds": {
            "max_fid_regression": 0.05,
            "max_absolute_fid": 20.0,
            "max_endpoint_regression": 0.05,
            "min_precision": 0.30,
            "min_recall": 0.30,
            "max_precision_regression": 0.05,
            "max_recall_regression": 0.05,
        },
        "summary": {
            "cofitok_fid": 19.0,
            "dense_fid": 19.0,
            "cofitok_precision": 0.6,
            "dense_precision": 0.6,
            "cofitok_recall": 0.4,
            "dense_recall": 0.4,
        },
    }


def _training_audit() -> dict:
    return {
        "status": "complete",
        "issues": [],
        "last_step": 300_000,
        "validation": {"logging_complete": True, "event_count": 150},
        "checkpoint": {
            "steps": list(MILESTONE_STEPS),
            "missing_required_steps": [],
        },
    }


def _runtime_selection() -> dict:
    return {
        "status": "selected",
        "git_revision": FULL_REVISION,
        "selected": {
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 64,
            "estimated_speedup_over_16x4": 1.0,
        },
    }


def _milestone(step: int, alerts: list[str] | None = None) -> dict:
    row = {
        "checkpoint_step": step,
        "sample_count": 2_048,
        "fid": 20.0,
    }
    return {
        "status": "completed",
        "milestone_step": step,
        "expected_samples": 2_048,
        "methods": {"cofitok": row, "dense_identity": dict(row)},
        "quality_alerts": alerts or [],
    }


def _generation(seed: str) -> dict:
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    return {
        "status": "completed",
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "runtime_environment": copy.deepcopy(runtime_environment),
        "runtime_environment_sha256": environment_sha,
        "paths": {"real_dir": "/datasets/imagenet_256/validation"},
        "counts": {"real_image_count": 50_000, "generated_image_count": 50_000},
        "real_set": {
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": REAL_SET_SHA,
            "root": "/datasets/imagenet_256/validation",
            "image_count": 50_000,
        },
        "parameters": {
            "real_cache_name": "imagenet256_val__cofitok_" + REAL_SET_SHA[:16]
        },
        "metrics": {
            "frechet_inception_distance": 19.0,
            "precision": 0.6,
            "recall": 0.4,
        },
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "sample_provenance": {
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": environment_sha,
            "git": {
                "revision": FULL_REVISION,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
            "checkpoint_step": 300_000,
            "weights": "ema",
            "checkpoint_sha256": seed * 64,
            "sample_set_sha256": seed.upper() * 64,
            "checkpoint_integrity_manifest": "/run/checkpoint_step_00300000.pt.integrity.json",
            "sampling_progress": {
                "status": "completed",
                "completed_samples": 50_000,
                "cumulative_elapsed_seconds": 10_000.0,
            },
            "sampling": {
                "batch_size": 64,
                "sample_steps": 250,
                "guidance_scale": 1.5,
                "inference_api": {
                    "name": "cofitok.generation.GenerationSession",
                    "version": 1,
                },
                "random_stream": {"batch_size_invariant": True},
            },
        },
    }


def _official_related() -> dict:
    rows = []
    for alias, method, fid in (
        ("d_ar", "D-AR", 2.6281),
        ("mar", "MAR", 2.3385),
        ("retok", "ReTok", 2.2189),
    ):
        rows.append(
            {
                "alias": alias,
                "method": method,
                "dataset": "imagenet_256",
                "resolution": 256,
                "status": "completed_eval_only_50k",
                "sample_count": 50_000,
                "fid": fid,
                "inception_score": 200.0,
                "precision": 0.8,
                "recall": 0.6,
                "protocol": "official pretrained eval-only",
                "paper_table_role": "secondary related-method only",
                "metrics_txt": f"/{alias}.txt",
            }
        )
    return {"schema_version": 1, "rows": rows}


def _comparison() -> dict:
    official = _official_related()
    return {
        "schema_version": 3,
        "status": "ready",
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "external_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
        },
        "official_context_source": {
            "path": "/reports/official_related_methods_table.json",
            "sha256": "9" * 64,
            "schema_version": 1,
        },
        "matched_training_rows": [
            {
                "method": "CoFiTok K=8",
                "comparison_tier": "matched_training_direct",
                "directly_comparable_to_cofitok": True,
                "dataset": "imagenet_256",
                "resolution": 256,
                "training_steps": 300_000,
                "parameter_count": 100_500,
                "effective_batch_size": 64,
                "training_images_seen": 19_200_000,
                "training_elapsed_seconds": 100_000.0,
                "training_images_per_second": 192.0,
                "peak_vram_bytes": 24 * 1024**3,
                "sample_count": 50_000,
                "sample_batch_size": 64,
                "sampling_elapsed_seconds": 10_000.0,
                "sampling_images_per_second": 5.0,
                "checkpoint_sha256": "a" * 64,
                "sample_set_sha256": "A" * 64,
                "real_set_digest_schema": "cofitok_image_tree_sha256_v1",
                "real_set_sha256": REAL_SET_SHA,
                "real_image_count": 50_000,
                "evaluator_runtime_environment_sha256": runtime_environment_sha256(
                    _runtime_environment()
                ),
            },
            {
                "method": "Dense identity",
                "comparison_tier": "matched_training_direct",
                "directly_comparable_to_cofitok": True,
                "dataset": "imagenet_256",
                "resolution": 256,
                "training_steps": 300_000,
                "parameter_count": 100_000,
                "effective_batch_size": 64,
                "training_images_seen": 19_200_000,
                "training_elapsed_seconds": 100_000.0,
                "training_images_per_second": 192.0,
                "peak_vram_bytes": 24 * 1024**3,
                "sample_count": 50_000,
                "sample_batch_size": 64,
                "sampling_elapsed_seconds": 10_000.0,
                "sampling_images_per_second": 5.0,
                "checkpoint_sha256": "b" * 64,
                "sample_set_sha256": "B" * 64,
                "real_set_digest_schema": "cofitok_image_tree_sha256_v1",
                "real_set_sha256": REAL_SET_SHA,
                "real_image_count": 50_000,
                "evaluator_runtime_environment_sha256": runtime_environment_sha256(
                    _runtime_environment()
                ),
            },
        ],
        "official_context_rows": [
            {
                "alias": row["alias"],
                "method": row["method"],
                "comparison_tier": "official_pretrained_contextual",
                "directly_comparable_to_cofitok": False,
                "dataset": row["dataset"],
                "resolution": row["resolution"],
                "training_steps": None,
                "parameter_count": None,
                "sample_count": row["sample_count"],
                "fid": row["fid"],
                "inception_score": row["inception_score"],
                "precision": row["precision"],
                "recall": row["recall"],
                "protocol_note": row["protocol"],
                "source_metrics": row["metrics_txt"],
                "source_status": row["status"],
                "paper_table_role": row["paper_table_role"],
            }
            for row in official["rows"]
        ],
    }


def _sampling_runtime_selection() -> dict:
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    return {
        "status": "selected",
        "git_revision": FULL_REVISION,
        "runtime_environment_sha256": environment_sha,
        "policy": {
            "shared_candidate_required": True,
            "batch_size_invariant_random_stream_required": True,
        },
        "selected": {
            "batch_size": 64,
            "estimated_speedup_over_baseline": 1.4,
        },
        "candidates": [
            {
                "batch_size": 64,
                "eligible": True,
                "methods": {
                    "cofitok": {
                        "runtime_environment": runtime_environment,
                        "runtime_environment_sha256": environment_sha,
                        "git": {
                            "revision": FULL_REVISION,
                            "tracked_dirty": False,
                        }
                    },
                    "dense_identity": {
                        "runtime_environment": runtime_environment,
                        "runtime_environment_sha256": environment_sha,
                        "git": {
                            "revision": FULL_REVISION,
                            "tracked_dirty": False,
                        }
                    },
                },
            }
        ],
        "checkpoints": {
            "cofitok": {"sha256": "a" * 64, "step": 300_000},
            "dense_identity": {"sha256": "b" * 64, "step": 300_000},
        },
    }


def _visual_audit() -> dict:
    return {
        "status": "completed",
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "claim_policy": {"quantitative_metric": False},
        "indices": [0, 1],
        "prefix_indices": [0, 1],
        "prefix_budgets": [1, 2, 4, 8],
        "sources": {
            "cofitok": {
                "checkpoint_sha256": "a" * 64,
                "sample_set_sha256": "A" * 64,
            },
            "dense_identity": {
                "checkpoint_sha256": "b" * 64,
                "sample_set_sha256": "B" * 64,
            },
        },
        "statistics": {
            "cofitok": {"exact_duplicate_count": 0, "pixel_std": 0.2},
            "dense_identity": {"exact_duplicate_count": 0, "pixel_std": 0.3},
        },
        "panels": {
            "cofitok": {"sha256": "c" * 64, "image_count": 2},
            "dense_identity": {"sha256": "d" * 64, "image_count": 2},
            "cofitok_prefix_paths": {"sha256": "e" * 64, "image_count": 8},
        },
    }


def _storage_preflights() -> dict:
    requirements = {
        "10pct_posteval": (0, 20_256, 32),
        "full_training": (16, 16_384, 64),
        "full_posteval": (0, 100_256, 64),
    }
    reports = {}
    for stage, (checkpoint_count, sample_count, safety_gib) in requirements.items():
        checkpoint_bytes = 1_000 if checkpoint_count else 0
        checkpoint_reserve = checkpoint_count * checkpoint_bytes
        sample_reserve = sample_count * 256 * 1024
        additional = 16 * 1024**3
        safety = safety_gib * 1024**3
        required = checkpoint_reserve + sample_reserve + additional + safety
        free = required + 1024**3
        reports[stage] = {
            "role": "generation_storage_capacity_preflight",
            "stage": stage,
            "status": "pass",
            "git": {
                "revision": FULL_REVISION,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
            "filesystem": {
                "path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
                "total_bytes": free + 2 * 1024**3,
                "used_bytes": 1024**3,
                "free_bytes": free,
            },
            "plan": {
                "checkpoint_count": checkpoint_count,
                "checkpoint_bytes_each": checkpoint_bytes,
                "checkpoint_reserve_bytes": checkpoint_reserve,
                "sample_count": sample_count,
                "estimated_sample_bytes_each": 256 * 1024,
                "sample_reserve_bytes": sample_reserve,
                "additional_bytes": additional,
                "safety_margin_bytes": safety,
                "required_free_bytes": required,
            },
            "headroom_bytes": free - required,
        }
    return reports


def _full_training_monitor() -> dict:
    runs = {}
    for method, directory in (
        ("cofitok", "imagenet256_full_cofitok_k8_300k"),
        ("dense_identity", "imagenet256_full_dense_300k"),
    ):
        runs[method] = {
            "run_dir": (
                "/root/autodl-tmp/CoFiTok/checkpoints/generation/" + directory
            ),
            "complete": True,
            "expected_steps": 300_000,
            "last_step": 300_000,
            "metric_rows": 6_000,
            "health_issues": [],
            "checkpoints": [
                {"step": step, "bytes": 1_000}
                for step in (50_000, 100_000, 200_000, 300_000)
            ],
        }
    return {
        "schema_version": 2,
        "monitor": "generation_full_matched_300k",
        "status": "pass",
        "stage": "complete",
        "issues": [],
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "runs": runs,
    }


def _inference_exports() -> dict:
    output = {}
    environment_sha = runtime_environment_sha256(_runtime_environment())
    source_git = {
        "dirty": False,
        "revision": FULL_REVISION,
        "branch": "scale/generative-system",
    }
    for method, source_sha, artifact_sha, count in (
        ("cofitok", "a" * 64, "f" * 64, 4),
        ("dense_identity", "b" * 64, "9" * 64, 2),
    ):
        artifact = (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/exports/"
            f"imagenet256_full_300k/{method}_ema_inference.pt"
        )
        output[f"{method}_export"] = {
            "status": "completed",
            "verified": True,
            "weights": "ema_export",
            "checkpoint_step": 300_000,
            "source_checkpoint_sha256": source_sha,
            "source_checkpoint_bytes": 1_000,
            "source_runtime_environment_sha256": environment_sha,
            "source_git": dict(source_git),
            "artifact_sha256": artifact_sha,
            "artifact_bytes": 400,
            "artifact": artifact,
            "artifact_integrity_manifest": f"{artifact}.integrity.json",
        }
        output[f"{method}_preflight"] = {
            "status": "passed",
            "checkpoint_sha256": artifact_sha,
            "artifact_type": "cofitok_generation_inference",
            "weights": "ema_export",
            "source_checkpoint_sha256": source_sha,
            "source_runtime_environment_sha256": environment_sha,
            "source_git": dict(source_git),
        }
        output[f"{method}_smoke"] = {
            "status": "completed",
            "output_count": count,
            "checkpoint": {
                "checkpoint_sha256": artifact_sha,
                "artifact_type": "cofitok_generation_inference",
                "source_checkpoint_sha256": source_sha,
                "source_runtime_environment_sha256": environment_sha,
                "source_git": dict(source_git),
            },
            "outputs": [{"sha256": str(index) * 64} for index in range(1, count + 1)],
        }
    return output


def _full_checkpoint_files() -> dict:
    output = {}
    environment_sha = runtime_environment_sha256(_runtime_environment())
    for method, run, checkpoint_sha in (
        ("cofitok", "imagenet256_full_cofitok_k8_300k", "a" * 64),
        ("dense_identity", "imagenet256_full_dense_300k", "b" * 64),
    ):
        checkpoint = "checkpoint_step_00300000.pt"
        path = f"/root/autodl-tmp/CoFiTok/checkpoints/generation/{run}/{checkpoint}"
        output[method] = {
            "status": "verified",
            "path": path,
            "integrity_manifest": f"{path}.integrity.json",
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": 1_000,
            "checkpoint_format_version": 1,
            "step": 300_000,
            "runtime_environment_sha256": environment_sha,
            "git_revision": FULL_REVISION,
            "git_branch": "scale/generative-system",
            "git_dirty": False,
        }
    return output


def _inference_artifact_files() -> dict:
    output = {}
    environment_sha = runtime_environment_sha256(_runtime_environment())
    for method, source_sha, artifact_sha in (
        ("cofitok", "a" * 64, "f" * 64),
        ("dense_identity", "b" * 64, "9" * 64),
    ):
        path = (
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/exports/"
            f"imagenet256_full_300k/{method}_ema_inference.pt"
        )
        output[method] = {
            "status": "verified",
            "path": path,
            "integrity_manifest": f"{path}.integrity.json",
            "artifact_sha256": artifact_sha,
            "artifact_bytes": 400,
            "source_checkpoint_sha256": source_sha,
            "source_runtime_environment_sha256": environment_sha,
            "source_git_revision": FULL_REVISION,
            "source_git_branch": "scale/generative-system",
            "source_git_dirty": False,
            "step": 300_000,
        }
    return output


def _kwargs() -> dict:
    return {
        "expected_10pct_revision": TEN_REVISION,
        "expected_full_revision": FULL_REVISION,
        "cofitok_10pct_training": _training(
            steps=50_000,
            dataset="imagenet_256_10pct",
            revision=TEN_REVISION,
            parameters=100_500,
            checkpoint_sha="c" * 64,
        ),
        "dense_10pct_training": _training(
            steps=50_000,
            dataset="imagenet_256_10pct",
            revision=TEN_REVISION,
            parameters=100_000,
            checkpoint_sha="d" * 64,
        ),
        "deployment_receipt": {
            "schema_version": 1,
            "status": "pass",
            "expected_training_revision": TEN_REVISION,
            "target_revision": FULL_REVISION,
            "git": {
                "revision": FULL_REVISION,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
            "bundle": {
                "bytes": 1234,
                "sha256": "e" * 64,
                "heads": [FULL_REVISION],
            },
            "training_pair_validation": {
                "status": "pass",
                "expected_revision": TEN_REVISION,
                "sha256": "f" * 64,
            },
            "verification": {
                "pytest": "pass",
                "runbook_syntax": "pass",
                "untracked_target_conflicts": 0,
            },
        },
        "storage_preflights": _storage_preflights(),
        "expected_storage_path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
        "full_training_monitor": _full_training_monitor(),
        "full_checkpoint_files": _full_checkpoint_files(),
        "scaling_gate": _gate("scaling"),
        "cofitok_full_training": _training(
            steps=300_000,
            dataset="imagenet_256",
            revision=FULL_REVISION,
            parameters=100_500,
            checkpoint_sha="a" * 64,
        ),
        "dense_full_training": _training(
            steps=300_000,
            dataset="imagenet_256",
            revision=FULL_REVISION,
            parameters=100_000,
            checkpoint_sha="b" * 64,
        ),
        "cofitok_training_audit": _training_audit(),
        "dense_training_audit": _training_audit(),
        "runtime_selection": _runtime_selection(),
        "sampling_runtime_selection": _sampling_runtime_selection(),
        "visual_audit": _visual_audit(),
        "inference_exports": _inference_exports(),
        "inference_artifact_files": _inference_artifact_files(),
        "milestones": {step: _milestone(step) for step in MILESTONE_STEPS},
        "cofitok_generation": _generation("a"),
        "dense_generation": _generation("b"),
        "final_gate": _gate("full"),
        "comparison": _comparison(),
        "official_related": _official_related(),
        "official_related_sha256": "9" * 64,
    }


def test_completion_audit_requires_every_large_scale_artifact() -> None:
    report = build_completion_audit(**_kwargs())

    assert report["status"] == "complete"
    assert report["complete"] is True
    assert report["failed_checks"] == []
    assert report["missing_checks"] == []
    assert len(report["checks"]) == 18


def test_completion_audit_requires_controlled_revision_transition() -> None:
    kwargs = _kwargs()
    kwargs["deployment_receipt"]["target_revision"] = "z" * 40

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["controlled_revision_transition"]


def test_completion_audit_reports_missing_work_as_in_progress() -> None:
    kwargs = _kwargs()
    kwargs["dense_full_training"] = None

    report = build_completion_audit(**kwargs)

    assert report["status"] == "in_progress"
    assert report["complete"] is False
    assert report["missing_checks"] == [
        "full_matched_training",
        "full_training_runtime_environment",
        "full_training_checkpoint_code_provenance",
        "reproducible_full_checkpoint_files",
        "full_runtime_selection",
        "formal_50k_generation",
        "deployable_ema_inference_artifacts",
        "final_comparison_report",
    ]


def test_completion_audit_rejects_failed_final_gate() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["status"] = "fail"
    kwargs["final_gate"]["decision"] = "hold"

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_generation_gate"]


def test_completion_audit_rejects_weakened_final_quality_threshold() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["thresholds"]["max_absolute_fid"] = 100.0

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_generation_gate"]


def test_completion_audit_rejects_unbound_final_quality_metrics() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["summary"]["cofitok_recall"] = 0.31

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_generation_gate"]


def test_completion_audit_requires_named_code_provenance_gates() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["gates"] = [
        gate
        for gate in kwargs["final_gate"]["gates"]
        if gate["name"] != "matched_evaluator_code_provenance"
    ]

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_generation_gate"]


def test_completion_audit_rejects_incomplete_formal_sampling() -> None:
    kwargs = _kwargs()
    kwargs["cofitok_generation"]["sample_provenance"]["sampling_progress"][
        "completed_samples"
    ] = 49_999

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_stale_gate_and_comparison_provenance() -> None:
    kwargs = _kwargs()
    gate_sampling = next(
        row["evidence"]
        for row in kwargs["final_gate"]["gates"]
        if row["name"] == "matched_sampling_provenance"
    )
    gate_sampling["cofitok_sample_set_sha256"] = "Z" * 64
    kwargs["comparison"]["matched_training_rows"][1]["checkpoint_sha256"] = "e" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "final_generation_gate",
        "final_comparison_report",
    ]


def test_completion_audit_rejects_mislabeled_official_context() -> None:
    kwargs = _kwargs()
    kwargs["comparison"]["official_context_rows"][0]["method"] = "MAR"

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_comparison_report"]


def test_completion_audit_rejects_unbound_official_context_source() -> None:
    kwargs = _kwargs()
    kwargs["comparison"]["official_context_source"]["sha256"] = "8" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_comparison_report"]


def test_completion_audit_rejects_cross_tier_ranking_policy() -> None:
    kwargs = _kwargs()
    kwargs["comparison"]["comparison_policy"]["cross_tier_numeric_ranking_allowed"] = True

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_comparison_report"]


def test_completion_audit_rejects_misreported_matched_compute() -> None:
    kwargs = _kwargs()
    kwargs["comparison"]["matched_training_rows"][0]["training_images_seen"] -= 1

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_comparison_report"]


def test_completion_audit_rejects_training_that_ignores_selected_runtime() -> None:
    kwargs = _kwargs()
    kwargs["runtime_selection"]["selected"].update(
        micro_batch_size=32,
        gradient_accumulation_steps=2,
        estimated_speedup_over_16x4=1.2,
    )

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["full_runtime_selection"]


def test_completion_audit_rejects_sampling_that_ignores_selected_batch() -> None:
    kwargs = _kwargs()
    kwargs["dense_generation"]["sample_provenance"]["sampling"]["batch_size"] = 32

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "formal_sampling_runtime_selection",
        "final_comparison_report",
    ]


def test_completion_audit_rejects_stale_selected_sampling_preflight() -> None:
    kwargs = _kwargs()
    selected = kwargs["sampling_runtime_selection"]["candidates"][0]
    selected["methods"]["cofitok"]["git"]["revision"] = "0" * 40

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_sampling_runtime_selection"]


def test_completion_audit_rejects_mismatched_sampling_environment() -> None:
    kwargs = _kwargs()
    dense_provenance = kwargs["dense_generation"]["sample_provenance"]
    dense_provenance["runtime_environment"]["device"]["name"] = "another-gpu"
    dense_provenance["runtime_environment_sha256"] = runtime_environment_sha256(
        dense_provenance["runtime_environment"]
    )

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "formal_50k_generation",
        "formal_sampling_runtime_selection",
        "final_generation_gate",
    ]


def test_completion_audit_rejects_mismatched_evaluator_environment() -> None:
    kwargs = _kwargs()
    dense_generation = kwargs["dense_generation"]
    dense_generation["runtime_environment"]["device"]["name"] = "another-gpu"
    dense_generation["runtime_environment_sha256"] = runtime_environment_sha256(
        dense_generation["runtime_environment"]
    )

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "formal_50k_generation",
        "final_generation_gate",
        "final_comparison_report",
    ]


def test_completion_audit_rejects_mismatched_real_set_content() -> None:
    kwargs = _kwargs()
    dense_generation = kwargs["dense_generation"]
    dense_generation["real_set"]["sha256"] = "8" * 64
    dense_generation["parameters"]["real_cache_name"] = (
        "imagenet256_val__cofitok_" + "8" * 16
    )

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "formal_50k_generation",
        "final_generation_gate",
        "final_comparison_report",
    ]


def test_completion_audit_rejects_dirty_formal_sampling_code() -> None:
    kwargs = _kwargs()
    kwargs["cofitok_generation"]["sample_provenance"]["git"][
        "tracked_dirty"
    ] = True

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_dirty_formal_evaluator_code() -> None:
    kwargs = _kwargs()
    kwargs["dense_generation"]["git"]["tracked_dirty"] = True

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_sampling_outside_stable_inference_api() -> None:
    kwargs = _kwargs()
    del kwargs["cofitok_generation"]["sample_provenance"]["sampling"]["inference_api"]

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_visual_audit_from_stale_sample_set() -> None:
    kwargs = _kwargs()
    kwargs["visual_audit"]["sources"]["cofitok"]["sample_set_sha256"] = "Z" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deterministic_visual_quality_audit"]


def test_completion_audit_rejects_visual_audit_from_dirty_code() -> None:
    kwargs = _kwargs()
    kwargs["visual_audit"]["git"]["tracked_dirty"] = True

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deterministic_visual_quality_audit"]


def test_completion_audit_rejects_weakened_storage_safety_margin() -> None:
    kwargs = _kwargs()
    report = kwargs["storage_preflights"]["full_posteval"]
    report["plan"]["safety_margin_bytes"] = 1
    report["plan"]["required_free_bytes"] = (
        report["plan"]["checkpoint_reserve_bytes"]
        + report["plan"]["sample_reserve_bytes"]
        + report["plan"]["additional_bytes"]
        + report["plan"]["safety_margin_bytes"]
    )
    report["headroom_bytes"] = (
        report["filesystem"]["free_bytes"] - report["plan"]["required_free_bytes"]
    )

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == ["generation_storage_capacity"]


def test_completion_audit_rejects_full_monitor_health_issue() -> None:
    kwargs = _kwargs()
    kwargs["full_training_monitor"]["runs"]["cofitok"]["health_issues"] = [
        "metric total is non-finite"
    ]

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == ["full_training_operational_monitor"]


def test_completion_audit_rejects_matched_runtime_environment_drift() -> None:
    kwargs = _kwargs()
    dense = kwargs["dense_full_training"]
    dense["runtime_environment"]["torch"]["cudnn_version"] = 99999
    changed_sha = runtime_environment_sha256(dense["runtime_environment"])
    dense["runtime_environment_sha256"] = changed_sha
    dense["latest_checkpoint"]["runtime_environment_sha256"] = changed_sha

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == [
        "full_training_runtime_environment",
        "reproducible_full_checkpoint_files",
        "deployable_ema_inference_artifacts",
    ]


def test_completion_audit_rejects_checkpoint_from_another_revision() -> None:
    kwargs = _kwargs()
    kwargs["dense_full_training"]["latest_checkpoint"]["git_revision"] = "c" * 40

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == [
        "full_training_checkpoint_code_provenance",
        "reproducible_full_checkpoint_files",
    ]


def test_completion_audit_rejects_missing_full_checkpoint_bytes() -> None:
    kwargs = _kwargs()
    kwargs["full_checkpoint_files"]["cofitok"] = {
        "status": "invalid",
        "path": "/missing/checkpoint_step_00300000.pt",
        "error": "Checkpoint integrity manifest is missing",
    }

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == ["reproducible_full_checkpoint_files"]


def test_completion_audit_rejects_checkpoint_sidecar_revision_drift() -> None:
    kwargs = _kwargs()
    kwargs["full_checkpoint_files"]["dense_identity"]["git_revision"] = "c" * 40

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == ["reproducible_full_checkpoint_files"]


def test_completion_audit_rejects_export_from_stale_training_checkpoint() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_export"][
        "source_checkpoint_sha256"
    ] = "Z" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_export_from_stale_training_environment() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_export"][
        "source_runtime_environment_sha256"
    ] = "Z" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_corrupted_inference_artifact_bytes() -> None:
    kwargs = _kwargs()
    kwargs["inference_artifact_files"]["dense_identity"] = {
        "status": "invalid",
        "path": "/corrupted/dense_identity_ema_inference.pt",
        "error": "Inference artifact SHA256 mismatch",
    }

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_preserves_nonblocking_milestone_alerts() -> None:
    kwargs = copy.deepcopy(_kwargs())
    kwargs["milestones"][50_000] = _milestone(
        50_000, ["cofitok_fid_more_than_25pct_above_dense"]
    )

    report = build_completion_audit(**kwargs)

    assert report["complete"] is True
    assert report["warnings"] == [
        "milestone_50000:cofitok_fid_more_than_25pct_above_dense"
    ]


def test_completion_audit_direct_cli_reports_in_progress(tmp_path) -> None:
    output_root = tmp_path / "empty_outputs"
    output_root.mkdir()
    output = tmp_path / "completion_audit.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_large_scale_generation_completion.py"),
            "--project-root",
            str(ROOT),
            "--output-root",
            str(output_root),
            "--expected-full-revision",
            FULL_REVISION,
            "--output",
            str(output),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "in_progress"' in result.stdout
    assert output.is_file()
