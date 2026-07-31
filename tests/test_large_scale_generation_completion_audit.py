from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    dataset_provenance_identity_sha256,
)
from cofitok.generation import (
    INFERENCE_API,
    SAMPLING_MANIFEST_SCHEMA_VERSION,
    SAMPLING_PROTOCOL_SCHEMA,
    SAMPLING_REPORT_SCHEMA_VERSION,
)
from cofitok.generation_gate import (
    GENERATION_GATE_SCHEMA_VERSION,
    REQUIRED_GENERATION_GATES,
)
from cofitok.generation_authorization import build_generation_gate_binding
from cofitok.image_integrity import image_tree_sha256, sample_set_sha256
from cofitok.reporting import file_sha256
from cofitok.training.authorization import build_generation_training_authorization
from scripts.audit_large_scale_generation_completion import (
    MILESTONE_STEPS,
    _inference_export_evidence,
    _verify_deployment_bundle_source,
    _verify_formal_real_set_files,
    _verify_formal_sample_files,
    _verify_inference_smoke_outputs,
    build_completion_audit,
)
from scripts.build_generation_milestone_report import expected_source_report_suffixes
from cofitok.generation_gate_sources import GATE_SOURCE_SUFFIXES
from cofitok.generation_paths import SCALING_REPORT_ID
from scripts.select_generation_sampling_batch import select_sampling_batch
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


def _dataset_provenance(dataset: str) -> dict:
    spec = FORMAL_GENERATION_DATASETS[dataset]
    report = {
        "schema_version": 1,
        "status": "pass",
        "formal": True,
        "dataset": dataset,
        "dataset_root": f"/root/autodl-tmp/CoFiTok/datasets/{dataset}",
        "manifest": {
            "relative_path": "metadata/image_manifest.jsonl",
            "bytes": spec.manifest_bytes,
            "sha256": spec.manifest_sha256,
        },
        "splits": {"train": spec.train_images, "val": spec.val_images},
        "issues": [],
    }
    report["identity_sha256"] = dataset_provenance_identity_sha256(report)
    return report


def _full_training_authorization() -> dict:
    return build_generation_training_authorization(
        _gate("scaling"),
        gate_path=(
            "/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/"
            f"generation/{SCALING_REPORT_ID}/"
            "promotion_gate.json"
        ),
        gate_bytes=10_000,
        gate_sha256="9" * 64,
    )


def _full_release_authorization() -> dict:
    return build_generation_gate_binding(
        _gate("full"),
        expected_stage="full",
        gate_path=(
            "/root/autodl-tmp/CoFiTok/CoFiTok-internal/artifacts/reports/"
            "generation/imagenet256_full_matched_300k/final_generation_gate.json"
        ),
        gate_bytes=20_000,
        gate_sha256="8" * 64,
    )


def _training(
    *, steps: int, dataset: str, revision: str, parameters: int, checkpoint_sha: str
) -> dict:
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    cofitok_method = parameters > 100_000
    if dataset == "imagenet_256_10pct":
        config_name = (
            "imagenet256_10pct_fixed_basis_cofitok_k8_50k.json"
            if cofitok_method
            else "imagenet256_10pct_fixed_basis_dense_50k.json"
        )
    else:
        config_name = (
            "imagenet256_cofitok_k8_300k.json"
            if cofitok_method
            else "imagenet256_dense_300k.json"
        )
    dataset_provenance = _dataset_provenance(dataset)
    training_authorization = None
    if dataset == "imagenet_256":
        training_authorization = _full_training_authorization()
    latest = {
        "checkpoint": f"checkpoint_step_{steps:08d}.pt",
        "step": steps,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": 1_000,
        "integrity_manifest": f"checkpoint_step_{steps:08d}.pt.integrity.json",
        "runtime_environment_sha256": environment_sha,
        "git_revision": revision,
        "git_branch": "scale/generative-system",
        "git_dirty": False,
        "dataset_identity_sha256": dataset_provenance["identity_sha256"],
    }
    if training_authorization is not None:
        latest.update(
            {
                "authorization_stage": training_authorization["stage"],
                "authorization_decision": training_authorization["decision"],
                "authorization_gate_bytes": training_authorization["gate_bytes"],
                "authorization_gate_sha256": training_authorization["gate_sha256"],
                "authorization_gate_identity_sha256": training_authorization[
                    "gate_identity_sha256"
                ],
            }
        )
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
        "latest_checkpoint": latest,
        "config": config_to_dict(
            load_config(ROOT / "configs/generation" / config_name)
        ),
        "dataset_provenance": dataset_provenance,
        "training_authorization": training_authorization,
    }


def _gate(stage: str) -> dict:
    decision = (
        "promote_to_full_imagenet256"
        if stage == "scaling"
        else "large_scale_generation_ready"
    )
    environment_sha = runtime_environment_sha256(_runtime_environment())
    gates = [{"name": "all_evidence", "passed": True, "evidence": {}}]
    for name in sorted(REQUIRED_GENERATION_GATES[stage]):
        evidence = {}
        if name in {
            "matched_sampling_runtime_environment",
            "matched_evaluator_runtime_environment",
        }:
            evidence = {
                "cofitok": {"sha256": environment_sha},
                "dense_identity": {"sha256": environment_sha},
            }
        elif name == "matched_real_set_provenance":
            evidence = {
                method: {
                    "valid": True,
                    "digest_schema": "cofitok_image_tree_sha256_v1",
                    "sha256": REAL_SET_SHA,
                    "root": "/datasets/imagenet_256/validation",
                    "image_count": 50_000,
                    "real_cache_name": "imagenet256_val__cofitok_" + REAL_SET_SHA[:16],
                }
                for method in ("cofitok", "dense_identity")
            }
        elif name == "matched_sampling_provenance":
            evidence = {
                "cofitok_checkpoint_sha256": "a" * 64,
                "dense_checkpoint_sha256": "b" * 64,
                "cofitok_sample_set_sha256": "A" * 64,
                "dense_sample_set_sha256": "B" * 64,
            }
        elif name == "fid_within_tolerance":
            evidence = {
                "cofitok_fid": 19.0,
                "dense_fid": 19.0,
                "max_regression": 0.05,
            }
        elif name == "absolute_fid_quality":
            evidence = {
                "cofitok_fid": 19.0,
                "max_absolute_fid": 100.0 if stage == "scaling" else 20.0,
            }
        elif name == "endpoint_within_tolerance":
            evidence = {
                "cofitok_endpoint_mse": 0.1,
                "dense_endpoint_mse": 0.1,
                "max_regression": 0.05,
            }
        elif name == "ordered_prefix_path":
            evidence = {"rank": 1, "order_count": 24}
        elif name == "coarse_token_utilization":
            evidence = {
                "valid": True,
                "source_metric": "component_energy_ratio_per_sample_mean",
                "token_count": 8,
                "coarse_token_count": 6,
                "component_energy_ratios": [0.01] * 6 + [0.30, 0.64],
                "coarse_token_energy_ratio": 0.06,
                "min_coarse_token_energy_ratio": 0.05,
            }
        elif name == "restricted_synthesis_contract":
            evidence = {"zero_token_max_abs": 0.0}
        elif name == "shuffle_mismatch":
            evidence = {"shuffled_to_ordered_endpoint_ratio": 1.2}
        elif name == "full_precision_recall_quality":
            evidence = {
                "enforced": True,
                "cofitok_precision": 0.6,
                "dense_precision": 0.6,
                "cofitok_recall": 0.4,
                "dense_recall": 0.4,
                "min_precision": 0.30,
                "min_recall": 0.30,
                "max_precision_regression": 0.05,
                "max_recall_regression": 0.05,
            }
        gates.append({"name": name, "passed": True, "evidence": evidence})
    return {
        "schema_version": GENERATION_GATE_SCHEMA_VERSION,
        "stage": stage,
        "status": "pass",
        "decision": decision,
        "gates": gates,
        "thresholds": {
            "min_samples": 10_000 if stage == "scaling" else 50_000,
            "max_fid_regression": 0.05,
            "max_absolute_fid": 100.0 if stage == "scaling" else 20.0,
            "max_endpoint_regression": 0.05,
            "min_coarse_token_energy_ratio": 0.05,
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
            "cofitok_endpoint_mse": 0.1,
            "dense_endpoint_mse": 0.1,
            "ordered_rank": 1,
            "order_count": 24,
            "coarse_token_energy_ratio": 0.06,
        },
        "source_reports": {
            name: {
                "path": f"/root/autodl-tmp/CoFiTok/{suffix}",
                "bytes": 100 + index,
                "sha256": str(index + 1) * 64,
            }
            for index, (name, suffix) in enumerate(
                GATE_SOURCE_SUFFIXES[stage].items()
            )
        },
    }


def _gate_source_verification(gate: dict) -> dict:
    return {
        "status": "verified",
        "stage": gate["stage"],
        "source_reports": copy.deepcopy(gate["source_reports"]),
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
    environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(environment)
    dataset_provenance = _dataset_provenance("imagenet_256")
    dataset_sha = dataset_provenance["identity_sha256"]
    config_paths = {
        "cofitok": ROOT / "configs/generation/imagenet256_cofitok_k8_300k.json",
        "dense_identity": ROOT / "configs/generation/imagenet256_dense_300k.json",
    }
    config_sha256 = {
        method: file_sha256(path) for method, path in config_paths.items()
    }
    candidates = []
    for index, (micro_batch, accumulation) in enumerate(
        ((16, 4), (32, 2), (64, 1))
    ):
        methods = {}
        for method, path in config_paths.items():
            config = config_to_dict(load_config(path))
            config["data"]["batch_size"] = micro_batch
            config["optimization"][
                "gradient_accumulation_steps"
            ] = accumulation
            seconds = 2.0 + index * 0.25
            methods[method] = {
                "status": "completed",
                "git": {
                    "revision": FULL_REVISION,
                    "branch": "scale/generative-system",
                    "dirty": False,
                },
                "config": config,
                "runtime_environment": copy.deepcopy(environment),
                "runtime_environment_sha256": environment_sha,
                "dataset_provenance": copy.deepcopy(dataset_provenance),
                "effective_batch_size": 64,
                "benchmark_steps": 8,
                "warmup_steps": 2,
                "checkpoint_written": False,
                "mean_optimizer_step_seconds": seconds,
                "images_per_second": 64 / seconds,
                "peak_vram_bytes": 50_000,
                "device_total_memory_bytes": 100_000,
            }
        candidates.append(
            {
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": 64,
                "eligible": True,
                "ineligible_reasons": [],
                "selection_score_seconds": 2.0 + index * 0.25,
                "max_memory_fraction": 0.5,
                "runtime_environment_sha256": environment_sha,
                "dataset_identity_sha256": dataset_sha,
                "methods": methods,
            }
        )
    benchmark_root = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "runtime_preflight/full_imagenet256_300k"
    )
    selection_lock = {
        "schema_version": 1,
        "mode": "freeze_on_training_state",
        "training_run_dirs": [
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_full_cofitok_k8_300k",
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_full_dense_300k",
        ],
        "candidates": [
            {"micro_batch_size": 16, "gradient_accumulation_steps": 4},
            {"micro_batch_size": 32, "gradient_accumulation_steps": 2},
            {"micro_batch_size": 64, "gradient_accumulation_steps": 1},
        ],
        "expected_effective_batch_size": 64,
        "benchmark_steps": 8,
        "warmup_steps": 2,
        "max_memory_fraction": 0.9,
        "training_target_steps": 300_000,
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "config_sha256": config_sha256,
        "benchmark_root": benchmark_root,
    }
    return {
        "schema_version": 2,
        "status": "selected",
        "git_revision": FULL_REVISION,
        "config_sha256": config_sha256,
        "benchmark_root": benchmark_root,
        "selection_lock": selection_lock,
        "runtime_environment_sha256": environment_sha,
        "dataset_identity_sha256": dataset_sha,
        "selected": {
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 64,
            "estimated_speedup_over_16x4": 1.0,
        },
        "candidates": candidates,
    }


def _milestone(step: int, alerts: list[str] | None = None) -> dict:
    alerts = alerts or []
    source_suffixes = expected_source_report_suffixes(step)
    source_reports = {
        name: {
            "path": f"/root/outputs/{source_suffixes[name]}",
            "bytes": 100 + index,
            "sha256": str(index + 1) * 64,
        }
        for index, name in enumerate(
            (
                "cofitok_generation",
                "dense_generation",
                "cofitok_checkpoint_eval",
                "dense_checkpoint_eval",
            )
        )
    }

    def row(method: str, budget: int, sha: str, fid: float) -> dict:
        sampling = {
            "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
            "inference_api": INFERENCE_API,
            "sampler": "ddim",
            "num_samples": 2_048,
            "start_index": 0,
            "batch_size": 32,
            "sample_steps": 50,
            "num_train_timesteps": 1_000,
            "actual_timesteps": select_sampling_timesteps(1_000, 50),
            "prefix_budgets": [budget],
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "eta": 0.0,
            "clip_x0": True,
            "seed": 0,
            "precision": "bf16",
            "image_shape": [3, 256, 256],
            "class_schedule": "balanced_modulo",
            "random_stream": {
                "prefix_budgets_share_stream": True,
                "batch_size_invariant": True,
                "resume_index_invariant": True,
            },
            "sample_set_digest": {
                "algorithm": "sha256",
                "framing": "filename_utf8_nul_file_bytes_nul",
            },
        }
        return {
            "checkpoint": f"/checkpoints/{method}_{step}.pt",
            "checkpoint_sha256": sha,
            "checkpoint_integrity_manifest": f"/checkpoints/{method}_{step}.pt.integrity.json",
            "checkpoint_step": step,
            "weights": "ema",
            "sample_set_sha256": ("c" if budget > 1 else "d") * 64,
            "selected_prefix_budget": budget,
            "sample_count": 2_048,
            "fid": fid,
            "inception_score": 4.0,
            "endpoint_clean_mse": 0.1,
            "prefix_path_mse_auc": 0.2,
            "ordered_rank_by_path_auc": 1,
            "order_count": 5 if budget > 1 else 1,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
            "sampling": sampling,
        }

    cofitok_fid = (
        30.0
        if "cofitok_fid_more_than_25pct_above_dense" in alerts
        else 20.0
    )
    return {
        "schema_version": 2,
        "status": "completed",
        "role": "training_quality_trend_only",
        "claim_policy": {"formal_generation_claim_allowed": False},
        "milestone_step": step,
        "expected_samples": 2_048,
        "source_reports": source_reports,
        "methods": {
            "cofitok": row("cofitok", 8, "a" * 64, cofitok_fid),
            "dense_identity": row("dense", 1, "b" * 64, 20.0),
        },
        "matched_comparison": {
            "fid_relative_change": (cofitok_fid - 20.0) / 20.0,
            "endpoint_clean_mse_relative_change": 0.0,
        },
        "quality_alerts": alerts,
        "quality_alert": bool(alerts),
    }


def _milestone_source_verification(step: int) -> dict:
    return {
        "status": "verified",
        "source_reports": copy.deepcopy(_milestone(step)["source_reports"]),
    }


def _generation(seed: str) -> dict:
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    run = (
        "imagenet256_full_cofitok_k8_300k"
        if seed == "a"
        else "imagenet256_full_dense_300k"
    )
    budget = 8 if seed == "a" else 1
    sampling_root = f"/outputs/{run}/samples_50k_ddim250_cfg15"
    generated_dir = f"{sampling_root}/prefix_{budget}"
    return {
        "status": "completed",
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "runtime_environment": copy.deepcopy(runtime_environment),
        "runtime_environment_sha256": environment_sha,
        "paths": {
            "real_dir": "/datasets/imagenet_256/validation",
            "generated_dir": generated_dir,
        },
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
            "inception_score_mean": 30.0,
            "precision": 0.6,
            "recall": 0.4,
        },
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "sample_provenance": {
            "report": f"{sampling_root}/sampling_report.json",
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": environment_sha,
            "git": {
                "revision": FULL_REVISION,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
            "checkpoint_step": 300_000,
            "weights": "ema",
            "checkpoint": f"/outputs/{run}/checkpoint_step_00300000.pt",
            "checkpoint_sha256": seed * 64,
            "sample_set_sha256": seed.upper() * 64,
            "checkpoint_integrity_manifest": "/run/checkpoint_step_00300000.pt.integrity.json",
            "sampling_progress": {
                "report": f"{sampling_root}/sampling_progress.json",
                "status": "completed",
                "completed_samples": 50_000,
                "cumulative_elapsed_seconds": 10_000.0,
                "invocation": 1,
            },
            "sampling": {
                "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_samples": 50_000,
                "batch_size": 64,
                "sample_steps": 250,
                "num_train_timesteps": 1000,
                "actual_timesteps": select_sampling_timesteps(1000, 250),
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "precision": "bf16",
                "seed": 0,
                "start_index": 0,
                "class_schedule": "balanced_modulo",
                "prefix_budgets": [budget],
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
            },
        },
    }


def _formal_sample_files() -> dict:
    output = {}
    for method, seed in (("cofitok", "a"), ("dense_identity", "b")):
        generation = _generation(seed)
        provenance = generation["sample_provenance"]
        output[method] = {
            "status": "verified",
            "generated_dir": generation["paths"]["generated_dir"],
            "sample_count": 50_000,
            "sample_set_sha256": provenance["sample_set_sha256"],
            "sampling_report": provenance["report"],
            "sampling_report_sha256": "1" * 64,
            "sampling_manifest": str(
                Path(provenance["report"]).with_name("sampling_manifest.json")
            ),
            "sampling_manifest_sha256": "2" * 64,
            "sampling_progress": provenance["sampling_progress"]["report"],
            "sampling_progress_sha256": "3" * 64,
        }
    return output


def _formal_real_set_files() -> dict:
    return {
        "status": "verified",
        "real_dir": "/datasets/imagenet_256/validation",
        "image_count": 50_000,
        "digest_schema": "cofitok_image_tree_sha256_v1",
        "real_set_sha256": REAL_SET_SHA,
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
        "schema_version": 5,
        "status": "ready",
        "final_gate": {
            "status": "pass",
            "decision": "large_scale_generation_ready",
        },
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
        "source_reports": _comparison_source_reports(),
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
                "sampling_invocations": 1,
                "fid": 19.0,
                "inception_score": 30.0,
                "precision": 0.6,
                "recall": 0.4,
                "evaluator": {"package": "torch_fidelity", "version": "0.4.0"},
                "weights": "ema",
                "sampling_protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "sampling_inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_train_timesteps": 1000,
                "sample_steps": 250,
                "actual_timesteps": select_sampling_timesteps(1000, 250),
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "sampling_precision": "bf16",
                "sampling_seed": 0,
                "sampling_start_index": 0,
                "class_schedule": "balanced_modulo",
                "prefix_budgets": [8],
                "sampling_random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
                "checkpoint_sha256": "a" * 64,
                "sample_set_sha256": "A" * 64,
                "real_set_digest_schema": "cofitok_image_tree_sha256_v1",
                "real_set_sha256": REAL_SET_SHA,
                "real_image_count": 50_000,
                "evaluator_runtime_environment_sha256": runtime_environment_sha256(
                    _runtime_environment()
                ),
                "protocol_note": "Same data, backbone family, optimizer, steps, and evaluator.",
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
                "sampling_invocations": 1,
                "fid": 19.0,
                "inception_score": 30.0,
                "precision": 0.6,
                "recall": 0.4,
                "evaluator": {"package": "torch_fidelity", "version": "0.4.0"},
                "weights": "ema",
                "sampling_protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
                "sampling_inference_api": INFERENCE_API,
                "sampler": "ddim",
                "num_train_timesteps": 1000,
                "sample_steps": 250,
                "actual_timesteps": select_sampling_timesteps(1000, 250),
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "clip_x0": True,
                "sampling_precision": "bf16",
                "sampling_seed": 0,
                "sampling_start_index": 0,
                "class_schedule": "balanced_modulo",
                "prefix_budgets": [1],
                "sampling_random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
                "checkpoint_sha256": "b" * 64,
                "sample_set_sha256": "B" * 64,
                "real_set_digest_schema": "cofitok_image_tree_sha256_v1",
                "real_set_sha256": REAL_SET_SHA,
                "real_image_count": 50_000,
                "evaluator_runtime_environment_sha256": runtime_environment_sha256(
                    _runtime_environment()
                ),
                "protocol_note": "Same data, backbone family, optimizer, steps, and evaluator.",
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
                "evaluator": {
                    "package": "ADM TensorFlow evaluation graph",
                    "version": "pinned baseline protocol",
                },
                "protocol_note": row["protocol"],
                "source_metrics": row["metrics_txt"],
                "source_status": row["status"],
                "paper_table_role": row["paper_table_role"],
            }
            for row in official["rows"]
        ],
        "matched_summary": {
            "cofitok_minus_dense_fid": 0.0,
            "cofitok_relative_fid": 0.0,
        },
    }


def _comparison_source_reports() -> dict:
    root = "/root/autodl-tmp/CoFiTok"
    return {
        "cofitok_training": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_cofitok_k8_300k/training_report.json",
            "bytes": 100,
            "sha256": "1" * 64,
        },
        "dense_training": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_dense_300k/training_report.json",
            "bytes": 100,
            "sha256": "2" * 64,
        },
        "cofitok_generation": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_cofitok_k8_300k/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json",
            "bytes": 100,
            "sha256": "3" * 64,
        },
        "dense_generation": {
            "path": f"{root}/checkpoints/generation/imagenet256_full_dense_300k/samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json",
            "bytes": 100,
            "sha256": "4" * 64,
        },
        "final_gate": {
            "path": f"{root}/CoFiTok-internal/artifacts/reports/generation/imagenet256_full_matched_300k/final_generation_gate.json",
            "bytes": 100,
            "sha256": "5" * 64,
        },
    }


def _comparison_source_verification() -> dict:
    return {
        "status": "verified",
        "source_reports": _comparison_source_reports(),
    }


def _sampling_runtime_selection() -> dict:
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    checkpoints = {
        "cofitok": {
            "path": "/checkpoints/cofitok.pt",
            "sha256": "a" * 64,
            "step": 300_000,
            "integrity_manifest": "/checkpoints/cofitok.pt.integrity.json",
        },
        "dense_identity": {
            "path": "/checkpoints/dense.pt",
            "sha256": "b" * 64,
            "step": 300_000,
            "integrity_manifest": "/checkpoints/dense.pt.integrity.json",
        },
    }
    candidates = []
    for batch_size, throughput in ((16, 30.0), (32, 50.0), (64, 70.0), (128, 65.0)):
        methods = {}
        for method, prefix_budget in (("cofitok", 8), ("dense_identity", 1)):
            methods[method] = {
                "status": "passed",
                "checkpoint": checkpoints[method]["path"],
                "checkpoint_sha256": checkpoints[method]["sha256"],
                "checkpoint_integrity_manifest": checkpoints[method][
                    "integrity_manifest"
                ],
                "checkpoint_step": 300_000,
                "weights": "ema",
                "runtime_environment": copy.deepcopy(runtime_environment),
                "runtime_environment_sha256": environment_sha,
                "git": {
                    "revision": FULL_REVISION,
                    "branch": "scale/generative-system",
                    "tracked_dirty": False,
                },
                "request": {
                    "batch_size": batch_size,
                    "effective_model_batch_size": batch_size * 2,
                    "prefix_budget": prefix_budget,
                    "token_count": prefix_budget,
                    "precision": "bf16",
                    "guidance_scale": 1.5,
                    "guidance_rescale": 0.0,
                    "cfg_batch_mode": "batched",
                    "warmup_forwards": 2,
                    "measured_forwards": 5,
                },
                "result": {
                    "output_images_per_second": throughput,
                    "cuda_memory_after_forward": {
                        "peak_allocated_bytes": 50_000,
                    },
                    "device_total_memory_bytes": 100_000,
                },
            }
        candidates.append({"batch_size": batch_size, "methods": methods})
    selection = select_sampling_batch(
        candidates,
        baseline_batch_size=32,
        max_memory_fraction=0.9,
    )
    benchmark_root = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "runtime_preflight/imagenet256_full_50k_sampling"
    )
    selection_lock = {
        "schema_version": 1,
        "mode": "freeze_on_sampling_state",
        "sampling_output_dirs": [
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "imagenet256_full_cofitok_k8_300k/samples_50k_ddim250_cfg15",
            "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
            "imagenet256_full_dense_300k/samples_50k_ddim250_cfg15",
        ],
        "candidates": [16, 32, 64, 128],
        "baseline_batch_size": 32,
        "max_memory_fraction": 0.9,
        "protocol": {
            "cofitok_prefix_budget": 8,
            "dense_prefix_budget": 1,
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "weights": "ema",
            "precision": "bf16",
            "warmup_forwards": 2,
            "measured_forwards": 5,
        },
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "checkpoints": checkpoints,
        "benchmark_root": benchmark_root,
    }
    selection.update(
        git_revision=FULL_REVISION,
        checkpoints=checkpoints,
        benchmark_root=benchmark_root,
        selection_lock=selection_lock,
    )
    return selection


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
    execution_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(execution_environment)
    training_authorization = _full_training_authorization()
    release_authorization = _full_release_authorization()
    execution_git = {
        "revision": FULL_REVISION,
        "branch": "scale/generative-system",
        "tracked_dirty": False,
    }
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
            "source_training_authorization": copy.deepcopy(
                training_authorization
            ),
            "release_authorization": copy.deepcopy(release_authorization),
            "artifact_sha256": artifact_sha,
            "artifact_bytes": 400,
            "artifact": artifact,
            "artifact_integrity_manifest": f"{artifact}.integrity.json",
        }
        output[f"{method}_preflight"] = {
            "status": "passed",
            "git": dict(execution_git),
            "runtime_environment": copy.deepcopy(execution_environment),
            "runtime_environment_sha256": environment_sha,
            "checkpoint_sha256": artifact_sha,
            "artifact_type": "cofitok_generation_inference",
            "weights": "ema_export",
            "source_checkpoint_sha256": source_sha,
            "source_runtime_environment_sha256": environment_sha,
            "source_git": dict(source_git),
            "training_authorization": copy.deepcopy(training_authorization),
            "release_authorization": copy.deepcopy(release_authorization),
            "release_authorization_required": True,
        }
        output[f"{method}_smoke"] = {
            "status": "completed",
            "git": dict(execution_git),
            "runtime_environment": copy.deepcopy(execution_environment),
            "runtime_environment_sha256": environment_sha,
            "output_count": count,
            "checkpoint": {
                "checkpoint_sha256": artifact_sha,
                "checkpoint_step": 300_000,
                "weights": "ema_export",
                "artifact_type": "cofitok_generation_inference",
                "source_checkpoint_sha256": source_sha,
                "source_runtime_environment_sha256": environment_sha,
                "source_git": dict(source_git),
                "training_authorization": copy.deepcopy(
                    training_authorization
                ),
                "release_authorization": copy.deepcopy(
                    release_authorization
                ),
                "release_authorization_required": True,
            },
            "request": {
                "seeds": [0, 1],
                "class_ids": [0, 0],
                "prefix_budgets": [1, 8] if method == "cofitok" else [1],
                "batch_size": 2,
                "sample_steps": 10,
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "cfg_batch_mode": "batched",
                "eta": 0.0,
                "precision": "bf16",
            },
            "outputs": [
                {
                    "path": (
                        "/root/autodl-tmp/CoFiTok/checkpoints/generation/exports/"
                        f"imagenet256_full_300k/smoke/{method}/output_{index}.png"
                    ),
                    "filename": f"output_{index}.png",
                    "sha256": str(index) * 64,
                    "seed": (index - 1) % 2,
                    "class_id": 0,
                    "prefix_budget": (
                        [1, 1, 8, 8][index - 1]
                        if method == "cofitok"
                        else 1
                    ),
                }
                for index in range(1, count + 1)
            ],
        }
    return output


def _full_checkpoint_files() -> dict:
    output = {}
    environment_sha = runtime_environment_sha256(_runtime_environment())
    authorization = _full_training_authorization()
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
            "authorization_stage": authorization["stage"],
            "authorization_decision": authorization["decision"],
            "authorization_gate_bytes": authorization["gate_bytes"],
            "authorization_gate_sha256": authorization["gate_sha256"],
            "authorization_gate_identity_sha256": authorization[
                "gate_identity_sha256"
            ],
        }
    return output


def _inference_artifact_files() -> dict:
    output = {}
    environment_sha = runtime_environment_sha256(_runtime_environment())
    training_authorization = _full_training_authorization()
    release_authorization = _full_release_authorization()
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
            "source_training_authorization": copy.deepcopy(
                training_authorization
            ),
            "release_authorization": copy.deepcopy(release_authorization),
            "step": 300_000,
        }
    return output


def _inference_smoke_files() -> dict:
    exports = _inference_exports()
    output = {}
    for method in ("cofitok", "dense_identity"):
        rows = exports[f"{method}_smoke"]["outputs"]
        output[method] = {
            "status": "verified",
            "root": str(Path(rows[0]["path"]).parent),
            "output_count": len(rows),
            "outputs": [
                {
                    "path": row["path"],
                    "bytes": 1024,
                    "sha256": row["sha256"],
                    "mode": "RGB",
                    "width": 256,
                    "height": 256,
                }
                for row in rows
            ],
        }
    return output


def _deployment_transition() -> tuple[dict, dict]:
    bundle_path = f"/outputs/deployment/cofitok-generation-upgrade-{FULL_REVISION}.bundle"
    conflict_path = "/outputs/generation_upgrade_conflict_scan.json"
    runbook_path = "/outputs/generation_upgrade_runbook_syntax.json"
    pytest_path = "/outputs/generation_upgrade_pytest.xml"
    conflict_scan = {
        "schema_version": 1,
        "status": "pass",
        "current_commit": TEN_REVISION,
        "target_commit": FULL_REVISION,
        "target_added_path_count": 145,
        "conflict_count": 0,
        "conflicts": [],
    }
    runbook_syntax = {
        "schema_version": 1,
        "role": "generation_runbook_syntax_check",
        "status": "pass",
        "enumeration": "git_ls_files",
        "discovered_count": 2,
        "checked_count": 2,
        "failed_count": 0,
        "runbooks": [
            "artifacts/runbooks/a.sh",
            "artifacts/runbooks/b.sh",
        ],
        "failures": [],
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
    }
    pytest_summary = {
        "status": "pass",
        "tests": 530,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    receipt = {
        "schema_version": 2,
        "status": "pass",
        "expected_training_revision": TEN_REVISION,
        "target_revision": FULL_REVISION,
        "git": {
            "revision": FULL_REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "bundle": {
            "path": bundle_path,
            "bytes": 1234,
            "sha256": "e" * 64,
            "heads": [FULL_REVISION],
            "prerequisites": [TEN_REVISION],
        },
        "training_pair_validation": {
            "status": "pass",
            "expected_revision": TEN_REVISION,
            "sha256": "f" * 64,
        },
        "verification": {
            "pytest": {
                "path": pytest_path,
                "bytes": 300,
                "sha256": "3" * 64,
                **pytest_summary,
            },
            "runbook_syntax": {
                "path": runbook_path,
                "bytes": 200,
                "sha256": "2" * 64,
                "status": "pass",
                "checked_count": 2,
            },
            "untracked_target_conflicts": {
                "path": conflict_path,
                "bytes": 100,
                "sha256": "1" * 64,
                "status": "pass",
                "target_added_path_count": 145,
                "conflict_count": 0,
            },
        },
    }
    sources = {
        "bundle": {
            "status": "verified",
            "path": bundle_path,
            "bytes": 1234,
            "sha256": "e" * 64,
            "heads": [FULL_REVISION],
            "prerequisites": [TEN_REVISION],
        },
        "conflict_scan": {
            "status": "verified",
            "path": conflict_path,
            "bytes": 100,
            "sha256": "1" * 64,
            "content": conflict_scan,
        },
        "runbook_syntax": {
            "status": "verified",
            "path": runbook_path,
            "bytes": 200,
            "sha256": "2" * 64,
            "content": runbook_syntax,
        },
        "pytest": {
            "status": "verified",
            "path": pytest_path,
            "bytes": 300,
            "sha256": "3" * 64,
            "summary": pytest_summary,
        },
    }
    return receipt, sources


def _kwargs() -> dict:
    deployment_receipt, deployment_verification_files = _deployment_transition()
    scaling_gate = _gate("scaling")
    final_gate = _gate("full")
    return {
        "expected_deployment_source_revision": TEN_REVISION,
        "expected_10pct_revision": FULL_REVISION,
        "expected_full_revision": FULL_REVISION,
        "cofitok_10pct_training": _training(
            steps=50_000,
            dataset="imagenet_256_10pct",
            revision=FULL_REVISION,
            parameters=100_500,
            checkpoint_sha="c" * 64,
        ),
        "dense_10pct_training": _training(
            steps=50_000,
            dataset="imagenet_256_10pct",
            revision=FULL_REVISION,
            parameters=100_000,
            checkpoint_sha="d" * 64,
        ),
        "deployment_receipt": deployment_receipt,
        "deployment_verification_files": deployment_verification_files,
        "storage_preflights": _storage_preflights(),
        "expected_storage_path": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
        "full_training_monitor": _full_training_monitor(),
        "full_checkpoint_files": _full_checkpoint_files(),
        "scaling_gate": scaling_gate,
        "scaling_gate_source_verification": _gate_source_verification(
            scaling_gate
        ),
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
        "inference_smoke_files": _inference_smoke_files(),
        "milestones": {step: _milestone(step) for step in MILESTONE_STEPS},
        "milestone_source_verifications": {
            step: _milestone_source_verification(step) for step in MILESTONE_STEPS
        },
        "cofitok_generation": _generation("a"),
        "dense_generation": _generation("b"),
        "formal_sample_files": _formal_sample_files(),
        "formal_real_set_files": _formal_real_set_files(),
        "final_gate": final_gate,
        "final_gate_source_verification": _gate_source_verification(final_gate),
        "comparison": _comparison(),
        "comparison_source_verification": _comparison_source_verification(),
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


def test_completion_audit_rehashes_deployment_verification_sources() -> None:
    kwargs = _kwargs()
    kwargs["deployment_verification_files"]["pytest"]["sha256"] = "0" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["controlled_revision_transition"]


def test_completion_audit_rehashes_archived_deployment_bundle() -> None:
    kwargs = _kwargs()
    kwargs["deployment_verification_files"]["bundle"]["sha256"] = "0" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["controlled_revision_transition"]


def test_completion_audit_revalidates_conflict_source_content() -> None:
    kwargs = _kwargs()
    conflict = kwargs["deployment_verification_files"]["conflict_scan"]["content"]
    conflict["status"] = "conflict"
    conflict["conflict_count"] = 1
    conflict["conflicts"] = ["src/cofitok/new.py"]

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["controlled_revision_transition"]


def test_completion_audit_verifies_archived_bundle_bytes_and_head(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "-b", "scale/generative-system"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "cofitok@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CoFiTok Test"],
        cwd=repository,
        check=True,
    )
    (repository / "tracked.txt").write_text("pinned\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "pinned"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    pinned_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repository / "tracked.txt").write_text("target\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "target"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bundle = tmp_path / "deployment.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle), "HEAD", f"^{pinned_revision}"],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    verified = _verify_deployment_bundle_source(bundle)

    assert verified["status"] == "verified"
    assert verified["heads"] == [revision]
    assert verified["prerequisites"] == [pinned_revision]
    bundle.write_bytes(bundle.read_bytes()[:-1])
    tampered = _verify_deployment_bundle_source(bundle)
    assert tampered["status"] == "verified"
    assert tampered["sha256"] != verified["sha256"]
    assert tampered["bytes"] == verified["bytes"] - 1


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
    assert report["failed_checks"] == [
        "deployable_ema_inference_artifacts",
        "final_generation_gate",
    ]


def test_completion_audit_rejects_weakened_final_quality_threshold() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["thresholds"]["max_absolute_fid"] = 100.0

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "deployable_ema_inference_artifacts",
        "final_generation_gate",
    ]


def test_completion_audit_rejects_weakened_scaling_quality_threshold() -> None:
    kwargs = _kwargs()
    kwargs["scaling_gate"]["thresholds"]["max_absolute_fid"] = 100.1

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "scaling_promotion_gate",
        "full_matched_training",
    ]


def test_completion_audit_rejects_scaling_gate_source_verification_drift() -> None:
    kwargs = _kwargs()
    kwargs["scaling_gate_source_verification"]["source_reports"][
        "cofitok_generation"
    ]["sha256"] = "0" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["scaling_promotion_gate"]


def test_completion_audit_rejects_final_gate_source_verification_drift() -> None:
    kwargs = _kwargs()
    kwargs["final_gate_source_verification"]["source_reports"][
        "dense_checkpoint_eval"
    ]["bytes"] += 1

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_generation_gate"]


def test_completion_audit_rejects_incomplete_scaling_gate_contract() -> None:
    kwargs = _kwargs()
    kwargs["scaling_gate"]["gates"] = [
        row
        for row in kwargs["scaling_gate"]["gates"]
        if row["name"] != "absolute_fid_quality"
    ]

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "scaling_promotion_gate",
        "full_matched_training",
    ]


def test_completion_audit_rejects_unbound_final_quality_metrics() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["summary"]["cofitok_recall"] = 0.31

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "deployable_ema_inference_artifacts",
        "final_generation_gate",
    ]


def test_completion_audit_requires_named_code_provenance_gates() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["gates"] = [
        gate
        for gate in kwargs["final_gate"]["gates"]
        if gate["name"] != "matched_evaluator_code_provenance"
    ]

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "deployable_ema_inference_artifacts",
        "final_generation_gate",
    ]


def test_completion_audit_rejects_incomplete_formal_sampling() -> None:
    kwargs = _kwargs()
    kwargs["cofitok_generation"]["sample_provenance"]["sampling_progress"][
        "completed_samples"
    ] = 49_999

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_missing_or_tampered_formal_sample_files() -> None:
    kwargs = _kwargs()
    kwargs["formal_sample_files"]["cofitok"] = {
        "status": "invalid",
        "generated_dir": kwargs["cofitok_generation"]["paths"]["generated_dir"],
        "error": "formal sample-set SHA256 differs from metrics provenance",
    }

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_tampered_formal_real_set_files() -> None:
    kwargs = _kwargs()
    kwargs["formal_real_set_files"] = {
        "status": "invalid",
        "real_dir": kwargs["cofitok_generation"]["paths"]["real_dir"],
        "error": "formal real-set report differs from physical image tree",
    }

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_matched_weakened_sampling_protocol() -> None:
    kwargs = _kwargs()
    for key in ("cofitok_generation", "dense_generation"):
        kwargs[key]["sample_provenance"]["sampling"]["clip_x0"] = False
    for row in kwargs["comparison"]["matched_training_rows"]:
        row["clip_x0"] = False

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
        "deployable_ema_inference_artifacts",
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


def test_completion_audit_rejects_changed_matched_comparison_source() -> None:
    kwargs = _kwargs()
    kwargs["comparison_source_verification"] = {
        "status": "invalid",
        "error": "comparison source report changed after binding: cofitok_generation",
    }

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_comparison_report"]


def test_completion_audit_rejects_misreported_matched_quality_metric() -> None:
    kwargs = _kwargs()
    kwargs["comparison"]["matched_training_rows"][0]["precision"] = 0.7

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


def test_completion_audit_rejects_identically_weakened_full_recipe() -> None:
    kwargs = _kwargs()
    for key in ("cofitok_full_training", "dense_full_training"):
        kwargs[key]["config"]["optimization"]["ema_decay"] = 0.9

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "full_matched_training",
        "full_runtime_selection",
    ]


def test_completion_audit_rejects_runtime_selection_lock_drift() -> None:
    kwargs = _kwargs()
    kwargs["runtime_selection"]["selection_lock"]["benchmark_steps"] = 9

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["full_runtime_selection"]


def test_completion_audit_rejects_runtime_benchmark_config_drift() -> None:
    kwargs = _kwargs()
    benchmark = kwargs["runtime_selection"]["candidates"][1]["methods"][
        "cofitok"
    ]
    benchmark["config"]["runtime"]["steps"] = 299_999

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["full_runtime_selection"]


def test_completion_audit_rejects_missing_full_dataset_provenance() -> None:
    kwargs = _kwargs()
    for key in ("cofitok_full_training", "dense_full_training"):
        kwargs[key].pop("dataset_provenance")
        kwargs[key]["latest_checkpoint"].pop("dataset_identity_sha256")

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "full_matched_training",
        "full_runtime_selection",
    ]


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


def test_completion_audit_rejects_sampling_selection_lock_drift() -> None:
    kwargs = _kwargs()
    kwargs["sampling_runtime_selection"]["selection_lock"]["protocol"][
        "measured_forwards"
    ] = 6

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_sampling_runtime_selection"]


def test_completion_audit_rejects_nonselected_preflight_environment_drift() -> None:
    kwargs = _kwargs()
    candidate = kwargs["sampling_runtime_selection"]["candidates"][0]
    for method in ("cofitok", "dense_identity"):
        preflight = candidate["methods"][method]
        preflight["runtime_environment"]["device"]["name"] = "another-gpu"
        preflight["runtime_environment_sha256"] = runtime_environment_sha256(
            preflight["runtime_environment"]
        )

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_sampling_runtime_selection"]


def test_completion_audit_rejects_preflight_integrity_path_drift() -> None:
    kwargs = _kwargs()
    preflight = kwargs["sampling_runtime_selection"]["candidates"][0]["methods"][
        "cofitok"
    ]
    preflight["checkpoint_integrity_manifest"] = "/checkpoints/other.integrity.json"

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
    assert report["failed_checks"] == [
        "formal_50k_generation",
        "final_comparison_report",
    ]


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
        "full_runtime_selection",
        "deployable_ema_inference_artifacts",
    ]


def test_completion_audit_rejects_training_benchmark_environment_drift() -> None:
    kwargs = _kwargs()
    benchmark = kwargs["runtime_selection"]["candidates"][0]["methods"][
        "dense_identity"
    ]
    benchmark["runtime_environment"]["device"]["name"] = "another GPU"
    benchmark["runtime_environment_sha256"] = runtime_environment_sha256(
        benchmark["runtime_environment"]
    )

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == ["full_runtime_selection"]


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


def test_completion_audit_rejects_full_training_authorization_drift() -> None:
    kwargs = _kwargs()
    kwargs["dense_full_training"]["training_authorization"][
        "gate_identity_sha256"
    ] = "c" * 64

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == [
        "full_matched_training",
        "deployable_ema_inference_artifacts",
    ]


def test_completion_audit_rejects_checkpoint_authorization_sidecar_drift() -> None:
    kwargs = _kwargs()
    kwargs["full_checkpoint_files"]["dense_identity"][
        "authorization_gate_identity_sha256"
    ] = "c" * 64

    completion = build_completion_audit(**kwargs)

    assert completion["status"] == "failed"
    assert completion["failed_checks"] == ["reproducible_full_checkpoint_files"]


def _write_formal_sample_fixture(tmp_path: Path) -> tuple[dict, Path]:
    sampling_root = tmp_path / "samples"
    generated = sampling_root / "prefix_2"
    generated.mkdir(parents=True)
    paths = []
    for index in range(2):
        path = generated / f"{index:06d}.png"
        Image.new("RGB", (8, 8), (index * 30, 10, 20)).save(path)
        paths.append(path)
    sample_sha = sample_set_sha256(paths)
    runtime_environment = _runtime_environment()
    environment_sha = runtime_environment_sha256(runtime_environment)
    git = {
        "revision": FULL_REVISION,
        "branch": "scale/generative-system",
        "tracked_dirty": False,
    }
    sampling = {"prefix_budgets": [2], "num_samples": 2}
    output_dirs = {"2": generated.resolve().as_posix()}
    checkpoint = (tmp_path / "checkpoint.pt").resolve().as_posix()
    fields = {
        "git": git,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": environment_sha,
        "checkpoint": checkpoint,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_integrity_manifest": f"{checkpoint}.integrity.json",
        "checkpoint_step": 300_000,
        "weights": "ema",
        "sampling": sampling,
        "output_dirs": output_dirs,
    }
    manifest_path = sampling_root / "sampling_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION, **fields},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha = file_sha256(manifest_path)
    sample_sets = {"2": {"count": 2, "sha256": sample_sha}}
    progress_path = sampling_root / "sampling_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "sampling_manifest_sha256": manifest_sha,
                "completed_samples": 2,
                "sample_sets": sample_sets,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sampling_report_path = sampling_root / "sampling_report.json"
    sampling_report_path.write_text(
        json.dumps(
            {
                "schema_version": SAMPLING_REPORT_SCHEMA_VERSION,
                "status": "completed",
                **fields,
                "sampling_manifest_sha256": manifest_sha,
                "sampling_progress": progress_path.resolve().as_posix(),
                "sample_sets": sample_sets,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = {
        "paths": {"generated_dir": generated.resolve().as_posix()},
        "sample_provenance": {
            "report": sampling_report_path.resolve().as_posix(),
            **{key: value for key, value in fields.items() if key != "output_dirs"},
            "sample_set_sha256": sample_sha,
            "sampling_progress": {"report": progress_path.resolve().as_posix()},
        },
    }
    return report, generated


def test_formal_sample_file_verifier_rehashes_sample_tree_and_sources(tmp_path) -> None:
    report, generated = _write_formal_sample_fixture(tmp_path)

    verified = _verify_formal_sample_files(
        report,
        expected_generated_dir=generated,
        expected_count=2,
    )

    assert verified is not None
    assert verified["status"] == "verified"
    assert verified["sample_count"] == 2
    Image.new("RGB", (8, 8), (255, 0, 0)).save(generated / "000000.png")
    tampered = _verify_formal_sample_files(
        report,
        expected_generated_dir=generated,
        expected_count=2,
    )
    assert tampered is not None
    assert tampered["status"] == "invalid"
    assert "sample-set SHA256 differs" in tampered["error"]


def test_formal_sample_file_verifier_rejects_missing_numbered_png(tmp_path) -> None:
    report, generated = _write_formal_sample_fixture(tmp_path)
    (generated / "000001.png").unlink()

    verified = _verify_formal_sample_files(
        report,
        expected_generated_dir=generated,
        expected_count=2,
    )

    assert verified is not None
    assert verified["status"] == "invalid"
    assert "numbered PNG set differs" in verified["error"]


def _write_formal_real_set_fixture(tmp_path: Path) -> tuple[dict[str, dict], Path]:
    real_dir = tmp_path / "imagenet_val"
    for class_index in range(2):
        class_dir = real_dir / f"class_{class_index:04d}"
        class_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), (class_index * 50, 10, 20)).save(
            class_dir / f"image_{class_index:04d}.jpg"
        )
    images = sorted(real_dir.rglob("*.jpg"))
    real_sha = image_tree_sha256(images, root=real_dir)
    reports = {}
    for method in ("cofitok", "dense_identity"):
        reports[method] = {
            "paths": {"real_dir": real_dir.resolve().as_posix()},
            "counts": {"real_image_count": 2},
            "real_set": {
                "digest_schema": "cofitok_image_tree_sha256_v1",
                "sha256": real_sha,
                "root": real_dir.resolve().as_posix(),
                "image_count": 2,
            },
        }
    return reports, real_dir


def test_formal_real_set_verifier_rehashes_physical_tree(tmp_path) -> None:
    reports, real_dir = _write_formal_real_set_fixture(tmp_path)

    verified = _verify_formal_real_set_files(
        reports,
        expected_real_dir=real_dir,
        expected_count=2,
    )

    assert verified is not None
    assert verified["status"] == "verified"
    assert verified["real_set_sha256"] == reports["cofitok"]["real_set"]["sha256"]
    Image.new("RGB", (8, 8), (255, 0, 0)).save(
        real_dir / "class_0000/image_0000.jpg"
    )
    tampered = _verify_formal_real_set_files(
        reports,
        expected_real_dir=real_dir,
        expected_count=2,
    )
    assert tampered is not None
    assert tampered["status"] == "invalid"
    assert "physical image tree" in tampered["error"]


def test_formal_real_set_verifier_rejects_unsupported_files(tmp_path) -> None:
    reports, real_dir = _write_formal_real_set_fixture(tmp_path)
    (real_dir / "notes.txt").write_text("unexpected", encoding="utf-8")

    verified = _verify_formal_real_set_files(
        reports,
        expected_real_dir=real_dir,
        expected_count=2,
    )

    assert verified is not None
    assert verified["status"] == "invalid"
    assert "unsupported file" in verified["error"]


def test_inference_smoke_file_verifier_rehashes_and_decodes_pngs(tmp_path) -> None:
    root = tmp_path / "smoke"
    root.mkdir()
    outputs = []
    for index in range(2):
        path = (root / f"output_{index}.png").resolve()
        Image.new("RGB", (256, 256), (index * 20, 10, 30)).save(path)
        outputs.append(
            {
                "path": path.as_posix(),
                "filename": path.name,
                "sha256": file_sha256(path),
            }
        )
    report = {"output_count": 2, "outputs": outputs}

    verified = _verify_inference_smoke_outputs(report, expected_root=root)

    assert verified is not None
    assert verified["status"] == "verified"
    assert verified["output_count"] == 2
    assert all(row["mode"] == "RGB" for row in verified["outputs"])
    Image.new("RGB", (256, 256), (255, 0, 0)).save(outputs[0]["path"])
    tampered = _verify_inference_smoke_outputs(report, expected_root=root)
    assert tampered is not None
    assert tampered["status"] == "invalid"
    assert "SHA256 differs" in tampered["error"]


def test_inference_smoke_file_verifier_rejects_path_escape(tmp_path) -> None:
    root = tmp_path / "smoke"
    root.mkdir()
    outside = (tmp_path / "outside.png").resolve()
    Image.new("RGB", (256, 256)).save(outside)
    report = {
        "output_count": 1,
        "outputs": [
            {
                "path": outside.as_posix(),
                "filename": outside.name,
                "sha256": file_sha256(outside),
            }
        ],
    }

    verified = _verify_inference_smoke_outputs(report, expected_root=root)

    assert verified is not None
    assert verified["status"] == "invalid"
    assert "escapes its output root" in verified["error"]


def test_completion_audit_rejects_export_from_stale_training_checkpoint() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_export"][
        "source_checkpoint_sha256"
    ] = "Z" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_inference_release_audit_binds_execution_revision_and_environment() -> None:
    kwargs = _kwargs()
    generation = {
        "cofitok": kwargs["cofitok_generation"],
        "dense_identity": kwargs["dense_generation"],
    }
    training = {
        "cofitok": kwargs["cofitok_full_training"],
        "dense_identity": kwargs["dense_full_training"],
    }

    evidence = _inference_export_evidence(
        kwargs["inference_exports"],
        kwargs["inference_artifact_files"],
        kwargs["inference_smoke_files"],
        generation,
        training,
        kwargs["final_gate"],
        expected_export_revision=FULL_REVISION,
        expected_export_branch="scale/generative-system",
    )

    assert evidence["cofitok"]["execution_git"]["revision"] == FULL_REVISION
    assert len(
        evidence["cofitok"]["execution_runtime_environment_sha256"]
    ) == 64

    kwargs["inference_exports"]["cofitok_smoke"]["git"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="smoke execution Git"):
        _inference_export_evidence(
            kwargs["inference_exports"],
            kwargs["inference_artifact_files"],
            kwargs["inference_smoke_files"],
            generation,
            training,
            kwargs["final_gate"],
            expected_export_revision=FULL_REVISION,
            expected_export_branch="scale/generative-system",
        )


def test_completion_audit_rejects_missing_or_tampered_smoke_pngs() -> None:
    kwargs = _kwargs()
    kwargs["inference_smoke_files"]["cofitok"] = {
        "status": "invalid",
        "root": "/missing/smoke/cofitok",
        "error": "inference smoke output SHA256 differs",
    }

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_weakened_inference_smoke_request() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_smoke"]["request"]["sample_steps"] = 1

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


def test_completion_audit_rejects_inference_artifact_authorization_drift() -> None:
    kwargs = _kwargs()
    kwargs["inference_artifact_files"]["dense_identity"][
        "source_training_authorization"
    ]["gate_identity_sha256"] = "d" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_inference_preflight_authorization_drift() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_preflight"][
        "training_authorization"
    ]["gate_identity_sha256"] = "d" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_inference_artifact_release_drift() -> None:
    kwargs = _kwargs()
    kwargs["inference_artifact_files"]["dense_identity"][
        "release_authorization"
    ]["gate_identity_sha256"] = "d" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_inference_preflight_release_drift() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_preflight"][
        "release_authorization"
    ]["gate_identity_sha256"] = "d" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_requires_release_only_export_preflight() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["cofitok_preflight"][
        "release_authorization_required"
    ] = False

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_requires_release_only_export_smoke() -> None:
    kwargs = _kwargs()
    kwargs["inference_exports"]["dense_identity_smoke"]["checkpoint"][
        "release_authorization_required"
    ] = False

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deployable_ema_inference_artifacts"]


def test_completion_audit_rejects_different_release_gate_files() -> None:
    kwargs = _kwargs()
    dense_release_locations = (
        kwargs["inference_exports"]["dense_identity_export"][
            "release_authorization"
        ],
        kwargs["inference_exports"]["dense_identity_preflight"][
            "release_authorization"
        ],
        kwargs["inference_exports"]["dense_identity_smoke"]["checkpoint"][
            "release_authorization"
        ],
        kwargs["inference_artifact_files"]["dense_identity"][
            "release_authorization"
        ],
    )
    for release in dense_release_locations:
        release["gate_sha256"] = "7" * 64

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


def test_completion_audit_rejects_milestone_source_report_drift() -> None:
    kwargs = _kwargs()
    kwargs["milestone_source_verifications"][50_000]["source_reports"][
        "cofitok_generation"
    ]["sha256"] = "f" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["full_milestone_evaluations"]


def test_completion_audit_rejects_milestone_sampling_protocol_drift() -> None:
    kwargs = _kwargs()
    kwargs["milestones"][50_000]["methods"]["cofitok"]["sampling"][
        "guidance_scale"
    ] = 2.0

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["full_milestone_evaluations"]


def test_completion_audit_direct_cli_reports_in_progress(tmp_path) -> None:
    project_root = tmp_path / "empty_project"
    project_root.mkdir()
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
            str(project_root),
            "--output-root",
            str(output_root),
            "--expected-full-revision",
            FULL_REVISION,
            "--expected-10pct-revision",
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
