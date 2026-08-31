from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cofitok.generation import sampling_recovery_audit as audit


def _identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json(path: Path, payload: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _identity(path)


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    output = tmp_path / "output"
    output.mkdir()
    real_root = tmp_path / "real"
    real_root.mkdir()
    real_set = {
        "digest_schema": "cofitok_image_tree_sha256_v1",
        "image_count": 50_000,
        "root": real_root.resolve().as_posix(),
        "sha256": "f" * 64,
    }
    design = _json(tmp_path / "design.json", {"role": "design"})
    separate_auth = _json(tmp_path / "separate_authorization.json", {"role": "user_stage"})
    execution_auth = _json(tmp_path / "execution_authorization.json", {"role": "execution"})
    preparation = _json(tmp_path / "preparation.json", {"status": "prepared"})
    observation_manifest = _json(tmp_path / "observation_manifest.json", {"status": "complete"})
    terminal = _json(tmp_path / "terminal_route.json", {"terminal_status": "hold"})
    real_identity_report = _json(tmp_path / "real_identity_report.json", {"role": "real"})
    artifact_reference_report = _json(tmp_path / "real_artifact_report.json", {"role": "artifact"})
    subset_manifest = _json(tmp_path / "real_subset_manifest.json", {"role": "subset"})

    source_git = {
        "branch": "scale/source",
        "revision": "c" * 40,
        "dirty": False,
    }
    evaluator_git = {
        "branch": "scale/evaluator",
        "revision": "e" * 40,
        "tracked_dirty": False,
    }
    execution_git = {
        **evaluator_git,
        "tree": "d" * 40,
    }
    seed = 123
    methods: dict[str, dict[str, object]] = {}
    for method in audit.METHODS:
        run = output / method
        run.mkdir()
        payload = (method + " checkpoint").encode("ascii")
        checkpoint = run / "checkpoint_step_00100000.pt"
        checkpoint.write_bytes(payload)
        checkpoint_identity = _identity(checkpoint)
        sidecar = _json(
            checkpoint.with_name(f"{checkpoint.name}.integrity.json"),
            {
                "checkpoint": checkpoint.name,
                "checkpoint_bytes": checkpoint_identity["bytes"],
                "checkpoint_sha256": checkpoint_identity["sha256"],
                "step": 100_000,
            },
        )
        latest = _json(
            run / "latest.json",
            {
                "checkpoint": checkpoint.name,
                "checkpoint_bytes": checkpoint_identity["bytes"],
                "checkpoint_sha256": checkpoint_identity["sha256"],
                "integrity_manifest": Path(sidecar["path"]).name,
                "step": 100_000,
            },
        )
        training_report = _json(
            run / "training_report.json",
            {
                "completed_steps": 100_000,
                "git": source_git,
                "output_dir": run.resolve().as_posix(),
                "target_steps": 100_000,
                "training_complete": True,
            },
        )
        methods[method] = {
            "checkpoint": checkpoint_identity,
            "checkpoint_step": 100_000,
            "integrity_sidecar": sidecar,
            "latest": latest,
            "prefix_budget": audit.PREFIX_BUDGETS[method],
            "training_report": training_report,
        }

    execution_sources: dict[str, dict[str, object]] = {}
    for name in (
        "classifier",
        "classifier_report",
        "cross_protocol_reconciliation",
        "dataset",
        "evaluator",
        "post_reconciliation_verification",
        "quality_bridge_result",
        "real_set",
        "runtime_environment",
        "training_pair_report",
    ):
        execution_sources[name] = _json(tmp_path / "execution_sources" / f"{name}.json", {"name": name})
    execution_sources["post_reconciliation_decision"] = terminal

    execution = {
        "authorized_actions": copy.deepcopy(audit.AUTHORIZED_ACTIONS),
        "design_identity": design,
        "evaluator_git": evaluator_git,
        "git": execution_git,
        "methods": methods,
        "output_root": output.resolve().as_posix(),
        "output_root_non_overlapping": True,
        "preparation_identity": preparation,
        "random_stream": {
            "fresh": True,
            "namespace": "test_sampling_recovery_123",
            "seed": seed,
            "start_index": 0,
        },
        "real_set": real_set,
        "runtime_environment_sha256s": {
            "artifact": "a" * 64,
            "class_fidelity": "b" * 64,
            "metrics": "b" * 64,
            "sampling": "c" * 64,
        },
        "schema": "cofitok_matched_epsilon_stability_execution_authorization_v1",
        "scope": "matched_1000_sample_epsilon_stability_sampling_diagnostic_only",
        "separate_execution_authorization_identity": separate_auth,
        "source_bindings": execution_sources,
        "status": "approved",
        "terminal_route_receipt_identity": terminal,
    }

    observations: list[dict[str, object]] = []
    fids: dict[str, dict[str, float]] = {}
    for case_index, case_id in enumerate(sorted(audit.CASE_IDS)):
        fids[case_id] = {}
        for method_index, method in enumerate(audit.METHODS):
            fid = 100.0 + method_index * 10.0 - (0.0 if case_id == audit.REFERENCE_CASE else case_index)
            metrics = {
                "channel_saturation_fraction": 0.01
                + (0.001 if case_id != audit.REFERENCE_CASE else 0.0),
                "class_top1": 0.01,
                "class_top5": 0.02,
                "fid": fid,
                "inception_score": 2.0,
                "median_filter_residual_fraction": 0.02,
                "total_variation": 0.1,
            }
            fids[case_id][method] = fid
            start975 = case_id.startswith("start975_")
            recompute_after_constraint = case_id.endswith("_recompute")
            sigma_scaled = case_id in {
                "start975_sigma_hard_clip",
                "start975_sigma_dynamic_threshold",
                "start975_sigma_dynamic_threshold_recompute",
            }
            sampling = {
                "actual_timesteps": audit._timesteps(975 if start975 else 999),
                "batch_size": 32,
                "cfg_batch_mode": "batched",
                "class_schedule": "balanced_modulo",
                "clip_x0": True,
                "dynamic_threshold_percentile": 0.0
                if case_id in {
                    audit.REFERENCE_CASE,
                    "start975_unit_hard_clip",
                    "start975_sigma_hard_clip",
                    "terminal_hard_clip_recompute",
                }
                else 0.995,
                "eta": 0.0,
                "guidance_rescale": 0.0,
                "guidance_scale": 1.5,
                "image_shape": [3, 256, 256],
                "inference_api": {"name": "cofitok.generation.GenerationSession", "version": 1},
                "initial_noise_scale": "schedule_sigma" if sigma_scaled else "unit",
                "num_samples": 1000,
                "num_train_timesteps": 1000,
                "precision": "bf16",
                "prefix_budgets": [audit.PREFIX_BUDGETS[method]],
                "protocol_schema": "cofitok_ddim_sampling_v1",
                "random_stream": {
                    "batch_size_invariant": True,
                    "prefix_budgets_share_stream": True,
                    "resume_index_invariant": True,
                    "scope": "per_global_sample_index",
                    "seed_formula": "(seed + global_index) mod 2^63",
                },
                "recompute_epsilon_after_x0_constraint": recompute_after_constraint,
                "requested_start_timestep": 975 if start975 else None,
                "sample_set_digest": {"algorithm": "sha256", "framing": "filename_utf8_nul_file_bytes_nul"},
                "sample_steps": 100,
                "sampler": "ddim",
                "scale_initial_noise_by_sigma": sigma_scaled,
                "seed": seed,
                "start_index": 0,
                "start_timestep": 975 if start975 else 999,
                "x0_constraint": "dynamic_threshold"
                if case_id in {
                    "terminal_dynamic_threshold",
                    "terminal_dynamic_threshold_recompute",
                    "start975_sigma_dynamic_threshold",
                    "start975_sigma_dynamic_threshold_recompute",
                }
                else "clip",
            }
            checkpoint = methods[method]["checkpoint"]
            sample_set_sha = hashlib.sha256(f"{case_id}:{method}".encode("ascii")).hexdigest()
            case_dir = output / "cases" / case_id / method
            artifact_metrics = {
                key: metrics[key]
                for key in ("channel_saturation_fraction", "median_filter_residual_fraction", "total_variation")
            }
            artifact_report = _json(
                case_dir / "artifact_report.json",
                {
                    "metrics": artifact_metrics,
                    "role": "generation_epsilon_stability_artifact_statistics_report",
                    "sample_count": 1000,
                    "sample_set_sha256": sample_set_sha,
                    "schema_version": 1,
                    "status": "completed",
                },
            )
            metrics_report = _json(
                case_dir / "metrics_report.json",
                {
                    "counts": {"generated_image_count": 1000},
                    "metrics": {"frechet_inception_distance": fid, "inception_score_mean": 2.0},
                    "protocol": "torch_fidelity_directory_metrics",
                    "sample_provenance": {
                        "checkpoint_sha256": checkpoint["sha256"],
                        "checkpoint_step": 100_000,
                        "sample_set_sha256": sample_set_sha,
                    },
                    "status": "completed",
                },
            )
            class_report = _json(
                case_dir / "class_fidelity_report.json",
                {
                    "metrics": {"sample_count": 1000, "top1_accuracy": 0.01, "top5_accuracy": 0.02},
                    "role": "generation_class_fidelity_report",
                    "sample_provenance": {
                        "checkpoint_sha256": checkpoint["sha256"],
                        "checkpoint_step": 100_000,
                        "sample_set_sha256": sample_set_sha,
                    },
                    "status": "completed",
                },
            )
            sampling_report = _json(
                case_dir / "sampling_report.json",
                {
                    "checkpoint_sha256": checkpoint["sha256"],
                    "checkpoint_step": 100_000,
                    "sampling": sampling,
                    "status": "completed",
                },
            )
            observation = {
                "authorization_boundary": copy.deepcopy(audit.AUTHORIZATION_BOUNDARY),
                "case_id": case_id,
                "checkpoint": checkpoint,
                "design_identity": design,
                "execution_authorization_identity": execution_auth,
                "integrity_sidecar": methods[method]["integrity_sidecar"],
                "method": method,
                "metrics": metrics,
                "real_set": real_set,
                "sample_count": 1000,
                "sample_set_sha256": sample_set_sha,
                "sampling": sampling,
                "schema": audit.OBSERVATION_SCHEMA,
                "source_reports": {
                    "artifact_report": artifact_report,
                    "class_fidelity_report": class_report,
                    "metrics_report": metrics_report,
                    "sampling_report": sampling_report,
                },
                "status": "pass",
            }
            observation_path = case_dir / "observation.json"
            observation_identity = _json(observation_path, observation)
            observation["identity"] = observation_identity
            observations.append(observation)

    real_artifact = {
        "metrics": {
            "channel_saturation_fraction": 0.01,
            "median_filter_residual_fraction": 0.02,
            "total_variation": 0.1,
        },
        "real_set": real_set,
        "real_set_identity": real_identity_report,
        "sample_count": 1000,
        "sample_set_sha256": "d" * 64,
        "source_reports": {
            "artifact_report": artifact_reference_report,
            "subset_manifest": subset_manifest,
        },
    }
    real_artifact_source = {
        "authorization_boundary": copy.deepcopy(audit.AUTHORIZATION_BOUNDARY),
        "design_identity": design,
        "execution_authorization_identity": execution_auth,
        **real_artifact,
        "schema": audit.REAL_ARTIFACT_SCHEMA,
        "status": "pass",
    }
    real_artifact_reference = _json(
        tmp_path / "real_artifact_reference.json",
        real_artifact_source,
    )
    top_sources = {
        "design": design,
        "execution_authorization": execution_auth,
        "observation_manifest": observation_manifest,
        "preparation": preparation,
        "real_artifact_reference": real_artifact_reference,
        "separate_execution_authorization": separate_auth,
        "terminal_route_receipt": terminal,
    }
    observations_by_case = {
        case_id: {observation["method"]: observation for observation in observations if observation["case_id"] == case_id}
        for case_id in audit.CASE_IDS
    }

    def comparison(candidate_case: str, method: str, reference_case: str) -> dict[str, object]:
        candidate_metrics = observations_by_case[candidate_case][method]["metrics"]
        reference_metrics = observations_by_case[reference_case][method]["metrics"]
        artifact_distance = {
            key: abs(candidate_metrics[key] - real_artifact["metrics"][key])
            for key in audit.ARTIFACT_METRIC_KEYS
        }
        reference_artifact_distance = {
            key: abs(reference_metrics[key] - real_artifact["metrics"][key])
            for key in audit.ARTIFACT_METRIC_KEYS
        }
        checks = {
            "channel_saturation_fraction_real_distance_non_regression": artifact_distance["channel_saturation_fraction"] <= reference_artifact_distance["channel_saturation_fraction"],
            "class_top1_non_regression": candidate_metrics["class_top1"] >= reference_metrics["class_top1"],
            "class_top5_non_regression": candidate_metrics["class_top5"] >= reference_metrics["class_top5"],
            "fid_strictly_lower": candidate_metrics["fid"] < reference_metrics["fid"],
            "median_filter_residual_fraction_real_distance_non_regression": artifact_distance["median_filter_residual_fraction"] <= reference_artifact_distance["median_filter_residual_fraction"],
            "total_variation_real_distance_non_regression": artifact_distance["total_variation"] <= reference_artifact_distance["total_variation"],
        }
        return {
            "artifact_distance_to_real": artifact_distance,
            "checks": checks,
            "fid_relative_improvement": (reference_metrics["fid"] - candidate_metrics["fid"]) / reference_metrics["fid"],
            "metrics": copy.deepcopy(candidate_metrics),
            "passes": all(checks.values()),
            "reference_artifact_distance_to_real": reference_artifact_distance,
            "reference_metrics": copy.deepcopy(reference_metrics),
        }

    candidates = []
    for case_id in sorted(audit.CANDIDATE_CASE_IDS):
        parent_case = audit.REFERENCE_CASE
        method_comparisons = {
            method: {
                "versus_legacy": comparison(case_id, method, audit.REFERENCE_CASE),
                "versus_parent": comparison(case_id, method, parent_case),
            }
            for method in audit.METHODS
        }
        worst = min(
            (fids[audit.REFERENCE_CASE][method] - fids[case_id][method])
            / fids[audit.REFERENCE_CASE][method]
            for method in audit.METHODS
        )
        candidates.append(
            {
                "attribution_role": "single_factor",
                "case_id": case_id,
                "incremental_control": "test_control",
                "methods": method_comparisons,
                "parent_case_id": parent_case,
                "passes_shared_recovery_screen": all(
                    method_comparisons[method]["versus_legacy"]["passes"]
                    for method in audit.METHODS
                ),
                "worst_method_fid_relative_improvement": worst,
            }
        )
    result = {
        "authorization_boundary": copy.deepcopy(audit.AUTHORIZATION_BOUNDARY),
        "candidates": candidates,
        "claim_boundary": copy.deepcopy(audit.CLAIM_BOUNDARY),
        "execution": execution,
        "generation_advantage_proven": False,
        "observations": observations,
        "real_artifact_reference": real_artifact,
        "schema": audit.RESULT_SCHEMA,
        "scientific_status": "screening_only",
        "selected_case_id": None,
        "selection_contract": copy.deepcopy(audit.SELECTION_CONTRACT),
        "selection_status": "no_shared_sampling_recovery_candidate",
        "source_bindings": top_sources,
        "status": "pass",
    }
    result_path = tmp_path / "sampling_recovery_result.json"
    _json(result_path, result)
    return result, result_path


def test_validate_result_rehashes_sources_and_preserves_boundary(tmp_path: Path) -> None:
    result, result_path = _fixture(tmp_path)
    summary = audit.validate_result(
        result,
        result_path=result_path,
        expected_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        allowed_output_prefix=tmp_path.resolve().as_posix(),
    )
    assert summary["observation_count"] == 16
    assert summary["case_count"] == 8
    assert summary["candidate_count"] == 7
    assert summary["generation_advantage_proven"] is False
    assert summary["authorization_boundary"] == audit.AUTHORIZATION_BOUNDARY


def test_validate_result_rejects_physical_source_drift(tmp_path: Path) -> None:
    result, result_path = _fixture(tmp_path)
    report = Path(result["observations"][0]["source_reports"]["metrics_report"]["path"])
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source bytes or SHA256 changed"):
        audit.validate_result(
            result,
            result_path=result_path,
            allowed_output_prefix=tmp_path.resolve().as_posix(),
        )


def test_validate_result_rejects_candidate_selection_upgrade(tmp_path: Path) -> None:
    result, result_path = _fixture(tmp_path)
    tampered = copy.deepcopy(result)
    tampered["selected_case_id"] = "start975_unit_hard_clip"
    with pytest.raises(ValueError, match="no-candidate boundary"):
        audit.validate_result(
            tampered,
            result_path=result_path,
            allowed_output_prefix=tmp_path.resolve().as_posix(),
        )


def test_validate_result_rejects_candidate_screen_upgrade(tmp_path: Path) -> None:
    result, result_path = _fixture(tmp_path)
    tampered = copy.deepcopy(result)
    tampered["candidates"][0]["passes_shared_recovery_screen"] = True
    with pytest.raises(ValueError, match="weakens the no-candidate result"):
        audit.validate_result(
            tampered,
            result_path=result_path,
            allowed_output_prefix=tmp_path.resolve().as_posix(),
        )
