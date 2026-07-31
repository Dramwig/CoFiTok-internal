from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from cofitok.environment import runtime_environment_sha256
from cofitok.generation import sampling_protocol_contract
from cofitok.generation.stability_scaling import (
    build_stability_scaling_decision,
    validate_stability_scaling_decision,
)
from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.generation_paths import generation_stability_workspace_paths
from cofitok.reporting import file_sha256, write_json_report

try:
    from scripts.audit_large_scale_generation_completion import (
        _comparison_evidence,
        _inference_export_evidence,
        _runtime_selection_evidence,
        _verify_checkpoint_file,
        _verify_formal_real_set_files,
        _verify_formal_sample_files,
        _verify_inference_artifact_file,
        _verify_inference_smoke_outputs,
    )
    from scripts.build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
    from scripts.build_generation_stability_50k_summary import build_summary
    from scripts.build_large_scale_generation_comparison import (
        verify_comparison_source_reports,
    )
    from scripts.validate_generation_training_pair import validate_training_pair
except ModuleNotFoundError:
    from audit_large_scale_generation_completion import (
        _comparison_evidence,
        _inference_export_evidence,
        _runtime_selection_evidence,
        _verify_checkpoint_file,
        _verify_formal_real_set_files,
        _verify_formal_sample_files,
        _verify_inference_artifact_file,
        _verify_inference_smoke_outputs,
    )
    from build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
    from build_generation_stability_50k_summary import build_summary
    from build_large_scale_generation_comparison import (
        verify_comparison_source_reports,
    )
    from validate_generation_training_pair import validate_training_pair


MILESTONE_STEPS = (50_000, 100_000, 200_000, 300_000)
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
FORMAL_REAL_DIR = Path(
    "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val"
)
OFFICIAL_RELATED = Path(
    "artifacts/reports/baselines/official_related_methods_2026-07-11_final/"
    "official_related_methods_table.json"
)


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"__load_error__": f"{path}: {error}"}
    if not isinstance(payload, dict):
        return {"__load_error__": f"{path}: expected a JSON object"}
    return payload


def _raise_load_error(payload: dict[str, Any]) -> None:
    error = payload.get("__load_error__")
    if error is not None:
        raise ValueError(str(error))


def _check(
    name: str,
    payloads: list[dict[str, Any] | None],
    validator: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if any(payload is None for payload in payloads):
        return {"name": name, "status": "missing", "evidence": None}
    try:
        evidence = validator()
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        return {
            "name": name,
            "status": "fail",
            "error": str(error),
            "evidence": None,
        }
    return {"name": name, "status": "pass", "evidence": evidence}


def aggregate_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row["name"] for row in checks if row["status"] == "fail"]
    missing = [row["name"] for row in checks if row["status"] == "missing"]
    complete = not failed and not missing
    return {
        "schema_version": 1,
        "status": "pass" if complete else ("failed" if failed else "incomplete"),
        "complete": complete,
        "failed_checks": failed,
        "missing_checks": missing,
        "checks": checks,
    }


def _validate_revision(value: str, *, name: str) -> str:
    if SHA1.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase Git SHA-1")
    return value


def _validate_sha256(value: str, *, name: str) -> str:
    if SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _verified_source(
    descriptor: dict[str, Any],
    *,
    expected_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(str(descriptor.get("path", ""))).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"source path differs: {path}")
    if not path.is_file():
        raise ValueError(f"source file is missing: {path}")
    identity = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if identity != descriptor:
        raise ValueError(f"source identity differs: {path}")
    payload = _read_optional(path)
    if payload is None:
        raise ValueError(f"source file disappeared: {path}")
    _raise_load_error(payload)
    return identity, payload


def stability_decision_evidence(
    decision: dict[str, Any],
    *,
    decision_path: Path,
    expected_sha256: str,
    expected_source_revision: str,
) -> dict[str, Any]:
    _raise_load_error(decision)
    actual_sha256 = file_sha256(decision_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("5K stability decision SHA256 differs")
    evidence = validate_stability_scaling_decision(
        decision,
        expected_source_revision=expected_source_revision,
        expected_next_stage="fresh_matched_50k_preparation",
    )
    sources = decision.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("5K stability decision source provenance is missing")
    screening_descriptor = sources.get("screening_report")
    robust_descriptors = sources.get("robust_reports")
    if not isinstance(screening_descriptor, dict) or not isinstance(
        robust_descriptors,
        list,
    ):
        raise ValueError("5K stability decision source descriptors are malformed")
    screening_identity, screening = _verified_source(screening_descriptor)
    robust_sources = []
    robust_reports = []
    for descriptor in robust_descriptors:
        if not isinstance(descriptor, dict):
            raise ValueError("5K stability robust source descriptor is malformed")
        identity, report = _verified_source(descriptor)
        robust_sources.append(identity)
        robust_reports.append(report)
    rebuilt = build_stability_scaling_decision(
        screening_report=screening,
        robust_reports=robust_reports,
        next_stage="fresh_matched_50k_preparation",
    )
    declared = {key: value for key, value in decision.items() if key != "sources"}
    if rebuilt != declared:
        raise ValueError("5K stability decision differs from its rehashed sources")
    return {
        **evidence,
        "decision_sha256": actual_sha256,
        "screening_source": screening_identity,
        "robust_sources": robust_sources,
    }


def monitor_evidence(
    report: dict[str, Any],
    *,
    expected_name: str,
    expected_steps: int,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    _raise_load_error(report)
    if (
        report.get("schema_version") != 2
        or report.get("monitor") != expected_name
        or report.get("status") != "pass"
        or report.get("stage") != "complete"
        or report.get("issues") != []
    ):
        raise ValueError(f"{expected_name} monitor did not pass cleanly")
    if report.get("git") != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError(f"{expected_name} monitor Git identity differs")
    runs = report.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"cofitok", "dense_identity"}:
        raise ValueError(f"{expected_name} monitor run set differs")
    run_evidence = {}
    for method, run in runs.items():
        if (
            run.get("complete") is not True
            or int(run.get("expected_steps", -1)) != expected_steps
            or int(run.get("last_step", -1)) != expected_steps
            or run.get("health_issues") != []
            or run.get("run_manifest", {}).get("status") != "verified"
        ):
            raise ValueError(f"{expected_name} {method} run is incomplete")
        integrity = run.get("checkpoint_integrity", {})
        manifests = integrity.get("manifests")
        if (
            integrity.get("policy") != "required"
            or integrity.get("expected_checkpoint_revision")
            != expected_revision
            or not isinstance(manifests, list)
            or not manifests
            or any(row.get("status") != "metadata_verified" for row in manifests)
            or integrity.get("latest_binding", {}).get("status")
            != "metadata_verified"
        ):
            raise ValueError(
                f"{expected_name} {method} checkpoint metadata is incomplete"
            )
        run_evidence[method] = {
            "last_step": expected_steps,
            "metric_rows": int(run.get("metric_rows", 0)),
            "checkpoint_steps": [int(row["step"]) for row in manifests],
            "runtime_environment_sha256": run["run_manifest"].get(
                "runtime_environment_sha256"
            ),
            "dataset_identity_sha256": run["run_manifest"].get(
                "dataset_identity_sha256"
            ),
        }
    return {
        "monitor": expected_name,
        "git": report["git"],
        "runs": run_evidence,
    }


def pair_summary_evidence(
    summary: dict[str, Any],
    *,
    cofitok_training_path: Path,
    dense_training_path: Path,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    _raise_load_error(summary)
    sources = summary.get("sources")
    expected_paths = {
        "cofitok_training": cofitok_training_path,
        "dense_training": dense_training_path,
    }
    if not isinstance(sources, dict) or set(sources) != {
        "cofitok_training",
        "dense_training",
        "decision_validation",
        "config_validation",
    }:
        raise ValueError("stability 50K pair summary sources are incomplete")
    loaded = {}
    verified = {}
    for name, descriptor in sources.items():
        if not isinstance(descriptor, dict):
            raise ValueError(f"stability 50K source is malformed: {name}")
        identity, payload = _verified_source(
            descriptor,
            expected_path=expected_paths.get(name),
        )
        verified[name] = identity
        loaded[name] = payload
    rebuilt = build_summary(
        cofitok_training=loaded["cofitok_training"],
        dense_training=loaded["dense_training"],
        decision_validation=loaded["decision_validation"],
        config_validation=loaded["config_validation"],
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    declared = {key: value for key, value in summary.items() if key != "sources"}
    if declared != rebuilt:
        raise ValueError("stability 50K pair summary differs from its sources")
    return {
        "git": rebuilt["git"],
        "completed_steps_per_method": rebuilt["completed_steps_per_method"],
        "images_seen_per_method": rebuilt["images_seen_per_method"],
        "effective_batch_size": rebuilt["effective_batch_size"],
        "source_sha256": {
            name: identity["sha256"] for name, identity in verified.items()
        },
    }


def gate_evidence(
    gate: dict[str, Any],
    *,
    gate_path: Path,
    expected_sha256: str,
    stage: str,
    source_profile: str,
    training_revision: str,
    training_branch: str,
    evaluation_revision: str,
    evaluation_branch: str,
) -> dict[str, Any]:
    _raise_load_error(gate)
    if file_sha256(gate_path) != expected_sha256:
        raise ValueError(f"{stage} gate SHA256 differs")
    if gate.get("source_profile") != source_profile:
        raise ValueError(f"{stage} gate source profile differs")
    provenance = gate.get("provenance_contract")
    expected_provenance = {
        "training_revision": training_revision,
        "training_branch": training_branch,
        "evaluation_revision": evaluation_revision,
        "evaluation_branch": evaluation_branch,
    }
    if provenance != expected_provenance:
        raise ValueError(f"{stage} gate code provenance differs")
    authorization = validate_generation_gate_authorization(
        gate,
        expected_stage=stage,
    )
    source_verification = verify_generation_gate_source_reports(gate)
    if source_verification.get("source_profile") != source_profile:
        raise ValueError(f"{stage} gate source verification profile differs")
    return {
        **authorization,
        "gate_sha256": expected_sha256,
        "provenance": provenance,
        "source_report_sha256": {
            name: identity["sha256"]
            for name, identity in source_verification["source_reports"].items()
        },
    }


def progress_audit_evidence(
    audits: dict[str, dict[str, Any]],
    *,
    expected_steps: int,
    required_checkpoint_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    evidence = {}
    for method, report in audits.items():
        _raise_load_error(report)
        validation = report.get("validation", {})
        checkpoint = report.get("checkpoint", {})
        if (
            report.get("status") != "complete"
            or report.get("issues") != []
            or int(report.get("last_step", -1)) != expected_steps
            or validation.get("logging_complete") is not True
            or checkpoint.get("missing_required_steps") != []
            or checkpoint.get("latest_integrity", {}).get("status")
            != "verified"
        ):
            raise ValueError(f"{method} training progress audit is incomplete")
        observed_required = tuple(
            int(step) for step in checkpoint.get("required_steps", [])
        )
        if required_checkpoint_steps and observed_required != required_checkpoint_steps:
            raise ValueError(f"{method} protected checkpoint contract differs")
        evidence[method] = {
            "last_step": expected_steps,
            "validation_event_count": int(validation.get("event_count", 0)),
            "checkpoint_steps": list(checkpoint.get("steps", [])),
            "required_checkpoint_steps": list(observed_required),
            "checkpoint_sha256": checkpoint["latest_integrity"].get(
                "checkpoint_sha256"
            ),
        }
    return evidence


def checkpoint_evidence(
    verified: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
    *,
    expected_step: int,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    evidence = {}
    for method in ("cofitok", "dense_identity"):
        file_report = verified[method]
        training = training_reports[method]
        latest = training.get("latest_checkpoint", {})
        if file_report.get("status") != "verified":
            raise ValueError(
                f"{method} checkpoint verification failed: "
                f"{file_report.get('error', 'unknown error')}"
            )
        if (
            int(file_report.get("step", -1)) != expected_step
            or file_report.get("checkpoint_sha256")
            != latest.get("checkpoint_sha256")
            or int(file_report.get("checkpoint_bytes", -1))
            != int(latest.get("checkpoint_bytes", -2))
            or file_report.get("git_revision") != expected_revision
            or file_report.get("git_branch") != expected_branch
            or file_report.get("git_dirty") is not False
        ):
            raise ValueError(f"{method} checkpoint identity differs")
        evidence[method] = {
            "path": file_report["path"],
            "checkpoint_sha256": file_report["checkpoint_sha256"],
            "checkpoint_bytes": int(file_report["checkpoint_bytes"]),
            "integrity_manifest": file_report["integrity_manifest"],
        }
    return evidence


def full_training_evidence(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
    *,
    promotion_gate: dict[str, Any],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    _raise_load_error(cofitok)
    _raise_load_error(dense)
    return validate_training_pair(
        cofitok,
        dense,
        expected_steps=300_000,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_dataset="imagenet_256",
        max_parameter_gap=0.02,
        expected_recipe_stage="stability_full",
        expected_authorization_gate=promotion_gate,
    )


def full_storage_capacity_evidence(
    report: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    expected_path: Path,
) -> dict[str, Any]:
    _raise_load_error(report)
    if (
        report.get("schema_version") != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != "full_training"
        or report.get("status") != "pass"
    ):
        raise ValueError("stability full storage capacity report is invalid")
    if report.get("git") != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("stability full storage capacity Git identity differs")
    filesystem = report.get("filesystem", {})
    observed_path = Path(str(filesystem.get("path", ""))).resolve()
    if observed_path != expected_path.resolve():
        raise ValueError("stability full storage capacity path differs")
    plan = report.get("plan", {})
    minimums = {
        "checkpoint_count": 16,
        "sample_count": 16_384,
        "estimated_sample_bytes_each": 256 * 1024,
        "additional_bytes": 16 * 1024**3,
        "safety_margin_bytes": 64 * 1024**3,
    }
    for key, minimum in minimums.items():
        if int(plan.get(key, -1)) < minimum:
            raise ValueError(f"stability full storage reserve {key} was weakened")
    reference_bytes = int(plan.get("reference_checkpoint_bytes_each", -1))
    multiplier = float(plan.get("checkpoint_size_multiplier", math.nan))
    planned_checkpoint_bytes = int(plan.get("checkpoint_bytes_each", -1))
    if reference_bytes < 1 or not math.isfinite(multiplier) or multiplier < 4.0:
        raise ValueError("stability full checkpoint scaling was weakened")
    if planned_checkpoint_bytes != math.ceil(reference_bytes * multiplier):
        raise ValueError("stability full checkpoint scaling arithmetic differs")
    checkpoint_reserve = int(plan["checkpoint_count"]) * planned_checkpoint_bytes
    sample_reserve = int(plan["sample_count"]) * int(
        plan["estimated_sample_bytes_each"]
    )
    if checkpoint_reserve != int(plan.get("checkpoint_reserve_bytes", -1)):
        raise ValueError("stability full checkpoint reserve arithmetic differs")
    if sample_reserve != int(plan.get("sample_reserve_bytes", -1)):
        raise ValueError("stability full sample reserve arithmetic differs")
    required = (
        checkpoint_reserve
        + sample_reserve
        + int(plan["additional_bytes"])
        + int(plan["safety_margin_bytes"])
    )
    if required != int(plan.get("required_free_bytes", -1)):
        raise ValueError("stability full storage requirement arithmetic differs")
    total = int(filesystem.get("total_bytes", -1))
    used = int(filesystem.get("used_bytes", -1))
    free = int(filesystem.get("free_bytes", -1))
    if total < 1 or used < 0 or free < required or used + free > total:
        raise ValueError("stability full storage filesystem headroom is invalid")
    if int(report.get("headroom_bytes", -1)) != free - required:
        raise ValueError("stability full storage headroom arithmetic differs")
    return {
        "reference_checkpoint_bytes_each": reference_bytes,
        "checkpoint_size_multiplier": multiplier,
        "checkpoint_bytes_each": planned_checkpoint_bytes,
        "required_free_bytes": required,
        "free_bytes": free,
        "headroom_bytes": free - required,
    }


def milestone_evidence(
    reports: dict[int, dict[str, Any]],
    checkpoint_files: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    evidence = {}
    warnings = []
    for step in MILESTONE_STEPS:
        report = reports[step]
        _raise_load_error(report)
        source_verification = verify_milestone_source_reports(report)
        step_evidence, step_warnings = validate_milestone_report(
            report,
            expected_step=step,
            source_verification=source_verification,
        )
        for method in ("cofitok", "dense_identity"):
            verified = checkpoint_files[step][method]
            if verified.get("status") != "verified":
                raise ValueError(f"{method} milestone {step} checkpoint is invalid")
            row = report["methods"][method]
            if (
                int(verified.get("step", -1)) != step
                or verified.get("checkpoint_sha256")
                != row.get("checkpoint_sha256")
            ):
                raise ValueError(
                    f"{method} milestone {step} checkpoint SHA256 differs"
                )
        evidence[str(step)] = step_evidence
        warnings.extend(step_warnings)
    return {"milestones": evidence, "warnings": warnings}


def formal_generation_evidence(
    reports: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
    sample_files: dict[str, dict[str, Any]],
    real_set_files: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if real_set_files.get("status") != "verified":
        raise ValueError(
            "formal real-set verification failed: "
            f"{real_set_files.get('error', 'unknown error')}"
        )
    evidence = {}
    sampling_environments = set()
    evaluator_environments = set()
    for method, report in reports.items():
        _raise_load_error(report)
        provenance = report.get("sample_provenance", {})
        sampling = provenance.get("sampling", {})
        if (
            report.get("status") != "completed"
            or int(report.get("counts", {}).get("generated_image_count", -1))
            != 50_000
            or report.get("git")
            != {
                "revision": expected_revision,
                "branch": expected_branch,
                "tracked_dirty": False,
            }
            or provenance.get("git")
            != {
                "revision": expected_revision,
                "branch": expected_branch,
                "tracked_dirty": False,
            }
            or int(provenance.get("checkpoint_step", -1)) != 300_000
            or provenance.get("weights") != "ema"
        ):
            raise ValueError(f"{method} formal generation provenance differs")
        contract = sampling_protocol_contract(
            sampling,
            stage="full",
            expected_num_train_timesteps=int(
                training_reports[method]["config"]["diffusion"][
                    "num_train_timesteps"
                ]
            ),
        )
        if contract["valid"] is not True:
            raise ValueError(
                f"{method} formal sampling protocol is invalid: "
                + ", ".join(contract["issues"])
            )
        if sampling.get("inference_api") != {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        }:
            raise ValueError(f"{method} formal sampling bypassed GenerationSession")
        latest = training_reports[method].get("latest_checkpoint", {})
        if provenance.get("checkpoint_sha256") != latest.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{method} formal sampling checkpoint differs")
        sample_verification = sample_files[method]
        if (
            sample_verification.get("status") != "verified"
            or sample_verification.get("sample_set_sha256")
            != provenance.get("sample_set_sha256")
        ):
            raise ValueError(f"{method} physical formal sample set is invalid")
        sampling_environment = provenance.get("runtime_environment")
        evaluator_environment = report.get("runtime_environment")
        if not isinstance(sampling_environment, dict) or not isinstance(
            evaluator_environment,
            dict,
        ):
            raise ValueError(f"{method} formal runtime environment is missing")
        sampling_sha = runtime_environment_sha256(sampling_environment)
        evaluator_sha = runtime_environment_sha256(evaluator_environment)
        if (
            provenance.get("runtime_environment_sha256") != sampling_sha
            or report.get("runtime_environment_sha256") != evaluator_sha
        ):
            raise ValueError(f"{method} formal runtime environment SHA256 differs")
        sampling_environments.add(sampling_sha)
        evaluator_environments.add(evaluator_sha)
        evidence[method] = {
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "sample_set_sha256": provenance["sample_set_sha256"],
            "sampling_runtime_environment_sha256": sampling_sha,
            "evaluator_runtime_environment_sha256": evaluator_sha,
            "sampling_protocol": contract,
        }
    if len(sampling_environments) != 1 or len(evaluator_environments) != 1:
        raise ValueError("matched formal methods used different runtime environments")
    return {
        "methods": evidence,
        "real_set": real_set_files,
    }


def runtime_and_visual_evidence(
    selection: dict[str, Any],
    visual: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    _raise_load_error(selection)
    _raise_load_error(visual)
    if (
        selection.get("schema_version") != 1
        or selection.get("status") != "selected"
        or selection.get("git_revision") != expected_revision
        or selection.get("selection_lock", {}).get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("formal sampling runtime selection identity differs")
    batch_size = int(selection.get("selected", {}).get("batch_size", -1))
    if batch_size < 1:
        raise ValueError("formal sampling runtime selection has no batch")
    for method, report in reports.items():
        if (
            int(
                report.get("sample_provenance", {})
                .get("sampling", {})
                .get("batch_size", -1)
            )
            != batch_size
        ):
            raise ValueError(f"{method} ignored the selected sampling batch")
    if (
        visual.get("status") != "completed"
        or visual.get("prefix_budgets") != [1, 2, 4, 8]
        or visual.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("formal visual audit identity differs")
    for method in ("cofitok", "dense_identity"):
        source = visual.get("sources", {}).get(method, {})
        provenance = reports[method].get("sample_provenance", {})
        if (
            source.get("checkpoint_sha256")
            != provenance.get("checkpoint_sha256")
            or source.get("sample_set_sha256")
            != provenance.get("sample_set_sha256")
        ):
            raise ValueError(f"visual audit {method} source identity differs")
    return {
        "sampling_batch_size": batch_size,
        "sampling_runtime_environment_sha256": selection.get(
            "runtime_environment_sha256"
        ),
        "visual_prefix_budgets": visual["prefix_budgets"],
    }


def strong_comparison_evidence(
    report: dict[str, Any],
    generation_reports: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
    official_related: dict[str, Any],
    official_related_sha256: str,
) -> dict[str, Any]:
    _raise_load_error(report)
    _raise_load_error(official_related)
    if report.get("source_profile") != "stability_full":
        raise ValueError("strong comparison source profile differs")
    return _comparison_evidence(
        report,
        generation_reports,
        training_reports,
        official_related,
        official_related_sha256,
        verify_comparison_source_reports(report),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the complete rollout-stable large-scale generation system."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/CoFiTok/checkpoints/generation",
    )
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument(
        "--expected-decision-source-revision",
        default="59db142fc45d69dc92bb0333be5ac2d0162d9dc4",
    )
    parser.add_argument("--expected-scaling-training-revision", required=True)
    parser.add_argument("--expected-scaling-training-branch", required=True)
    parser.add_argument("--expected-scaling-evaluation-revision", required=True)
    parser.add_argument("--expected-scaling-evaluation-branch", required=True)
    parser.add_argument("--expected-scaling-gate-sha256", required=True)
    parser.add_argument("--expected-full-training-revision", required=True)
    parser.add_argument("--expected-full-training-branch", required=True)
    parser.add_argument("--expected-full-evaluation-revision", required=True)
    parser.add_argument("--expected-full-evaluation-branch", required=True)
    parser.add_argument("--expected-final-gate-sha256", required=True)
    parser.add_argument("--expected-export-revision", required=True)
    parser.add_argument("--expected-export-branch", required=True)
    parser.add_argument("--real-dir", default=str(FORMAL_REAL_DIR))
    parser.add_argument("--official-related", default=str(OFFICIAL_RELATED))
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    paths = generation_stability_workspace_paths(output_root=output_root)
    expectations = {
        "decision_source_revision": _validate_revision(
            args.expected_decision_source_revision,
            name="decision source revision",
        ),
        "scaling_training_revision": _validate_revision(
            args.expected_scaling_training_revision,
            name="scaling training revision",
        ),
        "scaling_evaluation_revision": _validate_revision(
            args.expected_scaling_evaluation_revision,
            name="scaling evaluation revision",
        ),
        "full_training_revision": _validate_revision(
            args.expected_full_training_revision,
            name="full training revision",
        ),
        "full_evaluation_revision": _validate_revision(
            args.expected_full_evaluation_revision,
            name="full evaluation revision",
        ),
        "export_revision": _validate_revision(
            args.expected_export_revision,
            name="export revision",
        ),
        "decision_sha256": _validate_sha256(
            args.expected_decision_sha256,
            name="decision SHA256",
        ),
        "scaling_gate_sha256": _validate_sha256(
            args.expected_scaling_gate_sha256,
            name="scaling gate SHA256",
        ),
        "final_gate_sha256": _validate_sha256(
            args.expected_final_gate_sha256,
            name="final gate SHA256",
        ),
    }
    for name in (
        "expected_scaling_training_branch",
        "expected_scaling_evaluation_branch",
        "expected_full_training_branch",
        "expected_full_evaluation_branch",
        "expected_export_branch",
    ):
        if not getattr(args, name):
            raise ValueError(f"{name} must not be empty")

    scaling_cofitok = paths["STABILITY_SCALING_COFITOK_RUN"]
    scaling_dense = paths["STABILITY_SCALING_DENSE_RUN"]
    scaling_reports = paths["STABILITY_SCALING_REPORT_ROOT"]
    full_cofitok = paths["STABILITY_FULL_COFITOK_RUN"]
    full_dense = paths["STABILITY_FULL_DENSE_RUN"]
    full_reports = paths["STABILITY_FULL_REPORT_ROOT"]
    export_root = paths["STABILITY_EXPORT_ROOT"]

    decision = _read_optional(paths["STABILITY_DECISION"])
    scaling_monitor = _read_optional(paths["STABILITY_SCALING_MONITOR"])
    pair_summary = _read_optional(paths["STABILITY_SCALING_PAIR_SUMMARY"])
    scaling_gate = _read_optional(paths["STABILITY_SCALING_GATE"])
    scaling_training = {
        "cofitok": _read_optional(scaling_cofitok / "training_report.json"),
        "dense_identity": _read_optional(scaling_dense / "training_report.json"),
    }
    scaling_audits = {
        "cofitok": _read_optional(
            scaling_cofitok / "prepromotion_training_audit.json"
        ),
        "dense_identity": _read_optional(
            scaling_dense / "prepromotion_training_audit.json"
        ),
    }
    scaling_checkpoints = {
        "cofitok": _verify_checkpoint_file(
            scaling_cofitok / "checkpoint_step_00050000.pt"
        ),
        "dense_identity": _verify_checkpoint_file(
            scaling_dense / "checkpoint_step_00050000.pt"
        ),
    }

    full_monitor = _read_optional(paths["STABILITY_FULL_MONITOR"])
    full_runtime_selection = _read_optional(
        full_reports / "runtime_selection.json"
    )
    full_storage_capacity = _read_optional(
        full_reports / "storage_capacity.json"
    )
    full_training = {
        "cofitok": _read_optional(full_cofitok / "training_report.json"),
        "dense_identity": _read_optional(full_dense / "training_report.json"),
    }
    full_audits = {
        "cofitok": _read_optional(full_reports / "cofitok_training_audit.json"),
        "dense_identity": _read_optional(
            full_reports / "dense_training_audit.json"
        ),
    }
    full_checkpoints = {
        "cofitok": _verify_checkpoint_file(
            full_cofitok / "checkpoint_step_00300000.pt"
        ),
        "dense_identity": _verify_checkpoint_file(
            full_dense / "checkpoint_step_00300000.pt"
        ),
    }
    milestones = {
        step: _read_optional(
            full_reports / "milestones" / f"step_{step:08d}.json"
        )
        for step in MILESTONE_STEPS
    }
    milestone_checkpoints = {
        step: {
            "cofitok": _verify_checkpoint_file(
                full_cofitok / f"checkpoint_step_{step:08d}.pt"
            ),
            "dense_identity": _verify_checkpoint_file(
                full_dense / f"checkpoint_step_{step:08d}.pt"
            ),
        }
        for step in MILESTONE_STEPS
    }
    generation_reports = {
        "cofitok": _read_optional(
            full_cofitok
            / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_identity": _read_optional(
            full_dense
            / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
    }
    final_gate = _read_optional(paths["STABILITY_FULL_GATE"])
    sampling_selection = _read_optional(
        full_reports / "sampling_runtime_selection.json"
    )
    visual_audit = _read_optional(
        full_reports / "visual_audit/visual_audit_report.json"
    )
    comparison = _read_optional(
        full_reports / "comparison/large_scale_generation_comparison.json"
    )
    official_path = Path(args.official_related)
    if not official_path.is_absolute():
        official_path = project / official_path
    official_related = _read_optional(official_path)

    inference_exports = {
        "cofitok_export": _read_optional(
            full_reports / "exports/cofitok_export_report.json"
        ),
        "dense_identity_export": _read_optional(
            full_reports / "exports/dense_export_report.json"
        ),
        "cofitok_preflight": _read_optional(
            full_reports / "exports/cofitok_export_preflight.json"
        ),
        "dense_identity_preflight": _read_optional(
            full_reports / "exports/dense_export_preflight.json"
        ),
        "cofitok_smoke": _read_optional(
            full_reports / "exports/cofitok_export_inference_smoke.json"
        ),
        "dense_identity_smoke": _read_optional(
            full_reports / "exports/dense_export_inference_smoke.json"
        ),
    }
    inference_artifacts = {
        "cofitok": _verify_inference_artifact_file(
            paths["STABILITY_COFITOK_INFERENCE_ARTIFACT"]
        ),
        "dense_identity": _verify_inference_artifact_file(
            paths["STABILITY_DENSE_INFERENCE_ARTIFACT"]
        ),
    }
    inference_smoke_files = {
        "cofitok": _verify_inference_smoke_outputs(
            inference_exports["cofitok_smoke"],
            expected_root=export_root / "smoke/cofitok",
        ),
        "dense_identity": _verify_inference_smoke_outputs(
            inference_exports["dense_identity_smoke"],
            expected_root=export_root / "smoke/dense_identity",
        ),
    }

    scaling_training_present = list(scaling_training.values())
    full_training_present = list(full_training.values())
    milestone_present = list(milestones.values())
    generation_present = list(generation_reports.values())
    inference_present = list(inference_exports.values())

    sample_files = {
        "cofitok": _verify_formal_sample_files(
            generation_reports["cofitok"],
            expected_generated_dir=(
                full_cofitok / "samples_50k_ddim250_cfg15/prefix_8"
            ),
        ),
        "dense_identity": _verify_formal_sample_files(
            generation_reports["dense_identity"],
            expected_generated_dir=(
                full_dense / "samples_50k_ddim250_cfg15/prefix_1"
            ),
        ),
    }
    real_set_files = _verify_formal_real_set_files(
        generation_reports,
        expected_real_dir=Path(args.real_dir),
    )

    checks = [
        _check(
            "stability_5k_authorization",
            [decision],
            lambda: stability_decision_evidence(
                decision,
                decision_path=paths["STABILITY_DECISION"],
                expected_sha256=expectations["decision_sha256"],
                expected_source_revision=expectations[
                    "decision_source_revision"
                ],
            ),
        ),
        _check(
            "stability_50k_monitor",
            [scaling_monitor],
            lambda: monitor_evidence(
                scaling_monitor,
                expected_name="generation_stability_ema_teacher_matched_50k",
                expected_steps=50_000,
                expected_revision=expectations[
                    "scaling_training_revision"
                ],
                expected_branch=args.expected_scaling_training_branch,
            ),
        ),
        _check(
            "stability_50k_pair_summary",
            [pair_summary, *scaling_training_present],
            lambda: pair_summary_evidence(
                pair_summary,
                cofitok_training_path=scaling_cofitok
                / "training_report.json",
                dense_training_path=scaling_dense / "training_report.json",
                expected_revision=expectations[
                    "scaling_training_revision"
                ],
                expected_branch=args.expected_scaling_training_branch,
            ),
        ),
        _check(
            "stability_50k_checkpoint_integrity",
            [*scaling_training_present, *scaling_audits.values()],
            lambda: {
                "progress": progress_audit_evidence(
                    scaling_audits,
                    expected_steps=50_000,
                ),
                "checkpoints": checkpoint_evidence(
                    scaling_checkpoints,
                    scaling_training,
                    expected_step=50_000,
                    expected_revision=expectations[
                        "scaling_training_revision"
                    ],
                    expected_branch=args.expected_scaling_training_branch,
                ),
            },
        ),
        _check(
            "stability_scaling_gate",
            [scaling_gate],
            lambda: gate_evidence(
                scaling_gate,
                gate_path=paths["STABILITY_SCALING_GATE"],
                expected_sha256=expectations["scaling_gate_sha256"],
                stage="scaling",
                source_profile="stability_scaling",
                training_revision=expectations[
                    "scaling_training_revision"
                ],
                training_branch=args.expected_scaling_training_branch,
                evaluation_revision=expectations[
                    "scaling_evaluation_revision"
                ],
                evaluation_branch=args.expected_scaling_evaluation_branch,
            ),
        ),
        _check(
            "stability_full_monitor",
            [full_monitor],
            lambda: monitor_evidence(
                full_monitor,
                expected_name=(
                    "generation_stability_ema_teacher_full_matched_300k"
                ),
                expected_steps=300_000,
                expected_revision=expectations["full_training_revision"],
                expected_branch=args.expected_full_training_branch,
            ),
        ),
        _check(
            "stability_full_training_pair",
            [*full_training_present, *full_audits.values(), scaling_gate],
            lambda: {
                "pair": full_training_evidence(
                    full_training["cofitok"],
                    full_training["dense_identity"],
                    promotion_gate=scaling_gate,
                    expected_revision=expectations[
                        "full_training_revision"
                    ],
                    expected_branch=args.expected_full_training_branch,
                ),
                "progress": progress_audit_evidence(
                    full_audits,
                    expected_steps=300_000,
                    required_checkpoint_steps=MILESTONE_STEPS,
                ),
                "checkpoints": checkpoint_evidence(
                    full_checkpoints,
                    full_training,
                    expected_step=300_000,
                    expected_revision=expectations[
                        "full_training_revision"
                    ],
                    expected_branch=args.expected_full_training_branch,
                ),
            },
        ),
        _check(
            "stability_full_runtime_selection",
            [
                full_runtime_selection,
                *full_training_present,
            ],
            lambda: _runtime_selection_evidence(
                full_runtime_selection,
                full_training,
                expected_revision=expectations["full_training_revision"],
                expected_branch=args.expected_full_training_branch,
                expected_candidates=[
                    {"micro_batch_size": 1, "gradient_accumulation_steps": 64},
                    {"micro_batch_size": 2, "gradient_accumulation_steps": 32},
                    {"micro_batch_size": 4, "gradient_accumulation_steps": 16},
                    {"micro_batch_size": 8, "gradient_accumulation_steps": 8},
                    {"micro_batch_size": 16, "gradient_accumulation_steps": 4},
                ],
                expected_baseline={
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 64,
                },
                expected_run_dirs=[
                    full_cofitok.as_posix(),
                    full_dense.as_posix(),
                ],
                expected_benchmark_root=(
                    paths["STABILITY_FULL_ROOT"]
                    / "runtime_preflight/training"
                ).as_posix(),
            ),
        ),
        _check(
            "stability_full_storage_capacity",
            [full_storage_capacity],
            lambda: full_storage_capacity_evidence(
                full_storage_capacity,
                expected_revision=expectations["full_training_revision"],
                expected_branch=args.expected_full_training_branch,
                expected_path=output_root,
            ),
        ),
        _check(
            "stability_full_milestones",
            milestone_present,
            lambda: milestone_evidence(
                milestones,
                milestone_checkpoints,
            ),
        ),
        _check(
            "stability_full_formal_generation",
            [*generation_present, *full_training_present],
            lambda: formal_generation_evidence(
                generation_reports,
                full_training,
                sample_files,
                real_set_files,
                expected_revision=expectations[
                    "full_evaluation_revision"
                ],
                expected_branch=args.expected_full_evaluation_branch,
            ),
        ),
        _check(
            "stability_full_runtime_and_visual",
            [sampling_selection, visual_audit, *generation_present],
            lambda: runtime_and_visual_evidence(
                sampling_selection,
                visual_audit,
                generation_reports,
                expected_revision=expectations[
                    "full_evaluation_revision"
                ],
                expected_branch=args.expected_full_evaluation_branch,
            ),
        ),
        _check(
            "stability_final_gate",
            [final_gate, *generation_present],
            lambda: gate_evidence(
                final_gate,
                gate_path=paths["STABILITY_FULL_GATE"],
                expected_sha256=expectations["final_gate_sha256"],
                stage="full",
                source_profile="stability_full",
                training_revision=expectations["full_training_revision"],
                training_branch=args.expected_full_training_branch,
                evaluation_revision=expectations[
                    "full_evaluation_revision"
                ],
                evaluation_branch=args.expected_full_evaluation_branch,
            ),
        ),
        _check(
            "stability_strong_baseline_comparison",
            [
                comparison,
                official_related,
                final_gate,
                *generation_present,
                *full_training_present,
            ],
            lambda: strong_comparison_evidence(
                comparison,
                generation_reports,
                full_training,
                official_related,
                file_sha256(official_path),
            ),
        ),
        _check(
            "stability_release_authorized_inference",
            [
                *inference_present,
                final_gate,
                *generation_present,
                *full_training_present,
            ],
            lambda: _inference_export_evidence(
                inference_exports,
                inference_artifacts,
                inference_smoke_files,
                generation_reports,
                full_training,
                final_gate,
                expected_export_revision=expectations["export_revision"],
                expected_export_branch=args.expected_export_branch,
            ),
        ),
    ]
    audit = aggregate_checks(checks)
    audit["profile"] = "stability_generation_system_v1"
    audit["expectations"] = {
        **expectations,
        "scaling_training_branch": args.expected_scaling_training_branch,
        "scaling_evaluation_branch": args.expected_scaling_evaluation_branch,
        "full_training_branch": args.expected_full_training_branch,
        "full_evaluation_branch": args.expected_full_evaluation_branch,
        "export_branch": args.expected_export_branch,
    }
    write_json_report(Path(args.output), audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["complete"] and not args.allow_incomplete:
        raise SystemExit("stability generation completion audit did not pass")


if __name__ == "__main__":
    main()
