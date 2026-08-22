from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.data.provenance import validate_dataset_provenance
from cofitok.generation import (
    SAMPLING_MANIFEST_SCHEMA_VERSION,
    SAMPLING_REPORT_SCHEMA_VERSION,
    sampling_protocol_contract,
)
from cofitok.generation.artifact import (
    inference_export_manifest_path,
    verify_inference_artifact,
    verify_inference_export_manifest,
)
from cofitok.generation_authorization import validate_generation_gate_binding
from cofitok.generation_cost import training_cost_summary
from cofitok.gpu_contention import validate_gpu_contention_evidence
from cofitok.generation_gate import validate_generation_gate_authorization
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.generation_class_fidelity import (
    validate_class_fidelity_qualification,
)
from cofitok.generation_paths import (
    generation_deployment_attestation_paths,
    generation_workspace_paths,
)
from cofitok.image_integrity import (
    IMAGE_TREE_DIGEST_SCHEMA,
    image_tree_sha256,
    sample_set_sha256,
)
from cofitok.inference_replay import (
    INFERENCE_REPORT_SCHEMA_VERSION,
    validate_completed_inference_evidence,
)
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)
from cofitok.training.metrics import verify_metrics_resume_history_evidence

try:
    from scripts.validate_generation_training_pair import validate_training_pair
except ModuleNotFoundError:
    from validate_generation_training_pair import validate_training_pair

try:
    from scripts.select_generation_sampling_batch import select_sampling_batch
except ModuleNotFoundError:
    from select_generation_sampling_batch import select_sampling_batch

try:
    from scripts.write_generation_deployment_receipt import (
        bundle_prerequisites,
        pytest_junit_summary,
        validate_conflict_scan,
        validate_pytest_summary,
        validate_runbook_syntax,
    )
except ModuleNotFoundError:
    from write_generation_deployment_receipt import (
        bundle_prerequisites,
        pytest_junit_summary,
        validate_conflict_scan,
        validate_pytest_summary,
        validate_runbook_syntax,
    )

try:
    from scripts.build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )
except ModuleNotFoundError:
    from build_generation_milestone_report import (
        validate_milestone_report,
        verify_milestone_source_reports,
    )

try:
    from scripts.build_large_scale_generation_comparison import (
        COMPARISON_REPORT_SCHEMA_VERSION,
        MATCHED_TRAINING_BUDGET_BASIS,
        MATCHED_TRAINING_PROTOCOL_NOTE,
        training_budget_policy,
        verify_comparison_source_reports,
    )
except ModuleNotFoundError:
    from build_large_scale_generation_comparison import (
        COMPARISON_REPORT_SCHEMA_VERSION,
        MATCHED_TRAINING_BUDGET_BASIS,
        MATCHED_TRAINING_PROTOCOL_NOTE,
        training_budget_policy,
        verify_comparison_source_reports,
    )


PINNED_10PCT_REVISION = "781a01444fddbf0d48a427ba58bdeed50167b5be"
MILESTONE_STEPS = (50_000, 100_000, 200_000, 300_000)
GIB = 1024**3
KIB = 1024
FORMAL_REAL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FORMAL_IMAGENET256_REAL_DIR = Path(
    "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val"
)


def _check(
    name: str,
    payloads: list[dict[str, Any] | None],
    validator: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if any(payload is None for payload in payloads):
        return {"name": name, "status": "missing", "evidence": None}
    try:
        evidence = validator()
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return {
            "name": name,
            "status": "fail",
            "error": str(error),
            "evidence": None,
        }
    return {"name": name, "status": "pass", "evidence": evidence}


def _gate_evidence(
    gate: dict[str, Any],
    source_verification: dict[str, Any],
    *,
    stage: str,
    decision: str,
) -> dict[str, Any]:
    evidence = validate_generation_gate_authorization(gate, expected_stage=stage)
    if evidence["decision"] != decision:
        raise ValueError(f"{stage} gate did not authorize {decision}")
    if (
        source_verification.get("status") != "verified"
        or source_verification.get("stage") != stage
        or source_verification.get("source_reports") != gate.get("source_reports")
    ):
        raise ValueError(f"{stage} gate source-report verification differs")
    evidence["source_reports"] = source_verification["source_reports"]
    return evidence


def _training_audit_evidence(audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = {}
    for method, report in audits.items():
        if report.get("status") != "complete":
            raise ValueError(f"{method} full training audit is not complete")
        if report.get("issues"):
            raise ValueError(f"{method} full training audit contains issues")
        validation = report.get("validation", {})
        if validation.get("logging_complete") is not True:
            raise ValueError(f"{method} full validation logging is incomplete")
        checkpoint = report.get("checkpoint", {})
        if checkpoint.get("missing_required_steps"):
            raise ValueError(f"{method} protected checkpoints are missing")
        evidence[method] = {
            "last_step": int(report["last_step"]),
            "validation_event_count": int(validation["event_count"]),
            "checkpoint_steps": list(checkpoint["steps"]),
        }
    if any(row["last_step"] != 300_000 for row in evidence.values()):
        raise ValueError("full training audit did not reach exactly 300000 steps")
    return evidence


def _runtime_environment_evidence(
    training_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence = {}
    for method, report in training_reports.items():
        environment = report.get("runtime_environment")
        if not isinstance(environment, dict) or environment.get("schema_version") != 1:
            raise ValueError(f"{method} training lacks runtime environment provenance")
        environment_sha = runtime_environment_sha256(environment)
        if report.get("runtime_environment_sha256") != environment_sha:
            raise ValueError(f"{method} runtime environment SHA256 differs")
        if report.get("latest_checkpoint", {}).get(
            "runtime_environment_sha256"
        ) != environment_sha:
            raise ValueError(f"{method} checkpoint pointer lacks environment binding")
        python = environment.get("python", {})
        torch_environment = environment.get("torch", {})
        device = environment.get("device", {})
        packages = environment.get("packages", {})
        project_files = environment.get("project_files", {})
        if not all(
            isinstance(value, str) and value
            for value in (
                python.get("implementation"),
                python.get("version"),
                torch_environment.get("version"),
            )
        ):
            raise ValueError(f"{method} runtime language/framework versions are incomplete")
        if device.get("type") != "cuda" or not device.get("name"):
            raise ValueError(f"{method} full training did not bind a CUDA device")
        if not isinstance(torch_environment.get("cudnn_version"), int):
            raise ValueError(f"{method} runtime cuDNN provenance is incomplete")
        for package in ("numpy", "pillow", "torch", "torchvision", "tqdm"):
            if not packages.get(package):
                raise ValueError(f"{method} runtime package {package} is missing")
        for filename in ("pyproject.toml", "uv.lock"):
            identity = project_files.get(filename)
            if not isinstance(identity, dict) or len(str(identity.get("sha256", ""))) != 64:
                raise ValueError(f"{method} project environment file {filename} is unbound")
        evidence[method] = {
            "runtime_environment_sha256": environment_sha,
            "python_version": python["version"],
            "torch_version": torch_environment["version"],
            "cuda_version": torch_environment.get("cuda_version"),
            "cudnn_version": torch_environment["cudnn_version"],
            "device_name": device["name"],
        }
    hashes = {row["runtime_environment_sha256"] for row in evidence.values()}
    if len(hashes) != 1:
        raise ValueError("full matched methods used different runtime environments")
    return evidence


def _checkpoint_code_provenance_evidence(
    training_reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    evidence = {}
    for method, report in training_reports.items():
        git = report.get("git", {})
        latest = report.get("latest_checkpoint", {})
        if (
            git.get("revision") != expected_revision
            or git.get("branch") != "scale/generative-system"
            or git.get("dirty") is not False
        ):
            raise ValueError(f"{method} training Git provenance is invalid")
        if (
            latest.get("git_revision") != expected_revision
            or latest.get("git_branch") != "scale/generative-system"
            or latest.get("git_dirty") is not False
        ):
            raise ValueError(f"{method} checkpoint pointer Git binding is invalid")
        evidence[method] = {
            "revision": expected_revision,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
            "checkpoint": latest.get("checkpoint"),
        }
    return evidence


def _full_checkpoint_file_evidence(
    files: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence = {}
    for method in ("cofitok", "dense_identity"):
        verified = files[method]
        latest = training_reports[method].get("latest_checkpoint", {})
        if verified.get("status") != "verified":
            raise ValueError(
                f"{method} full checkpoint file is invalid: "
                f"{verified.get('error', 'verification failed')}"
            )
        if int(verified.get("step", -1)) != 300_000:
            raise ValueError(f"{method} verified checkpoint is not step 300K")
        if verified.get("checkpoint") != latest.get("checkpoint"):
            raise ValueError(f"{method} verified checkpoint filename differs")
        if verified.get("checkpoint_sha256") != latest.get("checkpoint_sha256"):
            raise ValueError(f"{method} verified checkpoint SHA256 differs")
        if int(verified.get("checkpoint_bytes", -1)) != int(
            latest.get("checkpoint_bytes", -2)
        ):
            raise ValueError(f"{method} verified checkpoint byte count differs")
        if Path(str(verified.get("integrity_manifest", ""))).name != Path(
            str(latest.get("integrity_manifest", ""))
        ).name:
            raise ValueError(f"{method} verified checkpoint integrity path differs")
        for key in (
            "runtime_environment_sha256",
            "git_revision",
            "git_branch",
            "git_dirty",
            "authorization_stage",
            "authorization_decision",
            "authorization_gate_bytes",
            "authorization_gate_sha256",
            "authorization_gate_identity_sha256",
        ):
            if verified.get(key) != latest.get(key):
                raise ValueError(
                    f"{method} verified checkpoint {key} binding differs"
                )
        if int(verified.get("checkpoint_format_version", -1)) != 1:
            raise ValueError(f"{method} verified checkpoint format differs")
        evidence[method] = {
            "path": verified["path"],
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "checkpoint_bytes": int(verified["checkpoint_bytes"]),
            "integrity_manifest": verified["integrity_manifest"],
        }
    return evidence


def _runtime_selection_evidence(
    selection: dict[str, Any],
    training_reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
    expected_branch: str = "scale/generative-system",
    expected_candidates: list[dict[str, int]] | None = None,
    expected_baseline: dict[str, int] | None = None,
    expected_run_dirs: list[str] | None = None,
    expected_benchmark_root: str | None = None,
) -> dict[str, Any]:
    schema_version = int(selection.get("schema_version", -1))
    if schema_version not in {2, 3}:
        raise ValueError("full training runtime selection schema is unsupported")
    if selection.get("status") != "selected":
        raise ValueError("full training runtime selection is incomplete")
    if selection.get("git_revision") != expected_revision:
        raise ValueError("runtime selection revision differs from full training")
    selection_lock = selection.get("selection_lock")
    expected_candidates = expected_candidates or [
        {"micro_batch_size": 16, "gradient_accumulation_steps": 4},
        {"micro_batch_size": 32, "gradient_accumulation_steps": 2},
        {"micro_batch_size": 64, "gradient_accumulation_steps": 1},
    ]
    expected_baseline = expected_baseline or {
        "micro_batch_size": 16,
        "gradient_accumulation_steps": 4,
    }
    expected_run_dirs = expected_run_dirs or [
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_full_cofitok_k8_300k",
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/imagenet256_full_dense_300k",
    ]
    expected_benchmark_root = expected_benchmark_root or (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "runtime_preflight/full_imagenet256_300k"
    )
    if not isinstance(selection_lock, dict):
        raise ValueError("full runtime selection lock is missing")
    if (
        selection_lock.get("schema_version") != (2 if schema_version == 3 else 1)
        or selection_lock.get("mode") != "freeze_on_training_state"
        or selection_lock.get("training_run_dirs") != expected_run_dirs
        or selection_lock.get("candidates") != expected_candidates
        or selection_lock.get("expected_effective_batch_size") != 64
        or selection_lock.get("benchmark_steps") != 8
        or selection_lock.get("warmup_steps") != 2
        or selection_lock.get("max_memory_fraction") != 0.9
        or selection_lock.get("training_target_steps") != 300_000
        or selection_lock.get("benchmark_root") != expected_benchmark_root
    ):
        raise ValueError("full runtime selection lock contract differs")
    if schema_version == 3:
        if selection_lock.get("baseline_candidate") != expected_baseline:
            raise ValueError("full runtime selection baseline lock differs")
        baseline = selection.get("baseline", {})
        if (
            int(baseline.get("micro_batch_size", -1))
            != expected_baseline["micro_batch_size"]
            or int(baseline.get("gradient_accumulation_steps", -1))
            != expected_baseline["gradient_accumulation_steps"]
            or int(baseline.get("effective_batch_size", -1)) != 64
        ):
            raise ValueError("full runtime selection baseline evidence differs")
    lock_git = selection_lock.get("git", {})
    if lock_git != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("full runtime selection lock Git state differs")
    config_sha256 = selection.get("config_sha256")
    if (
        not isinstance(config_sha256, dict)
        or selection_lock.get("config_sha256") != config_sha256
        or set(config_sha256) != {"cofitok", "dense_identity"}
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in config_sha256.values()
        )
    ):
        raise ValueError("full runtime selection config identity differs")
    if selection.get("benchmark_root") != expected_benchmark_root:
        raise ValueError("full runtime selection benchmark root differs")
    selected = selection.get("selected", {})
    micro_batch = int(selected.get("micro_batch_size", -1))
    accumulation = int(selected.get("gradient_accumulation_steps", -1))
    effective_batch = int(selected.get("effective_batch_size", -1))
    if micro_batch < 1 or accumulation < 1 or effective_batch != 64:
        raise ValueError("selected full runtime is invalid")
    if micro_batch * accumulation != effective_batch:
        raise ValueError("selected full runtime changes effective batch")
    candidates = selection.get("candidates", [])
    observed_candidates = [
        {
            "micro_batch_size": int(row.get("micro_batch_size", -1)),
            "gradient_accumulation_steps": int(
                row.get("gradient_accumulation_steps", -1)
            ),
        }
        for row in candidates
    ]
    if observed_candidates != expected_candidates:
        raise ValueError("full runtime benchmark candidate set differs")
    selected_candidates = [
        row
        for row in candidates
        if int(row.get("micro_batch_size", -1)) == micro_batch
        and int(row.get("gradient_accumulation_steps", -1)) == accumulation
    ]
    if (
        len(selected_candidates) != 1
        or selected_candidates[0].get("eligible") is not True
    ):
        raise ValueError("selected full runtime benchmark evidence is invalid")
    baseline_candidates = [
        row
        for row in candidates
        if int(row.get("micro_batch_size", -1))
        == expected_baseline["micro_batch_size"]
        and int(row.get("gradient_accumulation_steps", -1))
        == expected_baseline["gradient_accumulation_steps"]
    ]
    if (
        len(baseline_candidates) != 1
        or baseline_candidates[0].get("eligible") is not True
    ):
        raise ValueError("full runtime baseline benchmark evidence is invalid")
    selected_environment_sha = str(selection.get("runtime_environment_sha256", ""))
    if len(selected_environment_sha) != 64:
        raise ValueError("selected full runtime environment SHA256 is malformed")
    selected_dataset_sha = str(selection.get("dataset_identity_sha256", ""))
    if len(selected_dataset_sha) != 64:
        raise ValueError("selected full runtime dataset identity SHA256 is malformed")
    for candidate in candidates:
        candidate_micro_batch = int(candidate["micro_batch_size"])
        candidate_accumulation = int(candidate["gradient_accumulation_steps"])
        if (
            int(candidate.get("effective_batch_size", -1)) != 64
            or candidate_micro_batch * candidate_accumulation != 64
        ):
            raise ValueError("training benchmark candidate changes effective batch")
        completed_methods = 0
        for method in ("cofitok", "dense_identity"):
            benchmark = candidate.get("methods", {}).get(method, {})
            if benchmark.get("status") != "completed":
                continue
            completed_methods += 1
            environment = benchmark.get("runtime_environment")
            git = benchmark.get("git", {})
            provenance = benchmark.get("dataset_provenance")
            expected_config = copy.deepcopy(training_reports[method].get("config", {}))
            expected_config.setdefault("data", {})["batch_size"] = candidate_micro_batch
            expected_config.setdefault("optimization", {})[
                "gradient_accumulation_steps"
            ] = candidate_accumulation
            if benchmark.get("config") != expected_config:
                raise ValueError(f"{method} training benchmark config differs")
            if (
                int(benchmark.get("effective_batch_size", -1)) != 64
                or int(benchmark.get("benchmark_steps", -1)) != 8
                or int(benchmark.get("warmup_steps", -1)) != 2
                or benchmark.get("checkpoint_written") is not False
            ):
                raise ValueError(f"{method} training benchmark contract differs")
            if (
                not isinstance(environment, dict)
                or runtime_environment_sha256(environment)
                != selected_environment_sha
                or benchmark.get("runtime_environment_sha256")
                != selected_environment_sha
            ):
                raise ValueError(
                    f"{method} training benchmark environment differs"
                )
            if not isinstance(provenance, dict) or validate_dataset_provenance(
                provenance,
                expected_dataset="imagenet_256",
            )["identity_sha256"] != selected_dataset_sha:
                raise ValueError(f"{method} training benchmark dataset differs")
            if (
                git.get("revision") != expected_revision
                or git.get("branch") != expected_branch
                or git.get("dirty") is not False
            ):
                raise ValueError(f"{method} training benchmark Git state differs")
        if (
            completed_methods == 2
            and candidate.get("runtime_environment_sha256")
            != selected_environment_sha
        ):
            raise ValueError("training benchmark candidate environment differs")
        if (
            completed_methods == 2
            and candidate.get("dataset_identity_sha256") != selected_dataset_sha
        ):
            raise ValueError("training benchmark candidate dataset differs")
    for method, report in training_reports.items():
        if int(report.get("target_steps", -1)) != 300_000:
            raise ValueError(f"{method} training horizon differs from runtime selection")
        config = report.get("config", {})
        if int(config.get("data", {}).get("batch_size", -1)) != micro_batch:
            raise ValueError(f"{method} training did not use selected microbatch")
        if (
            int(
                config.get("optimization", {}).get(
                    "gradient_accumulation_steps", -1
                )
            )
            != accumulation
        ):
            raise ValueError(f"{method} training did not use selected accumulation")
        if report.get("runtime_environment_sha256") != selected_environment_sha:
            raise ValueError(f"{method} training environment differs from runtime selection")
        provenance = report.get("dataset_provenance")
        if not isinstance(provenance, dict) or validate_dataset_provenance(
            provenance,
            expected_dataset="imagenet_256",
        )["identity_sha256"] != selected_dataset_sha:
            raise ValueError(f"{method} training dataset differs from runtime selection")
    speedup_key = (
        "estimated_speedup_over_baseline"
        if schema_version == 3
        else "estimated_speedup_over_16x4"
    )
    speedup = float(selected.get(speedup_key, math.nan))
    if not math.isfinite(speedup) or speedup <= 0.0:
        raise ValueError("full runtime selection speedup is invalid")
    selected_score = float(
        selected_candidates[0].get("selection_score_seconds", math.nan)
    )
    baseline_score = float(
        baseline_candidates[0].get("selection_score_seconds", math.nan)
    )
    if (
        not math.isfinite(selected_score)
        or selected_score <= 0.0
        or not math.isfinite(baseline_score)
        or baseline_score <= 0.0
        or not math.isclose(speedup, baseline_score / selected_score)
    ):
        raise ValueError("full runtime selection score arithmetic differs")
    if schema_version == 3:
        baseline = selection["baseline"]
        if (
            float(selected.get("selection_score_seconds", math.nan))
            != selected_score
            or float(baseline.get("selection_score_seconds", math.nan))
            != baseline_score
        ):
            raise ValueError("full runtime baseline score differs")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "baseline": dict(expected_baseline),
        "estimated_speedup_over_baseline": speedup,
        "runtime_environment_sha256": selected_environment_sha,
        "dataset_identity_sha256": selected_dataset_sha,
    }


def _milestone_evidence(
    milestones: dict[int, dict[str, Any]],
    source_verifications: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    evidence = {}
    warnings = []
    for step in MILESTONE_STEPS:
        report = milestones[step]
        step_evidence, step_warnings = validate_milestone_report(
            report,
            expected_step=step,
            source_verification=source_verifications[step],
        )
        warnings.extend(step_warnings)
        evidence[str(step)] = step_evidence
    return evidence, warnings


def _verify_formal_sample_files(
    report: dict[str, Any] | None,
    *,
    expected_generated_dir: str | Path,
    expected_count: int = 50_000,
) -> dict[str, Any] | None:
    if report is None:
        return None
    declared_generated_dir = Path(expected_generated_dir)
    generated_dir = declared_generated_dir.resolve()
    try:
        if expected_count < 1:
            raise ValueError("formal sample count must be positive")
        if declared_generated_dir.is_symlink() or not generated_dir.is_dir():
            raise ValueError(f"formal sample directory is invalid: {generated_dir}")
        if Path(str(report.get("paths", {}).get("generated_dir", ""))).resolve() != generated_dir:
            raise ValueError("formal metrics generated directory differs")
        entries = list(generated_dir.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError("formal sample directory contains a non-file or symlink")
        expected_names = {f"{index:06d}.png" for index in range(expected_count)}
        if {entry.name for entry in entries} != expected_names:
            raise ValueError("formal sample directory numbered PNG set differs")
        sample_paths = sorted(entries, key=lambda path: path.name)
        actual_sha = sample_set_sha256(sample_paths)
        provenance = report.get("sample_provenance", {})
        if actual_sha != provenance.get("sample_set_sha256"):
            raise ValueError("formal sample-set SHA256 differs from metrics provenance")

        sampling_root = generated_dir.parent
        sampling_report_path = sampling_root / "sampling_report.json"
        provenance_report = Path(str(provenance.get("report", ""))).resolve()
        if provenance_report != sampling_report_path.resolve() or not sampling_report_path.is_file():
            raise ValueError("formal sampling report is missing or outside the sample run")
        sampling_report = json.loads(sampling_report_path.read_text(encoding="utf-8"))
        sampling = provenance.get("sampling", {})
        budgets = sampling.get("prefix_budgets", [])
        if not isinstance(budgets, list) or len(budgets) != 1:
            raise ValueError("formal sampling report must select exactly one prefix budget")
        budget = str(int(budgets[0]))
        if (
            int(sampling_report.get("schema_version", -1))
            != SAMPLING_REPORT_SCHEMA_VERSION
            or sampling_report.get("status") != "completed"
            or sampling_report.get("sampling") != sampling
            or Path(str(sampling_report.get("output_dirs", {}).get(budget, ""))).resolve()
            != generated_dir
            or sampling_report.get("sample_sets", {}).get(budget)
            != {"count": expected_count, "sha256": actual_sha}
        ):
            raise ValueError("formal sampling report differs from generated files")
        for field in (
            "git",
            "runtime_environment",
            "runtime_environment_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_integrity_manifest",
            "checkpoint_step",
            "weights",
        ):
            if sampling_report.get(field) != provenance.get(field):
                raise ValueError(f"formal sampling report {field} differs from metrics provenance")

        manifest_path = sampling_root / "sampling_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("formal immutable sampling manifest is missing")
        manifest_sha = file_sha256(manifest_path)
        if sampling_report.get("sampling_manifest_sha256") != manifest_sha:
            raise ValueError("formal sampling manifest SHA256 differs")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != SAMPLING_MANIFEST_SCHEMA_VERSION:
            raise ValueError("formal sampling manifest schema is unsupported")
        for field in (
            "git",
            "runtime_environment",
            "runtime_environment_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_integrity_manifest",
            "checkpoint_step",
            "weights",
            "sampling",
            "output_dirs",
        ):
            if manifest.get(field) != sampling_report.get(field):
                raise ValueError(f"formal sampling manifest {field} differs from report")

        progress_path = sampling_root / "sampling_progress.json"
        if (
            Path(str(sampling_report.get("sampling_progress", ""))).resolve()
            != progress_path.resolve()
            or Path(str(provenance.get("sampling_progress", {}).get("report", ""))).resolve()
            != progress_path.resolve()
            or not progress_path.is_file()
        ):
            raise ValueError("formal sampling progress is missing or outside the sample run")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            int(progress.get("schema_version", -1)) != 1
            or progress.get("status") != "completed"
            or progress.get("sampling_manifest_sha256") != manifest_sha
            or int(progress.get("completed_samples", -1)) != expected_count
            or progress.get("sample_sets") != sampling_report.get("sample_sets")
        ):
            raise ValueError("formal sampling progress differs from completed sample set")
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "generated_dir": generated_dir.as_posix(),
            "error": str(error),
        }
    return {
        "status": "verified",
        "generated_dir": generated_dir.as_posix(),
        "sample_count": expected_count,
        "sample_set_sha256": actual_sha,
        "sampling_report": sampling_report_path.resolve().as_posix(),
        "sampling_report_sha256": file_sha256(sampling_report_path),
        "sampling_manifest": manifest_path.resolve().as_posix(),
        "sampling_manifest_sha256": manifest_sha,
        "sampling_progress": progress_path.resolve().as_posix(),
        "sampling_progress_sha256": file_sha256(progress_path),
    }


def _verify_formal_real_set_files(
    reports: dict[str, dict[str, Any] | None],
    *,
    expected_real_dir: str | Path,
    expected_count: int = 50_000,
) -> dict[str, Any] | None:
    if any(report is None for report in reports.values()):
        return None
    declared_real_dir = Path(expected_real_dir)
    real_dir = declared_real_dir.resolve()

    def collect_image_tree() -> tuple[list[Path], tuple[str, ...]]:
        entries = list(real_dir.rglob("*"))
        symlinks = [entry for entry in entries if entry.is_symlink()]
        if symlinks:
            raise ValueError(f"formal real set contains a symlink: {symlinks[0]}")
        non_files = [
            entry for entry in entries if not entry.is_dir() and not entry.is_file()
        ]
        if non_files:
            raise ValueError(f"formal real set contains a non-file entry: {non_files[0]}")
        unsupported = [
            entry
            for entry in entries
            if entry.is_file()
            and entry.suffix.lower() not in FORMAL_REAL_IMAGE_EXTENSIONS
        ]
        if unsupported:
            raise ValueError(
                f"formal real set contains an unsupported file: {unsupported[0]}"
            )
        images = sorted(
            (
                entry
                for entry in entries
                if entry.is_file()
                and entry.suffix.lower() in FORMAL_REAL_IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(real_dir).as_posix(),
        )
        relative_paths = tuple(
            path.relative_to(real_dir).as_posix() for path in images
        )
        return images, relative_paths

    try:
        if expected_count < 1:
            raise ValueError("formal real-set image count must be positive")
        if declared_real_dir.is_symlink() or not real_dir.is_dir():
            raise ValueError(f"formal real-set directory is invalid: {real_dir}")
        images, relative_paths = collect_image_tree()
        if len(images) != expected_count:
            raise ValueError(
                f"formal real-set image count is {len(images)}, expected {expected_count}"
            )
        actual_sha = image_tree_sha256(images, root=real_dir)
        _, relative_paths_after = collect_image_tree()
        if relative_paths_after != relative_paths:
            raise RuntimeError("formal real-set membership changed while hashing")
        for method, report in reports.items():
            if report is None:
                raise ValueError(f"{method} formal real-set report is missing")
            reported_real_set = report.get("real_set", {})
            if (
                Path(str(report.get("paths", {}).get("real_dir", ""))).resolve()
                != real_dir
                or Path(str(reported_real_set.get("root", ""))).resolve()
                != real_dir
                or reported_real_set.get("root") != real_dir.as_posix()
                or reported_real_set.get("digest_schema")
                != IMAGE_TREE_DIGEST_SCHEMA
                or int(reported_real_set.get("image_count", -1)) != expected_count
                or int(report.get("counts", {}).get("real_image_count", -1))
                != expected_count
                or reported_real_set.get("sha256") != actual_sha
            ):
                raise ValueError(
                    f"{method} formal real-set report differs from physical image tree"
                )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "real_dir": real_dir.as_posix(),
            "error": str(error),
        }
    return {
        "status": "verified",
        "real_dir": real_dir.as_posix(),
        "image_count": expected_count,
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "real_set_sha256": actual_sha,
    }


def _generation_evidence(
    reports: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
    sample_files: dict[str, dict[str, Any]],
    real_set_files: dict[str, Any],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    if real_set_files.get("status") != "verified":
        raise ValueError(
            "formal real-set files are invalid: "
            f"{real_set_files.get('error', 'verification failed')}"
        )
    evidence = {}
    sampling_environment_shas = set()
    evaluator_environment_shas = set()
    real_set_identities = set()
    for method, report in reports.items():
        if report.get("status") != "completed":
            raise ValueError(f"{method} formal generation metrics are incomplete")
        evaluator_git = report.get("git", {})
        if (
            evaluator_git.get("revision") != expected_revision
            or evaluator_git.get("branch") != "scale/generative-system"
            or evaluator_git.get("tracked_dirty") is not False
        ):
            raise ValueError(f"{method} formal evaluator code provenance is invalid")
        evaluator_environment = report.get("runtime_environment")
        if not isinstance(evaluator_environment, dict):
            raise ValueError(f"{method} formal evaluator environment is missing")
        evaluator_environment_sha = runtime_environment_sha256(evaluator_environment)
        if report.get("runtime_environment_sha256") != evaluator_environment_sha:
            raise ValueError(f"{method} formal evaluator environment SHA256 differs")
        evaluator_environment_shas.add(evaluator_environment_sha)
        if int(report.get("counts", {}).get("generated_image_count", -1)) != 50_000:
            raise ValueError(f"{method} formal generated sample count is not 50000")
        real_set = report.get("real_set", {})
        real_set_sha = str(real_set.get("sha256", ""))
        real_count = int(report.get("counts", {}).get("real_image_count", -1))
        real_root = report.get("paths", {}).get("real_dir")
        real_cache_name = str(report.get("parameters", {}).get("real_cache_name", ""))
        if (
            real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
            or len(real_set_sha) != 64
            or any(character not in "0123456789abcdef" for character in real_set_sha)
            or int(real_set.get("image_count", -1)) != real_count
            or real_count != 50_000
            or real_set.get("root") != real_root
            or not real_cache_name.endswith(f"__cofitok_{real_set_sha[:16]}")
            or real_set_files.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
            or real_set_files.get("real_set_sha256") != real_set_sha
            or real_set_files.get("real_dir") != real_root
            or int(real_set_files.get("image_count", -1)) != real_count
        ):
            raise ValueError(f"{method} formal real-set provenance is invalid")
        real_set_identities.add(
            (IMAGE_TREE_DIGEST_SCHEMA, real_set_sha, real_root, real_count, real_cache_name)
        )
        provenance = report.get("sample_provenance", {})
        sampling = provenance.get("sampling", {})
        sampling_contract = sampling_protocol_contract(
            sampling,
            stage="full",
            expected_num_train_timesteps=int(
                training_reports[method]["config"]["diffusion"][
                    "num_train_timesteps"
                ]
            ),
        )
        if sampling_contract["valid"] is not True:
            raise ValueError(
                f"{method} formal sampling protocol is invalid: "
                + ", ".join(sampling_contract["issues"])
            )
        git = provenance.get("git", {})
        progress = provenance.get("sampling_progress", {})
        inference_api = sampling.get("inference_api", {})
        sampling_environment = provenance.get("runtime_environment")
        if not isinstance(sampling_environment, dict):
            raise ValueError(f"{method} formal sampling environment is missing")
        sampling_environment_sha = runtime_environment_sha256(sampling_environment)
        if (
            provenance.get("runtime_environment_sha256")
            != sampling_environment_sha
        ):
            raise ValueError(f"{method} formal sampling environment SHA256 differs")
        sampling_environment_shas.add(sampling_environment_sha)
        if int(provenance.get("checkpoint_step", -1)) != 300_000:
            raise ValueError(f"{method} formal samples do not use the 300K checkpoint")
        if provenance.get("weights") != "ema":
            raise ValueError(f"{method} formal samples do not use EMA weights")
        if (
            git.get("revision") != expected_revision
            or git.get("branch") != "scale/generative-system"
            or git.get("tracked_dirty") is not False
        ):
            raise ValueError(f"{method} formal sampling code provenance is invalid")
        if len(str(provenance.get("checkpoint_sha256", ""))) != 64:
            raise ValueError(f"{method} checkpoint SHA256 is malformed")
        if len(str(provenance.get("sample_set_sha256", ""))) != 64:
            raise ValueError(f"{method} sample-set SHA256 is malformed")
        if not str(provenance.get("checkpoint_integrity_manifest", "")).endswith(
            ".pt.integrity.json"
        ):
            raise ValueError(f"{method} checkpoint integrity manifest is missing")
        latest = training_reports[method].get("latest_checkpoint", {})
        if latest.get("checkpoint_sha256") != provenance["checkpoint_sha256"]:
            raise ValueError(f"{method} training and sampling checkpoint SHA256 differ")
        if Path(str(latest.get("integrity_manifest", ""))).name != Path(
            str(provenance["checkpoint_integrity_manifest"])
        ).name:
            raise ValueError(f"{method} training and sampling integrity manifests differ")
        if progress.get("status") != "completed":
            raise ValueError(f"{method} formal sampling progress is incomplete")
        if inference_api != {
            "name": "cofitok.generation.GenerationSession",
            "version": 1,
        }:
            raise ValueError(f"{method} formal sampling bypassed the stable inference API")
        if int(progress.get("completed_samples", -1)) != 50_000:
            raise ValueError(f"{method} formal sampling progress count mismatch")
        elapsed = float(progress.get("cumulative_elapsed_seconds", math.nan))
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError(f"{method} formal sampling elapsed time is invalid")
        verified_samples = sample_files[method]
        if (
            verified_samples.get("status") != "verified"
            or int(verified_samples.get("sample_count", -1)) != 50_000
            or verified_samples.get("sample_set_sha256")
            != provenance.get("sample_set_sha256")
            or verified_samples.get("generated_dir")
            != report.get("paths", {}).get("generated_dir")
        ):
            raise ValueError(
                f"{method} formal sample files are invalid: "
                f"{verified_samples.get('error', 'verification failed')}"
            )
        evidence[method] = {
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "sample_set_sha256": provenance["sample_set_sha256"],
            "sampling_elapsed_seconds": elapsed,
            "runtime_environment_sha256": sampling_environment_sha,
            "evaluator_runtime_environment_sha256": evaluator_environment_sha,
            "real_set_sha256": real_set_sha,
            "real_set_physical_verification": {
                "digest_schema": real_set_files["digest_schema"],
                "image_count": real_set_files["image_count"],
                "real_dir": real_set_files["real_dir"],
                "sha256": real_set_files["real_set_sha256"],
            },
            "sampling_protocol_contract": sampling_contract,
            "sampling_report_sha256": verified_samples["sampling_report_sha256"],
            "sampling_manifest_sha256": verified_samples[
                "sampling_manifest_sha256"
            ],
            "sampling_progress_sha256": verified_samples[
                "sampling_progress_sha256"
            ],
        }
    if len(sampling_environment_shas) != 1:
        raise ValueError("formal matched methods used different sampling environments")
    if len(evaluator_environment_shas) != 1:
        raise ValueError("formal matched methods used different evaluator environments")
    if len(real_set_identities) != 1:
        raise ValueError("formal matched methods used different real sets")
    return evidence


def _sampling_runtime_selection_evidence(
    selection: dict[str, Any],
    generation_reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    if selection.get("schema_version") != 1:
        raise ValueError("formal sampling runtime selection schema is unsupported")
    if selection.get("status") != "selected":
        raise ValueError("formal sampling runtime selection is incomplete")
    if selection.get("git_revision") != expected_revision:
        raise ValueError("sampling runtime selection revision differs from full training")
    policy = selection.get("policy", {})
    if policy.get("shared_candidate_required") is not True:
        raise ValueError("sampling runtime selection is not shared")
    if policy.get("batch_size_invariant_random_stream_required") is not True:
        raise ValueError("sampling selection omits batch-invariant random streams")
    selection_lock = selection.get("selection_lock")
    expected_candidates = [16, 32, 64, 128]
    expected_benchmark_root = (
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "runtime_preflight/imagenet256_full_50k_sampling"
    )
    expected_output_dirs = [
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "imagenet256_full_cofitok_k8_300k/samples_50k_ddim250_cfg15",
        "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
        "imagenet256_full_dense_300k/samples_50k_ddim250_cfg15",
    ]
    expected_protocol = {
        "cofitok_prefix_budget": 8,
        "dense_prefix_budget": 1,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "weights": "ema",
        "precision": "bf16",
        "warmup_forwards": 2,
        "measured_forwards": 5,
    }
    if not isinstance(selection_lock, dict) or (
        selection_lock.get("schema_version") != 1
        or selection_lock.get("mode") != "freeze_on_sampling_state"
        or selection_lock.get("sampling_output_dirs") != expected_output_dirs
        or selection_lock.get("candidates") != expected_candidates
        or selection_lock.get("baseline_batch_size") != 32
        or selection_lock.get("max_memory_fraction") != 0.9
        or selection_lock.get("protocol") != expected_protocol
        or selection_lock.get("benchmark_root") != expected_benchmark_root
    ):
        raise ValueError("formal sampling runtime selection lock differs")
    if selection_lock.get("git") != {
        "revision": expected_revision,
        "branch": "scale/generative-system",
        "tracked_dirty": False,
    }:
        raise ValueError("formal sampling selection lock Git state differs")
    if selection.get("benchmark_root") != expected_benchmark_root:
        raise ValueError("formal sampling selection benchmark root differs")
    identities = selection.get("checkpoints", {})
    if selection_lock.get("checkpoints") != identities:
        raise ValueError("formal sampling selection lock checkpoints differ")
    candidates = selection.get("candidates", [])
    if [int(row.get("batch_size", -1)) for row in candidates] != expected_candidates:
        raise ValueError("formal sampling candidate set differs")
    recomputed = select_sampling_batch(
        copy.deepcopy(candidates),
        baseline_batch_size=32,
        max_memory_fraction=0.9,
    )
    for key in (
        "schema_version",
        "status",
        "policy",
        "selected",
        "runtime_environment_sha256",
        "candidates",
    ):
        if selection.get(key) != recomputed.get(key):
            raise ValueError(f"formal sampling selection {key} is not reproducible")
    batch_size = int(selection.get("selected", {}).get("batch_size", -1))
    if batch_size < 1:
        raise ValueError("selected formal sampling batch is invalid")
    selected_candidates = [
        row
        for row in candidates
        if int(row.get("batch_size", -1)) == batch_size
    ]
    if (
        len(selected_candidates) != 1
        or selected_candidates[0].get("eligible") is not True
    ):
        raise ValueError("selected formal sampling candidate evidence is invalid")
    selected_methods = selected_candidates[0].get("methods", {})
    for method in ("cofitok", "dense_identity"):
        selected_git = selected_methods.get(method, {}).get("git", {})
        if (
            selected_git.get("revision") != expected_revision
            or selected_git.get("tracked_dirty") is not False
        ):
            raise ValueError(f"{method} selected sampling preflight revision differs")
    selected_environment_sha = selection.get("runtime_environment_sha256")
    if len(str(selected_environment_sha)) != 64:
        raise ValueError("sampling selection runtime environment SHA256 is malformed")
    observed_preflight_environment_shas = set()
    for candidate in candidates:
        candidate_batch = int(candidate["batch_size"])
        for method, prefix_budget in (("cofitok", 8), ("dense_identity", 1)):
            preflight = candidate.get("methods", {}).get(method, {})
            if preflight.get("status") != "passed":
                continue
            preflight_environment = preflight.get("runtime_environment")
            if not isinstance(preflight_environment, dict):
                raise ValueError(f"{method} sampling preflight environment is missing")
            preflight_environment_sha = runtime_environment_sha256(
                preflight_environment
            )
            observed_preflight_environment_shas.add(preflight_environment_sha)
            request = preflight.get("request", {})
            git = preflight.get("git", {})
            if (
                preflight.get("runtime_environment_sha256")
                != preflight_environment_sha
                or git.get("revision") != expected_revision
                or git.get("branch") != "scale/generative-system"
                or git.get("tracked_dirty") is not False
                or preflight.get("checkpoint_sha256")
                != identities.get(method, {}).get("sha256")
                or preflight.get("checkpoint")
                != identities.get(method, {}).get("path")
                or preflight.get("checkpoint_integrity_manifest")
                != identities.get(method, {}).get("integrity_manifest")
                or int(preflight.get("checkpoint_step", -1)) != 300_000
                or preflight.get("weights") != "ema"
                or int(request.get("batch_size", -1)) != candidate_batch
                or int(request.get("prefix_budget", -1)) != prefix_budget
                or float(request.get("guidance_scale", math.nan)) != 1.5
                or float(request.get("guidance_rescale", math.nan)) != 0.0
                or request.get("cfg_batch_mode") != "batched"
                or request.get("precision") != "bf16"
                or int(request.get("warmup_forwards", -1)) != 2
                or int(request.get("measured_forwards", -1)) != 5
            ):
                raise ValueError(f"{method} sampling preflight contract differs")
    if observed_preflight_environment_shas != {selected_environment_sha}:
        raise ValueError("sampling preflight environment changed across candidates")

    for method in ("cofitok", "dense_identity"):
        selected_method = selected_methods.get(method, {})
        selected_method_environment = selected_method.get("runtime_environment")
        if not isinstance(selected_method_environment, dict) or (
            runtime_environment_sha256(selected_method_environment)
            != selected_environment_sha
            or selected_method.get("runtime_environment_sha256")
            != selected_environment_sha
        ):
            raise ValueError(f"{method} selected preflight environment differs")
        provenance = generation_reports[method].get("sample_provenance", {})
        if provenance.get("runtime_environment_sha256") != selected_environment_sha:
            raise ValueError(f"{method} sampling environment differs from selection")
        if identities.get(method, {}).get("sha256") != provenance.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{method} sampling selection checkpoint SHA256 differs")
        if int(identities.get(method, {}).get("step", -1)) != 300_000:
            raise ValueError(f"{method} sampling selection checkpoint is not step 300K")
        if int(provenance.get("sampling", {}).get("batch_size", -1)) != batch_size:
            raise ValueError(f"{method} formal generation ignored selected sampling batch")
        random_stream = provenance.get("sampling", {}).get("random_stream", {})
        if random_stream.get("batch_size_invariant") is not True:
            raise ValueError(f"{method} formal sampling random stream is batch-dependent")
    return {
        "batch_size": batch_size,
        "estimated_speedup_over_baseline": float(
            selection["selected"]["estimated_speedup_over_baseline"]
        ),
        "checkpoint_sha256": {
            method: identities[method]["sha256"]
            for method in ("cofitok", "dense_identity")
        },
        "runtime_environment_sha256": selected_environment_sha,
    }


def _visual_audit_evidence(
    report: dict[str, Any],
    generation_reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    if report.get("status") != "completed":
        raise ValueError("final deterministic visual audit is incomplete")
    if report.get("claim_policy", {}).get("quantitative_metric") is not False:
        raise ValueError("visual audit incorrectly claims quantitative metric status")
    if report.get("prefix_budgets") != [1, 2, 4, 8]:
        raise ValueError("visual audit lacks the required prefix budgets")
    git = report.get("git", {})
    if (
        git.get("revision") != expected_revision
        or git.get("branch") != "scale/generative-system"
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("visual audit was not built from the clean full-training revision")
    sources = report.get("sources", {})
    for method in ("cofitok", "dense_identity"):
        provenance = generation_reports[method].get("sample_provenance", {})
        source = sources.get(method, {})
        if source.get("checkpoint_sha256") != provenance.get("checkpoint_sha256"):
            raise ValueError(f"visual audit {method} checkpoint SHA256 differs")
        if source.get("sample_set_sha256") != provenance.get("sample_set_sha256"):
            raise ValueError(f"visual audit {method} sample-set SHA256 differs")
        statistics = report.get("statistics", {}).get(method, {})
        if int(statistics.get("exact_duplicate_count", -1)) != 0:
            raise ValueError(f"visual audit {method} contains exact duplicates")
        pixel_std = float(statistics.get("pixel_std", math.nan))
        if not math.isfinite(pixel_std) or pixel_std <= 0.0:
            raise ValueError(f"visual audit {method} pixel variation is invalid")
    panels = report.get("panels", {})
    if set(panels) != {"cofitok", "dense_identity", "cofitok_prefix_paths"}:
        raise ValueError("visual audit panel set is incomplete")
    for name, panel in panels.items():
        if len(str(panel.get("sha256", ""))) != 64 or int(
            panel.get("image_count", 0)
        ) < 1:
            raise ValueError(f"visual audit panel {name} provenance is invalid")
    return {
        "indices": list(report.get("indices", [])),
        "prefix_budgets": list(report["prefix_budgets"]),
        "panel_sha256": {name: panel["sha256"] for name, panel in panels.items()},
    }


def _verify_inference_smoke_outputs(
    report: dict[str, Any] | None,
    *,
    expected_root: str | Path,
) -> dict[str, Any] | None:
    if report is None:
        return None
    declared_root = Path(expected_root)
    root = declared_root.resolve()
    try:
        if declared_root.is_symlink() or not root.is_dir():
            raise ValueError(f"inference smoke directory is missing: {root}")
        rows = report.get("outputs")
        if not isinstance(rows, list) or int(report.get("output_count", -1)) != len(rows):
            raise ValueError("inference smoke report output count differs")
        verified = []
        paths: set[Path] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("inference smoke output row is malformed")
            raw_path = str(row.get("path", ""))
            path = Path(raw_path)
            if not path.is_absolute():
                raise ValueError("inference smoke output path is not absolute")
            resolved = path.resolve(strict=True)
            if path.as_posix() != resolved.as_posix():
                raise ValueError("inference smoke output path is not canonical")
            if not resolved.is_relative_to(root):
                raise ValueError("inference smoke output escapes its output root")
            if resolved.suffix.lower() != ".png" or resolved.name != row.get("filename"):
                raise ValueError("inference smoke output filename differs")
            if resolved in paths:
                raise ValueError("inference smoke report repeats an output path")
            paths.add(resolved)
            actual_sha = file_sha256(resolved)
            if actual_sha != row.get("sha256"):
                raise ValueError("inference smoke output SHA256 differs")
            with Image.open(resolved) as image:
                image.verify()
            with Image.open(resolved) as image:
                mode = image.mode
                width, height = image.size
            if mode != "RGB" or (width, height) != (256, 256):
                raise ValueError("inference smoke output is not an RGB 256x256 PNG")
            verified.append(
                {
                    "path": resolved.as_posix(),
                    "bytes": resolved.stat().st_size,
                    "sha256": actual_sha,
                    "mode": mode,
                    "width": width,
                    "height": height,
                }
            )
        discovered = {
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".png"
        }
        if discovered != paths:
            raise ValueError("inference smoke directory PNG set differs from report")
        replay_evidence = (
            validate_completed_inference_evidence(
                report,
                expected_root=root,
            )
            if int(report.get("schema_version", -1)) == 2
            else {
                "status": "legacy_unverified",
                "reason": "inference report predates resumable evidence",
            }
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "root": root.as_posix(),
            "error": str(error),
        }
    return {
        "status": "verified",
        "root": root.as_posix(),
        "output_count": len(verified),
        "outputs": sorted(verified, key=lambda row: row["path"]),
        "replay_evidence": replay_evidence,
    }


def _inference_export_evidence(
    exports: dict[str, dict[str, Any]],
    artifact_files: dict[str, dict[str, Any]],
    smoke_files: dict[str, dict[str, Any]],
    generation_reports: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
    final_gate: dict[str, Any],
    *,
    expected_export_revision: str | None = None,
    expected_export_branch: str | None = None,
) -> dict[str, Any]:
    if (expected_export_revision is None) != (expected_export_branch is None):
        raise ValueError(
            "inference export revision and branch must be provided together"
        )
    expected_execution_git = (
        None
        if expected_export_revision is None
        else {
            "revision": expected_export_revision,
            "branch": expected_export_branch,
            "tracked_dirty": False,
        }
    )
    evidence = {}
    release_authorizations = {}
    export_environment_shas = set()
    inference_environment_shas = set()
    for method, expected_smoke_count in (("cofitok", 4), ("dense_identity", 2)):
        export = exports[f"{method}_export"]
        preflight = exports[f"{method}_preflight"]
        smoke = exports[f"{method}_smoke"]
        source_provenance = generation_reports[method]["sample_provenance"]
        source_training = training_reports[method]
        expected_environment_sha = source_training.get(
            "runtime_environment_sha256"
        )
        expected_source_git = source_training.get("git")
        expected_authorization = source_training.get("training_authorization")
        if not isinstance(expected_authorization, dict):
            raise ValueError(
                f"{method} full training authorization is missing from export provenance"
            )
        release_authorization = export.get("release_authorization")
        if not isinstance(release_authorization, dict):
            raise ValueError(
                f"{method} inference artifact lacks final release authorization"
            )
        validate_generation_gate_binding(
            release_authorization,
            expected_stage="full",
            expected_gate=final_gate,
        )
        release_authorizations[method] = release_authorization
        if export.get("status") != "completed" or export.get("verified") is not True:
            raise ValueError(f"{method} inference export is incomplete")
        if export.get("resume_requested") is not True or not isinstance(
            export.get("partial_outputs_recovered"),
            bool,
        ):
            raise ValueError(
                f"{method} inference export lacks explicit resume evidence"
            )
        if export.get("weights") != "ema_export":
            raise ValueError(f"{method} inference export is not EMA-only")
        if int(export.get("checkpoint_step", -1)) != 300_000:
            raise ValueError(f"{method} inference export is not from step 300K")
        if export.get("source_checkpoint_sha256") != source_provenance.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{method} inference export source checkpoint differs")
        if (
            export.get("source_runtime_environment_sha256")
            != expected_environment_sha
            or export.get("source_git") != expected_source_git
            or export.get("source_training_authorization")
            != expected_authorization
        ):
            raise ValueError(f"{method} inference export source provenance differs")
        artifact_sha = str(export.get("artifact_sha256", ""))
        artifact_bytes = int(export.get("artifact_bytes", 0))
        source_bytes = int(export.get("source_checkpoint_bytes", 0))
        if len(artifact_sha) != 64 or not 0 < artifact_bytes < source_bytes:
            raise ValueError(f"{method} inference export size/hash is invalid")
        verified_file = artifact_files[method]
        if verified_file.get("status") != "verified":
            raise ValueError(
                f"{method} inference artifact file is invalid: "
                f"{verified_file.get('error', 'verification failed')}"
            )
        if verified_file.get("path") != export.get("artifact"):
            raise ValueError(f"{method} inference artifact path differs")
        if verified_file.get("integrity_manifest") != export.get(
            "artifact_integrity_manifest"
        ):
            raise ValueError(f"{method} inference artifact integrity path differs")
        export_manifest = export.get("export_manifest")
        verified_manifest = verified_file.get("export_manifest")
        manifest_payload = verified_file.get("export_manifest_payload")
        if (
            not isinstance(export_manifest, dict)
            or export_manifest != verified_manifest
            or not isinstance(manifest_payload, dict)
        ):
            raise ValueError(f"{method} inference export manifest differs")
        manifest_source = manifest_payload.get("source")
        manifest_target = manifest_payload.get("target")
        expected_manifest_authorization = {
            "authorization_stage": expected_authorization["stage"],
            "authorization_decision": expected_authorization["decision"],
            "authorization_gate_bytes": expected_authorization["gate_bytes"],
            "authorization_gate_sha256": expected_authorization["gate_sha256"],
            "authorization_gate_identity_sha256": expected_authorization[
                "gate_identity_sha256"
            ],
        }
        if (
            manifest_payload.get("artifact_type")
            != "cofitok_generation_inference"
            or manifest_payload.get("artifact_format_version") != 4
            or manifest_payload.get("weights") != "ema_export"
            or not isinstance(manifest_source, dict)
            or manifest_source.get("path") != export.get("source_checkpoint")
            or manifest_source.get("bytes") != source_bytes
            or manifest_source.get("sha256")
            != export.get("source_checkpoint_sha256")
            or manifest_source.get("step") != 300_000
            or manifest_source.get("runtime_environment_sha256")
            != expected_environment_sha
            or manifest_source.get("git") != expected_source_git
            or manifest_source.get("training_authorization")
            != expected_manifest_authorization
            or not isinstance(manifest_target, dict)
            or manifest_target.get("artifact") != export.get("artifact")
            or manifest_target.get("integrity_manifest")
            != export.get("artifact_integrity_manifest")
            or manifest_payload.get("release_authorization")
            != release_authorization
            or manifest_payload.get("execution") != export.get("execution")
        ):
            raise ValueError(f"{method} inference export manifest provenance differs")
        if (
            verified_file.get("artifact_sha256") != artifact_sha
            or int(verified_file.get("artifact_bytes", -1)) != artifact_bytes
            or int(verified_file.get("step", -1)) != 300_000
            or verified_file.get("source_checkpoint_sha256")
            != export.get("source_checkpoint_sha256")
            or verified_file.get("source_runtime_environment_sha256")
            != expected_environment_sha
            or {
                "revision": verified_file.get("source_git_revision"),
                "branch": verified_file.get("source_git_branch"),
                "dirty": verified_file.get("source_git_dirty"),
            }
            != expected_source_git
            or verified_file.get("source_training_authorization")
            != expected_authorization
            or verified_file.get("release_authorization")
            != release_authorization
        ):
            raise ValueError(f"{method} inference artifact bytes differ from report")
        if preflight.get("status") != "passed":
            raise ValueError(f"{method} inference export preflight failed")
        if preflight.get("checkpoint_sha256") != artifact_sha:
            raise ValueError(f"{method} export preflight artifact SHA256 differs")
        if preflight.get("artifact_type") != "cofitok_generation_inference":
            raise ValueError(f"{method} export preflight artifact type differs")
        if preflight.get("weights") != "ema_export":
            raise ValueError(f"{method} export preflight did not load exported EMA")
        if preflight.get("release_authorization_required") is not True:
            raise ValueError(
                f"{method} export preflight did not enforce release authorization"
            )
        if preflight.get("source_checkpoint_sha256") != export.get(
            "source_checkpoint_sha256"
        ):
            raise ValueError(f"{method} export preflight source provenance differs")
        if (
            preflight.get("source_runtime_environment_sha256")
            != expected_environment_sha
            or preflight.get("source_git") != expected_source_git
            or preflight.get("training_authorization")
            != expected_authorization
            or preflight.get("release_authorization")
            != release_authorization
        ):
            raise ValueError(f"{method} export preflight source identity differs")
        checkpoint = smoke.get("checkpoint", {})
        if (
            smoke.get("schema_version") != INFERENCE_REPORT_SCHEMA_VERSION
            or smoke.get("status") != "completed"
            or int(
            smoke.get("output_count", -1)
            )
            != expected_smoke_count
        ):
            raise ValueError(f"{method} inference export smoke is incomplete")
        expected_budgets = [1, 8] if method == "cofitok" else [1]
        expected_request = {
            "seeds": [0, 1],
            "class_ids": [0, 0],
            "prefix_budgets": expected_budgets,
            "batch_size": 2,
            "sample_steps": 10,
            "actual_timesteps": select_sampling_timesteps(1_000, 10),
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "eta": 0.0,
            "clip_x0": True,
            "precision": "bf16",
            "image_shape": [3, 256, 256],
        }
        if smoke.get("request") != expected_request:
            raise ValueError(f"{method} inference export smoke request differs")
        if checkpoint.get("checkpoint_sha256") != artifact_sha:
            raise ValueError(f"{method} export smoke artifact SHA256 differs")
        if checkpoint.get("artifact_type") != "cofitok_generation_inference":
            raise ValueError(f"{method} export smoke artifact type differs")
        if checkpoint.get("release_authorization_required") is not True:
            raise ValueError(
                f"{method} export smoke did not enforce release authorization"
            )
        if (
            checkpoint.get("weights") != "ema_export"
            or int(checkpoint.get("checkpoint_step", -1)) != 300_000
        ):
            raise ValueError(f"{method} export smoke did not load step-300K EMA")
        if checkpoint.get("source_checkpoint_sha256") != export.get(
            "source_checkpoint_sha256"
        ):
            raise ValueError(f"{method} export smoke source provenance differs")
        if (
            checkpoint.get("source_runtime_environment_sha256")
            != expected_environment_sha
            or checkpoint.get("source_git") != expected_source_git
            or checkpoint.get("training_authorization")
            != expected_authorization
            or checkpoint.get("release_authorization")
            != release_authorization
        ):
            raise ValueError(f"{method} export smoke source identity differs")
        expected_rows = {
            (seed, 0, budget)
            for budget in expected_budgets
            for seed in (0, 1)
        }
        reported_outputs = smoke.get("outputs", [])
        if len(reported_outputs) != expected_smoke_count or {
            (row.get("seed"), row.get("class_id"), row.get("prefix_budget"))
            for row in reported_outputs
        } != expected_rows:
            raise ValueError(f"{method} export smoke output request rows differ")
        verified_smoke = smoke_files[method]
        if (
            verified_smoke.get("status") != "verified"
            or int(verified_smoke.get("output_count", -1)) != expected_smoke_count
        ):
            raise ValueError(
                f"{method} inference smoke files are invalid: "
                f"{verified_smoke.get('error', 'verification failed')}"
            )
        if verified_smoke.get("replay_evidence", {}).get("status") != "verified":
            raise ValueError(
                f"{method} inference smoke lacks verified resume evidence"
            )
        verified_by_path = {
            row["path"]: row for row in verified_smoke.get("outputs", [])
        }
        for row in reported_outputs:
            path = str(row.get("path", ""))
            verified = verified_by_path.get(path)
            if (
                verified is None
                or verified.get("sha256") != row.get("sha256")
                or verified.get("mode") != "RGB"
                or verified.get("width") != 256
                or verified.get("height") != 256
            ):
                raise ValueError(f"{method} inference smoke PNG differs from report")
        execution_environment_sha = None
        export_environment_sha = None
        if expected_execution_git is not None:
            export_execution = export.get("execution")
            if not isinstance(export_execution, dict):
                raise ValueError(
                    f"{method} export execution provenance is missing"
                )
            if export_execution.get("git") != expected_execution_git:
                raise ValueError(
                    f"{method} export execution Git provenance differs"
                )
            export_environment = export_execution.get("runtime_environment")
            if not isinstance(export_environment, dict):
                raise ValueError(
                    f"{method} export execution runtime environment is missing"
                )
            export_environment_sha = runtime_environment_sha256(
                export_environment
            )
            if (
                export_execution.get("runtime_environment_sha256")
                != export_environment_sha
            ):
                raise ValueError(
                    f"{method} export execution runtime environment differs"
                )
            export_environment_shas.add(export_environment_sha)
            if preflight.get("git") != expected_execution_git:
                raise ValueError(
                    f"{method} export preflight execution Git provenance differs"
                )
            if smoke.get("git") != expected_execution_git:
                raise ValueError(
                    f"{method} export smoke execution Git provenance differs"
                )
            preflight_environment = preflight.get("runtime_environment")
            smoke_environment = smoke.get("runtime_environment")
            if not isinstance(preflight_environment, dict) or not isinstance(
                smoke_environment,
                dict,
            ):
                raise ValueError(
                    f"{method} export execution runtime environment is missing"
                )
            preflight_environment_sha = runtime_environment_sha256(
                preflight_environment
            )
            smoke_environment_sha = runtime_environment_sha256(smoke_environment)
            if (
                preflight.get("runtime_environment_sha256")
                != preflight_environment_sha
                or smoke.get("runtime_environment_sha256")
                != smoke_environment_sha
                or preflight_environment_sha != smoke_environment_sha
            ):
                raise ValueError(
                    f"{method} export execution runtime environment differs"
                )
            execution_environment_sha = preflight_environment_sha
            inference_environment_shas.add(execution_environment_sha)
        evidence[method] = {
            "artifact_path": verified_file["path"],
            "artifact_sha256": artifact_sha,
            "artifact_bytes": artifact_bytes,
            "export_manifest": export_manifest,
            "source_checkpoint_sha256": export["source_checkpoint_sha256"],
            "source_checkpoint_bytes": source_bytes,
            "source_runtime_environment_sha256": expected_environment_sha,
            "source_git": expected_source_git,
            "execution_git": expected_execution_git,
            "export_runtime_environment_sha256": export_environment_sha,
            "execution_runtime_environment_sha256": execution_environment_sha,
            "training_authorization": expected_authorization,
            "release_authorization": release_authorization,
            "smoke_output_count": expected_smoke_count,
            "smoke_output_sha256": sorted(
                row["sha256"] for row in verified_smoke["outputs"]
            ),
        }
    if release_authorizations["cofitok"] != release_authorizations["dense_identity"]:
        raise ValueError("CoFiTok and dense artifacts used different release authorizations")
    if expected_execution_git is not None and (
        len(export_environment_shas) != 1
        or len(inference_environment_shas) != 1
    ):
        raise ValueError(
            "CoFiTok and dense artifacts used different execution environments"
        )
    return evidence


def _final_gate_evidence(
    gate: dict[str, Any],
    generation_reports: dict[str, dict[str, Any]],
    source_verification: dict[str, Any],
) -> dict[str, Any]:
    evidence = _gate_evidence(
        gate,
        source_verification,
        stage="full",
        decision="large_scale_generation_ready",
    )
    indexed_gates: dict[str, dict[str, Any]] = {}
    for row in gate.get("gates", []):
        name = str(row.get("name", ""))
        if name in indexed_gates:
            raise ValueError(f"final gate contains duplicate check: {name}")
        indexed_gates[name] = row
    required_quality_gates = {
        "generation_metrics_complete",
        "matched_real_set_provenance",
        "formal_sampling_protocol",
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
        "coarse_token_utilization",
        "restricted_synthesis_contract",
        "shuffle_mismatch",
        "full_training_checkpoint_integrity",
    }
    missing_quality_gates = sorted(required_quality_gates - indexed_gates.keys())
    if missing_quality_gates:
        raise ValueError(
            "final gate lacks required quality checks: " + ", ".join(missing_quality_gates)
        )
    thresholds = gate.get("thresholds", {})
    required_thresholds = {
        "max_fid_regression": ("max", 0.05),
        "max_absolute_fid": ("max", 20.0),
        "max_endpoint_regression": ("max", 0.05),
        "min_coarse_token_energy_ratio": ("min", 0.05),
        "min_precision": ("min", 0.30),
        "min_recall": ("min", 0.30),
        "max_precision_regression": ("max", 0.05),
        "max_recall_regression": ("max", 0.05),
    }
    for name, (direction, boundary) in required_thresholds.items():
        value = float(thresholds.get(name, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"final gate threshold {name} is missing or non-finite")
        if (direction == "max" and value > boundary) or (
            direction == "min" and value < boundary
        ):
            raise ValueError(f"final gate threshold {name} is weaker than required")
    matches = [
        row.get("evidence", {})
        for row in gate.get("gates", [])
        if row.get("name") == "matched_sampling_provenance"
    ]
    if len(matches) != 1:
        raise ValueError("final gate lacks unique matched sampling provenance")
    gate_sampling = matches[0]
    for method, prefix in (("cofitok", "cofitok"), ("dense_identity", "dense")):
        provenance = generation_reports[method]["sample_provenance"]
        if gate_sampling.get(f"{prefix}_checkpoint_sha256") != provenance.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"final gate {method} checkpoint SHA256 differs")
        if gate_sampling.get(f"{prefix}_sample_set_sha256") != provenance.get(
            "sample_set_sha256"
        ):
            raise ValueError(f"final gate {method} sample-set SHA256 differs")
        metrics = generation_reports[method].get("metrics", {})
        summary = gate.get("summary", {})
        for metric, suffix in (
            ("frechet_inception_distance", "fid"),
            ("precision", "precision"),
            ("recall", "recall"),
        ):
            gate_value = float(summary.get(f"{prefix}_{suffix}", math.nan))
            report_value = float(metrics.get(metric, math.nan))
            if (
                not math.isfinite(gate_value)
                or not math.isfinite(report_value)
                or not math.isclose(gate_value, report_value, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"final gate {method} {metric} differs from sample metrics")
    evidence["sampling_provenance_bound"] = True
    environment_matches = [
        row.get("evidence", {})
        for row in gate.get("gates", [])
        if row.get("name") == "matched_sampling_runtime_environment"
    ]
    if len(environment_matches) != 1:
        raise ValueError("final gate lacks unique sampling runtime environment")
    gate_environments = environment_matches[0]
    for method in ("cofitok", "dense_identity"):
        expected_sha = generation_reports[method]["sample_provenance"].get(
            "runtime_environment_sha256"
        )
        if gate_environments.get(method, {}).get("sha256") != expected_sha:
            raise ValueError(f"final gate {method} sampling environment differs")
    evaluator_environment_matches = [
        row.get("evidence", {})
        for row in gate.get("gates", [])
        if row.get("name") == "matched_evaluator_runtime_environment"
    ]
    if len(evaluator_environment_matches) != 1:
        raise ValueError("final gate lacks unique evaluator runtime environment")
    gate_evaluator_environments = evaluator_environment_matches[0]
    for method in ("cofitok", "dense_identity"):
        expected_sha = generation_reports[method].get("runtime_environment_sha256")
        if gate_evaluator_environments.get(method, {}).get("sha256") != expected_sha:
            raise ValueError(f"final gate {method} evaluator environment differs")
    real_set_matches = [
        row.get("evidence", {})
        for row in gate.get("gates", [])
        if row.get("name") == "matched_real_set_provenance"
    ]
    if len(real_set_matches) != 1:
        raise ValueError("final gate lacks unique real-set provenance")
    gate_real_sets = real_set_matches[0]
    for method in ("cofitok", "dense_identity"):
        expected_real_set = generation_reports[method].get("real_set", {})
        gate_real_set = gate_real_sets.get(method, {})
        if (
            gate_real_set.get("valid") is not True
            or gate_real_set.get("real_cache_name")
            != generation_reports[method].get("parameters", {}).get("real_cache_name")
            or gate_real_set.get("sha256") != expected_real_set.get("sha256")
            or gate_real_set.get("digest_schema")
            != expected_real_set.get("digest_schema")
            or gate_real_set.get("root") != expected_real_set.get("root")
            or int(gate_real_set.get("image_count", -1))
            != int(expected_real_set.get("image_count", -2))
        ):
            raise ValueError(f"final gate {method} real-set provenance differs")
    evidence["quality_metrics_bound"] = True
    evidence["quality_thresholds"] = {
        name: thresholds[name] for name in required_thresholds
    }
    return evidence


_CLASS_FIDELITY_ROW_FIELDS = {
    "class_fidelity_sample_count": "sample_count",
    "class_top1_accuracy": "top1_accuracy",
    "class_top5_accuracy": "top5_accuracy",
    "class_mean_target_probability": "mean_target_probability",
    "class_target_negative_log_likelihood": "target_negative_log_likelihood",
    "class_predicted_class_fraction": "predicted_class_fraction",
    "class_normalized_predicted_entropy": (
        "normalized_predicted_class_entropy"
    ),
}


def _class_fidelity_comparison_evidence(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    qualification: dict[str, Any],
    final_gate: dict[str, Any],
) -> dict[str, Any]:
    provenance = final_gate.get("provenance_contract")
    provenance = provenance if isinstance(provenance, dict) else {}
    validated = validate_class_fidelity_qualification(
        qualification,
        expected_stage="full",
        expected_revision=provenance.get("evaluation_revision"),
        expected_branch=provenance.get("evaluation_branch"),
        require_pass=True,
    )
    expected_summary = {
        "status": validated["status"],
        "valid": validated["valid"],
        "classifier": validated["classifier"],
        "thresholds": validated["thresholds"],
        "sampling_contract": validated["sampling_contract"],
        "claim_boundary": validated["claim_boundary"],
    }
    if report.get("class_fidelity") != expected_summary:
        raise ValueError("comparison class-fidelity summary differs from qualification")

    gate_rows = [
        row
        for row in final_gate.get("gates", [])
        if row.get("name") == "class_conditional_fidelity"
    ]
    if len(gate_rows) != 1 or gate_rows[0].get("passed") is not True:
        raise ValueError("final gate lacks passing class-fidelity evidence")
    gate_evidence = gate_rows[0].get("evidence")
    if not isinstance(gate_evidence, dict):
        raise ValueError("final gate class-fidelity evidence is malformed")
    gate_checks = gate_evidence.get("checks")
    if (
        gate_evidence.get("qualification") != qualification
        or gate_evidence.get("valid") is not True
        or gate_evidence.get("status") != "pass"
        or gate_evidence.get("classifier") != validated["classifier"]
        or gate_evidence.get("sampling_contract") != validated["sampling_contract"]
        or gate_evidence.get("metrics") != validated["metrics"]
        or not isinstance(gate_checks, dict)
        or not gate_checks
        or any(value is not True for value in gate_checks.values())
    ):
        raise ValueError("final gate class-fidelity evidence differs from qualification")

    indexed = {row.get("method"): row for row in rows}
    method_metrics = {
        "CoFiTok K=8": validated["metrics"]["cofitok"],
        "Dense identity": validated["metrics"]["dense_identity"],
    }
    if set(indexed) != set(method_metrics):
        raise ValueError("class-fidelity comparison method identities differ")
    for method, metrics in method_metrics.items():
        row = indexed[method]
        for row_field, metric_field in _CLASS_FIDELITY_ROW_FIELDS.items():
            observed = row.get(row_field)
            expected = metrics[metric_field]
            if row_field == "class_fidelity_sample_count":
                if int(observed) != int(expected):
                    raise ValueError(
                        f"comparison {method} {row_field} differs from qualification"
                    )
            elif not math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"comparison {method} {row_field} differs from qualification"
                )
    contract = validated["sampling_contract"]
    identity_expectations = {
        "CoFiTok K=8": (
            contract["cofitok_checkpoint_sha256"],
            contract["cofitok_sample_set_sha256"],
        ),
        "Dense identity": (
            contract["dense_checkpoint_sha256"],
            contract["dense_sample_set_sha256"],
        ),
    }
    for method, (checkpoint_sha, sample_sha) in identity_expectations.items():
        row = indexed[method]
        if (
            row.get("checkpoint_sha256") != checkpoint_sha
            or row.get("sample_set_sha256") != sample_sha
        ):
            raise ValueError(
                f"comparison {method} class-fidelity sample identity differs"
            )
    matched_summary = report.get("matched_summary")
    matched_summary = matched_summary if isinstance(matched_summary, dict) else {}
    for field, expected in {
        "cofitok_minus_dense_class_top1": (
            float(method_metrics["CoFiTok K=8"]["top1_accuracy"])
            - float(method_metrics["Dense identity"]["top1_accuracy"])
        ),
        "cofitok_minus_dense_class_top5": (
            float(method_metrics["CoFiTok K=8"]["top5_accuracy"])
            - float(method_metrics["Dense identity"]["top5_accuracy"])
        ),
    }.items():
        if not math.isclose(
            float(matched_summary.get(field, math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"comparison matched summary differs: {field}")
    return {
        "status": "pass",
        "classifier": validated["classifier"],
        "sample_count_per_method": contract["sample_count_per_method"],
        "cofitok_top1_accuracy": method_metrics["CoFiTok K=8"][
            "top1_accuracy"
        ],
        "dense_top1_accuracy": method_metrics["Dense identity"][
            "top1_accuracy"
        ],
        "release_authorization_allowed": validated["claim_boundary"][
            "release_authorization_allowed"
        ],
    }


def _comparison_evidence(
    report: dict[str, Any],
    generation_reports: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
    training_contention_report: dict[str, Any],
    official_related: dict[str, Any],
    official_related_sha256: str,
    source_verification: dict[str, Any],
    *,
    final_gate: dict[str, Any] | None = None,
    class_fidelity_qualification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if report.get("schema_version") != COMPARISON_REPORT_SCHEMA_VERSION:
        raise ValueError("large-scale comparison schema is stale")
    if report.get("status") != "ready":
        raise ValueError("large-scale comparison is not ready")
    if report.get("final_gate") != {
        "status": "pass",
        "decision": "large_scale_generation_ready",
    }:
        raise ValueError("comparison final-gate decision differs")
    if (
        source_verification.get("status") != "verified"
        or source_verification.get("source_reports") != report.get("source_reports")
    ):
        raise ValueError("comparison source-report verification differs")
    training_contention = validate_gpu_contention_evidence(
        training_contention_report
    )
    if report.get("training_contention") != training_contention:
        raise ValueError("comparison training contention summary differs")
    rows = report.get("matched_training_rows", [])
    if len(rows) != 2 or any(int(row.get("sample_count", -1)) != 50_000 for row in rows):
        raise ValueError("large-scale comparison lacks the matched 50K pair")
    policy = report.get("comparison_policy", {})
    if policy.get("primary_direct_tier") != "matched_training_direct":
        raise ValueError("comparison primary tier is not matched training")
    if policy.get("external_context_tier") != "official_pretrained_contextual":
        raise ValueError("comparison external tier is mislabeled")
    if policy.get("cross_tier_numeric_ranking_allowed") is not False:
        raise ValueError("comparison incorrectly permits cross-tier numeric ranking")
    if (
        policy.get("training_wall_clock_direct_comparison_allowed")
        is not training_contention["direct_comparison_allowed"]
        or policy.get("training_wall_clock_comparison_reason")
        != training_contention["reason"]
    ):
        raise ValueError("comparison training wall-clock policy differs")
    expected_budget_policy = training_budget_policy(rows, training_contention)
    if report.get("training_budget_policy") != expected_budget_policy:
        raise ValueError("comparison training budget policy differs")
    source = report.get("official_context_source", {})
    if (
        source.get("sha256") != official_related_sha256
        or source.get("schema_version") != 1
    ):
        raise ValueError("comparison official-context source binding differs")
    indexed = {row.get("method"): row for row in rows}
    expected = {
        "CoFiTok K=8": generation_reports["cofitok"]["sample_provenance"],
        "Dense identity": generation_reports["dense_identity"]["sample_provenance"],
    }
    if set(indexed) != set(expected):
        raise ValueError("large-scale comparison method identities differ")
    source_profile = str(report.get("source_profile", "full"))
    if source_profile == "stability_full":
        if final_gate is None or class_fidelity_qualification is None:
            raise ValueError(
                "stability-full comparison audit requires class-fidelity evidence"
            )
        class_fidelity_evidence = _class_fidelity_comparison_evidence(
            report,
            rows,
            class_fidelity_qualification,
            final_gate,
        )
    else:
        class_fidelity_evidence = None
        if report.get("class_fidelity") is not None:
            raise ValueError("legacy comparison unexpectedly contains class fidelity")
        for row in rows:
            if any(row.get(field) is not None for field in _CLASS_FIDELITY_ROW_FIELDS):
                raise ValueError("legacy comparison class-fidelity row is not empty")
        summary = report.get("matched_summary", {})
        if any(
            summary.get(field) is not None
            for field in (
                "cofitok_minus_dense_class_top1",
                "cofitok_minus_dense_class_top5",
            )
        ):
            raise ValueError("legacy comparison class-fidelity summary is not empty")
    method_keys = {"CoFiTok K=8": "cofitok", "Dense identity": "dense_identity"}
    for method, provenance in expected.items():
        row = indexed[method]
        key = method_keys[method]
        training = training_reports[key]
        cost = training_cost_summary(training)
        if cost["valid"] is not True:
            raise ValueError(f"comparison {method} source training cost is invalid")
        if (
            row.get("comparison_tier") != "matched_training_direct"
            or row.get("directly_comparable_to_cofitok") is not True
            or row.get("dataset") != "imagenet_256"
            or int(row.get("resolution", -1)) != 256
            or int(row.get("training_steps", -1)) != 300_000
            or int(row.get("sample_count", -1)) != 50_000
            or row.get("protocol_note")
            != MATCHED_TRAINING_PROTOCOL_NOTE
            or row.get("training_time_measurement")
            != training_contention["measurement"]
            or row.get("training_budget_basis")
            != MATCHED_TRAINING_BUDGET_BASIS
            or row.get("compute_matched_claim_allowed") is not False
            or row.get("training_cost_fields_role") != "measured_outcomes"
            or row.get("training_wall_clock_directly_comparable")
            is not training_contention["direct_comparison_allowed"]
            or row.get("training_throughput_directly_comparable")
            is not training_contention["direct_comparison_allowed"]
            or row.get("training_wall_clock_comparison_reason")
            != training_contention["reason"]
        ):
            raise ValueError(f"comparison {method} matched protocol metadata differs")
        if row.get("checkpoint_sha256") != provenance.get("checkpoint_sha256"):
            raise ValueError(f"comparison {method} checkpoint SHA256 differs")
        if row.get("sample_set_sha256") != provenance.get("sample_set_sha256"):
            raise ValueError(f"comparison {method} sample-set SHA256 differs")
        generation = generation_reports[key]
        sampling = provenance.get("sampling", {})
        if row.get("weights") != provenance.get("weights"):
            raise ValueError(f"comparison {method} weights differ from formal sampling")
        protocol_fields = {
            "sampling_protocol_schema": "protocol_schema",
            "sampling_inference_api": "inference_api",
            "sampler": "sampler",
            "num_train_timesteps": "num_train_timesteps",
            "sample_steps": "sample_steps",
            "actual_timesteps": "actual_timesteps",
            "guidance_scale": "guidance_scale",
            "guidance_rescale": "guidance_rescale",
            "cfg_batch_mode": "cfg_batch_mode",
            "eta": "eta",
            "clip_x0": "clip_x0",
            "sampling_precision": "precision",
            "sampling_seed": "seed",
            "sampling_start_index": "start_index",
            "class_schedule": "class_schedule",
            "prefix_budgets": "prefix_budgets",
            "sampling_random_stream": "random_stream",
        }
        for row_field, sampling_field in protocol_fields.items():
            if row.get(row_field) != sampling.get(sampling_field):
                raise ValueError(
                    f"comparison {method} {row_field} differs from formal sampling"
                )
        generation_real_set = generation.get("real_set", {})
        if (
            row.get("real_set_digest_schema")
            != generation_real_set.get("digest_schema")
            or row.get("real_set_sha256") != generation_real_set.get("sha256")
            or int(row.get("real_image_count", -1))
            != int(generation_real_set.get("image_count", -2))
        ):
            raise ValueError(f"comparison {method} real-set SHA256 differs")
        if row.get("evaluator_runtime_environment_sha256") != generation.get(
            "runtime_environment_sha256"
        ):
            raise ValueError(f"comparison {method} evaluator environment differs")
        if row.get("evaluator") != generation.get("implementation"):
            raise ValueError(f"comparison {method} evaluator implementation differs")
        expected_exact = {
            "parameter_count": int(training["parameter_count"]),
            "effective_batch_size": int(cost["effective_batch_size"]),
            "training_images_seen": int(cost["samples_seen"]),
            "peak_vram_bytes": int(cost["peak_vram_bytes"]),
        }
        for field, value in expected_exact.items():
            if int(row.get(field, -1)) != value:
                raise ValueError(f"comparison {method} {field} differs from training report")
        expected_float = {
            "training_elapsed_seconds": float(cost["elapsed_seconds"]),
            "training_images_per_second": float(cost["images_per_second"]),
        }
        sampling_progress = generation_reports[key]["sample_provenance"][
            "sampling_progress"
        ]
        sampling = generation_reports[key]["sample_provenance"]["sampling"]
        sampling_elapsed = float(sampling_progress["cumulative_elapsed_seconds"])
        expected_exact["sample_batch_size"] = int(sampling["batch_size"])
        if int(row.get("sample_batch_size", -1)) != expected_exact["sample_batch_size"]:
            raise ValueError(f"comparison {method} sampling batch differs")
        if int(row.get("sampling_invocations", -1)) != int(
            sampling_progress.get("invocation", -2)
        ):
            raise ValueError(f"comparison {method} sampling invocation differs")
        expected_float.update(
            {
                "sampling_elapsed_seconds": sampling_elapsed,
                "sampling_images_per_second": 50_000 / sampling_elapsed,
            }
        )
        for field, value in expected_float.items():
            if not math.isclose(
                float(row.get(field, math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"comparison {method} {field} differs from source evidence")
        metric_fields = {
            "fid": "frechet_inception_distance",
            "inception_score": "inception_score_mean",
            "precision": "precision",
            "recall": "recall",
        }
        for row_field, source_field in metric_fields.items():
            if not math.isclose(
                float(row.get(row_field, math.nan)),
                float(generation["metrics"].get(source_field, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"comparison {method} {row_field} differs from generation metrics"
                )
    official_rows = official_related.get("rows", [])
    expected_aliases = {"d_ar": "D-AR", "mar": "MAR", "retok": "ReTok"}
    if official_related.get("schema_version") != 1:
        raise ValueError("official contextual source schema differs")
    source_index = {row.get("alias"): row for row in official_rows}
    report_index = {
        row.get("alias"): row for row in report.get("official_context_rows", [])
    }
    if set(source_index) != set(report_index) or set(source_index) != set(expected_aliases):
        raise ValueError("comparison official contextual method identities differ")
    metric_names = ("fid", "inception_score", "precision", "recall")
    for alias, expected_method in expected_aliases.items():
        source_row = source_index[alias]
        row = report_index[alias]
        if (
            source_row.get("method") != expected_method
            or row.get("method") != expected_method
            or row.get("comparison_tier") != "official_pretrained_contextual"
            or row.get("directly_comparable_to_cofitok") is not False
            or row.get("dataset") != source_row.get("dataset")
            or source_row.get("dataset") != "imagenet_256"
            or int(row.get("resolution", -1)) != int(source_row.get("resolution", -2))
            or int(source_row.get("resolution", -1)) != 256
            or int(row.get("sample_count", -1)) != int(source_row.get("sample_count", -2))
            or int(source_row.get("sample_count", -1)) != 50_000
            or row.get("training_steps") is not None
            or row.get("parameter_count") is not None
            or row.get("source_status") != source_row.get("status")
            or source_row.get("status") != "completed_eval_only_50k"
            or row.get("paper_table_role") != source_row.get("paper_table_role")
            or source_row.get("paper_table_role") != "secondary related-method only"
            or row.get("protocol_note") != source_row.get("protocol")
            or row.get("source_metrics") != source_row.get("metrics_txt")
            or row.get("source_npz") != source_row.get("npz")
            or row.get("source_kind") != source_row.get("source_kind")
            or row.get("sample_steps") is not None
            or row.get("guidance_scale") is not None
            or row.get("checkpoint_sha256") is not None
            or row.get("sample_set_sha256") is not None
            or row.get("evaluator")
            != {
                "package": "ADM TensorFlow evaluation graph",
                "version": "pinned baseline protocol",
            }
        ):
            raise ValueError(f"comparison official contextual metadata differs: {alias}")
        for metric in metric_names:
            if not math.isclose(
                float(row.get(metric, math.nan)),
                float(source_row.get(metric, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"comparison official contextual metric differs: {alias}/{metric}")
    cofitok_row = indexed["CoFiTok K=8"]
    dense_row = indexed["Dense identity"]
    expected_summary = {
        "cofitok_minus_dense_fid": cofitok_row["fid"] - dense_row["fid"],
        "cofitok_relative_fid": cofitok_row["fid"] / dense_row["fid"] - 1.0,
    }
    summary = report.get("matched_summary", {})
    for field, expected_value in expected_summary.items():
        if not math.isclose(
            float(summary.get(field, math.nan)),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"comparison matched summary differs: {field}")
    return {
        "matched_methods": [row["method"] for row in rows],
        "official_context_methods": sorted(expected_aliases),
        "official_context_source_sha256": official_related_sha256,
        "source_report_sha256": {
            name: identity["sha256"]
            for name, identity in report["source_reports"].items()
        },
        "cross_tier_numeric_ranking_allowed": False,
        "class_fidelity": class_fidelity_evidence,
    }


def _deployment_transition_evidence(
    receipt: dict[str, Any],
    *,
    expected_training_revision: str,
    expected_target_revision: str,
    verification_files: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    if receipt.get("schema_version") != 2 or receipt.get("status") != "pass":
        raise ValueError("generation deployment receipt did not pass")
    if receipt.get("expected_training_revision") != expected_training_revision:
        raise ValueError("deployment receipt training revision differs")
    if receipt.get("target_revision") != expected_target_revision:
        raise ValueError("deployment receipt target revision differs")
    git = receipt.get("git", {})
    if (
        git.get("revision") != expected_target_revision
        or git.get("branch") != "scale/generative-system"
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("deployment receipt Git state is invalid")
    bundle = receipt.get("bundle", {})
    if (
        int(bundle.get("bytes", 0)) < 1
        or len(str(bundle.get("sha256", ""))) != 64
        or expected_target_revision not in bundle.get("heads", [])
        or bundle.get("prerequisites") != [expected_training_revision]
    ):
        raise ValueError("deployment receipt bundle integrity is invalid")
    bundle_source = verification_files.get("bundle")
    if bundle_source is None or bundle_source.get("status") != "verified":
        raise ValueError("deployment bundle source is missing or invalid")
    if (
        bundle.get("path") != bundle_source.get("path")
        or bundle.get("bytes") != bundle_source.get("bytes")
        or bundle.get("sha256") != bundle_source.get("sha256")
        or sorted(bundle.get("heads", [])) != bundle_source.get("heads")
        or bundle.get("prerequisites") != bundle_source.get("prerequisites")
    ):
        raise ValueError("deployment bundle source binding differs")
    pair = receipt.get("training_pair_validation", {})
    if (
        pair.get("status") != "pass"
        or pair.get("expected_revision") != expected_training_revision
        or len(str(pair.get("sha256", ""))) != 64
    ):
        raise ValueError("deployment receipt training-pair binding is invalid")
    verification = receipt.get("verification", {})
    expected_sources = {
        "pytest": "pytest",
        "runbook_syntax": "runbook_syntax",
        "untracked_target_conflicts": "conflict_scan",
    }
    verified_sources = {}
    for receipt_name, source_name in expected_sources.items():
        binding = verification.get(receipt_name, {})
        observed = verification_files.get(source_name)
        if observed is None or observed.get("status") != "verified":
            raise ValueError(f"deployment {source_name} source is missing or invalid")
        if (
            binding.get("path") != observed.get("path")
            or binding.get("bytes") != observed.get("bytes")
            or binding.get("sha256") != observed.get("sha256")
        ):
            raise ValueError(f"deployment {source_name} source binding differs")
        verified_sources[source_name] = observed

    conflict_scan = verified_sources["conflict_scan"].get("content", {})
    validate_conflict_scan(
        conflict_scan,
        expected_training_revision=expected_training_revision,
        target_revision=expected_target_revision,
    )
    conflict_binding = verification["untracked_target_conflicts"]
    if (
        conflict_binding.get("status") != "pass"
        or conflict_binding.get("conflict_count") != 0
        or conflict_binding.get("target_added_path_count")
        != conflict_scan.get("target_added_path_count")
    ):
        raise ValueError("deployment conflict scan receipt summary differs")

    runbook_syntax = verified_sources["runbook_syntax"].get("content", {})
    checked_count = validate_runbook_syntax(
        runbook_syntax, target_revision=expected_target_revision
    )
    runbook_binding = verification["runbook_syntax"]
    if (
        runbook_binding.get("status") != "pass"
        or runbook_binding.get("checked_count") != checked_count
    ):
        raise ValueError("deployment runbook receipt summary differs")

    pytest_summary = verified_sources["pytest"].get("summary", {})
    validate_pytest_summary(pytest_summary)
    pytest_binding = verification["pytest"]
    for key in ("status", "tests", "failures", "errors", "skipped"):
        if pytest_binding.get(key) != pytest_summary.get(key):
            raise ValueError("deployment pytest receipt summary differs")
    return {
        "training_revision": expected_training_revision,
        "target_revision": expected_target_revision,
        "bundle_sha256": bundle["sha256"],
        "training_pair_validation_sha256": pair["sha256"],
        "pytest_sha256": verified_sources["pytest"]["sha256"],
        "runbook_syntax_sha256": verified_sources["runbook_syntax"]["sha256"],
        "conflict_scan_sha256": verified_sources["conflict_scan"]["sha256"],
        "runbook_count": checked_count,
    }


def _storage_capacity_evidence(
    reports: dict[str, dict[str, Any]],
    *,
    expected_revision: str,
    expected_path: str,
) -> dict[str, Any]:
    requirements = {
        "10pct_posteval": {
            "checkpoint_count": 0,
            "sample_count": 20_256,
            "additional_bytes": 16 * GIB,
            "safety_margin_bytes": 32 * GIB,
        },
        "full_training": {
            "checkpoint_count": 16,
            "sample_count": 16_384,
            "additional_bytes": 16 * GIB,
            "safety_margin_bytes": 64 * GIB,
        },
        "full_posteval": {
            "checkpoint_count": 0,
            "sample_count": 100_256,
            "additional_bytes": 16 * GIB,
            "safety_margin_bytes": 64 * GIB,
        },
    }
    if set(reports) != set(requirements):
        raise ValueError("storage capacity report set is incomplete")

    evidence = {}
    normalized_expected_path = expected_path.replace("\\", "/").rstrip("/")
    for stage, minimums in requirements.items():
        report = reports[stage]
        if (
            report.get("role") != "generation_storage_capacity_preflight"
            or report.get("stage") != stage
            or report.get("status") != "pass"
        ):
            raise ValueError(f"storage capacity report is invalid for {stage}")
        git = report.get("git", {})
        if (
            git.get("revision") != expected_revision
            or git.get("branch") != "scale/generative-system"
            or git.get("tracked_dirty") is not False
        ):
            raise ValueError(f"storage capacity Git provenance is invalid for {stage}")
        filesystem = report.get("filesystem", {})
        observed_path = str(filesystem.get("path", "")).replace("\\", "/").rstrip("/")
        if observed_path != normalized_expected_path:
            raise ValueError(f"storage capacity path differs for {stage}")
        plan = report.get("plan", {})
        schema_version = int(report.get("schema_version", 1))
        if schema_version not in {1, 2}:
            raise ValueError(f"storage capacity schema is unsupported for {stage}")
        if int(plan.get("estimated_sample_bytes_each", -1)) < 256 * KIB:
            raise ValueError(f"storage sample-size estimate was weakened for {stage}")
        for key, minimum in minimums.items():
            if int(plan.get(key, -1)) < minimum:
                raise ValueError(f"storage capacity reserve {key} was weakened for {stage}")
        if stage == "full_training" and int(plan.get("checkpoint_bytes_each", 0)) < 1:
            raise ValueError("full-training storage plan lacks measured checkpoint bytes")
        if schema_version == 2:
            reference_bytes = int(plan.get("reference_checkpoint_bytes_each", -1))
            multiplier = float(plan.get("checkpoint_size_multiplier", math.nan))
            if reference_bytes < 0 or not math.isfinite(multiplier) or multiplier < 1.0:
                raise ValueError(f"storage checkpoint scaling is invalid for {stage}")
            expected_checkpoint_bytes = math.ceil(reference_bytes * multiplier)
            if expected_checkpoint_bytes != int(plan.get("checkpoint_bytes_each", -1)):
                raise ValueError(f"storage checkpoint scaling arithmetic differs for {stage}")
        checkpoint_reserve = int(plan.get("checkpoint_count", -1)) * int(
            plan.get("checkpoint_bytes_each", -1)
        )
        sample_reserve = int(plan.get("sample_count", -1)) * int(
            plan.get("estimated_sample_bytes_each", -1)
        )
        if checkpoint_reserve != int(plan.get("checkpoint_reserve_bytes", -2)):
            raise ValueError(f"storage checkpoint reserve arithmetic differs for {stage}")
        if sample_reserve != int(plan.get("sample_reserve_bytes", -2)):
            raise ValueError(f"storage sample reserve arithmetic differs for {stage}")
        required = (
            checkpoint_reserve
            + sample_reserve
            + int(plan.get("additional_bytes", -1))
            + int(plan.get("safety_margin_bytes", -1))
        )
        total = int(filesystem.get("total_bytes", -1))
        used = int(filesystem.get("used_bytes", -1))
        free = int(filesystem.get("free_bytes", -1))
        if total < 1 or used < 0 or free < 0 or used + free > total:
            raise ValueError(f"storage filesystem usage is invalid for {stage}")
        if required != int(plan.get("required_free_bytes", -2)) or required < 1:
            raise ValueError(f"storage capacity arithmetic differs for {stage}")
        if free < required or int(report.get("headroom_bytes", -1)) != free - required:
            raise ValueError(f"storage capacity headroom is invalid for {stage}")
        evidence[stage] = {
            "free_bytes": free,
            "required_free_bytes": required,
            "headroom_bytes": free - required,
        }
    return evidence


def _full_training_monitor_evidence(
    report: dict[str, Any],
    *,
    expected_revision: str,
    expected_output_root: str,
) -> dict[str, Any]:
    if (
        int(report.get("schema_version", 0)) < 2
        or report.get("monitor") != "generation_full_matched_300k"
        or report.get("status") != "pass"
        or report.get("stage") != "complete"
        or report.get("issues") != []
    ):
        raise ValueError("full-training monitor did not reach a clean pass state")
    git = report.get("git", {})
    if (
        git.get("revision") != expected_revision
        or git.get("branch") != "scale/generative-system"
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("full-training monitor Git provenance is invalid")
    expected_dirs = {
        "cofitok": "imagenet256_full_cofitok_k8_300k",
        "dense_identity": "imagenet256_full_dense_300k",
    }
    protected = {50_000, 100_000, 200_000, 300_000}
    root = expected_output_root.replace("\\", "/").rstrip("/")
    evidence = {}
    for method, directory in expected_dirs.items():
        run = report.get("runs", {}).get(method, {})
        expected_dir = f"{root}/{directory}"
        observed_dir = str(run.get("run_dir", "")).replace("\\", "/").rstrip("/")
        if observed_dir != expected_dir:
            raise ValueError(f"full-training monitor path differs for {method}")
        if (
            run.get("complete") is not True
            or int(run.get("expected_steps", -1)) != 300_000
            or int(run.get("last_step", -1)) != 300_000
            or int(run.get("metric_rows", 0)) < 1
            or run.get("health_issues") != []
        ):
            raise ValueError(f"full-training monitor run is invalid for {method}")
        checkpoints = {
            int(row.get("step", -1)): int(row.get("bytes", 0))
            for row in run.get("checkpoints", [])
        }
        if not protected.issubset(checkpoints) or any(
            checkpoints[step] < 1 for step in protected
        ):
            raise ValueError(f"full-training monitor lacks protected checkpoints for {method}")
        evidence[method] = {
            "last_step": 300_000,
            "metric_rows": int(run.get("metric_rows", 0)),
            "protected_checkpoint_bytes": {
                str(step): checkpoints[step] for step in sorted(protected)
            },
        }
    return evidence


def _complete_resume_history_evidence(
    histories: dict[str, dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_methods = {"cofitok", "dense_identity"}
    if set(histories) != expected_methods:
        raise ValueError("matched full training resume-history method set differs")
    evidence = {}
    for method in sorted(expected_methods):
        verified = histories[method]
        if not isinstance(verified, dict):
            raise ValueError(f"{method} resume-history verification is malformed")
        row = verified.get("evidence", {})
        if row.get("status") != "verified":
            raise ValueError(
                f"{method} resume-history verification failed: "
                f"{verified.get('error', row.get('error', 'invalid evidence'))}"
            )
        if (
            row.get("complete_recovery_chain_verified") is not True
            or row.get("history_sha256_verified") is not True
            or row.get("manifest_journal_equality_verified") is not True
            or row.get("training_report_history_equality_verified") is not True
            or row.get("known_physical_report_bindings_verified") is not True
            or row.get("known_physical_orphan_bindings_verified") is not True
            or row.get("metrics_prefix_bindings_verified") is not True
            or row.get("current_manifest_binding_verified") is not True
            or row.get("current_training_report_binding_verified") is not True
            or row.get("checkpoint_payload_sha256_recomputed") is not False
        ):
            raise ValueError(f"{method} complete exact-resume history is not proven")
        report = training_reports[method]
        embedded = report.get("metrics_resume_history")
        if not isinstance(embedded, dict):
            raise ValueError(f"{method} training report lacks metrics resume history")
        if embedded.get("history_sha256") != row.get("history_sha256"):
            raise ValueError(f"{method} training report resume-history digest differs")
        output_dir = str(report.get("output_dir", "")).replace("\\", "/").rstrip("/")
        run_dir = str(row.get("run_dir", "")).replace("\\", "/").rstrip("/")
        if not output_dir or output_dir != run_dir:
            raise ValueError(f"{method} resume-history run directory differs")
        evidence[method] = {
            "run_dir": run_dir,
            "history_sha256": row["history_sha256"],
            "checkpoint_bound_event_count": int(
                row.get("checkpoint_bound_event_count", -1)
            ),
            "complete_recovery_chain_verified": True,
            "checkpoint_payload_sha256_recomputed": False,
        }
    return evidence


def _verify_resume_history_file(
    run_dir: Path,
    *,
    label: str,
    training_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return verify_metrics_resume_history_evidence(
            manifest_path=run_dir / "run_manifest.json",
            metrics_path=run_dir / "train_metrics.jsonl",
            training_report_path=run_dir / "training_report.json",
            training_report=training_report,
            require_complete=True,
            label=label,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "run_dir": run_dir.resolve().as_posix(),
            "error": str(error),
        }


def build_completion_audit(
    *,
    expected_10pct_revision: str,
    expected_full_revision: str,
    cofitok_10pct_training: dict[str, Any] | None,
    dense_10pct_training: dict[str, Any] | None,
    deployment_receipt: dict[str, Any] | None,
    deployment_verification_files: dict[str, dict[str, Any] | None],
    storage_preflights: dict[str, dict[str, Any] | None],
    expected_storage_path: str,
    full_training_monitor: dict[str, Any] | None,
    full_checkpoint_files: dict[str, dict[str, Any] | None],
    scaling_gate: dict[str, Any] | None,
    scaling_gate_source_verification: dict[str, Any] | None,
    cofitok_full_training: dict[str, Any] | None,
    dense_full_training: dict[str, Any] | None,
    cofitok_training_audit: dict[str, Any] | None,
    dense_training_audit: dict[str, Any] | None,
    runtime_selection: dict[str, Any] | None,
    sampling_runtime_selection: dict[str, Any] | None,
    visual_audit: dict[str, Any] | None,
    inference_exports: dict[str, dict[str, Any] | None],
    inference_artifact_files: dict[str, dict[str, Any] | None],
    inference_smoke_files: dict[str, dict[str, Any] | None],
    milestones: dict[int, dict[str, Any] | None],
    milestone_source_verifications: dict[int, dict[str, Any] | None],
    cofitok_generation: dict[str, Any] | None,
    dense_generation: dict[str, Any] | None,
    formal_sample_files: dict[str, dict[str, Any] | None],
    formal_real_set_files: dict[str, Any] | None,
    final_gate: dict[str, Any] | None,
    final_gate_source_verification: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
    comparison_source_verification: dict[str, Any] | None,
    official_related: dict[str, Any] | None,
    official_related_sha256: str | None,
    class_fidelity_qualification: dict[str, Any] | None = None,
    expected_deployment_source_revision: str | None = None,
    full_resume_histories: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    deployment_source_revision = (
        expected_deployment_source_revision or expected_10pct_revision
    )
    if (
        len(expected_10pct_revision) != 40
        or len(expected_full_revision) != 40
        or len(deployment_source_revision) != 40
    ):
        raise ValueError("completion audit requires full 40-character revisions")

    checks = []
    checks.append(
        _check(
            "ten_percent_matched_training",
            [cofitok_10pct_training, dense_10pct_training],
            lambda: validate_training_pair(
                cofitok_10pct_training,
                dense_10pct_training,
                expected_steps=50_000,
                expected_revision=expected_10pct_revision,
                expected_recipe_stage="scaling",
                allow_legacy_missing_dataset_provenance=False,
            ),
        )
    )
    checks.append(
        _check(
            "controlled_revision_transition",
            [deployment_receipt],
            lambda: _deployment_transition_evidence(
                deployment_receipt,
                expected_training_revision=deployment_source_revision,
                expected_target_revision=expected_full_revision,
                verification_files=deployment_verification_files,
            ),
        )
    )
    checks.append(
        _check(
            "generation_storage_capacity",
            list(storage_preflights.values()),
            lambda: _storage_capacity_evidence(
                storage_preflights,
                expected_revision=expected_full_revision,
                expected_path=expected_storage_path,
            ),
        )
    )
    checks.append(
        _check(
            "full_training_operational_monitor",
            [full_training_monitor],
            lambda: _full_training_monitor_evidence(
                full_training_monitor,
                expected_revision=expected_full_revision,
                expected_output_root=expected_storage_path,
            ),
        )
    )
    checks.append(
        _check(
            "scaling_promotion_gate",
            [scaling_gate, scaling_gate_source_verification],
            lambda: _gate_evidence(
                scaling_gate,
                scaling_gate_source_verification,
                stage="scaling",
                decision="promote_to_full_imagenet256",
            ),
        )
    )
    checks.append(
        _check(
            "full_matched_training",
            [
                cofitok_full_training,
                dense_full_training,
                (full_resume_histories or {}).get("cofitok"),
                (full_resume_histories or {}).get("dense_identity"),
            ],
            lambda: {
                "pair": validate_training_pair(
                    cofitok_full_training,
                    dense_full_training,
                    expected_steps=300_000,
                    expected_revision=expected_full_revision,
                    expected_dataset="imagenet_256",
                    expected_recipe_stage="full",
                    expected_authorization_gate=scaling_gate,
                ),
                "metrics_resume_history": _complete_resume_history_evidence(
                    {
                        "cofitok": full_resume_histories["cofitok"],
                        "dense_identity": full_resume_histories["dense_identity"],
                    },
                    {
                        "cofitok": cofitok_full_training,
                        "dense_identity": dense_full_training,
                    },
                ),
            },
        )
    )
    checks.append(
        _check(
            "full_training_runtime_environment",
            [cofitok_full_training, dense_full_training],
            lambda: _runtime_environment_evidence(
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                }
            ),
        )
    )
    checks.append(
        _check(
            "full_training_checkpoint_code_provenance",
            [cofitok_full_training, dense_full_training],
            lambda: _checkpoint_code_provenance_evidence(
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "reproducible_full_checkpoint_files",
            [
                full_checkpoint_files.get("cofitok"),
                full_checkpoint_files.get("dense_identity"),
                cofitok_full_training,
                dense_full_training,
            ],
            lambda: _full_checkpoint_file_evidence(
                full_checkpoint_files,
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
            ),
        )
    )
    checks.append(
        _check(
            "full_runtime_selection",
            [runtime_selection, cofitok_full_training, dense_full_training],
            lambda: _runtime_selection_evidence(
                runtime_selection,
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "full_training_audits",
            [cofitok_training_audit, dense_training_audit],
            lambda: _training_audit_evidence(
                {
                    "cofitok": cofitok_training_audit,
                    "dense_identity": dense_training_audit,
                }
            ),
        )
    )

    milestone_payloads = [milestones.get(step) for step in MILESTONE_STEPS] + [
        milestone_source_verifications.get(step) for step in MILESTONE_STEPS
    ]
    milestone_warnings: list[str] = []

    def validate_milestones() -> dict[str, Any]:
        evidence, warnings = _milestone_evidence(
            {step: milestones[step] for step in MILESTONE_STEPS},
            {
                step: milestone_source_verifications[step]
                for step in MILESTONE_STEPS
            },
        )
        milestone_warnings.extend(warnings)
        return evidence

    checks.append(
        _check("full_milestone_evaluations", milestone_payloads, validate_milestones)
    )
    checks.append(
        _check(
            "formal_50k_generation",
            [
                cofitok_generation,
                dense_generation,
                formal_sample_files.get("cofitok"),
                formal_sample_files.get("dense_identity"),
                formal_real_set_files,
                cofitok_full_training,
                dense_full_training,
            ],
            lambda: _generation_evidence(
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
                formal_sample_files,
                formal_real_set_files,
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "formal_sampling_runtime_selection",
            [sampling_runtime_selection, cofitok_generation, dense_generation],
            lambda: _sampling_runtime_selection_evidence(
                sampling_runtime_selection,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "deterministic_visual_quality_audit",
            [visual_audit, cofitok_generation, dense_generation],
            lambda: _visual_audit_evidence(
                visual_audit,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                expected_revision=expected_full_revision,
            ),
        )
    )
    checks.append(
        _check(
            "deployable_ema_inference_artifacts",
            [
                inference_exports.get(name)
                for name in (
                    "cofitok_export",
                    "dense_identity_export",
                    "cofitok_preflight",
                    "dense_identity_preflight",
                    "cofitok_smoke",
                    "dense_identity_smoke",
                )
            ]
            + [
                inference_artifact_files.get("cofitok"),
                inference_artifact_files.get("dense_identity"),
                inference_smoke_files.get("cofitok"),
                inference_smoke_files.get("dense_identity"),
                cofitok_full_training,
                dense_full_training,
                final_gate,
            ],
            lambda: _inference_export_evidence(
                inference_exports,
                inference_artifact_files,
                inference_smoke_files,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
                final_gate,
            ),
        )
    )
    checks.append(
        _check(
            "final_generation_gate",
            [final_gate, final_gate_source_verification],
            lambda: _final_gate_evidence(
                final_gate,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                final_gate_source_verification,
            ),
        )
    )
    checks.append(
        _check(
            "final_comparison_report",
            [
                comparison,
                comparison_source_verification,
                official_related,
                official_related_sha256,
                cofitok_full_training,
                dense_full_training,
                full_training_monitor,
                final_gate,
            ]
            + (
                [class_fidelity_qualification]
                if isinstance(comparison, dict)
                and comparison.get("source_profile") == "stability_full"
                else []
            ),
            lambda: _comparison_evidence(
                comparison,
                {"cofitok": cofitok_generation, "dense_identity": dense_generation},
                {
                    "cofitok": cofitok_full_training,
                    "dense_identity": dense_full_training,
                },
                full_training_monitor,
                official_related,
                official_related_sha256,
                comparison_source_verification,
                final_gate=final_gate,
                class_fidelity_qualification=class_fidelity_qualification,
            ),
        )
    )

    failed = [row["name"] for row in checks if row["status"] == "fail"]
    missing = [row["name"] for row in checks if row["status"] == "missing"]
    complete = not failed and not missing
    return {
        "schema_version": 1,
        "status": "complete" if complete else ("failed" if failed else "in_progress"),
        "complete": complete,
        "expected_revisions": {
            "deployment_source": deployment_source_revision,
            "ten_percent_training": expected_10pct_revision,
            "full_training": expected_full_revision,
        },
        "checks": checks,
        "failed_checks": failed,
        "missing_checks": missing,
        "warnings": milestone_warnings,
    }


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _verify_deployment_json_source(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        content = _read_optional(path)
        if not isinstance(content, dict):
            raise ValueError("deployment verification JSON must contain an object")
        return {
            "status": "verified",
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "content": content,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "path": path.resolve().as_posix(),
            "error": str(error),
        }


def _verify_deployment_pytest_source(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return {
            "status": "verified",
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "summary": pytest_junit_summary(path),
        }
    except (OSError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "path": path.resolve().as_posix(),
            "error": str(error),
        }


def _verify_deployment_bundle_source(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = subprocess.run(
            ["git", "bundle", "list-heads", path.as_posix()],
            check=True,
            capture_output=True,
            text=True,
        )
        heads = sorted(
            {line.split()[0] for line in result.stdout.splitlines() if line.split()}
        )
        if not heads:
            raise ValueError("deployment bundle has no advertised heads")
        return {
            "status": "verified",
            "path": path.resolve().as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "heads": heads,
            "prerequisites": bundle_prerequisites(path),
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        return {
            "status": "invalid",
            "path": path.resolve().as_posix(),
            "error": str(error),
        }


def _verify_milestone_sources_optional(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    try:
        return verify_milestone_source_reports(report)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"status": "invalid", "error": str(error)}


def _verify_comparison_sources_optional(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    try:
        return verify_comparison_source_reports(report)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"status": "invalid", "error": str(error)}


def _verify_gate_sources_optional(
    gate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if gate is None:
        return None
    try:
        return verify_generation_gate_source_reports(gate)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"status": "invalid", "error": str(error)}


def _verify_checkpoint_file(path: Path) -> dict[str, Any]:
    try:
        integrity = verify_training_checkpoint(path)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "path": path.resolve().as_posix(),
            "error": str(error),
        }
    return {
        "status": "verified",
        "path": path.resolve().as_posix(),
        "integrity_manifest": checkpoint_integrity_path(path).resolve().as_posix(),
        **integrity,
    }


def _verify_inference_artifact_file(path: Path) -> dict[str, Any]:
    try:
        integrity = verify_inference_artifact(path)
        manifest_path = inference_export_manifest_path(path)
        export_manifest_payload = verify_inference_export_manifest(
            manifest_path,
            expected_artifact=path,
        )
        export_manifest = {
            "path": manifest_path.resolve().as_posix(),
            "bytes": manifest_path.stat().st_size,
            "sha256": file_sha256(manifest_path),
        }
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {
            "status": "invalid",
            "path": path.resolve().as_posix(),
            "error": str(error),
        }
    return {
        "status": "verified",
        "path": path.resolve().as_posix(),
        "integrity_manifest": checkpoint_integrity_path(path).resolve().as_posix(),
        "export_manifest": export_manifest,
        "export_manifest_payload": export_manifest_payload,
        **integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit end-to-end completion of large-scale CoFiTok generation."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/CoFiTok/checkpoints/generation",
    )
    parser.add_argument(
        "--expected-deployment-source-revision",
        default=PINNED_10PCT_REVISION,
    )
    parser.add_argument("--expected-10pct-revision", required=True)
    parser.add_argument("--expected-full-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    workspace = generation_workspace_paths(
        project_root=project,
        output_root=output_root,
    )
    ten_root = workspace["SCALING_REPORT_ROOT"]
    full_root = workspace["FULL_REPORT_ROOT"]
    cofitok_10 = workspace["SCALING_COFITOK_RUN"]
    dense_10 = workspace["SCALING_DENSE_RUN"]
    cofitok_full = workspace["FULL_COFITOK_RUN"]
    dense_full = workspace["FULL_DENSE_RUN"]
    official_related_path = (
        project
        / "artifacts/reports/baselines/official_related_methods_2026-07-11_final"
        / "official_related_methods_table.json"
    )
    milestones = {
        step: _read_optional(full_root / "milestones" / f"step_{step:08d}.json")
        for step in MILESTONE_STEPS
    }
    milestone_source_verifications = {
        step: _verify_milestone_sources_optional(report)
        for step, report in milestones.items()
    }
    comparison_path = full_root / "comparison/large_scale_generation_comparison.json"
    comparison = _read_optional(comparison_path)
    scaling_gate = _read_optional(ten_root / "promotion_gate.json")
    final_gate = _read_optional(full_root / "final_generation_gate.json")
    deployment = generation_deployment_attestation_paths(
        output_root=output_root,
        target_revision=args.expected_full_revision,
    )
    deployment_receipt = _read_optional(deployment["DEPLOYMENT_RECEIPT"])
    inference_exports = {
        "cofitok_export": _read_optional(
            full_root / "exports/cofitok_export_report.json"
        ),
        "dense_identity_export": _read_optional(
            full_root / "exports/dense_export_report.json"
        ),
        "cofitok_preflight": _read_optional(
            full_root / "exports/cofitok_export_preflight.json"
        ),
        "dense_identity_preflight": _read_optional(
            full_root / "exports/dense_export_preflight.json"
        ),
        "cofitok_smoke": _read_optional(
            full_root / "exports/cofitok_export_inference_smoke.json"
        ),
        "dense_identity_smoke": _read_optional(
            full_root / "exports/dense_export_inference_smoke.json"
        ),
    }
    cofitok_generation = _read_optional(
        cofitok_full
        / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
    )
    dense_generation = _read_optional(
        dense_full
        / "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
    )
    cofitok_full_training = _read_optional(cofitok_full / "training_report.json")
    dense_full_training = _read_optional(dense_full / "training_report.json")

    audit = build_completion_audit(
        expected_deployment_source_revision=args.expected_deployment_source_revision,
        expected_10pct_revision=args.expected_10pct_revision,
        expected_full_revision=args.expected_full_revision,
        cofitok_10pct_training=_read_optional(cofitok_10 / "training_report.json"),
        dense_10pct_training=_read_optional(dense_10 / "training_report.json"),
        deployment_receipt=deployment_receipt,
        deployment_verification_files={
            "bundle": _verify_deployment_bundle_source(
                deployment["DEPLOYMENT_BUNDLE"]
            ),
            "conflict_scan": _verify_deployment_json_source(
                deployment["DEPLOYMENT_CONFLICT_SCAN"]
            ),
            "runbook_syntax": _verify_deployment_json_source(
                deployment["DEPLOYMENT_RUNBOOK_SYNTAX"]
            ),
            "pytest": _verify_deployment_pytest_source(
                deployment["DEPLOYMENT_PYTEST"]
            ),
        },
        storage_preflights={
            "10pct_posteval": _read_optional(ten_root / "storage_preflight.json"),
            "full_training": _read_optional(
                full_root / "storage_preflight_training.json"
            ),
            "full_posteval": _read_optional(
                full_root / "storage_preflight_posteval.json"
            ),
        },
        expected_storage_path=output_root.as_posix(),
        full_training_monitor=_read_optional(
            output_root / "generation_full_matched_300k_monitor.json"
        ),
        full_checkpoint_files={
            "cofitok": _verify_checkpoint_file(
                cofitok_full / "checkpoint_step_00300000.pt"
            ),
            "dense_identity": _verify_checkpoint_file(
                dense_full / "checkpoint_step_00300000.pt"
            ),
        },
        scaling_gate=scaling_gate,
        scaling_gate_source_verification=_verify_gate_sources_optional(
            scaling_gate
        ),
        cofitok_full_training=cofitok_full_training,
        dense_full_training=dense_full_training,
        full_resume_histories={
            "cofitok": _verify_resume_history_file(
                cofitok_full,
                label="cofitok full training",
                training_report=cofitok_full_training,
            ),
            "dense_identity": _verify_resume_history_file(
                dense_full,
                label="dense full training",
                training_report=dense_full_training,
            ),
        },
        cofitok_training_audit=_read_optional(full_root / "cofitok_training_audit.json"),
        dense_training_audit=_read_optional(full_root / "dense_training_audit.json"),
        runtime_selection=_read_optional(full_root / "runtime_selection.json"),
        sampling_runtime_selection=_read_optional(
            full_root / "sampling_runtime_selection.json"
        ),
        visual_audit=_read_optional(full_root / "visual_audit/visual_audit_report.json"),
        inference_exports=inference_exports,
        inference_artifact_files={
            "cofitok": _verify_inference_artifact_file(
                output_root
                / "exports/imagenet256_full_300k/cofitok_k8_ema_inference.pt"
            ),
            "dense_identity": _verify_inference_artifact_file(
                output_root
                / "exports/imagenet256_full_300k/dense_identity_ema_inference.pt"
            ),
        },
        inference_smoke_files={
            "cofitok": _verify_inference_smoke_outputs(
                inference_exports["cofitok_smoke"],
                expected_root=(
                    output_root / "exports/imagenet256_full_300k/smoke/cofitok"
                ),
            ),
            "dense_identity": _verify_inference_smoke_outputs(
                inference_exports["dense_identity_smoke"],
                expected_root=(
                    output_root
                    / "exports/imagenet256_full_300k/smoke/dense_identity"
                ),
            ),
        },
        milestones=milestones,
        milestone_source_verifications=milestone_source_verifications,
        cofitok_generation=cofitok_generation,
        dense_generation=dense_generation,
        formal_sample_files={
            "cofitok": _verify_formal_sample_files(
                cofitok_generation,
                expected_generated_dir=(
                    cofitok_full / "samples_50k_ddim250_cfg15/prefix_8"
                ),
            ),
            "dense_identity": _verify_formal_sample_files(
                dense_generation,
                expected_generated_dir=(
                    dense_full / "samples_50k_ddim250_cfg15/prefix_1"
                ),
            ),
        },
        formal_real_set_files=_verify_formal_real_set_files(
            {
                "cofitok": cofitok_generation,
                "dense_identity": dense_generation,
            },
            expected_real_dir=FORMAL_IMAGENET256_REAL_DIR,
        ),
        final_gate=final_gate,
        final_gate_source_verification=_verify_gate_sources_optional(final_gate),
        comparison=comparison,
        comparison_source_verification=_verify_comparison_sources_optional(comparison),
        class_fidelity_qualification=_read_optional(
            full_root / "class_fidelity/qualification_report.json"
        ),
        official_related=_read_optional(official_related_path),
        official_related_sha256=(
            file_sha256(official_related_path) if official_related_path.is_file() else None
        ),
    )
    write_json_report(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["complete"] and not args.allow_incomplete:
        raise SystemExit("large-scale generation completion audit did not pass")


if __name__ == "__main__":
    main()
