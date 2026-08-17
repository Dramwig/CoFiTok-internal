from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cofitok.generation_cost import (
    RESUME_COMPUTE_ADJUSTMENT_METHODS,
    resume_compute_adjustment_source_identity,
    training_cost_summary,
    validate_resume_compute_adjustment_source_identities,
    verify_resume_compute_adjustment_source,
)
from cofitok.generation_paths import (
    FULL_COFITOK_RUN_ID,
    FULL_DENSE_RUN_ID,
    SCALING_COFITOK_RUN_ID,
    SCALING_DENSE_RUN_ID,
    STABILITY_SCALING_COFITOK_RUN_ID,
    STABILITY_SCALING_DENSE_RUN_ID,
    STABILITY_SCALING_ROOT_ID,
    STABILITY_FULL_COFITOK_RUN_ID,
    STABILITY_FULL_DENSE_RUN_ID,
    STABILITY_FULL_ROOT_ID,
)
from cofitok.reporting import file_sha256


GATE_SOURCE_SUFFIXES = {
    "scaling": {
        "cofitok_training": (
            f"checkpoints/generation/{SCALING_COFITOK_RUN_ID}/"
            "training_report.json"
        ),
        "dense_training": (
            f"checkpoints/generation/{SCALING_DENSE_RUN_ID}/"
            "training_report.json"
        ),
        "cofitok_generation": (
            f"checkpoints/generation/{SCALING_COFITOK_RUN_ID}/"
            "samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            f"checkpoints/generation/{SCALING_DENSE_RUN_ID}/"
            "samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            f"checkpoints/generation/{SCALING_COFITOK_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            f"checkpoints/generation/{SCALING_DENSE_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
    "stability_scaling": {
        "cofitok_training": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_COFITOK_RUN_ID}/training_report.json"
        ),
        "dense_training": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_DENSE_RUN_ID}/training_report.json"
        ),
        "cofitok_generation": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_COFITOK_RUN_ID}/"
            "samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_DENSE_RUN_ID}/"
            "samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_COFITOK_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_DENSE_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
    "full": {
        "cofitok_training": (
            f"checkpoints/generation/{FULL_COFITOK_RUN_ID}/"
            "training_report.json"
        ),
        "dense_training": (
            f"checkpoints/generation/{FULL_DENSE_RUN_ID}/training_report.json"
        ),
        "cofitok_generation": (
            f"checkpoints/generation/{FULL_COFITOK_RUN_ID}/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            f"checkpoints/generation/{FULL_DENSE_RUN_ID}/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            f"checkpoints/generation/{FULL_COFITOK_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            f"checkpoints/generation/{FULL_DENSE_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
    "stability_full": {
        "cofitok_training": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_COFITOK_RUN_ID}/training_report.json"
        ),
        "dense_training": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_DENSE_RUN_ID}/training_report.json"
        ),
        "cofitok_generation": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_COFITOK_RUN_ID}/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_DENSE_RUN_ID}/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_COFITOK_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_DENSE_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
}
GATE_SOURCE_PROFILE_STAGES = {
    "scaling": "scaling",
    "stability_scaling": "scaling",
    "full": "full",
    "stability_full": "full",
}
GATE_DIAGNOSTIC_SUFFIXES = {
    "stability_scaling": {
        "rollout_stability_qualification": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/reports/"
            "ema_rollout_stability/qualification_report.json"
        ),
        "cofitok_class_fidelity": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_COFITOK_RUN_ID}/samples_gate10k_ddim100_cfg15/"
            "class_fidelity/class_fidelity_report.json"
        ),
        "dense_class_fidelity": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/"
            f"{STABILITY_SCALING_DENSE_RUN_ID}/samples_gate10k_ddim100_cfg15/"
            "class_fidelity/class_fidelity_report.json"
        ),
        "class_fidelity_qualification": (
            f"checkpoints/generation/{STABILITY_SCALING_ROOT_ID}/reports/"
            "class_fidelity/qualification_report.json"
        ),
    },
    "stability_full": {
        "rollout_stability_qualification": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/reports/"
            "ema_rollout_stability/qualification_report.json"
        ),
        "cofitok_class_fidelity": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_COFITOK_RUN_ID}/samples_50k_ddim250_cfg15/"
            "class_fidelity/class_fidelity_report.json"
        ),
        "dense_class_fidelity": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/"
            f"{STABILITY_FULL_DENSE_RUN_ID}/samples_50k_ddim250_cfg15/"
            "class_fidelity/class_fidelity_report.json"
        ),
        "class_fidelity_qualification": (
            f"checkpoints/generation/{STABILITY_FULL_ROOT_ID}/reports/"
            "class_fidelity/qualification_report.json"
        ),
    },
}
LEGACY_GATE_DIAGNOSTIC_SUFFIXES = {
    profile: {"rollout_stability_qualification": rows["rollout_stability_qualification"]}
    for profile, rows in GATE_DIAGNOSTIC_SUFFIXES.items()
}


def _diagnostic_suffixes(
    profile: str,
    *,
    schema_version: int | None,
) -> dict[str, str] | None:
    if schema_version is not None and schema_version < 5:
        return LEGACY_GATE_DIAGNOSTIC_SUFFIXES.get(profile)
    return GATE_DIAGNOSTIC_SUFFIXES.get(profile)


def gate_source_report_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"generation gate source report does not exist: {source}"
        )
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def build_generation_gate_resume_compute_adjustments(
    paths: dict[str, str | Path],
) -> dict[str, dict[str, Any]]:
    if not set(paths).issubset(RESUME_COMPUTE_ADJUSTMENT_METHODS):
        raise ValueError("generation gate resume-compute methods are invalid")
    identities = {
        method: resume_compute_adjustment_source_identity(path)
        for method, path in paths.items()
    }
    validate_resume_compute_adjustment_source_identities(identities)
    return identities


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _validate_gate_source_report_identities(
    source_reports: dict[str, dict[str, Any]],
    *,
    profile: str,
) -> None:
    expected = GATE_SOURCE_SUFFIXES.get(profile)
    if expected is None or set(source_reports) != set(expected):
        raise ValueError("generation gate source-report set is incomplete")
    for name, suffix in expected.items():
        identity = source_reports[name]
        normalized_path = str(identity.get("path", "")).replace("\\", "/")
        sha256 = str(identity.get("sha256", ""))
        if (
            not normalized_path.endswith(suffix)
            or int(identity.get("bytes", 0)) < 1
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                f"generation gate source-report identity is invalid: {name}"
            )


def build_generation_gate_source_reports(
    *,
    stage: str,
    paths: dict[str, str | Path],
    profile: str | None = None,
) -> dict[str, dict[str, Any]]:
    source_profile = profile or stage
    if GATE_SOURCE_PROFILE_STAGES.get(source_profile) != stage:
        raise ValueError(
            f"generation gate source profile {source_profile} is incompatible "
            f"with stage {stage}"
        )
    expected = GATE_SOURCE_SUFFIXES.get(source_profile)
    if expected is None or set(paths) != set(expected):
        raise ValueError("generation gate source paths are incomplete")
    reports = {
        name: gate_source_report_identity(path) for name, path in paths.items()
    }
    _validate_gate_source_report_identities(reports, profile=source_profile)
    return reports


def _validate_gate_diagnostic_report_identities(
    diagnostic_reports: dict[str, dict[str, Any]],
    *,
    profile: str,
    schema_version: int | None,
) -> None:
    expected = _diagnostic_suffixes(profile, schema_version=schema_version)
    if expected is None or set(diagnostic_reports) != set(expected):
        raise ValueError("generation gate diagnostic-report set is incomplete")
    for name, suffix in expected.items():
        identity = diagnostic_reports[name]
        normalized_path = str(identity.get("path", "")).replace("\\", "/")
        sha256 = str(identity.get("sha256", ""))
        if (
            not normalized_path.endswith(suffix)
            or int(identity.get("bytes", 0)) < 1
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                f"generation gate diagnostic-report identity is invalid: {name}"
            )


def build_generation_gate_diagnostic_reports(
    *,
    profile: str,
    paths: dict[str, str | Path],
    schema_version: int | None = None,
) -> dict[str, dict[str, Any]]:
    if schema_version is None:
        latest = GATE_DIAGNOSTIC_SUFFIXES.get(profile)
        legacy = LEGACY_GATE_DIAGNOSTIC_SUFFIXES.get(profile)
        if latest is not None and set(paths) == set(latest):
            schema_version = 5
        elif legacy is not None and set(paths) == set(legacy):
            schema_version = 4
    expected = _diagnostic_suffixes(profile, schema_version=schema_version)
    if expected is None or set(paths) != set(expected):
        raise ValueError("generation gate diagnostic paths are incomplete")
    reports = {
        name: gate_source_report_identity(path) for name, path in paths.items()
    }
    _validate_gate_diagnostic_report_identities(
        reports,
        profile=profile,
        schema_version=schema_version,
    )
    return reports


def verify_generation_gate_source_reports(
    gate: dict[str, Any],
) -> dict[str, Any]:
    schema_version = int(gate.get("schema_version", 0) or 0)
    stage = str(gate.get("stage", ""))
    source_profile = str(gate.get("source_profile", stage))
    if GATE_SOURCE_PROFILE_STAGES.get(source_profile) != stage:
        raise ValueError(
            f"generation gate source profile {source_profile} is incompatible "
            f"with stage {stage}"
        )
    source_reports = gate.get("source_reports")
    if not isinstance(source_reports, dict):
        raise ValueError("generation gate is missing source-report identities")
    _validate_gate_source_report_identities(
        source_reports,
        profile=source_profile,
    )
    verified = {}
    for name, expected in source_reports.items():
        actual = gate_source_report_identity(expected["path"])
        if actual != expected:
            raise ValueError(
                f"generation gate source report changed after binding: {name}"
            )
        verified[name] = actual
    result = {
        "status": "verified",
        "stage": stage,
        "source_profile": source_profile,
        "source_reports": verified,
    }
    diagnostic_reports = gate.get("diagnostic_reports")
    if diagnostic_reports is not None:
        if not isinstance(diagnostic_reports, dict):
            raise ValueError("generation gate diagnostic-report identities are malformed")
        raw_schema_version = gate.get("schema_version")
        diagnostic_schema_version = (
            int(raw_schema_version) if raw_schema_version is not None else None
        )
        _validate_gate_diagnostic_report_identities(
            diagnostic_reports,
            profile=source_profile,
            schema_version=diagnostic_schema_version,
        )
        verified_diagnostics = {}
        for name, expected in diagnostic_reports.items():
            actual = gate_source_report_identity(expected["path"])
            if actual != expected:
                raise ValueError(
                    f"generation gate diagnostic report changed after binding: {name}"
                )
            verified_diagnostics[name] = actual
        result["diagnostic_reports"] = verified_diagnostics
    if schema_version >= 6:
        adjustments = gate.get("resume_compute_adjustments")
        if not isinstance(adjustments, dict):
            raise ValueError(
                "generation gate resume-compute adjustment bindings are missing"
            )
        validate_resume_compute_adjustment_source_identities(adjustments)
        training_reports = {
            "cofitok": _read_json_object(
                verified["cofitok_training"]["path"],
                label="CoFiTok training report",
            ),
            "dense_identity": _read_json_object(
                verified["dense_training"]["path"],
                label="dense training report",
            ),
        }
        verified_adjustments: dict[str, dict[str, Any]] = {}
        training_costs: dict[str, dict[str, Any]] = {}
        for method, training_report in training_reports.items():
            identity = adjustments.get(method)
            if identity is None:
                cost = training_cost_summary(training_report)
                if cost["valid"] is not True:
                    raise ValueError(
                        f"generation gate {method} training cost requires a "
                        "resume-compute adjustment"
                    )
            else:
                verification = verify_resume_compute_adjustment_source(
                    identity,
                    method=method,
                    training_report=training_report,
                )
                verified_adjustments[method] = verification
                cost = verification["training_cost"]
            training_costs[method] = cost

        summary = gate.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        if (
            summary.get("cofitok_training_cost") != training_costs["cofitok"]
            or summary.get("dense_training_cost")
            != training_costs["dense_identity"]
        ):
            raise ValueError(
                "generation gate training-cost summary differs from physical sources"
            )
        raw_gates = gate.get("gates")
        if not isinstance(raw_gates, list):
            raise ValueError("generation gate checks are malformed")
        cost_rows = [
            row
            for row in raw_gates
            if isinstance(row, dict)
            and row.get("name") == "training_cost_accounting"
        ]
        if (
            len(cost_rows) != 1
            or cost_rows[0].get("evidence")
            != {
                "cofitok": training_costs["cofitok"],
                "dense": training_costs["dense_identity"],
            }
        ):
            raise ValueError(
                "generation gate training-cost evidence differs from physical sources"
            )
        result["resume_compute_adjustments"] = verified_adjustments
        result["training_costs"] = training_costs
    return result
