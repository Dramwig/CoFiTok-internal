from __future__ import annotations

from pathlib import Path
from typing import Any

from cofitok.generation_paths import (
    CAPACITY_FULL_COFITOK_RUN_ID,
    CAPACITY_FULL_DENSE_RUN_ID,
    CAPACITY_FULL_ROOT_ID,
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
    "capacity_full": {
        "cofitok_training": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_COFITOK_RUN_ID}/training_report.json"
        ),
        "dense_training": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_DENSE_RUN_ID}/training_report.json"
        ),
        "cofitok_generation": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_COFITOK_RUN_ID}/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_DENSE_RUN_ID}/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_COFITOK_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_DENSE_RUN_ID}/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
}
GATE_SOURCE_PROFILE_STAGES = {
    "scaling": "scaling",
    "stability_scaling": "scaling",
    "full": "full",
    "stability_full": "full",
    "capacity_full": "full",
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
    "capacity_full": {
        "rollout_stability_qualification": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/reports/"
            "ema_rollout_stability/qualification_report.json"
        ),
        "cofitok_class_fidelity": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_COFITOK_RUN_ID}/samples_50k_ddim250_cfg15/"
            "class_fidelity/class_fidelity_report.json"
        ),
        "dense_class_fidelity": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/"
            f"{CAPACITY_FULL_DENSE_RUN_ID}/samples_50k_ddim250_cfg15/"
            "class_fidelity/class_fidelity_report.json"
        ),
        "class_fidelity_qualification": (
            f"checkpoints/generation/{CAPACITY_FULL_ROOT_ID}/reports/"
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
    return result
