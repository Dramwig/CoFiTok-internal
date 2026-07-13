from __future__ import annotations

from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256


GATE_SOURCE_SUFFIXES = {
    "scaling": {
        "cofitok_training": (
            "checkpoints/generation/imagenet256_10pct_cofitok_k8_50k_2026-07-12/"
            "training_report.json"
        ),
        "dense_training": (
            "checkpoints/generation/imagenet256_10pct_dense_50k_2026-07-12/"
            "training_report.json"
        ),
        "cofitok_generation": (
            "checkpoints/generation/imagenet256_10pct_cofitok_k8_50k_2026-07-12/"
            "samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            "checkpoints/generation/imagenet256_10pct_dense_50k_2026-07-12/"
            "samples_gate10k_ddim100_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            "checkpoints/generation/imagenet256_10pct_cofitok_k8_50k_2026-07-12/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            "checkpoints/generation/imagenet256_10pct_dense_50k_2026-07-12/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
    "full": {
        "cofitok_training": (
            "checkpoints/generation/imagenet256_full_cofitok_k8_300k/"
            "training_report.json"
        ),
        "dense_training": (
            "checkpoints/generation/imagenet256_full_dense_300k/training_report.json"
        ),
        "cofitok_generation": (
            "checkpoints/generation/imagenet256_full_cofitok_k8_300k/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "dense_generation": (
            "checkpoints/generation/imagenet256_full_dense_300k/"
            "samples_50k_ddim250_cfg15/metrics/generation_metrics_report.json"
        ),
        "cofitok_checkpoint_eval": (
            "checkpoints/generation/imagenet256_full_cofitok_k8_300k/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
        "dense_checkpoint_eval": (
            "checkpoints/generation/imagenet256_full_dense_300k/"
            "checkpoint_eval_ema_t500_1024/checkpoint_evaluation_report.json"
        ),
    },
}


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
    stage: str,
) -> None:
    expected = GATE_SOURCE_SUFFIXES.get(stage)
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
) -> dict[str, dict[str, Any]]:
    expected = GATE_SOURCE_SUFFIXES.get(stage)
    if expected is None or set(paths) != set(expected):
        raise ValueError("generation gate source paths are incomplete")
    reports = {
        name: gate_source_report_identity(path) for name, path in paths.items()
    }
    _validate_gate_source_report_identities(reports, stage=stage)
    return reports


def verify_generation_gate_source_reports(
    gate: dict[str, Any],
) -> dict[str, Any]:
    stage = str(gate.get("stage", ""))
    source_reports = gate.get("source_reports")
    if not isinstance(source_reports, dict):
        raise ValueError("generation gate is missing source-report identities")
    _validate_gate_source_report_identities(source_reports, stage=stage)
    verified = {}
    for name, expected in source_reports.items():
        actual = gate_source_report_identity(expected["path"])
        if actual != expected:
            raise ValueError(
                f"generation gate source report changed after binding: {name}"
            )
        verified[name] = actual
    return {"status": "verified", "stage": stage, "source_reports": verified}
