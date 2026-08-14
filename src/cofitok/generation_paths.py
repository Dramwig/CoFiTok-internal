from __future__ import annotations

import string
from pathlib import Path


SCALING_COFITOK_RUN_ID = "imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3"
SCALING_DENSE_RUN_ID = "imagenet256_10pct_fixed_basis_dense_50k_v3"
SCALING_REPORT_ID = "imagenet256_10pct_fixed_basis_matched_50k_v3"
STABILITY_PROBE_ROOT_ID = "stability_probe_2026-07-29"
STABILITY_DECISION_ID = "scaling_decision5k_rollout_x0_u2_ema_teacher_to_50k"
STABILITY_SCALING_ROOT_ID = "stability_scaling_50k_ema_teacher"
STABILITY_SCALING_COFITOK_RUN_ID = "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
STABILITY_SCALING_DENSE_RUN_ID = "dense_rollout_x0_u2_ema_teacher"
STABILITY_QUALITY_BRIDGE_ROOT_ID = "stability_full_data_100k_base128_quality_bridge_v1"
STABILITY_QUALITY_BRIDGE_COFITOK_RUN_ID = (
    "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
)
STABILITY_QUALITY_BRIDGE_DENSE_RUN_ID = "dense_rollout_x0_u2_ema_teacher"
STABILITY_FULL_ROOT_ID = "stability_full_300k_ema_teacher"
STABILITY_FULL_COFITOK_RUN_ID = "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
STABILITY_FULL_DENSE_RUN_ID = "dense_rollout_x0_u2_ema_teacher"
STABILITY_EXPORT_ROOT_ID = "exports/stability_full_300k_ema_teacher"
CAPACITY_FULL_ROOT_ID = "stability_capacity_full_300k_v1"
CAPACITY_FULL_COFITOK_RUN_ID = "cofitok"
CAPACITY_FULL_DENSE_RUN_ID = "dense_identity"
CAPACITY_FULL_EXPORT_ROOT_ID = "exports/stability_capacity_full_300k_v1"
STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT_ID = "deployment/large_capacity"
FULL_COFITOK_RUN_ID = "imagenet256_full_cofitok_k8_300k"
FULL_DENSE_RUN_ID = "imagenet256_full_dense_300k"
FULL_REPORT_ID = "imagenet256_full_matched_300k"


def _full_git_revision(value: str) -> str:
    revision = value.strip().lower()
    if len(revision) != 40 or any(char not in string.hexdigits for char in revision):
        raise ValueError("deployment target revision must be a full Git SHA-1")
    return revision


def generation_workspace_paths(
    *,
    project_root: str | Path,
    output_root: str | Path,
) -> dict[str, Path]:
    project = Path(project_root).resolve()
    output = Path(output_root).resolve()
    report_root = project / "artifacts/reports/generation"
    scaling_report = report_root / SCALING_REPORT_ID
    full_report = report_root / FULL_REPORT_ID
    return {
        "SCALING_COFITOK_RUN": output / SCALING_COFITOK_RUN_ID,
        "SCALING_DENSE_RUN": output / SCALING_DENSE_RUN_ID,
        "SCALING_REPORT_ROOT": scaling_report,
        "SCALING_GATE": scaling_report / "promotion_gate.json",
        "FULL_COFITOK_RUN": output / FULL_COFITOK_RUN_ID,
        "FULL_DENSE_RUN": output / FULL_DENSE_RUN_ID,
        "FULL_REPORT_ROOT": full_report,
        "FULL_GATE": full_report / "final_generation_gate.json",
        "STABILITY_FULL_COFITOK_RUN": (
            output / STABILITY_FULL_ROOT_ID / STABILITY_FULL_COFITOK_RUN_ID
        ),
        "STABILITY_FULL_DENSE_RUN": (
            output / STABILITY_FULL_ROOT_ID / STABILITY_FULL_DENSE_RUN_ID
        ),
        "STABILITY_FULL_REPORT_ROOT": output / STABILITY_FULL_ROOT_ID / "reports",
        "STABILITY_FULL_GATE": (
            output / STABILITY_FULL_ROOT_ID / "reports/final_generation_gate.json"
        ),
    }


def generation_deployment_attestation_paths(
    *,
    output_root: str | Path,
    target_revision: str,
) -> dict[str, Path]:
    output = Path(output_root).resolve()
    revision = _full_git_revision(target_revision)
    evidence_root = output / "deployment"
    stem = f"generation-upgrade-{revision}"
    return {
        "DEPLOYMENT_EVIDENCE_ROOT": evidence_root,
        "DEPLOYMENT_BUNDLE": (
            evidence_root / f"cofitok-generation-upgrade-{revision}.bundle"
        ),
        "DEPLOYMENT_RECEIPT": evidence_root / f"{stem}.receipt.json",
        "DEPLOYMENT_CONFLICT_SCAN": evidence_root / f"{stem}.conflict-scan.json",
        "DEPLOYMENT_RUNBOOK_SYNTAX": (
            evidence_root / f"{stem}.runbook-syntax.json"
        ),
        "DEPLOYMENT_PYTEST": evidence_root / f"{stem}.pytest.xml",
    }


def generation_large_capacity_deployment_paths(
    *,
    output_root: str | Path,
    target_revision: str,
) -> dict[str, Path]:
    output = Path(output_root).resolve()
    revision = _full_git_revision(target_revision)
    deployment_root = output / STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT_ID
    evidence_root = deployment_root / "deployments" / revision
    short_revision = revision[:7]
    return {
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT": deployment_root,
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_EVIDENCE_ROOT": evidence_root,
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_CHECKOUT": (
            deployment_root / f"checkout-{short_revision}"
        ),
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_RECEIPT": (
            evidence_root / "deployment_receipt.json"
        ),
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_RUNBOOK_SYNTAX": (
            evidence_root / "runbook_syntax.json"
        ),
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_PYTEST": evidence_root / "pytest.xml",
    }


def generation_stability_workspace_paths(
    *,
    output_root: str | Path,
) -> dict[str, Path]:
    output = Path(output_root).resolve()
    probe_root = output / STABILITY_PROBE_ROOT_ID
    scaling_root = output / STABILITY_SCALING_ROOT_ID
    scaling_reports = scaling_root / "reports"
    scaling_cofitok = scaling_root / STABILITY_SCALING_COFITOK_RUN_ID
    scaling_dense = scaling_root / STABILITY_SCALING_DENSE_RUN_ID
    quality_bridge_root = output / STABILITY_QUALITY_BRIDGE_ROOT_ID
    quality_bridge_reports = quality_bridge_root / "reports"
    quality_bridge_cofitok = (
        quality_bridge_root / STABILITY_QUALITY_BRIDGE_COFITOK_RUN_ID
    )
    quality_bridge_dense = (
        quality_bridge_root / STABILITY_QUALITY_BRIDGE_DENSE_RUN_ID
    )
    full_root = output / STABILITY_FULL_ROOT_ID
    full_reports = full_root / "reports"
    full_cofitok = full_root / STABILITY_FULL_COFITOK_RUN_ID
    full_dense = full_root / STABILITY_FULL_DENSE_RUN_ID
    export_root = output / STABILITY_EXPORT_ROOT_ID
    deployment_root = output / STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT_ID
    return {
        "STABILITY_DECISION": (
            probe_root / STABILITY_DECISION_ID / "scaling_decision.json"
        ),
        "STABILITY_SCALING_ROOT": scaling_root,
        "STABILITY_SCALING_COFITOK_RUN": scaling_cofitok,
        "STABILITY_SCALING_DENSE_RUN": scaling_dense,
        "STABILITY_SCALING_MONITOR": scaling_root / "pair_monitor.json",
        "STABILITY_SCALING_REPORT_ROOT": scaling_reports,
        "STABILITY_SCALING_PAIR_SUMMARY": scaling_reports / "pair_summary.json",
        "STABILITY_SCALING_GATE": scaling_reports / "promotion_gate.json",
        "STABILITY_QUALITY_BRIDGE_ROOT": quality_bridge_root,
        "STABILITY_QUALITY_BRIDGE_COFITOK_RUN": quality_bridge_cofitok,
        "STABILITY_QUALITY_BRIDGE_DENSE_RUN": quality_bridge_dense,
        "STABILITY_QUALITY_BRIDGE_MONITOR": quality_bridge_root / "pair_monitor.json",
        "STABILITY_QUALITY_BRIDGE_REPORT_ROOT": quality_bridge_reports,
        "STABILITY_QUALITY_BRIDGE_PREPARATION": (
            quality_bridge_reports / "preparation.json"
        ),
        "STABILITY_FULL_ROOT": full_root,
        "STABILITY_FULL_COFITOK_RUN": full_cofitok,
        "STABILITY_FULL_DENSE_RUN": full_dense,
        "STABILITY_FULL_MONITOR": full_root / "pair_monitor.json",
        "STABILITY_FULL_REPORT_ROOT": full_reports,
        "STABILITY_FULL_GATE": full_reports / "final_generation_gate.json",
        "STABILITY_EXPORT_ROOT": export_root,
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_ROOT": deployment_root,
        "STABILITY_LARGE_CAPACITY_DEPLOYMENT_RECEIPT": (
            deployment_root / "deployment_receipt.json"
        ),
        "STABILITY_COFITOK_INFERENCE_ARTIFACT": (
            export_root / "cofitok_k8_ema_inference.pt"
        ),
        "STABILITY_DENSE_INFERENCE_ARTIFACT": (
            export_root / "dense_identity_ema_inference.pt"
        ),
    }


def generation_capacity_full_workspace_paths(
    *,
    output_root: str | Path,
) -> dict[str, Path]:
    output = Path(output_root).resolve()
    full_root = output / CAPACITY_FULL_ROOT_ID
    report_root = full_root / "reports"
    export_root = output / CAPACITY_FULL_EXPORT_ROOT_ID
    return {
        "CAPACITY_FULL_ROOT": full_root,
        "CAPACITY_FULL_COFITOK_RUN": full_root / CAPACITY_FULL_COFITOK_RUN_ID,
        "CAPACITY_FULL_DENSE_RUN": full_root / CAPACITY_FULL_DENSE_RUN_ID,
        "CAPACITY_FULL_MONITOR": full_root / "pair_monitor.json",
        "CAPACITY_FULL_REPORT_ROOT": report_root,
        "CAPACITY_FULL_TRAINING_AUTHORIZATION": (
            report_root / "capacity_full_300k_training_launch_receipt.json"
        ),
        "CAPACITY_FULL_GATE": report_root / "final_generation_gate.json",
        "CAPACITY_FULL_EXPORT_ROOT": export_root,
        "CAPACITY_FULL_COFITOK_INFERENCE_ARTIFACT": (
            export_root / "cofitok_k8_ema_inference.pt"
        ),
        "CAPACITY_FULL_DENSE_INFERENCE_ARTIFACT": (
            export_root / "dense_identity_ema_inference.pt"
        ),
    }
