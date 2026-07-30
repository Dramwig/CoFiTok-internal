from __future__ import annotations

import string
from pathlib import Path


SCALING_COFITOK_RUN_ID = "imagenet256_10pct_fixed_basis_cofitok_k8_50k_v3"
SCALING_DENSE_RUN_ID = "imagenet256_10pct_fixed_basis_dense_50k_v3"
SCALING_REPORT_ID = "imagenet256_10pct_fixed_basis_matched_50k_v3"
STABILITY_SCALING_ROOT_ID = "stability_scaling_50k_ema_teacher"
STABILITY_SCALING_COFITOK_RUN_ID = "cofitok_rgbtail3_rollout_x0_u2_ema_teacher"
STABILITY_SCALING_DENSE_RUN_ID = "dense_rollout_x0_u2_ema_teacher"
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
