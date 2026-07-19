from __future__ import annotations

from pathlib import Path


SCALING_COFITOK_RUN_ID = "imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2"
SCALING_DENSE_RUN_ID = "imagenet256_10pct_rankcomplete_dense_50k_v2"
SCALING_REPORT_ID = "imagenet256_10pct_rankcomplete_matched_50k_v2"
FULL_COFITOK_RUN_ID = "imagenet256_full_cofitok_k8_300k"
FULL_DENSE_RUN_ID = "imagenet256_full_dense_300k"
FULL_REPORT_ID = "imagenet256_full_matched_300k"


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
