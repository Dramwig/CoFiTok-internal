#!/usr/bin/env python
"""Audit whether current evidence supports CoFiTok's scoped top-tier claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(evidence: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    diagnostics = evidence.get("diagnostic_summary", {})
    repeated = evidence.get("long_budget_repeat", {}).get("summary", {})
    confirmatory = evidence.get("imagenet256_confirmatory_summary", {})
    related = evidence.get("official_related_summary", {})
    generation = evidence.get("generation_summary", {})
    diagnostic_rows = evidence.get("diagnostic_rows", [])

    short_dataset_count = int(diagnostics.get("dataset_count", 0))
    short_auc_wins = int(
        diagnostics.get("cofitok_path_auc_better_than_endpoint_only", 0)
    )
    repeated_pairs = int(repeated.get("pair_count", 0))
    repeated_auc_wins = int(repeated.get("path_auc_better_pairs", 0))
    repeated_endpoint_change = float(repeated.get("mean_final_mse_relative_change", 1.0))
    restricted_zero = len(diagnostic_rows) >= 8 and all(
        row.get("cofitok_zero_ratio") is not None
        and abs(float(row["cofitok_zero_ratio"])) <= 1e-12
        for row in diagnostic_rows
    )
    completed_related = set(related.get("completed_50k", []))

    checks = {
        "broad_short_budget_path_auc": (
            short_dataset_count >= 8 and short_auc_wins == short_dataset_count
        ),
        "matched_20k_two_seed_path_auc": (
            repeated_pairs >= 4 and repeated_auc_wins == repeated_pairs
        ),
        "matched_20k_endpoint_cost_within_5pct": repeated_endpoint_change <= 0.05,
        "imagenet256_confirmatory_all_gates": bool(confirmatory.get("overall_pass")),
        "restricted_zero_token_contract": restricted_zero,
        "defined_protocols_complete": coverage.get("scientific_status")
        == "completed_defined_protocols",
        "official_related_50k_complete": completed_related
        == {"D-AR", "MAR", "ReTok"},
    }
    scoped_ready = all(checks.values())
    dataset_count = int(generation.get("dataset_count", 0))
    broad_generation_supported = dataset_count > 0 and (
        int(generation.get("cofitok_best_lowres_count", 0)) == dataset_count
        and int(generation.get("cofitok_best_inception_count", 0)) == dataset_count
    )

    residual_risks = [
        "The strongest evidence is mechanistic prefix control, not generation SOTA.",
        "Architecture-incompatible D-AR/MAR/ReTok rows are official eval-only, not matched-dataset/step retraining.",
        "The main repeated experiments still use small backbones and two seeds.",
        "ReTok's pinned repository does not declare a license; do not redistribute its code or weights.",
    ]
    return {
        "schema_version": 1,
        "decision": {
            "scoped_top_tier_evidence_ready": scoped_ready,
            "supported_claim": (
                "ordered restricted dense-noise factorization with prefix-controllable denoising"
                if scoped_ready
                else "pending required evidence gates"
            ),
            "broad_generation_superiority_supported": broad_generation_supported,
            "top_tier_acceptance_guaranteed": False,
        },
        "checks": checks,
        "observations": {
            "short_path_auc": f"{short_auc_wins}/{short_dataset_count}",
            "repeated_path_auc": f"{repeated_auc_wins}/{repeated_pairs}",
            "repeated_endpoint_mse_change": repeated_endpoint_change,
            "imagenet256_confirmatory": confirmatory,
            "official_related_completed": related.get("completed_50k", []),
            "generation": generation,
            "literal_cartesian_product_complete": coverage.get(
                "literal_cartesian_product_complete", False
            ),
        },
        "residual_risks": residual_risks,
        "interpretation": (
            "A true scoped_top_tier_evidence_ready value means the experimental package is "
            "coherent enough to support the scoped submission claim. It is not a prediction "
            "or guarantee of venue acceptance."
        ),
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    decision = payload["decision"]
    lines = [
        "# CoFiTok Top-Tier Claim Audit",
        "",
        f"Scoped evidence ready: **{decision['scoped_top_tier_evidence_ready']}**",
        f"Broad generation superiority supported: **{decision['broad_generation_superiority_supported']}**",
        "",
        "## Required checks",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for name, passed in payload["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(["", "## Residual risks", ""])
    lines.extend(f"- {risk}" for risk in payload["residual_risks"])
    lines.extend(["", payload["interpretation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = audit(read_json(args.evidence), read_json(args.coverage))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "top_tier_claim_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "top_tier_claim_audit.md")
    print(args.output_dir)


if __name__ == "__main__":
    main()
