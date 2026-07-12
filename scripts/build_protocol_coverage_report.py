#!/usr/bin/env python
"""Separate fair-training, tokenizer, and official-eval comparison coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PRIMARY_FAIR_METHODS = {
    "cofitok_light",
    "same_backbone_dense",
    "endpoint_only_factorized",
    "channel_mask",
    "cofitok_no_prefix_loss",
    "cofitok_no_monotonic_loss",
    "cofitok_simultaneous",
    "cofitok_deep_synthesis",
    "edm",
    "improved_diffusion",
}
TOKENIZER_EVAL_METHODS = {"ml_flextok", "titok_1d_tokenizer"}
OFFICIAL_RELATED_METHODS = {"D-AR", "MAR", "ReTok"}
GUARDED_MATRIX_METHODS = {"d_ar", "mar", "retok"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--official-related", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-complete-defined-protocols",
        action="store_true",
        help="Fail after writing the report unless all three defined protocol views are complete.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["status"]) for row in rows).items()))


def build(matrix: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    matrix_rows = matrix.get("rows", [])
    primary_rows = [row for row in matrix_rows if row.get("method") in PRIMARY_FAIR_METHODS]
    tokenizer_rows = [
        row for row in matrix_rows if row.get("method") in TOKENIZER_EVAL_METHODS
    ]
    blocked_rows = [row for row in matrix_rows if row.get("status") == "protocol_blocked"]
    official_rows = [
        row
        for row in official.get("rows", [])
        if row.get("method") in OFFICIAL_RELATED_METHODS
    ]

    datasets = sorted({str(row["dataset"]) for row in matrix_rows})
    expected_primary = len(datasets) * len(PRIMARY_FAIR_METHODS)
    expected_tokenizer = len(datasets) * len(TOKENIZER_EVAL_METHODS)
    expected_primary_pairs = {
        (dataset, method) for dataset in datasets for method in PRIMARY_FAIR_METHODS
    }
    expected_tokenizer_pairs = {
        (dataset, method) for dataset in datasets for method in TOKENIZER_EVAL_METHODS
    }
    expected_blocked_pairs = {
        (dataset, method) for dataset in datasets for method in GUARDED_MATRIX_METHODS
    }
    primary_pairs = {(str(row["dataset"]), str(row["method"])) for row in primary_rows}
    tokenizer_pairs = {
        (str(row["dataset"]), str(row["method"])) for row in tokenizer_rows
    }
    blocked_pairs = {(str(row["dataset"]), str(row["method"])) for row in blocked_rows}
    primary_complete = (
        len(primary_rows) == expected_primary
        and primary_pairs == expected_primary_pairs
        and all(row.get("status") == "completed" for row in primary_rows)
    )
    tokenizer_complete = (
        len(tokenizer_rows) == expected_tokenizer
        and tokenizer_pairs == expected_tokenizer_pairs
        and all(row.get("status") == "completed_eval_only" for row in tokenizer_rows)
    )
    blocked_complete = (
        len(blocked_rows) == len(expected_blocked_pairs)
        and blocked_pairs == expected_blocked_pairs
    )
    official_methods = {str(row.get("method")) for row in official_rows}
    official_complete = (
        len(official_rows) == len(OFFICIAL_RELATED_METHODS)
        and official_methods == OFFICIAL_RELATED_METHODS
        and all(
            str(row.get("status", "")).startswith("completed_eval_only_50k")
            for row in official_rows
        )
    )
    official_running = any(
        str(row.get("status", "")) in {
            "sampling_or_eval_pending",
            "official_ema_sampling_or_eval_pending",
            "community_non_ema_audit_completed_official_ema_pending",
            "pilot128_metrics_completed_50k_running",
        }
        for row in official_rows
    )

    if primary_complete and tokenizer_complete and blocked_complete and official_complete:
        scientific_status = "completed_defined_protocols"
    elif primary_complete and tokenizer_complete and blocked_complete and official_running:
        scientific_status = "official_eval_running"
    else:
        scientific_status = "incomplete_defined_protocols"

    return {
        "schema_version": 1,
        "scientific_status": scientific_status,
        "literal_cartesian_product_complete": len(blocked_rows) == 0,
        "literal_cartesian_product_note": (
            "The guarded master matrix intentionally retains cross-task cells as "
            "protocol_blocked; they are not evidence-equivalent missing jobs."
        ),
        "datasets": datasets,
        "views": {
            "primary_matched_dataset_step_training": {
                "expected": expected_primary,
                "observed": len(primary_rows),
                "complete": primary_complete,
                "status_counts": status_counts(primary_rows),
                "methods": sorted(PRIMARY_FAIR_METHODS),
            },
            "tokenizer_reconstruction_eval_only": {
                "expected": expected_tokenizer,
                "observed": len(tokenizer_rows),
                "complete": tokenizer_complete,
                "status_counts": status_counts(tokenizer_rows),
                "methods": sorted(TOKENIZER_EVAL_METHODS),
            },
            "official_imagenet256_eval_only": {
                "expected": len(OFFICIAL_RELATED_METHODS),
                "observed": len(official_rows),
                "complete": official_complete,
                "running": official_running,
                "status_counts": status_counts(official_rows),
                "methods": sorted(OFFICIAL_RELATED_METHODS),
                "rows": official_rows,
            },
        },
        "guarded_cross_product": {
            "protocol_blocked_count": len(blocked_rows),
            "expected": len(expected_blocked_pairs),
            "complete": blocked_complete,
            "status_counts": status_counts(blocked_rows),
            "methods": sorted({str(row["method"]) for row in blocked_rows}),
            "rows": blocked_rows,
            "rationale": {
                "d_ar": "Official diffusion-as-AR ImageNet-256 generation is not matched-dataset/step retraining on each local dataset.",
                "mar": "Official continuous-token AR is ImageNet-256 class-conditional and uses a different pretrained protocol.",
                "retok": "Official GPT+VQ generation is ImageNet-256 pretrained and is not a matched-dataset/step dense-noise baseline.",
            },
        },
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    views = payload["views"]
    rows = []
    for name, view in views.items():
        rows.append(
            [
                name,
                view["observed"],
                view["expected"],
                view["complete"],
                json.dumps(view["status_counts"], sort_keys=True),
            ]
        )
    body = "\n\n".join(
        [
            "# Protocol Coverage Report",
            f"Scientific status: `{payload['scientific_status']}`",
            markdown_table(
                ["protocol view", "observed", "expected", "complete", "status counts"],
                rows,
            ),
            "## Guarded Cartesian Product",
            (
                f"The master matrix retains "
                f"{payload['guarded_cross_product']['protocol_blocked_count']} "
                "cross-task cells as `protocol_blocked`. They must not be relabeled as "
                "completed fair training from official/eval-only evidence."
            ),
            "## Interpretation",
            (
                "Paper-facing completion should be judged within each defined protocol: "
                "matched-dataset/optimizer-step generation, tokenizer reconstruction, and official "
                "ImageNet-256 related-method evaluation. Literal method-by-dataset "
                "Cartesian completion remains false by design."
            ),
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build(read_json(args.matrix), read_json(args.official_related))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol_coverage_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "protocol_coverage_report.md")
    if (
        args.require_complete_defined_protocols
        and payload["scientific_status"] != "completed_defined_protocols"
    ):
        raise RuntimeError(
            "defined protocol coverage is incomplete: " + payload["scientific_status"]
        )
    print(args.output_dir)


if __name__ == "__main__":
    main()
