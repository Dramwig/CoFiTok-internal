#!/usr/bin/env python
"""Build concise main-paper result rows from the locked evidence package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_NAMES = {
    "cifar10": "CIFAR-10",
    "tiny_imagenet_200": "Tiny-IN",
    "imagenet_1k_64x64_hf": "ImageNet-64 HF",
    "downsampled_imagenet_64": "ImageNet-64 strict",
    "ffhq_64": "FFHQ-64",
    "afhqv2_64": "AFHQv2-64",
    "imagenet_256_10pct": r"ImageNet-256 10\%",
    "imagenet_256": "ImageNet-256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_all_dataset_rows(evidence: dict[str, Any]) -> list[str]:
    lines = []
    rows = evidence["diagnostic_rows"]
    for index, row in enumerate(rows):
        end = r" \\" if index < len(rows) - 1 else ""
        lines.append(
            f"{DATASET_NAMES[row['dataset']]} & {row['cofitok_psnr']:.3f} & "
            f"{row['dense_psnr']:.3f} & {row['cofitok_path_auc']:.4f} & "
            f"{row['endpoint_only_path_auc']:.4f} & {row['cofitok_effective_tokens']:.2f} & "
            f"{row['endpoint_only_effective_tokens']:.2f} & {row['cofitok_zero_ratio']:.4f} & "
            f"{row['deep_synthesis_zero_ratio']:.4f}{end}"
        )
    return lines


def build_confirmatory_rows(evidence: dict[str, Any]) -> list[str]:
    rows = evidence["imagenet256_confirmatory"]["rows"]
    lines = []
    for index, row in enumerate(rows):
        end = r" \\" if index < len(rows) - 1 else ""
        lines.append(
            f"{row['seed']} & {row['endpoint_only_path_auc']:.4f} & "
            f"{row['cofitok_ordered_path_auc']:.4f} & "
            f"{row['exhaustive_nonidentity_mean_path_auc']:.4f} & "
            f"{row['cofitok_reverse_path_auc']:.4f} & {row['ordered_rank_of_24']} & "
            f"{row['nonidentity_delta_ci_low']:.4f} & "
            f"{row['dense_monolithic_endpoint_mse']:.4f} & "
            f"{row['cofitok_endpoint_mse']:.4f}{end}"
        )
    return lines


def main() -> None:
    args = parse_args()
    evidence = read_json(args.evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_dataset_result_rows.tex").write_text(
        "\n".join(build_all_dataset_rows(evidence)) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "imagenet256_confirmatory_rows.tex").write_text(
        "\n".join(build_confirmatory_rows(evidence)) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source_evidence": str(args.evidence),
        "all_dataset_row_count": len(evidence["diagnostic_rows"]),
        "confirmatory_row_count": len(evidence["imagenet256_confirmatory"]["rows"]),
        "notes": [
            "The concise main table excludes the legacy clean-prefix objective row.",
            "Numerical values are copied from the locked evidence JSON without modification.",
        ],
    }
    (args.output_dir / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
