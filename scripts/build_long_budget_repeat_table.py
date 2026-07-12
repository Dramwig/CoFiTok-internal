#!/usr/bin/env python
"""Build the cross-batch matched 20k CoFiTok versus endpoint-only table."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cofitok.configs import LossConfig, ModelConfig


REPORT_SPECS = [
    {
        "dataset": "tiny_imagenet_200",
        "dataset_display": "Tiny ImageNet-200",
        "method": "endpoint_only_factorized",
        "method_display": "Endpoint-only factorized",
        "reports": [
            "train_tiny_imagenet_k8_epsilononly_p150eval_20k_2026-07-08/report.json",
            "train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda_2026-07-08/report.json",
        ],
        "quality_reports": [
            "quality_path_tiny_endpoint_only_20k_seed103_1024_t500_2026-07-11_confirmatory/quality_report.json",
            "quality_path_tiny_endpoint_only_20k_seed139_1024_t500_2026-07-11_confirmatory/quality_report.json",
        ],
    },
    {
        "dataset": "tiny_imagenet_200",
        "dataset_display": "Tiny ImageNet-200",
        "method": "cofitok",
        "method_display": "CoFiTok",
        "reports": [
            "train_tiny_imagenet_k8_denoisepath_p150_light_20k_2026-07-08/report.json",
            "train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda_2026-07-08/report.json",
        ],
        "quality_reports": [
            "quality_path_tiny_cofitok_20k_seed103_1024_t500_2026-07-11_confirmatory/quality_report.json",
            "quality_path_tiny_cofitok_20k_seed139_1024_t500_2026-07-11_confirmatory/quality_report.json",
        ],
    },
    {
        "dataset": "imagenet_1k_64x64_hf",
        "dataset_display": "ImageNet-64 HF",
        "method": "endpoint_only_factorized",
        "method_display": "Endpoint-only factorized",
        "reports": [
            "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_2026-07-08/report.json",
            "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda_2026-07-08/report.json",
        ],
        "quality_reports": [
            "quality_path_imagenet_hf_endpoint_only_20k_seed103_1024_t500_2026-07-11_confirmatory/quality_report.json",
            "quality_path_imagenet_hf_endpoint_only_20k_seed139_1024_t500_2026-07-11_confirmatory/quality_report.json",
        ],
    },
    {
        "dataset": "imagenet_1k_64x64_hf",
        "dataset_display": "ImageNet-64 HF",
        "method": "cofitok",
        "method_display": "CoFiTok",
        "reports": [
            "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_2026-07-08/report.json",
            "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda_2026-07-08/report.json",
        ],
        "quality_reports": [
            "quality_path_imagenet_hf_cofitok_20k_seed103_1024_t500_2026-07-11_confirmatory/quality_report.json",
            "quality_path_imagenet_hf_cofitok_20k_seed139_1024_t500_2026-07-11_confirmatory/quality_report.json",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_config_section(raw: dict[str, Any], config_type: type[Any]) -> dict[str, Any]:
    values = asdict(config_type())
    values.update(raw)
    return values


def extract_run(
    path: Path,
    quality_path: Path,
    expected_dataset: str,
    method: str,
) -> dict[str, Any]:
    payload = read_json(path)
    config = payload["config"]
    prefix = payload["prefix_summary"]
    dataset = config["data"]["dataset"]
    steps = int(config["runtime"]["steps"])
    token_count = int(config["model"]["token_count"])
    if dataset != expected_dataset:
        raise ValueError(f"{path}: dataset {dataset!r} != {expected_dataset!r}")
    if steps != 20_000 or token_count != 8:
        raise ValueError(f"{path}: expected 20k K=8, got steps={steps}, K={token_count}")
    denoise_path_weight = float(config.get("loss", {}).get("denoise_path_prefix_weight", 0.0))
    if method == "cofitok" and denoise_path_weight <= 0.0:
        raise ValueError(f"{path}: CoFiTok report lacks denoise-path prefix loss")
    if method == "endpoint_only_factorized" and denoise_path_weight != 0.0:
        raise ValueError(f"{path}: endpoint-only report has denoise-path prefix loss")
    quality = read_json(quality_path)
    quality_config = quality["config"]
    evaluation = quality["evaluation"]
    if quality_config["data"]["dataset"] != expected_dataset:
        raise ValueError(f"{quality_path}: dataset mismatch")
    if int(quality_config["runtime"]["seed"]) != int(config["runtime"]["seed"]):
        raise ValueError(f"{quality_path}: seed mismatch")
    if int(quality_config["runtime"]["steps"]) != 20_000:
        raise ValueError(f"{quality_path}: expected 20k quality report")
    if normalize_config_section(
        quality_config["model"], ModelConfig
    ) != normalize_config_section(config["model"], ModelConfig) or normalize_config_section(
        quality_config.get("loss", {}), LossConfig
    ) != normalize_config_section(config.get("loss", {}), LossConfig):
        raise ValueError(f"{quality_path}: train/quality model or loss config mismatch")
    reported_checkpoint = Path(str(quality.get("checkpoint", "")))
    if (
        reported_checkpoint.name != "checkpoint_final.pt"
        or reported_checkpoint.parent.name != path.parent.name
    ):
        raise ValueError(f"{quality_path}: quality checkpoint does not match train report")
    if (
        int(evaluation["image_count"]) != 1_024
        or int(evaluation["fixed_timestep"]) != 500
        or evaluation["prefix_budgets"] != list(range(1, 9))
        or evaluation.get("component_order") != "ordered"
        or int(evaluation.get("random_order_seed", -1)) != 0
    ):
        raise ValueError(f"{quality_path}: invalid cross-batch path protocol")
    effective_k = quality.get("component_energy", {}).get(
        "energy_effective_token_count"
    )
    if effective_k is None:
        raise ValueError(f"{quality_path}: missing cross-batch effective token count")
    return {
        "dataset": dataset,
        "method": method,
        "seed": int(config["runtime"]["seed"]),
        "steps": steps,
        "final_mse": float(quality["metrics_by_prefix"]["8"]["mse"]),
        "path_auc": float(quality["curve_auc"]["denoise_path_mse"]),
        "effective_k": float(effective_k),
        "zero_ratio": float(quality["diagnostics"]["zero_token_component_energy_ratio"]),
        "train_report": str(path),
        "quality_report": str(quality_path),
    }


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build(reports_root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for spec in REPORT_SPECS:
        if len(spec["reports"]) != len(spec["quality_reports"]):
            raise ValueError(f"mismatched train/quality reports for {spec['dataset']}/{spec['method']}")
        spec_runs = [
            extract_run(
                reports_root / train_relative,
                reports_root / quality_relative,
                spec["dataset"],
                spec["method"],
            )
            for train_relative, quality_relative in zip(
                spec["reports"], spec["quality_reports"]
            )
        ]
        seeds = [row["seed"] for row in spec_runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds for {spec['dataset']}/{spec['method']}: {seeds}")
        runs.extend(spec_runs)
        aggregates.append(
            {
                "dataset": spec["dataset"],
                "dataset_display": spec["dataset_display"],
                "method": spec["method"],
                "method_display": spec["method_display"],
                "seeds": sorted(seeds),
                "seed_count": len(seeds),
                "final_mse": aggregate([row["final_mse"] for row in spec_runs]),
                "path_auc": aggregate([row["path_auc"] for row in spec_runs]),
                "effective_k": aggregate([row["effective_k"] for row in spec_runs]),
                "zero_ratio": aggregate([row["zero_ratio"] for row in spec_runs]),
            }
        )

    by_key = {(row["dataset"], row["method"], row["seed"]): row for row in runs}
    paired: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in runs}):
        seeds = sorted(
            row["seed"]
            for row in runs
            if row["dataset"] == dataset and row["method"] == "cofitok"
        )
        for seed in seeds:
            cofitok = by_key[(dataset, "cofitok", seed)]
            dense = by_key[(dataset, "endpoint_only_factorized", seed)]
            paired.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "path_auc_relative_reduction": 1.0 - cofitok["path_auc"] / dense["path_auc"],
                    "final_mse_relative_change": cofitok["final_mse"] / dense["final_mse"] - 1.0,
                    "path_auc_better": cofitok["path_auc"] < dense["path_auc"],
                    "final_mse_better": cofitok["final_mse"] < dense["final_mse"],
                }
            )

    summary = {
        "pair_count": len(paired),
        "path_auc_better_pairs": sum(row["path_auc_better"] for row in paired),
        "final_mse_better_pairs": sum(row["final_mse_better"] for row in paired),
        "mean_path_auc_relative_reduction": statistics.mean(
            row["path_auc_relative_reduction"] for row in paired
        ),
        "mean_final_mse_relative_change": statistics.mean(
            row["final_mse_relative_change"] for row in paired
        ),
    }
    return {
        "schema_version": 1,
        "protocol": "matched_20k_k8_two_seed_cofitok_vs_endpoint_only_factorized",
        "runs": runs,
        "aggregates": aggregates,
        "paired": paired,
        "summary": summary,
    }


def format_pm(metric: dict[str, float], digits: int) -> str:
    return f"{metric['mean']:.{digits}f} +/- {metric['std']:.{digits}f}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Matched 20k Two-Seed Comparison",
        "",
        "All rows use K=8, 20,000 training steps, seeds 103/139, and 1,024-image full-prefix evaluation at t=500. The endpoint-only control retains token factorization and is not a monolithic dense head.",
        "",
        "| dataset | method | endpoint x0 MSE at t=500 | path AUC | effective K | zero ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregates"]:
        lines.append(
            "| {dataset} | {method} | {mse} | {auc} | {eff} | {zero} |".format(
                dataset=row["dataset_display"],
                method=row["method_display"],
                mse=format_pm(row["final_mse"], 4),
                auc=format_pm(row["path_auc"], 4),
                eff=format_pm(row["effective_k"], 3),
                zero=format_pm(row["zero_ratio"], 4),
            )
        )
    summary = payload["summary"]
    lines.extend(
        [
            "",
            (
                f"CoFiTok has lower path AUC in {summary['path_auc_better_pairs']}/"
                f"{summary['pair_count']} paired dataset-seed comparisons, with a mean "
                f"relative reduction of {100 * summary['mean_path_auc_relative_reduction']:.2f}%."
            ),
            (
                f"The mean endpoint x0-MSE change at t=500 is "
                f"{100 * summary['mean_final_mse_relative_change']:+.2f}%; endpoint MSE "
                f"is better in {summary['final_mse_better_pairs']}/{summary['pair_count']} pairs."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(payload: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "dataset",
        "method",
        "seed_count",
        "seeds",
        "final_mse_mean",
        "final_mse_std",
        "path_auc_mean",
        "path_auc_std",
        "effective_k_mean",
        "effective_k_std",
        "zero_ratio_mean",
        "zero_ratio_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["aggregates"]:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "seed_count": row["seed_count"],
                    "seeds": ",".join(str(seed) for seed in row["seeds"]),
                    "final_mse_mean": row["final_mse"]["mean"],
                    "final_mse_std": row["final_mse"]["std"],
                    "path_auc_mean": row["path_auc"]["mean"],
                    "path_auc_std": row["path_auc"]["std"],
                    "effective_k_mean": row["effective_k"]["mean"],
                    "effective_k_std": row["effective_k"]["std"],
                    "zero_ratio_mean": row["zero_ratio"]["mean"],
                    "zero_ratio_std": row["zero_ratio"]["std"],
                }
            )


def write_latex_rows(payload: dict[str, Any], path: Path) -> None:
    lines = []
    rows = payload["aggregates"]
    for index, row in enumerate(rows):
        row_end = r" \\" if index < len(rows) - 1 else ""
        lines.append(
            "{dataset} & {method} & ${mse}$ & ${auc}$ & ${eff}$ & ${zero}${row_end}".format(
                dataset=row["dataset_display"],
                method=row["method_display"],
                mse=format_pm(row["final_mse"], 4).replace("+/-", r"\pm"),
                auc=format_pm(row["path_auc"], 4).replace("+/-", r"\pm"),
                eff=format_pm(row["effective_k"], 3).replace("+/-", r"\pm"),
                zero=format_pm(row["zero_ratio"], 4).replace("+/-", r"\pm"),
                row_end=row_end,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build(args.reports_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "long_budget_repeat_table.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "long_budget_repeat_table.md")
    write_csv(payload, args.output_dir / "long_budget_repeat_table.csv")
    write_latex_rows(payload, args.output_dir / "long_budget_repeat_rows.tex")
    print(args.output_dir)


if __name__ == "__main__":
    main()
