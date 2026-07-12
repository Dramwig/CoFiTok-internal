#!/usr/bin/env python
"""Validate and summarize the matched AAAI-27 ablation suite."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable


COMPONENT_ORDER = (
    "full",
    "endpoint_only",
    "no_path_prefix",
    "no_path_component",
    "simultaneous",
    "deep_synthesis",
)
ARCHITECTURE_ORDER = ("full", "direct_dense")
SWEEP_ORDER = ("lambda_prefix", "lambda_component", "progress_power", "token_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot aggregate an empty sequence")
    return {
        "mean": statistics.mean(materialized),
        "std": statistics.stdev(materialized) if len(materialized) > 1 else 0.0,
    }


def extract_run(
    entry: dict[str, Any],
    suite_root: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    train_path = Path(entry["checkpoint_dir"]) / "report.json"
    quality_path = suite_root / "eval" / str(entry["id"]) / "quality_report.json"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not quality_path.exists():
        raise FileNotFoundError(quality_path)
    train = read_json(train_path)
    quality = read_json(quality_path)
    train_config = train["config"]
    quality_config = quality["config"]
    evaluation = quality["evaluation"]
    seed = int(entry["seed"])
    token_count = int(entry["token_count"])

    for label, config in (("train", train_config), ("quality", quality_config)):
        if config["data"]["dataset"] != entry["dataset"]:
            raise ValueError(f"{entry['id']}: {label} dataset mismatch")
        if int(config["runtime"]["seed"]) != seed:
            raise ValueError(f"{entry['id']}: {label} seed mismatch")
        if int(config["runtime"]["steps"]) != 5000:
            raise ValueError(f"{entry['id']}: {label} training-step mismatch")
        if int(config["model"]["token_count"]) != token_count:
            raise ValueError(f"{entry['id']}: {label} token-count mismatch")

    expected_budgets = list(range(1, token_count + 1))
    if (
        int(evaluation["image_count"]) != int(protocol["image_count"])
        or int(evaluation["fixed_timestep"]) != int(protocol["timestep"])
        or evaluation["prefix_budgets"] != expected_budgets
        or evaluation["component_order"] != protocol["component_order"]
        or abs(
            float(evaluation["evaluation_denoise_path_progress_power"])
            - float(protocol["evaluation_progress_power"])
        )
        > 1e-12
    ):
        raise ValueError(f"{entry['id']}: quality protocol mismatch")

    endpoint = quality["metrics_by_prefix"][str(token_count)]
    energy = quality["component_energy"]
    row = {
        "id": entry["id"],
        "variant": entry["variant"],
        "display": entry["display"],
        "dataset": entry["dataset"],
        "seed": seed,
        "token_count": token_count,
        "component_ablation": bool(entry.get("component_ablation", False)),
        "architecture_control": bool(entry.get("architecture_control", False)),
        "sweep_values": dict(entry.get("sweep_values", {})),
        "endpoint_mse": float(endpoint["mse"]),
        "endpoint_psnr_db": float(endpoint["psnr_db"]),
        "path_auc": (
            float(quality["curve_auc"]["denoise_path_mse"])
            if token_count > 1
            else None
        ),
        "effective_k": float(energy["energy_effective_token_count"]),
        "energy_entropy_normalized": float(energy["energy_entropy_normalized"]),
        "zero_ratio": float(quality["diagnostics"]["zero_token_component_energy_ratio"]),
        "mean_abs_component_cosine": float(quality["component_correlation"]["mean_abs_cosine"]),
        "train_report": str(train_path),
        "quality_report": str(quality_path),
    }
    return row, train_path, quality_path


def aggregate_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    if not selected:
        raise ValueError(f"missing variant: {variant}")
    seeds = sorted(int(row["seed"]) for row in selected)
    if seeds != [103, 139]:
        raise ValueError(f"{variant}: expected seeds [103, 139], got {seeds}")
    metrics = {
        key: aggregate(row[key] for row in selected)
        for key in (
            "endpoint_mse",
            "endpoint_psnr_db",
            "effective_k",
            "energy_entropy_normalized",
            "zero_ratio",
            "mean_abs_component_cosine",
        )
    }
    path_values = [row["path_auc"] for row in selected if row["path_auc"] is not None]
    metrics["path_auc"] = aggregate(path_values) if path_values else None
    return {
        "variant": variant,
        "display": selected[0]["display"],
        "seeds": seeds,
        "seed_count": len(seeds),
        "token_count": selected[0]["token_count"],
        **metrics,
    }


def build_sweeps(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sweeps: dict[str, list[dict[str, Any]]] = {}
    for axis in SWEEP_ORDER:
        grouped: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            if axis in row["sweep_values"]:
                value = float(row["sweep_values"][axis])
                grouped.setdefault(value, []).append(row)
        points = []
        for value, selected in sorted(grouped.items()):
            seeds = sorted(int(row["seed"]) for row in selected)
            if seeds != [103, 139]:
                raise ValueError(f"{axis}={value}: expected seeds [103, 139], got {seeds}")
            points.append(
                {
                    "value": value,
                    "seeds": seeds,
                    "endpoint_mse": aggregate(row["endpoint_mse"] for row in selected),
                    "path_auc": aggregate(row["path_auc"] for row in selected if row["path_auc"] is not None),
                    "effective_k": aggregate(row["effective_k"] for row in selected),
                }
            )
        if len(points) < 3:
            raise ValueError(f"{axis}: expected at least three sweep points")
        sweeps[axis] = points
    return sweeps


def paired_summary(rows: list[dict[str, Any]], baseline_variant: str) -> dict[str, Any]:
    full = {int(row["seed"]): row for row in rows if row["variant"] == "full"}
    baseline = {int(row["seed"]): row for row in rows if row["variant"] == baseline_variant}
    seeds = sorted(set(full) & set(baseline))
    if seeds != [103, 139]:
        raise ValueError(f"{baseline_variant}: incomplete paired seeds")
    path_pairs = [
        {
            "seed": seed,
            "full": full[seed]["path_auc"],
            "baseline": baseline[seed]["path_auc"],
        }
        for seed in seeds
        if full[seed]["path_auc"] is not None and baseline[seed]["path_auc"] is not None
    ]
    return {
        "baseline_variant": baseline_variant,
        "seeds": seeds,
        "full_path_better_count": sum(pair["full"] < pair["baseline"] for pair in path_pairs),
        "path_pair_count": len(path_pairs),
        "mean_path_relative_change": (
            statistics.mean(pair["full"] / pair["baseline"] - 1.0 for pair in path_pairs)
            if path_pairs
            else None
        ),
        "mean_endpoint_relative_change": statistics.mean(
            full[seed]["endpoint_mse"] / baseline[seed]["endpoint_mse"] - 1.0
            for seed in seeds
        ),
    }


def build_payload(manifest: dict[str, Any], suite_root: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path, Path]]]:
    extracted = [
        extract_run(entry, suite_root, manifest["evaluation_protocol"])
        for entry in manifest["entries"]
    ]
    rows = [item[0] for item in extracted]
    component = [aggregate_variant(rows, variant) for variant in COMPONENT_ORDER]
    architecture = [aggregate_variant(rows, variant) for variant in ARCHITECTURE_ORDER]
    sweeps = build_sweeps(rows)
    paired = {
        variant: paired_summary(rows, variant)
        for variant in ("endpoint_only", "no_path_prefix", "no_path_component", "simultaneous", "direct_dense")
    }
    deep = next(row for row in component if row["variant"] == "deep_synthesis")
    full = next(row for row in component if row["variant"] == "full")
    summary = {
        "run_count": len(rows),
        "component_variant_count": len(component),
        "deep_synthesis_zero_ratio_nonzero": deep["zero_ratio"]["mean"] > 0.0,
        "restricted_zero_ratio_max": max(
            row["zero_ratio"] for row in rows if row["variant"] == "full"
        ),
        "full_vs_endpoint_path": paired["endpoint_only"],
        "full_vs_no_path_prefix": paired["no_path_prefix"],
        "full_vs_no_path_component": paired["no_path_component"],
        "full_vs_simultaneous": paired["simultaneous"],
        "full_endpoint_mse": full["endpoint_mse"],
    }
    return (
        {
            "schema_version": 1,
            "suite": manifest["suite"],
            "training_protocol": manifest["training_protocol"],
            "evaluation_protocol": manifest["evaluation_protocol"],
            "runs": sorted(rows, key=lambda row: str(row["id"])),
            "component_ablations": component,
            "architecture_controls": architecture,
            "sweeps": sweeps,
            "paired_comparisons": paired,
            "summary": summary,
        },
        extracted,
    )


def pm(metric: dict[str, float] | None, digits: int = 4) -> str:
    if metric is None:
        return "--"
    return f"{metric['mean']:.{digits}f} +/- {metric['std']:.{digits}f}"


def latex_pm(metric: dict[str, float] | None, digits: int = 4) -> str:
    return pm(metric, digits).replace("+/-", r"\pm")


def write_component_rows(rows: list[dict[str, Any]], path: Path) -> None:
    lines = []
    for index, row in enumerate(rows):
        end = r" \\" if index < len(rows) - 1 else ""
        lines.append(
            f"{row['display']} & ${latex_pm(row['endpoint_mse'])}$ & "
            f"${latex_pm(row['path_auc'])}$ & ${latex_pm(row['effective_k'], 3)}$ & "
            f"${latex_pm(row['zero_ratio'])}${end}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sweep_rows(sweeps: dict[str, list[dict[str, Any]]], path: Path) -> None:
    axis_labels = {
        "lambda_prefix": r"$\lambda_p$",
        "lambda_component": r"$\lambda_c$",
        "progress_power": r"$p$",
        "token_count": r"$K$",
    }
    lines: list[str] = []
    axes = list(SWEEP_ORDER)
    for axis_index, axis in enumerate(axes):
        points = sweeps[axis]
        for point_index, point in enumerate(points):
            label = axis_labels[axis] if point_index == 0 else ""
            is_final_row = axis_index == len(axes) - 1 and point_index == len(points) - 1
            row_end = "" if is_final_row else r" \\"
            lines.append(
                f"{label} & {point['value']:g} & ${latex_pm(point['endpoint_mse'])}$ & "
                f"${latex_pm(point['path_auc'])}$ & ${latex_pm(point['effective_k'], 3)}${row_end}"
            )
        if axis_index < len(axes) - 1:
            lines.append(r"\midrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csvs(payload: dict[str, Any], output_dir: Path) -> None:
    run_fields = [
        "id", "variant", "display", "dataset", "seed", "token_count", "endpoint_mse",
        "endpoint_psnr_db", "path_auc", "effective_k", "energy_entropy_normalized",
        "zero_ratio", "mean_abs_component_cosine", "train_report", "quality_report",
    ]
    with (output_dir / "ablation_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["runs"])
    with (output_dir / "hyperparameter_sweeps.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["axis", "value", "seeds", "endpoint_mse_mean", "endpoint_mse_std", "path_auc_mean", "path_auc_std", "effective_k_mean", "effective_k_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for axis, points in payload["sweeps"].items():
            for point in points:
                writer.writerow(
                    {
                        "axis": axis,
                        "value": point["value"],
                        "seeds": ",".join(str(seed) for seed in point["seeds"]),
                        "endpoint_mse_mean": point["endpoint_mse"]["mean"],
                        "endpoint_mse_std": point["endpoint_mse"]["std"],
                        "path_auc_mean": point["path_auc"]["mean"],
                        "path_auc_std": point["path_auc"]["std"],
                        "effective_k_mean": point["effective_k"]["mean"],
                        "effective_k_std": point["effective_k"]["std"],
                    }
                )


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    protocol = payload["evaluation_protocol"]
    lines = [
        "# AAAI-27 Component and Hyperparameter Ablations",
        "",
        (
            f"All rows use ImageNet-64 HF, 5,000 steps, seeds 103/139, and "
            f"{protocol['image_count']} validation images at t={protocol['timestep']}. "
            f"Every path AUC is scored against the same p={protocol['evaluation_progress_power']} reference path."
        ),
        "",
        "| variant | endpoint x0 MSE | path AUC | effective K | zero ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["component_ablations"]:
        lines.append(
            f"| {row['display']} | {pm(row['endpoint_mse'])} | {pm(row['path_auc'])} | "
            f"{pm(row['effective_k'], 3)} | {pm(row['zero_ratio'])} |"
        )
    lines.extend(["", "## Hyperparameter Sweeps", ""])
    for axis, points in payload["sweeps"].items():
        lines.extend(
            [
                f"### {axis}",
                "",
                "| value | endpoint x0 MSE | path AUC | effective K |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for point in points:
            lines.append(
                f"| {point['value']:g} | {pm(point['endpoint_mse'])} | "
                f"{pm(point['path_auc'])} | {pm(point['effective_k'], 3)} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_raw_reports(
    extracted: list[tuple[dict[str, Any], Path, Path]],
    output_dir: Path,
) -> None:
    train_dir = output_dir / "raw" / "train"
    quality_dir = output_dir / "raw" / "quality"
    train_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    for row, train_path, quality_path in extracted:
        shutil.copy2(train_path, train_dir / f"{row['id']}.json")
        shutil.copy2(quality_path, quality_dir / f"{row['id']}.json")


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    suite_root = args.suite_root or Path(manifest["suite_root"])
    payload, extracted = build_payload(manifest, suite_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(args.manifest, args.output_dir / "suite_manifest.json")
    copy_raw_reports(extracted, args.output_dir)
    write_csvs(payload, args.output_dir)
    write_component_rows(payload["component_ablations"], args.output_dir / "component_ablation_rows.tex")
    write_component_rows(payload["architecture_controls"], args.output_dir / "architecture_control_rows.tex")
    write_sweep_rows(payload["sweeps"], args.output_dir / "hyperparameter_sweep_rows.tex")
    write_markdown(payload, args.output_dir / "README.md")
    print(args.output_dir)


if __name__ == "__main__":
    main()
