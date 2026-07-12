#!/usr/bin/env python
"""Plot publication-ready two-seed ablation curves from the validated report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AXES = (
    ("lambda_prefix", r"Prefix weight $\lambda_p$", 0.15),
    ("lambda_component", r"Component weight $\lambda_c$", 0.30),
    ("progress_power", r"Progress power $p$", 1.50),
    ("token_count", r"Token count $K$", 8.0),
)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def values(points: list[dict[str, Any]], metric: str, field: str) -> list[float]:
    return [float(point[metric][field]) for point in points]


def plot(report: dict[str, Any], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(7.0, 3.35), constrained_layout=True)

    for column, (axis, title, default) in enumerate(AXES):
        points = report["sweeps"][axis]
        x = [float(point["value"]) for point in points]
        path_mean = values(points, "path_auc", "mean")
        path_std = values(points, "path_auc", "std")
        mse_mean = values(points, "endpoint_mse", "mean")
        mse_std = values(points, "endpoint_mse", "std")

        top = axes[0, column]
        bottom = axes[1, column]
        top.errorbar(x, path_mean, yerr=path_std, color=BLUE, marker="o", capsize=2.0)
        bottom.errorbar(x, mse_mean, yerr=mse_std, color=ORANGE, marker="s", capsize=2.0)
        top.axvline(default, color=GRAY, linestyle="--", linewidth=0.8, zorder=0)
        bottom.axvline(default, color=GRAY, linestyle="--", linewidth=0.8, zorder=0)
        top.set_title(title, pad=3)
        bottom.set_xlabel(title)
        top.grid(axis="y", color="#d9d9d9", linewidth=0.45)
        bottom.grid(axis="y", color="#d9d9d9", linewidth=0.45)
        for panel in (top, bottom):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.tick_params(length=2.5, width=0.6)
        if axis == "token_count":
            top.set_xticks(x, [str(int(value)) for value in x])
            bottom.set_xticks(x, [str(int(value)) for value in x])

    axes[0, 0].set_ylabel(r"Path AUC $\downarrow$")
    axes[1, 0].set_ylabel(r"Endpoint $x_0$ MSE $\downarrow$")
    figure.suptitle(
        "Matched ImageNet-64 ablations (mean and sample s.d.; seeds 103/139)",
        fontsize=8.6,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "aaai27_ablation_hyperparameters.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "aaai27_ablation_hyperparameters.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    report = read_json(args.report)
    plot(report, args.output_dir)
    manifest = {
        "schema_version": 1,
        "source_report": str(args.report),
        "outputs": [
            str(args.output_dir / "aaai27_ablation_hyperparameters.pdf"),
            str(args.output_dir / "aaai27_ablation_hyperparameters.png"),
        ],
        "protocol": report["evaluation_protocol"],
        "axes": [axis for axis, _, _ in AXES],
        "statistics": "mean +/- sample standard deviation over seeds 103 and 139",
        "style": "Matplotlib; Okabe-Ito blue/orange; vector PDF and 300-DPI PNG",
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
