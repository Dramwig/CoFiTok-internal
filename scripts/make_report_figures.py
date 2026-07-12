from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "epsilon_only": "#3b82f6",
    "light_denoise_path": "#ef4444",
    "ordered": "#16a34a",
    "random": "#f59e0b",
    "reverse": "#7c3aed",
}

DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "tiny_imagenet_200": "Tiny ImageNet",
    "imagenet_1k_64x64_hf": "ImageNet-64 HF",
}

VARIANT_LABELS = {
    "epsilon_only": "epsilon-only",
    "light_denoise_path": "K8 light",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render CoFiTok report figures from experiment_summary.json.")
    parser.add_argument(
        "--summary",
        default="artifacts/reports/summary_2026-07-08/experiment_summary.json",
        help="Path to summary JSON produced by summarize_experiments.py.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for figure PNGs and manifest JSON.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def _variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def _require_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError(f"No rows available for {label}")
    return rows


def _figure_record(path: Path, source: str, metric: str) -> dict[str, str]:
    return {"path": path.as_posix(), "source": source, "metric": metric}


def select_20k_training_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("train", []):
        if row.get("dataset") not in {"tiny_imagenet_200", "imagenet_1k_64x64_hf"}:
            continue
        if row.get("variant") not in {"epsilon_only", "light_denoise_path"}:
            continue
        if row.get("token_count") != 8 or row.get("steps") != 20000:
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("variant"))))
    return _require_rows(rows, "20k training tradeoff")


def select_20k_generated_rows(
    summary: dict[str, Any],
    sample_image_count: int = 256,
    real_image_count: int = 1024,
) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("generated_quality", []):
        if row.get("dataset") not in {"tiny_imagenet_200", "imagenet_1k_64x64_hf"}:
            continue
        if row.get("variant") not in {"epsilon_only", "light_denoise_path"}:
            continue
        if row.get("token_count") != 8 or row.get("steps") != 20000:
            continue
        if row.get("sample_image_count") != sample_image_count or row.get("real_image_count") != real_image_count:
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("dataset")), str(row.get("variant"))))
    return _require_rows(rows, f"20k generated quality {sample_image_count}-vs-{real_image_count}")


def select_order_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("order_eval", []):
        if row.get("dataset") != "imagenet_1k_64x64_hf":
            continue
        if row.get("variant") != "light_denoise_path":
            continue
        if row.get("token_count") not in {4, 8, 16}:
            continue
        if row.get("seed") != 139:
            continue
        if row.get("component_order") not in {"ordered", "random", "reverse"}:
            continue
        rows.append(row)
    order_rank = {"ordered": 0, "random": 1, "reverse": 2}
    rows.sort(key=lambda row: (int(row.get("token_count") or 0), order_rank.get(str(row.get("component_order")), 99)))
    return _require_rows(rows, "ImageNet-64 order ablation")


def _draw_grouped_bars(
    output_path: Path,
    title: str,
    subtitle: str,
    groups: list[str],
    series: list[str],
    values: dict[tuple[str, str], float],
    colors: dict[str, str],
    y_label: str,
) -> None:
    width, height = 1260, 760
    margin_left, margin_right = 110, 45
    margin_top, margin_bottom = 120, 150
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    subtitle_font = _font(18)
    label_font = _font(17)
    small_font = _font(14)

    draw.text((40, 28), title, fill="#111827", font=title_font)
    draw.text((40, 70), subtitle, fill="#4b5563", font=subtitle_font)
    draw.text((18, plot_top + plot_height // 2 - 20), y_label, fill="#374151", font=label_font)

    all_values = [value for value in values.values() if value is not None]
    max_value = max(all_values) if all_values else 1.0
    max_value = max_value * 1.15 if max_value > 0 else 1.0
    tick_count = 5
    for tick in range(tick_count + 1):
        y_value = max_value * tick / tick_count
        y = plot_bottom - int((y_value / max_value) * plot_height)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((plot_left - 86, y - 8), _format_value(y_value), fill="#6b7280", font=small_font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#111827", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#111827", width=2)

    group_width = plot_width / max(len(groups), 1)
    bar_gap = 10
    bar_width = min(80, (group_width - 34) / max(len(series), 1) - bar_gap)
    for group_index, group in enumerate(groups):
        group_center = plot_left + group_width * (group_index + 0.5)
        total_bar_width = len(series) * bar_width + (len(series) - 1) * bar_gap
        start_x = group_center - total_bar_width / 2
        for series_index, series_name in enumerate(series):
            value = values.get((group, series_name))
            if value is None:
                continue
            x0 = start_x + series_index * (bar_width + bar_gap)
            x1 = x0 + bar_width
            y0 = plot_bottom - int((value / max_value) * plot_height)
            draw.rectangle((x0, y0, x1, plot_bottom), fill=colors.get(series_name, "#9ca3af"))
            label = _format_value(value)
            text_box = draw.textbbox((0, 0), label, font=small_font)
            text_width = text_box[2] - text_box[0]
            draw.text((x0 + bar_width / 2 - text_width / 2, y0 - 22), label, fill="#111827", font=small_font)
        group_text = group.replace(" ", "\n")
        text_box = draw.multiline_textbbox((0, 0), group_text, font=label_font, spacing=2)
        text_width = text_box[2] - text_box[0]
        draw.multiline_text(
            (group_center - text_width / 2, plot_bottom + 18),
            group_text,
            fill="#111827",
            font=label_font,
            spacing=2,
            align="center",
        )

    legend_x = width - 390
    legend_y = 32
    for index, series_name in enumerate(series):
        y = legend_y + index * 28
        draw.rectangle((legend_x, y + 5, legend_x + 18, y + 23), fill=colors.get(series_name, "#9ca3af"))
        draw.text((legend_x + 28, y + 3), series_name, fill="#111827", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_training_figures(summary: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rows = select_20k_training_rows(summary)
    groups = sorted({_dataset_label(str(row["dataset"])) for row in rows})
    series = ["epsilon-only", "K8 light"]
    variant_to_series = {"epsilon_only": "epsilon-only", "light_denoise_path": "K8 light"}
    colors = {"epsilon-only": PALETTE["epsilon_only"], "K8 light": PALETTE["light_denoise_path"]}

    metric_specs = [
        (
            "path_auc_20k.png",
            "Prefix Path Error, 20k Steps",
            "Lower is better. K8 light preserves endpoint quality while making prefixes meaningful.",
            "path AUC",
            "path_auc",
        ),
        (
            "final_clean_mse_20k.png",
            "Endpoint Clean MSE, 20k Steps",
            "Lower is better. Endpoint quality remains close under the CoFiTok objective.",
            "final MSE",
            "final_clean_mse",
        ),
        (
            "effective_tokens_20k.png",
            "Effective Token Usage, 20k Steps",
            "Higher is better. K8 light spreads denoising work across more ordered components.",
            "effective K",
            "effective_tokens",
        ),
    ]
    figures = []
    for filename, title, subtitle, y_label, metric in metric_specs:
        values = {}
        for row in rows:
            group = _dataset_label(str(row["dataset"]))
            series_name = variant_to_series[str(row["variant"])]
            values[(group, series_name)] = float(row[metric])
        path = output_dir / filename
        _draw_grouped_bars(path, title, subtitle, groups, series, values, colors, y_label)
        figures.append(_figure_record(path, "experiment_summary.train", metric))
    return figures


def _render_generated_quality_figure(
    summary: dict[str, Any],
    output_dir: Path,
    sample_image_count: int,
    real_image_count: int,
    filename: str,
) -> dict[str, Any]:
    rows = select_20k_generated_rows(summary, sample_image_count=sample_image_count, real_image_count=real_image_count)
    groups = sorted({_dataset_label(str(row["dataset"])) for row in rows})
    series = ["epsilon-only", "K8 light"]
    variant_to_series = {"epsilon_only": "epsilon-only", "light_denoise_path": "K8 light"}
    colors = {"epsilon-only": PALETTE["epsilon_only"], "K8 light": PALETTE["light_denoise_path"]}
    values = {}
    for row in rows:
        group = _dataset_label(str(row["dataset"]))
        series_name = variant_to_series[str(row["variant"])]
        values[(group, series_name)] = float(row["inception_frechet"])
    path = output_dir / filename
    _draw_grouped_bars(
        path,
        "Generated-Sample Inception Frechet, 20k Steps",
        f"{sample_image_count} generated images vs {real_image_count} validation images. Lower is better.",
        groups,
        series,
        values,
        colors,
        "Frechet",
    )
    return _figure_record(path, "experiment_summary.generated_quality", "inception_frechet")


def render_generated_quality_figures(summary: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    figures = [
        _render_generated_quality_figure(
            summary,
            output_dir,
            sample_image_count=256,
            real_image_count=1024,
            filename="generated_inception_20k_256.png",
        ),
        _render_generated_quality_figure(
            summary,
            output_dir,
            sample_image_count=1024,
            real_image_count=4096,
            filename="generated_inception_20k_1024.png",
        ),
        _render_generated_quality_figure(
            summary,
            output_dir,
            sample_image_count=8192,
            real_image_count=8192,
            filename="generated_inception_20k_8192.png",
        ),
    ]
    formal_rows = [
        row
        for row in summary.get("generated_quality", [])
        if row.get("dataset") == "imagenet_1k_64x64_hf"
        and row.get("variant") in {"epsilon_only", "light_denoise_path"}
        and row.get("token_count") == 8
        and row.get("steps") == 20000
        and row.get("sample_image_count") == 50000
        and row.get("real_image_count") == 50000
    ]
    if formal_rows:
        figures.append(
            _render_generated_quality_figure(
                summary,
                output_dir,
                sample_image_count=50000,
                real_image_count=50000,
                filename="generated_inception_20k_hf_50000.png",
            )
        )
    return figures


def render_order_figure(summary: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rows = select_order_rows(summary)
    groups = [f"K={token_count}" for token_count in [4, 8, 16]]
    series = ["ordered", "random", "reverse"]
    colors = {name: PALETTE[name] for name in series}
    values = {}
    for row in rows:
        group = f"K={row['token_count']}"
        series_name = str(row["component_order"])
        values[(group, series_name)] = float(row["path_auc"])
    path = output_dir / "imagenet_hf_order_path_auc.png"
    _draw_grouped_bars(
        path,
        "Order Ablation on ImageNet-64 HF",
        "Lower is better. Random/reverse order damages prefix path alignment.",
        groups,
        series,
        values,
        colors,
        "path AUC",
    )
    return [_figure_record(path, "experiment_summary.order_eval", "path_auc")]


def make_figures(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    figures.extend(render_training_figures(summary, output_dir))
    figures.extend(render_generated_quality_figures(summary, output_dir))
    figures.extend(render_order_figure(summary, output_dir))
    manifest = {
        "summary": summary_path.as_posix(),
        "figure_count": len(figures),
        "figures": figures,
        "notes": [
            "All figures are rendered from experiment_summary.json.",
            "Generated-sample Frechet values are smoke metrics, not publication-scale FID.",
        ],
    }
    _write_json(output_dir / "figure_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = make_figures(Path(args.summary), Path(args.output_dir))
    print(f"wrote {manifest['figure_count']} figures")
    print(f"wrote {Path(args.output_dir) / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
