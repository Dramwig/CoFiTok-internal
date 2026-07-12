from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class PanelItem:
    label: str
    relative_path: str


PREFIX_COMPARISON_ITEMS = [
    PanelItem("Tiny / endpoint-only factorized", "train_tiny_imagenet_k8_epsilononly_p150eval_20k_2026-07-08/prefix_final.png"),
    PanelItem("Tiny / CoFiTok K8", "train_tiny_imagenet_k8_denoisepath_p150_light_20k_2026-07-08/prefix_final.png"),
    PanelItem("ImageNet-64 / endpoint-only factorized", "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_2026-07-08/prefix_final.png"),
    PanelItem("ImageNet-64 / CoFiTok K8", "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_2026-07-08/prefix_final.png"),
]

ORDER_ABLATION_ITEMS = [
    PanelItem("ordered", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_ordered_2026-07-08/prefix_final.png"),
    PanelItem("random order", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_random0_2026-07-08/prefix_final.png"),
    PanelItem("reverse order", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_reverse_2026-07-08/prefix_final.png"),
]

SK_DIAGNOSTIC_ITEMS = [
    PanelItem("restricted S_k / ordered", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_ordered_2026-07-08/prefix_final.png"),
    PanelItem("restricted S_k / shuffled", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_ordered_2026-07-08/prefix_final_shuffled.png"),
    PanelItem("deep S_k / ordered", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_ordered_2026-07-08/prefix_final.png"),
    PanelItem("deep S_k / shuffled", "eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_ordered_2026-07-08/prefix_final_shuffled.png"),
]

SAMPLING_ITEMS = [
    PanelItem("Tiny / endpoint-only factorized", "generated_tiny_epsilononly_20k_256_ddim20_2026-07-08/samples_prefix_8.png"),
    PanelItem("Tiny / CoFiTok K8", "generated_tiny_imagenet_k8_light_20k_256_ddim20_2026-07-08/samples_prefix_8.png"),
    PanelItem("ImageNet-64 / endpoint-only factorized", "generated_imagenet_hf_epsilononly_20k_256_ddim20_2026-07-08/samples_prefix_8.png"),
    PanelItem("ImageNet-64 / CoFiTok K8", "generated_imagenet_hf_k8_light_20k_256_ddim20_2026-07-08/samples_prefix_8.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose paper-facing visual panels from CoFiTok report PNG artifacts.")
    parser.add_argument("--reports-root", default="artifacts/reports", help="Root directory containing report image artifacts.")
    parser.add_argument("--output-dir", required=True, help="Directory for visual panel PNGs and manifest JSON.")
    parser.add_argument(
        "--sample-crop-width",
        type=int,
        default=1058,
        help="Left crop width for wide sample grids; default keeps the first 16 columns from 64x64 grids.",
    )
    return parser.parse_args()


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _open_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing visual source image: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _crop_left(image: Image.Image, crop_width: int | None) -> Image.Image:
    if crop_width is None or crop_width <= 0 or image.width <= crop_width:
        return image
    return image.crop((0, 0, crop_width, image.height))


def _resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _center_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x0, y0, x1, y1 = xy
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2), text, font=font, fill=fill)


def _render_panel(
    *,
    reports_root: Path,
    output_path: Path,
    title: str,
    subtitle: str,
    items: list[PanelItem],
    columns: int,
    tile_width: int,
    crop_width: int | None = None,
) -> dict[str, Any]:
    images = []
    for item in items:
        source = reports_root / item.relative_path
        image = _resize_to_width(_crop_left(_open_rgb(source), crop_width), tile_width)
        images.append((item, source, image))

    margin = 34
    gap = 24
    title_height = 84
    label_height = 34
    rows = (len(images) + columns - 1) // columns
    max_image_height = max(image.height for _, _, image in images)
    cell_height = label_height + max_image_height
    canvas_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    canvas_height = margin + title_height + rows * cell_height + (rows - 1) * gap + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28, bold=True)
    subtitle_font = _font(17)
    label_font = _font(16, bold=True)

    draw.text((margin, 24), title, font=title_font, fill="#111827")
    draw.text((margin, 60), subtitle, font=subtitle_font, fill="#4b5563")

    y0 = margin + title_height
    for index, (item, _source, image) in enumerate(images):
        row = index // columns
        column = index % columns
        x = margin + column * (tile_width + gap)
        y = y0 + row * (cell_height + gap)
        _center_text(draw, (x, y, x + tile_width, y + label_height), item.label, label_font, "#111827")
        canvas.paste(image, (x, y + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": output_path.as_posix(),
        "title": title,
        "source_images": [str(source.as_posix()) for _, source, _ in images],
    }


def make_visual_panels(reports_root: Path, output_dir: Path, sample_crop_width: int = 1058) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        _render_panel(
            reports_root=reports_root,
            output_path=output_dir / "prefix_comparison_20k.png",
            title="Prefix Denoising at 20k Steps",
            subtitle="Rows show real images followed by prefix reconstructions. K8 light is trained for meaningful prefixes.",
            items=PREFIX_COMPARISON_ITEMS,
            columns=2,
            tile_width=430,
        ),
        _render_panel(
            reports_root=reports_root,
            output_path=output_dir / "order_ablation_prefix_panel.png",
            title="Order Ablation Prefix Grids",
            subtitle="Endpoint sums are similar, but prefix progression depends on component order.",
            items=ORDER_ABLATION_ITEMS,
            columns=3,
            tile_width=390,
        ),
        _render_panel(
            reports_root=reports_root,
            output_path=output_dir / "sk_diagnostic_panel.png",
            title="Restricted vs Deep S_k Diagnostics",
            subtitle="Restricted S_k keeps token-only, bias-free synthesis; deep S_k is an ablation for degeneration risk.",
            items=SK_DIAGNOSTIC_ITEMS,
            columns=2,
            tile_width=430,
        ),
        _render_panel(
            reports_root=reports_root,
            output_path=output_dir / "sampling_comparison_20k.png",
            title="Generated Samples at 20k Steps",
            subtitle="First 16 columns from DDIM-20 sample grids. Distribution metrics remain smoke-level.",
            items=SAMPLING_ITEMS,
            columns=1,
            tile_width=960,
            crop_width=sample_crop_width,
        ),
    ]
    manifest = {
        "panel_count": len(panels),
        "panels": panels,
        "reports_root": reports_root.as_posix(),
        "notes": [
            "Panels are composed from existing report PNG artifacts.",
            "Generated-sample panels are qualitative smoke evidence, not formal FID evidence.",
            "Deep S_k panels are ablation evidence and should not be presented as the default method.",
        ],
    }
    _write_json(output_dir / "visual_panel_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = make_visual_panels(Path(args.reports_root), Path(args.output_dir), sample_crop_width=args.sample_crop_width)
    print(f"wrote {manifest['panel_count']} visual panels")
    print(f"wrote {Path(args.output_dir) / 'visual_panel_manifest.json'}")


if __name__ == "__main__":
    main()
