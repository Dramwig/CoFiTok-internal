from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final visual claim panel from existing evidence artifacts.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--visual-panels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _open_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[tuple[str, ImageFont.ImageFont, str]],
    line_gap: int = 7,
) -> int:
    x, y = xy
    for text, font, fill in lines:
        draw.text((x, y), text, font=font, fill=fill)
        box = draw.textbbox((x, y), text, font=font)
        y = box[3] + line_gap
    return y


def _card(
    canvas: Image.Image,
    xy: tuple[int, int, int, int],
    title: str,
    value: str,
    detail: str,
    accent: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=18, fill="#f8fafc", outline="#d1d5db", width=2)
    draw.rectangle((x0, y0, x0 + 10, y1), fill=accent)
    title_font = _font(24, bold=True)
    value_font = _font(44, bold=True)
    detail_font = _font(20)
    _draw_text_block(
        draw,
        (x0 + 28, y0 + 22),
        [
            (title, title_font, "#111827"),
            (value, value_font, accent),
            (detail, detail_font, "#4b5563"),
        ],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_panel(evidence_path: Path, visual_panels_dir: Path, output_dir: Path) -> dict[str, Any]:
    evidence = _read_json(evidence_path)
    generation = evidence["generation_summary"]
    diagnostics = evidence["diagnostic_summary"]
    core_ready = bool(
        evidence.get("claim_stance", {}).get("core_empirical_gates_ready")
    )
    subtitle = (
        "Core empirical gates support prefix-controllable dense-noise factorization; generation SOTA is not the claim."
        if core_ready
        else "Core empirical gates remain pending or failed; generation SOTA is not the claim."
    )

    prefix = _resize_width(_open_rgb(visual_panels_dir / "prefix_comparison_20k.png"), 720)
    sk = _resize_width(_open_rgb(visual_panels_dir / "sk_diagnostic_panel.png"), 720)

    margin = 42
    gap = 34
    title_h = 108
    card_h = 166
    label_h = 44
    body_top = margin + title_h + card_h + gap
    canvas_w = margin * 2 + prefix.width + gap + sk.width
    body_h = label_h + max(prefix.height, sk.height)
    canvas_h = body_top + body_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(38, bold=True)
    subtitle_font = _font(22)
    label_font = _font(24, bold=True)

    draw.text((margin, 28), "CoFiTok Evidence for the Scoped Claim", font=title_font, fill="#111827")
    draw.text(
        (margin, 76),
        subtitle,
        font=subtitle_font,
        fill="#4b5563",
    )

    card_gap = 24
    card_w = (canvas_w - margin * 2 - card_gap * 2) // 3
    y_card = margin + title_h
    _card(
        canvas,
        (margin, y_card, margin + card_w, y_card + card_h),
        "Prefix control",
        f"{diagnostics['cofitok_path_auc_better_than_endpoint_only']}/{diagnostics['dataset_count']}",
        "beat endpoint-only path AUC",
        "#2563eb",
    )
    _card(
        canvas,
        (margin + card_w + card_gap, y_card, margin + 2 * card_w + card_gap, y_card + card_h),
        "Restricted synthesis",
        "0 -> 0",
        f"deep-S leaks on {diagnostics['deep_synthesis_nonzero_zero_ratio_count']}/{diagnostics['dataset_count']}",
        "#059669",
    )
    _card(
        canvas,
        (margin + 2 * (card_w + card_gap), y_card, canvas_w - margin, y_card + card_h),
        "Generation metrics",
        f"{generation['cofitok_best_lowres_count']}/{generation['dataset_count']}",
        "lowres wins; do not claim SOTA",
        "#b45309",
    )

    x_left = margin
    x_right = margin + prefix.width + gap
    y_label = body_top
    draw.text((x_left, y_label), "Prefix denoising evidence", font=label_font, fill="#111827")
    draw.text((x_right, y_label), "Shuffle and deep-S_k diagnostics", font=label_font, fill="#111827")
    canvas.paste(prefix, (x_left, y_label + label_h))
    canvas.paste(sk, (x_right, y_label + label_h))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cofitok_final_visual_claim_panel.png"
    canvas.save(output_path)
    manifest = {
        "path": output_path.as_posix(),
        "inputs": {
            "evidence": evidence_path.as_posix(),
            "prefix_panel": (visual_panels_dir / "prefix_comparison_20k.png").as_posix(),
            "sk_diagnostic_panel": (visual_panels_dir / "sk_diagnostic_panel.png").as_posix(),
        },
        "metrics": {
            "core_empirical_gates_ready": core_ready,
            "cofitok_path_auc_better_than_endpoint_only": diagnostics[
                "cofitok_path_auc_better_than_endpoint_only"
            ],
            "dataset_count": diagnostics["dataset_count"],
            "deep_synthesis_nonzero_zero_ratio_count": diagnostics["deep_synthesis_nonzero_zero_ratio_count"],
            "cofitok_best_lowres_count": generation["cofitok_best_lowres_count"],
        },
        "notes": [
            "This panel summarizes visual evidence and machine-checked metrics from paper_evidence_report.json.",
            (
                "It supports the scoped claim: prefix-controllable dense-noise factorization, not generation SOTA."
                if core_ready
                else "It records evidence while the scoped claim remains pending; generation SOTA is not supported."
            ),
        ],
    }
    _write_json(output_dir / "final_visual_claim_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_panel(args.evidence, args.visual_panels_dir, args.output_dir)
    print(f"wrote {manifest['path']}")


if __name__ == "__main__":
    main()
