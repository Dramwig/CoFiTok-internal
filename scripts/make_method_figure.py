from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS_WIDTH = 1760
CANVAS_HEIGHT = 1080
METHOD_SUBTITLE = (
    "Dense pixel-space noise prediction is factorized into ordered restricted "
    "denoising components."
)


@dataclass(frozen=True)
class Box:
    key: str
    x: int
    y: int
    width: int
    height: int
    title: str
    lines: tuple[str, ...]
    fill: str
    border: str


@dataclass(frozen=True)
class Arrow:
    x1: int
    y1: int
    x2: int
    y2: int
    color: str = "#374151"
    dashed: bool = False
    label: str | None = None
    label_dx: int = 0
    label_dy: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the CoFiTok method overview figure.")
    parser.add_argument("--output-dir", required=True, help="Directory for method_overview PNG/SVG and manifest.")
    return parser.parse_args()


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


def build_method_scene() -> tuple[list[Box], list[Arrow]]:
    token_x = [850, 1010, 1170, 1330, 1490]
    token_titles = ("z_1", "z_2", "...", "z_m", "z_K")
    synthesis_titles = ("S_1", "S_2", "...", "S_m", "S_K")
    component_titles = ("eps_1", "eps_2", "...", "eps_m", "eps_K")

    boxes = [
        Box(
            "input",
            70,
            140,
            270,
            150,
            "Noisy state",
            ("x_t and timestep t", "optional condition stays in T"),
            "#eff6ff",
            "#2563eb",
        ),
        Box(
            "predictor",
            430,
            140,
            340,
            150,
            "Expressive predictor T",
            ("may use x_t, t, z_<k", "outputs ordered tokens"),
            "#f0fdf4",
            "#16a34a",
        ),
    ]

    for index, x in enumerate(token_x):
        boxes.append(
            Box(
                f"token_{index}",
                x,
                115,
                120,
                110,
                token_titles[index],
                ("ordered", "denoising token") if token_titles[index] != "..." else ("ordered", "sequence"),
                "#fff7ed",
                "#f97316",
            )
        )
        boxes.append(
            Box(
                f"synthesis_{index}",
                x,
                350,
                120,
                110,
                synthesis_titles[index],
                ("token only", "weak local map") if synthesis_titles[index] != "..." else ("same rule", "per token"),
                "#fdf2f8",
                "#db2777",
            )
        )
        boxes.append(
            Box(
                f"component_{index}",
                x,
                595,
                120,
                110,
                component_titles[index],
                ("dense", "component") if component_titles[index] != "..." else ("component", "sequence"),
                "#f8fafc",
                "#475569",
            )
        )

    boxes.extend(
        [
            Box(
                "prefix_sum",
                805,
                750,
                380,
                125,
                "Prefix sum",
                ("eps_hat_1:m = sum_{k<=m} S_k(z_k)", "valid partial noise prediction"),
                "#ecfeff",
                "#0891b2",
            ),
            Box(
                "full_sum",
                1305,
                750,
                360,
                125,
                "Full sum",
                ("eps_hat = sum_{k<=K} S_k(z_k)", "standard reverse update"),
                "#eef2ff",
                "#4f46e5",
            ),
            Box(
                "partial_result",
                805,
                920,
                380,
                110,
                "Partial denoising",
                ("x0_hat^m from eps_hat_1:m", "prefix-controllable result"),
                "#ecfdf5",
                "#059669",
            ),
            Box(
                "final_result",
                1305,
                920,
                360,
                110,
                "Final denoising",
                ("x0_hat^K from eps_hat", "pixel-space output"),
                "#eef2ff",
                "#4f46e5",
            ),
            Box(
                "restrictions",
                70,
                825,
                660,
                195,
                "Default S_k restrictions",
                (
                    "sees only current z_k",
                    "no x_t, t, label, prompt, z_<k, skips, or constants",
                    "bias-free shallow local synthesis; S_k(0)=0",
                    "tokens are noise components, not VAE image latents",
                ),
                "#fef2f2",
                "#dc2626",
            ),
        ]
    )

    arrows = [
        Arrow(340, 215, 430, 215),
        Arrow(770, 215, 850, 160, label="ordered z_k", label_dy=-28),
        Arrow(970, 160, 1010, 160),
        Arrow(1130, 160, 1170, 160),
        Arrow(1290, 160, 1330, 160),
        Arrow(1450, 160, 1490, 160),
        Arrow(930, 225, 930, 350),
        Arrow(1090, 225, 1090, 350),
        Arrow(1250, 225, 1250, 350),
        Arrow(1410, 225, 1410, 350),
        Arrow(1570, 225, 1570, 350),
        Arrow(930, 460, 930, 595),
        Arrow(1090, 460, 1090, 595),
        Arrow(1250, 460, 1250, 595),
        Arrow(1410, 460, 1410, 595),
        Arrow(1570, 460, 1570, 595),
        Arrow(930, 705, 910, 750),
        Arrow(1090, 705, 1015, 750),
        Arrow(1410, 705, 1085, 750, label="sum first m components", label_dx=-110, label_dy=20),
        Arrow(1570, 705, 1485, 750, label="sum all K components", label_dx=-130, label_dy=20),
        Arrow(995, 875, 995, 920),
        Arrow(1485, 875, 1485, 920),
        Arrow(
            320,
            290,
            930,
            350,
            color="#dc2626",
            dashed=True,
            label="blocked into S_k",
            label_dx=130,
            label_dy=14,
        ),
    ]
    return boxes, arrows


def _draw_dashed_line(draw: ImageDraw.ImageDraw, points: tuple[int, int, int, int], fill: str, width: int) -> None:
    x1, y1, x2, y2 = points
    dash = 12
    gap = 8
    total = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    dx = (x2 - x1) / total
    dy = (y2 - y1) / total
    cursor = 0
    while cursor < total:
        end = min(cursor + dash, total)
        draw.line(
            (x1 + dx * cursor, y1 + dy * cursor, x1 + dx * end, y1 + dy * end),
            fill=fill,
            width=width,
        )
        cursor += dash + gap


def _draw_arrowhead(draw: ImageDraw.ImageDraw, arrow: Arrow) -> None:
    dx = arrow.x2 - arrow.x1
    dy = arrow.y2 - arrow.y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    size = 12
    wing = 6
    points = [
        (arrow.x2, arrow.y2),
        (arrow.x2 - ux * size + px * wing, arrow.y2 - uy * size + py * wing),
        (arrow.x2 - ux * size - px * wing, arrow.y2 - uy * size - py * wing),
    ]
    draw.polygon(points, fill=arrow.color)


def _draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    title_font = _font(23, bold=True)
    body_font = _font(16)
    draw.rounded_rectangle(
        (box.x, box.y, box.x + box.width, box.y + box.height),
        radius=16,
        fill=box.fill,
        outline=box.border,
        width=3,
    )
    title_bbox = draw.textbbox((0, 0), box.title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text((box.x + (box.width - title_width) / 2, box.y + 18), box.title, fill="#111827", font=title_font)
    y = box.y + 58
    for line in box.lines:
        line_bbox = draw.textbbox((0, 0), line, font=body_font)
        line_width = line_bbox[2] - line_bbox[0]
        draw.text((box.x + (box.width - line_width) / 2, y), line, fill="#374151", font=body_font)
        y += 24


def render_method_png(output_path: Path, boxes: list[Box], arrows: list[Arrow]) -> None:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(38, bold=True)
    subtitle_font = _font(20)
    label_font = _font(15, bold=True)

    draw.text((60, 34), "CoFiTok method overview", fill="#111827", font=title_font)
    draw.text(
        (60, 82),
        METHOD_SUBTITLE,
        fill="#4b5563",
        font=subtitle_font,
    )

    for arrow in arrows:
        if arrow.dashed:
            _draw_dashed_line(draw, (arrow.x1, arrow.y1, arrow.x2, arrow.y2), arrow.color, 3)
        else:
            draw.line((arrow.x1, arrow.y1, arrow.x2, arrow.y2), fill=arrow.color, width=3)
        _draw_arrowhead(draw, arrow)

    for box in boxes:
        _draw_box(draw, box)

    for arrow in arrows:
        if arrow.label:
            label_x = (arrow.x1 + arrow.x2) / 2 + arrow.label_dx
            label_y = (arrow.y1 + arrow.y2) / 2 + arrow.label_dy
            draw.text((label_x, label_y), arrow.label, fill=arrow.color, font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _svg_text_block(x: int, y: int, width: int, title: str, lines: tuple[str, ...]) -> str:
    chunks = [
        f'<text x="{x + width / 2:.1f}" y="{y + 34}" text-anchor="middle" '
        'font-family="Arial, DejaVu Sans, sans-serif" font-size="23" font-weight="700" fill="#111827">'
        f"{escape(title)}</text>"
    ]
    for index, line in enumerate(lines):
        chunks.append(
            f'<text x="{x + width / 2:.1f}" y="{y + 68 + index * 24}" text-anchor="middle" '
            'font-family="Arial, DejaVu Sans, sans-serif" font-size="16" fill="#374151">'
            f"{escape(line)}</text>"
        )
    return "\n".join(chunks)


def render_method_svg(output_path: Path, boxes: list[Box], arrows: list[Arrow]) -> None:
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" '
        f'viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">',
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="context-stroke" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="60" y="64" font-family="Arial, DejaVu Sans, sans-serif" font-size="38" '
        'font-weight="700" fill="#111827">CoFiTok method overview</text>',
        '<text x="60" y="102" font-family="Arial, DejaVu Sans, sans-serif" font-size="20" '
        f'fill="#4b5563">{escape(METHOD_SUBTITLE)}</text>',
    ]
    for arrow in arrows:
        dash = ' stroke-dasharray="12 8"' if arrow.dashed else ""
        svg.append(
            f'<line x1="{arrow.x1}" y1="{arrow.y1}" x2="{arrow.x2}" y2="{arrow.y2}" '
            f'stroke="{arrow.color}" stroke-width="3" marker-end="url(#arrow)"{dash} />'
        )
        if arrow.label:
            label_x = (arrow.x1 + arrow.x2) / 2 + arrow.label_dx
            label_y = (arrow.y1 + arrow.y2) / 2 + arrow.label_dy
            svg.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-family="Arial, DejaVu Sans, sans-serif" '
                f'font-size="15" font-weight="700" fill="{arrow.color}">{escape(arrow.label)}</text>'
            )
    for box in boxes:
        svg.append(
            f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" rx="16" '
            f'fill="{box.fill}" stroke="{box.border}" stroke-width="3" />'
        )
        svg.append(_svg_text_block(box.x, box.y, box.width, box.title, box.lines))
    svg.append("</svg>\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def make_method_figure(output_dir: Path) -> dict[str, Any]:
    boxes, arrows = build_method_scene()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "method_overview.png"
    svg_path = output_dir / "method_overview.svg"
    render_method_png(png_path, boxes, arrows)
    render_method_svg(svg_path, boxes, arrows)
    manifest = {
        "artifact_count": 2,
        "canvas": {"width": CANVAS_WIDTH, "height": CANVAS_HEIGHT},
        "subtitle": METHOD_SUBTITLE,
        "figures": [
            {"path": png_path.as_posix(), "format": "png", "role": "raster draft"},
            {"path": svg_path.as_posix(), "format": "svg", "role": "vector draft"},
        ],
        "constraints": [
            "S_k receives only the current token z_k.",
            "S_k receives no x_t, timestep, label, prompt, previous tokens, skips, or learned constants.",
            "Default S_k is bias-free, shallow, local, and condition-free.",
            "S_k(0)=0 by construction for the restricted operator.",
            "CoFiTok tokens are dense noise components, not VAE-style image latents.",
        ],
        "notes": [
            "This schematic is a paper-facing method overview, not a numeric result.",
            "The expressive predictor T can be upgraded without changing the restricted S_k interface.",
            "The blocked dashed path marks information that must not enter S_k in the default method.",
        ],
    }
    _write_json(output_dir / "method_overview_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = make_method_figure(Path(args.output_dir))
    print(f"wrote {manifest['artifact_count']} method overview artifacts")
    print(f"wrote {Path(args.output_dir) / 'method_overview_manifest.json'}")


if __name__ == "__main__":
    main()
