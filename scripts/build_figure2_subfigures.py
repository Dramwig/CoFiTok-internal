#!/usr/bin/env python
"""Build the four standalone bitmap subfigures used by main-paper Figure 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class SubfigureSpec:
    key: str
    filename: str
    left_label: str
    right_label: str
    left_source: str
    right_source: str


SUBFIGURE_SPECS = (
    SubfigureSpec(
        key="a",
        filename="figure2a_tiny_prefix.png",
        left_label="Tiny ImageNet endpoint-only",
        right_label="Tiny ImageNet CoFiTok K8",
        left_source="train_tiny_imagenet_k8_epsilononly_p150eval_20k_2026-07-08/prefix_final.png",
        right_source="train_tiny_imagenet_k8_denoisepath_p150_light_20k_2026-07-08/prefix_final.png",
    ),
    SubfigureSpec(
        key="b",
        filename="figure2b_imagenet64_prefix.png",
        left_label="ImageNet-64 endpoint-only",
        right_label="ImageNet-64 CoFiTok K8",
        left_source="train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_2026-07-08/prefix_final.png",
        right_source="train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_2026-07-08/prefix_final.png",
    ),
    SubfigureSpec(
        key="c",
        filename="figure2c_restricted_synthesis.png",
        left_label="Restricted synthesis ordered",
        right_label="Restricted synthesis shuffled",
        left_source="eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_ordered_2026-07-08/prefix_final.png",
        right_source="eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_5k_seed2_ordered_2026-07-08/prefix_final_shuffled.png",
    ),
    SubfigureSpec(
        key="d",
        filename="figure2d_deep_synthesis.png",
        left_label="Deep synthesis ordered",
        right_label="Deep synthesis shuffled",
        left_source="eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_ordered_2026-07-08/prefix_final.png",
        right_source="eval_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_ordered_2026-07-08/prefix_final_shuffled.png",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap", type=int, default=12, help="White pixel gap between the left and right grids.")
    return parser.parse_args()


def _open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _pair_images(left: Image.Image, right: Image.Image, gap: int) -> Image.Image:
    if gap < 0:
        raise ValueError("gap must be >= 0")
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + gap + right.width, height), "white")
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + gap, (height - right.height) // 2))
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_subfigures(reports_root: Path, output_dir: Path, gap: int = 12) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in SUBFIGURE_SPECS:
        left_path = reports_root / spec.left_source
        right_path = reports_root / spec.right_source
        image = _pair_images(_open_rgb(left_path), _open_rgb(right_path), gap)
        output_path = output_dir / spec.filename
        image.save(output_path)
        rows.append(
            {
                "key": spec.key,
                "path": output_path.as_posix(),
                "width": image.width,
                "height": image.height,
                "left": {"label": spec.left_label, "source": left_path.as_posix()},
                "right": {"label": spec.right_label, "source": right_path.as_posix()},
                "sha256": _sha256(output_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "figure": 2,
        "subfigure_count": len(rows),
        "latex_layout": "2x2 minipages; labels and captions are rendered by LaTeX",
        "gap_pixels": gap,
        "subfigures": rows,
        "notes": [
            "Each output is a standalone bitmap subfigure, not a precomposed full Figure 2.",
            "No title, metric card, subfigure letter, or caption text is baked into the bitmap.",
            "Within each bitmap, the left/right experimental conditions are recorded in this manifest and the LaTeX caption.",
        ],
    }
    manifest_path = output_dir / "figure2_subfigures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_subfigures(args.reports_root, args.output_dir, gap=args.gap)
    print(f"wrote {manifest['subfigure_count']} Figure 2 subfigures")


if __name__ == "__main__":
    main()
