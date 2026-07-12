from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export generated image arrays from an NPZ file to PNG images.")
    parser.add_argument("--input", required=True, type=Path, help="Input .npz file, usually samples_*.npz.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where PNG files will be written.")
    parser.add_argument("--key", default="arr_0", help="NPZ array key containing images.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to export; 0 exports all images.")
    parser.add_argument("--prefix", default="sample", help="Output filename prefix.")
    return parser.parse_args()


def _as_nhwc_uint8(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"expected a 4D image array, got shape {array.shape}")

    if array.shape[-1] in {1, 3, 4}:
        nhwc = array
    elif array.shape[1] in {1, 3, 4}:
        nhwc = np.transpose(array, (0, 2, 3, 1))
    else:
        raise ValueError(f"cannot infer channel axis for shape {array.shape}")

    if nhwc.dtype != np.uint8:
        if np.issubdtype(nhwc.dtype, np.floating):
            if nhwc.min() >= -1.0 and nhwc.max() <= 1.0:
                nhwc = (nhwc + 1.0) * 127.5
            elif nhwc.min() >= 0.0 and nhwc.max() <= 1.0:
                nhwc = nhwc * 255.0
        nhwc = np.clip(nhwc, 0, 255).astype(np.uint8)
    return nhwc


def export_npz_to_png(input_path: Path, output_dir: Path, key: str, limit: int, prefix: str) -> dict[str, Any]:
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if not input_path.is_file():
        raise FileNotFoundError(f"input NPZ does not exist: {input_path}")

    with np.load(input_path) as payload:
        if key not in payload:
            raise KeyError(f"key {key!r} not found in {input_path}; available keys: {sorted(payload.files)}")
        images = _as_nhwc_uint8(payload[key])

    count = int(images.shape[0] if limit == 0 else min(limit, images.shape[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images[:count]):
        if image.shape[-1] == 1:
            pil_image = Image.fromarray(image[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(image[:, :, :3], mode="RGB")
        pil_image.save(output_dir / f"{prefix}_{index:06d}.png")

    manifest = {
        "input": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "key": key,
        "available_images": int(images.shape[0]),
        "exported_images": count,
        "shape": list(images.shape),
        "dtype": "uint8",
    }
    (output_dir / "npz_to_png_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = export_npz_to_png(args.input, args.output_dir, args.key, args.limit, args.prefix)
    print(f"wrote {manifest['exported_images']} PNG images to {manifest['output_dir']}")


if __name__ == "__main__":
    main()
