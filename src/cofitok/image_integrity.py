from __future__ import annotations

from pathlib import Path

from PIL import Image


def is_valid_png(
    path: str | Path,
    *,
    width: int,
    height: int,
    channels: int,
) -> bool:
    expected_mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
    if expected_mode is None:
        raise ValueError("PNG integrity checks support 1, 3, or 4 image channels")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return (
                image.format == "PNG"
                and image.size == (width, height)
                and image.mode == expected_mode
            )
    except (OSError, SyntaxError, ValueError):
        return False
