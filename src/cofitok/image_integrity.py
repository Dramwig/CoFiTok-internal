from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from PIL import Image


def sample_set_sha256(paths: Iterable[str | Path], chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    normalized = sorted((Path(path) for path in paths), key=lambda path: path.name)
    for path in normalized:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


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
