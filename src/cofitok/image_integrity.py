from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from PIL import Image


IMAGE_TREE_DIGEST_SCHEMA = "cofitok_image_tree_sha256_v1"


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


def image_tree_sha256(
    paths: Iterable[str | Path],
    *,
    root: str | Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Hash image bytes and root-relative paths with unambiguous framing."""

    declared_root = Path(root)
    if declared_root.is_symlink():
        raise ValueError(f"image tree root must not be a symlink: {declared_root}")
    root_path = declared_root.resolve()
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value)
        if path.is_symlink():
            raise ValueError(f"image tree contains a symlink: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root_path).as_posix()
        except ValueError as error:
            raise ValueError(f"image is outside the declared tree root: {path}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"image tree entry is not a file: {resolved}")
        if relative in seen:
            raise ValueError(f"image tree contains a duplicate entry: {relative}")
        seen.add(relative)
        normalized.append((relative, resolved))

    digest = hashlib.sha256()
    digest.update(IMAGE_TREE_DIGEST_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for relative, path in sorted(normalized):
        relative_bytes = relative.encode("utf-8")
        before = path.stat()
        size = before.st_size
        digest.update(len(relative_bytes).to_bytes(8, byteorder="big"))
        digest.update(relative_bytes)
        digest.update(size.to_bytes(8, byteorder="big"))
        bytes_read = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                bytes_read += len(block)
                digest.update(block)
        after = path.stat()
        if (
            bytes_read != size
            or after.st_size != size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise RuntimeError(f"image changed while hashing: {path}")
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
