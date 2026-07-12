"""Local filesystem subset of blobfile used by improved-diffusion.

The upstream code imports ``blobfile`` for both local and cloud paths. CoFiTok
baseline runs use only local paths, so this shim intentionally supports local
filesystem operations and fails clearly for unsupported behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Any


def join(*parts: str) -> str:
    return os.path.join(*parts)


def dirname(path: str) -> str:
    return os.path.dirname(path)


def basename(path: str) -> str:
    return os.path.basename(path)


def exists(path: str) -> bool:
    return os.path.exists(path)


def isdir(path: str) -> bool:
    return os.path.isdir(path)


def listdir(path: str) -> list[str]:
    return os.listdir(path)


def makedirs(path: str, exist_ok: bool = True) -> None:
    os.makedirs(path, exist_ok=exist_ok)


def BlobFile(path: str, mode: str = "r", *args: Any, **kwargs: Any) -> IO[Any]:
    Path(path).parent.mkdir(parents=True, exist_ok=True) if any(flag in mode for flag in "wax+") else None
    return open(path, mode, *args, **kwargs)
