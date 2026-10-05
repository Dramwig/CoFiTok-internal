from __future__ import annotations

import os
from pathlib import Path


def reject_symlink_chain(path: str | Path, *, name: str) -> Path:
    """Return an absolute path only when it and every parent are real paths."""
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = absolute
    while True:
        if current.is_symlink():
            raise ValueError(f"{name} path must not contain a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    return absolute
