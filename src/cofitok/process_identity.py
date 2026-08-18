from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


LINUX_PROCESS_IDENTITY_FIELDS = (
    "pid",
    "start_ticks",
    "executable",
    "cwd",
    "cmdline_sha256",
)


def read_linux_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any] | None:
    if pid < 1:
        return None
    process = proc_root / str(pid)
    if not process.is_dir():
        return None
    try:
        raw_cmdline = (process / "cmdline").read_bytes()
        argv = raw_cmdline.replace(b"\0", b" ").decode(
            errors="replace"
        ).strip()
        stat = (process / "stat").read_text(encoding="utf-8").rsplit(
            ")", 1
        )[1].split()
        executable = os.readlink(process / "exe")
        cwd = os.readlink(process / "cwd")
        start_ticks = int(stat[19])
    except (FileNotFoundError, IndexError, OSError, PermissionError, ValueError):
        return None
    if not argv:
        return None
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "argv": argv,
        "cwd": cwd,
        "executable": executable,
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
    }


def linux_process_identity_mismatches(
    observed: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, Any],
) -> list[str]:
    if observed is None:
        return ["process_missing"]
    return [
        field
        for field in LINUX_PROCESS_IDENTITY_FIELDS
        if observed.get(field) != expected.get(field)
    ]
