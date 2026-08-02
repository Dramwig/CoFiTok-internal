from __future__ import annotations

import errno
import json
import os
import socket
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


OUTPUT_LOCK_SCHEMA_VERSION = 1


class OutputLockError(RuntimeError):
    """Another process already owns an output target."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute_without_symlinks(path: str | Path, *, name: str) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = absolute
    while True:
        if current.is_symlink():
            raise ValueError(f"{name} path must not contain a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    return absolute


def output_lock_path(target: str | Path) -> Path:
    output = _absolute_without_symlinks(target, name="output lock target")
    if not output.name:
        raise ValueError("output lock target must not be a filesystem root")
    return output.parent / f".{output.name}.cofitok-output.lock"


def _open_lock(path: Path):
    _absolute_without_symlinks(path, name="output lock")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "r+b", buffering=0)


def _acquire(handle) -> None:
    if os.name == "nt":
        import msvcrt

        if os.fstat(handle.fileno()).st_size < 1:
            handle.seek(0)
            handle.write(b"\0")
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _is_contention(error: OSError) -> bool:
    return error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or getattr(
        error, "winerror", None
    ) in {32, 33, 36}


def _read_owner(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_owner(handle, *, target: Path, role: str) -> dict[str, object]:
    owner: dict[str, object] = {
        "schema_version": OUTPUT_LOCK_SCHEMA_VERSION,
        "role": role,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "target": target.as_posix(),
        "acquired_at": _utc_now(),
    }
    encoded = (json.dumps(owner, sort_keys=True) + "\n").encode("utf-8")
    handle.seek(0)
    handle.truncate(0)
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
    return owner


@contextmanager
def exclusive_output_lock(
    target: str | Path,
    *,
    role: str,
) -> Iterator[dict[str, object]]:
    """Hold one non-blocking OS lock for an output target.

    The lock file intentionally persists after release.  Exclusivity belongs to
    the open file descriptor, so a crashed process cannot leave a stale lock.
    """

    if not isinstance(role, str) or not role.strip():
        raise ValueError("output lock role must be a non-empty string")
    output = _absolute_without_symlinks(target, name="output lock target")
    lock_path = output_lock_path(output)
    _absolute_without_symlinks(lock_path.parent, name="output lock parent")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _absolute_without_symlinks(lock_path.parent, name="output lock parent")
    handle = _open_lock(lock_path)
    acquired = False
    try:
        try:
            _acquire(handle)
        except OSError as error:
            if not _is_contention(error):
                raise
            owner = _read_owner(lock_path)
            detail = ""
            if owner is not None:
                detail = (
                    f" (owner pid={owner.get('pid')} host={owner.get('hostname')} "
                    f"role={owner.get('role')})"
                )
            raise OutputLockError(
                f"output target is already locked: {output}{detail}"
            ) from error
        acquired = True
        yield _write_owner(handle, target=output, role=role.strip())
    finally:
        if acquired:
            _release(handle)
        handle.close()


@contextmanager
def exclusive_output_locks(
    targets: Iterable[str | Path],
    *,
    role: str,
) -> Iterator[list[dict[str, object]]]:
    """Hold non-blocking locks for multiple output targets in stable order.

    Stable acquisition order prevents two matched-output writers from taking
    opposite first locks.  ``ExitStack`` releases every previously acquired
    lock if a later target is contended or invalid.
    """

    outputs: list[Path] = []
    seen: set[str] = set()
    for target in targets:
        output = _absolute_without_symlinks(target, name="output lock target")
        key = os.path.normcase(str(output))
        if key in seen:
            raise ValueError(f"duplicate output lock target: {output}")
        seen.add(key)
        outputs.append(output)
    if not outputs:
        raise ValueError("at least one output lock target is required")

    ordered = sorted(outputs, key=lambda path: os.path.normcase(str(path)))
    with ExitStack() as stack:
        owners = [
            stack.enter_context(exclusive_output_lock(output, role=role))
            for output in ordered
        ]
        yield owners
