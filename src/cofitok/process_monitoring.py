from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from math import isfinite
from typing import Any


def publish_child_heartbeat(
    heartbeat: Callable[[], None],
    *,
    error_reported: bool,
) -> bool:
    """Publish child liveness without letting a status-write failure detach it."""
    try:
        heartbeat()
    except Exception as error:
        if not error_reported:
            try:
                print(
                    f"child heartbeat failed; continuing supervision: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
        return True
    return False


def wait_for_child_with_heartbeat(
    child: subprocess.Popen[Any],
    *,
    poll_seconds: float,
    heartbeat: Callable[[], None],
) -> int:
    """Wait for a child while periodically publishing a liveness heartbeat."""
    if not isfinite(poll_seconds) or poll_seconds <= 0.0:
        raise ValueError("child heartbeat poll interval must be finite and positive")
    heartbeat_error_reported = False
    while True:
        try:
            return int(child.wait(timeout=poll_seconds))
        except subprocess.TimeoutExpired:
            heartbeat_error_reported = publish_child_heartbeat(
                heartbeat,
                error_reported=heartbeat_error_reported,
            )
