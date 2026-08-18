from __future__ import annotations

import subprocess
from collections.abc import Callable
from math import isfinite
from typing import Any


def wait_for_child_with_heartbeat(
    child: subprocess.Popen[Any],
    *,
    poll_seconds: float,
    heartbeat: Callable[[], None],
) -> int:
    """Wait for a child while periodically publishing a liveness heartbeat."""
    if not isfinite(poll_seconds) or poll_seconds <= 0.0:
        raise ValueError("child heartbeat poll interval must be finite and positive")
    while True:
        try:
            return int(child.wait(timeout=poll_seconds))
        except subprocess.TimeoutExpired:
            heartbeat()
