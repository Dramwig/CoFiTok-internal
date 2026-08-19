from __future__ import annotations

import subprocess
import sys
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
    heartbeat_error_reported = False
    while True:
        try:
            return int(child.wait(timeout=poll_seconds))
        except subprocess.TimeoutExpired:
            try:
                heartbeat()
            except Exception as error:
                # A status-write failure must not detach a still-running child.
                if not heartbeat_error_reported:
                    try:
                        print(
                            f"child heartbeat failed; continuing supervision: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    except Exception:
                        pass
                heartbeat_error_reported = True
            else:
                heartbeat_error_reported = False
