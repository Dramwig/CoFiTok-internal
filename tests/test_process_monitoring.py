from __future__ import annotations

import subprocess

import pytest

from cofitok.process_monitoring import wait_for_child_with_heartbeat


def test_wait_for_child_refreshes_heartbeat_until_completion() -> None:
    calls: list[float] = []

    class Child:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, *, timeout: float) -> int:
            self.wait_calls += 1
            if self.wait_calls < 3:
                raise subprocess.TimeoutExpired(["child"], timeout)
            return 0

    child = Child()
    result = wait_for_child_with_heartbeat(
        child,  # type: ignore[arg-type]
        poll_seconds=2.5,
        heartbeat=lambda: calls.append(2.5),
    )

    assert result == 0
    assert child.wait_calls == 3
    assert calls == [2.5, 2.5]


@pytest.mark.parametrize("poll_seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_wait_for_child_rejects_invalid_poll_interval(poll_seconds: float) -> None:
    with pytest.raises(ValueError, match="poll interval must be finite and positive"):
        wait_for_child_with_heartbeat(
            object(),  # type: ignore[arg-type]
            poll_seconds=poll_seconds,
            heartbeat=lambda: None,
        )
