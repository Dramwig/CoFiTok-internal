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


def test_wait_for_child_rejects_nonpositive_poll_interval() -> None:
    with pytest.raises(ValueError, match="poll interval must be positive"):
        wait_for_child_with_heartbeat(
            object(),  # type: ignore[arg-type]
            poll_seconds=0.0,
            heartbeat=lambda: None,
        )


def test_wait_for_child_does_not_detach_when_heartbeat_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    heartbeat_calls = 0

    class Child:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, *, timeout: float) -> int:
            self.wait_calls += 1
            if self.wait_calls < 3:
                raise subprocess.TimeoutExpired(["child"], timeout)
            return 7

    def failed_heartbeat() -> None:
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        raise OSError("status disk is full")

    child = Child()
    result = wait_for_child_with_heartbeat(
        child,  # type: ignore[arg-type]
        poll_seconds=1.0,
        heartbeat=failed_heartbeat,
    )

    assert result == 7
    assert child.wait_calls == 3
    assert heartbeat_calls == 2
    assert capsys.readouterr().err.count("status disk is full") == 1
