from __future__ import annotations

import subprocess

import pytest

from cofitok.process_monitoring import (
    publish_child_heartbeat,
    wait_for_child_with_heartbeat,
)


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


def test_publish_child_heartbeat_suppresses_repeated_write_errors_and_recovers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def flaky_heartbeat() -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("status disk is full")

    error_reported = publish_child_heartbeat(
        flaky_heartbeat,
        error_reported=False,
    )
    assert error_reported is True
    error_reported = publish_child_heartbeat(
        flaky_heartbeat,
        error_reported=error_reported,
    )
    assert error_reported is True
    error_reported = publish_child_heartbeat(
        flaky_heartbeat,
        error_reported=error_reported,
    )
    assert error_reported is False
    assert calls == 3
    assert capsys.readouterr().err.count("status disk is full") == 1
