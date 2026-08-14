from __future__ import annotations

from pathlib import Path

import pytest

from cofitok.reporting import write_json_report
from scripts.wait_for_generation_capacity_full_300k_readiness_decision import (
    WAIT_BOUNDARY,
    _publish_immutable,
    _result_ready,
)


def test_capacity_full_readiness_waiter_is_cpu_only() -> None:
    assert WAIT_BOUNDARY["cpu_only_waiter"] is True
    assert WAIT_BOUNDARY["gpu_execution_allowed"] is False
    assert WAIT_BOUNDARY["readiness_gpu_benchmark_launch_allowed"] is False
    assert WAIT_BOUNDARY["training_launch_allowed"] is False
    assert WAIT_BOUNDARY["full_300k_launch_allowed"] is False
    assert WAIT_BOUNDARY["process_signaling_allowed"] is False


def test_capacity_full_readiness_waiter_requires_completed_result(
    tmp_path: Path,
) -> None:
    status = tmp_path / "status.json"
    result = tmp_path / "result.json"
    assert _result_ready(status, result) is None
    write_json_report(status, {"status": "waiting"})
    assert _result_ready(status, result) is None
    write_json_report(status, {"status": "completed"})
    with pytest.raises(RuntimeError, match="has no result"):
        _result_ready(status, result)
    write_json_report(result, {"status": "completed"})
    ready = _result_ready(status, result)
    assert ready is not None
    write_json_report(status, {"status": "failed", "detail": "fixture"})
    with pytest.raises(RuntimeError, match="terminal"):
        _result_ready(status, result)


def test_capacity_full_readiness_waiter_publishes_immutably(
    tmp_path: Path,
) -> None:
    target = tmp_path / "decision.json"
    payload = {"status": "authorized", "training_launch_allowed": False}
    _publish_immutable(target, payload, name="fixture decision")
    _publish_immutable(target, payload, name="fixture decision")
    with pytest.raises(ValueError, match="not byte-equivalent"):
        _publish_immutable(
            target,
            {"status": "authorized", "training_launch_allowed": True},
            name="fixture decision",
        )
