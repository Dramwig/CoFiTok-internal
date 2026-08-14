from __future__ import annotations

from pathlib import Path

import pytest

from cofitok.reporting import write_json_report
from scripts.wait_for_generation_capacity_completion_decision import (
    WAIT_BOUNDARY,
    _execution_completed,
    _publish_immutable,
    source_paths,
)


def test_capacity_completion_waiter_paths_and_boundary(tmp_path: Path) -> None:
    paths = source_paths(tmp_path / "capacity")
    assert paths["capacity_scaling_execution_status"].as_posix().endswith(
        "reports/capacity_scaling_50k_execution_status.json"
    )
    assert paths["cofitok_training_validation"].as_posix().endswith(
        "reports/capacity_scaling_50k/training/cofitok.json"
    )
    assert paths["capacity_scaling_result"].as_posix().endswith(
        "reports/capacity_scaling_50k_result.json"
    )
    assert paths["capacity_completion_decision"].as_posix().endswith(
        "reports/capacity_completion_100k_decision.json"
    )
    assert WAIT_BOUNDARY["cpu_only_waiter"] is True
    assert WAIT_BOUNDARY["gpu_execution_allowed"] is False
    assert WAIT_BOUNDARY["process_signaling_allowed"] is False
    assert WAIT_BOUNDARY["full_300k_launch_allowed"] is False


def test_capacity_completion_waiter_publishes_only_byte_equivalent_json(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    payload = {"status": "completed", "value": 1}
    _publish_immutable(target, payload, name="fixture result")
    _publish_immutable(target, payload, name="fixture result")
    with pytest.raises(ValueError, match="not byte-equivalent"):
        _publish_immutable(
            target,
            {"status": "completed", "value": 2},
            name="fixture result",
        )


def test_capacity_completion_waiter_requires_exact_completed_execution(
    tmp_path: Path,
) -> None:
    status = tmp_path / "execution.json"
    assert _execution_completed(status) is False
    write_json_report(status, {"status": "running", "stage": "training"})
    assert _execution_completed(status) is False
    write_json_report(status, {"status": "completed", "stage": "complete"})
    assert _execution_completed(status) is True
    write_json_report(status, {"status": "failed", "detail": "fixture"})
    with pytest.raises(RuntimeError, match="terminal"):
        _execution_completed(status)


def test_capacity_completion_waiter_entrypoint_imports() -> None:
    __import__("scripts.wait_for_generation_capacity_completion_decision")
