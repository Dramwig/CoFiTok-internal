from __future__ import annotations

from pathlib import Path

import pytest

from cofitok.generation.capacity_probe_execution import (
    STANDING_AUTHORIZATION_EXACT_TEXT,
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import write_json_report
from scripts.wait_for_generation_capacity_completion_100k_result import (
    WAIT_BOUNDARY,
    _execution_completed,
    _publish_immutable,
    _standing_authorization_evidence,
)


def test_capacity_completion_result_waiter_is_cpu_only_and_non_authorizing() -> None:
    assert WAIT_BOUNDARY["cpu_only_waiter"] is True
    assert WAIT_BOUNDARY["gpu_execution_allowed"] is False
    assert WAIT_BOUNDARY["training_launch_allowed"] is False
    assert WAIT_BOUNDARY["process_signaling_allowed"] is False
    assert WAIT_BOUNDARY["full_300k_launch_allowed"] is False
    assert WAIT_BOUNDARY["new_source_compatible_gate_or_decision_required"] is True


def test_capacity_completion_result_waiter_requires_exact_completed_execution(
    tmp_path: Path,
) -> None:
    status = tmp_path / "execution.json"
    assert _execution_completed(status) is False
    write_json_report(status, {"status": "running", "stage": "terminal_eval"})
    assert _execution_completed(status) is False
    write_json_report(status, {"status": "completed", "stage": "complete"})
    assert _execution_completed(status) is True
    write_json_report(status, {"status": "failed", "detail": "fixture"})
    with pytest.raises(RuntimeError, match="terminal"):
        _execution_completed(status)


def test_capacity_completion_result_waiter_publishes_only_byte_equivalent_json(
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


def test_capacity_completion_result_waiter_replays_standing_authorization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "standing_authorization.json"
    write_json_report(
        path,
        {
            "schema_version": 1,
            "role": STANDING_AUTHORIZATION_ROLE,
            "status": "active",
            "instruction": {
                "language": "zh-CN",
                "exact_text": STANDING_AUTHORIZATION_EXACT_TEXT,
                "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
                "received_at": "2026-08-14T00:00:00+00:00",
            },
            "preserved_safety_boundaries": (
                STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
            ),
        },
    )
    identity = file_identity(path)
    evidence = _standing_authorization_evidence(
        path,
        expected_sha256=identity["sha256"],
    )
    assert evidence["source"] == identity
    assert evidence["validated_record"]["status"] == "active"
    with pytest.raises(ValueError, match="SHA256 differs"):
        _standing_authorization_evidence(path, expected_sha256="0" * 64)
