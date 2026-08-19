from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "expected_guarded_publications"),
    [
        ("scripts/run_generation_capacity_scaling_50k_supervisor.py", 2),
        ("scripts/run_generation_capacity_completion_100k_supervisor.py", 2),
        ("scripts/run_generation_capacity_full_300k_readiness_supervisor.py", 1),
        ("scripts/run_generation_capacity_full_300k_training_supervisor.py", 1),
        ("scripts/run_generation_capacity_full_300k_posteval_supervisor.py", 1),
        ("scripts/run_generation_capacity_full_300k_finalization_supervisor.py", 1),
    ],
)
def test_capacity_child_loops_keep_supervision_when_heartbeat_write_fails(
    relative_path: str,
    expected_guarded_publications: int,
) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    ast.parse(source)
    assert "from cofitok.process_monitoring import publish_child_heartbeat" in source
    assert "while child.poll() is None:" in source
    assert (
        source.count("heartbeat_error_reported = publish_child_heartbeat(")
        == expected_guarded_publications
    )
