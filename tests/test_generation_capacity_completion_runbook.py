from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_completion_100k_decision_waiter.sh"
)


def test_capacity_completion_waiter_runbook_is_cpu_only_and_bounded() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=\"\"" in source
    assert "nice -n 19" in source
    assert "flock -n" in source
    assert "wait_for_generation_capacity_completion_decision.py" in source
    assert "--expected-self-revision" in source
    assert "--expected-training-tree" in source
    assert "--expected-execution-tree" in source
    assert "scripts/train_generation.py" not in source
    assert "generate_samples.py" not in source
    assert "kill " not in source
    assert "pkill" not in source
    assert "nvidia-smi" not in source
