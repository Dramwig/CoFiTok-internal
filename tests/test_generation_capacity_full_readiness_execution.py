from __future__ import annotations

from pathlib import Path

from scripts.run_generation_capacity_full_300k_readiness_supervisor import (
    SUPERVISOR_BOUNDARY,
)


ROOT = Path(__file__).resolve().parents[1]
READINESS_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_full_300k_readiness_after_decision.sh"
)
SUPERVISOR_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_readiness_supervisor.sh"
)
SUPERVISOR = ROOT / "scripts/run_generation_capacity_full_300k_readiness_supervisor.py"


def test_capacity_full_readiness_runbook_is_benchmark_only() -> None:
    source = READINESS_RUNBOOK.read_text(encoding="utf-8")
    assert "validate_generation_capacity_full_300k_readiness_decision.py" in source
    assert "select_generation_training_runtime.py" in source
    assert "build_generation_capacity_full_300k_readiness.py" in source
    assert "verify_generation_capacity_full_300k_readiness.py" in source
    assert "--checkpoint-size-multiplier 1.0" in source
    assert "--sample-count 116640" in source
    assert "base256_cofitok/checkpoint_step_00100000.pt" in source
    assert "base256_dense_identity/checkpoint_step_00100000.pt" in source
    assert "1x64,2x32,4x16,8x8,16x4" in source
    assert "nvidia-smi" in source
    assert "exit 9" in source
    assert "scripts/train_generation.py" not in source
    assert "full_matched_300k" not in source
    assert "kill " not in source
    assert "pkill" not in source


def test_capacity_full_readiness_supervisor_cannot_launch_training() -> None:
    wrapper = SUPERVISOR_RUNBOOK.read_text(encoding="utf-8")
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "flock -n" in wrapper
    assert "nice -n 19" in wrapper
    assert "run_generation_capacity_full_300k_readiness_supervisor.py" in wrapper
    assert "CUDA_VISIBLE_DEVICES=\"\"" not in wrapper
    assert "generation_capacity_full_300k_readiness_after_decision.sh" in source
    assert "scripts/train_generation.py" not in source
    assert "full_matched_300k" not in source
    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "pkill" not in source
    assert SUPERVISOR_BOUNDARY["training_launch_allowed_by_supervisor"] is False
    assert SUPERVISOR_BOUNDARY["full_300k_launch_allowed_by_supervisor"] is False
    assert SUPERVISOR_BOUNDARY["process_signaling_allowed"] is False
    assert SUPERVISOR_BOUNDARY["capacity_100k_checkpoint_resume_allowed"] is False
