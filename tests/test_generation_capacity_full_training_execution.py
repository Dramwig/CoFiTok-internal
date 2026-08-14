from __future__ import annotations

from pathlib import Path

from scripts.run_generation_capacity_full_300k_training_supervisor import (
    SUPERVISOR_BOUNDARY,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_execute.sh"
)
SUPERVISOR_RUNBOOK = (
    ROOT / "artifacts/runbooks/generation_capacity_full_300k_training_supervisor.sh"
)
SUPERVISOR = ROOT / "scripts/run_generation_capacity_full_300k_training_supervisor.py"


def test_capacity_full_execution_is_fresh_receipted_and_milestoned() -> None:
    source = EXECUTION_RUNBOOK.read_text(encoding="utf-8")
    assert "verify_generation_capacity_full_300k_training_launch_receipt.py" in source
    assert '--authorization-gate "$TRAINING_LAUNCH_RECEIPT"' in source
    assert "for milestone in 50000 100000 200000 300000" in source
    assert "--required-checkpoint-steps 50000,100000,200000,300000" in source
    assert "--checkpoint-size-multiplier 1.0" in source
    assert "--sample-count 116640" in source
    assert "--expected-steps 300000" in source
    assert "--expected-dataset imagenet_256" in source
    assert "capacity_full_experimental" in source
    assert "stability_capacity_full_300k_v1" in source
    assert "stability_full_300k_ema_teacher" not in source
    assert "promotion_gate.json" not in source
    assert "base256_cofitok/checkpoint_step_00100000.pt" in source
    assert "--resume auto" in source
    assert "capacity_100k" not in source.split("train_to_milestone", 1)[1]


def test_capacity_full_supervisor_waits_and_never_signals_processes() -> None:
    wrapper = SUPERVISOR_RUNBOOK.read_text(encoding="utf-8")
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "flock -n" in wrapper
    assert "nice -n 19" in wrapper
    assert "run_generation_capacity_full_300k_training_supervisor.py" in wrapper
    assert "generation_capacity_full_300k_execute.sh" in source
    assert "required_idle_polls" in source
    assert "start_new_session=True" in source
    assert "build_generation_capacity_full_300k_training_launch_receipt.py" in source
    assert "verify_generation_capacity_full_300k_training_launch_receipt.py" in source
    assert "os.kill" not in source
    assert ".terminate(" not in source
    assert ".kill(" not in source
    assert "pkill" not in source
    assert SUPERVISOR_BOUNDARY[
        "fresh_matched_300k_training_launch_allowed_after_exact_receipt"
    ] is True
    assert SUPERVISOR_BOUNDARY["capacity_100k_checkpoint_resume_allowed"] is False
    assert SUPERVISOR_BOUNDARY["training_authorization_is_quality_promotion_gate"] is False
    assert SUPERVISOR_BOUNDARY["formal_generation_claim_allowed"] is False
    assert SUPERVISOR_BOUNDARY["release_authorization_allowed"] is False
    assert SUPERVISOR_BOUNDARY["process_signaling_allowed"] is False
