from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_scaling_250m_50k_execute.sh"
)
SUPERVISOR_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_scaling_250m_50k_supervisor.sh"
)


def test_capacity_scaling_runbook_is_exact_resume_to_50k_only() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "--resume auto" in source
    assert source.count("verify_generation_capacity_scaling_launch_receipt.py") >= 2
    assert "--stop-after-steps" in source
    assert "checkpoint_step_00050000.pt" in source
    assert "source-profile capacity_scaling" in source
    assert "samples_2048_ddim50_cfg15" in source
    assert "configured_100k_completion_allowed':False" in source
    assert "full_300k_launch_allowed':False" in source
    assert "--authorization-gate" not in source
    assert "full_matched_300k" not in source
    assert "checkpoint_step_00100000.pt" not in source


def test_capacity_scaling_runbook_binds_all_source_replay_layers() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    for entrypoint in (
        "verify_generation_capacity_scaling_decision.py",
        "build_generation_capacity_scaling_launch_receipt.py",
        "verify_generation_capacity_scaling_launch_receipt.py",
        "validate_generation_capacity_scaling_training.py",
        "verify_generation_capacity_scaling_training.py",
        "build_generation_milestone_report.py",
        "validate_generation_milestone_report.py",
    ):
        assert entrypoint in source
    assert "EXPECTED_TRAINING_TREE" in source
    assert "EXPECTED_EXECUTION_TREE" in source
    assert "EXPECTED_DECISION_TREE" in source
    assert "flock -n" in source


def test_capacity_scaling_runbook_preserves_method_order_and_matched_protocol() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    assert source.index("set_stage cofitok_training") < source.index(
        "set_stage dense_training"
    )
    assert 'evaluate_50k cofitok "$COFITOK_RUN" 8 4' in source
    assert 'evaluate_50k dense_identity "$DENSE_RUN" 1 0' in source
    assert "--sample-count 4096" in source
    assert "--checkpoint-count 6" in source


def test_capacity_scaling_supervisor_wrapper_preserves_bounded_scope() -> None:
    source = SUPERVISOR_RUNBOOK.read_text(encoding="utf-8")
    assert "run_generation_capacity_scaling_50k_supervisor.py" in source
    assert "--required-idle-polls" in source
    assert "--stall-seconds" in source
    assert "--termination-grace-seconds" in source
    assert "--expected-training-tree" in source
    assert "train_generation.py" not in source
    assert "full_matched_300k" not in source
