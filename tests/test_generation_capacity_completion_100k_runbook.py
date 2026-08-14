from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_completion_250m_100k_execute.sh"
)
SUPERVISOR_RUNBOOK = (
    ROOT
    / "artifacts/runbooks/generation_capacity_completion_250m_100k_supervisor.sh"
)


def test_capacity_completion_runbook_is_exact_50k_to_100k_only() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    assert "--resume auto" in source
    assert "--stop-after-steps" in source
    assert "checkpoint_step_00050000.pt" in source
    assert "checkpoint_step_00100000.pt" in source
    assert "matched_250m_resume_to_100000_allowed':True" in source
    assert "additional_training_allowed':False" in source
    assert "full_300k_launch_allowed':False" in source
    assert "checkpoint_step_00300000.pt" not in source
    assert "full_matched_300k" not in source
    assert "--authorization-gate" not in source


def test_capacity_completion_runbook_binds_archived_source_and_training() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    for entrypoint in (
        "verify_generation_capacity_completion_decision.py",
        "archive_generation_capacity_completion_sources.py",
        "restore_generation_capacity_completion_sources.py",
        "verify_generation_capacity_completion_source_archive.py",
        "build_generation_capacity_completion_launch_receipt.py",
        "verify_generation_capacity_completion_launch_receipt.py",
        "validate_generation_capacity_completion_training.py",
        "verify_generation_capacity_completion_training.py",
    ):
        assert entrypoint in source
    assert "EXPECTED_DECISION_TREE" in source
    assert "EXPECTED_SCALING_EXECUTION_TREE" in source
    assert "EXPECTED_TRAINING_TREE" in source
    assert "EXPECTED_EXECUTION_TREE" in source
    assert "flock -n" in source


def test_capacity_completion_terminal_protocol_is_matched_and_source_bound() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")
    assert source.index("set_stage cofitok_training") < source.index(
        "set_stage dense_training"
    )
    assert "source-profile capacity_scaling" in source
    assert "samples_2048_ddim50_cfg15" in source
    assert "samples_10000_ddim100_cfg15" in source
    assert "--num-samples 10000" in source
    assert "--sample-steps 100" in source
    assert "--weights ema" in source
    assert "--precision bf16" in source
    assert "--guidance-scale 1.5" in source
    assert "--min-samples 10000" in source
    assert "evaluate_generation_class_fidelity.py" in source
    assert "new result required" in source


def test_capacity_completion_supervisor_wrapper_preserves_bounded_scope() -> None:
    source = SUPERVISOR_RUNBOOK.read_text(encoding="utf-8")
    assert "run_generation_capacity_completion_100k_supervisor.py" in source
    assert "--required-idle-polls" in source
    assert "--stall-seconds" in source
    assert "--termination-grace-seconds" in source
    assert "--expected-scaling-execution-tree" in source
    assert "--expected-training-tree" in source
    assert "train_generation.py" not in source
    assert "full_matched_300k" not in source
