from pathlib import Path

import pytest

from scripts import run_generation_factorization_quality_regression_supervisor as supervisor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    PROJECT_ROOT
    / "artifacts/runbooks/generation_factorization_quality_regression_probe_v1.sh"
)


def test_supervisor_parses_only_numeric_gpu_process_rows() -> None:
    assert supervisor.parse_gpu_process_pids("123\n456\n123\n") == [123, 456]
    assert supervisor.parse_gpu_process_pids("") == []

    with pytest.raises(ValueError, match="unparseable"):
        supervisor.parse_gpu_process_pids("not-a-pid\n")


def test_supervisor_authorization_boundary_is_diagnostic_only() -> None:
    boundary = supervisor.AUTHORIZATION_BOUNDARY

    assert boundary["standing_authorization_required"] is True
    assert boundary["terminal_system_guard_required"] is True
    assert boundary["five_consecutive_idle_gpu_polls_required"] is True
    assert boundary["unrelated_process_signaling_allowed"] is False
    assert boundary["training_launch_allowed"] is False
    assert boundary["followup_experiment_launch_allowed"] is False
    assert boundary["full_300k_launch_allowed"] is False
    assert boundary["release_authorization_allowed"] is False


def test_runbook_uses_exact_bounded_rollout_protocol() -> None:
    source = RUNBOOK.read_text(encoding="utf-8")

    assert source.count("scripts/evaluate_generation_rollout_stability.py") == 2
    assert source.count("--num-images 64") == 2
    assert source.count("--sample-steps 100") == 2
    assert source.count("--teacher-timesteps 999,900,750,500,250,100,10") == 2
    assert source.count("--guidance-scale 1.5") == 2
    assert source.count("--weights ema") == 5
    assert "for seed in 2029 2039" in source
    assert "scripts/build_generation_stability_qualification.py" in source
    assert "scripts/build_generation_factorization_quality_regression_report.py" in source
    assert "scripts/verify_generation_factorization_quality_regression_execution_authorization.py" in source
    assert "scripts/train_generation.py" not in source
    assert "scripts/generate_samples.py" not in source
    assert "full_300k" not in source


def test_supervisor_source_requires_exact_route_guard_and_five_idle_polls() -> None:
    source = Path(supervisor.__file__).read_text(encoding="utf-8")

    assert "classify_followup_decision" in source
    assert "terminal_system_guard" in source
    assert "validate_standing_experiment_authorization" in source
    assert "args.required_idle_polls != 5" in source
    assert "output_root.exists()" in source
    assert "gpu_compute_pids()" in source
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert "os.kill" not in source
    assert "terminate()" not in source
    assert "kill()" not in source
