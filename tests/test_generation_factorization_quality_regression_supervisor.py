from argparse import Namespace
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
    assert "deployment_receipt_output" in source
    assert "build_deployment_receipt" in source
    assert "os.kill" not in source
    assert "terminate()" not in source
    assert "kill()" not in source


def test_supervisor_deployment_receipt_binds_exact_control_plane(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    args = Namespace(
        project=PROJECT_ROOT,
        preparation=control_root / "preparation.json",
        standing_authorization=tmp_path / "standing.json",
        followup_decision=tmp_path / "followup.json",
        terminal_system_guard=tmp_path / "terminal.json",
        source_binding=control_root / "source_binding.json",
        execution_authorization=control_root / "execution_authorization.json",
        output_root=Path(supervisor.OUTPUT_ROOT),
        status_output=control_root / "status.json",
        pid_file=control_root / "supervisor.pid.json",
        deployment_receipt_output=control_root / "deployment_receipt.json",
        poll_seconds=60.0,
        required_idle_polls=5,
        timeout_seconds=2_592_000.0,
    )
    git = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "branch": "scale/generation-factorization-quality-regression-v1",
        "tracked_dirty": False,
    }

    receipt = supervisor.build_deployment_receipt(
        args,
        control_git=git,
        supervisor_source={"path": "/source.py", "bytes": 1, "sha256": "1" * 64},
        runbook={"path": "/runbook.sh", "bytes": 2, "sha256": "2" * 64},
        python_runtime={"path": "/python", "bytes": 3, "sha256": "3" * 64},
        preparation={"path": "/preparation.json", "bytes": 4, "sha256": "4" * 64},
        standing_authorization={
            "path": "/standing.json",
            "bytes": 5,
            "sha256": "5" * 64,
        },
    )

    assert receipt["role"] == supervisor.DEPLOYMENT_ROLE
    assert receipt["control"]["checkout"]["revision"] == "a" * 40
    assert receipt["targets"]["control_root"] == control_root.resolve().as_posix()
    assert receipt["targets"]["diagnostic_output_root"] == (
        args.output_root.resolve().as_posix()
    )
    assert receipt["timing"]["required_idle_gpu_polls"] == 5
    assert receipt["authorization_boundary"] == supervisor.AUTHORIZATION_BOUNDARY
    assert receipt["authorization_boundary"]["training_launch_allowed"] is False
