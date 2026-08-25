from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.monitor_generation_min_snr_pilot import (
    EXPECTED_RESULT_BOUNDARY,
    NON_AUTHORIZING_BOUNDARY,
    classify_status,
    evaluate_process_health,
    fdinfo_declares_write_flock,
    inspect_metrics,
    inspect_result,
    inspect_run_manifest,
)


REVISION = "8" * 40
BRANCH = "scale/generation-min-snr-matched-pilot-v1-20260825"


def _write_manifest(path: Path, *, method: str) -> None:
    model = (
        {
            "synthesis_mode": "fixed_basis",
            "token_count": 8,
            "predictor_use_feedback": True,
        }
        if method == "cofitok"
        else {
            "synthesis_mode": "dense_identity",
            "token_count": 1,
            "predictor_use_feedback": False,
        }
    )
    path.write_text(
        json.dumps(
            {
                "config": {
                    "runtime": {
                        "steps": 100_000,
                        "checkpoint_interval": 5_000,
                        "seed": 2027,
                    },
                    "loss": {"min_snr_gamma": 5.0},
                    "diffusion": {"prediction_target": "epsilon"},
                    "data": {"dataset": "imagenet_256", "batch_size": 64},
                    "optimization": {"gradient_accumulation_steps": 1},
                    "model": model,
                },
                "git": {"revision": REVISION, "branch": BRANCH, "dirty": False},
                "runtime_environment_sha256": (
                    "d5bfcd085ea467ee5d24dfccc6e147da06dd7a0a0efdcdab355882b8547c985e"
                ),
                "dataset_provenance": {
                    "identity_sha256": (
                        "6ec1d96ac3cd8a41fc66c40d424bf8e005c6a08bf9f580f5379c93772c8fe659"
                    )
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _metric(step: int) -> dict:
    return {
        "step": step,
        "samples_seen": step * 64,
        "total": 0.5,
        "epsilon": 0.4,
        "epsilon_unweighted": 0.6,
        "min_snr_weight_mean": 0.8,
        "learning_rate": 1e-4,
        "grad_norm": 1.0,
    }


def _arm(method: str, *, step: int, age: float) -> dict:
    return {
        "method": method,
        "metrics": {"last_step": step, "activity_age_seconds": age},
    }


def _process(pid: int, ppid: int, argv: list[str], *, start_ticks: int = 10) -> dict:
    return {
        "pid": pid,
        "ppid": ppid,
        "start_ticks": start_ticks,
        "argv": argv,
        "cmdline": " ".join(argv),
    }


def test_metrics_accept_100k_horizon_trajectory_at_50k_physical_stop(tmp_path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    path.write_text(
        "".join(json.dumps(_metric(step)) + "\n" for step in (1, 50, 50_000)),
        encoding="utf-8",
    )

    report = inspect_metrics(path, now=path.stat().st_mtime + 1)

    assert report["last_step"] == 50_000
    assert report["issues"] == []
    assert report["samples_seen_binding_verified"] is True


def test_metrics_fail_closed_on_exposure_nonfinite_and_boundary(tmp_path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    rows = [_metric(1), _metric(50_001)]
    rows[-1]["samples_seen"] = 1
    rows[-1]["epsilon"] = math.nan
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = inspect_metrics(path, now=path.stat().st_mtime)

    assert any("authorized 50K" in issue for issue in report["issues"])
    assert any("samples_seen" in issue for issue in report["issues"])
    assert any("epsilon" in issue and "finite" in issue for issue in report["issues"])


def test_manifest_distinguishes_scheduler_horizon_from_physical_stop(tmp_path) -> None:
    _write_manifest(tmp_path / "run_manifest.json", method="cofitok")

    report = inspect_run_manifest(
        tmp_path,
        method="cofitok",
        expected_revision=REVISION,
        expected_branch=BRANCH,
        metrics_exist=True,
    )

    assert report["status"] == "verified"
    assert report["scheduler_horizon_steps"] == 100_000
    assert report["physical_stop_step"] == 50_000


def test_process_health_accepts_one_controller_one_trainer_and_workers() -> None:
    runbook = "/checkout/artifacts/runbooks/generation_min_snr_matched_50k_pilot_v1.sh"
    trainer = [
        "python",
        "scripts/train_generation.py",
        "--config",
        "imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k_horizon_50k_pilot.json",
        "--output-dir",
        "/output/min_snr_gamma5_matched_50k_pilot_v1/cofitok_gamma5",
    ]
    processes = [
        _process(10, 1, ["bash", runbook]),
        _process(20, 10, trainer),
        _process(21, 20, trainer),
        _process(22, 20, trainer),
    ]

    health = evaluate_process_health(
        processes=processes,
        gpu_processes=[{"pid": 20}],
        controller_pid=10,
        controller_start_ticks=10,
        controller_runbook=runbook,
        controller_phase="training_cofitok",
        arms={
            "cofitok": _arm("cofitok", step=100, age=5),
            "dense_identity": _arm("dense_identity", step=0, age=0),
        },
        phase_elapsed_seconds=1_000,
        startup_grace_seconds=600,
        stall_seconds=1_800,
    )

    assert health["issues"] == []
    assert health["stall_issues"] == []
    assert [row["pid"] for row in health["direct_trainers"]] == [20]


def test_process_health_rejects_duplicate_trainer_and_external_gpu() -> None:
    runbook = "/checkout/controller.sh"
    trainer = [
        "python",
        "scripts/train_generation.py",
        "imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k_horizon_50k_pilot.json",
        "/min_snr_gamma5_matched_50k_pilot_v1/cofitok_gamma5",
    ]
    processes = [
        _process(10, 1, ["bash", runbook]),
        _process(20, 10, trainer),
        _process(30, 10, trainer),
    ]

    health = evaluate_process_health(
        processes=processes,
        gpu_processes=[{"pid": 20}, {"pid": 999}],
        controller_pid=10,
        controller_start_ticks=10,
        controller_runbook=runbook,
        controller_phase="training_cofitok",
        arms={
            "cofitok": _arm("cofitok", step=100, age=5),
            "dense_identity": _arm("dense_identity", step=0, age=0),
        },
        phase_elapsed_seconds=1_000,
        startup_grace_seconds=600,
        stall_seconds=1_800,
    )

    assert any("multiple direct" in issue for issue in health["issues"])
    assert any("outside" in issue for issue in health["issues"])


def test_process_health_marks_stale_metrics_without_signaling() -> None:
    runbook = "/checkout/controller.sh"
    trainer = [
        "python",
        "scripts/train_generation.py",
        "imagenet256_min_snr_gamma5_quality_repair_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k_horizon_50k_pilot.json",
        "/min_snr_gamma5_matched_50k_pilot_v1/cofitok_gamma5",
    ]
    health = evaluate_process_health(
        processes=[_process(10, 1, ["bash", runbook]), _process(20, 10, trainer)],
        gpu_processes=[{"pid": 20}],
        controller_pid=10,
        controller_start_ticks=10,
        controller_runbook=runbook,
        controller_phase="training_cofitok",
        arms={
            "cofitok": _arm("cofitok", step=100, age=2_000),
            "dense_identity": _arm("dense_identity", step=0, age=0),
        },
        phase_elapsed_seconds=1_000,
        startup_grace_seconds=600,
        stall_seconds=1_800,
    )

    assert health["issues"] == []
    assert len(health["stall_issues"]) == 1


def test_fdinfo_flock_binding_handles_container_pid_zero_record() -> None:
    value = (
        "pos:\t0\n"
        "flags:\t0100001\n"
        "lock:\t1: FLOCK  ADVISORY  WRITE 0 00:4b:2151402318 0 EOF\n"
    )

    assert fdinfo_declares_write_flock(value, inode=2_151_402_318) is True
    assert fdinfo_declares_write_flock(value, inode=123) is False


def test_result_can_pass_operationally_but_never_authorizes_continuation(tmp_path) -> None:
    result_path = tmp_path / "reports/min_snr_pilot_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "scientific_status": "screening_only",
                "terminal_status": "hold",
                "generation_advantage_proven": False,
                "selection_status": "shared_min_snr_candidate_for_separately_gated_100k_continuation",
                "recommended_next_stage": {
                    "execution_ready": False,
                    "gpu_execution_allowed": False,
                    "continuation_beyond_50000_allowed": False,
                },
                "authorization_boundary": dict(EXPECTED_RESULT_BOUNDARY),
            }
        ),
        encoding="utf-8",
    )

    result = inspect_result(tmp_path)
    status = classify_status(
        controller_status="completed",
        controller_phase="completed",
        issues=[],
        stall_issues=[],
        result_verified=result["verified_non_authorizing"],
    )

    assert result["verified_non_authorizing"] is True
    assert status == ("pass", "completed")
    assert all(value is False for value in NON_AUTHORIZING_BOUNDARY.values())
    assert EXPECTED_RESULT_BOUNDARY["new_gate_required_for_any_followup"] is True


def test_completed_controller_may_exit_and_release_its_execution_flock() -> None:
    health = evaluate_process_health(
        processes=[],
        gpu_processes=[],
        controller_pid=10,
        controller_start_ticks=10,
        controller_runbook="/checkout/controller.sh",
        controller_phase="completed",
        arms={
            "cofitok": _arm("cofitok", step=50_000, age=10),
            "dense_identity": _arm("dense_identity", step=50_000, age=10),
        },
        phase_elapsed_seconds=10,
        startup_grace_seconds=600,
        stall_seconds=1_800,
        controller_completed=True,
    )

    assert health["issues"] == []


def test_monitor_source_has_no_process_signal_or_torch_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/monitor_generation_min_snr_pilot.py"
    ).read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "os.kill(" not in source
    assert ".terminate(" not in source
    assert ".send_signal(" not in source
