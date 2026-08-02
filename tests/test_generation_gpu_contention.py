from __future__ import annotations

import copy

import pytest

from cofitok.gpu_contention import (
    update_gpu_contention_evidence,
    validate_gpu_contention_evidence,
)


def _pair(*, status: str, stage: str, step: int) -> dict:
    return {
        "monitor": "generation_stability_ema_teacher_full_matched_300k",
        "status": status,
        "stage": stage,
        "issues": [],
        "git": {
            "revision": "a" * 40,
            "branch": "scale/generation-large-capacity",
            "tracked_dirty": False,
        },
        "runs": {
            "cofitok": {"last_step": step},
            "dense_identity": {"last_step": step},
        },
    }


def _training_process(pid: int = 10) -> dict:
    return {
        "pid": pid,
        "start_ticks": 1_000,
        "argv": "python scripts/train_generation.py --output-dir /runs/cofitok",
        "cwd": "/checkout",
    }


def _gpu_process(
    pid: int = 10,
    *,
    argv: str = "python scripts/train_generation.py --output-dir /runs/cofitok",
    cwd: str = "/checkout",
    memory: int = 70_000,
) -> dict:
    return {
        "pid": pid,
        "start_ticks": pid * 100,
        "argv": argv,
        "cwd": cwd,
        "process_name": "python",
        "used_memory_mib": memory,
    }


def _complete_sequence(*, unrelated: bool = False, final_seconds: int = 600) -> dict:
    evidence = update_gpu_contention_evidence(
        None,
        pair_report=_pair(status="running", stage="cofitok_training", step=0),
        training_processes=[],
        gpu_compute_processes=[],
        observed_at="2026-08-02T00:00:00+00:00",
        poll_seconds=300.0,
    )
    compute = [_gpu_process()]
    if unrelated:
        compute.append(
            _gpu_process(
                99,
                argv="python -m fieldscope.cli extract-dataset",
                cwd="/root/autodl-tmp/FieldScope",
                memory=15_412,
            )
        )
    evidence = update_gpu_contention_evidence(
        evidence,
        pair_report=_pair(
            status="running", stage="cofitok_training", step=50
        ),
        training_processes=[_training_process()],
        gpu_compute_processes=compute,
        observed_at="2026-08-02T00:05:00+00:00",
        poll_seconds=300.0,
    )
    final_pair = _pair(status="pass", stage="complete", step=300_000)
    evidence = update_gpu_contention_evidence(
        evidence,
        pair_report=final_pair,
        training_processes=[],
        gpu_compute_processes=[],
        observed_at=f"2026-08-02T00:{final_seconds // 60:02d}:00+00:00",
        poll_seconds=300.0,
    )
    final_pair["gpu_contention"] = evidence
    return final_pair


def test_gpu_contention_allows_wall_clock_only_with_complete_exclusive_coverage() -> None:
    report = _complete_sequence()

    verified = validate_gpu_contention_evidence(report)

    assert report["gpu_contention"]["status"] == "pass_exclusive"
    assert report["gpu_contention"]["coverage"]["complete"] is True
    assert verified == {
        "status": "verified",
        "direct_comparison_allowed": True,
        "measurement": "raw_process_wall_clock",
        "reason": "exclusive_gpu_observation_coverage",
        "coverage_complete": True,
        "unrelated_gpu_compute_observed": False,
        "observation_count": 3,
    }


def test_gpu_contention_preserves_external_identity_and_marks_time_observational() -> None:
    report = _complete_sequence(unrelated=True)

    verified = validate_gpu_contention_evidence(report)
    identities = report["gpu_contention"]["unrelated_gpu_compute"]["identities"]

    assert report["gpu_contention"]["status"] == "pass_observational_only"
    assert verified["direct_comparison_allowed"] is False
    assert verified["reason"] == "external_gpu_contention_observed"
    assert identities == [
        {
            "identity_key": "99:9900",
            "pid": 99,
            "start_ticks": 9_900,
            "argv": "python -m fieldscope.cli extract-dataset",
            "cwd": "/root/autodl-tmp/FieldScope",
            "process_name": "python",
            "first_observed_at": "2026-08-02T00:05:00+00:00",
            "last_observed_at": "2026-08-02T00:05:00+00:00",
            "observation_count": 1,
            "maximum_used_memory_mib": 15_412,
        }
    ]


def test_gpu_contention_rejects_wall_clock_comparison_when_coverage_has_gap() -> None:
    report = _complete_sequence(final_seconds=20 * 60)

    verified = validate_gpu_contention_evidence(report)

    assert report["gpu_contention"]["coverage"]["continuous"] is False
    assert verified["direct_comparison_allowed"] is False
    assert verified["reason"] == "incomplete_gpu_observation_coverage"


def test_gpu_contention_rejects_prior_binding_drift() -> None:
    report = _complete_sequence()
    prior = report["gpu_contention"]
    changed = _pair(status="running", stage="cofitok_training", step=50)
    changed["git"]["revision"] = "b" * 40

    with pytest.raises(ValueError, match="binding differs"):
        update_gpu_contention_evidence(
            prior,
            pair_report=changed,
            training_processes=[_training_process()],
            gpu_compute_processes=[_gpu_process()],
            observed_at="2026-08-02T00:15:00+00:00",
            poll_seconds=300.0,
        )


def test_gpu_contention_rejects_poll_interval_drift() -> None:
    report = _complete_sequence()

    with pytest.raises(ValueError, match="binding differs"):
        update_gpu_contention_evidence(
            report["gpu_contention"],
            pair_report=_pair(
                status="running", stage="cofitok_training", step=50
            ),
            training_processes=[_training_process()],
            gpu_compute_processes=[_gpu_process()],
            observed_at="2026-08-02T00:15:00+00:00",
            poll_seconds=60.0,
        )


def test_gpu_contention_marks_incomplete_nvidia_query_as_observational_only() -> None:
    evidence = update_gpu_contention_evidence(
        None,
        pair_report=_pair(status="running", stage="cofitok_training", step=0),
        training_processes=[],
        gpu_compute_processes=[],
        observed_at="2026-08-02T00:00:00+00:00",
        poll_seconds=300.0,
        gpu_query_complete=False,
    )
    evidence = update_gpu_contention_evidence(
        evidence,
        pair_report=_pair(status="pass", stage="complete", step=300_000),
        training_processes=[_training_process()],
        gpu_compute_processes=[_gpu_process()],
        observed_at="2026-08-02T00:05:00+00:00",
        poll_seconds=300.0,
    )
    report = _pair(status="pass", stage="complete", step=300_000)
    report["gpu_contention"] = evidence

    verified = validate_gpu_contention_evidence(report)

    assert evidence["coverage"]["all_gpu_queries_complete"] is False
    assert verified["direct_comparison_allowed"] is False
    assert verified["reason"] == "incomplete_gpu_observation_coverage"


def test_gpu_contention_validator_rejects_promoted_direct_comparison() -> None:
    report = _complete_sequence(unrelated=True)
    tampered = copy.deepcopy(report)
    tampered["gpu_contention"]["training_wall_clock"][
        "direct_comparison_allowed"
    ] = True

    with pytest.raises(ValueError, match="decision is inconsistent"):
        validate_gpu_contention_evidence(tampered)
