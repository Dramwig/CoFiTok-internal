from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


GPU_CONTENTION_SCHEMA_VERSION = 1
GPU_CONTENTION_ROLE = "generation_gpu_contention_evidence"
MAX_UNRELATED_IDENTITIES = 128


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GPU contention timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _process_identity(process: dict[str, Any]) -> dict[str, Any]:
    pid = int(process.get("pid", -1))
    start_ticks = int(process.get("start_ticks", -1))
    argv = str(process.get("argv", ""))
    cwd = str(process.get("cwd", ""))
    process_name = str(process.get("process_name", ""))
    used_memory_mib = int(process.get("used_memory_mib", -1))
    if pid < 1 or start_ticks < 0 or not argv or not cwd or used_memory_mib < 0:
        raise ValueError("GPU compute process identity is incomplete")
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "argv": argv,
        "cwd": cwd,
        "process_name": process_name,
        "used_memory_mib": used_memory_mib,
    }


def _identity_key(process: dict[str, Any]) -> str:
    return f"{int(process['pid'])}:{int(process['start_ticks'])}"


def _pair_steps(pair_report: dict[str, Any]) -> list[int]:
    runs = pair_report.get("runs", {})
    if not isinstance(runs, dict) or not runs:
        raise ValueError("pair monitor runs are missing for GPU contention evidence")
    steps = []
    for run in runs.values():
        if not isinstance(run, dict):
            raise ValueError("pair monitor run entry is malformed")
        steps.append(int(run.get("last_step", 0)))
    return steps


def _binding(pair_report: dict[str, Any]) -> dict[str, Any]:
    git = pair_report.get("git", {})
    monitor_name = str(pair_report.get("monitor", ""))
    revision = str(git.get("revision", "")) if isinstance(git, dict) else ""
    branch = str(git.get("branch", "")) if isinstance(git, dict) else ""
    if not monitor_name or not revision or not branch:
        raise ValueError("pair monitor identity is incomplete for GPU contention evidence")
    return {
        "monitor_name": monitor_name,
        "training_revision": revision,
        "training_branch": branch,
    }


def _prior_state(
    previous: dict[str, Any] | None,
    *,
    binding: dict[str, Any],
    poll_seconds: float,
) -> dict[str, Any]:
    if previous is None:
        return {
            "observation_count": 0,
            "first_observed_at": None,
            "last_observed_at": None,
            "maximum_gap_seconds": 0.0,
            "started_before_training": False,
            "saw_training_active": False,
            "completed_after_training": False,
            "all_gpu_queries_complete": True,
            "unrelated_observation_count": 0,
            "unrelated_identities": [],
            "identity_overflow": False,
        }
    if (
        previous.get("schema_version") != GPU_CONTENTION_SCHEMA_VERSION
        or previous.get("role") != GPU_CONTENTION_ROLE
        or previous.get("binding") != binding
        or float(previous.get("poll_seconds", -1.0)) != poll_seconds
    ):
        raise ValueError("prior GPU contention evidence binding differs")
    coverage = previous.get("coverage", {})
    unrelated = previous.get("unrelated_gpu_compute", {})
    identities = unrelated.get("identities", [])
    if not isinstance(coverage, dict) or not isinstance(unrelated, dict):
        raise ValueError("prior GPU contention evidence is malformed")
    if not isinstance(identities, list):
        raise ValueError("prior unrelated GPU identity list is malformed")
    return {
        "observation_count": int(previous.get("observation_count", 0)),
        "first_observed_at": previous.get("first_observed_at"),
        "last_observed_at": previous.get("last_observed_at"),
        "maximum_gap_seconds": float(coverage.get("maximum_gap_seconds", 0.0)),
        "started_before_training": coverage.get("started_before_training") is True,
        "saw_training_active": coverage.get("saw_training_active") is True,
        "completed_after_training": coverage.get("completed_after_training") is True,
        "all_gpu_queries_complete": coverage.get("all_gpu_queries_complete") is True,
        "unrelated_observation_count": int(
            unrelated.get("observation_count", 0)
        ),
        "unrelated_identities": [dict(identity) for identity in identities],
        "identity_overflow": unrelated.get("identity_overflow") is True,
    }


def update_gpu_contention_evidence(
    previous: dict[str, Any] | None,
    *,
    pair_report: dict[str, Any],
    training_processes: list[dict[str, Any]],
    gpu_compute_processes: list[dict[str, Any]],
    observed_at: str,
    poll_seconds: float,
    gpu_query_complete: bool = True,
) -> dict[str, Any]:
    if poll_seconds <= 0.0:
        raise ValueError("GPU contention polling interval must be positive")
    current_time = _timestamp(observed_at)
    binding = _binding(pair_report)
    prior = _prior_state(
        previous,
        binding=binding,
        poll_seconds=float(poll_seconds),
    )
    if prior["last_observed_at"] is not None:
        last_time = _timestamp(str(prior["last_observed_at"]))
        gap_seconds = (current_time - last_time).total_seconds()
        if gap_seconds <= 0.0:
            raise ValueError("GPU contention observations are not strictly increasing")
    else:
        gap_seconds = 0.0

    training_pids = {
        int(process.get("pid", -1))
        for process in training_processes
        if int(process.get("pid", -1)) > 0
    }
    compute = [_process_identity(process) for process in gpu_compute_processes]
    expected = [process for process in compute if process["pid"] in training_pids]
    unrelated_current = [
        process for process in compute if process["pid"] not in training_pids
    ]
    training_active = bool(training_processes) or bool(expected)
    steps = _pair_steps(pair_report)
    first_observation = prior["observation_count"] == 0
    started_before_training = prior["started_before_training"] or (
        first_observation and not training_active and max(steps) == 0
    )
    pair_status = str(pair_report.get("status", ""))
    pair_stage = str(pair_report.get("stage", ""))
    completed_after_training = prior["completed_after_training"] or (
        pair_status == "pass" and pair_stage == "complete"
    )

    identity_index = {
        str(identity.get("identity_key")): identity
        for identity in prior["unrelated_identities"]
    }
    identity_overflow = prior["identity_overflow"]
    for process in unrelated_current:
        key = _identity_key(process)
        existing = identity_index.get(key)
        if existing is None:
            if len(identity_index) >= MAX_UNRELATED_IDENTITIES:
                identity_overflow = True
                continue
            existing = {
                "identity_key": key,
                "pid": process["pid"],
                "start_ticks": process["start_ticks"],
                "argv": process["argv"],
                "cwd": process["cwd"],
                "process_name": process["process_name"],
                "first_observed_at": observed_at,
                "last_observed_at": observed_at,
                "observation_count": 0,
                "maximum_used_memory_mib": 0,
            }
            identity_index[key] = existing
        elif (
            existing.get("argv") != process["argv"]
            or existing.get("cwd") != process["cwd"]
            or existing.get("process_name") != process["process_name"]
        ):
            raise ValueError("unrelated GPU process identity changed for one PID start")
        existing["last_observed_at"] = observed_at
        existing["observation_count"] = int(existing["observation_count"]) + 1
        existing["maximum_used_memory_mib"] = max(
            int(existing["maximum_used_memory_mib"]),
            process["used_memory_mib"],
        )

    observation_count = prior["observation_count"] + 1
    maximum_gap = max(prior["maximum_gap_seconds"], gap_seconds)
    continuous = observation_count >= 2 and maximum_gap <= poll_seconds * 2.5
    coverage_complete = (
        started_before_training
        and (prior["saw_training_active"] or training_active)
        and completed_after_training
        and continuous
        and prior["all_gpu_queries_complete"]
        and gpu_query_complete
        and not identity_overflow
    )
    unrelated_observed = bool(identity_index) or identity_overflow
    if not coverage_complete:
        comparison_reason = "incomplete_gpu_observation_coverage"
    elif unrelated_observed:
        comparison_reason = "external_gpu_contention_observed"
    else:
        comparison_reason = "exclusive_gpu_observation_coverage"
    direct_comparison_allowed = coverage_complete and not unrelated_observed

    if pair_status in {"failed", "stalled"}:
        status = "failed_training"
    elif completed_after_training:
        status = (
            "pass_exclusive"
            if direct_comparison_allowed
            else "pass_observational_only"
        )
    else:
        status = "running"

    return {
        "schema_version": GPU_CONTENTION_SCHEMA_VERSION,
        "role": GPU_CONTENTION_ROLE,
        "status": status,
        "binding": binding,
        "poll_seconds": float(poll_seconds),
        "observation_count": observation_count,
        "first_observed_at": prior["first_observed_at"] or observed_at,
        "last_observed_at": observed_at,
        "coverage": {
            "started_before_training": started_before_training,
            "saw_training_active": prior["saw_training_active"] or training_active,
            "completed_after_training": completed_after_training,
            "all_gpu_queries_complete": (
                prior["all_gpu_queries_complete"] and gpu_query_complete
            ),
            "maximum_gap_seconds": maximum_gap,
            "continuous": continuous,
            "complete": coverage_complete,
        },
        "current": {
            "pair_status": pair_status,
            "pair_stage": pair_stage,
            "training_process_pids": sorted(training_pids),
            "expected_training_gpu_processes": expected,
            "unrelated_gpu_processes": unrelated_current,
        },
        "unrelated_gpu_compute": {
            "observed": unrelated_observed,
            "observation_count": prior["unrelated_observation_count"]
            + (1 if unrelated_current else 0),
            "identity_overflow": identity_overflow,
            "identities": sorted(
                identity_index.values(), key=lambda item: item["identity_key"]
            ),
        },
        "training_wall_clock": {
            "measurement": "raw_process_wall_clock",
            "direct_comparison_allowed": direct_comparison_allowed,
            "reason": comparison_reason,
        },
    }


def validate_gpu_contention_evidence(
    pair_report: dict[str, Any],
) -> dict[str, Any]:
    evidence = pair_report.get("gpu_contention")
    if not isinstance(evidence, dict):
        raise ValueError("pair monitor is missing GPU contention evidence")
    if (
        evidence.get("schema_version") != GPU_CONTENTION_SCHEMA_VERSION
        or evidence.get("role") != GPU_CONTENTION_ROLE
        or evidence.get("binding") != _binding(pair_report)
    ):
        raise ValueError("pair monitor GPU contention evidence identity differs")
    wall_clock = evidence.get("training_wall_clock")
    coverage = evidence.get("coverage")
    unrelated = evidence.get("unrelated_gpu_compute")
    if (
        not isinstance(wall_clock, dict)
        or not isinstance(coverage, dict)
        or not isinstance(unrelated, dict)
        or wall_clock.get("measurement") != "raw_process_wall_clock"
    ):
        raise ValueError("pair monitor GPU contention evidence is malformed")
    if pair_report.get("status") != "pass" or pair_report.get("stage") != "complete":
        raise ValueError("GPU contention evidence is not bound to a completed pair")
    observation_count = int(evidence.get("observation_count", 0))
    poll_seconds = float(evidence.get("poll_seconds", 0.0))
    maximum_gap = float(coverage.get("maximum_gap_seconds", math.inf))
    expected_coverage = (
        observation_count >= 2
        and poll_seconds > 0.0
        and math.isfinite(maximum_gap)
        and maximum_gap <= poll_seconds * 2.5
        and coverage.get("started_before_training") is True
        and coverage.get("saw_training_active") is True
        and coverage.get("completed_after_training") is True
        and coverage.get("continuous") is True
        and coverage.get("all_gpu_queries_complete") is True
        and unrelated.get("identity_overflow") is False
    )
    if (coverage.get("complete") is True) != expected_coverage:
        raise ValueError("GPU contention coverage decision is inconsistent")
    identities = unrelated.get("identities")
    if (
        not isinstance(coverage.get("complete"), bool)
        or not isinstance(unrelated.get("observed"), bool)
        or not isinstance(unrelated.get("identity_overflow"), bool)
        or not isinstance(wall_clock.get("direct_comparison_allowed"), bool)
        or not isinstance(identities, list)
        or (unrelated["observed"] is False and identities)
        or (
            unrelated["observed"] is True
            and not identities
            and unrelated["identity_overflow"] is False
        )
    ):
        raise ValueError("GPU contention unrelated-process evidence is inconsistent")
    direct = wall_clock.get("direct_comparison_allowed") is True
    expected_direct = (
        coverage.get("complete") is True
        and unrelated.get("observed") is False
        and unrelated.get("identity_overflow") is False
    )
    if direct != expected_direct:
        raise ValueError("GPU contention wall-clock comparison decision is inconsistent")
    expected_reason = (
        "exclusive_gpu_observation_coverage"
        if expected_direct
        else (
            "external_gpu_contention_observed"
            if coverage.get("complete") is True
            and unrelated.get("observed") is True
            else "incomplete_gpu_observation_coverage"
        )
    )
    if wall_clock.get("reason") != expected_reason:
        raise ValueError("GPU contention wall-clock comparison reason is inconsistent")
    expected_status = (
        "pass_exclusive" if expected_direct else "pass_observational_only"
    )
    if evidence.get("status") != expected_status:
        raise ValueError("GPU contention terminal status is inconsistent")
    return {
        "status": "verified",
        "direct_comparison_allowed": direct,
        "measurement": wall_clock["measurement"],
        "reason": wall_clock["reason"],
        "coverage_complete": coverage.get("complete") is True,
        "unrelated_gpu_compute_observed": unrelated.get("observed") is True,
        "observation_count": observation_count,
    }
