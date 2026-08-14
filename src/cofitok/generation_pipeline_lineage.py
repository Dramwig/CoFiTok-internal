from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.inference_replay import file_identity


PLAN_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "capacity_generation_pipeline_lineage_observer"
SUCCESS_STATUSES = frozenset(
    {"pass", "passed", "complete", "completed", "ready", "success", "succeeded"}
)
ACTIVE_STATUSES = frozenset(
    {"active", "observing", "retrying", "running", "starting", "waiting"}
)
HOLD_STATUSES = frozenset({"blocked", "hold", "not_selected", "rejected"})
FAILURE_STATUSES = frozenset({"fail", "failed", "invalid", "stalled"})
AUTHORIZATION_BOUNDARY = {
    "training_launch_allowed": False,
    "sampling_or_evaluation_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signaling_allowed": False,
    "gpu_allocation_allowed": False,
    "unrelated_gpu_process_modification_allowed": False,
    "read_only_status_observation_allowed": True,
    "atomic_observer_report_write_allowed": True,
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("lineage observer timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _classification(status: Any) -> str:
    if not isinstance(status, str):
        return "invalid"
    normalized = status.strip().lower()
    if normalized in SUCCESS_STATUSES:
        return "success"
    if normalized in ACTIVE_STATUSES:
        return "active"
    if normalized in HOLD_STATUSES:
        return "hold"
    if normalized in FAILURE_STATUSES:
        return "failure"
    return "invalid"


def load_lineage_plan(
    path: str | Path,
    *,
    variables: Mapping[str, str],
) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("capacity lineage plan must be a JSON object")
    if (
        payload.get("schema_version") != PLAN_SCHEMA_VERSION
        or payload.get("role") != "capacity_generation_pipeline_lineage_plan"
    ):
        raise ValueError("capacity lineage plan identity differs")
    rows = payload.get("stages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("capacity lineage plan stages are missing")
    stages = []
    names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "blocking",
            "name",
            "role",
            "status_path",
        }:
            raise ValueError(f"capacity lineage stage {index} has another schema")
        name = row["name"]
        role = row["role"]
        template = row["status_path"]
        blocking = row["blocking"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(role, str)
            or not role
            or not isinstance(template, str)
            or not template
            or type(blocking) is not bool
        ):
            raise ValueError(f"capacity lineage stage {index} is malformed")
        try:
            expanded = template.format_map(dict(variables))
        except KeyError as error:
            raise ValueError(
                f"capacity lineage stage {name} uses an unknown path variable"
            ) from error
        status_path = Path(expanded)
        if not status_path.is_absolute() and not PurePosixPath(expanded).is_absolute():
            raise ValueError(f"capacity lineage stage {name} path is not absolute")
        names.add(name)
        stages.append(
            {
                "index": index,
                "name": name,
                "role": role,
                "blocking": blocking,
                "status_path": status_path,
            }
        )
    return {
        "identity": file_identity(plan_path),
        "role": payload["role"],
        "stages": stages,
    }


def inspect_lineage_stage(
    spec: Mapping[str, Any],
    *,
    now: datetime,
    stale_seconds: float,
    process_exists: Callable[[int], bool],
) -> dict[str, Any]:
    if stale_seconds <= 0:
        raise ValueError("capacity lineage stale threshold must be positive")
    observed_at = _utc(now)
    path = Path(spec["status_path"])
    base = {
        "index": int(spec["index"]),
        "name": str(spec["name"]),
        "expected_role": str(spec["role"]),
        "blocking": bool(spec["blocking"]),
        "status_path": path.resolve().as_posix(),
    }
    if not path.is_file():
        return {
            **base,
            "identity": None,
            "role": None,
            "status": "missing",
            "detail": "status_file_missing",
            "classification": "missing",
            "health": "missing",
            "pid": None,
            "process_alive": False,
            "updated_at": None,
            "age_seconds": None,
            "issues": ["status_file_missing"],
        }
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {
            **base,
            "identity": file_identity(path),
            "role": None,
            "status": "invalid",
            "detail": "status_json_invalid",
            "classification": "invalid",
            "health": "invalid",
            "pid": None,
            "process_alive": False,
            "updated_at": None,
            "age_seconds": None,
            "issues": [f"status_json_invalid:{error}"],
        }
    if not isinstance(payload, dict):
        return {
            **base,
            "identity": file_identity(path),
            "role": None,
            "status": "invalid",
            "detail": "status_payload_not_object",
            "classification": "invalid",
            "health": "invalid",
            "pid": None,
            "process_alive": False,
            "updated_at": None,
            "age_seconds": None,
            "issues": ["status_payload_not_object"],
        }

    role = payload.get("role")
    status = payload.get("status")
    classification = _classification(status)
    raw_pid = payload.get("pid", payload.get("supervisor_pid"))
    pid = raw_pid if type(raw_pid) is int and raw_pid > 0 else None
    alive = process_exists(pid) if pid is not None else False
    raw_updated_at = payload.get("updated_at", payload.get("updated_at_utc"))
    updated_at = _parse_timestamp(raw_updated_at)
    age = (
        max((observed_at - updated_at).total_seconds(), 0.0)
        if updated_at is not None
        else None
    )
    issues = []
    if role != spec["role"]:
        issues.append("status_role_mismatch")
    if classification == "invalid":
        issues.append("status_value_invalid")
    if classification == "active":
        if pid is None:
            issues.append("active_status_missing_pid")
        elif not alive:
            issues.append("active_status_process_dead")
        if updated_at is None:
            issues.append("active_status_timestamp_invalid")
        elif age is not None and age > stale_seconds:
            issues.append("active_status_stale")

    if "active_status_stale" in issues:
        health = "stalled"
    elif issues:
        health = "failed"
    elif classification == "success":
        health = "pass"
    elif classification == "active":
        health = "running" if str(status).lower() == "running" else "waiting"
    elif classification == "hold":
        health = "hold"
    elif classification == "failure":
        health = "failed"
    else:
        health = "invalid"
    return {
        **base,
        "identity": file_identity(path),
        "role": role,
        "status": status,
        "detail": payload.get("detail"),
        "decision": payload.get("decision"),
        "attempt": payload.get("attempt"),
        "classification": classification,
        "health": health,
        "pid": pid,
        "process_alive": alive,
        "updated_at": raw_updated_at,
        "age_seconds": age,
        "issues": issues,
    }


def build_lineage_report(
    *,
    stages: Sequence[Mapping[str, Any]],
    plan_identity: Mapping[str, Any],
    observed_at: datetime,
    stale_seconds: float,
    observer_git: Mapping[str, Any],
    formal_checkout: Mapping[str, Any],
    gpu_compute_processes: Sequence[Mapping[str, Any]],
    gpu_query_complete: bool,
    disk: Mapping[str, Any],
    hostname: str,
    external_issues: Sequence[str] = (),
) -> dict[str, Any]:
    if not stages:
        raise ValueError("capacity lineage report requires stages")
    observed = _utc(observed_at)
    blocking = [stage for stage in stages if stage.get("blocking") is True]
    if not blocking:
        raise ValueError("capacity lineage report requires a blocking stage")
    issues = [
        f"{stage['name']}:{issue}"
        for stage in stages
        for issue in stage.get("issues", [])
        if stage.get("blocking") is True
    ]
    issues.extend(str(issue) for issue in external_issues)
    unhealthy = next(
        (
            stage
            for stage in blocking
            if stage.get("health") in {"failed", "invalid", "missing", "stalled"}
        ),
        None,
    )
    current = next(
        (stage for stage in blocking if stage.get("classification") != "success"),
        None,
    )
    if unhealthy is not None or external_issues:
        selected = unhealthy or current or blocking[-1]
        status = "stalled" if selected.get("health") == "stalled" else "failed"
        detail = f"lineage_health_failure:{selected['name']}"
    elif current is None:
        selected = blocking[-1]
        status = "pass"
        detail = "capacity_generation_pipeline_completed"
    elif current.get("classification") == "active":
        selected = current
        status = "running" if current.get("status") == "running" else "waiting"
        detail = f"waiting_at:{current['name']}:{current.get('detail')}"
    elif current.get("classification") == "hold":
        selected = current
        status = "hold"
        detail = f"lineage_held_at:{current['name']}"
    else:
        selected = current
        status = "failed"
        detail = f"lineage_failed_at:{current['name']}"
    succeeded = sum(stage.get("classification") == "success" for stage in blocking)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": status,
        "detail": detail,
        "complete": status == "pass",
        "observed_at": observed.isoformat(),
        "hostname": hostname,
        "plan": dict(plan_identity),
        "observer_git": dict(observer_git),
        "formal_checkout": dict(formal_checkout),
        "thresholds": {"active_status_stale_seconds": float(stale_seconds)},
        "progress": {
            "blocking_stage_count": len(blocking),
            "successful_blocking_stage_count": succeeded,
            "fraction": succeeded / len(blocking),
            "current_stage": selected["name"],
            "current_stage_index": int(selected["index"]),
            "current_stage_status": selected.get("status"),
            "current_stage_detail": selected.get("detail"),
        },
        "stages": [dict(stage) for stage in stages],
        "gpu": {
            "query_complete": bool(gpu_query_complete),
            "compute_processes": [dict(row) for row in gpu_compute_processes],
        },
        "disk": dict(disk),
        "issues": issues,
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
    }
