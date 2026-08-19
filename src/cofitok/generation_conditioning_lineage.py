from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.inference_replay import file_identity

PLAN_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PLAN_ROLE = "generation_conditioning_pipeline_lineage_plan"
REPORT_ROLE = "generation_conditioning_pipeline_lineage_observer"
ACTIVE_STATUSES = frozenset({"waiting", "running", "starting"})
SUCCESS_STATUSES = frozenset({"complete", "completed", "pass", "passed"})
FAILURE_STATUSES = frozenset({"fail", "failed", "invalid", "stalled"})
AUTHORIZATION_BOUNDARY = {
    "read_only_source_observation_allowed": True,
    "atomic_observer_report_write_allowed": True,
    "training_launch_allowed": False,
    "sampling_or_evaluation_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signaling_allowed": False,
    "gpu_allocation_allowed": False,
    "standing_authorization_modified": False,
    "supervisor_or_child_process_modified": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and (
        Path(value).is_absolute() or PurePosixPath(value).is_absolute()
    )


def _identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "path", "sha256"}:
        raise ValueError(f"{label} identity schema differs")
    path = value["path"]
    size = value["bytes"]
    sha256 = value["sha256"]
    if (
        not _is_absolute_path(path)
        or type(size) is not int
        or size < 0
        or not _is_sha256(sha256)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": sha256}


def _project(value: Any, *, label: str) -> dict[str, Any]:
    required = {"branch", "path", "revision", "tracked_dirty", "tree"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} project schema differs")
    if (
        not _is_absolute_path(value["path"])
        or not isinstance(value["branch"], str)
        or not value["branch"]
        or not isinstance(value["revision"], str)
        or len(value["revision"]) != 40
        or not isinstance(value["tree"], str)
        or len(value["tree"]) != 40
        or value["tracked_dirty"] is not False
    ):
        raise ValueError(f"{label} project is malformed")
    return dict(value)


def _process(value: Any, *, label: str) -> dict[str, Any]:
    required = {"cmdline_sha256", "cwd", "pid", "start_ticks"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{label} process schema differs")
    if (
        type(value["pid"]) is not int
        or value["pid"] <= 0
        or type(value["start_ticks"]) is not int
        or value["start_ticks"] <= 0
        or not _is_absolute_path(value["cwd"])
        or not _is_sha256(value["cmdline_sha256"])
    ):
        raise ValueError(f"{label} process is malformed")
    return dict(value)


def load_conditioning_lineage_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != {
        "role",
        "schema_version",
        "stages",
        "standing_authorization",
    }:
        raise ValueError("conditioning lineage plan schema differs")
    if payload["schema_version"] != PLAN_SCHEMA_VERSION or payload["role"] != PLAN_ROLE:
        raise ValueError("conditioning lineage plan identity differs")
    standing = _identity(
        payload["standing_authorization"],
        label="standing authorization",
    )
    rows = payload["stages"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("conditioning lineage stages are missing")
    expected_keys = {
        "expected_status_output_root",
        "idle_gpu_evidence_path",
        "launch_receipt_path",
        "max_launches",
        "name",
        "process",
        "project",
        "required_idle_polls",
        "required_status_sources",
        "role",
        "source_files",
        "stage_output_path",
        "status_path",
        "status_pid_required",
    }
    stages = []
    names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise ValueError(f"conditioning lineage stage {index} schema differs")
        name = row["name"]
        role = row["role"]
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(role, str)
            or not role
        ):
            raise ValueError(f"conditioning lineage stage {index} identity differs")
        absolute_fields = (
            "expected_status_output_root",
            "stage_output_path",
            "status_path",
        )
        if any(not _is_absolute_path(row[field]) for field in absolute_fields):
            raise ValueError(f"conditioning lineage stage {name} path differs")
        optional_paths = ("idle_gpu_evidence_path", "launch_receipt_path")
        if any(
            value is not None and not _is_absolute_path(value)
            for value in (row[field] for field in optional_paths)
        ):
            raise ValueError(f"conditioning lineage stage {name} optional path differs")
        if (
            type(row["required_idle_polls"]) is not int
            or row["required_idle_polls"] not in {0, 5}
            or row["max_launches"] != 1
            or type(row["status_pid_required"]) is not bool
        ):
            raise ValueError(f"conditioning lineage stage {name} policy differs")
        required_sources = row["required_status_sources"]
        if (
            not isinstance(required_sources, list)
            or any(
                not isinstance(value, str) or not value for value in required_sources
            )
            or len(set(required_sources)) != len(required_sources)
        ):
            raise ValueError(
                f"conditioning lineage stage {name} source requirements differ"
            )
        raw_sources = row["source_files"]
        if not isinstance(raw_sources, list) or len(raw_sources) < 2:
            raise ValueError(f"conditioning lineage stage {name} sources differ")
        source_files = [
            _identity(value, label=f"conditioning lineage stage {name} source")
            for value in raw_sources
        ]
        if len({value["path"] for value in source_files}) != len(source_files):
            raise ValueError(f"conditioning lineage stage {name} repeats a source")
        names.add(name)
        stages.append(
            {
                **dict(row),
                "index": index,
                "project": _project(row["project"], label=f"stage {name}"),
                "process": _process(row["process"], label=f"stage {name}"),
                "required_status_sources": list(required_sources),
                "source_files": source_files,
            }
        )
    return {
        "role": PLAN_ROLE,
        "identity": file_identity(plan_path),
        "standing_authorization": standing,
        "stages": stages,
    }


def _status_classification(value: Any) -> str:
    if not isinstance(value, str):
        return "invalid"
    normalized = value.strip().lower()
    if normalized in ACTIVE_STATUSES:
        return "active"
    if normalized in SUCCESS_STATUSES:
        return "success"
    if normalized in FAILURE_STATUSES:
        return "failure"
    return "invalid"


def _timestamp(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("updated_at_unix")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = payload.get("updated_at", payload.get("updated_at_utc"))
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).timestamp()


def _reported_identity_issues(value: Any, *, label: str) -> list[str]:
    issues: list[str] = []
    if isinstance(value, Mapping):
        if {"bytes", "path", "sha256"}.issubset(value):
            try:
                expected = _identity(
                    {key: value[key] for key in ("bytes", "path", "sha256")},
                    label=label,
                )
                path = Path(expected["path"])
                if not path.is_file():
                    issues.append(f"{label}:reported_source_missing")
                elif file_identity(path) != expected:
                    issues.append(f"{label}:reported_source_identity_mismatch")
            except ValueError:
                issues.append(f"{label}:reported_source_identity_invalid")
            return issues
        for key, child in value.items():
            issues.extend(_reported_identity_issues(child, label=f"{label}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_reported_identity_issues(child, label=f"{label}[{index}]"))
    return issues


def _idle_evidence_issues(path: Path, *, required_polls: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ["idle_gpu_evidence_invalid"]
    if not isinstance(payload, Mapping):
        return ["idle_gpu_evidence_invalid"]
    observations = payload.get("observations")
    if (
        payload.get("status") not in SUCCESS_STATUSES
        or payload.get("required_consecutive_idle_polls") != required_polls
        or not isinstance(observations, list)
        or len(observations) != required_polls
    ):
        return ["idle_gpu_evidence_policy_mismatch"]
    timestamps = []
    for index, observation in enumerate(observations, start=1):
        if (
            not isinstance(observation, Mapping)
            or observation.get("poll_index") != index
            or observation.get("gpu_compute_pids") != []
            or not isinstance(observation.get("observed_at_unix"), (int, float))
            or isinstance(observation.get("observed_at_unix"), bool)
        ):
            return ["idle_gpu_evidence_observation_invalid"]
        timestamps.append(float(observation["observed_at_unix"]))
    if any(right <= left for left, right in pairwise(timestamps)):
        return ["idle_gpu_evidence_timestamps_not_increasing"]
    return []


def _initial_history() -> dict[str, Any]:
    return {
        "child_pids": [],
        "first_observed_at_unix": None,
        "idle_gate_observed": False,
        "launch_receipt_sha256s": [],
        "max_idle_gpu_polls": 0,
        "observation_count": 0,
        "unbound_output_observations": 0,
    }


def inspect_conditioning_stage(
    spec: Mapping[str, Any],
    *,
    now_unix: float,
    stale_seconds: float,
    standing_authorization: Mapping[str, Any],
    previous_history: Mapping[str, Any] | None,
    git_inspector: Callable[[Path], Mapping[str, Any]],
    process_inspector: Callable[[int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stale_seconds <= 0:
        raise ValueError("conditioning lineage stale threshold must be positive")
    history = _initial_history()
    if previous_history is not None:
        for key in history:
            if key in previous_history:
                history[key] = copy.deepcopy(previous_history[key])
    history["observation_count"] = int(history["observation_count"]) + 1
    if history["first_observed_at_unix"] is None:
        history["first_observed_at_unix"] = float(now_unix)

    issues: list[str] = []
    status_path = Path(str(spec["status_path"]))
    payload: dict[str, Any] = {}
    status_identity = None
    if not status_path.is_file():
        issues.append("status_file_missing")
    else:
        status_identity = file_identity(status_path)
        try:
            with status_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise TypeError("status is not an object")
            payload = loaded
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            issues.append("status_json_invalid")

    status = payload.get("status")
    classification = _status_classification(status)
    if payload:
        if payload.get("role") != spec["role"]:
            issues.append("status_role_mismatch")
        if classification == "invalid":
            issues.append("status_value_invalid")
        if (
            payload.get("project", payload.get("evaluator_project"))
            != spec["project"]["path"]
        ):
            issues.append("status_project_mismatch")
        if payload.get("output_root") != spec["expected_status_output_root"]:
            issues.append("status_output_root_mismatch")

    expected_process = dict(spec["process"])
    actual_process = dict(process_inspector(int(expected_process["pid"])))
    process_alive = bool(actual_process.get("alive"))
    if classification == "active" and actual_process != {
        **expected_process,
        "alive": True,
    }:
        issues.append("active_supervisor_process_identity_mismatch")
    raw_status_pid = payload.get("pid")
    if spec["status_pid_required"]:
        if raw_status_pid != expected_process["pid"]:
            issues.append("status_pid_mismatch")
    elif raw_status_pid is not None and raw_status_pid != expected_process["pid"]:
        issues.append("optional_status_pid_mismatch")

    updated_at = _timestamp(payload)
    age_seconds = (
        max(float(now_unix) - updated_at, 0.0) if updated_at is not None else None
    )
    if classification == "active":
        if updated_at is None:
            issues.append("active_status_timestamp_invalid")
        elif age_seconds is not None and age_seconds > stale_seconds:
            issues.append("active_status_stale")
        if not process_alive:
            issues.append("active_supervisor_process_dead")

    actual_git = dict(git_inspector(Path(spec["project"]["path"])))
    if actual_git != dict(spec["project"]):
        issues.append("project_git_identity_mismatch")
    source_files = []
    for expected in spec["source_files"]:
        path = Path(expected["path"])
        actual = file_identity(path) if path.is_file() else None
        if actual != expected:
            issues.append(f"source_file_identity_mismatch:{path.name}")
        source_files.append({"expected": dict(expected), "actual": actual})

    sources = payload.get("sources")
    if not isinstance(sources, Mapping):
        if spec["required_status_sources"]:
            issues.append("status_sources_missing")
        sources = {}
    for key in spec["required_status_sources"]:
        if key not in sources:
            issues.append(f"required_status_source_missing:{key}")
    issues.extend(_reported_identity_issues(sources, label="status.sources"))
    standing_source = sources.get("standing_authorization")
    if (
        "standing_authorization" in spec["required_status_sources"]
        and standing_source != standing_authorization
    ):
        issues.append("standing_authorization_status_binding_mismatch")

    raw_idle_polls = payload.get("idle_gpu_polls", 0)
    idle_polls = raw_idle_polls if type(raw_idle_polls) is int else -1
    if idle_polls < 0 or idle_polls > spec["required_idle_polls"]:
        issues.append("idle_gpu_poll_count_invalid")
    history["max_idle_gpu_polls"] = max(
        int(history["max_idle_gpu_polls"]), max(idle_polls, 0)
    )
    if history["max_idle_gpu_polls"] >= spec["required_idle_polls"]:
        history["idle_gate_observed"] = True

    child_pid = payload.get("child_pid")
    if child_pid is not None and (type(child_pid) is not int or child_pid <= 0):
        issues.append("child_pid_invalid")
        child_pid = None
    child_pids = {int(value) for value in history["child_pids"]}
    if child_pid is not None:
        child_pids.add(child_pid)
    history["child_pids"] = sorted(child_pids)

    launch_path_value = spec.get("launch_receipt_path")
    launch_path = Path(launch_path_value) if launch_path_value is not None else None
    launch_identity = None
    launch_receipts = set(history["launch_receipt_sha256s"])
    if launch_path is not None and launch_path.is_file():
        launch_identity = file_identity(launch_path)
        launch_receipts.add(str(launch_identity["sha256"]))
        issues.extend(
            _reported_identity_issues(
                json.loads(launch_path.read_text(encoding="utf-8")),
                label="launch_receipt",
            )
        )
    history["launch_receipt_sha256s"] = sorted(launch_receipts)
    output_exists = Path(spec["stage_output_path"]).exists()
    launch_count_lower_bound = max(
        len(history["child_pids"]),
        len(history["launch_receipt_sha256s"]),
        int(output_exists),
    )
    if launch_count_lower_bound > spec["max_launches"]:
        issues.append("runbook_launch_count_exceeded")
    launch_bound = bool(history["child_pids"] or history["launch_receipt_sha256s"])
    history["unbound_output_observations"] = (
        int(history["unbound_output_observations"]) + 1
        if output_exists and not launch_bound
        else 0
    )
    if history["unbound_output_observations"] >= 3:
        issues.append("stage_output_exists_without_observed_launch_binding")

    launch_observed = bool(child_pid is not None or launch_identity is not None)
    idle_path_value = spec.get("idle_gpu_evidence_path")
    idle_path = Path(idle_path_value) if idle_path_value is not None else None
    idle_identity = None
    if idle_path is not None and idle_path.is_file():
        idle_identity = file_identity(idle_path)
    if launch_observed and spec["required_idle_polls"]:
        if idle_path is not None:
            if idle_identity is None:
                issues.append("launch_missing_idle_gpu_evidence")
            else:
                issues.extend(
                    _idle_evidence_issues(
                        idle_path,
                        required_polls=int(spec["required_idle_polls"]),
                    )
                )
                if not any(issue.startswith("idle_gpu_evidence") for issue in issues):
                    history["idle_gate_observed"] = True
        elif not history["idle_gate_observed"]:
            issues.append("launch_before_observed_idle_gpu_gate")
    if launch_observed and launch_path is not None and launch_identity is None:
        issues.append("launch_missing_immutable_receipt")

    health = "running"
    if issues:
        health = "stalled" if "active_status_stale" in issues else "failed"
    elif classification == "success":
        health = "pass"
    elif classification == "failure":
        health = "failed"
    elif classification == "active":
        health = "running" if status == "running" else "waiting"
    else:
        health = "invalid"
    return (
        {
            "index": spec["index"],
            "name": spec["name"],
            "role": payload.get("role"),
            "expected_role": spec["role"],
            "status": status,
            "detail": payload.get("detail"),
            "classification": classification,
            "health": health,
            "status_identity": status_identity,
            "status_updated_at_unix": updated_at,
            "status_age_seconds": age_seconds,
            "project": {"expected": dict(spec["project"]), "actual": actual_git},
            "supervisor_process": {
                "expected": expected_process,
                "actual": actual_process,
            },
            "source_files": source_files,
            "standing_authorization_bound": (
                standing_source == standing_authorization
                if "standing_authorization" in spec["required_status_sources"]
                else None
            ),
            "idle_gpu_polls": max(idle_polls, 0),
            "required_idle_gpu_polls": spec["required_idle_polls"],
            "child_pid": child_pid,
            "launch_receipt": launch_identity,
            "idle_gpu_evidence": idle_identity,
            "stage_output_exists": output_exists,
            "launch_count_lower_bound": launch_count_lower_bound,
            "max_launches": spec["max_launches"],
            "issues": issues,
        },
        history,
    )


def build_conditioning_lineage_report(
    *,
    stages: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    observed_at_unix: float,
    stale_seconds: float,
    plan_identity: Mapping[str, Any],
    observer_git: Mapping[str, Any],
    standing_authorization: Mapping[str, Any],
    gpu_compute_processes: Sequence[Mapping[str, Any]],
    gpu_query_complete: bool,
    disk: Mapping[str, Any],
    hostname: str,
    external_issues: Sequence[str] = (),
) -> dict[str, Any]:
    if not stages:
        raise ValueError("conditioning lineage report requires stages")
    issues = [
        f"{stage['name']}:{issue}"
        for stage in stages
        for issue in stage.get("issues", [])
    ]
    issues.extend(str(value) for value in external_issues)
    current = next(
        (stage for stage in stages if stage.get("classification") != "success"),
        stages[-1],
    )
    unhealthy = next((stage for stage in stages if stage.get("issues")), None)
    all_success = all(stage.get("classification") == "success" for stage in stages)
    if issues:
        status = (
            "stalled"
            if any(stage.get("health") == "stalled" for stage in stages)
            else "failed"
        )
        detail = f"conditioning_lineage_health_failure:{(unhealthy or current)['name']}"
    elif all_success:
        status = "pass"
        detail = "conditioning_pipeline_completed"
    elif current.get("classification") == "failure":
        status = "failed"
        detail = f"conditioning_pipeline_failed_at:{current['name']}"
    else:
        status = "running" if current.get("status") == "running" else "waiting"
        detail = f"waiting_at:{current['name']}:{current.get('detail')}"
    succeeded = sum(stage.get("classification") == "success" for stage in stages)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": status,
        "detail": detail,
        "complete": status == "pass",
        "observed_at": datetime.fromtimestamp(
            observed_at_unix, tz=timezone.utc
        ).isoformat(),
        "observed_at_unix": float(observed_at_unix),
        "hostname": hostname,
        "plan": dict(plan_identity),
        "observer_git": dict(observer_git),
        "standing_authorization": dict(standing_authorization),
        "thresholds": {"active_status_stale_seconds": float(stale_seconds)},
        "progress": {
            "stage_count": len(stages),
            "successful_stage_count": succeeded,
            "fraction": succeeded / len(stages),
            "current_stage": current["name"],
            "current_stage_status": current.get("status"),
            "current_stage_detail": current.get("detail"),
        },
        "stages": [copy.deepcopy(dict(stage)) for stage in stages],
        "history": copy.deepcopy(dict(histories)),
        "gpu": {
            "query_complete": bool(gpu_query_complete),
            "compute_processes": [dict(value) for value in gpu_compute_processes],
        },
        "disk": dict(disk),
        "issues": issues,
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
    }
