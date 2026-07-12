from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report


def load_sampling_progress_state(
    path: str | Path,
    *,
    sampling_manifest_sha256: str,
    total_samples: int,
    start_index: int,
    prefix_budgets: list[int],
) -> dict[str, Any]:
    if len(sampling_manifest_sha256) != 64:
        raise ValueError("sampling manifest SHA256 is malformed")
    if total_samples < 1 or start_index < 0 or not prefix_budgets:
        raise ValueError("sampling progress identity is invalid")
    progress_path = Path(path)
    previous = None
    if progress_path.is_file():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "sampling_manifest_sha256": sampling_manifest_sha256,
            "total_samples": total_samples,
            "start_index": start_index,
            "prefix_budgets": prefix_budgets,
        }
        actual = {key: previous.get(key) for key in expected}
        if actual != expected:
            raise ValueError("existing sampling progress belongs to another manifest")
        prior_elapsed = float(previous.get("cumulative_elapsed_seconds", math.nan))
        prior_completed = int(previous.get("completed_samples", -1))
        prior_invocation = int(previous.get("invocation", 0))
        if (
            not math.isfinite(prior_elapsed)
            or prior_elapsed < 0.0
            or not 0 <= prior_completed <= total_samples
            or prior_invocation < 1
        ):
            raise ValueError("existing sampling progress is invalid")
    else:
        prior_elapsed = 0.0
        prior_completed = 0
        prior_invocation = 0
    return {
        "schema_version": 1,
        "path": progress_path.resolve().as_posix(),
        "sampling_manifest_sha256": sampling_manifest_sha256,
        "total_samples": total_samples,
        "start_index": start_index,
        "prefix_budgets": list(prefix_budgets),
        "prior_cumulative_elapsed_seconds": prior_elapsed,
        "prior_completed_samples": prior_completed,
        "invocation": prior_invocation + 1,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def build_sampling_progress(
    state: dict[str, Any],
    *,
    status: str,
    completed_samples: int,
    invocation_elapsed_seconds: float,
    error: BaseException | None = None,
    sample_sets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"running", "failed", "completed"}:
        raise ValueError("unsupported sampling progress status")
    total_samples = int(state["total_samples"])
    completed_samples = max(int(state["prior_completed_samples"]), completed_samples)
    if not 0 <= completed_samples <= total_samples:
        raise ValueError("completed sampling count is outside the requested range")
    if not math.isfinite(invocation_elapsed_seconds) or invocation_elapsed_seconds < 0.0:
        raise ValueError("invocation elapsed seconds must be finite and non-negative")
    if status == "failed" and error is None:
        raise ValueError("failed sampling progress requires an error")
    if status != "failed" and error is not None:
        raise ValueError("sampling error is valid only for failed progress")
    if status == "completed":
        if completed_samples != total_samples or sample_sets is None:
            raise ValueError("completed sampling progress requires all samples and digests")
    cumulative_elapsed = (
        float(state["prior_cumulative_elapsed_seconds"]) + invocation_elapsed_seconds
    )
    throughput = completed_samples / cumulative_elapsed if cumulative_elapsed > 0.0 else None
    remaining = total_samples - completed_samples
    eta_seconds = remaining / throughput if throughput is not None and throughput > 0.0 else None
    payload = {
        "schema_version": 1,
        "status": status,
        "sampling_manifest_sha256": state["sampling_manifest_sha256"],
        "total_samples": total_samples,
        "start_index": int(state["start_index"]),
        "prefix_budgets": list(state["prefix_budgets"]),
        "completed_samples": completed_samples,
        "completed_fraction": completed_samples / total_samples,
        "last_completed_index": (
            int(state["start_index"]) + completed_samples - 1
            if completed_samples > 0
            else None
        ),
        "invocation": int(state["invocation"]),
        "invocation_elapsed_seconds": invocation_elapsed_seconds,
        "cumulative_elapsed_seconds": cumulative_elapsed,
        "samples_per_second": throughput,
        "eta_seconds": eta_seconds,
        "hostname": state["hostname"],
        "pid": int(state["pid"]),
        "sample_sets": sample_sets,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }
    return payload


def write_sampling_progress(path: str | Path, payload: dict[str, Any]) -> None:
    write_json_report(Path(path), payload)
