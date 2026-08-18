from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


RESUME_COMPUTE_ADJUSTMENT_ROLE = "generation_resume_compute_adjustment"
RESUME_COMPUTE_ADJUSTMENT_SCHEMA_VERSION = 1
RESUME_COMPUTE_ADJUSTMENT_METHODS = {"cofitok", "dense_identity"}
_ORPHAN_METRICS_NAME = re.compile(
    r"^train_metrics_orphaned_at_resume_(\d{8})_([0-9a-f]{12})\.jsonl$"
)
_FINAL_COST_FORMULA = (
    "training_report.cumulative_elapsed_seconds + "
    "orphaned_compute_seconds_lower_bound"
)


def _normalized_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def resume_compute_adjustment_source_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = source.read_bytes()
    return {
        "path": source.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _identity_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        _normalized_path(actual.get("path"))
        == _normalized_path(expected.get("path"))
        and int(actual.get("bytes", -1)) == int(expected.get("bytes", -2))
        and actual.get("sha256") == expected.get("sha256")
    )


def _effective_batch(report: dict[str, Any]) -> int:
    config = report.get("config")
    config = config if isinstance(config, dict) else {}
    data = config.get("data")
    data = data if isinstance(data, dict) else {}
    optimization = config.get("optimization")
    optimization = optimization if isinstance(optimization, dict) else {}
    micro_batch = int(data.get("batch_size", 0) or 0)
    accumulation = int(
        optimization.get("gradient_accumulation_steps", 0) or 0
    )
    effective_batch = micro_batch * accumulation
    if effective_batch < 1:
        raise ValueError("training effective batch is invalid")
    return effective_batch


def _discovered_orphan_archives(report: dict[str, Any]) -> list[Path]:
    raw_output_dir = str(report.get("output_dir", "")).strip()
    if not raw_output_dir:
        return []
    output_dir = Path(raw_output_dir)
    if not output_dir.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in output_dir.glob(
                "train_metrics_orphaned_at_resume_????????_????????????.jsonl"
            )
            if path.is_file() and _ORPHAN_METRICS_NAME.match(path.name)
        ),
        key=lambda path: path.as_posix(),
    )


def validate_resume_compute_adjustment_source_identities(
    sources: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(sources, dict) or not set(sources).issubset(
        RESUME_COMPUTE_ADJUSTMENT_METHODS
    ):
        raise ValueError("resume-compute adjustment source methods are invalid")
    for method, identity in sources.items():
        path = _normalized_path(identity.get("path"))
        if (
            not path.endswith("/resume_compute_adjustment.json")
            or int(identity.get("bytes", 0)) < 1
            or not _valid_sha256(identity.get("sha256"))
        ):
            raise ValueError(
                f"resume-compute adjustment source identity is invalid: {method}"
            )


def _read_jsonl_prefix(path: Path, *, end_step: int) -> tuple[bytes, list[dict[str, Any]]]:
    raw_prefix: list[bytes] = []
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid canonical metrics JSONL at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError("canonical metrics JSONL row is not an object")
        step = int(row.get("step", -1))
        if step <= end_step:
            raw_prefix.append(raw)
            rows.append(row)
    return b"".join(raw_prefix), rows


def verify_resume_compute_adjustment_files(
    adjustment: dict[str, Any],
    *,
    effective_batch: int,
) -> dict[str, Any]:
    canonical = adjustment.get("canonical_metrics")
    canonical = canonical if isinstance(canonical, dict) else {}
    continuity = canonical.get("continuity_prefix")
    continuity = continuity if isinstance(continuity, dict) else {}
    metrics_path = Path(str(canonical.get("path", "")))
    if not metrics_path.is_file():
        raise ValueError("resume-compute canonical metrics file is missing")
    end_step = int(continuity.get("end_step", -1))
    if end_step < 1:
        raise ValueError("resume-compute continuity end step is invalid")
    prefix_payload, prefix_rows = _read_jsonl_prefix(
        metrics_path,
        end_step=end_step,
    )
    prefix_sha256 = hashlib.sha256(prefix_payload).hexdigest()
    if (
        len(prefix_payload) != int(continuity.get("bytes", -1))
        or len(prefix_rows) != int(continuity.get("row_count", -1))
        or prefix_sha256 != continuity.get("sha256")
    ):
        raise ValueError("resume-compute canonical continuity prefix changed")
    steps = [int(row.get("step", -1)) for row in prefix_rows]
    cumulative = [
        float(row.get("cumulative_elapsed_seconds", math.nan))
        for row in prefix_rows
    ]
    if (
        not steps
        or steps[-1] != end_step
        or not all(left < right for left, right in zip(steps, steps[1:]))
        or not all(math.isfinite(value) for value in cumulative)
        or not all(left < right for left, right in zip(cumulative, cumulative[1:]))
        or any(
            int(row.get("samples_seen", -1))
            != int(row.get("step", -1)) * effective_batch
            for row in prefix_rows
        )
    ):
        raise ValueError("resume-compute canonical continuity prefix is invalid")

    canonical_by_step = {int(row["step"]): row for row in prefix_rows}
    events = adjustment.get("recovery_events")
    if not isinstance(events, list) or not events:
        raise ValueError("resume-compute recovery events are missing")
    verified_orphans: list[dict[str, Any]] = []
    seen_orphan_paths: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("resume-compute recovery event is malformed")
        identity = event.get("orphan_metrics")
        identity = identity if isinstance(identity, dict) else {}
        path = Path(str(identity.get("path", "")))
        if not path.is_file():
            raise ValueError("resume-compute orphan metrics file is missing")
        actual = resume_compute_adjustment_source_identity(path)
        if not _identity_matches(actual, identity):
            raise ValueError("resume-compute orphan metrics identity changed")
        normalized_path = _normalized_path(actual["path"])
        if normalized_path in seen_orphan_paths:
            raise ValueError("resume-compute orphan metrics source is duplicated")
        seen_orphan_paths.add(normalized_path)

        name_match = _ORPHAN_METRICS_NAME.match(path.name)
        if name_match is None:
            raise ValueError("resume-compute orphan metrics filename is malformed")
        checkpoint_step = int(event.get("resume_checkpoint_step", -1))
        if (
            checkpoint_step != int(name_match.group(1))
            or not actual["sha256"].startswith(name_match.group(2))
        ):
            raise ValueError("resume-compute orphan metrics filename binding differs")

        _, orphan_rows = _read_jsonl_prefix(path, end_step=10**18)
        orphan_steps = [int(row.get("step", -1)) for row in orphan_rows]
        orphan_cumulative = [
            float(row.get("cumulative_elapsed_seconds", math.nan))
            for row in orphan_rows
        ]
        if (
            not orphan_steps
            or orphan_steps[0] <= checkpoint_step
            or not all(
                left < right for left, right in zip(orphan_steps, orphan_steps[1:])
            )
            or not all(math.isfinite(value) for value in orphan_cumulative)
            or not all(
                left < right
                for left, right in zip(orphan_cumulative, orphan_cumulative[1:])
            )
            or any(
                int(row.get("samples_seen", -1))
                != int(row.get("step", -1)) * effective_batch
                for row in orphan_rows
            )
        ):
            raise ValueError("resume-compute orphan metrics rows are invalid")
        if any(step not in canonical_by_step for step in orphan_steps):
            raise ValueError("resume-compute canonical prefix omits replacement steps")
        replacement_rows = [canonical_by_step[step] for step in orphan_steps]
        resume_bases = [
            float(row.get("cumulative_elapsed_seconds", math.nan))
            - float(row.get("elapsed_seconds", math.nan))
            for row in replacement_rows
        ]
        resume_base = resume_bases[0]
        checkpoint_row = canonical_by_step.get(checkpoint_step)
        if (
            checkpoint_row is None
            or not all(math.isfinite(value) for value in resume_bases)
            or not all(
                math.isclose(value, resume_base, rel_tol=0.0, abs_tol=1e-6)
                for value in resume_bases
            )
        ):
            raise ValueError("resume-compute canonical replacement rows are invalid")
        checkpoint_cumulative = float(
            checkpoint_row.get("cumulative_elapsed_seconds", math.nan)
        )
        orphaned_seconds = orphan_cumulative[-1] - resume_base
        optimizer_steps = orphan_steps[-1] - checkpoint_step
        expected_images = optimizer_steps * effective_batch
        if (
            event.get("canonical_replacement_steps") != orphan_steps
            or int(event.get("orphan_first_step", -1)) != orphan_steps[0]
            or int(event.get("orphan_last_step", -1)) != orphan_steps[-1]
            or int(event.get("orphan_logged_row_count", -1)) != len(orphan_rows)
            or not math.isclose(
                float(event.get("checkpoint_cumulative_elapsed_seconds", math.nan)),
                checkpoint_cumulative,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(event.get("canonical_resume_base_seconds", math.nan)),
                resume_base,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(event.get("orphan_end_cumulative_elapsed_seconds", math.nan)),
                orphan_cumulative[-1],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(event.get("orphaned_compute_seconds_lower_bound", math.nan)),
                orphaned_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or int(event.get("orphaned_optimizer_steps_lower_bound", -1))
            != optimizer_steps
            or int(event.get("orphaned_images_lower_bound", -1))
            != expected_images
            or event.get("canonical_replacement_resume_base_constant") is not True
            or event.get("canonical_replacement_first_row") != replacement_rows[0]
            or event.get("canonical_replacement_last_row") != replacement_rows[-1]
        ):
            raise ValueError("resume-compute recovery event differs from physical rows")
        verified_orphans.append(actual)
    return {
        "status": "verified",
        "canonical_metrics": metrics_path.resolve().as_posix(),
        "continuity_end_step": end_step,
        "continuity_prefix_bytes": len(prefix_payload),
        "continuity_prefix_sha256": prefix_sha256,
        "orphan_metrics": verified_orphans,
    }


def verify_resume_compute_adjustment_source(
    identity: dict[str, Any],
    *,
    method: str,
    training_report: dict[str, Any],
) -> dict[str, Any]:
    validate_resume_compute_adjustment_source_identities({method: identity})
    actual = resume_compute_adjustment_source_identity(identity["path"])
    if not _identity_matches(actual, identity):
        raise ValueError("resume-compute adjustment report changed after binding")
    with Path(identity["path"]).open("r", encoding="utf-8") as handle:
        adjustment = json.load(handle)
    if not isinstance(adjustment, dict):
        raise ValueError("resume-compute adjustment report is not an object")
    effective_batch = _effective_batch(training_report)
    physical_sources = verify_resume_compute_adjustment_files(
        adjustment,
        effective_batch=effective_batch,
    )
    summary = resume_compute_adjustment_summary(
        training_report,
        adjustment,
        effective_batch=effective_batch,
    )
    if summary["valid"] is not True:
        raise ValueError(
            "resume-compute adjustment report is invalid: "
            + "; ".join(summary["issues"])
        )
    training_cost = training_cost_summary(training_report, adjustment)
    if training_cost["valid"] is not True:
        raise ValueError("resume-compute adjusted training cost is invalid")
    return {
        "status": "verified",
        "identity": actual,
        "physical_sources": physical_sources,
        "summary": summary,
        "training_cost": training_cost,
    }


def resume_compute_adjustment_summary(
    report: dict[str, Any],
    adjustment: dict[str, Any] | None,
    *,
    effective_batch: int,
) -> dict[str, Any]:
    reconciliation = report.get("metrics_resume_reconciliation")
    reconciliation = reconciliation if isinstance(reconciliation, dict) else {}
    reconciliation_status = str(reconciliation.get("status", ""))
    reconciled_orphan_rows = int(reconciliation.get("orphaned_rows", 0) or 0)
    reconciliation_requires_adjustment = (
        reconciliation_status == "reconciled"
        and (
            reconciled_orphan_rows > 0
            or bool(reconciliation.get("orphan_archive"))
        )
    )
    discovered_orphans = _discovered_orphan_archives(report)
    required = reconciliation_requires_adjustment or bool(discovered_orphans)
    required_reasons = []
    if reconciliation_requires_adjustment:
        required_reasons.append("training reconciliation contains orphaned compute")
    if discovered_orphans:
        required_reasons.append("training run contains orphan metrics archives")
    if adjustment is None:
        return {
            "valid": not required,
            "required": required,
            "provided": False,
            "applied": False,
            "seconds": 0.0,
            "hours": 0.0,
            "event_count": 0,
            "orphaned_optimizer_steps_lower_bound": 0,
            "orphaned_images_lower_bound": 0,
            "discovered_orphan_archive_count": len(discovered_orphans),
            "covered_orphan_archive_count": 0,
            "required_reasons": required_reasons,
            "issues": (
                ["training orphaned compute lacks an adjustment report"]
                if required
                else []
            ),
        }

    issues: list[str] = []
    if int(adjustment.get("schema_version", 0)) != RESUME_COMPUTE_ADJUSTMENT_SCHEMA_VERSION:
        issues.append("adjustment schema differs")
    if adjustment.get("role") != RESUME_COMPUTE_ADJUSTMENT_ROLE:
        issues.append("adjustment role differs")
    if adjustment.get("status") != "pass":
        issues.append("adjustment status is not pass")
    boundary = adjustment.get("claim_boundary")
    boundary = boundary if isinstance(boundary, dict) else {}
    expected_boundary = {
        "adjusts_compute_accounting_only": True,
        "changes_training_trajectory": False,
        "changes_checkpoint": False,
        "changes_quality_metrics": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
    }
    if any(boundary.get(key) is not value for key, value in expected_boundary.items()):
        issues.append("adjustment claim boundary differs")

    canonical = adjustment.get("canonical_metrics")
    canonical = canonical if isinstance(canonical, dict) else {}
    expected_metrics = _normalized_path(
        reconciliation.get("metrics")
        or (Path(str(report.get("output_dir", ""))) / "train_metrics.jsonl")
    )
    if _normalized_path(canonical.get("path")) != expected_metrics:
        issues.append("adjustment canonical metrics path differs")
    continuity = canonical.get("continuity_prefix")
    continuity = continuity if isinstance(continuity, dict) else {}
    if (
        int(continuity.get("end_step", 0)) < 1
        or int(continuity.get("row_count", 0)) < 1
        or int(continuity.get("bytes", 0)) < 1
        or not _valid_sha256(continuity.get("sha256"))
        or continuity.get("steps_strictly_increasing") is not True
        or continuity.get("cumulative_elapsed_strictly_increasing") is not True
        or continuity.get("samples_seen_binding_verified") is not True
    ):
        issues.append("adjustment continuity prefix is invalid")

    events = adjustment.get("recovery_events")
    events = events if isinstance(events, list) else []
    seconds = 0.0
    optimizer_steps = 0
    orphaned_images = 0
    latest_orphan_matched = not reconciliation_requires_adjustment
    max_orphan_step = 0
    event_orphan_paths: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            issues.append("adjustment recovery event is malformed")
            continue
        identity = event.get("orphan_metrics")
        identity = identity if isinstance(identity, dict) else {}
        event_seconds = event.get("orphaned_compute_seconds_lower_bound")
        event_steps = int(event.get("orphaned_optimizer_steps_lower_bound", 0) or 0)
        event_images = int(event.get("orphaned_images_lower_bound", 0) or 0)
        first_step = int(event.get("orphan_first_step", 0) or 0)
        last_step = int(event.get("orphan_last_step", 0) or 0)
        logged_rows = int(event.get("orphan_logged_row_count", 0) or 0)
        if (
            not _positive_finite(event_seconds)
            or event_steps < 1
            or event_images != event_steps * effective_batch
            or first_step < 1
            or last_step < first_step
            or logged_rows < 1
            or int(identity.get("bytes", 0)) < 1
            or not _valid_sha256(identity.get("sha256"))
        ):
            issues.append("adjustment recovery event values are invalid")
            continue
        normalized_orphan = _normalized_path(identity.get("path"))
        if normalized_orphan in event_orphan_paths:
            issues.append("adjustment repeats an orphan metrics source")
        event_orphan_paths.add(normalized_orphan)
        output_dir = _normalized_path(report.get("output_dir"))
        if output_dir and not normalized_orphan.startswith(output_dir + "/"):
            issues.append("adjustment orphan metrics path escapes the training run")
        if reconciliation_requires_adjustment and (
            normalized_orphan == _normalized_path(reconciliation.get("orphan_archive"))
            and identity.get("sha256") == reconciliation.get("orphan_sha256")
            and logged_rows == reconciled_orphan_rows
        ):
            latest_orphan_matched = True
        seconds += float(event_seconds)
        optimizer_steps += event_steps
        orphaned_images += event_images
        max_orphan_step = max(max_orphan_step, last_step)
    if not events:
        issues.append("adjustment has no recovery events")
    if not latest_orphan_matched:
        issues.append("adjustment does not bind the reconciled orphan archive")
    discovered_paths = {
        _normalized_path(path.resolve().as_posix()) for path in discovered_orphans
    }
    if discovered_paths and event_orphan_paths != discovered_paths:
        issues.append("adjustment does not cover the physical orphan archive set")
    if int(continuity.get("end_step", 0)) < max_orphan_step:
        issues.append("adjustment continuity prefix ends before orphan replacement")
    completed_steps = int(report.get("completed_steps", report.get("target_steps", 0)) or 0)
    if int(continuity.get("end_step", 0)) > completed_steps:
        issues.append("adjustment continuity prefix exceeds completed training")

    summary = adjustment.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    if (
        not math.isclose(
            float(summary.get("orphaned_compute_seconds_lower_bound", math.nan)),
            seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or int(summary.get("orphaned_optimizer_steps_lower_bound", -1))
        != optimizer_steps
        or int(summary.get("orphaned_images_lower_bound", -1)) != orphaned_images
        or int(summary.get("event_count", -1)) != len(events)
        or summary.get("matched_compute_report_must_add_adjustment") is not True
        or summary.get("final_cost_formula") != _FINAL_COST_FORMULA
    ):
        issues.append("adjustment summary differs from recovery events")
    observed_canonical = float(
        canonical.get("observed_canonical_elapsed_seconds", math.nan)
    )
    observed_adjusted = float(
        summary.get("observed_adjusted_elapsed_seconds_lower_bound", math.nan)
    )
    if (
        not math.isfinite(observed_canonical)
        or not math.isclose(
            observed_canonical + seconds,
            observed_adjusted,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        issues.append("adjustment observed elapsed accounting differs")
    valid = not issues
    return {
        "valid": valid,
        "required": required,
        "provided": True,
        "applied": valid,
        "seconds": seconds if valid else 0.0,
        "hours": seconds / 3600.0 if valid else 0.0,
        "event_count": len(events),
        "orphaned_optimizer_steps_lower_bound": optimizer_steps,
        "orphaned_images_lower_bound": orphaned_images,
        "continuity_end_step": continuity.get("end_step"),
        "discovered_orphan_archive_count": len(discovered_orphans),
        "covered_orphan_archive_count": len(event_orphan_paths & discovered_paths),
        "required_reasons": required_reasons,
        "issues": issues,
    }


def training_cost_summary(
    report: dict[str, Any],
    resume_compute_adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = report["config"]
    target_steps = int(report["target_steps"])
    micro_batch = int(config["data"]["batch_size"])
    accumulation = int(config["optimization"]["gradient_accumulation_steps"])
    effective_batch = micro_batch * accumulation
    expected_samples = target_steps * effective_batch
    samples_seen = int(report.get("final_metrics", {}).get("samples_seen", -1))
    reported_elapsed_seconds = float(report.get("elapsed_seconds", math.nan))
    adjustment = resume_compute_adjustment_summary(
        report,
        resume_compute_adjustment,
        effective_batch=effective_batch,
    )
    elapsed_seconds = reported_elapsed_seconds + float(adjustment["seconds"])
    peak_vram_bytes = int(report.get("peak_vram_bytes", -1))
    cuda_run = config["runtime"].get("device") == "cuda"
    valid = (
        samples_seen == expected_samples
        and adjustment["valid"] is True
        and math.isfinite(elapsed_seconds)
        and elapsed_seconds > 0.0
        and peak_vram_bytes >= 0
        and (not cuda_run or peak_vram_bytes > 0)
    )
    return {
        "valid": valid,
        "target_steps": target_steps,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "expected_samples_seen": expected_samples,
        "samples_seen": samples_seen,
        "reported_elapsed_seconds": reported_elapsed_seconds,
        "resume_compute_adjustment": adjustment,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_role": (
            "physical_lower_bound_including_orphaned_recovery_compute"
            if adjustment["applied"] is True
            else "reported_training_elapsed_seconds"
        ),
        "images_per_second": (
            samples_seen / elapsed_seconds
            if samples_seen >= 0 and math.isfinite(elapsed_seconds) and elapsed_seconds > 0.0
            else None
        ),
        "peak_vram_bytes": peak_vram_bytes,
    }
