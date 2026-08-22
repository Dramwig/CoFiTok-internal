from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any

from cofitok.generation_pair import generation_pair_contract
from cofitok.reporting import file_sha256, git_provenance, write_json_report
from cofitok.training.metrics import validate_metrics_resume_history


_SHARED_SCHEDULES = (
    "rollout_consistency",
    "ema_teacher_consistency",
)
_SCHEDULE_PHASES = (
    "disabled",
    "inactive",
    "warmup",
    "full_scale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound diagnostic comparison of matched CoFiTok and "
            "dense training trajectories through an exact cutoff step."
        )
    )
    parser.add_argument("--cofitok-metrics", type=Path, required=True)
    parser.add_argument("--dense-metrics", type=Path, required=True)
    parser.add_argument("--cofitok-manifest", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument(
        "--cofitok-reconciliation",
        type=Path,
        action="append",
        help=(
            "Historical CoFiTok metrics-resume reconciliation report. Repeat in "
            "strictly increasing resume-step order when the run resumed more than once."
        ),
    )
    parser.add_argument(
        "--dense-reconciliation",
        type=Path,
        action="append",
        help=(
            "Historical dense metrics-resume reconciliation report. Repeat in "
            "strictly increasing resume-step order when the run resumed more than once."
        ),
    )
    parser.add_argument("--cutoff-step", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--cofitok-origin")
    parser.add_argument("--dense-origin")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help=(
            "Optionally write exact metrics-prefix snapshots into this directory and "
            "bind the report to those snapshots instead of the growing inputs."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": path.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _builder_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    project = resolved.parents[1]
    git = git_provenance(project)
    revision = git.get("revision")
    branch = git.get("branch")
    if (
        git.get("tracked_dirty") is not False
        or not _is_hex_digest(revision, length=40)
        or not isinstance(branch, str)
        or not branch
    ):
        raise ValueError("trajectory builder Git provenance is invalid or dirty")
    return {
        "path": resolved.relative_to(project).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
        "git": git,
    }


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _is_hex_digest(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _normalize_reconciliation(
    reconciliation: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(reconciliation, dict):
        raise ValueError(f"{label} metrics resume reconciliation is not an object")
    if int(reconciliation.get("schema_version", 0)) != 1:
        raise ValueError(f"{label} metrics resume reconciliation schema is invalid")
    status = reconciliation.get("status")
    if status not in {"absent", "unchanged", "reconciled"}:
        raise ValueError(f"{label} metrics resume reconciliation status is invalid")
    try:
        resume_step = int(reconciliation["resume_step"])
        retained_rows = int(reconciliation["retained_rows"])
        orphaned_rows = int(reconciliation["orphaned_rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} metrics resume reconciliation counts are invalid"
        ) from error
    if retained_rows < 0 or orphaned_rows < 0:
        raise ValueError(f"{label} metrics resume reconciliation counts are negative")
    orphan_archive = reconciliation.get("orphan_archive")
    orphan_sha256 = reconciliation.get("orphan_sha256")
    if status in {"absent", "unchanged"}:
        if (
            orphaned_rows != 0
            or orphan_archive is not None
            or orphan_sha256 is not None
        ):
            raise ValueError(
                f"{label} zero-orphan metrics reconciliation contains orphan evidence"
            )
        if status == "absent" and retained_rows != 0:
            raise ValueError(
                f"{label} absent metrics reconciliation retained rows are nonzero"
            )
    else:
        if (
            orphaned_rows < 1
            or not isinstance(orphan_archive, str)
            or not orphan_archive
            or not _is_hex_digest(orphan_sha256, length=64)
        ):
            raise ValueError(
                f"{label} reconciled metrics resume lacks bound orphan evidence"
            )
    return {
        "resume_step": resume_step,
        "first_post_resume_step": resume_step + 1,
        "reconciliation_status": status,
        "retained_rows": retained_rows,
        "orphaned_rows": orphaned_rows,
        "orphan_archive": orphan_archive,
        "orphan_sha256": orphan_sha256,
    }


def _validate_metrics_resume(
    manifest: dict[str, Any],
    *,
    label: str,
    reconciliation_history: list[dict[str, Any]] | None = None,
    resume_history_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "metrics_resume_history" in manifest and resume_history_evidence is None:
        raise ValueError(
            f"{label} manifest metrics resume history was not physically resolved"
        )
    resume = manifest.get("resume")
    reconciliation = manifest.get("metrics_resume_reconciliation")
    if resume is None:
        if reconciliation is not None:
            raise ValueError(
                f"{label} manifest has metrics resume evidence without a resume checkpoint"
            )
        if reconciliation_history:
            raise ValueError(
                f"{label} reconciliation history exists without a resume checkpoint"
            )
        if resume_history_evidence is not None and int(
            resume_history_evidence.get("reconciliation_event_count", -1)
        ) != 0:
            raise ValueError(
                f"{label} resume history contains events without a resume checkpoint"
            )
        return {
            "resumed": False,
            "checkpoint": None,
            "resume_step": None,
            "first_post_resume_step": None,
            "reconciliation_status": None,
            "retained_rows": None,
            "orphaned_rows": None,
            "orphan_archive": None,
            "orphan_sha256": None,
            "event_count": 0,
            "events": [],
            "history_evidence": resume_history_evidence
            or {
                "mode": "manifest_without_append_only_history",
                "history_sha256": None,
                "legacy_history_complete": False,
                "reconciliation_event_count": 0,
                "complete_recovery_chain_verified": False,
                "limitation": (
                    "The manifest predates append-only metrics resume history, so "
                    "absence of a recovery event is not independently proven."
                ),
            },
        }
    if not isinstance(resume, str) or not resume:
        raise ValueError(f"{label} manifest has an invalid resume checkpoint")
    checkpoint_name = resume.replace("\\", "/").rsplit("/", 1)[-1]
    checkpoint_match = re.fullmatch(r"checkpoint_step_(\d{8})\.pt", checkpoint_name)
    if checkpoint_match is None:
        raise ValueError(f"{label} manifest resume checkpoint name is invalid")
    checkpoint_step = int(checkpoint_match.group(1))
    current = _normalize_reconciliation(reconciliation, label=label)
    if current["resume_step"] != checkpoint_step:
        raise ValueError(
            f"{label} resume checkpoint and metrics reconciliation steps differ"
        )
    if reconciliation_history is None:
        events = [current]
    else:
        events = [
            _normalize_reconciliation(
                event,
                label=f"{label} reconciliation history event {index}",
            )
            for index, event in enumerate(reconciliation_history)
        ]
        if not events or events[-1] != current:
            raise ValueError(
                f"{label} reconciliation history does not end at the manifest resume"
            )
    resume_steps = [int(event["resume_step"]) for event in events]
    if resume_steps != sorted(set(resume_steps)):
        raise ValueError(
            f"{label} reconciliation history steps are not strictly increasing"
        )
    if resume_history_evidence is None:
        resume_history_evidence = {
            "mode": (
                "legacy_cli_reconciliation_reports"
                if reconciliation_history is not None
                else "manifest_current_reconciliation_only"
            ),
            "history_sha256": None,
            "legacy_history_complete": False,
            "reconciliation_event_count": len(events),
            "complete_recovery_chain_verified": False,
            "limitation": (
                "No append-only manifest metrics_resume_history proves that every "
                "earlier exact-resume event is represented."
            ),
        }
    elif int(resume_history_evidence.get("reconciliation_event_count", -1)) != len(
        events
    ):
        raise ValueError(f"{label} resume history event count differs from its evidence")
    return {
        "resumed": True,
        "checkpoint": resume,
        **current,
        "event_count": len(events),
        "events": events,
        "history_evidence": resume_history_evidence,
    }


def _load_reconciliation_history(
    paths: list[Path] | None,
    *,
    metrics_path: Path,
    manifest: dict[str, Any],
    label: str,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    if paths is None:
        return None, []
    if not paths:
        raise ValueError(f"{label} reconciliation path list is empty")
    resolved_metrics = metrics_path.resolve()
    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    resolved_paths: list[Path] = []
    for index, path in enumerate(paths):
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"{label} reconciliation report is absent: {resolved}")
        payload = _read_object(resolved)
        normalized = _normalize_reconciliation(
            payload,
            label=f"{label} reconciliation report {index}",
        )
        payload_metrics = payload.get("metrics")
        if (
            not isinstance(payload_metrics, str)
            or Path(payload_metrics).resolve() != resolved_metrics
        ):
            raise ValueError(
                f"{label} reconciliation report {index} binds another metrics file"
            )
        orphan_source = None
        if normalized["reconciliation_status"] == "reconciled":
            archive = Path(str(normalized["orphan_archive"])).resolve()
            if not archive.is_file():
                raise ValueError(
                    f"{label} reconciliation report {index} orphan archive is absent"
                )
            orphan_source = _source(archive)
            if orphan_source["sha256"] != normalized["orphan_sha256"]:
                raise ValueError(
                    f"{label} reconciliation report {index} orphan SHA256 differs"
                )
        events.append(payload)
        resolved_paths.append(resolved)
        sources.append(
            {
                "report": _source(resolved),
                "orphan_archive": orphan_source,
                "resume_step": normalized["resume_step"],
            }
        )
    current = manifest.get("metrics_resume_reconciliation")
    if not isinstance(current, dict):
        raise ValueError(f"{label} manifest lacks metrics resume reconciliation")
    current_report = current.get("report")
    if isinstance(current_report, str):
        if Path(current_report).resolve() != resolved_paths[-1]:
            raise ValueError(
                f"{label} latest reconciliation source does not match the manifest report"
            )
    elif current_report is None and current.get("status") == "unchanged":
        if _normalize_reconciliation(
            events[-1],
            label=f"{label} latest persisted reconciliation",
        ) != _normalize_reconciliation(
            current,
            label=f"{label} manifest reconciliation",
        ):
            events.append(current)
            sources.append(
                {
                    "report": None,
                    "orphan_archive": None,
                    "resume_step": int(current["resume_step"]),
                    "manifest_bound": True,
                }
            )
    else:
        raise ValueError(
            f"{label} latest reconciliation source is not bound by the manifest"
        )
    return events, sources


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_history_artifact(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{label} path is unsafe")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*pure.parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the training output") from error
    return resolved


def _resolve_document_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _canonical_reconciliation_document(
    document: Any,
    *,
    root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError(f"{label} is not an object")
    result = json.loads(json.dumps(document))
    for field in ("report", "metrics", "orphan_archive"):
        value = result.get(field)
        if value is not None:
            result[field] = _resolve_document_path(
                root,
                value,
                label=f"{label} {field}",
            ).as_posix()
    return result


def _load_history_report(
    identity: dict[str, Any],
    *,
    root: Path,
    metrics_path: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = _resolve_history_artifact(
        root,
        identity.get("path"),
        label=f"{label} report",
    )
    report = _read_object(report_path)
    normalized = _normalize_reconciliation(report, label=label)
    report_metrics = _resolve_document_path(
        root,
        report.get("metrics"),
        label=f"{label} metrics",
    )
    if report_metrics != metrics_path.resolve():
        raise ValueError(f"{label} report binds another metrics file")
    orphan_source = None
    if normalized["reconciliation_status"] == "reconciled":
        orphan_path = _resolve_document_path(
            root,
            normalized["orphan_archive"],
            label=f"{label} orphan archive",
        )
        orphan_source = _source(orphan_path)
        if orphan_source["sha256"] != normalized["orphan_sha256"]:
            raise ValueError(f"{label} orphan SHA256 differs")
    return report, {
        "report": _source(report_path),
        "orphan_archive": orphan_source,
        "resume_step": normalized["resume_step"],
    }


def _legacy_reconciliation_sha256(report: dict[str, Any], *, root: Path) -> str:
    metrics_path = _resolve_document_path(
        root,
        report.get("metrics"),
        label="legacy metrics reconciliation metrics",
    )
    try:
        relative_metrics = metrics_path.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            "legacy metrics reconciliation names metrics outside the training output"
        ) from error
    core = {
        "status": report.get("status"),
        "resume_step": int(report.get("resume_step", -1)),
        "metrics": relative_metrics,
        "retained_rows": int(report.get("retained_rows", -1)),
        "orphaned_rows": int(report.get("orphaned_rows", -1)),
        "orphan_sha256": report.get("orphan_sha256"),
    }
    return _canonical_json_sha256(core)


def _assert_current_manifest_history_binding(
    manifest: dict[str, Any],
    *,
    root: Path,
    event: dict[str, Any],
    report: dict[str, Any],
    label: str,
) -> None:
    checkpoint = event["checkpoint"]
    expected_checkpoint = _resolve_history_artifact(
        root,
        checkpoint.get("path"),
        label=f"{label} latest history checkpoint",
    )
    actual_checkpoint = _resolve_document_path(
        root,
        manifest.get("resume"),
        label=f"{label} manifest resume checkpoint",
    )
    if actual_checkpoint != expected_checkpoint:
        raise ValueError(
            f"{label} manifest resume checkpoint differs from the latest history event"
        )
    reconciliation = event["reconciliation"]
    report_identity = reconciliation["report"]
    report_path = _resolve_history_artifact(
        root,
        report_identity.get("path"),
        label=f"{label} latest reconciliation report",
    )
    expected = {
        **report,
        "report": report_path.as_posix(),
        "report_bytes": report_identity["bytes"],
        "report_sha256": report_identity["sha256"],
    }
    actual = manifest.get("metrics_resume_reconciliation")
    if _canonical_reconciliation_document(
        actual,
        root=root,
        label=f"{label} manifest reconciliation",
    ) != _canonical_reconciliation_document(
        expected,
        root=root,
        label=f"{label} history reconciliation",
    ):
        raise ValueError(
            f"{label} manifest reconciliation differs from the latest history event"
        )


def _load_manifest_resume_history(
    manifest_path: Path,
    *,
    manifest: dict[str, Any],
    metrics_path: Path,
    label: str,
) -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    raw_history = manifest.get("metrics_resume_history")
    if raw_history is None:
        return None, [], None
    root = manifest_path.resolve().parent
    validated = validate_metrics_resume_history(raw_history, output_dir=root)
    journal_path = root / "metrics_resume_history.json"
    if not journal_path.is_file():
        raise FileNotFoundError(f"{label} metrics resume history journal is missing")
    journal = _read_object(journal_path)
    if journal != validated:
        raise ValueError(f"{label} manifest and journal metrics resume histories diverge")

    records: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    unresolved_legacy: list[dict[str, Any]] = []
    for index, legacy in enumerate(validated["legacy_reconciliations"]):
        report_identity = legacy.get("report")
        if report_identity is None:
            unresolved_legacy.append(
                {
                    "resume_step": legacy["resume_step"],
                    "status": legacy["status"],
                    "reconciliation_sha256": legacy["reconciliation_sha256"],
                    "checkpoint_binding_available": False,
                    "report_binding_available": False,
                }
            )
            continue
        report, source = _load_history_report(
            report_identity,
            root=root,
            metrics_path=metrics_path,
            label=f"{label} legacy history reconciliation {index}",
        )
        if _legacy_reconciliation_sha256(report, root=root) != legacy.get(
            "reconciliation_sha256"
        ):
            raise ValueError(
                f"{label} legacy reconciliation digest differs from its report"
            )
        source.update(
            {
                "history_kind": "legacy_unbound",
                "checkpoint_binding_available": False,
            }
        )
        records.append((report, source, False))

    latest_event_report: dict[str, Any] | None = None
    for index, event in enumerate(validated["events"]):
        report, source = _load_history_report(
            event["reconciliation"]["report"],
            root=root,
            metrics_path=metrics_path,
            label=f"{label} checkpoint-bound history reconciliation {index}",
        )
        if int(report.get("resume_step", -1)) != int(event["resume_step"]):
            raise ValueError(f"{label} history event and report resume steps differ")
        checkpoint = event["checkpoint"]
        checkpoint_path = _resolve_history_artifact(
            root,
            checkpoint.get("path"),
            label=f"{label} history checkpoint {index}",
        )
        integrity_path = _resolve_history_artifact(
            root,
            checkpoint["integrity_manifest"].get("path"),
            label=f"{label} history checkpoint integrity {index}",
        )
        source.update(
            {
                "history_kind": "checkpoint_bound",
                "checkpoint_binding_available": True,
                "checkpoint": {
                    "path": checkpoint_path.as_posix(),
                    "bytes": checkpoint["bytes"],
                    "sha256": checkpoint["sha256"],
                    "payload_present": checkpoint_path.is_file(),
                    "payload_bytes_verified": (
                        checkpoint_path.is_file()
                        and checkpoint_path.stat().st_size == checkpoint["bytes"]
                    ),
                    "payload_sha256_recomputed": False,
                    "integrity_manifest": _source(integrity_path),
                },
            }
        )
        records.append((report, source, True))
        latest_event_report = report

    selected: dict[int, tuple[dict[str, Any], dict[str, Any], bool]] = {}
    for report, source, checkpoint_bound in records:
        normalized = _normalize_reconciliation(report, label=label)
        step = int(normalized["resume_step"])
        previous = selected.get(step)
        if previous is not None:
            previous_normalized = _normalize_reconciliation(previous[0], label=label)
            if previous_normalized != normalized:
                raise ValueError(
                    f"{label} resume history contains divergent events at step {step}"
                )
            if checkpoint_bound and not previous[2]:
                selected[step] = (report, source, checkpoint_bound)
        else:
            selected[step] = (report, source, checkpoint_bound)
    ordered = [selected[step] for step in sorted(selected)]
    reconciliations = [entry[0] for entry in ordered]
    sources = [entry[1] for entry in ordered]

    if validated["events"]:
        if latest_event_report is None:
            raise ValueError(f"{label} latest metrics resume event lacks its report")
        _assert_current_manifest_history_binding(
            manifest,
            root=root,
            event=validated["events"][-1],
            report=latest_event_report,
            label=label,
        )
    elif manifest.get("resume") is not None or manifest.get(
        "metrics_resume_reconciliation"
    ) is not None:
        raise ValueError(
            f"{label} manifest resume is not represented by a checkpoint-bound history event"
        )

    legacy_complete = validated["legacy_history_complete"] is True
    complete_recovery_chain_verified = (
        legacy_complete
        and not validated["legacy_reconciliations"]
        and len(reconciliations) == len(validated["events"])
    )
    evidence = {
        "schema_version": 1,
        "mode": "manifest_metrics_resume_history_v1",
        "history_sha256": validated["history_sha256"],
        "history_sha256_verified": True,
        "manifest_journal_equality_verified": True,
        "journal": _source(journal_path),
        "legacy_history_complete": legacy_complete,
        "known_legacy_reconciliation_count": len(
            validated["legacy_reconciliations"]
        ),
        "checkpoint_bound_event_count": len(validated["events"]),
        "reconciliation_event_count": len(reconciliations),
        "known_physical_report_bindings_verified": True,
        "known_physical_orphan_bindings_verified": True,
        "metrics_prefix_bindings_verified": True,
        "current_manifest_binding_verified": True,
        "manual_cli_corroborated": False,
        "complete_recovery_chain_verified": complete_recovery_chain_verified,
        "unresolved_legacy_reconciliations": unresolved_legacy,
        "limitation": (
            None
            if complete_recovery_chain_verified
            else (
                "Legacy reconciliation evidence predates checkpoint-bound append-only "
                "history, so the complete recovery chain is not proven."
            )
        ),
    }
    return reconciliations, sources, evidence


def _normalized_reconciliation_sequence(
    events: list[dict[str, Any]] | None,
    *,
    label: str,
) -> list[dict[str, Any]]:
    return [
        _normalize_reconciliation(event, label=f"{label} event {index}")
        for index, event in enumerate(events or [])
    ]


def _select_reconciliation_history(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    metrics_path: Path,
    manual_paths: list[Path] | None,
    label: str,
) -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    automatic, automatic_sources, evidence = _load_manifest_resume_history(
        manifest_path,
        manifest=manifest,
        metrics_path=metrics_path,
        label=label,
    )
    if automatic is None:
        manual, manual_sources = _load_reconciliation_history(
            manual_paths,
            metrics_path=metrics_path,
            manifest=manifest,
            label=label,
        )
        if manual is None:
            return None, [], None
        return manual, manual_sources, {
            "schema_version": 1,
            "mode": "legacy_cli_reconciliation_reports",
            "history_sha256": None,
            "legacy_history_complete": False,
            "reconciliation_event_count": len(manual),
            "complete_recovery_chain_verified": False,
            "manual_cli_corroborated": True,
            "limitation": (
                "CLI-bound reconciliation reports do not prove that no earlier "
                "resume event is omitted."
            ),
        }
    if manual_paths is None:
        return automatic, automatic_sources, evidence

    manual, manual_sources = _load_reconciliation_history(
        manual_paths,
        metrics_path=metrics_path,
        manifest=manifest,
        label=label,
    )
    if _normalized_reconciliation_sequence(
        manual,
        label=f"{label} CLI reconciliation history",
    ) != _normalized_reconciliation_sequence(
        automatic,
        label=f"{label} manifest reconciliation history",
    ):
        raise ValueError(
            f"{label} CLI reconciliation history diverges from manifest metrics resume history"
        )
    automatic_reports = [
        entry["report"]["path"]
        for entry in automatic_sources
        if entry.get("report") is not None
    ]
    manual_reports = [
        entry["report"]["path"]
        for entry in manual_sources
        if entry.get("report") is not None
    ]
    if manual_reports != automatic_reports:
        raise ValueError(
            f"{label} CLI reconciliation report paths diverge from manifest history"
        )
    corroborated = {**evidence, "manual_cli_corroborated": True}
    return automatic, automatic_sources, corroborated


def _read_metrics_prefix(path: Path, *, cutoff_step: int) -> dict[str, Any]:
    if cutoff_step < 1:
        raise ValueError("cutoff_step must be positive")
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    prefix_parts: list[bytes] = []
    seen_after_cutoff = False
    for line_number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number} has an invalid step") from error
        if step <= cutoff_step:
            if seen_after_cutoff:
                raise ValueError(f"{path} contains a cutoff row after a later step")
            rows.append(row)
            prefix_parts.append(raw_line)
        else:
            seen_after_cutoff = True
    if not rows or int(rows[-1]["step"]) != cutoff_step:
        raise ValueError(f"{path} does not contain the exact cutoff step {cutoff_step}")
    prefix = b"".join(prefix_parts)
    return {
        "rows": rows,
        "raw_prefix": prefix,
        "identity": {
            "row_count": len(rows),
            "last_step": cutoff_step,
            "bytes": len(prefix),
            "sha256": hashlib.sha256(prefix).hexdigest(),
        },
        "observed_file": _source(path),
    }


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    label: str,
    expected_revision: str,
    expected_branch: str,
    reconciliation_history: list[dict[str, Any]] | None = None,
    resume_history_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git = manifest.get("git")
    if not isinstance(git, dict):
        raise ValueError(f"{label} manifest lacks Git identity")
    if git.get("revision") != expected_revision:
        raise ValueError(f"{label} manifest revision does not match the expected revision")
    if git.get("branch") != expected_branch:
        raise ValueError(f"{label} manifest branch does not match the expected branch")
    if git.get("dirty") is not False:
        raise ValueError(f"{label} manifest used a dirty tracked worktree")
    provenance = manifest.get("dataset_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") != "pass"
        or provenance.get("formal") is not True
    ):
        raise ValueError(f"{label} manifest lacks passing dataset provenance")
    dataset_identity = provenance.get("identity_sha256")
    runtime_identity = manifest.get("runtime_environment_sha256")
    if not _is_hex_digest(dataset_identity, length=64):
        raise ValueError(f"{label} manifest has an invalid dataset identity")
    if not _is_hex_digest(runtime_identity, length=64):
        raise ValueError(f"{label} manifest has an invalid runtime identity")
    parameters = int(manifest.get("parameter_count", 0))
    if parameters < 1:
        raise ValueError(f"{label} manifest lacks a positive parameter count")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} manifest lacks its resolved config")
    if config.get("data", {}).get("dataset") != provenance.get("dataset"):
        raise ValueError(f"{label} manifest config and provenance datasets differ")
    return {
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
            "dirty": False,
        },
        "dataset": provenance.get("dataset"),
        "dataset_identity_sha256": dataset_identity,
        "runtime_environment_sha256": runtime_identity,
        "parameter_count": parameters,
        "config": config,
        "metrics_resume": _validate_metrics_resume(
            manifest,
            label=label,
            reconciliation_history=reconciliation_history,
            resume_history_evidence=resume_history_evidence,
        ),
    }


def _expected_steps(
    *,
    cutoff_step: int,
    log_interval: int,
    metrics_resume: dict[str, Any],
) -> list[int]:
    if log_interval < 1:
        raise ValueError("log_interval must be positive")
    if cutoff_step % log_interval:
        raise ValueError("cutoff_step must be divisible by the matched log interval")
    expected = {1, *range(log_interval, cutoff_step + 1, log_interval)}
    events = metrics_resume.get("events")
    if events is None:
        first_post_resume_step = metrics_resume.get("first_post_resume_step")
        events = (
            []
            if first_post_resume_step is None
            else [{"first_post_resume_step": first_post_resume_step}]
        )
    for event in events:
        step = int(event["first_post_resume_step"])
        if step <= cutoff_step:
            expected.add(step)
    return sorted(expected)


def _protected_checkpoint_steps(
    config: dict[str, Any],
    *,
    label: str,
) -> list[int]:
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} config lacks runtime settings")
    raw = runtime.get("protected_checkpoint_steps", [])
    if not isinstance(raw, list):
        raise ValueError(f"{label} protected checkpoint steps are invalid")
    try:
        result = [int(step) for step in raw]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} protected checkpoint steps are invalid") from error
    if any(step < 1 for step in result) or result != sorted(set(result)):
        raise ValueError(
            f"{label} protected checkpoint steps are not strictly increasing"
        )
    return result


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    cutoff_step: int,
    log_interval: int,
    evaluation_interval: int,
    effective_batch: int,
    metrics_resume: dict[str, Any],
    protected_checkpoint_steps: list[int],
) -> dict[str, Any]:
    steps = [int(row["step"]) for row in rows]
    required = _expected_steps(
        cutoff_step=cutoff_step,
        log_interval=log_interval,
        metrics_resume=metrics_resume,
    )
    base = {1, *range(log_interval, cutoff_step + 1, log_interval)}
    required_set = set(required)
    observed_set = set(steps)
    protected_boundaries = {
        step + 1
        for step in protected_checkpoint_steps
        if step + 1 <= cutoff_step
    }
    extra = observed_set - required_set
    if extra - protected_boundaries:
        raise ValueError(f"{label} metrics contain an unbound irregular logging step")
    expected = sorted(required_set | extra)
    if steps != expected:
        raise ValueError(f"{label} metrics do not follow the exact logging schedule")
    for event in metrics_resume.get("events", []):
        resume_step = int(event["resume_step"])
        retained_rows = int(event["retained_rows"])
        observed_retained_rows = sum(step <= resume_step for step in steps)
        if int(resume_step) <= cutoff_step:
            if observed_retained_rows != int(retained_rows):
                raise ValueError(
                    f"{label} metrics resume retained-row count does not match "
                    f"the prefix at step {resume_step}"
                )
        elif int(retained_rows) < observed_retained_rows:
            raise ValueError(
                f"{label} metrics resume retained-row count is smaller than "
                f"the prefix at step {resume_step}"
            )
    for row in rows:
        step = int(row["step"])
        if int(row.get("samples_seen", -1)) != step * effective_batch:
            raise ValueError(f"{label} samples_seen is invalid at step {step}")
        for field, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _finite_number(value, label=f"{label} step {step} {field}")
        for field in (
            "epsilon",
            "rollout_consistency",
            "rollout_consistency_scale",
            "ema_teacher_consistency",
            "ema_teacher_consistency_scale",
        ):
            _finite_number(row.get(field), label=f"{label} step {step} {field}")
    validation_rows = [row for row in rows if "validation_epsilon_mse" in row]
    expected_validation_steps = list(
        range(evaluation_interval, cutoff_step + 1, evaluation_interval)
    )
    if [int(row["step"]) for row in validation_rows] != expected_validation_steps:
        raise ValueError(f"{label} validation events do not follow the exact schedule")
    for event_index, row in enumerate(validation_rows):
        step = int(row["step"])
        mse = _finite_number(
            row.get("validation_epsilon_mse"),
            label=f"{label} step {step} validation_epsilon_mse",
        )
        if mse < 0.0:
            raise ValueError(f"{label} validation MSE is negative at step {step}")
        expected_metadata = {
            "validation_event_index": event_index,
            "validation_batch_index": event_index,
        }
        for field, expected_value in expected_metadata.items():
            if int(row.get(field, -1)) != expected_value:
                raise ValueError(f"{label} {field} is invalid at step {step}")
        if int(row.get("validation_num_images", 0)) < 1:
            raise ValueError(f"{label} validation_num_images is invalid at step {step}")
        if int(row.get("validation_noise_seed", -1)) < 0:
            raise ValueError(f"{label} validation_noise_seed is invalid at step {step}")
    endpoint = rows[-1]
    return {
        "row_count": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "samples_seen": int(endpoint["samples_seen"]),
        "all_numeric_metrics_finite": True,
        "steps_exact": True,
        "metrics_resume": metrics_resume,
        "metrics_resume_retained_rows_verified": True,
        "resume_boundary_steps": sorted(observed_set - base),
        "resume_boundary_evidence": [
            {
                "resume_step": step - 1,
                "first_post_resume_step": step,
                "metrics_reconciliation": next(
                    (
                        event
                        for event in metrics_resume.get("events", [])
                        if int(event["first_post_resume_step"]) == step
                    ),
                    None,
                ),
                "protected_checkpoint_schedule": (step - 1)
                in protected_checkpoint_steps,
            }
            for step in sorted(observed_set - base)
        ],
        "samples_seen_exact": True,
        "validation_event_count": len(validation_rows),
        "validation_steps": expected_validation_steps,
        "validation_provenance_complete": True,
        "endpoint": {
            "epsilon": float(endpoint["epsilon"]),
            "rollout_consistency": float(endpoint["rollout_consistency"]),
            "rollout_consistency_scale": float(endpoint["rollout_consistency_scale"]),
        },
        "validation_rows": validation_rows,
    }


def _schedule_contract(config: dict[str, Any], *, label: str) -> dict[str, Any]:
    loss = config.get("loss")
    if not isinstance(loss, dict):
        raise ValueError(f"{label} config lacks its loss schedule")
    result: dict[str, Any] = {}
    for name in _SHARED_SCHEDULES:
        weight = _finite_number(
            loss.get(f"{name}_weight"),
            label=f"{label} {name}_weight",
        )
        if weight < 0.0:
            raise ValueError(f"{label} {name}_weight must be nonnegative")
        try:
            start_step = int(loss[f"{name}_start_step"])
            warmup_steps = int(loss[f"{name}_warmup_steps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{label} {name} schedule is incomplete") from error
        if start_step < 0 or warmup_steps < 0:
            raise ValueError(f"{label} {name} schedule steps must be nonnegative")
        enabled = weight > 0.0
        result[name] = {
            "enabled": enabled,
            "weight": weight,
            "start_step": start_step,
            "warmup_steps": warmup_steps,
            "full_scale_step": start_step + warmup_steps if enabled else None,
        }
    return result


def _schedule_phase(step: int, contract: dict[str, Any]) -> str:
    if contract["enabled"] is not True:
        return "disabled"
    start_step = int(contract["start_step"])
    warmup_steps = int(contract["warmup_steps"])
    if step < start_step:
        return "inactive"
    if warmup_steps > 0 and step < start_step + warmup_steps:
        return "warmup"
    return "full_scale"


def _validate_observed_schedule_scale(
    value: Any,
    *,
    label: str,
    phase: str,
) -> float:
    scale = _finite_number(value, label=label)
    if scale < 0.0 or scale > 1.0:
        raise ValueError(f"{label} must be between zero and one")
    if phase in {"disabled", "inactive"} and scale != 0.0:
        raise ValueError(f"{label} must be zero during the {phase} phase")
    if phase == "warmup" and scale >= 1.0:
        raise ValueError(f"{label} must remain below one during warmup")
    if phase == "full_scale" and not math.isclose(scale, 1.0, abs_tol=1e-8):
        raise ValueError(f"{label} must equal one during the full-scale phase")
    return scale


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot summarize an empty validation regime")
    cofitok_values = [event["cofitok_epsilon_mse"] for event in events]
    dense_values = [event["dense_epsilon_mse"] for event in events]
    deltas = [event["cofitok_minus_dense"] for event in events]
    relative_deltas = [
        event["relative_delta"]
        for event in events
        if event["relative_delta"] is not None
    ]
    cofitok_mean = mean(cofitok_values)
    dense_mean = mean(dense_values)
    return {
        "event_count": len(events),
        "cofitok_mean_epsilon_mse": cofitok_mean,
        "dense_mean_epsilon_mse": dense_mean,
        "mean_cofitok_minus_dense": mean(deltas),
        "relative_delta_of_means": (
            (cofitok_mean - dense_mean) / dense_mean if dense_mean != 0.0 else None
        ),
        "mean_absolute_delta": mean(abs(delta) for delta in deltas),
        "median_absolute_delta": median(abs(delta) for delta in deltas),
        "max_absolute_relative_delta": (
            max(abs(value) for value in relative_deltas) if relative_deltas else None
        ),
        "cofitok_lower_event_count": sum(delta < 0.0 for delta in deltas),
        "dense_lower_event_count": sum(delta > 0.0 for delta in deltas),
        "tie_event_count": sum(delta == 0.0 for delta in deltas),
        "endpoint_relative_delta": events[-1]["relative_delta"],
    }


def _schedule_regime_summaries(
    events: list[dict[str, Any]],
    *,
    contracts: dict[str, Any],
) -> dict[str, Any]:
    by_schedule: dict[str, Any] = {}
    for name, contract in contracts.items():
        phase_summaries = []
        for phase in _SCHEDULE_PHASES:
            selected = [
                event
                for event in events
                if event["schedule_phases"][name] == phase
            ]
            if not selected:
                continue
            phase_summaries.append(
                {
                    "phase": phase,
                    "first_step": selected[0]["step"],
                    "last_step": selected[-1]["step"],
                    "event_steps": [event["step"] for event in selected],
                    "summary": _event_summary(selected),
                }
            )
        by_schedule[name] = phase_summaries
    return {
        "basis": "predeclared shared loss schedules from both resolved manifests",
        "contracts": contracts,
        "by_schedule": by_schedule,
        "individual_regime_significance_claim_allowed": False,
    }


def _paired_validation(
    cofitok_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    *,
    schedule_contracts: dict[str, Any],
) -> dict[str, Any]:
    if len(cofitok_rows) != len(dense_rows):
        raise ValueError("matched validation event counts differ")
    events: list[dict[str, Any]] = []
    for cofitok, dense in zip(cofitok_rows, dense_rows, strict=True):
        step = int(cofitok["step"])
        metadata_fields = (
            "step",
            "validation_event_index",
            "validation_batch_index",
            "validation_num_images",
            "validation_noise_seed",
        )
        if any(cofitok.get(field) != dense.get(field) for field in metadata_fields):
            raise ValueError(f"validation provenance differs between methods at step {step}")
        cofitok_mse = float(cofitok["validation_epsilon_mse"])
        dense_mse = float(dense["validation_epsilon_mse"])
        delta = cofitok_mse - dense_mse
        relative = delta / dense_mse if dense_mse != 0.0 else None
        schedule_phases = {
            name: _schedule_phase(step, contract)
            for name, contract in schedule_contracts.items()
        }
        observed_schedule_scales: dict[str, float] = {}
        for name, phase in schedule_phases.items():
            field = f"{name}_scale"
            cofitok_scale = _validate_observed_schedule_scale(
                cofitok.get(field),
                label=f"cofitok step {step} {field}",
                phase=phase,
            )
            dense_scale = _validate_observed_schedule_scale(
                dense.get(field),
                label=f"dense_identity step {step} {field}",
                phase=phase,
            )
            if not math.isclose(cofitok_scale, dense_scale, abs_tol=1e-12):
                raise ValueError(f"matched {field} differs at step {step}")
            observed_schedule_scales[name] = cofitok_scale
        events.append(
            {
                "step": step,
                "validation_event_index": int(cofitok["validation_event_index"]),
                "validation_batch_index": int(cofitok["validation_batch_index"]),
                "validation_num_images": int(cofitok["validation_num_images"]),
                "validation_noise_seed": int(cofitok["validation_noise_seed"]),
                "cofitok_epsilon_mse": cofitok_mse,
                "dense_epsilon_mse": dense_mse,
                "cofitok_minus_dense": delta,
                "relative_delta": relative,
                "schedule_phases": schedule_phases,
                "observed_schedule_scales": observed_schedule_scales,
                "lower_mse": (
                    "cofitok" if delta < 0.0 else "dense_identity" if delta > 0.0 else "tie"
                ),
            }
        )
    return {
        "pairing_basis": (
            "same validation steps, event indices, batch indices, image counts, "
            "and fixed noise seed"
        ),
        "events": events,
        "summary": _event_summary(events),
        "schedule_regimes": _schedule_regime_summaries(
            events,
            contracts=schedule_contracts,
        ),
    }


def build_report(
    *,
    cofitok_metrics: dict[str, Any],
    dense_metrics: dict[str, Any],
    cofitok_manifest: dict[str, Any],
    dense_manifest: dict[str, Any],
    cutoff_step: int,
    expected_revision: str,
    expected_branch: str,
    cofitok_reconciliation_history: list[dict[str, Any]] | None = None,
    dense_reconciliation_history: list[dict[str, Any]] | None = None,
    cofitok_resume_history_evidence: dict[str, Any] | None = None,
    dense_resume_history_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _is_hex_digest(expected_revision, length=40):
        raise ValueError("expected_revision must be a full 40-character revision")
    validated_cofitok = _validate_manifest(
        cofitok_manifest,
        label="cofitok",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        reconciliation_history=cofitok_reconciliation_history,
        resume_history_evidence=cofitok_resume_history_evidence,
    )
    validated_dense = _validate_manifest(
        dense_manifest,
        label="dense_identity",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        reconciliation_history=dense_reconciliation_history,
        resume_history_evidence=dense_resume_history_evidence,
    )
    for field in ("dataset", "dataset_identity_sha256", "runtime_environment_sha256"):
        if validated_cofitok[field] != validated_dense[field]:
            raise ValueError(f"matched manifests differ in {field}")
    contract = generation_pair_contract(
        validated_cofitok["config"], validated_dense["config"]
    )
    if contract["valid"] is not True:
        raise ValueError("training pair contract failed: " + "; ".join(contract["issues"]))
    cofitok_config = validated_cofitok["config"]
    dense_config = validated_dense["config"]
    schedule_contracts = _schedule_contract(cofitok_config, label="cofitok")
    dense_schedule_contracts = _schedule_contract(
        dense_config,
        label="dense_identity",
    )
    if schedule_contracts != dense_schedule_contracts:
        raise ValueError("matched manifests differ in shared loss schedules")
    protected_checkpoint_steps = _protected_checkpoint_steps(
        cofitok_config,
        label="cofitok",
    )
    dense_protected_checkpoint_steps = _protected_checkpoint_steps(
        dense_config,
        label="dense_identity",
    )
    if protected_checkpoint_steps != dense_protected_checkpoint_steps:
        raise ValueError("matched manifests differ in protected checkpoint steps")
    optimization = cofitok_config["optimization"]
    data = cofitok_config["data"]
    runtime = cofitok_config["runtime"]
    effective_batch = int(data["batch_size"]) * int(
        optimization["gradient_accumulation_steps"]
    )
    log_interval = int(optimization["log_interval"])
    evaluation_interval = int(runtime["evaluation_interval"])
    if effective_batch < 1 or evaluation_interval < 1:
        raise ValueError("matched effective batch and evaluation interval must be positive")
    cofitok_trajectory = _validate_rows(
        cofitok_metrics["rows"],
        label="cofitok",
        cutoff_step=cutoff_step,
        log_interval=log_interval,
        evaluation_interval=evaluation_interval,
        effective_batch=effective_batch,
        metrics_resume=validated_cofitok["metrics_resume"],
        protected_checkpoint_steps=protected_checkpoint_steps,
    )
    dense_trajectory = _validate_rows(
        dense_metrics["rows"],
        label="dense_identity",
        cutoff_step=cutoff_step,
        log_interval=log_interval,
        evaluation_interval=evaluation_interval,
        effective_batch=effective_batch,
        metrics_resume=validated_dense["metrics_resume"],
        protected_checkpoint_steps=protected_checkpoint_steps,
    )
    paired = _paired_validation(
        cofitok_trajectory.pop("validation_rows"),
        dense_trajectory.pop("validation_rows"),
        schedule_contracts=schedule_contracts,
    )
    dense_parameters = validated_dense["parameter_count"]
    parameter_gap = (
        validated_cofitok["parameter_count"] - dense_parameters
    ) / dense_parameters
    if abs(parameter_gap) > 0.02:
        raise ValueError("matched parameter gap exceeds 2%")
    complete_resume_history_verified = all(
        trajectory["metrics_resume"]["history_evidence"][
            "complete_recovery_chain_verified"
        ]
        for trajectory in (cofitok_trajectory, dense_trajectory)
    )
    return {
        "schema_version": 5,
        "status": "pass",
        "role": "matched_training_trajectory_diagnostic",
        "cutoff_step": cutoff_step,
        "images_seen_per_method": cutoff_step * effective_batch,
        "contract": {
            "git": validated_cofitok["git"],
            "dataset": validated_cofitok["dataset"],
            "dataset_identity_sha256": validated_cofitok["dataset_identity_sha256"],
            "runtime_environment_sha256": validated_cofitok[
                "runtime_environment_sha256"
            ],
            "effective_batch_size": effective_batch,
            "log_interval": log_interval,
            "evaluation_interval": evaluation_interval,
            "protected_checkpoint_steps": protected_checkpoint_steps,
            "cofitok_parameter_count": validated_cofitok["parameter_count"],
            "dense_parameter_count": dense_parameters,
            "relative_parameter_gap": parameter_gap,
            "generation_pair_contract": contract,
            "complete_recovery_chain_verified": complete_resume_history_verified,
        },
        "trajectories": {
            "cofitok": cofitok_trajectory,
            "dense_identity": dense_trajectory,
        },
        "paired_fixed_validation": paired,
        "comparison_policy": {
            "shared_primary_training_epsilon_reported_descriptively": True,
            "shared_rollout_consistency_reported_descriptively": True,
            "total_loss_comparison_allowed": False,
            "training_wall_clock_comparison_allowed": False,
            "reason": (
                "CoFiTok total loss includes factorization-only auxiliaries, and the "
                "dense run experienced externally observed GPU contention."
            ),
        },
        "claim_boundary": {
            "supports": [
                "matched resolved training contract through the cutoff",
                "source-bound fixed-validation epsilon trajectory through the cutoff",
                "predeclared fixed-validation summaries by shared schedule regime",
            ],
            "schedule_regime_quality_claim_allowed": False,
            "quality_claim_allowed": False,
            "formal_50k_gate_substitute": False,
            "promotion_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "sample_quality_metrics_present": False,
            "complete_recovery_chain_claim_allowed": (
                complete_resume_history_verified
            ),
            "resume_history_limitation": (
                None
                if complete_resume_history_verified
                else (
                    "At least one resumed run lacks a complete checkpoint-bound "
                    "append-only metrics resume history."
                )
            ),
            "required_next_evidence": (
                "exact healthy matched target completion followed by terminal EMA post-eval"
            ),
        },
    }


def main() -> None:
    args = _parse_args()
    cofitok_metrics = _read_metrics_prefix(
        args.cofitok_metrics, cutoff_step=args.cutoff_step
    )
    dense_metrics = _read_metrics_prefix(
        args.dense_metrics, cutoff_step=args.cutoff_step
    )
    if args.snapshot_dir is not None:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_paths = {
            "cofitok": args.snapshot_dir
            / f"cofitok_train_metrics_through_step_{args.cutoff_step:08d}.jsonl",
            "dense": args.snapshot_dir
            / f"dense_train_metrics_through_step_{args.cutoff_step:08d}.jsonl",
        }
        snapshot_paths["cofitok"].write_bytes(cofitok_metrics["raw_prefix"])
        snapshot_paths["dense"].write_bytes(dense_metrics["raw_prefix"])
        cofitok_metrics = _read_metrics_prefix(
            snapshot_paths["cofitok"], cutoff_step=args.cutoff_step
        )
        dense_metrics = _read_metrics_prefix(
            snapshot_paths["dense"], cutoff_step=args.cutoff_step
        )
    cofitok_manifest = _read_object(args.cofitok_manifest)
    dense_manifest = _read_object(args.dense_manifest)
    (
        cofitok_reconciliation_history,
        cofitok_reconciliation_sources,
        cofitok_resume_history_evidence,
    ) = _select_reconciliation_history(
        manifest_path=args.cofitok_manifest,
        manifest=cofitok_manifest,
        metrics_path=args.cofitok_metrics,
        manual_paths=args.cofitok_reconciliation,
        label="cofitok",
    )
    (
        dense_reconciliation_history,
        dense_reconciliation_sources,
        dense_resume_history_evidence,
    ) = _select_reconciliation_history(
        manifest_path=args.dense_manifest,
        manifest=dense_manifest,
        metrics_path=args.dense_metrics,
        manual_paths=args.dense_reconciliation,
        label="dense_identity",
    )
    report = build_report(
        cofitok_metrics=cofitok_metrics,
        dense_metrics=dense_metrics,
        cofitok_manifest=cofitok_manifest,
        dense_manifest=dense_manifest,
        cutoff_step=args.cutoff_step,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        cofitok_reconciliation_history=cofitok_reconciliation_history,
        dense_reconciliation_history=dense_reconciliation_history,
        cofitok_resume_history_evidence=cofitok_resume_history_evidence,
        dense_resume_history_evidence=dense_resume_history_evidence,
    )
    builder_path = Path(__file__).resolve()
    report["builder"] = _builder_identity(builder_path)
    report["sources"] = {
        "cofitok_metrics": {
            "origin": args.cofitok_origin,
            "bound_prefix": cofitok_metrics["identity"],
            "observed_file": cofitok_metrics["observed_file"],
        },
        "dense_metrics": {
            "origin": args.dense_origin,
            "bound_prefix": dense_metrics["identity"],
            "observed_file": dense_metrics["observed_file"],
        },
        "cofitok_manifest": _source(args.cofitok_manifest),
        "dense_manifest": _source(args.dense_manifest),
        "cofitok_reconciliations": cofitok_reconciliation_sources,
        "dense_reconciliations": dense_reconciliation_sources,
        "cofitok_metrics_resume_history": cofitok_resume_history_evidence,
        "dense_metrics_resume_history": dense_resume_history_evidence,
    }
    write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "cutoff_step": report["cutoff_step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
