from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.reporting import file_sha256, write_json_report, write_text_report


# Keep reconciliation schema v1 for existing trajectory/audit consumers. The
# additional physical identities are backward-compatible fields.
METRICS_RECONCILIATION_SCHEMA_VERSION = 1
METRICS_RESUME_HISTORY_SCHEMA_VERSION = 1
METRICS_RESUME_EVENT_SCHEMA_VERSION = 1
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _relative_artifact_path(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"resume artifact is outside the training output: {resolved}") from error
    return relative.as_posix()


def _resolve_artifact_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("resume artifact path must be a non-empty relative path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"resume artifact path is unsafe: {relative_path}")
    resolved = root.joinpath(*pure.parts).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"resume artifact escapes the training output: {relative_path}") from error
    return resolved


def _artifact_identity(path: str | Path, root: Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"resume artifact is missing: {artifact}")
    return {
        "path": _relative_artifact_path(artifact, root),
        "bytes": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }


def _validate_artifact_identity(
    identity: Any,
    *,
    root: Path,
    label: str,
    allow_missing: bool = False,
) -> Path:
    if not isinstance(identity, Mapping) or set(identity) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} identity is malformed")
    expected_bytes = identity.get("bytes")
    expected_sha256 = identity.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes < 0 or not _is_sha256(
        expected_sha256
    ):
        raise ValueError(f"{label} identity is malformed")
    path = _resolve_artifact_path(root, identity.get("path"))
    if not path.is_file():
        if allow_missing:
            return path
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte size differs from resume history")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from resume history")
    return path


def empty_metrics_resume_history() -> dict[str, Any]:
    legacy_reconciliations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    return {
        "schema_version": METRICS_RESUME_HISTORY_SCHEMA_VERSION,
        "legacy_history_complete": True,
        "legacy_reconciliations": legacy_reconciliations,
        "events": events,
        "history_sha256": _canonical_json_sha256(
            {
                "legacy_history_complete": True,
                "legacy_reconciliations": legacy_reconciliations,
                "events": events,
            }
        ),
    }


def _legacy_reconciliation_identity(
    reconciliation: Any,
    *,
    root: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(reconciliation, Mapping):
        raise ValueError("legacy metrics reconciliation is not an object")
    status = reconciliation.get("status")
    if status not in {"absent", "unchanged", "reconciled"}:
        raise ValueError("legacy metrics reconciliation status is invalid")
    resume_step = int(reconciliation.get("resume_step", -1))
    retained_rows = int(reconciliation.get("retained_rows", -1))
    orphaned_rows = int(reconciliation.get("orphaned_rows", -1))
    if resume_step < 0 or retained_rows < 0 or orphaned_rows < 0:
        raise ValueError("legacy metrics reconciliation counts are invalid")
    metrics_path = Path(str(reconciliation.get("metrics", ""))).resolve()
    metrics_relative = _relative_artifact_path(metrics_path, root)
    orphan_path = reconciliation.get("orphan_archive")
    orphan_sha256 = reconciliation.get("orphan_sha256")
    orphan_identity = None
    if orphaned_rows > 0:
        if not isinstance(orphan_path, str) or not _is_sha256(orphan_sha256):
            raise ValueError("legacy metrics reconciliation lacks its orphan identity")
        orphan_identity = _artifact_identity(orphan_path, root)
        if orphan_identity["sha256"] != orphan_sha256:
            raise ValueError("legacy metrics reconciliation orphan SHA256 differs")
    elif orphan_path is not None or orphan_sha256 is not None:
        raise ValueError("legacy zero-orphan reconciliation binds an unexpected archive")

    declared_report = reconciliation.get("report")
    if report_path is None and isinstance(declared_report, str):
        report_path = Path(declared_report)
    report_identity = _artifact_identity(report_path, root) if report_path is not None else None
    core = {
        "status": status,
        "resume_step": resume_step,
        "metrics": metrics_relative,
        "retained_rows": retained_rows,
        "orphaned_rows": orphaned_rows,
        "orphan_sha256": orphan_sha256,
    }
    return {
        "schema_version": 1,
        "resume_step": resume_step,
        "status": status,
        "reconciliation_sha256": _canonical_json_sha256(core),
        "report": report_identity,
        "orphan_archive": orphan_identity,
        "checkpoint_binding_available": False,
    }


def _merge_legacy_reconciliations(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entry in entries:
        identity = str(entry["reconciliation_sha256"])
        previous = merged.get(identity)
        if previous is None:
            merged[identity] = entry
            continue
        for artifact_key in ("report", "orphan_archive"):
            previous_artifact = previous[artifact_key]
            current_artifact = entry[artifact_key]
            if previous_artifact is None:
                previous[artifact_key] = current_artifact
            elif current_artifact is not None and current_artifact != previous_artifact:
                raise ValueError(
                    f"legacy metrics reconciliation {artifact_key} identities diverge"
                )
    return sorted(
        merged.values(),
        key=lambda entry: (entry["resume_step"], entry["reconciliation_sha256"]),
    )


def ensure_fresh_training_output(directory: str | Path) -> None:
    root = Path(directory)
    candidates = [
        root / "run_manifest.json",
        root / "training_report.json",
        root / "latest.json",
        root / "train_metrics.jsonl",
        root / "metrics_resume_history.json",
    ]
    candidates.extend(root.glob("checkpoint_step_*"))
    candidates.extend(root.glob("train_metrics_orphaned_*"))
    candidates.extend(root.glob("metrics_resume_reconciliation_*"))
    existing = sorted({path.resolve() for path in candidates if path.exists()})
    if existing:
        preview = ", ".join(path.name for path in existing[:3])
        raise FileExistsError(
            f"Refusing to start a fresh run in {root}: existing training state "
            f"includes {preview}; pass --resume auto or choose a new output directory"
        )


def _normalized_line(line: str) -> str:
    return line if line.endswith("\n") else f"{line}\n"


def _persist_reconciliation_report(
    metrics_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    report_payload_sha256 = _canonical_json_sha256(report)
    report_path = metrics_path.with_name(
        f"metrics_resume_reconciliation_{int(report['resume_step']):08d}_"
        f"{report_payload_sha256[:12]}.json"
    )
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != report:
            raise ValueError("existing metrics reconciliation report conflicts with its address")
    else:
        write_json_report(report_path, report)
    return {
        **report,
        "report": report_path.resolve().as_posix(),
        "report_bytes": report_path.stat().st_size,
        "report_sha256": file_sha256(report_path),
    }


def reconcile_metrics_for_resume(
    path: str | Path,
    *,
    resume_step: int,
) -> dict[str, Any]:
    """Make a metrics JSONL canonical for an exact checkpoint resume.

    Rows after the checkpoint and older duplicate rows are preserved in a
    content-addressed orphan archive. The active JSONL keeps the latest row for
    each step at or before the checkpoint, ordered strictly by step. Every
    outcome, including ``absent`` and ``unchanged``, receives an immutable
    reconciliation report so a later resume can replay the complete history.
    """
    if resume_step < 0:
        raise ValueError("resume_step must be non-negative")
    metrics_path = Path(path)
    resolved_metrics = metrics_path.resolve().as_posix()
    if not metrics_path.is_file():
        report = {
            "schema_version": METRICS_RECONCILIATION_SCHEMA_VERSION,
            "status": "absent",
            "resume_step": resume_step,
            "metrics": resolved_metrics,
            "source_metrics_bytes": 0,
            "source_metrics_sha256": _EMPTY_SHA256,
            "metrics_bytes": 0,
            "metrics_sha256": _EMPTY_SHA256,
            "retained_rows": 0,
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        }
        return _persist_reconciliation_report(metrics_path, report)

    source_bytes = metrics_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("metrics JSONL is not valid UTF-8") from error
    raw_lines = source_text.splitlines(keepends=True)
    rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics JSON at line {line_number}") from error
        step = int(payload.get("step", -1))
        if step < 0:
            raise ValueError(f"metrics line {line_number} has an invalid step")
        rows.append((step, _normalized_line(line)))

    latest_index_by_step = {step: index for index, (step, _) in enumerate(rows)}
    retained_indices = {
        index for step, index in latest_index_by_step.items() if step <= resume_step
    }
    retained = sorted(
        (rows[index] for index in retained_indices),
        key=lambda item: item[0],
    )
    orphaned = [
        line for index, (_, line) in enumerate(rows) if index not in retained_indices
    ]
    canonical_content = "".join(line for _, line in retained)
    canonical_bytes = canonical_content.encode("utf-8")
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    current_content = "".join(
        _normalized_line(line) for line in raw_lines if line.strip()
    )

    orphan_archive: str | None = None
    orphan_sha256: str | None = None
    if not orphaned:
        status = "unchanged"
        if canonical_content != current_content:
            write_text_report(metrics_path, canonical_content)
    else:
        status = "reconciled"
        orphan_content = "".join(orphaned)
        orphan_bytes = orphan_content.encode("utf-8")
        orphan_sha256 = hashlib.sha256(orphan_bytes).hexdigest()
        archive = metrics_path.with_name(
            f"{metrics_path.stem}_orphaned_at_resume_{resume_step:08d}_"
            f"{orphan_sha256[:12]}{metrics_path.suffix}"
        )
        if archive.is_file():
            if archive.read_bytes() != orphan_bytes:
                raise ValueError(
                    "existing orphan archive does not match its content address"
                )
        else:
            write_text_report(archive, orphan_content)
        orphan_archive = archive.resolve().as_posix()
        write_text_report(metrics_path, canonical_content)

    if metrics_path.stat().st_size != len(canonical_bytes):
        raise ValueError("canonical metrics byte size differs after reconciliation")
    if file_sha256(metrics_path) != canonical_sha256:
        raise ValueError("canonical metrics SHA256 differs after reconciliation")
    report = {
        "schema_version": METRICS_RECONCILIATION_SCHEMA_VERSION,
        "status": status,
        "resume_step": resume_step,
        "metrics": resolved_metrics,
        "source_metrics_bytes": len(source_bytes),
        "source_metrics_sha256": source_sha256,
        "metrics_bytes": len(canonical_bytes),
        "metrics_sha256": canonical_sha256,
        "retained_rows": len(retained),
        "orphaned_rows": len(orphaned),
        "orphan_archive": orphan_archive,
        "orphan_sha256": orphan_sha256,
    }
    return _persist_reconciliation_report(metrics_path, report)


def _validate_reconciliation_binding(
    reconciliation: Any,
    *,
    event: Mapping[str, Any],
    root: Path,
    label: str,
) -> None:
    if not isinstance(reconciliation, Mapping):
        raise ValueError(f"{label} lacks its current metrics reconciliation")
    bound = event["reconciliation"]
    report_path = _resolve_artifact_path(root, bound["report"]["path"])
    metrics_path = _resolve_artifact_path(root, bound["metrics"]["path"])
    expected = {
        "status": bound["status"],
        "resume_step": event["resume_step"],
        "report": report_path.as_posix(),
        "report_bytes": bound["report"]["bytes"],
        "report_sha256": bound["report"]["sha256"],
        "metrics": metrics_path.as_posix(),
        "metrics_bytes": bound["metrics"]["prefix_bytes"],
        "metrics_sha256": bound["metrics"]["prefix_sha256"],
        "retained_rows": bound["metrics"]["retained_rows"],
        "orphaned_rows": bound["orphaned_rows"],
    }
    for key, value in expected.items():
        actual = reconciliation.get(key)
        if key in {"report", "metrics"} and isinstance(actual, str):
            actual = Path(actual).resolve().as_posix()
        if actual != value:
            raise ValueError(f"{label} metrics reconciliation differs at {key}")
    orphan_identity = bound["orphan_archive"]
    if orphan_identity is None:
        if reconciliation.get("orphan_archive") is not None or reconciliation.get(
            "orphan_sha256"
        ) is not None:
            raise ValueError(f"{label} metrics orphan binding differs")
    else:
        orphan_path = _resolve_artifact_path(root, orphan_identity["path"])
        actual_path = reconciliation.get("orphan_archive")
        if not isinstance(actual_path, str) or Path(actual_path).resolve() != orphan_path:
            raise ValueError(f"{label} metrics orphan path differs")
        if reconciliation.get("orphan_sha256") != orphan_identity["sha256"]:
            raise ValueError(f"{label} metrics orphan SHA256 differs")


def validate_metrics_resume_history(
    history: Any,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if not isinstance(history, Mapping) or set(history) != {
        "schema_version",
        "legacy_history_complete",
        "legacy_reconciliations",
        "events",
        "history_sha256",
    }:
        raise ValueError("metrics resume history is malformed")
    if history.get("schema_version") != METRICS_RESUME_HISTORY_SCHEMA_VERSION:
        raise ValueError("unsupported metrics resume history schema")
    if not isinstance(history.get("legacy_history_complete"), bool):
        raise ValueError("metrics resume legacy-completeness flag is malformed")
    legacy_reconciliations = history.get("legacy_reconciliations")
    events = history.get("events")
    if not isinstance(legacy_reconciliations, list):
        raise ValueError("legacy metrics reconciliations must be a list")
    if not isinstance(events, list):
        raise ValueError("metrics resume history events must be a list")
    expected_history_sha256 = _canonical_json_sha256(
        {
            "legacy_history_complete": history["legacy_history_complete"],
            "legacy_reconciliations": legacy_reconciliations,
            "events": events,
        }
    )
    if history.get("history_sha256") != expected_history_sha256:
        raise ValueError("metrics resume history digest is inconsistent")

    previous_legacy_key: tuple[int, str] | None = None
    seen_legacy_identities: set[str] = set()
    for legacy in legacy_reconciliations:
        if not isinstance(legacy, Mapping) or set(legacy) != {
            "schema_version",
            "resume_step",
            "status",
            "reconciliation_sha256",
            "report",
            "orphan_archive",
            "checkpoint_binding_available",
        }:
            raise ValueError("legacy metrics reconciliation identity is malformed")
        resume_step = legacy.get("resume_step")
        reconciliation_sha256 = legacy.get("reconciliation_sha256")
        if (
            legacy.get("schema_version") != 1
            or not isinstance(resume_step, int)
            or resume_step < 0
            or legacy.get("status") not in {"absent", "unchanged", "reconciled"}
            or not _is_sha256(reconciliation_sha256)
            or legacy.get("checkpoint_binding_available") is not False
        ):
            raise ValueError("legacy metrics reconciliation identity is malformed")
        legacy_key = (resume_step, reconciliation_sha256)
        if previous_legacy_key is not None and legacy_key <= previous_legacy_key:
            raise ValueError("legacy metrics reconciliations are not strictly ordered")
        if reconciliation_sha256 in seen_legacy_identities:
            raise ValueError("legacy metrics reconciliation identity is duplicated")
        previous_legacy_key = legacy_key
        seen_legacy_identities.add(reconciliation_sha256)
        report_identity = legacy.get("report")
        if report_identity is not None:
            _validate_artifact_identity(
                report_identity,
                root=root,
                label="legacy metrics reconciliation report",
            )
        orphan_identity = legacy.get("orphan_archive")
        if orphan_identity is not None:
            _validate_artifact_identity(
                orphan_identity,
                root=root,
                label="legacy metrics orphan archive",
            )

    previous_step = -1
    latest_legacy_step = max(
        (int(entry["resume_step"]) for entry in legacy_reconciliations),
        default=-1,
    )
    for expected_index, event in enumerate(events, start=1):
        if not isinstance(event, Mapping) or set(event) != {
            "schema_version",
            "event_index",
            "resume_step",
            "checkpoint",
            "reconciliation",
        }:
            raise ValueError("metrics resume history event is malformed")
        if event.get("schema_version") != METRICS_RESUME_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported metrics resume event schema")
        if event.get("event_index") != expected_index:
            raise ValueError("metrics resume history event indices are not contiguous")
        resume_step = event.get("resume_step")
        if (
            not isinstance(resume_step, int)
            or resume_step <= previous_step
            or resume_step < latest_legacy_step
        ):
            raise ValueError("metrics resume history steps are not strictly increasing")
        previous_step = resume_step

        checkpoint = event.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
            "path",
            "bytes",
            "sha256",
            "integrity_manifest",
        }:
            raise ValueError("metrics resume checkpoint identity is malformed")
        if (
            not isinstance(checkpoint.get("bytes"), int)
            or checkpoint["bytes"] < 1
            or not _is_sha256(checkpoint.get("sha256"))
        ):
            raise ValueError("metrics resume checkpoint identity is malformed")
        checkpoint_path = _resolve_artifact_path(root, checkpoint.get("path"))
        integrity_path = _validate_artifact_identity(
            checkpoint.get("integrity_manifest"),
            root=root,
            label="metrics resume checkpoint integrity manifest",
            allow_missing=not checkpoint_path.is_file(),
        )
        if checkpoint_path.is_file():
            if checkpoint_path.stat().st_size != checkpoint["bytes"]:
                raise ValueError("metrics resume checkpoint byte size differs")
            if not integrity_path.is_file():
                raise FileNotFoundError(
                    "metrics resume checkpoint exists without its integrity manifest"
                )
            with integrity_path.open("r", encoding="utf-8") as handle:
                integrity = json.load(handle)
            if (
                integrity.get("checkpoint") != checkpoint_path.name
                or int(integrity.get("checkpoint_bytes", -1)) != checkpoint["bytes"]
                or integrity.get("checkpoint_sha256") != checkpoint["sha256"]
                or int(integrity.get("step", -1)) != resume_step
            ):
                raise ValueError(
                    "metrics resume checkpoint differs from its integrity manifest"
                )

        reconciliation = event.get("reconciliation")
        if not isinstance(reconciliation, Mapping) or set(reconciliation) != {
            "status",
            "report",
            "metrics",
            "orphaned_rows",
            "orphan_archive",
        }:
            raise ValueError("metrics resume reconciliation binding is malformed")
        if reconciliation.get("status") not in {"absent", "unchanged", "reconciled"}:
            raise ValueError("metrics resume reconciliation status is invalid")
        report_path = _validate_artifact_identity(
            reconciliation.get("report"),
            root=root,
            label="metrics reconciliation report",
        )
        metrics = reconciliation.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "path",
            "prefix_bytes",
            "prefix_sha256",
            "retained_rows",
        }:
            raise ValueError("metrics resume prefix binding is malformed")
        prefix_bytes = metrics.get("prefix_bytes")
        retained_rows = metrics.get("retained_rows")
        if (
            not isinstance(prefix_bytes, int)
            or prefix_bytes < 0
            or not isinstance(retained_rows, int)
            or retained_rows < 0
            or not _is_sha256(metrics.get("prefix_sha256"))
        ):
            raise ValueError("metrics resume prefix binding is malformed")
        metrics_path = _resolve_artifact_path(root, metrics.get("path"))
        if prefix_bytes > 0 and not metrics_path.is_file():
            raise FileNotFoundError("metrics resume prefix source is missing")
        if metrics_path.is_file():
            with metrics_path.open("rb") as handle:
                prefix = handle.read(prefix_bytes)
            if len(prefix) != prefix_bytes:
                raise ValueError("metrics file is shorter than a recorded resume prefix")
            if hashlib.sha256(prefix).hexdigest() != metrics["prefix_sha256"]:
                raise ValueError("metrics resume prefix SHA256 differs")

        orphaned_rows = reconciliation.get("orphaned_rows")
        if not isinstance(orphaned_rows, int) or orphaned_rows < 0:
            raise ValueError("metrics resume orphan row count is invalid")
        orphan_identity = reconciliation.get("orphan_archive")
        if orphaned_rows == 0:
            if orphan_identity is not None:
                raise ValueError("zero-orphan reconciliation unexpectedly binds an archive")
        else:
            _validate_artifact_identity(
                orphan_identity,
                root=root,
                label="metrics orphan archive",
            )

        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if (
            int(report.get("resume_step", -1)) != resume_step
            or report.get("status") != reconciliation["status"]
            or int(report.get("metrics_bytes", -1)) != prefix_bytes
            or report.get("metrics_sha256") != metrics["prefix_sha256"]
            or int(report.get("retained_rows", -1)) != retained_rows
            or int(report.get("orphaned_rows", -1)) != orphaned_rows
        ):
            raise ValueError("metrics reconciliation report differs from resume history")
        expected_metrics_path = metrics_path.as_posix()
        if Path(str(report.get("metrics", ""))).resolve().as_posix() != expected_metrics_path:
            raise ValueError("metrics reconciliation report names another metrics file")
        if orphan_identity is None:
            if report.get("orphan_archive") is not None or report.get(
                "orphan_sha256"
            ) is not None:
                raise ValueError("metrics reconciliation report has an unexpected orphan")
        else:
            orphan_path = _resolve_artifact_path(root, orphan_identity["path"])
            if (
                Path(str(report.get("orphan_archive", ""))).resolve() != orphan_path
                or report.get("orphan_sha256") != orphan_identity["sha256"]
            ):
                raise ValueError("metrics reconciliation orphan binding differs")

    return json.loads(json.dumps(history, sort_keys=True))


def restore_metrics_resume_history(
    output_dir: str | Path,
    *,
    checkpoint_extra_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Restore the longest mutually compatible history from durable sources."""
    root = Path(output_dir).resolve()
    candidates: list[tuple[str, dict[str, Any]]] = []
    legacy_reconciliations: list[dict[str, Any]] = []

    def add_candidate(
        label: str,
        raw_history: Any,
        current_reconciliation: Any,
        *,
        bind_current_reconciliation: bool = True,
    ) -> None:
        if raw_history is None:
            return
        history = validate_metrics_resume_history(raw_history, output_dir=root)
        if history["events"] and bind_current_reconciliation:
            _validate_reconciliation_binding(
                current_reconciliation,
                event=history["events"][-1],
                root=root,
                label=label,
            )
        elif not history["events"] and current_reconciliation is not None:
            raise ValueError(f"{label} has a reconciliation but an empty resume history")
        candidates.append((label, history))

    checkpoint_state = dict(checkpoint_extra_state or {})
    if "metrics_resume_history" in checkpoint_state:
        add_candidate(
            "checkpoint",
            checkpoint_state.get("metrics_resume_history"),
            checkpoint_state.get("metrics_resume_reconciliation"),
        )
    elif checkpoint_state.get("metrics_resume_reconciliation") is not None:
        legacy_reconciliations.append(
            _legacy_reconciliation_identity(
                checkpoint_state["metrics_resume_reconciliation"],
                root=root,
            )
        )

    journal_path = root / "metrics_resume_history.json"
    if journal_path.is_file():
        with journal_path.open("r", encoding="utf-8") as handle:
            add_candidate(
                "history journal",
                json.load(handle),
                None,
                bind_current_reconciliation=False,
            )

    for filename, label in (
        ("run_manifest.json", "run manifest"),
        ("training_report.json", "training report"),
    ):
        path = root / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if "metrics_resume_history" in document:
            add_candidate(
                label,
                document.get("metrics_resume_history"),
                document.get("metrics_resume_reconciliation"),
            )
        elif document.get("metrics_resume_reconciliation") is not None:
            legacy_reconciliations.append(
                _legacy_reconciliation_identity(
                    document["metrics_resume_reconciliation"],
                    root=root,
                )
            )

    if not candidates:
        for report_path in sorted(root.glob("metrics_resume_reconciliation_*.json")):
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            legacy_reconciliations.append(
                _legacy_reconciliation_identity(
                    report,
                    root=root,
                    report_path=report_path,
                )
            )
        migrated = empty_metrics_resume_history()
        migrated["legacy_reconciliations"] = _merge_legacy_reconciliations(
            legacy_reconciliations
        )
        migrated["legacy_history_complete"] = not bool(
            migrated["legacy_reconciliations"]
        )
        migrated["history_sha256"] = _canonical_json_sha256(
            {
                "legacy_history_complete": migrated["legacy_history_complete"],
                "legacy_reconciliations": migrated["legacy_reconciliations"],
                "events": migrated["events"],
            }
        )
        return validate_metrics_resume_history(migrated, output_dir=root)
    longest_label, longest = max(candidates, key=lambda item: len(item[1]["events"]))
    longest_events = longest["events"]
    for label, candidate in candidates:
        if candidate["legacy_reconciliations"] != longest["legacy_reconciliations"]:
            raise ValueError(
                f"legacy metrics resume history sources diverge: {label} vs {longest_label}"
            )
        if candidate["legacy_history_complete"] != longest["legacy_history_complete"]:
            raise ValueError(
                f"metrics resume legacy-completeness sources diverge: {label} vs {longest_label}"
            )
        candidate_events = candidate["events"]
        if candidate_events != longest_events[: len(candidate_events)]:
            raise ValueError(
                f"metrics resume history sources diverge: {label} vs {longest_label}"
            )
    stored_legacy = {
        entry["reconciliation_sha256"]: entry
        for entry in longest["legacy_reconciliations"]
    }
    for observed in _merge_legacy_reconciliations(legacy_reconciliations):
        stored = stored_legacy.get(observed["reconciliation_sha256"])
        if stored is None:
            raise ValueError(
                "legacy metrics reconciliation source is absent from resume history"
            )
        for artifact_key in ("report", "orphan_archive"):
            if (
                observed[artifact_key] is not None
                and stored[artifact_key] != observed[artifact_key]
            ):
                raise ValueError(
                    f"legacy metrics reconciliation {artifact_key} differs from history"
                )
    return longest


def append_metrics_resume_event(
    history: Mapping[str, Any],
    *,
    output_dir: str | Path,
    resume_checkpoint: str | Path,
    checkpoint_integrity: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    validated = validate_metrics_resume_history(history, output_dir=root)
    checkpoint_path = Path(resume_checkpoint).resolve()
    checkpoint_relative = _relative_artifact_path(checkpoint_path, root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing: {checkpoint_path}")
    resume_step = int(checkpoint_integrity.get("step", -1))
    checkpoint_bytes = int(checkpoint_integrity.get("checkpoint_bytes", -1))
    checkpoint_sha256 = checkpoint_integrity.get("checkpoint_sha256")
    if (
        resume_step < 0
        or checkpoint_integrity.get("checkpoint") != checkpoint_path.name
        or checkpoint_bytes != checkpoint_path.stat().st_size
        or not _is_sha256(checkpoint_sha256)
    ):
        raise ValueError("resume checkpoint integrity identity is malformed")
    integrity_path = checkpoint_path.with_name(f"{checkpoint_path.name}.integrity.json")
    integrity_identity = _artifact_identity(integrity_path, root)
    with integrity_path.open("r", encoding="utf-8") as handle:
        on_disk_integrity = json.load(handle)
    if dict(checkpoint_integrity) != on_disk_integrity:
        raise ValueError("resume checkpoint integrity manifest changed after verification")

    if int(reconciliation.get("resume_step", -1)) != resume_step:
        raise ValueError("metrics reconciliation step differs from the resume checkpoint")
    report_identity = _artifact_identity(str(reconciliation.get("report", "")), root)
    if (
        report_identity["bytes"] != reconciliation.get("report_bytes")
        or report_identity["sha256"] != reconciliation.get("report_sha256")
    ):
        raise ValueError("metrics reconciliation report identity is inconsistent")
    metrics_path = Path(str(reconciliation.get("metrics", ""))).resolve()
    metrics_relative = _relative_artifact_path(metrics_path, root)
    metrics_bytes = int(reconciliation.get("metrics_bytes", -1))
    metrics_sha256 = reconciliation.get("metrics_sha256")
    retained_rows = int(reconciliation.get("retained_rows", -1))
    orphaned_rows = int(reconciliation.get("orphaned_rows", -1))
    if (
        metrics_bytes < 0
        or retained_rows < 0
        or orphaned_rows < 0
        or not _is_sha256(metrics_sha256)
    ):
        raise ValueError("metrics reconciliation identity is malformed")
    if metrics_bytes > 0 and not metrics_path.is_file():
        raise FileNotFoundError("reconciled metrics file is missing")
    if metrics_path.is_file():
        with metrics_path.open("rb") as handle:
            prefix = handle.read(metrics_bytes)
        if len(prefix) != metrics_bytes or hashlib.sha256(prefix).hexdigest() != metrics_sha256:
            raise ValueError("reconciled metrics prefix identity is inconsistent")

    orphan_identity = None
    orphan_path = reconciliation.get("orphan_archive")
    if orphaned_rows > 0:
        if not isinstance(orphan_path, str):
            raise ValueError("reconciled metrics orphan archive is missing")
        orphan_identity = _artifact_identity(orphan_path, root)
        if orphan_identity["sha256"] != reconciliation.get("orphan_sha256"):
            raise ValueError("metrics orphan archive identity is inconsistent")
    elif orphan_path is not None or reconciliation.get("orphan_sha256") is not None:
        raise ValueError("zero-orphan reconciliation unexpectedly binds an archive")

    event = {
        "schema_version": METRICS_RESUME_EVENT_SCHEMA_VERSION,
        "event_index": len(validated["events"]) + 1,
        "resume_step": resume_step,
        "checkpoint": {
            "path": checkpoint_relative,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
            "integrity_manifest": integrity_identity,
        },
        "reconciliation": {
            "status": reconciliation.get("status"),
            "report": report_identity,
            "metrics": {
                "path": metrics_relative,
                "prefix_bytes": metrics_bytes,
                "prefix_sha256": metrics_sha256,
                "retained_rows": retained_rows,
            },
            "orphaned_rows": orphaned_rows,
            "orphan_archive": orphan_identity,
        },
    }
    events = list(validated["events"])
    if events and resume_step == events[-1]["resume_step"]:
        candidate = {**event, "event_index": events[-1]["event_index"]}
        if candidate != events[-1]:
            raise ValueError("repeated resume step differs from its recorded history event")
        return validated
    if events and resume_step < events[-1]["resume_step"]:
        raise ValueError("resume checkpoint predates the append-only metrics history")
    events.append(event)
    updated = {
        "schema_version": METRICS_RESUME_HISTORY_SCHEMA_VERSION,
        "legacy_history_complete": validated["legacy_history_complete"],
        "legacy_reconciliations": validated["legacy_reconciliations"],
        "events": events,
        "history_sha256": _canonical_json_sha256(
            {
                "legacy_history_complete": validated["legacy_history_complete"],
                "legacy_reconciliations": validated["legacy_reconciliations"],
                "events": events,
            }
        ),
    }
    return validate_metrics_resume_history(updated, output_dir=root)


def persist_metrics_resume_history(
    output_dir: str | Path,
    history: Mapping[str, Any],
) -> Path:
    root = Path(output_dir).resolve()
    validated = validate_metrics_resume_history(history, output_dir=root)
    path = root / "metrics_resume_history.json"
    write_json_report(path, validated)
    return path
