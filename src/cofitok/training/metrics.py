from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report, write_text_report


def ensure_fresh_training_output(directory: str | Path) -> None:
    root = Path(directory)
    candidates = [
        root / "run_manifest.json",
        root / "training_report.json",
        root / "latest.json",
        root / "train_metrics.jsonl",
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


def reconcile_metrics_for_resume(
    path: str | Path,
    *,
    resume_step: int,
) -> dict[str, Any]:
    """Make a metrics JSONL canonical for an exact checkpoint resume.

    Rows after the checkpoint and older duplicate rows are preserved in a
    content-addressed orphan archive. The active JSONL keeps the latest row for
    each step at or before the checkpoint, ordered strictly by step.
    """
    if resume_step < 0:
        raise ValueError("resume_step must be non-negative")
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return {
            "schema_version": 1,
            "status": "absent",
            "resume_step": resume_step,
            "metrics": metrics_path.resolve().as_posix(),
            "retained_rows": 0,
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        }

    raw_lines = metrics_path.read_text(encoding="utf-8").splitlines(keepends=True)
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
        index
        for step, index in latest_index_by_step.items()
        if step <= resume_step
    }
    retained = sorted(
        (rows[index] for index in retained_indices),
        key=lambda item: item[0],
    )
    orphaned = [
        line
        for index, (_, line) in enumerate(rows)
        if index not in retained_indices
    ]
    canonical_content = "".join(line for _, line in retained)
    current_content = "".join(_normalized_line(line) for line in raw_lines if line.strip())
    if not orphaned and canonical_content == current_content:
        return {
            "schema_version": 1,
            "status": "unchanged",
            "resume_step": resume_step,
            "metrics": metrics_path.resolve().as_posix(),
            "retained_rows": len(retained),
            "orphaned_rows": 0,
            "orphan_archive": None,
            "orphan_sha256": None,
        }

    orphan_content = "".join(orphaned)
    orphan_sha256 = hashlib.sha256(orphan_content.encode("utf-8")).hexdigest()
    archive = metrics_path.with_name(
        f"{metrics_path.stem}_orphaned_at_resume_{resume_step:08d}_"
        f"{orphan_sha256[:12]}{metrics_path.suffix}"
    )
    if archive.is_file():
        existing_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        if existing_sha256 != orphan_sha256:
            raise ValueError("existing orphan archive does not match its content address")
    else:
        write_text_report(archive, orphan_content)
    write_text_report(metrics_path, canonical_content)
    report = {
        "schema_version": 1,
        "status": "reconciled",
        "resume_step": resume_step,
        "metrics": metrics_path.resolve().as_posix(),
        "retained_rows": len(retained),
        "orphaned_rows": len(orphaned),
        "orphan_archive": archive.resolve().as_posix(),
        "orphan_sha256": orphan_sha256,
    }
    report_path = metrics_path.with_name(
        f"metrics_resume_reconciliation_{resume_step:08d}_{orphan_sha256[:12]}.json"
    )
    write_json_report(report_path, report)
    return {**report, "report": report_path.resolve().as_posix()}
