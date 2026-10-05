from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cofitok.path_security import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report


INFERENCE_MANIFEST_SCHEMA_VERSION = 1
INFERENCE_PROGRESS_SCHEMA_VERSION = 1
INFERENCE_REPORT_SCHEMA_VERSION = 2
INFERENCE_MANIFEST_ROLE = "generation_inference_manifest"
INFERENCE_PROGRESS_ROLE = "generation_inference_progress"


def read_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    import json

    source = reject_symlink_chain(path, name=name)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def file_identity(path: str | Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="inference evidence")
    if not source.is_file():
        raise FileNotFoundError(f"inference evidence file is missing: {source}")
    resolved = source.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def prepare_manifest(
    path: str | Path,
    expected: dict[str, Any],
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    target = reject_symlink_chain(path, name="inference manifest")
    if resume and overwrite:
        raise ValueError("inference --resume and --overwrite are mutually exclusive")
    if resume:
        if not target.is_file():
            raise FileNotFoundError(
                f"inference resume manifest is missing: {target}"
            )
        existing = read_json_object(target, name="inference manifest")
        if existing != expected:
            raise ValueError("inference resume manifest does not match the request")
    else:
        if target.exists() and not overwrite:
            raise FileExistsError(f"Inference manifest already exists: {target}")
        write_json_report(target, expected)
    return file_identity(target)


def _expected_index(
    expected_outputs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for output in expected_outputs:
        if not isinstance(output, dict):
            raise ValueError("inference manifest output entry is malformed")
        filename = str(output.get("filename", ""))
        if not filename or Path(filename).name != filename or filename in indexed:
            raise ValueError("inference manifest output filenames are invalid")
        indexed[filename] = dict(output)
    if not indexed:
        raise ValueError("inference manifest has no expected outputs")
    return indexed


def validate_output_directory_layout(
    root: str | Path,
    *,
    expected_outputs: list[dict[str, Any]],
) -> dict[str, Path]:
    root_path = reject_symlink_chain(root, name="inference output root")
    if not root_path.is_dir():
        raise FileNotFoundError(f"inference output root is missing: {root_path}")
    resolved_root = root_path.resolve()
    expected = _expected_index(expected_outputs)
    paths: dict[str, Path] = {}
    for filename, row in expected.items():
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("inference manifest output path is invalid")
        path = reject_symlink_chain(raw_path, name="inference output")
        if path.parent.resolve() != resolved_root or path.name != filename:
            raise ValueError("inference manifest output escapes its output root")
        paths[filename] = path

    unexpected = []
    for candidate in resolved_root.rglob("*"):
        if candidate.suffix.lower() != ".png":
            continue
        if candidate.is_symlink():
            raise ValueError(f"inference output PNG must not be a symlink: {candidate}")
        if not candidate.is_file() or candidate.parent != resolved_root:
            unexpected.append(candidate.as_posix())
        elif candidate.name not in expected:
            unexpected.append(candidate.as_posix())
    if unexpected:
        raise ValueError(
            "inference output root contains unexpected PNG files: "
            + ", ".join(sorted(unexpected))
        )
    return paths


def load_progress(
    path: str | Path,
    *,
    manifest_identity: dict[str, Any],
    expected_outputs: list[dict[str, Any]],
    resume: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    target = reject_symlink_chain(path, name="inference progress")
    expected = _expected_index(expected_outputs)
    if target.exists() and not target.is_file():
        raise ValueError(f"inference progress is not a regular file: {target}")
    if not target.is_file():
        if resume:
            return {
                "status": "not_started",
                "attempt_count": 0,
                "cumulative_elapsed_seconds": 0.0,
                "outputs": {},
            }
        return {
            "status": "not_started",
            "attempt_count": 0,
            "cumulative_elapsed_seconds": 0.0,
            "outputs": {},
        }
    if not resume and not overwrite:
        raise FileExistsError(f"Inference progress already exists: {target}")
    if not resume:
        return {
            "status": "not_started",
            "attempt_count": 0,
            "cumulative_elapsed_seconds": 0.0,
            "outputs": {},
        }
    progress = read_json_object(target, name="inference progress")
    if (
        progress.get("schema_version") != INFERENCE_PROGRESS_SCHEMA_VERSION
        or progress.get("role") != INFERENCE_PROGRESS_ROLE
        or progress.get("manifest") != manifest_identity
        or progress.get("status") not in {"running", "failed", "completed"}
        or int(progress.get("expected_output_count", -1)) != len(expected)
    ):
        raise ValueError("inference progress binding is invalid")
    rows = progress.get("outputs")
    if not isinstance(rows, list):
        raise ValueError("inference progress outputs are malformed")
    valid: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("inference progress output row is malformed")
        filename = str(row.get("filename", ""))
        if filename in seen or filename not in expected:
            raise ValueError("inference progress output set differs")
        seen.add(filename)
        expected_row = expected[filename]
        for field in ("filename", "path", "seed", "class_id", "prefix_budget"):
            if row.get(field) != expected_row.get(field):
                raise ValueError("inference progress output identity differs")
        sha256 = str(row.get("sha256", ""))
        if len(sha256) != 64:
            raise ValueError("inference progress output digest is malformed")
        output_path = reject_symlink_chain(row["path"], name="inference output")
        if output_path.exists() and not output_path.is_file():
            raise ValueError("inference output path is not a regular file")
        if output_path.is_file() and file_sha256(output_path) == sha256:
            valid[filename] = dict(row)
    completed_count = int(progress.get("completed_output_count", -1))
    if completed_count != len(rows):
        raise ValueError("inference progress completed count differs")
    elapsed = float(progress.get("cumulative_elapsed_seconds", -1.0))
    attempts = int(progress.get("attempt_count", -1))
    if not math.isfinite(elapsed) or elapsed < 0.0 or attempts < 1:
        raise ValueError("inference progress timing metadata is invalid")
    return {
        "status": progress["status"],
        "attempt_count": attempts,
        "cumulative_elapsed_seconds": elapsed,
        "outputs": valid,
    }


def write_progress(
    path: str | Path,
    *,
    manifest_identity: dict[str, Any],
    expected_output_count: int,
    outputs: list[dict[str, Any]],
    status: str,
    attempt_count: int,
    cumulative_elapsed_seconds: float,
    updated_at: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if status not in {"running", "failed", "completed"}:
        raise ValueError("inference progress status is invalid")
    if (
        attempt_count < 1
        or not math.isfinite(cumulative_elapsed_seconds)
        or cumulative_elapsed_seconds < 0.0
    ):
        raise ValueError("inference progress timing is invalid")
    payload = {
        "schema_version": INFERENCE_PROGRESS_SCHEMA_VERSION,
        "role": INFERENCE_PROGRESS_ROLE,
        "status": status,
        "manifest": manifest_identity,
        "attempt_count": attempt_count,
        "expected_output_count": expected_output_count,
        "completed_output_count": len(outputs),
        "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
        "outputs": outputs,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "updated_at": updated_at,
    }
    write_json_report(path, payload)
    return file_identity(path)


def validate_report_binding(
    report: dict[str, Any],
    *,
    manifest_identity: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if (
        report.get("schema_version") != INFERENCE_REPORT_SCHEMA_VERSION
        or report.get("status") not in {"running", "failed", "completed"}
        or report.get("manifest") != manifest_identity
        or report.get("checkpoint") != manifest.get("checkpoint")
        or report.get("request") != manifest.get("request")
        or report.get("git") != manifest.get("git")
        or report.get("runtime_environment_sha256")
        != manifest.get("runtime_environment_sha256")
    ):
        raise ValueError("existing inference report binding differs")


def reusable_completed_report(
    report: dict[str, Any],
    *,
    progress_path: str | Path,
    progress_status: str,
    expected_outputs: list[dict[str, Any]],
    valid_outputs: dict[str, dict[str, Any]],
) -> bool:
    if report.get("status") != "completed" or progress_status != "completed":
        return False
    if len(valid_outputs) != len(expected_outputs):
        return False
    ordered = [valid_outputs[row["filename"]] for row in expected_outputs]
    return (
        report.get("progress") == file_identity(progress_path)
        and int(report.get("output_count", -1)) == len(ordered)
        and report.get("outputs") == ordered
    )


def validate_completed_inference_evidence(
    report: dict[str, Any],
    *,
    expected_root: str | Path,
) -> dict[str, Any]:
    root_path = reject_symlink_chain(expected_root, name="inference output root")
    if not root_path.is_dir():
        raise FileNotFoundError(f"inference output root is missing: {root_path}")
    root = root_path.resolve()
    if (
        report.get("schema_version") != INFERENCE_REPORT_SCHEMA_VERSION
        or report.get("status") != "completed"
    ):
        raise ValueError("inference report is not a completed replayable report")
    identities = {}
    for name, filename in (
        ("manifest", "inference_manifest.json"),
        ("progress", "inference_progress.json"),
    ):
        claimed = report.get(name)
        if not isinstance(claimed, dict):
            raise ValueError(f"inference report {name} identity is missing")
        path = reject_symlink_chain(
            str(claimed.get("path", "")),
            name=f"inference {name}",
        )
        if (
            not path.is_file()
            or path.resolve().parent != root
            or path.name != filename
        ):
            raise ValueError(f"inference {name} path is not canonical")
        actual = file_identity(path)
        if claimed != actual:
            raise ValueError(f"inference {name} identity differs")
        identities[name] = actual

    manifest = read_json_object(
        identities["manifest"]["path"],
        name="inference manifest",
    )
    if (
        manifest.get("schema_version") != INFERENCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("role") != INFERENCE_MANIFEST_ROLE
        or manifest.get("output_root") != root.as_posix()
    ):
        raise ValueError("inference manifest contract differs")
    validate_report_binding(
        report,
        manifest_identity=identities["manifest"],
        manifest=manifest,
    )
    progress = read_json_object(
        identities["progress"]["path"],
        name="inference progress",
    )
    expected_outputs = manifest.get("expected_outputs")
    outputs = report.get("outputs")
    if not isinstance(expected_outputs, list) or not isinstance(outputs, list):
        raise ValueError("inference output evidence is malformed")
    expected_paths = validate_output_directory_layout(
        root,
        expected_outputs=expected_outputs,
    )
    try:
        progress_attempts = int(progress.get("attempt_count", -1))
        progress_elapsed = float(progress.get("cumulative_elapsed_seconds", -1.0))
        report_attempts = int(report.get("attempt_count", -1))
        report_elapsed = float(report.get("elapsed_seconds", -1.0))
        report_attempt_elapsed = float(report.get("attempt_elapsed_seconds", -1.0))
    except (TypeError, ValueError) as error:
        raise ValueError("inference completion timing evidence is malformed") from error
    if (
        progress.get("schema_version") != INFERENCE_PROGRESS_SCHEMA_VERSION
        or progress.get("role") != INFERENCE_PROGRESS_ROLE
        or progress.get("status") != "completed"
        or progress.get("manifest") != identities["manifest"]
        or int(progress.get("expected_output_count", -1)) != len(expected_outputs)
        or int(progress.get("completed_output_count", -1)) != len(expected_outputs)
        or progress.get("outputs") != outputs
        or int(report.get("output_count", -1)) != len(expected_outputs)
        or progress_attempts < 1
        or report_attempts != progress_attempts
        or not math.isfinite(progress_elapsed)
        or progress_elapsed < 0.0
        or report_elapsed != progress_elapsed
        or not math.isfinite(report_attempt_elapsed)
        or report_attempt_elapsed < 0.0
        or report_attempt_elapsed > report_elapsed
        or progress.get("error_type") is not None
        or progress.get("error") is not None
    ):
        raise ValueError("inference progress completion evidence differs")
    expected_by_name = _expected_index(expected_outputs)
    if len(outputs) != len(expected_by_name):
        raise ValueError("inference report output count differs from manifest")
    seen: set[str] = set()
    for row in outputs:
        if not isinstance(row, dict):
            raise ValueError("inference report output row is malformed")
        filename = str(row.get("filename", ""))
        if filename in seen or filename not in expected_by_name:
            raise ValueError("inference report output set differs from manifest")
        seen.add(filename)
        expected = expected_by_name[filename]
        for field in ("filename", "path", "seed", "class_id", "prefix_budget"):
            if row.get(field) != expected.get(field):
                raise ValueError("inference report output identity differs from manifest")
        path = expected_paths[filename]
        digest = str(row.get("sha256", ""))
        if not path.is_file() or len(digest) != 64 or file_sha256(path) != digest:
            raise ValueError("inference report output digest differs from physical PNG")
    if seen != set(expected_by_name):
        raise ValueError("inference report output set is incomplete")
    return {
        "status": "verified",
        "manifest_sha256": identities["manifest"]["sha256"],
        "progress_sha256": identities["progress"]["sha256"],
        "attempt_count": int(progress.get("attempt_count", -1)),
        "output_count": len(outputs),
    }
