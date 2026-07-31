from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256


SCHEMA_VERSION = 1
ROLE = "generation_checkpoint_retention_inventory"
RUNWAY_ROLE = "generation_checkpoint_retention_runway"
CHECKPOINT_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")
REFERENCE_PATTERN = re.compile(r"[^\s\"'<>]*checkpoint_step_\d+\.pt")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".sh", ".txt"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unsupported checkpoint filename: {path.name}")
    return int(match.group(1))


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _strength(path: Path, policy: str) -> str:
    if policy == "authoritative":
        return "strong"
    return (
        "weak"
        if any(
            word in path.name.lower()
            for word in ("log", "monitor", "observer", "progress", "status", "watchdog")
        )
        else "strong"
    )


def _resolve_reference(
    raw: str,
    *,
    source: Path,
    inventory_root: Path,
    checkpoints: dict[str, Path],
) -> tuple[Path | None, str | None]:
    match = CHECKPOINT_PATTERN.search(raw.replace("\\", "/"))
    if match is None:
        return None, None
    name = match.group(0)
    cleaned = raw.strip("`[](),:;")
    value = Path(cleaned)
    candidates = (
        [value.resolve()]
        if value.is_absolute()
        else [
            (source.parent / value).resolve(),
            (inventory_root / value).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.as_posix() in checkpoints:
            return checkpoints[candidate.as_posix()], None
    same_name = [path for path in checkpoints.values() if path.name == name]
    return (same_name[0], None) if len(same_name) == 1 else (None, name)


def _scan_references(
    *,
    inventory_root: Path,
    checkpoints: dict[str, Path],
    reference_roots: list[tuple[str, Path, str]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    references = {path: [] for path in checkpoints}
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, root, policy in [("inventory", inventory_root, "operational"), *reference_roots]:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"reference root is missing: {root}")
        for source in sorted(root.rglob("*")):
            source_key = source.resolve().as_posix()
            if (
                not source.is_file()
                or source.suffix.lower() not in TEXT_SUFFIXES
                or source.name.endswith(".pt.integrity.json")
                or source_key in seen
            ):
                continue
            seen.add(source_key)
            try:
                text = source.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if source.suffix.lower() == ".json":
                try:
                    role = json.loads(text).get("role")
                except (AttributeError, json.JSONDecodeError):
                    role = None
                if role in {ROLE, RUNWAY_ROLE}:
                    continue
            raw_references = sorted(set(REFERENCE_PATTERN.findall(text)))
            if not raw_references:
                continue
            source_index = len(sources)
            strength = _strength(source, policy)
            sources.append(
                {
                    **_identity(source),
                    "label": label,
                    "policy": policy,
                    "strength": strength,
                }
            )
            for raw in raw_references:
                target, ambiguous = _resolve_reference(
                    raw,
                    source=source,
                    inventory_root=inventory_root,
                    checkpoints=checkpoints,
                )
                if target is not None:
                    references[target.as_posix()].append(
                        {"source_index": source_index, "strength": strength}
                    )
                elif ambiguous is not None:
                    unresolved.append(
                        {
                            "source_index": source_index,
                            "checkpoint_name": ambiguous,
                        }
                    )
    return sources, references, unresolved


def _run_state(run_dir: Path, checkpoints: list[Path]) -> dict[str, Any]:
    latest_path = run_dir / "latest.json"
    training_path = run_dir / "training_report.json"
    latest = _read_json(latest_path) if latest_path.is_file() else {}
    training = _read_json(training_path) if training_path.is_file() else {}
    max_step = max(_step(path) for path in checkpoints)
    latest_name = latest.get("checkpoint")
    issues = []
    if latest_name not in {path.name for path in checkpoints} or int(
        latest.get("step", -1)
    ) != max_step:
        issues.append("latest_checkpoint_binding_invalid")
    complete = bool(
        training.get("training_complete") is True
        and int(training.get("completed_steps", -1)) == max_step
    )
    if not complete:
        issues.append("training_report_not_complete_at_latest_step")
    protected = (
        training.get("config", {})
        .get("runtime", {})
        .get("protected_checkpoint_steps", [])
    )
    if not isinstance(protected, list) or not all(
        isinstance(item, int) for item in protected
    ):
        protected = []
        issues.append("protected_checkpoint_steps_invalid")
    return {
        "path": run_dir.resolve().as_posix(),
        "latest": _identity(latest_path) if latest_path.is_file() else None,
        "training_report": _identity(training_path) if training_path.is_file() else None,
        "latest_checkpoint": latest_name,
        "latest_step": int(latest.get("step", -1)),
        "max_checkpoint_step": max_step,
        "training_complete": complete,
        "protected_steps": sorted(set(protected)),
        "issues": issues,
    }


def _integrity(checkpoint: Path, physical_hash: bool) -> dict[str, Any]:
    sidecar = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    issues = []
    manifest = _read_json(sidecar) if sidecar.is_file() else {}
    if not manifest:
        issues.append("integrity_manifest_missing")
    expected_sha = manifest.get("checkpoint_sha256")
    if manifest and (
        manifest.get("schema_version") != 1
        or manifest.get("checkpoint") != checkpoint.name
        or int(manifest.get("step", -1)) != _step(checkpoint)
        or int(manifest.get("checkpoint_bytes", -1)) != checkpoint.stat().st_size
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
    ):
        issues.append("integrity_manifest_mismatch")
    actual_sha = file_sha256(checkpoint) if physical_hash else None
    if physical_hash and expected_sha != actual_sha:
        issues.append("checkpoint_sha256_mismatch")
    return {
        "manifest": _identity(sidecar) if sidecar.is_file() else None,
        "expected_sha256": expected_sha,
        "physical_hash_performed": physical_hash,
        "actual_sha256": actual_sha,
        "status": "verified" if not issues else "invalid",
        "issues": sorted(set(issues)),
    }


def build_checkpoint_retention_inventory(
    *,
    inventory_root: Path,
    reference_roots: list[tuple[str, Path, str]],
    physical_hash: bool,
) -> dict[str, Any]:
    root = inventory_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"inventory root is missing: {root}")
    if any(policy not in {"authoritative", "operational"} for _, _, policy in reference_roots):
        raise ValueError("reference root policy must be authoritative or operational")
    checkpoint_paths = sorted(root.rglob("checkpoint_step_*.pt"))
    if not checkpoint_paths:
        raise ValueError("inventory root contains no checkpoints")
    checkpoint_map = {path.resolve().as_posix(): path.resolve() for path in checkpoint_paths}
    sources, references, unresolved = _scan_references(
        inventory_root=root,
        checkpoints=checkpoint_map,
        reference_roots=reference_roots,
    )
    unresolved_names = {item["checkpoint_name"] for item in unresolved}
    run_dirs = sorted({path.parent.resolve() for path in checkpoint_paths})
    runs = [
        _run_state(run_dir, sorted(run_dir.glob("checkpoint_step_*.pt")))
        for run_dir in run_dirs
    ]
    run_by_path = {run["path"]: run for run in runs}
    run_index = {run["path"]: index for index, run in enumerate(runs)}
    counts = {"required": 0, "candidate_for_archive": 0, "indeterminate": 0}
    bytes_by_disposition = dict(counts)
    checkpoints = []
    invalid_integrity_count = 0
    for checkpoint in checkpoint_paths:
        checkpoint = checkpoint.resolve()
        run = run_by_path[checkpoint.parent.as_posix()]
        integrity = _integrity(checkpoint, physical_hash)
        invalid_integrity_count += integrity["status"] != "verified"
        bound = references[checkpoint.as_posix()]
        strong = any(item["strength"] == "strong" for item in bound)
        weak = any(item["strength"] == "weak" for item in bound)
        reasons = []
        if checkpoint.name == run["latest_checkpoint"]:
            reasons.append("latest_checkpoint")
        if _step(checkpoint) in run["protected_steps"]:
            reasons.append("protected_checkpoint_step")
        if strong:
            reasons.append("authoritative_reference")
        if reasons:
            disposition = "required"
        elif (
            integrity["status"] != "verified"
            or not physical_hash
            or run["issues"]
            or checkpoint.name in unresolved_names
            or weak
        ):
            disposition = "indeterminate"
            if integrity["status"] != "verified":
                reasons.append("integrity_not_verified")
            if not physical_hash:
                reasons.append("physical_hash_not_performed")
            if run["issues"]:
                reasons.append("run_state_invalid")
            if checkpoint.name in unresolved_names:
                reasons.append("ambiguous_reference")
            if weak:
                reasons.append("operational_reference")
        else:
            disposition = "candidate_for_archive"
            reasons.append("completed_unreferenced_intermediate_checkpoint")
        size = checkpoint.stat().st_size
        counts[disposition] += 1
        bytes_by_disposition[disposition] += size
        checkpoints.append(
            {
                "path": checkpoint.as_posix(),
                "run_index": run_index[run["path"]],
                "step": _step(checkpoint),
                "bytes": size,
                "integrity": integrity,
                "references": bound,
                "disposition": disposition,
                "reasons": sorted(set(reasons)),
            }
        )
    status = "pass" if invalid_integrity_count == 0 else "fail"
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": status,
        "archive_readiness": (
            "eligible_for_explicit_archive_review"
            if status == "pass" and counts["indeterminate"] == 0
            else "blocked"
        ),
        "read_only": True,
        "archive_or_deletion_authorized": False,
        "inventory_root": root.as_posix(),
        "reference_roots": [
            {
                "label": label,
                "path": path.resolve().as_posix(),
                "policy": policy,
            }
            for label, path, policy in reference_roots
        ],
        "physical_checkpoint_hashes_required_for_candidates": True,
        "physical_checkpoint_hashes_performed": physical_hash,
        "runs": runs,
        "reference_sources": sources,
        "unresolved_references": unresolved,
        "checkpoints": checkpoints,
        "summary": {
            "checkpoint_count": len(checkpoints),
            "checkpoint_bytes": sum(item["bytes"] for item in checkpoints),
            "counts_by_disposition": counts,
            "bytes_by_disposition": bytes_by_disposition,
            "invalid_integrity_count": invalid_integrity_count,
            "unresolved_reference_count": len(unresolved),
            "currently_reclaimable_bytes": 0,
            "potential_archive_candidate_bytes": bytes_by_disposition[
                "candidate_for_archive"
            ],
        },
    }


def verify_checkpoint_retention_inventory(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("role") != ROLE:
        raise ValueError("checkpoint retention inventory identity is invalid")
    if (
        report.get("read_only") is not True
        or report.get("archive_or_deletion_authorized") is not False
        or report.get("summary", {}).get("currently_reclaimable_bytes") != 0
    ):
        raise ValueError("checkpoint retention inventory weakens read-only policy")
    roots = []
    for root in report.get("reference_roots", []):
        policy = root.get("policy")
        if policy not in {"authoritative", "operational"}:
            raise ValueError("checkpoint retention reference policy is invalid")
        roots.append((root.get("label", ""), Path(root.get("path", "")), policy))
    replay = build_checkpoint_retention_inventory(
        inventory_root=Path(report.get("inventory_root", "")),
        reference_roots=roots,
        physical_hash=bool(report.get("physical_checkpoint_hashes_performed")),
    )
    if replay != report:
        raise ValueError("checkpoint retention inventory replay changed")
    return report


def build_retention_runway_report(
    *,
    retention_report: dict[str, Any],
    retention_report_path: Path,
    filesystem_path: Path,
    total_bytes: int,
    used_bytes: int,
    free_bytes: int,
    required_free_bytes: int,
) -> dict[str, Any]:
    verify_checkpoint_retention_inventory(retention_report)
    if min(total_bytes, used_bytes, free_bytes, required_free_bytes) < 0:
        raise ValueError("retention runway bytes must be non-negative")
    if used_bytes + free_bytes > total_bytes:
        raise ValueError("retention runway filesystem usage exceeds total capacity")
    candidate_bytes = retention_report["summary"]["potential_archive_candidate_bytes"]
    headroom = free_bytes - required_free_bytes
    return {
        "schema_version": 1,
        "role": RUNWAY_ROLE,
        "status": "pass" if headroom >= 0 else "fail",
        "read_only": True,
        "archive_or_deletion_authorized": False,
        "retention_inventory": _identity(retention_report_path),
        "filesystem": {
            "path": filesystem_path.resolve().as_posix(),
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "free_bytes": free_bytes,
        },
        "required_free_bytes": required_free_bytes,
        "current_headroom_bytes": headroom,
        "currently_reclaimable_bytes": 0,
        "potential_archive_candidate_bytes": candidate_bytes,
        "projected_headroom_after_unapproved_archive_bytes": headroom + candidate_bytes,
        "potential_archive_bytes_counted_as_current_capacity": False,
    }
