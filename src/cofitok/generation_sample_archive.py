from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.reporting import file_sha256


GENERATION_SAMPLE_ARCHIVE_AUDIT_SCHEMA_VERSION = 1
_IMMUTABLE_SAMPLING_FIELDS = (
    "git",
    "runtime_environment",
    "runtime_environment_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_integrity_manifest",
    "checkpoint_step",
    "weights",
    "sampling",
    "output_dirs",
)


def _canonical_member_name(name: str) -> str:
    if not name or "\\" in name:
        raise ValueError(f"archive member name is malformed: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"archive member path is unsafe: {name!r}")
    canonical = path.as_posix()
    if canonical != name.rstrip("/"):
        raise ValueError(f"archive member path is not canonical: {name!r}")
    return canonical


def _read_member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member is not readable: {member.name}")
    payload = handle.read()
    if len(payload) != member.size:
        raise ValueError(f"archive member size changed while reading: {member.name}")
    return payload


def _json_member(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    name: str,
) -> tuple[dict[str, Any], str]:
    member = members.get(name)
    if member is None or not member.isfile():
        raise ValueError(f"archive is missing required JSON file: {name}")
    payload = _read_member_bytes(archive, member)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"archive JSON file is invalid: {name}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"archive JSON file must contain an object: {name}")
    return decoded, hashlib.sha256(payload).hexdigest()


def _sample_set_digest(
    archive: tarfile.TarFile,
    sample_members: list[tarfile.TarInfo],
) -> str:
    digest = hashlib.sha256()
    for member in sorted(sample_members, key=lambda item: PurePosixPath(item.name).name):
        name = PurePosixPath(member.name).name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"sample member is not readable: {member.name}")
        bytes_read = 0
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            bytes_read += len(block)
            digest.update(block)
        if bytes_read != member.size:
            raise ValueError(f"sample member size changed while reading: {member.name}")
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_source_path(value: Any, *, expected_suffix: str, field: str) -> str:
    path = str(value)
    if not path.startswith("/") or not path.endswith(f"/{expected_suffix}"):
        raise ValueError(f"{field} does not preserve the expected remote suffix")
    return path


def _audit_sampling_root(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    root: str,
) -> dict[str, Any]:
    report, report_sha = _json_member(archive, members, f"{root}/sampling_report.json")
    manifest, manifest_sha = _json_member(
        archive, members, f"{root}/sampling_manifest.json"
    )
    progress, progress_sha = _json_member(
        archive, members, f"{root}/sampling_progress.json"
    )
    if report.get("status") != "completed" or progress.get("status") != "completed":
        raise ValueError(f"sampling root is not completed: {root}")
    if report.get("sampling_manifest_sha256") != manifest_sha:
        raise ValueError(f"sampling manifest digest differs from report: {root}")
    if progress.get("sampling_manifest_sha256") != manifest_sha:
        raise ValueError(f"sampling progress belongs to another manifest: {root}")
    for field in _IMMUTABLE_SAMPLING_FIELDS:
        if manifest.get(field) != report.get(field):
            raise ValueError(f"sampling report {field} differs from manifest: {root}")

    sampling = report.get("sampling")
    output_dirs = report.get("output_dirs")
    sample_sets = report.get("sample_sets")
    if not isinstance(sampling, dict):
        raise ValueError(f"sampling contract is missing: {root}")
    if not isinstance(output_dirs, dict) or not isinstance(sample_sets, dict):
        raise ValueError(f"sampling outputs or digests are missing: {root}")
    budgets = [int(value) for value in sampling.get("prefix_budgets", [])]
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError(f"sampling prefix budgets are malformed: {root}")
    budget_keys = {str(value) for value in budgets}
    if set(output_dirs) != budget_keys or set(sample_sets) != budget_keys:
        raise ValueError(f"sampling prefix budgets disagree across reports: {root}")
    if progress.get("prefix_budgets") != budgets:
        raise ValueError(f"sampling progress prefix budgets differ: {root}")
    if progress.get("sample_sets") != sample_sets:
        raise ValueError(f"sampling progress sample digests differ: {root}")

    num_samples = int(sampling.get("num_samples", -1))
    start_index = int(sampling.get("start_index", 0))
    if num_samples <= 0 or start_index < 0:
        raise ValueError(f"sampling range is invalid: {root}")
    if int(progress.get("completed_samples", -1)) != num_samples:
        raise ValueError(f"sampling progress count differs: {root}")
    expected_names = {
        f"{index:06d}.png" for index in range(start_index, start_index + num_samples)
    }

    source_roots: set[str] = set()
    audited_sets: dict[str, Any] = {}
    for budget in budgets:
        key = str(budget)
        prefix = f"{root}/prefix_{budget}"
        source_dir = _validate_source_path(
            output_dirs[key], expected_suffix=prefix, field=f"output_dirs[{key}]"
        )
        source_roots.add(source_dir[: -(len(f"/prefix_{budget}"))])
        prefix_members = [
            member
            for name, member in members.items()
            if member.isfile() and name.startswith(f"{prefix}/")
        ]
        if any(PurePosixPath(member.name).parent.as_posix() != prefix for member in prefix_members):
            raise ValueError(f"sample prefix contains nested files: {prefix}")
        actual_names = {PurePosixPath(member.name).name for member in prefix_members}
        if actual_names != expected_names or len(prefix_members) != len(expected_names):
            raise ValueError(f"sample filenames are incomplete or unexpected: {prefix}")
        declared = sample_sets[key]
        if not isinstance(declared, dict) or int(declared.get("count", -1)) != num_samples:
            raise ValueError(f"sample-set count is invalid: {prefix}")
        actual_sha = _sample_set_digest(archive, prefix_members)
        if actual_sha != declared.get("sha256"):
            raise ValueError(f"sample-set digest differs from report: {prefix}")
        audited_sets[key] = {
            "archive_directory": prefix,
            "source_directory": source_dir,
            "count": num_samples,
            "logical_bytes": sum(member.size for member in prefix_members),
            "sha256": actual_sha,
        }
    if len(source_roots) != 1:
        raise ValueError(f"sampling outputs do not share one source root: {root}")
    source_root = next(iter(source_roots))
    _validate_source_path(
        report.get("sampling_progress"),
        expected_suffix=f"{root}/sampling_progress.json",
        field="sampling_progress",
    )

    root_files = [
        member
        for name, member in members.items()
        if member.isfile() and name.startswith(f"{root}/")
    ]
    return {
        "archive_root": root,
        "source_root": source_root,
        "file_count": len(root_files),
        "logical_bytes": sum(member.size for member in root_files),
        "metadata_sha256": {
            "sampling_manifest.json": manifest_sha,
            "sampling_progress.json": progress_sha,
            "sampling_report.json": report_sha,
        },
        "checkpoint": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "git": report.get("git"),
        "runtime_environment_sha256": report.get("runtime_environment_sha256"),
        "sample_sets": audited_sets,
    }


def audit_generation_sample_archive(
    *,
    archive_path: str | Path,
    expected_archive_sha256: str,
    expected_roots: list[str],
) -> dict[str, Any]:
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"sample archive is missing: {archive_path}")
    if len(expected_archive_sha256) != 64:
        raise ValueError("expected archive SHA256 is malformed")
    archive_sha = file_sha256(archive_path)
    if archive_sha != expected_archive_sha256:
        raise ValueError("sample archive SHA256 differs from the expected digest")
    canonical_roots = [_canonical_member_name(root) for root in expected_roots]
    if not canonical_roots or len(set(canonical_roots)) != len(canonical_roots):
        raise ValueError("expected archive roots are missing or duplicated")
    if any("/" not in root for root in canonical_roots):
        raise ValueError("each expected archive root must include a run and sample directory")

    with tarfile.open(archive_path, mode="r:") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            name = _canonical_member_name(member.name)
            if name in members:
                raise ValueError(f"archive contains a duplicate member: {name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"archive contains a non-regular member: {name}")
            inside_root = any(
                name == root or name.startswith(f"{root}/") for root in canonical_roots
            )
            ancestor_directory = member.isdir() and any(
                root.startswith(f"{name}/") for root in canonical_roots
            )
            if not (inside_root or ancestor_directory):
                raise ValueError(f"archive member is outside expected roots: {name}")
            members[name] = member
        for root in canonical_roots:
            root_member = members.get(root)
            if root_member is None or not root_member.isdir():
                raise ValueError(f"archive root directory is missing: {root}")
        audited_roots = [
            _audit_sampling_root(archive, members, root) for root in canonical_roots
        ]

    source_roots = [item["source_root"] for item in audited_roots]
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("archive sampling roots map to duplicate source directories")
    archive_stat = archive_path.stat()
    return {
        "schema_version": GENERATION_SAMPLE_ARCHIVE_AUDIT_SCHEMA_VERSION,
        "status": "pass",
        "archive": archive_path.as_posix(),
        "archive_bytes": archive_stat.st_size,
        "archive_sha256": archive_sha,
        "member_policy": {
            "absolute_paths_allowed": False,
            "path_traversal_allowed": False,
            "links_allowed": False,
            "extra_roots_allowed": False,
        },
        "summary": {
            "root_count": len(audited_roots),
            "regular_file_count": sum(item["file_count"] for item in audited_roots),
            "logical_file_bytes": sum(item["logical_bytes"] for item in audited_roots),
            "sample_count": sum(
                sample_set["count"]
                for item in audited_roots
                for sample_set in item["sample_sets"].values()
            ),
            "source_roots": source_roots,
        },
        "roots": audited_roots,
        "restore_contract": {
            "target_root": "/root/autodl-tmp/CoFiTok/checkpoints/generation",
            "existing_source_roots_must_be_absent": True,
            "post_restore_archive_audit_required": True,
        },
        "remote_source_deletion_performed": False,
    }
