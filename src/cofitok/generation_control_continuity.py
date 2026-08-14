from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PLAN_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PLAN_ROLE = "generation_control_plane_continuity_plan"
MANIFEST_ROLE = "generation_control_plane_continuity_archive"
VERIFY_ROLE = "generation_control_plane_continuity_verification"
RESTORE_ROLE = "generation_control_plane_continuity_restore"
AUTHORIZATION_BOUNDARY = {
    "archive_static_control_assets_allowed": True,
    "restore_into_absent_bounded_paths_allowed": True,
    "training_launch_allowed": False,
    "sampling_or_evaluation_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "process_signaling_allowed": False,
    "gpu_allocation_allowed": False,
    "active_source_process_modification_allowed": False,
    "formal_checkout_modification_allowed": False,
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID = re.compile(r"[0-9a-f]{40}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} is unsafe")
    if not path.parts:
        raise ValueError(f"{label} is empty")
    return path


def _under(root: Path, value: str, *, label: str) -> Path:
    relative = _relative_path(value, label=label)
    return root.joinpath(*relative.parts)


def _run_git(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_state(project: Path) -> dict[str, Any]:
    project = project.resolve()
    return {
        "path": project.as_posix(),
        "revision": _run_git(["rev-parse", "HEAD"], cwd=project),
        "tree": _run_git(["rev-parse", "HEAD^{tree}"], cwd=project),
        "branch": _run_git(["branch", "--show-current"], cwd=project),
        "tracked_dirty": bool(
            _run_git(
                ["status", "--porcelain=v1", "--untracked-files=no"],
                cwd=project,
            )
        ),
    }


def bundle_heads(path: Path) -> list[dict[str, str]]:
    rows = []
    output = _run_git(["bundle", "list-heads", str(path.resolve())])
    for line in output.splitlines():
        values = line.split(maxsplit=1)
        if len(values) != 2 or not _OBJECT_ID.fullmatch(values[0]):
            raise ValueError(f"git bundle advertised head is malformed: {path}")
        rows.append({"revision": values[0], "ref": values[1]})
    if not rows:
        raise ValueError(f"git bundle advertises no heads: {path}")
    return rows


def _validate_expected_file(row: Mapping[str, Any], *, label: str) -> None:
    if (
        type(row.get("expected_bytes")) is not int
        or row["expected_bytes"] <= 0
        or not isinstance(row.get("expected_sha256"), str)
        or not _SHA256.fullmatch(row["expected_sha256"])
    ):
        raise ValueError(f"{label} identity is malformed")


def load_continuity_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    payload = _read_json_object(plan_path, label="control continuity plan")
    if set(payload) != {
        "schema_version",
        "role",
        "authorization_boundary",
        "bundles",
        "checkouts",
        "runtime_files",
    }:
        raise ValueError("control continuity plan schema differs")
    if (
        payload["schema_version"] != PLAN_SCHEMA_VERSION
        or payload["role"] != PLAN_ROLE
        or payload["authorization_boundary"] != AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("control continuity plan identity differs")

    raw_bundles = payload["bundles"]
    if not isinstance(raw_bundles, list) or not raw_bundles:
        raise ValueError("control continuity plan has no git bundles")
    bundles = []
    names: set[str] = set()
    archive_paths: set[str] = set()
    restore_paths: set[str] = set()
    seed_count = 0
    for index, raw in enumerate(raw_bundles):
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "source_path",
            "archive_relative_path",
            "restore_relative_path",
            "expected_bytes",
            "expected_sha256",
            "expected_revision",
            "expected_ref",
            "repository_seed",
        }:
            raise ValueError(f"control continuity bundle {index} schema differs")
        name = raw["name"]
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ValueError(f"control continuity bundle {index} name is invalid")
        source_text = raw["source_path"]
        if not isinstance(source_text, str) or not source_text:
            raise ValueError(f"control continuity bundle {name} source is invalid")
        source = Path(source_text)
        archive = _relative_path(
            raw["archive_relative_path"], label=f"bundle {name} archive path"
        ).as_posix()
        restore = _relative_path(
            raw["restore_relative_path"], label=f"bundle {name} restore path"
        ).as_posix()
        if not source.is_absolute() and not PurePosixPath(source_text).is_absolute():
            raise ValueError(f"control continuity bundle {name} source is not absolute")
        if archive in archive_paths or restore in restore_paths:
            raise ValueError(f"control continuity bundle {name} path is duplicated")
        _validate_expected_file(raw, label=f"control continuity bundle {name}")
        if (
            not isinstance(raw["expected_revision"], str)
            or not _OBJECT_ID.fullmatch(raw["expected_revision"])
            or not isinstance(raw["expected_ref"], str)
            or not raw["expected_ref"].startswith("refs/heads/")
            or type(raw["repository_seed"]) is not bool
        ):
            raise ValueError(f"control continuity bundle {name} git identity is invalid")
        seed_count += int(raw["repository_seed"])
        names.add(name)
        archive_paths.add(archive)
        restore_paths.add(restore)
        bundles.append({**raw, "source_path": source, "archive_relative_path": archive, "restore_relative_path": restore})
    if seed_count != 1 or not bundles[0]["repository_seed"]:
        raise ValueError("the first continuity bundle must be the sole repository seed")

    raw_checkouts = payload["checkouts"]
    if not isinstance(raw_checkouts, list) or not raw_checkouts:
        raise ValueError("control continuity plan has no checkouts")
    checkouts = []
    checkout_names: set[str] = set()
    checkout_paths: set[str] = set()
    bundle_revisions = {row["expected_revision"] for row in bundles}
    for index, raw in enumerate(raw_checkouts):
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "revision",
            "tree",
            "branch",
            "restore_relative_path",
        }:
            raise ValueError(f"control continuity checkout {index} schema differs")
        name = raw["name"]
        restore = _relative_path(
            raw["restore_relative_path"], label=f"checkout {name} restore path"
        ).as_posix()
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in checkout_names
            or not isinstance(raw["revision"], str)
            or not _OBJECT_ID.fullmatch(raw["revision"])
            or raw["revision"] not in bundle_revisions
            or not isinstance(raw["tree"], str)
            or not _OBJECT_ID.fullmatch(raw["tree"])
            or not isinstance(raw["branch"], str)
            or not raw["branch"].startswith("scale/")
            or restore in checkout_paths
        ):
            raise ValueError(f"control continuity checkout {index} is invalid")
        checkout_names.add(name)
        checkout_paths.add(restore)
        checkouts.append({**raw, "restore_relative_path": restore})

    raw_runtime = payload["runtime_files"]
    if not isinstance(raw_runtime, list) or not raw_runtime:
        raise ValueError("control continuity plan has no runtime files")
    runtime_files = []
    runtime_names: set[str] = set()
    for index, raw in enumerate(raw_runtime):
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "source_path",
            "archive_relative_path",
            "restore_relative_path",
            "expected_bytes",
            "expected_sha256",
        }:
            raise ValueError(f"control continuity runtime file {index} schema differs")
        name = raw["name"]
        source_text = raw["source_path"]
        if not isinstance(source_text, str) or not source_text:
            raise ValueError(
                f"control continuity runtime file {name} source is invalid"
            )
        source = Path(source_text)
        archive = _relative_path(
            raw["archive_relative_path"], label=f"runtime {name} archive path"
        ).as_posix()
        restore = _relative_path(
            raw["restore_relative_path"], label=f"runtime {name} restore path"
        ).as_posix()
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in runtime_names
            or (
                not source.is_absolute()
                and not PurePosixPath(source_text).is_absolute()
            )
            or archive in archive_paths
            or restore in restore_paths
        ):
            raise ValueError(f"control continuity runtime file {index} is invalid")
        _validate_expected_file(raw, label=f"control continuity runtime file {name}")
        runtime_names.add(name)
        archive_paths.add(archive)
        restore_paths.add(restore)
        runtime_files.append({**raw, "source_path": source, "archive_relative_path": archive, "restore_relative_path": restore})

    return {
        "identity": _identity(plan_path),
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "bundles": bundles,
        "checkouts": checkouts,
        "runtime_files": runtime_files,
    }


def _require_source_identity(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(row["source_path"])
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    observed = _identity(path)
    if (
        observed["bytes"] != row["expected_bytes"]
        or observed["sha256"] != row["expected_sha256"]
    ):
        raise ValueError(f"{label} identity differs")
    return observed


def _copy_bound_file(source: Path, destination: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"continuity archive destination exists: {destination}")
    source_mode = stat.S_IMODE(source.stat().st_mode)
    shutil.copyfile(source, destination)
    os.chmod(destination, source_mode)
    with destination.open("rb+") as handle:
        os.fsync(handle.fileno())
    archived = _identity(destination)
    source_after = _identity(source)
    if (
        archived["bytes"] != expected["expected_bytes"]
        or archived["sha256"] != expected["expected_sha256"]
        or source_after["bytes"] != expected["expected_bytes"]
        or source_after["sha256"] != expected["expected_sha256"]
    ):
        raise ValueError(f"continuity source changed while copying: {source}")
    return {
        "bytes": archived["bytes"],
        "sha256": archived["sha256"],
        "mode": source_mode,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_continuity_archive(
    *,
    project: Path,
    plan_path: Path,
    output_root: Path,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    project = project.resolve()
    output_root = output_root.resolve()
    expected_git = {
        "path": project.as_posix(),
        "revision": expected_revision,
        "tree": expected_tree,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if git_state(project) != expected_git:
        raise ValueError("control continuity builder checkout differs")
    plan = load_continuity_plan(plan_path)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"continuity archive output exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=output_root.parent,
            prefix=f".{output_root.name}.building.",
        )
    )
    try:
        archived_bundles = []
        for row in plan["bundles"]:
            source_identity = _require_source_identity(
                row, label=f"continuity bundle {row['name']}"
            )
            heads = bundle_heads(Path(row["source_path"]))
            expected_heads = [
                {"revision": row["expected_revision"], "ref": row["expected_ref"]}
            ]
            if heads != expected_heads:
                raise ValueError(f"continuity bundle {row['name']} heads differ")
            relative = f"payload/{row['archive_relative_path']}"
            archived = _copy_bound_file(
                Path(row["source_path"]),
                _under(staging, relative, label="archive bundle path"),
                row,
            )
            archived_bundles.append(
                {
                    "name": row["name"],
                    "source": source_identity,
                    "archive_relative_path": relative,
                    "restore_relative_path": row["restore_relative_path"],
                    "bytes": archived["bytes"],
                    "sha256": archived["sha256"],
                    "mode": archived["mode"],
                    "expected_revision": row["expected_revision"],
                    "expected_ref": row["expected_ref"],
                    "repository_seed": row["repository_seed"],
                    "advertised_heads": heads,
                }
            )

        archived_runtime = []
        for row in plan["runtime_files"]:
            source_identity = _require_source_identity(
                row, label=f"continuity runtime file {row['name']}"
            )
            relative = f"payload/{row['archive_relative_path']}"
            archived = _copy_bound_file(
                Path(row["source_path"]),
                _under(staging, relative, label="archive runtime path"),
                row,
            )
            archived_runtime.append(
                {
                    "name": row["name"],
                    "source": source_identity,
                    "archive_relative_path": relative,
                    "restore_relative_path": row["restore_relative_path"],
                    "bytes": archived["bytes"],
                    "sha256": archived["sha256"],
                    "mode": archived["mode"],
                }
            )

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "role": MANIFEST_ROLE,
            "status": "pass",
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "builder_git": expected_git,
            "plan": plan["identity"],
            "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
            "bundles": archived_bundles,
            "checkouts": plan["checkouts"],
            "runtime_files": archived_runtime,
            "effects": {
                "source_files_modified": False,
                "source_processes_modified": False,
                "processes_launched": False,
                "processes_signaled": False,
                "gpu_queried_or_allocated": False,
                "formal_checkout_modified": False,
            },
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest_path = output_root / "manifest.json"
    verification = verify_continuity_archive(
        archive_root=output_root,
        expected_manifest_sha256=_sha256(manifest_path),
    )
    return {
        "manifest": _identity(manifest_path),
        "archive_root": output_root.as_posix(),
        "verification": verification,
    }


def verify_continuity_archive(
    *,
    archive_root: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    archive_root = archive_root.resolve()
    manifest_path = archive_root / "manifest.json"
    if not _SHA256.fullmatch(expected_manifest_sha256):
        raise ValueError("expected continuity manifest SHA256 is malformed")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("continuity archive manifest is missing or a symlink")
    manifest_identity = _identity(manifest_path)
    if manifest_identity["sha256"] != expected_manifest_sha256:
        raise ValueError("continuity archive manifest SHA256 differs")
    manifest = _read_json_object(manifest_path, label="continuity archive manifest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("role") != MANIFEST_ROLE
        or manifest.get("status") != "pass"
        or manifest.get("complete") is not True
        or manifest.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or manifest.get("effects", {}).get("processes_launched") is not False
        or manifest.get("effects", {}).get("source_processes_modified") is not False
    ):
        raise ValueError("continuity archive manifest contract differs")

    expected_payload_paths: set[str] = set()
    verified_bundles = []
    bundles = manifest.get("bundles")
    runtime_files = manifest.get("runtime_files")
    checkouts = manifest.get("checkouts")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("continuity archive bundles are missing")
    if not isinstance(runtime_files, list) or not runtime_files:
        raise ValueError("continuity archive runtime files are missing")
    if not isinstance(checkouts, list) or not checkouts:
        raise ValueError("continuity archive checkouts are missing")
    if sum(row.get("repository_seed") is True for row in bundles) != 1 or not bundles[0].get("repository_seed"):
        raise ValueError("continuity archive repository seed differs")

    for row in [*bundles, *runtime_files]:
        if not isinstance(row, dict):
            raise ValueError("continuity archive file row is malformed")
        relative = _relative_path(
            row.get("archive_relative_path"), label="manifest archive path"
        ).as_posix()
        if relative in expected_payload_paths:
            raise ValueError("continuity archive path is duplicated")
        expected_payload_paths.add(relative)
        path = _under(archive_root, relative, label="manifest archive path")
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"continuity archived file is missing or unsafe: {relative}")
        identity = _identity(path)
        if identity["bytes"] != row.get("bytes") or identity["sha256"] != row.get("sha256"):
            raise ValueError(f"continuity archived file identity differs: {relative}")
        if stat.S_IMODE(path.stat().st_mode) != row.get("mode"):
            raise ValueError(f"continuity archived file mode differs: {relative}")
        if row in bundles:
            heads = bundle_heads(path)
            if heads != row.get("advertised_heads") or heads != [
                {
                    "revision": row.get("expected_revision"),
                    "ref": row.get("expected_ref"),
                }
            ]:
                raise ValueError(f"continuity archived bundle heads differ: {relative}")
            verified_bundles.append(row["name"])

    payload_root = archive_root / "payload"
    observed_payload_paths = {
        path.relative_to(archive_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_payload_paths != expected_payload_paths:
        raise ValueError("continuity archive payload membership differs")
    return {
        "schema_version": 1,
        "role": VERIFY_ROLE,
        "status": "pass",
        "complete": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": archive_root.as_posix(),
        "manifest": manifest_identity,
        "verified_bundle_names": verified_bundles,
        "verified_file_count": len(expected_payload_paths),
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "effects": {
            "processes_launched": False,
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "source_files_modified": False,
        },
    }


def _require_absent_targets(destination_root: Path, relatives: Sequence[str]) -> None:
    seen = set()
    for relative in relatives:
        top = _relative_path(relative, label="restore target").parts[0]
        if top in seen:
            continue
        seen.add(top)
        target = destination_root / top
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"continuity restore target exists: {target}")


def restore_continuity_archive(
    *,
    archive_root: Path,
    expected_manifest_sha256: str,
    destination_root: Path,
) -> dict[str, Any]:
    verification = verify_continuity_archive(
        archive_root=archive_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    archive_root = archive_root.resolve()
    destination_root = destination_root.resolve()
    manifest = _read_json_object(
        archive_root / "manifest.json", label="continuity archive manifest"
    )
    bare_relative = ".cofitok-capacity-control-continuity-v1.git"
    targets = [bare_relative]
    targets.extend(row["restore_relative_path"] for row in manifest["bundles"])
    targets.extend(row["restore_relative_path"] for row in manifest["checkouts"])
    targets.extend(row["restore_relative_path"] for row in manifest["runtime_files"])
    destination_root.mkdir(parents=True, exist_ok=True)
    _require_absent_targets(destination_root, targets)

    bare = destination_root / bare_relative
    _run_git(["init", "--bare", str(bare)])
    restored_bundles = []
    for row in manifest["bundles"]:
        archived = _under(
            archive_root,
            row["archive_relative_path"],
            label="archived bundle path",
        )
        destination = _under(
            destination_root,
            row["restore_relative_path"],
            label="restored bundle path",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archived, destination)
        os.chmod(destination, row["mode"])
        identity = _identity(destination)
        if identity["bytes"] != row["bytes"] or identity["sha256"] != row["sha256"]:
            raise ValueError(f"restored bundle identity differs: {row['name']}")
        continuity_ref = f"refs/heads/continuity/{row['name']}"
        _run_git(
            [
                f"--git-dir={bare}",
                "fetch",
                "--no-tags",
                str(destination),
                f"{row['expected_ref']}:{continuity_ref}",
            ]
        )
        observed_revision = _run_git(
            [f"--git-dir={bare}", "rev-parse", continuity_ref]
        )
        if observed_revision != row["expected_revision"]:
            raise ValueError(f"restored bundle revision differs: {row['name']}")
        restored_bundles.append(
            {
                "name": row["name"],
                "path": destination.as_posix(),
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
                "revision": observed_revision,
                "ref": continuity_ref,
            }
        )

    restored_checkouts = []
    for row in manifest["checkouts"]:
        revision = _run_git(
            [f"--git-dir={bare}", "rev-parse", row["revision"]]
        )
        tree = _run_git(
            [f"--git-dir={bare}", "rev-parse", f"{row['revision']}^{{tree}}"]
        )
        if revision != row["revision"] or tree != row["tree"]:
            raise ValueError(f"continuity checkout object differs: {row['name']}")
        branch_ref = f"refs/heads/{row['branch']}"
        _run_git([f"--git-dir={bare}", "update-ref", branch_ref, row["revision"]])
        destination = _under(
            destination_root,
            row["restore_relative_path"],
            label="restored checkout path",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            [
                f"--git-dir={bare}",
                "worktree",
                "add",
                str(destination),
                row["branch"],
            ]
        )
        observed = git_state(destination)
        expected = {
            "path": destination.resolve().as_posix(),
            "revision": row["revision"],
            "tree": row["tree"],
            "branch": row["branch"],
            "tracked_dirty": False,
        }
        if observed != expected:
            raise ValueError(f"restored checkout state differs: {row['name']}")
        restored_checkouts.append({"name": row["name"], **observed})

    restored_runtime = []
    for row in manifest["runtime_files"]:
        archived = _under(
            archive_root,
            row["archive_relative_path"],
            label="archived runtime path",
        )
        destination = _under(
            destination_root,
            row["restore_relative_path"],
            label="restored runtime path",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"continuity runtime restore target exists: {destination}")
        shutil.copyfile(archived, destination)
        os.chmod(destination, row["mode"])
        identity = _identity(destination)
        if identity["bytes"] != row["bytes"] or identity["sha256"] != row["sha256"]:
            raise ValueError(f"restored runtime identity differs: {row['name']}")
        restored_runtime.append(
            {
                "name": row["name"],
                "path": destination.as_posix(),
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
                "mode": stat.S_IMODE(destination.stat().st_mode),
            }
        )

    return {
        "schema_version": 1,
        "role": RESTORE_ROLE,
        "status": "pass",
        "complete": True,
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "destination_root": destination_root.as_posix(),
        "archive_verification": verification,
        "repository": {
            "path": bare.as_posix(),
            "bare": True,
        },
        "bundles": restored_bundles,
        "checkouts": restored_checkouts,
        "runtime_files": restored_runtime,
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "effects": {
            "processes_launched": False,
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "formal_checkout_modified": False,
            "experiment_execution_authorized": False,
        },
    }
