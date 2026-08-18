from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_ROLE = "generation_postbridge_two_stage_deployment_manifest"
REPORT_ROLE = "generation_postbridge_two_stage_deployment"
FORMAL_BRANCH = "scale/generative-system"
DEFAULT_TERMINAL_STATUSES = [
    "cancelled",
    "complete",
    "completed",
    "failed",
    "hold",
    "pass",
    "stopped",
]
DEFAULT_BLOCKED_PROCESS_MARKERS = [
    "scripts/train_generation.py",
    "generation_stability_full_data_quality_bridge_100k_execute.sh",
]
AUTHORIZATION_BOUNDARY = {
    "default_mode": "preflight",
    "formal_repository_read_only_in_preflight": True,
    "apply_requires_explicit_mode_and_exact_target_confirmation": True,
    "apply_while_bound_training_or_controller_active_allowed": False,
    "training_launch_allowed": False,
    "sampling_or_evaluation_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "full_300k_launch_allowed": False,
    "process_signaling_allowed": False,
    "gpu_query_or_allocation_allowed": False,
}

_OBJECT_ID = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_object_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_OBJECT_ID.fullmatch(value.lower()))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value.lower()))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"deployment source is not a regular file: {source}")
    payload = source.read_bytes()
    return {
        "path": source.as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=text,
    )
    if check and completed.returncode:
        stdout = completed.stdout if isinstance(completed.stdout, str) else b""
        stderr = completed.stderr if isinstance(completed.stderr, str) else b""
        detail = (stderr or stdout or "").strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}: {detail}"
        )
    return completed


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return _run(["git", *arguments], cwd=repository, check=check)


def _git_text(repository: Path, *arguments: str) -> str:
    return str(_git(repository, *arguments).stdout).strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = _run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=False,
    )
    return bytes(completed.stdout)


def _git_object_exists(repository: Path, revision: str) -> bool:
    return (
        _git(repository, "cat-file", "-e", f"{revision}^{{commit}}", check=False).returncode
        == 0
    )


def _git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    return (
        _git(
            repository,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            check=False,
        ).returncode
        == 0
    )


def _status_row_count(payload: bytes) -> int:
    return sum(bool(value) for value in payload.split(b"\0"))


def git_state(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"formal repository is not a directory: {root}")
    tracked = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=no",
    )
    full = _git_bytes(root, "status", "--porcelain=v1", "-z")
    refs = _git(root, "show-ref", "--head", "-d", check=False)
    if refs.returncode not in (0, 1):
        raise RuntimeError("unable to enumerate formal repository refs")
    object_store = _git_text(root, "count-objects", "-v")
    return {
        "path": root.as_posix(),
        "revision": _git_text(root, "rev-parse", "HEAD").lower(),
        "tree": _git_text(root, "rev-parse", "HEAD^{tree}").lower(),
        "branch": _git_text(root, "branch", "--show-current"),
        "tracked_dirty": bool(tracked),
        "tracked_status_rows": _status_row_count(tracked),
        "tracked_status_sha256": _sha256_bytes(tracked),
        "full_status_rows": _status_row_count(full),
        "full_status_sha256": _sha256_bytes(full),
        "refs_sha256": _sha256_bytes(str(refs.stdout).encode("utf-8")),
        "object_store_sha256": _sha256_bytes(object_store.encode("utf-8")),
    }


def bundle_prerequisites(path: str | Path) -> list[str]:
    source = Path(path).resolve()
    prerequisites: list[str] = []
    with source.open("rb") as handle:
        first = handle.readline()
        if not first.startswith(b"# v") or b"git bundle" not in first:
            raise ValueError(f"Git bundle header is invalid: {source}")
        for raw_line in handle:
            line = raw_line.rstrip(b"\r\n")
            if not line:
                break
            if not line.startswith(b"-"):
                continue
            try:
                revision = line[1:].split(b" ", 1)[0].decode("ascii").lower()
            except UnicodeDecodeError as error:
                raise ValueError("Git bundle prerequisite is not ASCII") from error
            if not _is_object_id(revision):
                raise ValueError(f"Git bundle prerequisite is malformed: {revision}")
            prerequisites.append(revision)
    if len(prerequisites) != len(set(prerequisites)):
        raise ValueError("Git bundle repeats a prerequisite")
    return sorted(prerequisites)


def bundle_heads(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).resolve()
    completed = _run(["git", "bundle", "list-heads", str(source)])
    heads: list[dict[str, str]] = []
    for line in str(completed.stdout).splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _is_object_id(fields[0]):
            raise ValueError(f"Git bundle advertised head is malformed: {source}")
        heads.append({"revision": fields[0].lower(), "ref": fields[1]})
    if not heads:
        raise ValueError(f"Git bundle advertises no heads: {source}")
    return heads


def verify_bundle(path: str | Path, *, repository: Path) -> dict[str, str]:
    source = Path(path).resolve()
    completed = _run(
        ["git", "bundle", "verify", str(source)],
        cwd=repository,
    )
    return {
        "status": "pass",
        "stdout": str(completed.stdout).strip(),
        "stderr": str(completed.stderr).strip(),
    }


def _bundle_manifest_row(
    path: Path,
    *,
    repository: Path,
    name: str,
    expected_revision: str,
    expected_ref: str,
    expected_prerequisites: Sequence[str],
) -> dict[str, Any]:
    identity = file_identity(path)
    heads = bundle_heads(path)
    prerequisites = bundle_prerequisites(path)
    verification = verify_bundle(path, repository=repository)
    expected_heads = [{"revision": expected_revision, "ref": expected_ref}]
    if heads != expected_heads:
        raise ValueError(f"{name} bundle advertised head differs")
    if prerequisites != sorted(expected_prerequisites):
        raise ValueError(f"{name} bundle prerequisites differ")
    return {
        "name": name,
        "filename": path.name,
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
        "advertised_heads": heads,
        "prerequisites": prerequisites,
        "builder_verification": verification,
    }


def _require_absolute_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", value))
    if not PurePosixPath(value).is_absolute() and not windows_absolute:
        raise ValueError(f"{label} must be absolute")
    return value.replace("\\", "/")


def build_manifest(
    *,
    repository: Path,
    formal_repository_path: str,
    formal_revision: str,
    formal_branch: str,
    bootstrap_bundle: Path,
    bootstrap_revision: str,
    bootstrap_ref: str,
    bootstrap_prerequisites: Sequence[str],
    integration_bundle: Path,
    integration_revision: str,
    integration_ref: str,
    quality_bridge_root: str,
) -> dict[str, Any]:
    repository = repository.resolve()
    for label, value in (
        ("formal revision", formal_revision),
        ("bootstrap revision", bootstrap_revision),
        ("integration revision", integration_revision),
        *(('bootstrap prerequisite', value) for value in bootstrap_prerequisites),
    ):
        if not _is_object_id(value):
            raise ValueError(f"{label} must be a full Git SHA-1")
    formal_revision = formal_revision.lower()
    bootstrap_revision = bootstrap_revision.lower()
    integration_revision = integration_revision.lower()
    bootstrap_prerequisites = sorted(value.lower() for value in bootstrap_prerequisites)
    if formal_branch != FORMAL_BRANCH:
        raise ValueError("post-bridge formal branch differs")
    if len(bootstrap_prerequisites) != 2 or len(set(bootstrap_prerequisites)) != 2:
        raise ValueError("post-bridge bootstrap requires two distinct prerequisites")
    if formal_revision not in bootstrap_prerequisites:
        raise ValueError("formal revision is not a bootstrap prerequisite")
    if not bootstrap_ref.startswith("refs/heads/"):
        raise ValueError("bootstrap advertised ref is not a branch")
    if not integration_ref.startswith("refs/heads/"):
        raise ValueError("integration advertised ref is not a branch")
    integration_branch = integration_ref.removeprefix("refs/heads/")

    builder = git_state(repository)
    if builder["tracked_dirty"]:
        raise ValueError("manifest builder repository has tracked changes")
    if (
        builder["revision"] != integration_revision
        or builder["branch"] != integration_branch
    ):
        raise ValueError("manifest builder Git identity differs from integration target")
    required_revisions = {
        formal_revision,
        bootstrap_revision,
        integration_revision,
        *bootstrap_prerequisites,
    }
    missing = sorted(
        revision
        for revision in required_revisions
        if not _git_object_exists(repository, revision)
    )
    if missing:
        raise ValueError(f"manifest builder is missing Git objects: {missing}")
    for prerequisite in bootstrap_prerequisites:
        if not _git_is_ancestor(repository, prerequisite, bootstrap_revision):
            raise ValueError("bootstrap prerequisite is not an ancestor of its head")
    if not _git_is_ancestor(repository, bootstrap_revision, integration_revision):
        raise ValueError("integration target does not descend from bootstrap head")

    bootstrap = _bundle_manifest_row(
        bootstrap_bundle,
        repository=repository,
        name="bootstrap",
        expected_revision=bootstrap_revision,
        expected_ref=bootstrap_ref,
        expected_prerequisites=bootstrap_prerequisites,
    )
    integration = _bundle_manifest_row(
        integration_bundle,
        repository=repository,
        name="integration",
        expected_revision=integration_revision,
        expected_ref=integration_ref,
        expected_prerequisites=[bootstrap_revision],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": MANIFEST_ROLE,
        "status": "prepared",
        "created_at": _utc_now(),
        "builder_git": {
            key: builder[key]
            for key in ("path", "revision", "tree", "branch", "tracked_dirty")
        },
        "formal_repository": {
            "path": _require_absolute_path(
                formal_repository_path,
                label="formal repository path",
            ),
            "revision": formal_revision,
            "tree": _git_text(repository, "rev-parse", f"{formal_revision}^{{tree}}").lower(),
            "branch": formal_branch,
        },
        "bootstrap_bundle": bootstrap,
        "integration_bundle": integration,
        "target": {
            "revision": integration_revision,
            "tree": _git_text(
                repository,
                "rev-parse",
                f"{integration_revision}^{{tree}}",
            ).lower(),
            "source_branch": integration_branch,
            "formal_branch_after_apply": formal_branch,
        },
        "active_training_guard": {
            "quality_bridge_root": _require_absolute_path(
                quality_bridge_root,
                label="quality bridge root",
            ),
            "execution_status_relative_path": "reports/execution_status.json",
            "pair_monitor_relative_path": "pair_monitor.json",
            "terminal_statuses": list(DEFAULT_TERMINAL_STATUSES),
            "blocked_process_markers": list(DEFAULT_BLOCKED_PROCESS_MARKERS),
        },
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
    }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} schema differs")


def validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "role",
            "status",
            "created_at",
            "builder_git",
            "formal_repository",
            "bootstrap_bundle",
            "integration_bundle",
            "target",
            "active_training_guard",
            "authorization_boundary",
        },
        label="post-bridge deployment manifest",
    )
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("role") != MANIFEST_ROLE
        or payload.get("status") != "prepared"
        or payload.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("post-bridge deployment manifest identity differs")
    formal = payload.get("formal_repository")
    target = payload.get("target")
    guard = payload.get("active_training_guard")
    builder = payload.get("builder_git")
    if not all(isinstance(value, dict) for value in (formal, target, guard, builder)):
        raise ValueError("post-bridge deployment manifest sections are malformed")
    _require_exact_keys(
        builder,
        {"path", "revision", "tree", "branch", "tracked_dirty"},
        label="post-bridge manifest builder Git",
    )
    _require_exact_keys(
        formal,
        {"path", "revision", "tree", "branch"},
        label="formal repository manifest",
    )
    _require_exact_keys(
        target,
        {"revision", "tree", "source_branch", "formal_branch_after_apply"},
        label="post-bridge target manifest",
    )
    _require_exact_keys(
        guard,
        {
            "quality_bridge_root",
            "execution_status_relative_path",
            "pair_monitor_relative_path",
            "terminal_statuses",
            "blocked_process_markers",
        },
        label="active training guard manifest",
    )
    if (
        formal.get("branch") != FORMAL_BRANCH
        or target.get("formal_branch_after_apply") != FORMAL_BRANCH
        or not _is_object_id(formal.get("revision"))
        or not _is_object_id(formal.get("tree"))
        or not _is_object_id(target.get("revision"))
        or not _is_object_id(target.get("tree"))
        or not isinstance(target.get("source_branch"), str)
        or not target["source_branch"]
        or not _is_object_id(builder.get("revision"))
        or not _is_object_id(builder.get("tree"))
        or not isinstance(builder.get("path"), str)
        or not builder["path"]
        or builder.get("tracked_dirty") is not False
        or builder.get("revision") != target.get("revision")
        or builder.get("tree") != target.get("tree")
        or builder.get("branch") != target.get("source_branch")
    ):
        raise ValueError("post-bridge repository or target identity is invalid")
    _require_absolute_path(str(formal.get("path", "")), label="formal repository path")
    _require_absolute_path(
        str(guard.get("quality_bridge_root", "")),
        label="quality bridge root",
    )
    if (
        guard.get("execution_status_relative_path") != "reports/execution_status.json"
        or guard.get("pair_monitor_relative_path") != "pair_monitor.json"
        or guard.get("terminal_statuses") != DEFAULT_TERMINAL_STATUSES
        or guard.get("blocked_process_markers") != DEFAULT_BLOCKED_PROCESS_MARKERS
    ):
        raise ValueError("active training guard contract differs")

    for name in ("bootstrap_bundle", "integration_bundle"):
        row = payload.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"{name} manifest is malformed")
        _require_exact_keys(
            row,
            {
                "name",
                "filename",
                "bytes",
                "sha256",
                "advertised_heads",
                "prerequisites",
                "builder_verification",
            },
            label=name,
        )
        if (
            row.get("name") != name.removesuffix("_bundle")
            or not isinstance(row.get("filename"), str)
            or Path(row["filename"]).name != row["filename"]
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or not _is_sha256(row.get("sha256"))
            or not isinstance(row.get("advertised_heads"), list)
            or len(row["advertised_heads"]) != 1
            or not isinstance(row.get("prerequisites"), list)
            or row.get("builder_verification", {}).get("status") != "pass"
        ):
            raise ValueError(f"{name} identity is invalid")
        head = row["advertised_heads"][0]
        if (
            not isinstance(head, dict)
            or set(head) != {"revision", "ref"}
            or not _is_object_id(head.get("revision"))
            or not isinstance(head.get("ref"), str)
            or not head["ref"].startswith("refs/heads/")
            or any(not _is_object_id(value) for value in row["prerequisites"])
            or row["prerequisites"] != sorted(set(row["prerequisites"]))
        ):
            raise ValueError(f"{name} Git contract is invalid")

    bootstrap = payload["bootstrap_bundle"]
    integration = payload["integration_bundle"]
    if (
        len(bootstrap["prerequisites"]) != 2
        or formal["revision"] not in bootstrap["prerequisites"]
        or integration["prerequisites"]
        != [bootstrap["advertised_heads"][0]["revision"]]
        or target["revision"] != integration["advertised_heads"][0]["revision"]
        or target["source_branch"]
        != integration["advertised_heads"][0]["ref"].removeprefix("refs/heads/")
    ):
        raise ValueError("post-bridge two-stage bundle chain differs")
    return dict(payload)


def load_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256.lower():
        raise ValueError("post-bridge deployment manifest SHA256 differs")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("post-bridge deployment manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("post-bridge deployment manifest must be a JSON object")
    return validate_manifest(payload), identity


def _verify_runtime_bundle(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    identity = file_identity(path)
    if (
        identity["bytes"] != expected["bytes"]
        or identity["sha256"] != expected["sha256"]
        or path.name != expected["filename"]
    ):
        raise ValueError(f"{expected['name']} runtime bundle identity differs")
    heads = bundle_heads(path)
    prerequisites = bundle_prerequisites(path)
    if (
        heads != expected["advertised_heads"]
        or prerequisites != expected["prerequisites"]
    ):
        raise ValueError(f"{expected['name']} runtime bundle Git contract differs")
    return {
        **identity,
        "advertised_heads": heads,
        "prerequisites": prerequisites,
    }


def _expected_formal_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    formal = manifest["formal_repository"]
    return {
        "revision": formal["revision"],
        "tree": formal["tree"],
        "branch": formal["branch"],
        "tracked_dirty": False,
    }


def _require_formal_state(
    state: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> None:
    expected = _expected_formal_state(manifest)
    observed = {key: state.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"formal repository Git identity differs: {observed}")


def _nul_paths(payload: bytes) -> list[str]:
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in payload.split(b"\0")
        if value
    ]


def _target_added_paths(
    repository: Path,
    *,
    current_revision: str,
    target_revision: str,
) -> list[str]:
    return _nul_paths(
        _git_bytes(
            repository,
            "diff",
            "--no-renames",
            "--diff-filter=A",
            "--name-only",
            "-z",
            current_revision,
            target_revision,
        )
    )


def find_untracked_conflicts(
    repository: Path,
    *,
    current_revision: str,
    target_added_paths: Sequence[str],
) -> dict[str, Any]:
    root = repository.resolve()
    current_tracked = set(
        _nul_paths(
            _git_bytes(
                root,
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                current_revision,
            )
        )
    )
    conflicts: set[str] = set()
    chunk_size = 256
    for start in range(0, len(target_added_paths), chunk_size):
        chunk = list(target_added_paths[start : start + chunk_size])
        if chunk:
            conflicts.update(
                _nul_paths(
                    _git_bytes(
                        root,
                        "ls-files",
                        "--others",
                        "-z",
                        "--",
                        *chunk,
                    )
                )
            )
    for target_path in target_added_paths:
        parts = PurePosixPath(target_path).parts
        for depth in range(len(parts) - 1, 0, -1):
            parent = PurePosixPath(*parts[:depth]).as_posix()
            local_parent = root.joinpath(*parts[:depth])
            if local_parent.is_symlink() or (
                local_parent.exists() and not local_parent.is_dir()
            ):
                if parent not in current_tracked:
                    conflicts.add(parent)
                break
            if local_parent.is_dir():
                break
    return {
        "status": "conflict" if conflicts else "pass",
        "target_added_path_count": len(target_added_paths),
        "conflict_count": len(conflicts),
        "conflicts": sorted(conflicts),
    }


def _fetch_bundle(
    repository: Path,
    *,
    bundle: Path,
    advertised_ref: str,
    expected_revision: str,
) -> str:
    _git(
        repository,
        "fetch",
        "--no-tags",
        str(bundle.resolve()),
        advertised_ref,
    )
    fetched = _git_text(repository, "rev-parse", "FETCH_HEAD").lower()
    if fetched != expected_revision:
        raise ValueError("fetched bundle head differs")
    return fetched


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def preflight_deployment(
    *,
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    formal_repository: Path,
    bootstrap_bundle: Path,
    integration_bundle: Path,
    temporary_parent: Path | None = None,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest)
    formal_repository = formal_repository.resolve()
    if formal_repository.as_posix() != manifest["formal_repository"]["path"]:
        raise ValueError("formal repository path differs from manifest")
    bootstrap = _verify_runtime_bundle(
        bootstrap_bundle,
        expected=manifest["bootstrap_bundle"],
    )
    integration = _verify_runtime_bundle(
        integration_bundle,
        expected=manifest["integration_bundle"],
    )
    before = git_state(formal_repository)
    _require_formal_state(before, manifest=manifest)

    bootstrap_prerequisites = manifest["bootstrap_bundle"]["prerequisites"]
    bootstrap_presence = {
        revision: _git_object_exists(formal_repository, revision)
        for revision in bootstrap_prerequisites
    }
    missing_bootstrap = [
        revision for revision, present in bootstrap_presence.items() if not present
    ]
    if missing_bootstrap:
        raise ValueError(
            f"formal repository is missing bootstrap prerequisites: {missing_bootstrap}"
        )
    integration_prerequisite = manifest["integration_bundle"]["prerequisites"][0]
    direct_prerequisite_present = _git_object_exists(
        formal_repository,
        integration_prerequisite,
    )
    bootstrap_verification = verify_bundle(
        bootstrap_bundle,
        repository=formal_repository,
    )

    temporary_parent_path = temporary_parent.resolve() if temporary_parent else None
    if temporary_parent_path is not None:
        temporary_parent_path.mkdir(parents=True, exist_ok=True)
        if _path_is_within(temporary_parent_path, formal_repository):
            raise ValueError("isolated preflight parent cannot be inside formal repository")
    with tempfile.TemporaryDirectory(
        dir=temporary_parent_path,
        prefix="cofitok-postbridge-preflight-",
    ) as temporary:
        isolated = Path(temporary) / "repository"
        _run(
            [
                "git",
                "clone",
                "--no-local",
                "--no-hardlinks",
                "--no-checkout",
                str(formal_repository),
                str(isolated),
            ]
        )
        _git(
            isolated,
            "checkout",
            "--detach",
            manifest["formal_repository"]["revision"],
        )
        _fetch_bundle(
            isolated,
            bundle=bootstrap_bundle,
            advertised_ref=manifest["bootstrap_bundle"]["advertised_heads"][0][
                "ref"
            ],
            expected_revision=manifest["bootstrap_bundle"]["advertised_heads"][0][
                "revision"
            ],
        )
        integration_verification = verify_bundle(
            integration_bundle,
            repository=isolated,
        )
        _fetch_bundle(
            isolated,
            bundle=integration_bundle,
            advertised_ref=manifest["integration_bundle"]["advertised_heads"][0][
                "ref"
            ],
            expected_revision=manifest["target"]["revision"],
        )
        target_tree = _git_text(
            isolated,
            "rev-parse",
            f"{manifest['target']['revision']}^{{tree}}",
        ).lower()
        if target_tree != manifest["target"]["tree"]:
            raise ValueError("isolated preflight target tree differs")
        if not _git_is_ancestor(
            isolated,
            manifest["formal_repository"]["revision"],
            manifest["target"]["revision"],
        ):
            raise ValueError("post-bridge target is not a formal-repository fast-forward")
        fsck = _git(isolated, "fsck", "--no-dangling")
        target_added_paths = _target_added_paths(
            isolated,
            current_revision=manifest["formal_repository"]["revision"],
            target_revision=manifest["target"]["revision"],
        )
        isolated_evidence = {
            "clone_mode": "no_local_no_hardlinks_no_checkout",
            "bootstrap_head": manifest["bootstrap_bundle"]["advertised_heads"][0][
                "revision"
            ],
            "integration_head": manifest["target"]["revision"],
            "target_tree": target_tree,
            "git_fsck_no_dangling": "pass",
            "git_fsck_stdout": str(fsck.stdout).strip(),
            "git_fsck_stderr": str(fsck.stderr).strip(),
        }

    conflict_scan = find_untracked_conflicts(
        formal_repository,
        current_revision=manifest["formal_repository"]["revision"],
        target_added_paths=target_added_paths,
    )
    if conflict_scan["status"] != "pass":
        raise ValueError(
            f"formal repository has target-added untracked conflicts: {conflict_scan['conflicts']}"
        )
    after = git_state(formal_repository)
    immutable_fields = (
        "revision",
        "tree",
        "branch",
        "tracked_dirty",
        "tracked_status_rows",
        "tracked_status_sha256",
        "full_status_rows",
        "full_status_sha256",
        "refs_sha256",
        "object_store_sha256",
    )
    if any(before[field] != after[field] for field in immutable_fields):
        raise ValueError("formal repository changed during preflight")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass",
        "mode": "preflight",
        "created_at": _utc_now(),
        "manifest": dict(manifest_identity),
        "formal_repository": {
            "before": before,
            "after": after,
            "unchanged": True,
            "fetch_performed": False,
            "fast_forward_performed": False,
        },
        "bundle_chain": {
            "bootstrap": {
                **bootstrap,
                "formal_prerequisite_presence": bootstrap_presence,
                "verification": bootstrap_verification,
            },
            "integration": {
                **integration,
                "prerequisite_present_in_formal_before_bootstrap": direct_prerequisite_present,
                "direct_formal_verify_possible_before_bootstrap": direct_prerequisite_present,
                "bootstrap_required": not direct_prerequisite_present,
                "isolated_verification_after_bootstrap": integration_verification,
            },
        },
        "isolated_preflight": isolated_evidence,
        "untracked_target_conflicts": conflict_scan,
        "target_added_paths": target_added_paths,
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "effects": {
            "formal_repository_modified": False,
            "training_or_controller_process_modified": False,
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "training_sampling_or_evaluation_launched": False,
        },
    }


def _read_stable_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before = path.read_bytes()
    after = path.read_bytes()
    if before != after:
        raise ValueError(f"{label} changed while reading")
    try:
        payload = json.loads(before.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, {
        "path": path.resolve().as_posix(),
        "bytes": len(before),
        "sha256": _sha256_bytes(before),
    }


def _integer_pids(values: Iterable[Any]) -> list[int]:
    pids: set[int] = set()
    for value in values:
        if type(value) is int and value > 0:
            pids.add(value)
    return sorted(pids)


def _known_training_pids(pair_monitor: Mapping[str, Any]) -> list[int]:
    gpu = pair_monitor.get("gpu_contention")
    current = gpu.get("current") if isinstance(gpu, dict) else None
    if not isinstance(current, dict):
        return []
    values: list[Any] = []
    for key in ("training_process_pids", "trainer_pids", "runbook_process_pids"):
        candidate = current.get(key)
        if isinstance(candidate, list):
            values.extend(candidate)
    return _integer_pids(values)


def _proc_cmdline(proc_root: Path, pid: int) -> str | None:
    path = proc_root / str(pid) / "cmdline"
    if not path.is_file():
        return None
    return " ".join(
        field.decode("utf-8", errors="replace")
        for field in path.read_bytes().split(b"\0")
        if field
    )


def _matching_live_processes(
    *,
    proc_root: Path,
    quality_bridge_root: str,
    markers: Sequence[str],
) -> list[dict[str, Any]]:
    if not proc_root.is_dir():
        return []
    matches: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        pid = int(entry.name)
        cmdline = _proc_cmdline(proc_root, pid)
        if (
            cmdline
            and quality_bridge_root in cmdline
            and any(marker in cmdline for marker in markers)
        ):
            matches.append({"pid": pid, "cmdline": cmdline})
    return sorted(matches, key=lambda row: row["pid"])


def require_inactive_bound_training(
    manifest: Mapping[str, Any],
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    if proc_root.is_symlink() or not proc_root.is_dir():
        raise ValueError("process filesystem is unavailable for apply guard")
    guard = manifest["active_training_guard"]
    root = Path(guard["quality_bridge_root"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("quality-bridge root is unavailable for apply guard")
    execution_path = root.joinpath(*PurePosixPath(guard["execution_status_relative_path"]).parts)
    pair_path = root.joinpath(*PurePosixPath(guard["pair_monitor_relative_path"]).parts)
    if (
        execution_path.is_symlink()
        or pair_path.is_symlink()
        or not execution_path.is_file()
        or not pair_path.is_file()
    ):
        raise ValueError("active-training guard sources are missing")
    execution, execution_identity = _read_stable_json(
        execution_path,
        label="quality-bridge execution status",
    )
    pair, pair_identity = _read_stable_json(
        pair_path,
        label="quality-bridge pair monitor",
    )
    terminal = set(guard["terminal_statuses"])
    execution_status = str(execution.get("status", "")).lower()
    pair_status = str(pair.get("status", "")).lower()
    if execution_status not in terminal or pair_status not in terminal:
        raise ValueError(
            "post-bridge apply is prohibited while quality-bridge status is non-terminal: "
            f"execution={execution_status!r}, pair={pair_status!r}"
        )
    controller_pid = execution.get("pid")
    candidate_pids = _integer_pids([controller_pid, *_known_training_pids(pair)])
    live_known = [
        {"pid": pid, "cmdline": _proc_cmdline(proc_root, pid)}
        for pid in candidate_pids
        if (proc_root / str(pid)).is_dir()
    ]
    scanned = _matching_live_processes(
        proc_root=proc_root,
        quality_bridge_root=guard["quality_bridge_root"],
        markers=guard["blocked_process_markers"],
    )
    live_by_pid = {row["pid"]: row for row in [*live_known, *scanned]}
    if live_by_pid:
        raise ValueError(
            "post-bridge apply is prohibited while bound trainer/controller processes are active: "
            f"{sorted(live_by_pid)}"
        )
    return {
        "status": "pass",
        "quality_bridge_root": guard["quality_bridge_root"],
        "execution_status": execution_status,
        "pair_status": pair_status,
        "execution_status_source": execution_identity,
        "pair_monitor_source": pair_identity,
        "known_candidate_pids": candidate_pids,
        "live_bound_processes": [],
        "process_signals_sent": False,
    }


def apply_deployment(
    *,
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    formal_repository: Path,
    bootstrap_bundle: Path,
    integration_bundle: Path,
    confirm_target_revision: str,
    proc_root: Path = Path("/proc"),
    temporary_parent: Path | None = None,
) -> dict[str, Any]:
    if confirm_target_revision != manifest["target"]["revision"]:
        raise ValueError("explicit apply target confirmation differs")
    preflight = preflight_deployment(
        manifest=manifest,
        manifest_identity=manifest_identity,
        formal_repository=formal_repository,
        bootstrap_bundle=bootstrap_bundle,
        integration_bundle=integration_bundle,
        temporary_parent=temporary_parent,
    )
    inactivity_before_fetch = require_inactive_bound_training(
        manifest,
        proc_root=proc_root,
    )
    formal_repository = formal_repository.resolve()
    before = git_state(formal_repository)
    _require_formal_state(before, manifest=manifest)
    conflicts = find_untracked_conflicts(
        formal_repository,
        current_revision=manifest["formal_repository"]["revision"],
        target_added_paths=preflight["target_added_paths"],
    )
    if conflicts["status"] != "pass":
        raise ValueError(
            f"formal repository has target-added untracked conflicts: {conflicts['conflicts']}"
        )

    verify_bundle(bootstrap_bundle, repository=formal_repository)
    bootstrap_head = manifest["bootstrap_bundle"]["advertised_heads"][0]
    _fetch_bundle(
        formal_repository,
        bundle=bootstrap_bundle,
        advertised_ref=bootstrap_head["ref"],
        expected_revision=bootstrap_head["revision"],
    )
    verify_bundle(integration_bundle, repository=formal_repository)
    integration_head = manifest["integration_bundle"]["advertised_heads"][0]
    _fetch_bundle(
        formal_repository,
        bundle=integration_bundle,
        advertised_ref=integration_head["ref"],
        expected_revision=integration_head["revision"],
    )
    before_merge = git_state(formal_repository)
    if (
        before_merge["revision"] != manifest["formal_repository"]["revision"]
        or before_merge["branch"] != manifest["formal_repository"]["branch"]
        or before_merge["tracked_dirty"]
    ):
        raise ValueError("formal repository changed before fast-forward")
    inactivity_before_merge = require_inactive_bound_training(
        manifest,
        proc_root=proc_root,
    )
    conflicts_before_merge = find_untracked_conflicts(
        formal_repository,
        current_revision=manifest["formal_repository"]["revision"],
        target_added_paths=preflight["target_added_paths"],
    )
    if conflicts_before_merge["status"] != "pass":
        raise ValueError("formal repository conflicts appeared before fast-forward")
    if not _git_is_ancestor(
        formal_repository,
        manifest["formal_repository"]["revision"],
        manifest["target"]["revision"],
    ):
        raise ValueError("target is not a fast-forward after bundle fetch")
    _git(
        formal_repository,
        "merge",
        "--ff-only",
        manifest["target"]["revision"],
    )
    after = git_state(formal_repository)
    expected_after = {
        "revision": manifest["target"]["revision"],
        "tree": manifest["target"]["tree"],
        "branch": manifest["target"]["formal_branch_after_apply"],
        "tracked_dirty": False,
    }
    if {key: after[key] for key in expected_after} != expected_after:
        raise ValueError("formal repository post-apply identity differs")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass",
        "mode": "apply",
        "created_at": _utc_now(),
        "manifest": dict(manifest_identity),
        "preflight": preflight,
        "active_training_guard": {
            "before_fetch": inactivity_before_fetch,
            "before_fast_forward": inactivity_before_merge,
        },
        "formal_repository": {
            "before": before,
            "after_fetch_before_fast_forward": before_merge,
            "after": after,
            "fetch_performed": True,
            "fast_forward_performed": True,
        },
        "untracked_target_conflicts": conflicts_before_merge,
        "authorization_boundary": dict(AUTHORIZATION_BOUNDARY),
        "effects": {
            "formal_repository_fast_forwarded": True,
            "training_or_controller_process_modified": False,
            "processes_signaled": False,
            "gpu_queried_or_allocated": False,
            "training_sampling_or_evaluation_launched": False,
            "promotion_release_or_full_300k_authorized": False,
        },
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
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
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or explicitly apply the fail-closed two-stage post-bridge "
            "Git bundle chain. The default mode is read-only preflight."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--formal-repository", type=Path, required=True)
    parser.add_argument("--bootstrap-bundle", type=Path, required=True)
    parser.add_argument("--integration-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "apply"),
        default="preflight",
    )
    parser.add_argument("--confirm-target-revision")
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--temporary-parent", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    formal_repository = args.formal_repository.resolve()
    output = args.output.resolve()
    if _path_is_within(output, formal_repository):
        raise ValueError("deployment report output cannot be inside formal repository")
    manifest, manifest_identity = load_manifest(
        args.manifest,
        expected_sha256=args.expected_manifest_sha256,
    )
    if args.mode == "apply":
        if not args.expected_manifest_sha256:
            raise ValueError("apply requires --expected-manifest-sha256")
        if not args.confirm_target_revision:
            raise ValueError("apply requires --confirm-target-revision")
        report = apply_deployment(
            manifest=manifest,
            manifest_identity=manifest_identity,
            formal_repository=formal_repository,
            bootstrap_bundle=args.bootstrap_bundle,
            integration_bundle=args.integration_bundle,
            confirm_target_revision=args.confirm_target_revision,
            proc_root=args.proc_root,
            temporary_parent=args.temporary_parent,
        )
    else:
        if args.confirm_target_revision is not None:
            raise ValueError("preflight does not accept apply target confirmation")
        report = preflight_deployment(
            manifest=manifest,
            manifest_identity=manifest_identity,
            formal_repository=formal_repository,
            bootstrap_bundle=args.bootstrap_bundle,
            integration_bundle=args.integration_bundle,
            temporary_parent=args.temporary_parent,
        )
    write_json_atomic(output, report)
    print(output)


if __name__ == "__main__":
    main()
