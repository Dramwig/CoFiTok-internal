"""Clone and pin external baseline repositories for CoFiTok.

The script clones only registry entries marked ``clone_on_setup`` by default and
writes a small manifest with commit/license provenance. It never downloads
weights or runs training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("baselines/registry.json")
DEFAULT_REPO_ROOT = Path("/root/autodl-tmp/CoFiTok/baselines/repos")
DEFAULT_MANIFEST = Path(
    "docs/experiment_conditions/baselines_repo_manifest_2026-07-09.json"
)
LICENSE_PATTERNS = ("LICENSE*", "COPYING*", "NOTICE*")


class GitError(RuntimeError):
    """Raised when a git command fails."""


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    if "entries" not in registry or not isinstance(registry["entries"], list):
        raise ValueError(f"Registry has no entries list: {path}")
    return registry


def select_clone_entries(
    registry: dict[str, Any],
    aliases: set[str] | None = None,
    priorities: set[str] | None = None,
    clone_all_marked: bool = True,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        if entry.get("repo_type") != "external":
            continue
        if aliases is not None and entry["alias"] not in aliases:
            continue
        if priorities is not None and entry.get("priority") not in priorities:
            continue
        if clone_all_marked and aliases is None and priorities is None:
            if not entry.get("clone_on_setup", False):
                continue
        selected.append(entry)
    return selected


def clone_target(repo_root: Path, entry: dict[str, Any]) -> Path:
    alias = entry["alias"]
    if not re.fullmatch(r"[a-z0-9_]+", alias):
        raise ValueError(f"Unsafe baseline alias: {alias!r}")
    return repo_root / alias


def run_git(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        command = "git " + " ".join(args)
        raise GitError(f"{command} failed: {proc.stderr.strip()}")
    return proc


def git_output(args: list[str], cwd: Path, timeout: int = 120) -> str | None:
    proc = run_git(args, cwd=cwd, timeout=timeout, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def find_license_files(repo_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in LICENSE_PATTERNS:
        for path in repo_dir.glob(pattern):
            if path.is_file():
                found.append(path.name)
    return sorted(set(found))


def inspect_repo(entry: dict[str, Any], target: Path) -> dict[str, Any]:
    dirty = git_output(["status", "--short"], cwd=target) or ""
    return {
        "alias": entry["alias"],
        "priority": entry.get("priority"),
        "family": entry.get("family"),
        "method_name": entry.get("method_name"),
        "repo_url": entry.get("repo_url"),
        "target_path": str(target),
        "origin_url": git_output(["remote", "get-url", "origin"], cwd=target),
        "commit": git_output(["rev-parse", "HEAD"], cwd=target),
        "commit_date": git_output(["log", "-1", "--format=%cI"], cwd=target),
        "license_files": find_license_files(target),
        "dirty_status": dirty.splitlines(),
    }


def clone_or_inspect(
    entry: dict[str, Any],
    repo_root: Path,
    *,
    dry_run: bool = False,
    timeout: int = 600,
) -> dict[str, Any]:
    target = clone_target(repo_root, entry)
    record: dict[str, Any] = {
        "alias": entry["alias"],
        "priority": entry.get("priority"),
        "family": entry.get("family"),
        "method_name": entry.get("method_name"),
        "repo_url": entry.get("repo_url"),
        "target_path": str(target),
        "status": "unknown",
        "error": None,
    }

    if dry_run:
        record["status"] = "dry_run"
        record["clone_command"] = [
            "git",
            "clone",
            "--depth",
            "1",
            "--no-tags",
            entry["repo_url"],
            str(target),
        ]
        return record

    try:
        repo_root.mkdir(parents=True, exist_ok=True)
        if target.exists() and not (target / ".git").exists():
            raise GitError(f"Target exists but is not a git repository: {target}")

        if (target / ".git").exists():
            record.update(inspect_repo(entry, target))
            record["status"] = (
                "exists_dirty" if record.get("dirty_status") else "exists"
            )
            return record

        clone_command = [
            "clone",
            "--depth",
            "1",
            "--no-tags",
            entry["repo_url"],
            str(target),
        ]
        record["clone_command"] = ["git", *clone_command]
        run_git(clone_command, timeout=timeout)
        record.update(inspect_repo(entry, target))
        record["status"] = "cloned"
        return record
    except Exception as exc:  # noqa: BLE001 - each failure is recorded in manifest
        record["status"] = "failed"
        record["error"] = str(exc)
        return record


def build_manifest(
    registry_path: Path,
    registry: dict[str, Any],
    repo_root: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": registry.get("project", "CoFiTok"),
        "registry_path": str(registry_path),
        "repo_root": str(repo_root),
        "record_count": len(records),
        "failed_count": sum(1 for record in records if record["status"] == "failed"),
        "records": records,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--alias",
        action="append",
        default=None,
        help="Clone/inspect only this alias. Can be repeated.",
    )
    parser.add_argument(
        "--include-priority",
        action="append",
        default=None,
        choices=("P0", "P1", "P2"),
        help="Clone/inspect external entries with this priority. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Exit nonzero if any selected repository fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = load_registry(args.registry)
    aliases = set(args.alias) if args.alias else None
    priorities = set(args.include_priority) if args.include_priority else None
    entries = select_clone_entries(registry, aliases=aliases, priorities=priorities)

    records = [
        clone_or_inspect(
            entry,
            args.repo_root,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
        for entry in entries
    ]
    manifest = build_manifest(args.registry, registry, args.repo_root, records)
    write_manifest(args.manifest_output, manifest)

    for record in records:
        print(
            f"{record['alias']}: {record['status']} "
            f"{record.get('commit') or record.get('error') or ''}"
        )

    if args.require_all and manifest["failed_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
