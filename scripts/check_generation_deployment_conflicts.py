from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath


CONFLICT_EXIT_CODE = 76
PATHSPEC_CHUNK_SIZE = 256


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _nul_paths(payload: bytes) -> list[str]:
    return [
        value.decode("utf-8", errors="surrogateescape")
        for value in payload.split(b"\0")
        if value
    ]


def find_untracked_target_conflicts(
    repository: str | Path,
    *,
    current_commit: str,
    target_commit: str,
) -> dict:
    root = Path(repository).resolve()
    added_paths = _nul_paths(
        _git(
            root,
            "diff",
            "--no-renames",
            "--diff-filter=A",
            "--name-only",
            "-z",
            current_commit,
            target_commit,
        )
    )
    current_tracked = set(
        _nul_paths(
            _git(
                root,
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                current_commit,
            )
        )
    )
    conflicts: set[str] = set()
    for start in range(0, len(added_paths), PATHSPEC_CHUNK_SIZE):
        chunk = added_paths[start : start + PATHSPEC_CHUNK_SIZE]
        if chunk:
            conflicts.update(
                _nul_paths(
                    _git(
                        root,
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "-z",
                        "--",
                        *chunk,
                    )
                )
            )

    for target_path in added_paths:
        parts = PurePosixPath(target_path).parts
        for depth in range(len(parts) - 1, 0, -1):
            parent = PurePosixPath(*parts[:depth]).as_posix()
            local_parent = root.joinpath(*parts[:depth])
            parent_blocks = local_parent.is_symlink() or (
                local_parent.exists() and not local_parent.is_dir()
            )
            if parent_blocks:
                if parent not in current_tracked:
                    conflicts.add(parent)
                break
            if local_parent.is_dir():
                break

    return {
        "schema_version": 1,
        "status": "conflict" if conflicts else "pass",
        "repository": root.as_posix(),
        "current_commit": current_commit,
        "target_commit": target_commit,
        "target_added_path_count": len(added_paths),
        "pathspec_chunk_size": PATHSPEC_CHUNK_SIZE,
        "conflict_count": len(conflicts),
        "conflicts": sorted(conflicts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check only target-added paths for remote untracked conflicts."
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--current-commit", required=True)
    parser.add_argument("--target-commit", required=True)
    args = parser.parse_args()
    report = find_untracked_target_conflicts(
        args.repository,
        current_commit=args.current_commit,
        target_commit=args.target_commit,
    )
    print(json.dumps(report, sort_keys=True))
    if report["conflict_count"]:
        raise SystemExit(CONFLICT_EXIT_CODE)


if __name__ == "__main__":
    main()
