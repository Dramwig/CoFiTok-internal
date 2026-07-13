from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from cofitok.reporting import git_provenance, write_json_report


SCHEMA_VERSION = 1
ROLE = "generation_runbook_syntax_check"


def tracked_runbooks(project_root: str | Path) -> list[Path]:
    root = Path(project_root).resolve()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "artifacts/runbooks"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = sorted(
        value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value and value.endswith(b".sh")
    )
    return [root.joinpath(*Path(relative).parts) for relative in relative_paths]


def check_runbook_syntax(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runbook_root = root / "artifacts" / "runbooks"
    runbooks = tracked_runbooks(root)
    if not runbooks:
        raise ValueError(f"no shell runbooks found under {runbook_root}")

    failures = []
    relative_paths = []
    for path in runbooks:
        relative = path.relative_to(root).as_posix()
        relative_paths.append(relative)
        result = subprocess.run(
            ["bash", "-n", path.as_posix()],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode:
            failures.append(
                {
                    "path": relative,
                    "exit_code": result.returncode,
                    "stderr": (result.stderr or "").strip(),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass" if not failures else "failed",
        "enumeration": "git_ls_files",
        "runbook_root": runbook_root.as_posix(),
        "discovered_count": len(relative_paths),
        "checked_count": len(relative_paths),
        "failed_count": len(failures),
        "runbooks": relative_paths,
        "failures": failures,
        "git": git_provenance(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Syntax-check every tracked generation shell runbook."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = check_runbook_syntax(args.project_root)
    write_json_report(args.output, report)
    print(args.output)
    if report["status"] != "pass":
        raise SystemExit("one or more generation runbooks failed bash -n")


if __name__ == "__main__":
    main()
