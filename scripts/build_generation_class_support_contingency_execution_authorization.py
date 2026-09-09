"""Build or replay a separately approved class-support execution authorization.

The builder cannot infer approval from a preparation.  It requires a distinct
user-created stage-approval record bound to the exact preparation and evaluator
Git identity.  Building this record does not train or sample and grants only
classifier evaluation of the frozen trees.
"""

# Parsed artifact type mismatches are schema-value failures, not API misuse.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation_class_support_contingency import (
    build_execution_authorization,
    stable_file_identity,
    validate_execution_authorization,
    validate_git_identity,
    validate_preparation,
)


def _read_json(
    path: Path, *, expected_sha256: str, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = stable_file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    if stable_file_identity(path) != identity:
        raise RuntimeError(f"{name} changed while it was read")
    return value, identity


def _checkout_identity(root: Path, *, script: Path | None = None) -> dict[str, Any]:
    project = root.resolve(strict=True)
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(project),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    )
    if status:
        raise ValueError(f"checkout must be completely clean: {project}")
    if script is not None:
        resolved = script.resolve(strict=True)
        relative = resolved.relative_to(project).as_posix()
        subprocess.run(
            ["git", "-C", str(project), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
        )
        if (
            subprocess.check_output(
                ["git", "-C", str(project), "show", f"HEAD:{relative}"]
            )
            != resolved.read_bytes()
        ):
            raise ValueError("authorization script differs from the exact HEAD blob")
    return validate_git_identity(
        {
            "revision": subprocess.check_output(
                ["git", "-C", str(project), "rev-parse", "HEAD^{commit}"], text=True
            ).strip(),
            "tree": subprocess.check_output(
                ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "branch": subprocess.check_output(
                ["git", "-C", str(project), "branch", "--show-current"], text=True
            ).strip(),
            "tracked_dirty": False,
        },
        "checkout Git",
    )


def _exclusive_write(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return stable_file_identity(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--preparation", type=Path, required=True)
        command.add_argument("--preparation-sha256", required=True)
        command.add_argument("--stage-approval", type=Path, required=True)
        command.add_argument("--stage-approval-sha256", required=True)
        command.add_argument("--evaluator-project-root", type=Path, required=True)
        command.add_argument("--authorization-project-root", type=Path, required=True)
        command.add_argument("--script", type=Path, required=True)
    commands.choices["build"].add_argument("--output", type=Path, required=True)
    commands.choices["validate"].add_argument(
        "--authorization", type=Path, required=True
    )
    commands.choices["validate"].add_argument("--authorization-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    preparation, preparation_id = _read_json(
        args.preparation,
        expected_sha256=args.preparation_sha256,
        name="class-support preparation",
    )
    prepared = validate_preparation(preparation)
    approval, approval_id = _read_json(
        args.stage_approval,
        expected_sha256=args.stage_approval_sha256,
        name="class-support stage approval",
    )
    expected = build_execution_authorization(
        preparation=prepared,
        preparation_identity=preparation_id,
        stage_approval=approval,
        stage_approval_identity=approval_id,
        evaluator_git=_checkout_identity(args.evaluator_project_root),
        authorization_git=_checkout_identity(
            args.authorization_project_root, script=args.script
        ),
    )
    if args.command == "build":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite authorization: {args.output}")
        identity = _exclusive_write(args.output, expected)
    else:
        actual, identity = _read_json(
            args.authorization,
            expected_sha256=args.authorization_sha256,
            name="class-support execution authorization",
        )
        validate_execution_authorization(
            actual,
            preparation=prepared,
            preparation_identity=preparation_id,
        )
        if actual != expected:
            raise ValueError("execution authorization differs from physical replay")
    print(
        json.dumps(
            {
                "command": args.command,
                "status": "pass",
                "authorization": identity,
                "evaluation_launch_allowed": True,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "full_training_launch_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
