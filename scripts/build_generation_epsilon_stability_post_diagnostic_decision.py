from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.quality_repair_decision import (
    SOURCE_NAMES,
    build_epsilon_stability_post_diagnostic_decision,
)
from cofitok.reporting import file_sha256, write_json_report


ARGUMENTS = {
    "epsilon_stability_sampling_result": "--epsilon-stability-sampling-result",
    "post_reconciliation_decision": "--post-reconciliation-decision",
    "cross_protocol_reconciliation": "--cross-protocol-reconciliation",
    "pair_monitor": "--pair-monitor",
    "training_exposure_report": "--training-exposure-report",
    "runtime_compute_fairness": "--runtime-compute-fairness",
    "terminal_system_claim_guard": "--terminal-system-claim-guard",
    "terminal_completion_audit": "--terminal-completion-audit",
    "terminal_route_supersession_receipt": (
        "--terminal-route-supersession-receipt"
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the CPU-only, permanently non-authorizing decision after the "
            "matched epsilon-stability sampling diagnostic."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    for destination, option in ARGUMENTS.items():
        parser.add_argument(option, type=Path, required=True, dest=destination)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reject_symlink_chain(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    current = absolute
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    return absolute


def _load_source(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _reject_symlink_chain(path, label=label)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return (
        {
            "path": source.as_posix(),
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
        },
        payload,
    )


def _git(project: Path) -> dict[str, Any]:
    root = _reject_symlink_chain(project, label="decision project")
    if not root.is_dir():
        raise FileNotFoundError(f"decision project is missing: {root}")

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    identity = {
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(
            run("status", "--porcelain", "--untracked-files=no")
        ),
    }
    if identity["tracked_dirty"]:
        raise ValueError("decision project has tracked changes")
    return identity


def main() -> None:
    args = _parse_args()
    output = _reject_symlink_chain(args.output, label="decision output")
    if output.exists():
        raise FileExistsError(f"post-diagnostic decision already exists: {output}")

    identities: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        identity, payload = _load_source(getattr(args, name), label=name)
        identities[name] = identity
        payloads[name] = payload
    report = build_epsilon_stability_post_diagnostic_decision(
        **payloads,
        source_identities=identities,
        builder_git=_git(args.project),
    )
    write_json_report(output, report)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "bytes": output.stat().st_size,
                "sha256": file_sha256(output),
                "status": report["status"],
                "terminal_status": report["terminal_status"],
                "decision": report["decision"],
                "training_launch_allowed": report["authorization_boundary"]
                ["training_launch_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
