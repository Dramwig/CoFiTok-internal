from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.quality_repair_decision import (
    POST_DIAGNOSTIC_AUTHORIZATION_BOUNDARY,
    POST_DIAGNOSTIC_VERIFICATION_SCHEMA,
    SOURCE_NAMES,
    validate_epsilon_stability_post_diagnostic_decision,
)
from cofitok.reporting import file_sha256, write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rehash and replay an epsilon-stability post-diagnostic decision."
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-builder-revision", required=True)
    parser.add_argument("--expected-builder-tree", required=True)
    parser.add_argument("--expected-builder-branch", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path, *, label: str) -> dict[str, Any]:
    source = path.absolute()
    current = source
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _rehash_source(identity: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity is malformed")
    path = Path(str(identity.get("path", "")))
    payload = _load(path, label=label)
    actual = {
        "path": path.absolute().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if actual != identity:
        raise ValueError(f"{label} source identity differs")
    return {"identity": actual, "payload": payload}


def main() -> None:
    args = _parse_args()
    decision_path = args.decision.absolute()
    decision = _load(decision_path, label="post-diagnostic decision")
    actual_sha256 = file_sha256(decision_path)
    if actual_sha256 != args.expected_decision_sha256:
        raise ValueError("post-diagnostic decision SHA256 differs")
    expected_git = {
        "revision": args.expected_builder_revision,
        "tree": args.expected_builder_tree,
        "branch": args.expected_builder_branch,
        "tracked_dirty": False,
    }
    if decision.get("builder_git") != expected_git:
        raise ValueError("post-diagnostic decision builder Git differs")
    source_identities = decision.get("source_identities")
    if not isinstance(source_identities, dict) or set(source_identities) != set(
        SOURCE_NAMES
    ):
        raise ValueError("post-diagnostic decision source set differs")
    sources = {
        name: _rehash_source(source_identities[name], label=name)
        for name in SOURCE_NAMES
    }
    validate_epsilon_stability_post_diagnostic_decision(
        decision,
        **{name: sources[name]["payload"] for name in SOURCE_NAMES},
        source_identities={
            name: sources[name]["identity"] for name in SOURCE_NAMES
        },
        builder_git=expected_git,
    )
    report = {
        "schema": POST_DIAGNOSTIC_VERIFICATION_SCHEMA,
        "status": "pass",
        "decision": {
            "path": decision_path.as_posix(),
            "bytes": decision_path.stat().st_size,
            "sha256": actual_sha256,
        },
        "builder_git": expected_git,
        "verified_source_identities": {
            name: sources[name]["identity"] for name in SOURCE_NAMES
        },
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "authorization_boundary": dict(POST_DIAGNOSTIC_AUTHORIZATION_BOUNDARY),
    }
    if args.output is not None:
        output = args.output.absolute()
        if output.exists():
            raise FileExistsError(
                f"post-diagnostic verification already exists: {output}"
            )
        write_json_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
