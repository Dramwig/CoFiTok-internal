from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_probe import (
    validate_conditioning_ranking_probe_execution_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _embedded_source(
    authorization: Mapping[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = authorization.get("source_reports")
    descriptor = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"execution authorization source is missing: {name}")
    expected = dict(descriptor)
    source = reject_symlink_chain(
        Path(str(expected.get("path", ""))),
        name=f"execution authorization {name}",
    ).resolve()
    identity = file_identity(source)
    if identity != expected:
        raise ValueError(f"execution authorization source changed: {name}")
    return read_json_object(source, name=name), identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and verify the source-bound standing authorization receipt for "
            "the four-arm 1K class-conditioning-ranking probe."
        )
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization_path = reject_symlink_chain(
        args.authorization,
        name="conditioning-ranking execution authorization",
    ).resolve()
    authorization_identity = file_identity(authorization_path)
    if authorization_identity["sha256"] != args.expected_authorization_sha256:
        raise ValueError("conditioning-ranking execution authorization SHA256 differs")
    authorization = read_json_object(
        authorization_path,
        name="conditioning-ranking execution authorization",
    )
    preparation, preparation_identity = _embedded_source(
        authorization,
        name="preparation",
    )
    standing, standing_identity = _embedded_source(
        authorization,
        name="standing_authorization",
    )
    followup, followup_identity = _embedded_source(
        authorization,
        name="quality_bridge_followup_decision",
    )
    terminal, terminal_identity = _embedded_source(
        authorization,
        name="terminal_system_claim_guard",
    )
    _embedded_source(
        authorization,
        name="quality_bridge_result",
    )
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("conditioning-ranking preparation SHA256 differs")
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    validate_conditioning_ranking_probe_execution_authorization(
        authorization,
        preparation=preparation,
        preparation_identity=preparation_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        followup_decision=followup,
        followup_decision_identity=followup_identity,
        terminal_system_guard=terminal,
        terminal_system_guard_identity=terminal_identity,
        authorization_git=git_provenance(PROJECT_ROOT),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.expected_output_root,
    )
    print(authorization_identity["sha256"])


if __name__ == "__main__":
    main()
