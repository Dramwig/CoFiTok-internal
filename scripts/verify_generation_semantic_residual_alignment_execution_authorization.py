from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.semantic_residual_alignment_probe import (
    validate_execution_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    "preparation",
    "standing_authorization",
    "quality_bridge_result",
    "source_postevaluation",
    "runbook",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the semantic residual-alignment execution authorization."
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", required=True)
    return parser.parse_args()


def _embedded_identity(
    authorization: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    sources = authorization.get("source_reports")
    descriptor = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"authorization source is missing: {name}")
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=f"authorization source {name}",
    ).resolve()
    identity = file_identity(source)
    if identity != dict(descriptor):
        raise ValueError(f"authorization source changed: {name}")
    return identity


def _git() -> dict[str, Any]:
    provenance = git_provenance(PROJECT_ROOT)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if provenance["tracked_dirty"] or status:
        raise ValueError("authorization replay requires a fully clean checkout")
    return {
        **provenance,
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip(),
    }


def main() -> None:
    args = parse_args()
    authorization_path = reject_symlink_chain(
        args.authorization,
        name="residual-alignment authorization",
    ).resolve()
    authorization_identity = file_identity(authorization_path)
    if authorization_identity["sha256"] != args.expected_authorization_sha256:
        raise ValueError("residual-alignment authorization SHA256 differs")
    authorization = read_json_object(
        authorization_path,
        name="residual-alignment authorization",
    )
    identities = {
        name: _embedded_identity(authorization, name=name) for name in SOURCE_NAMES
    }
    payloads = {
        name: read_json_object(Path(identities[name]["path"]), name=name)
        for name in SOURCE_NAMES
        if name != "runbook"
    }
    validate_execution_authorization(
        authorization,
        preparation=payloads["preparation"],
        preparation_identity=identities["preparation"],
        standing_authorization=payloads["standing_authorization"],
        standing_authorization_identity=identities["standing_authorization"],
        quality_bridge_result=payloads["quality_bridge_result"],
        quality_bridge_result_identity=identities["quality_bridge_result"],
        source_postevaluation=payloads["source_postevaluation"],
        source_postevaluation_identity=identities["source_postevaluation"],
        runbook_identity=identities["runbook"],
        authorization_git=_git(),
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
        expected_output_root=args.expected_output_root,
    )
    print(authorization_identity["sha256"])


if __name__ == "__main__":
    main()
