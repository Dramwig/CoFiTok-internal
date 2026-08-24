from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.semantic_residual_alignment_probe import (
    build_execution_authorization,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact source-bound standing authorization for the four-arm "
            "semantic residual-alignment probe."
        )
    )
    for name in (
        "preparation",
        "standing_authorization",
        "quality_bridge_result",
        "source_postevaluation",
    ):
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path, required=True)
        parser.add_argument(f"--expected-{option}-sha256", required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--expected-runbook-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _git() -> dict[str, Any]:
    provenance = git_provenance(PROJECT_ROOT)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if provenance["tracked_dirty"] or status:
        raise ValueError("authorization builder requires a fully clean checkout")
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
    loaded: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for name in (
        "preparation",
        "standing_authorization",
        "quality_bridge_result",
        "source_postevaluation",
    ):
        loaded[name], identities[name] = _bound_json(
            getattr(args, name),
            expected_sha256=getattr(args, f"expected_{name}_sha256"),
            label=name.replace("_", " "),
        )
    runbook = reject_symlink_chain(args.runbook, name="residual-alignment runbook")
    runbook_identity = file_identity(runbook)
    if runbook_identity["sha256"] != args.expected_runbook_sha256:
        raise ValueError("residual-alignment runbook SHA256 differs")
    report = build_execution_authorization(
        preparation=loaded["preparation"],
        preparation_identity=identities["preparation"],
        standing_authorization=loaded["standing_authorization"],
        standing_authorization_identity=identities["standing_authorization"],
        quality_bridge_result=loaded["quality_bridge_result"],
        quality_bridge_result_identity=identities["quality_bridge_result"],
        source_postevaluation=loaded["source_postevaluation"],
        source_postevaluation_identity=identities["source_postevaluation"],
        runbook_identity=runbook_identity,
        authorization_git=_git(),
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
        expected_output_root=args.output_root,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
