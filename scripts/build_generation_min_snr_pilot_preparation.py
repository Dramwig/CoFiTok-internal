from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.min_snr_pilot import (
    LEGACY_REVISION,
    METHODS,
    TRAINING_SEMANTIC_FILES,
    build_preparation,
)
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the non-authorizing matched Min-SNR pilot preparation."
    )
    parser.add_argument("--post-diagnostic-decision", required=True)
    parser.add_argument("--expected-post-diagnostic-decision-sha256", required=True)
    parser.add_argument("--pair-monitor", required=True)
    parser.add_argument("--legacy-cofitok-config", required=True)
    parser.add_argument("--legacy-dense-config", required=True)
    parser.add_argument("--pilot-cofitok-config", required=True)
    parser.add_argument("--pilot-dense-config", required=True)
    parser.add_argument("--legacy-cofitok-training-report", required=True)
    parser.add_argument("--legacy-dense-training-report", required=True)
    parser.add_argument("--legacy-cofitok-checkpoint-audit-50k", required=True)
    parser.add_argument("--legacy-dense-checkpoint-audit-50k", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _builder_git() -> dict[str, Any]:
    provenance = git_provenance(PROJECT_ROOT)
    return {**provenance, "tree": _git("rev-parse", "HEAD^{tree}")}


def _training_source_delta() -> dict[str, Any]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", LEGACY_REVISION, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    changed = sorted(
        line
        for line in _git(
            "diff",
            "--name-only",
            f"{LEGACY_REVISION}..HEAD",
            "--",
            *TRAINING_SEMANTIC_FILES,
        ).splitlines()
        if line
    )
    return {
        "base_revision": LEGACY_REVISION,
        "base_is_ancestor": ancestor,
        "changed_semantic_files": changed,
        "semantic_file_identities": {
            path: _identity(PROJECT_ROOT / path) for path in TRAINING_SEMANTIC_FILES
        },
    }


def main() -> None:
    args = parse_args()
    paths = {
        "post_diagnostic_decision": args.post_diagnostic_decision,
        "pair_monitor": args.pair_monitor,
        "legacy_cofitok_config": args.legacy_cofitok_config,
        "legacy_dense_config": args.legacy_dense_config,
        "pilot_cofitok_config": args.pilot_cofitok_config,
        "pilot_dense_config": args.pilot_dense_config,
        "legacy_cofitok_training_report": args.legacy_cofitok_training_report,
        "legacy_dense_training_report": args.legacy_dense_training_report,
        "legacy_cofitok_checkpoint_audit_50k": (
            args.legacy_cofitok_checkpoint_audit_50k
        ),
        "legacy_dense_checkpoint_audit_50k": (
            args.legacy_dense_checkpoint_audit_50k
        ),
    }
    identities = {name: _identity(path) for name, path in paths.items()}
    if (
        identities["post_diagnostic_decision"]["sha256"]
        != args.expected_post_diagnostic_decision_sha256
    ):
        raise ValueError("post-diagnostic decision SHA256 differs")
    report = build_preparation(
        post_diagnostic_decision=_read(args.post_diagnostic_decision),
        pair_monitor=_read(args.pair_monitor),
        legacy_configs={
            "cofitok": _read(args.legacy_cofitok_config),
            "dense_identity": _read(args.legacy_dense_config),
        },
        pilot_configs={
            "cofitok": _read(args.pilot_cofitok_config),
            "dense_identity": _read(args.pilot_dense_config),
        },
        legacy_training_reports={
            "cofitok": _read(args.legacy_cofitok_training_report),
            "dense_identity": _read(args.legacy_dense_training_report),
        },
        legacy_checkpoint_audits={
            "cofitok": _read(args.legacy_cofitok_checkpoint_audit_50k),
            "dense_identity": _read(args.legacy_dense_checkpoint_audit_50k),
        },
        source_identities=identities,
        builder_git=_builder_git(),
        training_source_delta=_training_source_delta(),
        output_root=args.output_root,
    )
    write_json_report(args.output, report)
    print(file_sha256(args.output))


if __name__ == "__main__":
    main()
