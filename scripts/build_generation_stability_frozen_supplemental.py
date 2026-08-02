from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.frozen_supplemental import (
    build_frozen_stability_supplemental_qualification,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _verify_nested_sources(report: dict[str, Any], *, label: str) -> None:
    nested = report.get("sources")
    if not isinstance(nested, dict) or not nested:
        raise ValueError(f"{label} report does not bind source identities")
    for name, identity in nested.items():
        if not isinstance(identity, dict) or _source(Path(str(identity.get("path", "")))) != identity:
            raise ValueError(f"{label} source changed: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the frozen stability gate with source-bound distribution "
            "support and later clean-checkout EMA rollout diagnostics."
        )
    )
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--posteval-verification", type=Path, required=True)
    parser.add_argument("--distribution-support", type=Path, required=True)
    parser.add_argument("--rollout-stability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-frozen-evaluation-revision", required=True)
    parser.add_argument("--expected-frozen-evaluation-branch", required=True)
    parser.add_argument("--expected-supplemental-revision", required=True)
    parser.add_argument("--expected-supplemental-branch", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def build_bound_report(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "promotion_gate": args.promotion_gate,
        "posteval_verification": args.posteval_verification,
        "distribution_support": args.distribution_support,
        "rollout_stability": args.rollout_stability,
    }
    sources = {name: _source(path) for name, path in paths.items()}
    reports = {name: _read(path) for name, path in paths.items()}
    posteval_source = reports["posteval_verification"].get("source")
    if (
        not isinstance(posteval_source, dict)
        or _source(Path(str(posteval_source.get("path", "")))) != posteval_source
    ):
        raise ValueError("post-evaluation verification source changed")
    _verify_nested_sources(reports["distribution_support"], label="distribution-support")
    _verify_nested_sources(reports["rollout_stability"], label="rollout-stability")
    report = build_frozen_stability_supplemental_qualification(
        promotion_gate=reports["promotion_gate"],
        posteval_verification=reports["posteval_verification"],
        distribution_support=reports["distribution_support"],
        rollout_stability=reports["rollout_stability"],
        sources=sources,
        builder_git=git_provenance(PROJECT_ROOT),
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
        expected_frozen_evaluation_revision=(
            args.expected_frozen_evaluation_revision
        ),
        expected_frozen_evaluation_branch=args.expected_frozen_evaluation_branch,
        expected_supplemental_revision=args.expected_supplemental_revision,
        expected_supplemental_branch=args.expected_supplemental_branch,
    )
    if {name: _source(path) for name, path in paths.items()} != sources:
        raise ValueError("frozen supplemental inputs changed while building")
    _verify_nested_sources(reports["distribution_support"], label="distribution-support")
    _verify_nested_sources(reports["rollout_stability"], label="rollout-stability")
    return report


def main() -> int:
    args = parse_args()
    with exclusive_output_lock(
        args.output,
        role="generation_stability_frozen_supplemental_qualification",
    ):
        report = build_bound_report(args)
        if args.output.exists():
            if not args.resume:
                raise FileExistsError(
                    "frozen supplemental report exists; pass --resume to replay it"
                )
            if _read(args.output) != report:
                raise ValueError(
                    "existing frozen supplemental report differs from exact replay"
                )
        else:
            write_json_report(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))
    return 2 if args.require_pass and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
