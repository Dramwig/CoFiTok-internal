from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.stability_scaling import (
    build_stability_scaling_decision,
    validate_stability_scaling_decision,
)
from cofitok.reporting import file_sha256, write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and rehash a rollout-stability decision before preparing "
            "a larger matched generation run."
        )
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument(
        "--expected-next-stage",
        choices=("matched_5k", "fresh_matched_50k_preparation"),
        default="fresh_matched_50k_preparation",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _verify_source(descriptor: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(descriptor["path"]))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"stability decision source is missing: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = file_sha256(path)
    if (
        actual_bytes != int(descriptor["bytes"])
        or actual_sha256 != descriptor["sha256"]
    ):
        raise ValueError(f"stability decision source identity differs: {path}")
    return {
        "path": path.resolve().as_posix(),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def main() -> None:
    args = _parse_args()
    decision_path = args.decision.resolve()
    if not decision_path.is_file():
        raise ValueError(f"stability decision is missing: {decision_path}")
    decision_sha256 = file_sha256(decision_path)
    if decision_sha256 != args.expected_decision_sha256:
        raise ValueError("stability decision SHA256 differs from the expected identity")
    decision = _load(decision_path)
    sources = decision.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("stability decision source provenance is missing")
    screening_descriptor = sources.get("screening_report")
    robust_descriptors = sources.get("robust_reports")
    if not isinstance(screening_descriptor, dict) or not isinstance(
        robust_descriptors, list
    ):
        raise ValueError("stability decision source descriptors are malformed")
    screening_source = _verify_source(screening_descriptor)
    robust_sources = [
        _verify_source(descriptor)
        for descriptor in robust_descriptors
    ]
    rebuilt = build_stability_scaling_decision(
        screening_report=_load(Path(screening_source["path"])),
        robust_reports=[
            _load(Path(descriptor["path"]))
            for descriptor in robust_sources
        ],
        next_stage=args.expected_next_stage,
    )
    declared_core = {
        key: value for key, value in decision.items() if key != "sources"
    }
    if declared_core != rebuilt:
        raise ValueError(
            "stability decision does not match its rehashed qualification sources"
        )
    evidence = validate_stability_scaling_decision(
        decision,
        expected_source_revision=args.expected_source_revision,
        expected_next_stage=args.expected_next_stage,
    )
    verified_sources = {
        "screening_report": screening_source,
        "robust_reports": robust_sources,
    }
    report = {
        **evidence,
        "decision_source": {
            "path": decision_path.as_posix(),
            "bytes": decision_path.stat().st_size,
            "sha256": decision_sha256,
        },
        "verified_sources": verified_sources,
    }
    if args.output is not None:
        write_json_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
