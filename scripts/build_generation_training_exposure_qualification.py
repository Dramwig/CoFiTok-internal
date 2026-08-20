from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.training_exposure_qualification import (
    CAPACITY_RESULT_SOURCE_NAMES,
    EXPOSURE_QUALIFICATION_OUTPUT_ROOT,
    build_training_exposure_qualification_preparation,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report

try:
    from scripts.build_generation_quality_bridge_followup_decision import (
        build_from_sources as replay_quality_bridge_followup_from_sources,
    )
except ModuleNotFoundError:
    from build_generation_quality_bridge_followup_decision import (
        build_from_sources as replay_quality_bridge_followup_from_sources,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--expected-followup-decision-sha256", required=True)
    parser.add_argument("--capacity-probe-result", type=Path, required=True)
    parser.add_argument("--expected-capacity-probe-result-sha256", required=True)
    parser.add_argument("--source-cofitok-config", type=Path, required=True)
    parser.add_argument("--source-dense-config", type=Path, required=True)
    parser.add_argument("--target-cofitok-config", type=Path, required=True)
    parser.add_argument("--target-dense-config", type=Path, required=True)
    parser.add_argument("--expected-followup-revision", required=True)
    parser.add_argument("--expected-followup-branch", required=True)
    parser.add_argument("--expected-capacity-revision", required=True)
    parser.add_argument("--expected-capacity-branch", required=True)
    parser.add_argument("--expected-preparation-revision", required=True)
    parser.add_argument("--expected-preparation-branch", required=True)
    parser.add_argument(
        "--output-root",
        default=EXPOSURE_QUALIFICATION_OUTPUT_ROOT,
    )


def _bound_json_source(
    value: object,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} identity is missing")
    expected = dict(value)
    actual = file_identity(str(value.get("path", "")))
    if actual != expected:
        raise ValueError(f"{label} identity differs")
    return read_json_object(actual["path"], name=label), actual


def replay_followup_decision(
    path: Path,
    *,
    expected_sha256: str,
    expected_revision: str,
    expected_branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("exposure-aware follow-up decision SHA256 differs")
    actual = read_json_object(path, name="exposure-aware follow-up decision")
    sources = actual.get("source_reports")
    if not isinstance(sources, Mapping) or set(sources) != {
        "quality_bridge_result",
        "milestones",
        "terminal_training_exposure",
    }:
        raise ValueError("exposure-aware follow-up source set differs")
    result = sources.get("quality_bridge_result")
    exposure = sources.get("terminal_training_exposure")
    if not isinstance(result, Mapping) or not isinstance(exposure, Mapping):
        raise ValueError("exposure-aware follow-up source identity is malformed")
    decision_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    expected = replay_quality_bridge_followup_from_sources(
        quality_bridge_result_path=Path(str(result.get("path", ""))),
        expected_quality_bridge_result_sha256=str(result.get("sha256", "")),
        training_exposure_report_path=Path(str(exposure.get("path", ""))),
        expected_training_exposure_report_sha256=str(
            exposure.get("sha256", "")
        ),
        decision_git=decision_git,
    )
    if actual != expected:
        raise ValueError("exposure-aware follow-up decision is not reproducible")
    return actual, identity


def replay_capacity_probe_result(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError("capacity probe result SHA256 differs")
    result = read_json_object(path, name="capacity probe result")
    sources = result.get("source_reports")
    if not isinstance(sources, Mapping) or set(sources) != CAPACITY_RESULT_SOURCE_NAMES:
        raise ValueError("capacity probe result source set differs")
    payloads = {
        name: _bound_json_source(
            value,
            label=f"capacity probe result source {name}",
        )[0]
        for name, value in sources.items()
    }
    preparation = payloads["preparation"]
    launch_receipt = payloads["launch_receipt"]
    if (
        launch_receipt.get("source_reports", {}).get("preparation")
        != sources["preparation"]
        or result.get("output_root") != preparation.get("output_root")
        or result.get("output_root") != launch_receipt.get("output_root")
    ):
        raise ValueError("capacity probe result source binding differs")
    return result, identity


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "followup_decision": args.followup_decision.resolve(),
        "capacity_probe_result": args.capacity_probe_result.resolve(),
        "source_cofitok_config": args.source_cofitok_config.resolve(),
        "source_dense_identity_config": args.source_dense_config.resolve(),
        "target_cofitok_config": args.target_cofitok_config.resolve(),
        "target_dense_identity_config": args.target_dense_config.resolve(),
    }


def build_from_args(
    args: argparse.Namespace,
    *,
    preparation_git: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _paths(args)
    followup, followup_identity = replay_followup_decision(
        paths["followup_decision"],
        expected_sha256=args.expected_followup_decision_sha256,
        expected_revision=args.expected_followup_revision,
        expected_branch=args.expected_followup_branch,
    )
    capacity, capacity_identity = replay_capacity_probe_result(
        paths["capacity_probe_result"],
        expected_sha256=args.expected_capacity_probe_result_sha256,
    )
    configs = {
        name: config_to_dict(load_config(path))
        for name, path in paths.items()
        if name.endswith("config")
    }
    identities = {
        "followup_decision": followup_identity,
        "capacity_probe_result": capacity_identity,
        **{
            name: file_identity(path)
            for name, path in paths.items()
            if name.endswith("config")
        },
    }
    return build_training_exposure_qualification_preparation(
        followup_decision=followup,
        capacity_probe_result=capacity,
        source_cofitok_config=configs["source_cofitok_config"],
        source_dense_identity_config=configs["source_dense_identity_config"],
        target_cofitok_config=configs["target_cofitok_config"],
        target_dense_identity_config=configs["target_dense_identity_config"],
        source_identities=identities,
        expected_followup_revision=args.expected_followup_revision,
        expected_followup_branch=args.expected_followup_branch,
        expected_capacity_revision=args.expected_capacity_revision,
        expected_capacity_branch=args.expected_capacity_branch,
        expected_preparation_revision=args.expected_preparation_revision,
        expected_preparation_branch=args.expected_preparation_branch,
        preparation_git=preparation_git,
        output_root=args.output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-replayed, permanently non-authorizing preparation "
            "for a fresh matched base128 full-data 300K exposure qualification."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"training exposure qualification preparation exists: {args.output}"
        )
    current_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_preparation_revision,
        "branch": args.expected_preparation_branch,
        "tracked_dirty": False,
    }
    if current_git != expected_git:
        raise ValueError("exposure qualification preparation checkout differs")
    report = build_from_args(args, preparation_git=current_git)
    write_json_report(args.output, report)
    print(report["selection"]["id"])


if __name__ == "__main__":
    main()
