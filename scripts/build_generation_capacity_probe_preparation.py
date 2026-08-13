from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.capacity_probe import (
    CAPACITY_PROBE_REFERENCE_REASON,
    CAPACITY_PROBE_STOP_STEP,
    build_capacity_probe_preparation,
)
from cofitok.generation.quality_bridge_followup import (
    QUALITY_BRIDGE_EXECUTION_BRANCH,
    QUALITY_BRIDGE_EXECUTION_REVISION,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_quality_bridge_followup_decision import (
    build_from_sources as rebuild_followup_decision,
)
from scripts.preserve_generation_checkpoint_reference import (
    verify_checkpoint_reference,
)
from scripts.validate_generation_configs import validate_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--expected-followup-decision-sha256", required=True)
    parser.add_argument("--base-cofitok-config", type=Path, required=True)
    parser.add_argument("--base-dense-config", type=Path, required=True)
    parser.add_argument("--capacity-cofitok-config", type=Path, required=True)
    parser.add_argument("--capacity-dense-config", type=Path, required=True)
    parser.add_argument("--cofitok-reference-receipt", type=Path, required=True)
    parser.add_argument("--dense-reference-receipt", type=Path, required=True)
    parser.add_argument("--expected-preparation-revision", required=True)
    parser.add_argument("--expected-preparation-branch", required=True)
    parser.add_argument("--output-root", required=True)


def _reference(path: Path, *, method: str) -> dict[str, Any]:
    report = read_json_object(path, name=f"{method} checkpoint reference receipt")
    verify_checkpoint_reference(
        report,
        expected_step=CAPACITY_PROBE_STOP_STEP,
        expected_revision=QUALITY_BRIDGE_EXECUTION_REVISION,
        expected_branch=QUALITY_BRIDGE_EXECUTION_BRANCH,
        expected_reason=CAPACITY_PROBE_REFERENCE_REASON,
    )
    return {"receipt": file_identity(path), "report": report}


def build_from_paths(args: argparse.Namespace) -> dict[str, Any]:
    decision_identity = file_identity(args.followup_decision)
    if decision_identity["sha256"] != args.expected_followup_decision_sha256:
        raise ValueError("quality bridge follow-up decision SHA256 differs")
    decision = read_json_object(
        args.followup_decision,
        name="quality bridge follow-up decision",
    )
    decision_git = decision.get("decision_builder_git")
    if not isinstance(decision_git, dict):
        raise ValueError("quality bridge follow-up decision Git identity is missing")
    source_result = decision.get("source_reports", {}).get("quality_bridge_result")
    if not isinstance(source_result, dict):
        raise ValueError("quality bridge follow-up source result is missing")
    rebuilt = rebuild_followup_decision(
        quality_bridge_result_path=Path(str(source_result["path"])),
        expected_quality_bridge_result_sha256=str(source_result["sha256"]),
        decision_git=decision_git,
    )
    if rebuilt != decision:
        raise ValueError("quality bridge follow-up decision does not replay")

    quality_bridge_result_path = Path(str(source_result["path"]))
    quality_bridge_result = read_json_object(
        quality_bridge_result_path,
        name="quality bridge result",
    )
    bridge_preparation_source = quality_bridge_result.get("source_reports", {}).get(
        "preparation"
    )
    if not isinstance(bridge_preparation_source, dict):
        raise ValueError("quality bridge result preparation source is missing")
    bridge_preparation_path = Path(str(bridge_preparation_source["path"]))
    if file_identity(bridge_preparation_path) != bridge_preparation_source:
        raise ValueError("quality bridge preparation identity differs")
    bridge_preparation = read_json_object(
        bridge_preparation_path,
        name="quality bridge preparation",
    )
    if (
        bridge_preparation.get("status") != "prepared"
        or bridge_preparation.get("role")
        != "stability_full_data_quality_bridge_preparation"
        or bridge_preparation.get("authorization_boundary", {}).get(
            "full_300k_launch_allowed"
        )
        is not False
    ):
        raise ValueError("quality bridge preparation contract differs")

    config_paths = {
        "base": {
            "cofitok": args.base_cofitok_config,
            "dense_identity": args.base_dense_config,
        },
        "capacity": {
            "cofitok": args.capacity_cofitok_config,
            "dense_identity": args.capacity_dense_config,
        },
    }
    config_identities = {
        group: {method: file_identity(path) for method, path in paths.items()}
        for group, paths in config_paths.items()
    }
    expected_base = bridge_preparation.get("matched_training_contract", {})
    expected_base_cofitok = expected_base.get("cofitok_config")
    expected_base_dense = expected_base.get("dense_config")
    if not isinstance(expected_base_cofitok, dict) or not isinstance(
        expected_base_dense, dict
    ):
        raise ValueError("quality bridge preparation config identities are missing")
    if (
        {
            key: config_identities["base"]["cofitok"][key]
            for key in ("bytes", "sha256")
        }
        != {key: expected_base_cofitok.get(key) for key in ("bytes", "sha256")}
        or {
            key: config_identities["base"]["dense_identity"][key]
            for key in ("bytes", "sha256")
        }
        != {key: expected_base_dense.get(key) for key in ("bytes", "sha256")}
    ):
        raise ValueError("capacity probe uses different base128 bridge configs")

    loaded = {
        group: {method: load_config(path) for method, path in paths.items()}
        for group, paths in config_paths.items()
    }
    configs = {
        group: {method: config_to_dict(config) for method, config in rows.items()}
        for group, rows in loaded.items()
    }
    base_validation = validate_pair(
        loaded["base"]["cofitok"],
        loaded["base"]["dense_identity"],
        max_parameter_gap=0.02,
        stage="stability_quality_bridge",
    )
    capacity_validation = validate_pair(
        loaded["capacity"]["cofitok"],
        loaded["capacity"]["dense_identity"],
        max_parameter_gap=0.02,
        stage="stability_capacity_probe",
    )

    preparation_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_preparation_revision,
        "branch": args.expected_preparation_branch,
        "tracked_dirty": False,
    }
    if preparation_git != expected_git:
        raise ValueError("capacity probe preparation checkout identity differs")

    references = {
        "cofitok": _reference(args.cofitok_reference_receipt, method="cofitok"),
        "dense_identity": _reference(
            args.dense_reference_receipt,
            method="dense_identity",
        ),
    }
    return build_capacity_probe_preparation(
        followup_decision=decision,
        followup_decision_identity=decision_identity,
        quality_bridge_result_identity=file_identity(quality_bridge_result_path),
        quality_bridge_preparation_identity=file_identity(bridge_preparation_path),
        base_configs=configs["base"],
        capacity_configs=configs["capacity"],
        base_config_identities=config_identities["base"],
        capacity_config_identities=config_identities["capacity"],
        base_config_validation=base_validation,
        capacity_config_validation=capacity_validation,
        checkpoint_references=references,
        preparation_git=preparation_git,
        output_root=args.output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-replayed, non-authorizing matched 250M/10K "
            "capacity-probe preparation."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity probe preparation exists: {args.output}")
    report = build_from_paths(args)
    write_json_report(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
