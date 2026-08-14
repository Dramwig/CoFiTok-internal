from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.capacity_probe_result import (
    CAPACITY_PROBE_ARM_NAMES,
    build_capacity_probe_result,
)
from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import write_json_report
from scripts.build_generation_quality_bridge_result import (
    _verify_physical_generation_evidence,
)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument(
        "--base256-cofitok-training-validation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base256-dense-training-validation",
        type=Path,
        required=True,
    )
    for arm in CAPACITY_PROBE_ARM_NAMES:
        flag = arm.replace("_", "-")
        parser.add_argument(
            f"--{flag}-sampling-preflight",
            dest=f"{arm}_sampling_preflight",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{flag}-generation",
            dest=f"{arm}_generation",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{flag}-checkpoint-eval",
            dest=f"{arm}_checkpoint_eval",
            type=Path,
            required=True,
        )
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "preparation": args.preparation.resolve(),
        "launch_receipt": args.launch_receipt.resolve(),
        "base256_cofitok_training_validation": (
            args.base256_cofitok_training_validation.resolve()
        ),
        "base256_dense_identity_training_validation": (
            args.base256_dense_training_validation.resolve()
        ),
    }
    for arm in CAPACITY_PROBE_ARM_NAMES:
        paths[f"{arm}_sampling_preflight"] = getattr(
            args,
            f"{arm}_sampling_preflight",
        ).resolve()
        paths[f"{arm}_generation"] = getattr(
            args,
            f"{arm}_generation",
        ).resolve()
        paths[f"{arm}_checkpoint_eval"] = getattr(
            args,
            f"{arm}_checkpoint_eval",
        ).resolve()
    return paths


def _verify_launch_sources(receipt: dict[str, Any]) -> None:
    sources = receipt.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError("capacity probe launch receipt source reports are missing")
    for name, expected in sources.items():
        if not isinstance(expected, dict):
            raise ValueError(f"capacity probe launch source is malformed: {name}")
        if file_identity(expected.get("path", "")) != expected:
            raise ValueError(f"capacity probe launch source changed: {name}")


def build_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    identities = {name: file_identity(path) for name, path in paths.items()}
    if identities["preparation"]["sha256"] != args.expected_preparation_sha256:
        raise ValueError("capacity probe preparation SHA256 differs")
    if identities["launch_receipt"]["sha256"] != args.expected_launch_receipt_sha256:
        raise ValueError("capacity probe launch receipt SHA256 differs")
    launch = read_json_object(paths["launch_receipt"], name="capacity probe launch receipt")
    _verify_launch_sources(launch)
    generation_reports = {
        arm: read_json_object(
            paths[f"{arm}_generation"],
            name=f"{arm} generation metrics",
        )
        for arm in CAPACITY_PROBE_ARM_NAMES
    }
    physical = {
        arm: _verify_physical_generation_evidence(
            generation_reports[arm],
            label=f"capacity probe {arm}",
        )
        for arm in CAPACITY_PROBE_ARM_NAMES
    }
    return build_capacity_probe_result(
        preparation=read_json_object(
            paths["preparation"],
            name="capacity probe preparation",
        ),
        launch_receipt=launch,
        partial_training_validations={
            "cofitok": read_json_object(
                paths["base256_cofitok_training_validation"],
                name="base256 CoFiTok partial training validation",
            ),
            "dense_identity": read_json_object(
                paths["base256_dense_identity_training_validation"],
                name="base256 dense partial training validation",
            ),
        },
        sampling_preflights={
            arm: read_json_object(
                paths[f"{arm}_sampling_preflight"],
                name=f"{arm} sampling preflight",
            )
            for arm in CAPACITY_PROBE_ARM_NAMES
        },
        generation_reports=generation_reports,
        checkpoint_evaluations={
            arm: read_json_object(
                paths[f"{arm}_checkpoint_eval"],
                name=f"{arm} checkpoint evaluation",
            )
            for arm in CAPACITY_PROBE_ARM_NAMES
        },
        physical_evidence=physical,
        source_identities=identities,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the physical-evidence-replayed, non-authorizing four-arm "
            "250M/10K capacity-probe result."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"capacity probe result exists: {args.output}")
    report = build_from_args(args)
    write_json_report(args.output, report)
    print(report["decision"]["recommendation"]["id"])


if __name__ == "__main__":
    main()
