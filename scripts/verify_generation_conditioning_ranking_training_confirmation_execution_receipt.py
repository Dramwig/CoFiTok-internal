from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_training_confirmation import (
    CONFIG_RELATIVE_PATHS,
    EXPECTED_OUTPUT_ROOT,
    RUNBOOK_RELATIVE_PATH,
    RUN_NAMES,
    build_training_confirmation_execution_receipt,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _descriptor(receipt: Mapping[str, Any], name: str) -> dict[str, Any]:
    sources = receipt.get("source_reports")
    descriptor = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"training confirmation receipt source is missing: {name}")
    return dict(descriptor)


def _bound_source(
    receipt: Mapping[str, Any],
    *,
    name: str,
    path: Path,
    expected_sha256: str,
    json_source: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source = reject_symlink_chain(
        path,
        name=f"training confirmation receipt {name}",
    ).resolve()
    identity = file_identity(source)
    if identity != _descriptor(receipt, name) or identity["sha256"] != expected_sha256:
        raise ValueError(f"training confirmation receipt source changed: {name}")
    value = read_json_object(source, name=name) if json_source else None
    return value, identity


def _config_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "control_cofitok": Path(args.control_cofitok),
        "control_dense_identity": Path(args.control_dense),
        "ranked_cofitok": Path(args.ranked_cofitok),
        "ranked_dense_identity": Path(args.ranked_dense),
    }
    resolved = {
        name: reject_symlink_chain(path, name=f"{name} config").resolve()
        for name, path in paths.items()
    }
    expected = {
        name: (PROJECT_ROOT / relative).resolve()
        for name, relative in CONFIG_RELATIVE_PATHS.items()
    }
    if resolved != expected:
        raise ValueError("training confirmation config paths differ")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact one-shot four-arm 5K training-confirmation "
            "execution receipt."
        )
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--sampling-validation", type=Path, required=True)
    parser.add_argument("--expected-sampling-validation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--idle-gpu-evidence", type=Path, required=True)
    parser.add_argument("--expected-idle-gpu-evidence-sha256", required=True)
    parser.add_argument("--control-cofitok", type=Path, required=True)
    parser.add_argument("--control-dense", type=Path, required=True)
    parser.add_argument("--ranked-cofitok", type=Path, required=True)
    parser.add_argument("--ranked-dense", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--expected-runbook-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", default=EXPECTED_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_path = reject_symlink_chain(
        args.receipt,
        name="training confirmation execution receipt",
    ).resolve()
    receipt_identity = file_identity(receipt_path)
    if receipt_identity["sha256"] != args.expected_receipt_sha256:
        raise ValueError("training confirmation execution receipt SHA256 differs")
    receipt = read_json_object(
        receipt_path,
        name="training confirmation execution receipt",
    )
    preparation, preparation_identity = _bound_source(
        receipt,
        name="preparation",
        path=args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        json_source=True,
    )
    sampling, sampling_identity = _bound_source(
        receipt,
        name="sampling_validation",
        path=args.sampling_validation,
        expected_sha256=args.expected_sampling_validation_sha256,
        json_source=True,
    )
    standing, standing_identity = _bound_source(
        receipt,
        name="standing_authorization",
        path=args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        json_source=True,
    )
    idle, idle_identity = _bound_source(
        receipt,
        name="idle_gpu_evidence",
        path=args.idle_gpu_evidence,
        expected_sha256=args.expected_idle_gpu_evidence_sha256,
        json_source=True,
    )
    config_descriptors = receipt.get("source_reports", {}).get("configs")
    if not isinstance(config_descriptors, Mapping) or set(config_descriptors) != set(
        RUN_NAMES
    ):
        raise ValueError("training confirmation receipt config set differs")
    config_identities: dict[str, dict[str, Any]] = {}
    for name, path in _config_paths(args).items():
        identity = file_identity(path)
        if identity != config_descriptors[name]:
            raise ValueError(f"training confirmation receipt config changed: {name}")
        config_identities[name] = identity
    runbook = reject_symlink_chain(
        args.runbook,
        name="training confirmation runbook",
    ).resolve()
    if runbook != (PROJECT_ROOT / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("training confirmation runbook path differs")
    runbook_identity = file_identity(runbook)
    if (
        runbook_identity != _descriptor(receipt, "runbook")
        or runbook_identity["sha256"] != args.expected_runbook_sha256
    ):
        raise ValueError("training confirmation receipt runbook changed")
    if not all(
        isinstance(value, dict)
        for value in (preparation, sampling, standing, idle)
    ):
        raise TypeError("training confirmation receipt JSON sources are malformed")
    expected = build_training_confirmation_execution_receipt(
        preparation=preparation,
        preparation_identity=preparation_identity,
        sampling_validation=sampling,
        sampling_validation_identity=sampling_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=idle_identity,
        config_identities=config_identities,
        runbook_identity=runbook_identity,
        receipt_git=git_provenance(PROJECT_ROOT),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.expected_output_root,
    )
    if receipt != expected:
        raise ValueError("training confirmation execution receipt replay differs")
    print(receipt_identity["sha256"])


if __name__ == "__main__":
    main()
