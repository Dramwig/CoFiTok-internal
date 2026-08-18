from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.conditioning_ranking_sampling import (
    EXPECTED_OUTPUT_ROOT,
    build_sampling_execution_receipt,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance
from scripts.prepare_generation_conditioning_ranking_sampling_validation import (
    replay_sampling_preparation,
)
from scripts.select_generation_conditioning_ranking_sampling_batch import (
    validate_completed_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _embedded_source(
    receipt: Mapping[str, Any],
    *,
    name: str,
) -> tuple[dict, dict]:
    sources = receipt.get("source_reports")
    descriptor = sources.get(name) if isinstance(sources, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"sampling execution receipt source is missing: {name}")
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=f"sampling execution receipt {name}",
    ).resolve()
    source_identity = file_identity(source)
    if source_identity != descriptor:
        raise ValueError(f"sampling execution receipt source changed: {name}")
    return read_json_object(source, name=name), source_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the exact four-arm 5K sampling execution receipt."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-idle-gpu-evidence-sha256", required=True)
    parser.add_argument("--expected-batch-selection-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", default=EXPECTED_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_path = reject_symlink_chain(
        args.receipt,
        name="conditioning-ranking sampling execution receipt",
    ).resolve()
    receipt_identity = file_identity(receipt_path)
    if receipt_identity["sha256"] != args.expected_receipt_sha256:
        raise ValueError("sampling execution receipt SHA256 differs")
    receipt = read_json_object(receipt_path, name="sampling execution receipt")
    preparation, preparation_identity = replay_sampling_preparation(
        receipt.get("source_reports", {}).get("preparation", {}).get("path", "")
    )
    if (
        preparation_identity
        != receipt.get("source_reports", {}).get("preparation")
        or preparation_identity["sha256"] != args.expected_preparation_sha256
    ):
        raise ValueError("sampling preparation identity differs")
    standing, standing_identity = _embedded_source(receipt, name="standing_authorization")
    idle, idle_identity = _embedded_source(receipt, name="idle_gpu_evidence")
    selection, selection_identity = _embedded_source(receipt, name="batch_selection")
    if standing_identity["sha256"] != args.expected_standing_authorization_sha256:
        raise ValueError("standing experiment authorization SHA256 differs")
    if idle_identity["sha256"] != args.expected_idle_gpu_evidence_sha256:
        raise ValueError("idle GPU evidence SHA256 differs")
    if selection_identity["sha256"] != args.expected_batch_selection_sha256:
        raise ValueError("sampling batch selection SHA256 differs")
    validate_completed_selection(
        selection,
        preparation=preparation,
        preparation_identity=preparation_identity,
    )
    expected = build_sampling_execution_receipt(
        preparation=preparation,
        preparation_identity=preparation_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=idle_identity,
        batch_selection=selection,
        batch_selection_identity=selection_identity,
        receipt_git=git_provenance(PROJECT_ROOT),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.expected_output_root,
    )
    if receipt != expected:
        raise ValueError("sampling execution receipt replay differs")
    print(receipt_identity["sha256"])


if __name__ == "__main__":
    main()
