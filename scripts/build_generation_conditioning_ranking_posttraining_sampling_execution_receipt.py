from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation.conditioning_ranking_posttraining_sampling import (
    EXPECTED_OUTPUT_ROOT,
    build_sampling_execution_receipt,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance
from scripts.prepare_generation_conditioning_ranking_posttraining_sampling_confirmation import (
    replay_sampling_preparation,
)
from scripts.select_generation_conditioning_ranking_sampling_batch import (
    validate_completed_selection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict, dict]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound execution receipt for the independent-stream "
            "posttraining four-arm sampling confirmation."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--idle-gpu-evidence", type=Path, required=True)
    parser.add_argument("--expected-idle-gpu-evidence-sha256", required=True)
    parser.add_argument("--batch-selection", type=Path, required=True)
    parser.add_argument("--expected-batch-selection-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparation, preparation_identity = replay_sampling_preparation(args.preparation)
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("posttraining sampling preparation SHA256 differs")
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    idle, idle_identity = _bound_json(
        args.idle_gpu_evidence,
        expected_sha256=args.expected_idle_gpu_evidence_sha256,
        label="five-poll idle GPU evidence",
    )
    selection, selection_identity = _bound_json(
        args.batch_selection,
        expected_sha256=args.expected_batch_selection_sha256,
        label="four-arm posttraining sampling batch selection",
    )
    validate_completed_selection(
        selection,
        preparation=preparation,
        preparation_identity=preparation_identity,
    )
    output = reject_symlink_chain(
        args.output,
        name="posttraining sampling execution receipt",
    ).resolve()
    expected_output = (
        Path(args.expected_output_root).resolve() / "reports" / "execution_receipt.json"
    )
    if output != expected_output:
        raise ValueError("posttraining sampling execution receipt output differs")
    report = build_sampling_execution_receipt(
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
    receipt_identity = prepare_manifest(
        output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(receipt_identity["sha256"])


if __name__ == "__main__":
    main()
