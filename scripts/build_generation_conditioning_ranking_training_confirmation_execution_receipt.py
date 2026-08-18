from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.conditioning_ranking_training_confirmation import (
    CONFIG_RELATIVE_PATHS,
    EXPECTED_OUTPUT_ROOT,
    RUNBOOK_RELATIVE_PATH,
    RUN_NAMES,
    build_training_confirmation_execution_receipt,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance
from scripts.build_generation_conditioning_ranking_sampling_validation import (
    replay_sampling_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _config_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "control_cofitok": reject_symlink_chain(
            args.control_cofitok,
            name="control CoFiTok config",
        ).resolve(),
        "control_dense_identity": reject_symlink_chain(
            args.control_dense,
            name="control dense config",
        ).resolve(),
        "ranked_cofitok": reject_symlink_chain(
            args.ranked_cofitok,
            name="ranked CoFiTok config",
        ).resolve(),
        "ranked_dense_identity": reject_symlink_chain(
            args.ranked_dense,
            name="ranked dense config",
        ).resolve(),
    }
    if set(paths) != set(RUN_NAMES) or any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("training confirmation config set is incomplete")
    expected_paths = {
        name: (PROJECT_ROOT / relative).resolve()
        for name, relative in CONFIG_RELATIVE_PATHS.items()
    }
    if paths != expected_paths:
        raise ValueError("training confirmation config paths differ")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the one-shot source-bound execution receipt for the fresh "
            "four-arm 5K conditioning-ranking training confirmation."
        )
    )
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparation, preparation_identity = _bound_json(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        label="training confirmation preparation",
    )
    sampling, sampling_identity = replay_sampling_validation(
        args.sampling_validation,
    )
    if sampling_identity["sha256"] != args.expected_sampling_validation_sha256:
        raise ValueError("conditioning-ranking 5K sampling validation SHA256 differs")
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
    configs = _config_paths(args)
    runbook = reject_symlink_chain(
        args.runbook,
        name="training confirmation runbook",
    ).resolve()
    if runbook != (PROJECT_ROOT / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("training confirmation runbook path differs")
    runbook_identity = file_identity(runbook)
    if runbook_identity["sha256"] != args.expected_runbook_sha256:
        raise ValueError("training confirmation runbook SHA256 differs")
    output = reject_symlink_chain(
        args.output,
        name="training confirmation execution receipt",
    ).resolve()
    if output.name != "execution_receipt.json" or output.parent != Path(
        args.preparation
    ).resolve().parent:
        raise ValueError("training confirmation execution receipt path differs")
    report = build_training_confirmation_execution_receipt(
        preparation=preparation,
        preparation_identity=preparation_identity,
        sampling_validation=sampling,
        sampling_validation_identity=sampling_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        idle_gpu_evidence=idle,
        idle_gpu_evidence_identity=idle_identity,
        config_identities={name: file_identity(path) for name, path in configs.items()},
        runbook_identity=runbook_identity,
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
