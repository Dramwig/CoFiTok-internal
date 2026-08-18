from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation.conditioning_ranking_sampling import (
    EXPECTED_OUTPUT_ROOT,
    build_sampling_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance
from scripts.build_generation_conditioning_ranking_sampling_validation import (
    replay_postevaluation,
    validate_training_checkpoints,
)
from scripts.evaluate_generation_class_fidelity import classifier_identity


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


def _embedded_source(
    descriptor: object,
    *,
    label: str,
) -> tuple[Path, dict]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} identity is missing")
    source = reject_symlink_chain(
        Path(str(descriptor.get("path", ""))),
        name=label,
    ).resolve()
    source_identity = file_identity(source)
    if source_identity != descriptor:
        raise ValueError(f"{label} identity differs")
    return source, source_identity


def replay_sampling_preparation(
    path: str | Path,
) -> tuple[dict, dict]:
    source = reject_symlink_chain(path, name="sampling preparation").resolve()
    source_identity = file_identity(source)
    preparation = read_json_object(source, name="sampling preparation")
    sources = preparation.get("source_reports")
    if not isinstance(sources, dict):
        raise ValueError("sampling preparation sources are missing")
    postevaluation_path, expected_postevaluation_identity = _embedded_source(
        sources.get("postevaluation"),
        label="conditioning-ranking postevaluation",
    )
    postevaluation, postevaluation_identity, training_reports = replay_postevaluation(
        postevaluation_path
    )
    if postevaluation_identity != expected_postevaluation_identity:
        raise ValueError("conditioning-ranking postevaluation identity differs")
    standing_path, standing_identity = _embedded_source(
        sources.get("standing_authorization"),
        label="standing experiment authorization",
    )
    standing = read_json_object(standing_path, name="standing experiment authorization")
    runbook_path, runbook_identity = _embedded_source(
        sources.get("runbook"),
        label="conditioning-ranking sampling runbook",
    )
    git = preparation.get("git")
    if not isinstance(git, dict):
        raise ValueError("sampling preparation Git identity is missing")
    rebuilt = build_sampling_preparation(
        postevaluation=postevaluation,
        postevaluation_identity=postevaluation_identity,
        checkpoint_evidence=validate_training_checkpoints(
            postevaluation,
            training_reports,
        ),
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        classifier=classifier_identity(
            str(preparation.get("classifier", {}).get("weights_path", ""))
        ),
        runbook_identity=runbook_identity,
        builder_git=git_provenance(PROJECT_ROOT),
        expected_revision=str(git.get("revision", "")),
        expected_branch=str(git.get("branch", "")),
        expected_output_root=str(preparation.get("output_root", "")),
    )
    if rebuilt != preparation:
        raise ValueError("sampling preparation replay differs")
    if runbook_path != Path(runbook_identity["path"]):
        raise ValueError("sampling runbook path differs")
    return preparation, source_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the immutable source-bound four-arm 5K generated-sample "
            "conditioning-ranking validation."
        )
    )
    parser.add_argument("--postevaluation", type=Path, required=True)
    parser.add_argument("--expected-postevaluation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    postevaluation, postevaluation_identity, training_reports = replay_postevaluation(
        args.postevaluation
    )
    if postevaluation_identity["sha256"] != args.expected_postevaluation_sha256:
        raise ValueError("conditioning-ranking postevaluation SHA256 differs")
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    runbook = reject_symlink_chain(args.runbook, name="sampling runbook").resolve()
    if not runbook.is_file():
        raise FileNotFoundError(f"sampling runbook is missing: {runbook}")
    report = build_sampling_preparation(
        postevaluation=postevaluation,
        postevaluation_identity=postevaluation_identity,
        checkpoint_evidence=validate_training_checkpoints(
            postevaluation,
            training_reports,
        ),
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        classifier=classifier_identity(args.classifier_checkpoint),
        runbook_identity=file_identity(runbook),
        builder_git=git_provenance(PROJECT_ROOT),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.output_root,
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
