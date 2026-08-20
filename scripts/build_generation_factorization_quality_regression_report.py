from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cofitok.generation.factorization_quality_regression import (
    SEEDS,
    build_diagnostic_report,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)


def _bound(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _seed_paths(values: list[str], *, label: str) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text.isdigit() or not path_text:
            raise ValueError(f"{label} must use SEED=PATH")
        seed = int(seed_text)
        if seed in paths:
            raise ValueError(f"duplicate {label} seed: {seed}")
        paths[seed] = Path(path_text)
    if set(paths) != set(SEEDS):
        raise ValueError(f"{label} seed set differs")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and aggregate two-seed matched rollout-stability evidence for the "
            "exact 100K factorization quality-regression diagnostic."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    parser.add_argument("--expected-source-binding-sha256", required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--expected-execution-authorization-sha256", required=True)
    parser.add_argument("--cofitok-training", type=Path, required=True)
    parser.add_argument("--dense-training", type=Path, required=True)
    parser.add_argument("--cofitok-checkpoint-eval", type=Path, required=True)
    parser.add_argument("--dense-checkpoint-eval", type=Path, required=True)
    parser.add_argument("--cofitok-rollout", action="append", default=[])
    parser.add_argument("--dense-rollout", action="append", default=[])
    parser.add_argument("--qualification", action="append", default=[])
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparation, preparation_identity = _bound(
        args.preparation,
        label="factorization-regression preparation",
        expected_sha256=args.expected_preparation_sha256,
    )
    binding, binding_identity = _bound(
        args.source_binding,
        label="factorization-regression source binding",
        expected_sha256=args.expected_source_binding_sha256,
    )
    authorization, authorization_identity = _bound(
        args.execution_authorization,
        label="factorization-regression execution authorization",
        expected_sha256=args.expected_execution_authorization_sha256,
    )
    cofitok_training, cofitok_training_identity = _bound(
        args.cofitok_training,
        label="CoFiTok training report",
    )
    dense_training, dense_training_identity = _bound(
        args.dense_training,
        label="dense training report",
    )
    cofitok_eval, cofitok_eval_identity = _bound(
        args.cofitok_checkpoint_eval,
        label="CoFiTok diagnostic checkpoint evaluation",
    )
    dense_eval, dense_eval_identity = _bound(
        args.dense_checkpoint_eval,
        label="dense diagnostic checkpoint evaluation",
    )
    cofitok_paths = _seed_paths(args.cofitok_rollout, label="CoFiTok rollout")
    dense_paths = _seed_paths(args.dense_rollout, label="dense rollout")
    qualification_paths = _seed_paths(args.qualification, label="qualification")
    rollout_reports: dict[int, dict[str, dict[str, Any]]] = {}
    qualifications: dict[int, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {
        "cofitok_training": cofitok_training_identity,
        "dense_training": dense_training_identity,
        "cofitok_diagnostic_checkpoint_evaluation": cofitok_eval_identity,
        "dense_diagnostic_checkpoint_evaluation": dense_eval_identity,
    }
    for seed in SEEDS:
        cofitok_rollout, cofitok_identity = _bound(
            cofitok_paths[seed],
            label=f"CoFiTok rollout seed {seed}",
        )
        dense_rollout, dense_identity = _bound(
            dense_paths[seed],
            label=f"dense rollout seed {seed}",
        )
        qualification, qualification_identity = _bound(
            qualification_paths[seed],
            label=f"stability qualification seed {seed}",
        )
        rollout_reports[seed] = {
            "cofitok": cofitok_rollout,
            "dense_identity": dense_rollout,
        }
        qualifications[seed] = qualification
        identities[f"cofitok_rollout_seed_{seed}"] = cofitok_identity
        identities[f"dense_rollout_seed_{seed}"] = dense_identity
        identities[f"qualification_seed_{seed}"] = qualification_identity
    report = build_diagnostic_report(
        preparation=preparation,
        preparation_identity=preparation_identity,
        source_binding=binding,
        source_binding_identity=binding_identity,
        execution_authorization=authorization,
        execution_authorization_identity=authorization_identity,
        training_reports={
            "cofitok": cofitok_training,
            "dense_identity": dense_training,
        },
        checkpoint_evaluations={
            "cofitok": cofitok_eval,
            "dense_identity": dense_eval,
        },
        rollout_reports=rollout_reports,
        qualification_reports=qualifications,
        source_identities=identities,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
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
