from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_training_confirmation import (
    EXPECTED_OUTPUT_ROOT,
    RUN_NAMES,
    build_training_confirmation_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import git_provenance


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


def _parameter_count(path: Path) -> int:
    model = CoFiTokTiny(load_config(path).model)
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, without launching GPU work, the source-bound four-arm "
            "5K class-conditioning-ranking training confirmation."
        )
    )
    parser.add_argument("--sampling-validation", type=Path, required=True)
    parser.add_argument("--expected-sampling-validation-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--control-cofitok", type=Path, required=True)
    parser.add_argument("--control-dense", type=Path, required=True)
    parser.add_argument("--ranked-cofitok", type=Path, required=True)
    parser.add_argument("--ranked-dense", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", default=EXPECTED_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sampling, sampling_identity = _bound_json(
        args.sampling_validation,
        expected_sha256=args.expected_sampling_validation_sha256,
        label="conditioning-ranking 5K sampling validation",
    )
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
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
    configs = {name: config_to_dict(load_config(path)) for name, path in paths.items()}
    report = build_training_confirmation_preparation(
        sampling_validation=sampling,
        sampling_validation_identity=sampling_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense_identity"],
        config_identities={name: file_identity(path) for name, path in paths.items()},
        parameter_counts={name: _parameter_count(path) for name, path in paths.items()},
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
