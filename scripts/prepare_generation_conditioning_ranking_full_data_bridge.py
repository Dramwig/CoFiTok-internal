from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_full_data_bridge import (
    BASE_CONFIG_RELATIVE_PATHS,
    EXPECTED_OUTPUT_ROOT,
    RANKED_CONFIG_RELATIVE_PATHS,
    build_full_data_ranked_bridge_preparation,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import git_provenance
from scripts.build_generation_conditioning_ranking_posttraining_sampling_confirmation import (
    replay_posttraining_sampling_confirmation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    source_identity = file_identity(source)
    if source_identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), source_identity


def _parameter_count(path: Path) -> int:
    model = CoFiTokTiny(load_config(path).model)
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()
    return count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, without launching GPU work, the source-bound fresh "
            "full-data matched 100K conditioning-ranking bridge."
        )
    )
    parser.add_argument("--posttraining-confirmation", type=Path, required=True)
    parser.add_argument("--expected-posttraining-confirmation-sha256", required=True)
    parser.add_argument("--quality-bridge-followup", type=Path, required=True)
    parser.add_argument("--expected-quality-bridge-followup-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--base-cofitok", type=Path, required=True)
    parser.add_argument("--base-dense", type=Path, required=True)
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
    posttraining, posttraining_identity = replay_posttraining_sampling_confirmation(
        args.posttraining_confirmation
    )
    if (
        posttraining_identity["sha256"]
        != args.expected_posttraining_confirmation_sha256
    ):
        raise ValueError("posttraining sampling confirmation SHA256 differs")
    followup, followup_identity = _bound_json(
        args.quality_bridge_followup,
        expected_sha256=args.expected_quality_bridge_followup_sha256,
        label="quality-bridge follow-up decision",
    )
    standing, standing_identity = _bound_json(
        args.standing_authorization,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )

    paths = {
        "base_cofitok": reject_symlink_chain(
            args.base_cofitok, name="base CoFiTok config"
        ).resolve(),
        "base_dense_identity": reject_symlink_chain(
            args.base_dense, name="base dense config"
        ).resolve(),
        "ranked_cofitok": reject_symlink_chain(
            args.ranked_cofitok, name="ranked CoFiTok config"
        ).resolve(),
        "ranked_dense_identity": reject_symlink_chain(
            args.ranked_dense, name="ranked dense config"
        ).resolve(),
    }
    expected_paths = {
        "base_cofitok": (
            PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATHS["cofitok"]
        ).resolve(),
        "base_dense_identity": (
            PROJECT_ROOT / BASE_CONFIG_RELATIVE_PATHS["dense_identity"]
        ).resolve(),
        "ranked_cofitok": (
            PROJECT_ROOT / RANKED_CONFIG_RELATIVE_PATHS["cofitok"]
        ).resolve(),
        "ranked_dense_identity": (
            PROJECT_ROOT / RANKED_CONFIG_RELATIVE_PATHS["dense_identity"]
        ).resolve(),
    }
    if paths != expected_paths or any(not path.is_file() for path in paths.values()):
        raise ValueError("ranked full-data bridge config paths differ")

    configs = {name: config_to_dict(load_config(path)) for name, path in paths.items()}
    report = build_full_data_ranked_bridge_preparation(
        posttraining_confirmation=posttraining,
        posttraining_confirmation_identity=posttraining_identity,
        quality_bridge_followup=followup,
        quality_bridge_followup_identity=followup_identity,
        standing_authorization=standing,
        standing_authorization_identity=standing_identity,
        base_cofitok=configs["base_cofitok"],
        base_dense_identity=configs["base_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense_identity=configs["ranked_dense_identity"],
        config_identities={name: file_identity(path) for name, path in paths.items()},
        parameter_counts={name: _parameter_count(path) for name, path in paths.items()},
        builder_git=git_provenance(PROJECT_ROOT),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_output_root=args.output_root,
    )
    output_identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(output_identity["sha256"])


if __name__ == "__main__":
    main()
