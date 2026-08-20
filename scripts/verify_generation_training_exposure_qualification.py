from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object
from cofitok.reporting import git_provenance

try:
    from scripts.build_generation_training_exposure_qualification import (
        PROJECT_ROOT,
        _common_arguments,
        build_from_args,
    )
except ModuleNotFoundError:
    from build_generation_training_exposure_qualification import (
        PROJECT_ROOT,
        _common_arguments,
        build_from_args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently replay a non-authorizing matched training-exposure "
            "qualification preparation."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = file_identity(args.preparation)
    if identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("training exposure qualification preparation SHA256 differs")
    current_git = git_provenance(PROJECT_ROOT)
    expected_git = {
        "revision": args.expected_preparation_revision,
        "branch": args.expected_preparation_branch,
        "tracked_dirty": False,
    }
    if current_git != expected_git:
        raise ValueError("exposure qualification verifier checkout differs")
    actual = read_json_object(
        args.preparation,
        name="training exposure qualification preparation",
    )
    expected = build_from_args(args, preparation_git=current_git)
    if actual != expected:
        raise ValueError(
            "training exposure qualification preparation is not reproducible"
        )
    print(
        json.dumps(
            {
                "status": "verified",
                "preparation": identity,
                "selection": actual["selection"],
                "authorization_boundary": actual["authorization_boundary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
