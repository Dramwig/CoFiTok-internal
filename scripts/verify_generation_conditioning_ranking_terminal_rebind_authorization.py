from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.inference_replay import file_identity, read_json_object, reject_symlink_chain

try:
    from scripts.build_generation_conditioning_ranking_terminal_rebind_authorization import (
        build_from_authorization,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback.
    from build_generation_conditioning_ranking_terminal_rebind_authorization import (
        build_from_authorization,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay the source-bound four-arm 1K terminal-rebind "
            "execution authorization."
        )
    )
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization_path = reject_symlink_chain(
        args.authorization,
        name="conditioning-ranking terminal-rebind authorization",
    ).resolve()
    identity = file_identity(authorization_path)
    if identity["sha256"] != args.expected_authorization_sha256:
        raise ValueError("terminal-rebind authorization SHA256 differs")
    actual = read_json_object(
        authorization_path,
        name="conditioning-ranking terminal-rebind authorization",
    )
    expected = build_from_authorization(
        actual,
        project=PROJECT_ROOT,
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
        expected_output_root=args.expected_output_root,
    )
    if actual != expected:
        raise ValueError("terminal-rebind authorization is not reproducible")
    print(identity["sha256"])


if __name__ == "__main__":
    main()
