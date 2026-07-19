from __future__ import annotations

import argparse
import json
import shlex

from cofitok.generation_paths import (
    generation_deployment_attestation_paths,
    generation_workspace_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the authoritative generation workspace path contract."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-revision")
    parser.add_argument("--format", choices=["json", "shell"], default="json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generation_workspace_paths(
        project_root=args.project_root,
        output_root=args.output_root,
    )
    if args.target_revision:
        paths.update(
            generation_deployment_attestation_paths(
                output_root=args.output_root,
                target_revision=args.target_revision,
            )
        )
    rendered = {name: path.as_posix() for name, path in paths.items()}
    if args.format == "shell":
        for name, value in rendered.items():
            print(f"{name}={shlex.quote(value)}")
        return
    print(json.dumps(rendered, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
