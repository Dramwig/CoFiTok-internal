from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation_control_continuity import build_continuity_archive


DEFAULT_PLAN = Path(
    "configs/generation/diagnostics/"
    "capacity_generation_control_plane_continuity_v3.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Persist the exact static control assets needed to reconstruct the "
            "active capacity-generation wait chain without launching or signaling it."
        )
    )
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project.resolve()
    plan = args.plan if args.plan.is_absolute() else project / args.plan
    report = build_continuity_archive(
        project=project,
        plan_path=plan,
        output_root=args.output_root,
        expected_revision=args.expected_revision,
        expected_tree=args.expected_tree,
        expected_branch=args.expected_branch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
