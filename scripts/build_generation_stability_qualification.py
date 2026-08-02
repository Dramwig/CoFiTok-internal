from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.stability_qualification import (
    DEFAULT_HIGH_FREQUENCY_TIMESTEPS,
    build_stability_qualification,
)
from cofitok.reporting import file_sha256, write_json_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a matched CoFiTok/dense rollout-stability qualification report."
    )
    parser.add_argument("--cofitok-training", type=Path, required=True)
    parser.add_argument("--dense-training", type=Path, required=True)
    parser.add_argument("--cofitok-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--cofitok-rollout", type=Path, required=True)
    parser.add_argument("--dense-rollout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--high-frequency-timesteps",
        default=",".join(str(value) for value in DEFAULT_HIGH_FREQUENCY_TIMESTEPS),
    )
    parser.add_argument("--max-tail-two-energy-ratio", type=float, default=0.65)
    parser.add_argument("--max-single-token-energy-ratio", type=float, default=0.35)
    parser.add_argument("--max-relative-regression", type=float, default=0.05)
    parser.add_argument("--max-high-frequency-ratio", type=float, default=1.5)
    parser.add_argument("--min-shuffle-mismatch-ratio", type=float, default=2.0)
    parser.add_argument("--max-zero-token-abs", type=float, default=1e-8)
    parser.add_argument(
        "--weights",
        choices=["model", "ema"],
        default="model",
        help="Require all checkpoint and rollout reports to use these weights.",
    )
    parser.add_argument("--expected-evaluation-revision")
    parser.add_argument("--expected-evaluation-branch")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> int:
    args = _parse_args()
    paths = {
        "cofitok_training": args.cofitok_training,
        "dense_training": args.dense_training,
        "cofitok_checkpoint": args.cofitok_checkpoint,
        "dense_checkpoint": args.dense_checkpoint,
        "cofitok_rollout": args.cofitok_rollout,
        "dense_rollout": args.dense_rollout,
    }
    timesteps = tuple(
        int(value.strip())
        for value in args.high_frequency_timesteps.split(",")
        if value.strip()
    )
    report = build_stability_qualification(
        **{name: _load(path) for name, path in paths.items()},
        expected_weights=args.weights,
        expected_evaluation_revision=args.expected_evaluation_revision,
        expected_evaluation_branch=args.expected_evaluation_branch,
        high_frequency_timesteps=timesteps,
        max_tail_two_energy_ratio=args.max_tail_two_energy_ratio,
        max_single_token_energy_ratio=args.max_single_token_energy_ratio,
        max_relative_regression=args.max_relative_regression,
        max_high_frequency_ratio=args.max_high_frequency_ratio,
        min_shuffle_mismatch_ratio=args.min_shuffle_mismatch_ratio,
        max_zero_token_abs=args.max_zero_token_abs,
    )
    report["sources"] = {name: _source(path) for name, path in paths.items()}
    output_path = args.output_dir / "qualification_report.json"
    write_json_report(output_path, report)
    print(json.dumps({"output": str(output_path), "status": report["status"]}, indent=2))
    return 2 if args.require_pass and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
