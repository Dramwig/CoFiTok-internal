from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.min_snr_pilot import (
    PILOT_STEP,
    validate_sampling_preflight,
)
from cofitok.training.checkpointing import verify_training_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one reusable matched Min-SNR pilot sampling preflight."
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--expected-prefix-budget", type=int, required=True)
    parser.add_argument("--sampling-preflight", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    preflight_path = Path(args.sampling_preflight).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    row = validate_sampling_preflight(
        _read(preflight_path),
        arm=args.arm,
        expected_prefix_budget=args.expected_prefix_budget,
    )
    if (
        row["git"]["revision"] != args.expected_revision
        or row["git"]["branch"] != args.expected_branch
        or _git("rev-parse", "HEAD") != args.expected_revision
        or _git("branch", "--show-current") != args.expected_branch
        or _git("status", "--porcelain")
    ):
        raise ValueError("sampling preflight source identity differs")
    integrity = verify_training_checkpoint(checkpoint_path)
    if (
        int(integrity.get("step", -1)) != PILOT_STEP
        or integrity.get("checkpoint_sha256") != row["checkpoint_sha256"]
    ):
        raise ValueError("sampling preflight checkpoint integrity differs")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
