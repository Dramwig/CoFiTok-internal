from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_generation_full_launch_receipt import (
        _common_arguments,
        _read_json,
        build_kwargs_from_args,
        verify_full_launch_receipt,
    )
except ModuleNotFoundError:
    from build_generation_full_launch_receipt import (
        _common_arguments,
        _read_json,
        build_kwargs_from_args,
        verify_full_launch_receipt,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an immutable stability-full 300K launch receipt."
    )
    _common_arguments(parser)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--allow-later-git-revision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verified = verify_full_launch_receipt(
        _read_json(args.receipt),
        receipt_path=args.receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
        **build_kwargs_from_args(
            args,
            require_current_git=not args.allow_later_git_revision,
            require_current_formal_repository=False,
            require_training_state_absent=False,
        ),
    )
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
