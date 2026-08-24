from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation import (
    EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE,
    build_epsilon_stability_user_authorization_receipt,
)
from cofitok.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact user-approved scope for the non-authorizing "
            "matched 1K epsilon-stability diagnostic."
        )
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--user-message",
        default=EPSILON_STABILITY_USER_AUTHORIZATION_MESSAGE,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_epsilon_stability_user_authorization_receipt(
        task_id=args.task_id,
        user_message=args.user_message,
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
