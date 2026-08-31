"""Rebuild and compare a bounded exposure continuation result."""

from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_result import build_result
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256
from exposure_capacity_result_cli import parse_args, result_kwargs


def main() -> None:
    args = parse_args()
    result_path = reject_symlink_chain(args.result, name="continuation result").resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"continuation result is missing: {result_path}")
    for path, expected, name in (
        (args.authorization, args.expected_authorization_sha256, "authorization"),
        (args.gate, args.expected_gate_sha256, "execution gate"),
        (args.preparation, args.expected_preparation_sha256, "preparation"),
        (
            args.standing_authorization,
            args.expected_standing_authorization_sha256,
            "standing authorization",
        ),
        (
            args.stage_authorization,
            args.expected_stage_authorization_sha256,
            "stage authorization",
        ),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{name} SHA256 differs")
    actual = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("continuation result must be a JSON object")
    expected = build_result(**result_kwargs(args))
    if actual != expected:
        raise ValueError("continuation result does not match revalidated source evidence")
    print(
        json.dumps(
            {
                "status": "pass",
                "result_sha256": file_sha256(result_path),
                "scientific_status": actual["scientific_status"],
                "generation_advantage_proven": actual["generation_advantage_proven"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
