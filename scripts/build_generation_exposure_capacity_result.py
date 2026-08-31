"""Build the fail-closed result for an authorized exposure continuation."""

from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_result import build_result
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
from exposure_capacity_result_cli import parse_args, result_kwargs


def main() -> None:
    args = parse_args()
    result_path = reject_symlink_chain(args.result, name="continuation result").resolve()
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite continuation result: {result_path}")
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
    payload = build_result(**result_kwargs(args))
    write_json_report(result_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scientific_status": payload["scientific_status"],
                "generation_advantage_proven": payload["generation_advantage_proven"],
                "result_sha256": file_sha256(result_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
