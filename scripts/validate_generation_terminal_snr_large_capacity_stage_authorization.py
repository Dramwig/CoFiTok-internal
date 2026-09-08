from __future__ import annotations

import json

from cofitok.generation.terminal_snr_large_capacity_execution import (
    validate_terminal_snr_large_capacity_stage_authorization,
)

try:
    from scripts.terminal_snr_large_capacity_stage_cli import (
        parse_validate_args,
        stable_load,
        stage_kwargs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_large_capacity_stage_cli import (
        parse_validate_args,
        stable_load,
        stage_kwargs,
    )


def main() -> None:
    args = parse_validate_args()
    report, _ = stable_load(
        args.stage_authorization,
        args.expected_stage_authorization_sha256,
        "terminal-SNR large-capacity stage authorization",
    )
    kwargs = stage_kwargs(args)
    expected_output_root = kwargs.pop("output_root")
    validated = validate_terminal_snr_large_capacity_stage_authorization(
        report,
        **kwargs,
        expected_output_root=expected_output_root,
    )
    print(
        json.dumps(
            {
                "status": validated["status"],
                "execution_ready": False,
                "training_launch_allowed": False,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
