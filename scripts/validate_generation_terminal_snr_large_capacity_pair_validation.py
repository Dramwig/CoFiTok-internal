from __future__ import annotations

import json

from cofitok.generation.terminal_snr_large_capacity_execution import (
    validate_terminal_snr_large_capacity_pair_validation,
)

try:
    from scripts.terminal_snr_large_capacity_pair_cli import (
        pair_kwargs,
        parse_validate_args,
        stable_load,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_large_capacity_pair_cli import (
        pair_kwargs,
        parse_validate_args,
        stable_load,
    )


def main() -> None:
    args = parse_validate_args()
    report, _ = stable_load(
        args.pair_validation,
        args.expected_pair_validation_sha256,
        "terminal-SNR large-capacity pair validation",
    )
    kwargs = pair_kwargs(args)
    expected_output_root = kwargs.pop("output_root")
    validated = validate_terminal_snr_large_capacity_pair_validation(
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
