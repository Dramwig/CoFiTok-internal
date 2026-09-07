from __future__ import annotations

import json

from cofitok.generation.exposure_capacity_authorization import read_object
from cofitok.generation.terminal_snr_screen_execution import (
    build_terminal_snr_screen_execution_authorization,
    validate_terminal_snr_screen_execution_authorization,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256

try:
    from scripts.terminal_snr_screen_execution_cli import (
        authorization_kwargs,
        parse_validate_authorization_args,
    )
except ModuleNotFoundError:  # pragma: no cover
    from terminal_snr_screen_execution_cli import (  # type: ignore[no-redef]
        authorization_kwargs,
        parse_validate_authorization_args,
    )


def main() -> None:
    args = parse_validate_authorization_args()
    path = reject_symlink_chain(
        args.authorization, name="terminal-SNR execution authorization"
    ).resolve()
    if file_sha256(path) != args.expected_authorization_sha256:
        raise ValueError("terminal-SNR execution authorization SHA256 differs")
    actual = read_object(path, name="terminal-SNR execution authorization")
    kwargs = authorization_kwargs(args)
    expected = build_terminal_snr_screen_execution_authorization(**kwargs)
    if actual != expected:
        raise ValueError("terminal-SNR execution authorization replay differs")
    validate_terminal_snr_screen_execution_authorization(
        actual,
        preparation_identity=kwargs["preparation_identity"],
        expected_execution_checkout=kwargs["execution_checkout"],
        expected_output_root=kwargs["output_root"],
    )
    print(json.dumps({
        "status": "pass",
        "authorization_sha256": file_sha256(path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
