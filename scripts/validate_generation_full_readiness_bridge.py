from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_generation_full_readiness import _read_json
    from scripts.build_generation_full_readiness_bridge import (
        _common_arguments,
        kwargs_from_args,
        verify_readiness_bridge,
    )
except ModuleNotFoundError:
    from build_generation_full_readiness import _read_json
    from build_generation_full_readiness_bridge import (
        _common_arguments,
        kwargs_from_args,
        verify_readiness_bridge,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a source-bound readiness revision bridge."
    )
    _common_arguments(parser)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--expected-bridge-sha256", required=True)
    parser.add_argument("--print-selected-runtime", action="store_true")
    args = parser.parse_args()
    verified = verify_readiness_bridge(
        _read_json(args.bridge),
        bridge_path=args.bridge,
        expected_bridge_sha256=args.expected_bridge_sha256,
        **kwargs_from_args(args),
    )
    if args.print_selected_runtime:
        runtime = verified["runtime_selection"]
        print(
            f"{runtime['micro_batch_size']} "
            f"{runtime['gradient_accumulation_steps']}"
        )
    else:
        print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
