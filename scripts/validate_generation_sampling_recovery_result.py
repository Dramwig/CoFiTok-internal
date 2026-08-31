"""Validate the approved non-authorizing matched sampling-recovery result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation.sampling_recovery_audit import validate_result
from cofitok.inference_replay import reject_symlink_chain


def _read(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="sampling recovery result")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"sampling recovery result is unreadable: {source}") from error
    if not isinstance(value, dict):
        raise ValueError("sampling recovery result must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Revalidate a completed, non-authorizing sampling-recovery result."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    result = _read(args.result)
    summary = validate_result(
        result,
        result_path=args.result,
        expected_result_sha256=args.expected_sha256,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
