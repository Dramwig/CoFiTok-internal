from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.generation.min_snr_pilot import validate_execution_gate
from cofitok.reporting import file_sha256


def _read(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Min-SNR pilot gate.")
    parser.add_argument("--gate", required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    args = parser.parse_args()
    if file_sha256(args.gate) != args.expected_gate_sha256:
        raise ValueError("Min-SNR pilot execution gate SHA256 differs")
    identity = _identity(args.preparation)
    if identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("Min-SNR pilot preparation SHA256 differs")
    validate_execution_gate(
        _read(args.gate),
        preparation=_read(args.preparation),
        preparation_identity=identity,
    )
    print(args.expected_gate_sha256)


if __name__ == "__main__":
    main()

