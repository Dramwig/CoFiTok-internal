from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation import build_epsilon_stability_execution_authorization
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import write_json_report


def _source(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = gate_source_report_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the bound source")
    with Path(identity["path"]).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return identity, payload


def build_from_paths(
    *,
    preparation_path: str | Path,
    expected_preparation_sha256: str,
    design_path: str | Path,
    expected_design_sha256: str,
    user_authorization_path: str | Path,
    expected_user_authorization_sha256: str,
) -> dict[str, Any]:
    preparation_identity, preparation = _source(
        preparation_path,
        expected_preparation_sha256,
        label="epsilon-stability preparation",
    )
    design_identity, design = _source(
        design_path,
        expected_design_sha256,
        label="sampling design",
    )
    if preparation.get("design_identity") != design_identity:
        raise ValueError("epsilon-stability preparation binds another design")
    user_identity, user_authorization = _source(
        user_authorization_path,
        expected_user_authorization_sha256,
        label="separate user authorization",
    )
    return build_epsilon_stability_execution_authorization(
        preparation=preparation,
        preparation_identity=preparation_identity,
        design=design,
        user_authorization=user_authorization,
        user_authorization_identity=user_identity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact matched-1K epsilon-stability execution gate from "
            "a replayed preparation and user authorization receipt."
        )
    )
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--user-authorization", required=True)
    parser.add_argument("--expected-user-authorization-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        preparation_path=args.preparation,
        expected_preparation_sha256=args.expected_preparation_sha256,
        design_path=args.design,
        expected_design_sha256=args.expected_design_sha256,
        user_authorization_path=args.user_authorization,
        expected_user_authorization_sha256=(
            args.expected_user_authorization_sha256
        ),
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
