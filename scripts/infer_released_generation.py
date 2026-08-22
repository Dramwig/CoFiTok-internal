from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.infer_generation import build_parser, run_inference
except ModuleNotFoundError:
    from infer_generation import build_parser, run_inference


def parse_args() -> argparse.Namespace:
    return build_parser(released=True).parse_args()


def run_released_inference(args: argparse.Namespace) -> dict:
    receipt = str(getattr(args, "completion_receipt", "") or "").strip()
    if not receipt:
        raise ValueError("released inference requires a completion receipt")
    if getattr(args, "weights", "ema") != "ema":
        raise ValueError("released inference exposes EMA weights only")
    enforced = argparse.Namespace(
        **{
            **vars(args),
            "completion_receipt": receipt,
            "weights": "ema",
            "require_release_authorization": True,
            "require_completion_authorization": True,
        }
    )
    return run_inference(enforced)


def main() -> None:
    args = parse_args()
    report = run_released_inference(args)
    output = (
        Path(args.report)
        if args.report
        else Path(args.output_dir) / "inference_report.json"
    )
    print(output)
    print(f"generated {report['output_count']} images")


if __name__ == "__main__":
    main()
