from __future__ import annotations

import argparse
from pathlib import Path

from cofitok.generation import export_ema_inference_artifact
from cofitok.reporting import write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a verified EMA-only CoFiTok generation artifact."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = export_ema_inference_artifact(args.checkpoint, args.output)
    write_json_report(Path(args.report), report)
    print(args.report)


if __name__ == "__main__":
    main()
